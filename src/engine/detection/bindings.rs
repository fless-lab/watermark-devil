use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use opencv::core::{Mat, Point, Rect, Size};
use anyhow::Result;
use tracing::{info, debug, warn};

use crate::types::{Detection, WatermarkType, Confidence};
use crate::config::DetectionConfig;

/// Wrapper pour les modèles Python
pub struct PyModelWrapper {
    instance: Arc<PyObject>,
    interpreter: Arc<Python<'static>>,
}

impl PyModelWrapper {
    pub fn new(module_path: &str, class_name: &str, config: Option<&PyDict>) -> Result<Self> {
        Python::with_gil(|py| {
            // Import dynamique du module
            let module = PyModule::import(py, module_path)?;
            let class = module.getattr(class_name)?;
            
            // Instanciation avec config
            let instance = match config {
                Some(cfg) => class.call1((cfg,))?,
                None => class.call0()?,
            };
            
            Ok(Self {
                instance: Arc::new(instance.into()),
                interpreter: Arc::new(py),
            })
        })
    }
    
    pub fn preprocess(&self, image: &Mat) -> Result<PyObject> {
        Python::with_gil(|py| {
            // Convertir Mat en array NumPy
            let numpy = py.import("numpy")?;
            let image_array = numpy.getattr("array")?.call1((image.data()?,))?;
            
            // Appeler preprocess
            let preprocessed = self.instance.call_method1(py, "preprocess", (image_array,))?;
            Ok(preprocessed.into())
        })
    }
    
    pub fn detect(&self, preprocessed: &PyObject) -> Result<Vec<Detection>> {
        Python::with_gil(|py| {
            // Appeler detect
            let detections = self.instance.call_method1(py, "detect", (preprocessed,))?;
            
            // Convertir les résultats
            let mut results = Vec::new();
            for detection in detections.extract::<&PyList>()?.iter() {
                let (bbox, confidence) = detection.extract::<(Vec<i32>, f32)>()?;
                
                if bbox.len() == 4 {
                    results.push(Detection {
                        watermark_type: WatermarkType::Unknown, // Sera défini par le détecteur spécifique
                        bbox: Rect::new(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
                        confidence: Confidence::new(confidence),
                        mask: None,
                        metadata: None,
                    });
                }
            }
            
            Ok(results)
        })
    }
}

/// Wrapper pour le détecteur de logos
pub struct LogoDetectorWrapper {
    model: PyModelWrapper,
}

impl LogoDetectorWrapper {
    pub fn new(config: &DetectionConfig) -> Result<Self> {
        let py_config = Python::with_gil(|py| {
            let config_dict = PyDict::new(py);
            config_dict.set_item("conf_threshold", config.confidence_threshold)?;
            config_dict.set_item("iou_threshold", config.iou_threshold)?;
            Ok(config_dict)
        })?;
        
        Ok(Self {
            model: PyModelWrapper::new(
                "ml.models.logo_detector.model",
                "LogoDetector",
                Some(&py_config)
            )?,
        })
    }
    
    pub fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        let preprocessed = self.model.preprocess(image)?;
        let mut detections = self.model.detect(&preprocessed)?;
        
        // Définir le type de watermark
        for detection in &mut detections {
            detection.watermark_type = WatermarkType::Logo;
        }
        
        Ok(detections)
    }
}

/// Wrapper pour le détecteur de texte
pub struct TextDetectorWrapper {
    model: PyModelWrapper,
}

impl TextDetectorWrapper {
    pub fn new(config: &DetectionConfig) -> Result<Self> {
        let py_config = Python::with_gil(|py| {
            let config_dict = PyDict::new(py);
            config_dict.set_item("conf_threshold", config.confidence_threshold)?;
            Ok(config_dict)
        })?;
        
        Ok(Self {
            model: PyModelWrapper::new(
                "ml.models.text_detector.model",
                "TextDetector",
                Some(&py_config)
            )?,
        })
    }
    
    pub fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        let preprocessed = self.model.preprocess(image)?;
        let mut detections = self.model.detect(&preprocessed)?;
        
        for detection in &mut detections {
            detection.watermark_type = WatermarkType::Text;
        }
        
        Ok(detections)
    }
}

/// Wrapper pour le détecteur de motifs
pub struct PatternDetectorWrapper {
    model: PyModelWrapper,
}

impl PatternDetectorWrapper {
    pub fn new(config: &DetectionConfig) -> Result<Self> {
        let py_config = Python::with_gil(|py| {
            let config_dict = PyDict::new(py);
            config_dict.set_item("min_pattern_size", 20)?;
            config_dict.set_item("max_pattern_size", 200)?;
            Ok(config_dict)
        })?;
        
        Ok(Self {
            model: PyModelWrapper::new(
                "ml.models.pattern_detector.model",
                "PatternDetector",
                Some(&py_config)
            )?,
        })
    }
    
    pub fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        let preprocessed = self.model.preprocess(image)?;
        let mut detections = self.model.detect(&preprocessed)?;
        
        for detection in &mut detections {
            detection.watermark_type = WatermarkType::Pattern;
        }
        
        Ok(detections)
    }
}

/// Wrapper pour le détecteur de transparence
pub struct TransparencyDetectorWrapper {
    model: PyModelWrapper,
}

impl TransparencyDetectorWrapper {
    pub fn new(config: &DetectionConfig) -> Result<Self> {
        let py_config = Python::with_gil(|py| {
            let config_dict = PyDict::new(py);
            config_dict.set_item("alpha_threshold", 0.1)?;
            config_dict.set_item("min_area", 100)?;
            Ok(config_dict)
        })?;
        
        Ok(Self {
            model: PyModelWrapper::new(
                "ml.models.transparency_detector.model",
                "TransparencyDetector",
                Some(&py_config)
            )?,
        })
    }
    
    pub fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        let preprocessed = self.model.preprocess(image)?;
        let mut detections = self.model.detect(&preprocessed)?;
        
        for detection in &mut detections {
            detection.watermark_type = WatermarkType::Transparent;
        }
        
        Ok(detections)
    }
}

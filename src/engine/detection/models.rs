use std::sync::Arc;
use opencv::{core::{Mat, Point, Rect, Size}, imgproc};
use pyo3::prelude::*;
use rayon::prelude::*;
use anyhow::Result;
use tracing::{info, debug, warn};

use crate::types::{Detection, WatermarkType, Confidence};
use crate::config::DetectionConfig;

mod bindings;
use bindings::{LogoDetectorWrapper, TextDetectorWrapper, PatternDetectorWrapper, TransparencyDetectorWrapper};

/// Interface for all watermark detectors
pub trait WatermarkDetector: Send + Sync {
    fn detect(&self, image: &Mat) -> Result<Vec<Detection>>;
    fn get_type(&self) -> WatermarkType;
    fn get_name(&self) -> String;
}

/// Logo detector using YOLOv5 via Python-Rust integration
pub struct LogoDetector {
    config: DetectionConfig,
    model: Arc<PyObject>,
    interpreter: Arc<Python<'static>>,
}

impl LogoDetector {
    pub fn new(config: &DetectionConfig) -> Result<Self> {
        Python::with_gil(|py| {
            // Import the Python model
            let model = PyModule::import(py, "ml.models.logo_detector.model")?
                .getattr("LogoDetector")?
                .call0()?;
            
            Ok(Self {
                config: config.clone(),
                model: Arc::new(model.into()),
                interpreter: Arc::new(py),
            })
        })
    }
}

impl WatermarkDetector for LogoDetector {
    fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        Python::with_gil(|py| {
            // Convert OpenCV Mat to NumPy array
            let numpy = PyModule::import(py, "numpy")?;
            let image_array = numpy.getattr("array")?.call1((image.data()?,))?;
            
            // Prétraitement de l'image
            let preprocessed = self.model.call_method1(py, "preprocess", (image_array,))?;
            
            // Détection avec le modèle
            let predictions = self.model.call_method1(py, "detect", (preprocessed,))?;
            
            // Convertir les prédictions en détections
            let mut detections = Vec::new();
            
            for pred in predictions.iter()? {
                let pred = pred?;
                let bbox = pred.get_item(0)?;
                let confidence = pred.get_item(1)?.extract::<f32>()?;
                
                // Créer le rectangle de détection
                let rect = Rect::new(
                    bbox.get_item(0)?.extract()?,
                    bbox.get_item(1)?.extract()?,
                    bbox.get_item(2)?.extract()?,
                    bbox.get_item(3)?.extract()?,
                );
                
                // Ajouter la détection
                detections.push(Detection {
                    watermark_type: self.get_type(),
                    bbox: rect,
                    confidence: Confidence::new(confidence),
                    mask: None,  // Le masque sera généré si nécessaire
                    metadata: Some(serde_json::json!({
                        "model": "yolov5",
                        "size": format!("{}x{}", rect.width, rect.height),
                    })),
                });
            }
            
            Ok(detections)
        })
    }

    fn get_type(&self) -> WatermarkType {
        WatermarkType::Logo
    }

    fn get_name(&self) -> String {
        "YOLOv5 Logo Detector".to_string()
    }
}

/// Text detector using OCR and deep learning
pub struct TextDetector {
    config: DetectionConfig,
    model: Arc<PyObject>,
    interpreter: Arc<Python<'static>>,
}

impl TextDetector {
    pub fn new(config: &DetectionConfig) -> Result<Self> {
        Python::with_gil(|py| {
            let model = PyModule::import(py, "ml.models.text_detector.model")?
                .getattr("TextDetector")?
                .call0()?;
            
            Ok(Self {
                config: config.clone(),
                model: Arc::new(model.into()),
                interpreter: Arc::new(py),
            })
        })
    }
}

impl WatermarkDetector for TextDetector {
    fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        Python::with_gil(|py| {
            let numpy = PyModule::import(py, "numpy")?;
            let image_array = numpy.getattr("array")?.call1((image.data()?,))?;
            
            let detections = self.model.call_method1(py, "detect", (image_array,))?;
            
            let mut results = Vec::new();
            for detection in detections.iter(py)? {
                let det = detection?;
                let bbox = det.getattr("bbox")?.extract::<(i32, i32, i32, i32)>()?;
                let confidence = det.getattr("confidence")?.extract::<f32>()?;
                let text = det.getattr("text")?.extract::<String>()?;
                
                results.push(Detection {
                    watermark_type: WatermarkType::Text,
                    confidence: Confidence::new(confidence),
                    bbox: Rect::new(bbox.0, bbox.1, bbox.2, bbox.3),
                    metadata: Some(serde_json::json!({
                        "text": text,
                    })),
                });
            }
            
            Ok(results)
        })
    }
    
    fn get_type(&self) -> WatermarkType {
        WatermarkType::Text
    }
    
    fn get_name(&self) -> String {
        "TextDetector".to_string()
    }
}

/// Pattern detector for repetitive watermarks
pub struct PatternDetector {
    config: DetectionConfig,
    model: Arc<PyObject>,
    interpreter: Arc<Python<'static>>,
}

impl PatternDetector {
    pub fn new(config: &DetectionConfig) -> Result<Self> {
        Python::with_gil(|py| {
            let model = PyModule::import(py, "ml.models.pattern_detector.model")?
                .getattr("PatternDetector")?
                .call0()?;
            
            Ok(Self {
                config: config.clone(),
                model: Arc::new(model.into()),
                interpreter: Arc::new(py),
            })
        })
    }
}

impl WatermarkDetector for PatternDetector {
    fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        Python::with_gil(|py| {
            let numpy = PyModule::import(py, "numpy")?;
            let image_array = numpy.getattr("array")?.call1((image.data()?,))?;
            
            let detections = self.model.call_method1(py, "detect", (image_array,))?;
            
            let mut results = Vec::new();
            for detection in detections.iter(py)? {
                let det = detection?;
                let bbox = det.getattr("bbox")?.extract::<(i32, i32, i32, i32)>()?;
                let confidence = det.getattr("confidence")?.extract::<f32>()?;
                let pattern_type = det.getattr("pattern_type")?.extract::<String>()?;
                
                results.push(Detection {
                    watermark_type: WatermarkType::Pattern,
                    confidence: Confidence::new(confidence),
                    bbox: Rect::new(bbox.0, bbox.1, bbox.2, bbox.3),
                    metadata: Some(serde_json::json!({
                        "pattern_type": pattern_type,
                    })),
                });
            }
            
            Ok(results)
        })
    }
    
    fn get_type(&self) -> WatermarkType {
        WatermarkType::Pattern
    }
    
    fn get_name(&self) -> String {
        "PatternDetector".to_string()
    }
}

/// Transparency detector for alpha channel and semi-transparent watermarks
pub struct TransparencyDetector {
    config: DetectionConfig,
    model: Arc<PyObject>,
    interpreter: Arc<Python<'static>>,
}

impl TransparencyDetector {
    pub fn new(config: &DetectionConfig) -> Result<Self> {
        Python::with_gil(|py| {
            let model = PyModule::import(py, "ml.models.transparency_detector.model")?
                .getattr("TransparencyDetector")?
                .call0()?;
            
            Ok(Self {
                config: config.clone(),
                model: Arc::new(model.into()),
                interpreter: Arc::new(py),
            })
        })
    }
}

impl WatermarkDetector for TransparencyDetector {
    fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        Python::with_gil(|py| {
            let numpy = PyModule::import(py, "numpy")?;
            let image_array = numpy.getattr("array")?.call1((image.data()?,))?;
            
            let detections = self.model.call_method1(py, "detect", (image_array,))?;
            
            let mut results = Vec::new();
            for detection in detections.iter(py)? {
                let det = detection?;
                let bbox = det.getattr("bbox")?.extract::<(i32, i32, i32, i32)>()?;
                let confidence = det.getattr("confidence")?.extract::<f32>()?;
                let alpha = det.getattr("alpha")?.extract::<f32>()?;
                
                results.push(Detection {
                    watermark_type: WatermarkType::Transparent,
                    confidence: Confidence::new(confidence),
                    bbox: Rect::new(bbox.0, bbox.1, bbox.2, bbox.3),
                    metadata: Some(serde_json::json!({
                        "alpha": alpha,
                    })),
                });
            }
            
            Ok(results)
        })
    }
    
    fn get_type(&self) -> WatermarkType {
        WatermarkType::Transparent
    }
    
    fn get_name(&self) -> String {
        "TransparencyDetector".to_string()
    }
}

/// Ensemble detector that combines results from multiple detectors
pub struct EnsembleDetector {
    config: DetectionConfig,
    logo_detector: LogoDetectorWrapper,
    text_detector: TextDetectorWrapper,
    pattern_detector: PatternDetectorWrapper,
    transparency_detector: TransparencyDetectorWrapper,
}

impl EnsembleDetector {
    pub fn new(config: &DetectionConfig) -> Result<Self> {
        Ok(Self {
            logo_detector: LogoDetectorWrapper::new(config)?,
            text_detector: TextDetectorWrapper::new(config)?,
            pattern_detector: PatternDetectorWrapper::new(config)?,
            transparency_detector: TransparencyDetectorWrapper::new(config)?,
            config: config.clone(),
        })
    }
    
    pub fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        let mut all_detections = Vec::new();
        
        // Détecter avec chaque détecteur
        if let Ok(detections) = self.logo_detector.detect(image) {
            all_detections.extend(detections);
        }
        
        if let Ok(detections) = self.text_detector.detect(image) {
            all_detections.extend(detections);
        }
        
        if let Ok(detections) = self.pattern_detector.detect(image) {
            all_detections.extend(detections);
        }
        
        if let Ok(detections) = self.transparency_detector.detect(image) {
            all_detections.extend(detections);
        }
        
        // Non-maximum suppression
        self.non_max_suppression(&mut all_detections);
        
        Ok(all_detections)
    }
    
    fn non_max_suppression(&self, detections: &mut Vec<Detection>) {
        if detections.is_empty() {
            return;
        }
        
        // Trier par confiance
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        let mut i = 0;
        while i < detections.len() {
            let mut j = i + 1;
            while j < detections.len() {
                if self.iou(&detections[i].bbox, &detections[j].bbox) > self.config.iou_threshold {
                    detections.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }
    
    fn iou(&self, bbox1: &Rect, bbox2: &Rect) -> f32 {
        let x1 = bbox1.x.max(bbox2.x);
        let y1 = bbox1.y.max(bbox2.y);
        let x2 = (bbox1.x + bbox1.width).min(bbox2.x + bbox2.width);
        let y2 = (bbox1.y + bbox1.height).min(bbox2.y + bbox2.height);
        
        if x2 < x1 || y2 < y1 {
            return 0.0;
        }
        
        let intersection = (x2 - x1) * (y2 - y1);
        let area1 = bbox1.width * bbox1.height;
        let area2 = bbox2.width * bbox2.height;
        
        intersection as f32 / (area1 + area2 - intersection) as f32
    }
}

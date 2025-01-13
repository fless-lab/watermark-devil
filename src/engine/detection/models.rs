use std::sync::Arc;
use opencv::{core::{Mat, Point, Rect, Size}, imgproc};
use pyo3::prelude::*;
use rayon::prelude::*;
use anyhow::Result;
use tracing::{info, debug, warn};

use crate::types::{Detection, WatermarkType, Confidence};
use crate::config::DetectionConfig;

/// Interface for all watermark detectors
pub trait WatermarkDetector: Send + Sync {
    fn detect(&self, image: &Mat) -> Result<Vec<Detection>>;
    fn get_type(&self) -> WatermarkType;
    fn get_name(&self) -> String;
}

/// Logo detector using deep learning
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
            
            // Call Python model
            let detections = self.model.call_method1(py, "detect", (image_array,))?;
            
            // Convert Python results to Rust structs
            let mut results = Vec::new();
            for detection in detections.iter(py)? {
                let det = detection?;
                let bbox = det.getattr("bbox")?.extract::<(i32, i32, i32, i32)>()?;
                let confidence = det.getattr("confidence")?.extract::<f32>()?;
                
                results.push(Detection {
                    watermark_type: WatermarkType::Logo,
                    confidence: Confidence::new(confidence),
                    bbox: Rect::new(bbox.0, bbox.1, bbox.2, bbox.3),
                    metadata: None,
                });
            }
            
            Ok(results)
        })
    }
    
    fn get_type(&self) -> WatermarkType {
        WatermarkType::Logo
    }
    
    fn get_name(&self) -> String {
        "LogoDetector".to_string()
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
    detectors: Vec<Box<dyn WatermarkDetector>>,
    config: DetectionConfig,
}

impl EnsembleDetector {
    pub fn new(config: &DetectionConfig) -> Result<Self> {
        let mut detectors: Vec<Box<dyn WatermarkDetector>> = Vec::new();
        
        // Initialize all detectors
        detectors.push(Box::new(LogoDetector::new(config)?));
        detectors.push(Box::new(TextDetector::new(config)?));
        detectors.push(Box::new(PatternDetector::new(config)?));
        detectors.push(Box::new(TransparencyDetector::new(config)?));
        
        Ok(Self {
            detectors,
            config: config.clone(),
        })
    }
    
    pub fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        // Run all detectors in parallel
        let all_detections: Vec<Vec<Detection>> = self.detectors.par_iter()
            .map(|detector| {
                match detector.detect(image) {
                    Ok(detections) => detections,
                    Err(e) => {
                        warn!("Detector {} failed: {}", detector.get_name(), e);
                        Vec::new()
                    }
                }
            })
            .collect();
        
        // Merge and filter detections
        let mut merged = Vec::new();
        for detections in all_detections {
            for detection in detections {
                if detection.confidence.value > self.config.min_confidence {
                    merged.push(detection);
                }
            }
        }
        
        // Non-maximum suppression
        self.non_max_suppression(&mut merged);
        
        Ok(merged)
    }
    
    fn non_max_suppression(&self, detections: &mut Vec<Detection>) {
        detections.sort_by(|a, b| b.confidence.value.partial_cmp(&a.confidence.value).unwrap());
        
        let mut i = 0;
        while i < detections.len() {
            let mut j = i + 1;
            while j < detections.len() {
                if self.iou(&detections[i].bbox, &detections[j].bbox) > self.config.nms_threshold {
                    detections.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }
    
    fn iou(&self, bbox1: &Rect, bbox2: &Rect) -> f32 {
        let intersection = bbox1 & bbox2;
        if intersection.width <= 0 || intersection.height <= 0 {
            return 0.0;
        }
        
        let intersection_area = intersection.width * intersection.height;
        let union_area = (bbox1.width * bbox1.height) + 
                        (bbox2.width * bbox2.height) - 
                        intersection_area;
        
        intersection_area as f32 / union_area as f32
    }
}

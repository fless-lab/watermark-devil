use std::sync::Arc;
use std::time::Instant;
use opencv::core::{Mat, UMat};
use opencv::prelude::*;
use anyhow::{Result, Context};
use rayon::prelude::*;
use metrics::{counter, gauge, histogram};

use crate::types::Detection;
use crate::config::DetectionConfig;
use crate::utils::validation;

/// Trait définissant l'interface commune pour tous les détecteurs
pub trait WatermarkDetector: Send + Sync {
    /// Détecte les filigranes dans une image
    fn detect(&self, image: &Mat) -> Result<Vec<Detection>>;
    
    /// Version optimisée GPU si disponible
    fn detect_gpu(&self, image: &UMat) -> Result<Vec<Detection>> {
        // Par défaut, on convertit en Mat et on utilise la version CPU
        let cpu_mat = image.get_mat()?;
        self.detect(&cpu_mat)
    }
}

/// Métriques de performance pour le moteur de détection
#[derive(Debug, Clone)]
pub struct DetectionMetrics {
    pub total_time_ms: f64,
    pub preprocessing_time_ms: f64,
    pub detection_time_ms: f64,
    pub postprocessing_time_ms: f64,
    pub num_detections: usize,
}

/// Détecteur qui combine les résultats de plusieurs détecteurs
#[derive(Clone)]
pub struct EnsembleDetector {
    detectors: Vec<Arc<dyn WatermarkDetector>>,
    config: DetectionConfig,
}

impl EnsembleDetector {
    pub fn new(detectors: Vec<Arc<dyn WatermarkDetector>>, config: DetectionConfig) -> Self {
        Self { detectors, config }
    }
    
    /// Valide l'image d'entrée
    fn validate_input(&self, image: &Mat) -> Result<()> {
        // Vérifier que l'image n'est pas vide
        validation::check_not_empty(image)
            .context("Image d'entrée vide")?;
            
        // Vérifier les dimensions
        validation::check_dimensions(
            image,
            self.config.min_image_size,
            self.config.max_image_size
        ).context("Dimensions d'image invalides")?;
        
        // Vérifier le type
        validation::check_image_type(image)
            .context("Type d'image non supporté")?;
            
        Ok(())
    }
    
    fn merge_detections(&self, all_detections: Vec<Vec<Detection>>) -> Vec<Detection> {
        let mut merged = Vec::new();
        
        // Aplatir toutes les détections
        for detections in all_detections {
            merged.extend(detections);
        }
        
        // Trier par confiance décroissante
        merged.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        // Non-maximum suppression
        self.non_max_suppression(&mut merged);
        
        merged
    }
    
    fn non_max_suppression(&self, detections: &mut Vec<Detection>) {
        if detections.is_empty() {
            return;
        }
        
        let mut i = 0;
        while i < detections.len() {
            let mut j = i + 1;
            while j < detections.len() {
                if validation::compute_iou(&detections[i].bbox, &detections[j].bbox) > self.config.iou_threshold {
                    detections.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }
    
    /// Détecte avec métriques de performance
    pub fn detect_with_metrics(&self, image: &Mat) -> Result<(Vec<Detection>, DetectionMetrics)> {
        let start = Instant::now();
        
        // Validation
        let preprocess_start = Instant::now();
        self.validate_input(image)?;
        let preprocess_time = preprocess_start.elapsed();
        
        // Détection parallèle
        let detect_start = Instant::now();
        let all_detections: Vec<_> = self.detectors.par_iter()
            .filter_map(|detector| {
                match detector.detect(image) {
                    Ok(detections) => Some(detections),
                    Err(e) => {
                        counter!("detection.errors", 1);
                        tracing::warn!("Erreur de détection: {}", e);
                        None
                    }
                }
            })
            .collect();
        let detect_time = detect_start.elapsed();
        
        // Post-traitement
        let postprocess_start = Instant::now();
        let merged = self.merge_detections(all_detections);
        let postprocess_time = postprocess_start.elapsed();
        
        // Métriques
        let total_time = start.elapsed();
        let metrics = DetectionMetrics {
            total_time_ms: total_time.as_secs_f64() * 1000.0,
            preprocessing_time_ms: preprocess_time.as_secs_f64() * 1000.0,
            detection_time_ms: detect_time.as_secs_f64() * 1000.0,
            postprocessing_time_ms: postprocess_time.as_secs_f64() * 1000.0,
            num_detections: merged.len(),
        };
        
        // Enregistrer les métriques
        gauge!("detection.total_time_ms", metrics.total_time_ms);
        gauge!("detection.preprocessing_time_ms", metrics.preprocessing_time_ms);
        gauge!("detection.detection_time_ms", metrics.detection_time_ms);
        gauge!("detection.postprocessing_time_ms", metrics.postprocessing_time_ms);
        histogram!("detection.num_detections", metrics.num_detections as f64);
        
        Ok((merged, metrics))
    }
}

impl WatermarkDetector for EnsembleDetector {
    fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        let (detections, _) = self.detect_with_metrics(image)?;
        Ok(detections)
    }
    
    fn detect_gpu(&self, image: &UMat) -> Result<Vec<Detection>> {
        // Version optimisée GPU
        let start = Instant::now();
        
        // Validation
        let cpu_mat = image.get_mat()?;
        self.validate_input(&cpu_mat)?;
        
        // Détection parallèle sur GPU
        let all_detections: Vec<_> = self.detectors.par_iter()
            .filter_map(|detector| {
                match detector.detect_gpu(image) {
                    Ok(detections) => Some(detections),
                    Err(e) => {
                        counter!("detection.gpu.errors", 1);
                        tracing::warn!("Erreur de détection GPU: {}", e);
                        None
                    }
                }
            })
            .collect();
            
        let merged = self.merge_detections(all_detections);
        
        // Métriques GPU
        let total_time = start.elapsed();
        gauge!("detection.gpu.total_time_ms", total_time.as_secs_f64() * 1000.0);
        
        Ok(merged)
    }
}

/// Logo detector using YOLOv5 via Python-Rust integration
pub struct LogoDetector {
    config: crate::config::DetectionConfig,
    model: Arc<PyObject>,
    interpreter: Arc<Python<'static>>,
}

impl LogoDetector {
    pub fn new(config: &crate::config::DetectionConfig) -> Result<Self> {
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
                    watermark_type: WatermarkType::Logo,
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
    
    fn detect_gpu(&self, image: &UMat) -> Result<Vec<Detection>> {
        // Version optimisée GPU
        let start = Instant::now();
        
        // Validation
        let cpu_mat = image.get_mat()?;
        
        // Détection parallèle sur GPU
        let all_detections: Vec<_> = self.detectors.par_iter()
            .filter_map(|detector| {
                match detector.detect_gpu(image) {
                    Ok(detections) => Some(detections),
                    Err(e) => {
                        counter!("detection.gpu.errors", 1);
                        tracing::warn!("Erreur de détection GPU: {}", e);
                        None
                    }
                }
            })
            .collect();
            
        let merged = self.merge_detections(all_detections);
        
        // Métriques GPU
        let total_time = start.elapsed();
        gauge!("detection.gpu.total_time_ms", total_time.as_secs_f64() * 1000.0);
        
        Ok(merged)
    }
}

/// Text detector using OCR and deep learning
pub struct TextDetector {
    config: crate::config::DetectionConfig,
    model: Arc<PyObject>,
    interpreter: Arc<Python<'static>>,
}

impl TextDetector {
    pub fn new(config: &crate::config::DetectionConfig) -> Result<Self> {
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
}

/// Pattern detector for repetitive watermarks
pub struct PatternDetector {
    config: crate::config::DetectionConfig,
    model: Arc<PyObject>,
    interpreter: Arc<Python<'static>>,
}

impl PatternDetector {
    pub fn new(config: &crate::config::DetectionConfig) -> Result<Self> {
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
}

/// Transparency detector for alpha channel and semi-transparent watermarks
pub struct TransparencyDetector {
    config: crate::config::DetectionConfig,
    model: Arc<PyObject>,
    interpreter: Arc<Python<'static>>,
}

impl TransparencyDetector {
    pub fn new(config: &crate::config::DetectionConfig) -> Result<Self> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{WatermarkType, BoundingBox, Confidence};
    use opencv::core::Mat;
    
    struct MockDetector {
        detections: Vec<Detection>,
    }
    
    impl MockDetector {
        fn new(detections: Vec<Detection>) -> Self {
            Self { detections }
        }
    }
    
    impl WatermarkDetector for MockDetector {
        fn detect(&self, _image: &Mat) -> Result<Vec<Detection>> {
            Ok(self.detections.clone())
        }
    }
    
    #[test]
    fn test_validation() -> Result<()> {
        let config = DetectionConfig::default();
        let ensemble = EnsembleDetector::new(vec![], config);
        
        // Image vide
        let empty_mat = Mat::default();
        assert!(ensemble.validate_input(&empty_mat).is_err());
        
        // Image valide
        let valid_mat = Mat::ones(100, 100, opencv::core::CV_8UC3)?;
        assert!(ensemble.validate_input(&valid_mat).is_ok());
        
        Ok(())
    }
    
    #[test]
    fn test_ensemble_detector() -> Result<()> {
        let config = DetectionConfig::default();
        
        // Créer des détections fictives
        let mock1 = MockDetector::new(vec![
            Detection {
                watermark_type: WatermarkType::Logo,
                bbox: BoundingBox {
                    x: 0,
                    y: 0,
                    width: 100,
                    height: 100,
                },
                confidence: Confidence::new(0.8),
                mask: None,
            }
        ]);
        
        let mock2 = MockDetector::new(vec![
            Detection {
                watermark_type: WatermarkType::Text,
                bbox: BoundingBox {
                    x: 50,
                    y: 50,
                    width: 200,
                    height: 50,
                },
                confidence: Confidence::new(0.9),
                mask: None,
            }
        ]);
        
        // Créer l'ensemble
        let ensemble = EnsembleDetector::new(
            vec![Arc::new(mock1), Arc::new(mock2)],
            config
        );
        
        // Tester la détection avec métriques
        let test_mat = Mat::ones(100, 100, opencv::core::CV_8UC3)?;
        let (detections, metrics) = ensemble.detect_with_metrics(&test_mat)?;
        
        // Vérifier les résultats
        assert_eq!(detections.len(), 2);
        assert!(detections[0].confidence.value > detections[1].confidence.value);
        
        // Vérifier les métriques
        assert!(metrics.total_time_ms > 0.0);
        assert!(metrics.preprocessing_time_ms > 0.0);
        assert!(metrics.detection_time_ms > 0.0);
        assert!(metrics.postprocessing_time_ms > 0.0);
        assert_eq!(metrics.num_detections, 2);
        
        Ok(())
    }
}

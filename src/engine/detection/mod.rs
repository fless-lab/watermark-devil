use std::sync::Arc;
use opencv::{
    core::{Mat, Size, CV_8UC3},
    imgproc,
    prelude::*,
};
use rayon::prelude::*;
use anyhow::{Result, Context};
use tracing::{info, debug, warn, instrument};

mod models;
mod frequency;
mod patterns;
mod neural;
mod utils;

use crate::types::{Detection, WatermarkType, BoundingBox, Confidence};
use crate::config::DetectionConfig;
use crate::error::EngineError;

use self::{
    models::{EnsembleDetector, WatermarkDetector},
    frequency::FrequencyDetector,
    patterns::PatternDetector,
    neural::NeuralDetector,
    utils::preprocess_image,
};

/// Engine principal de détection de filigranes
pub struct DetectionEngine {
    config: Arc<DetectionConfig>,
    ensemble: Arc<EnsembleDetector>,
    frequency_detector: Arc<FrequencyDetector>,
    pattern_detector: Arc<PatternDetector>,
    neural_detector: Arc<NeuralDetector>,
}

impl DetectionEngine {
    /// Crée une nouvelle instance du moteur de détection
    pub fn new(config: DetectionConfig) -> Result<Self> {
        let config = Arc::new(config);
        
        Ok(Self {
            ensemble: Arc::new(EnsembleDetector::new(&config)?),
            frequency_detector: Arc::new(FrequencyDetector::new(&config)?),
            pattern_detector: Arc::new(PatternDetector::new(&config)?),
            neural_detector: Arc::new(NeuralDetector::new(&config)?),
            config,
        })
    }

    /// Détecte les filigranes dans une image
    #[instrument(skip(self, image), fields(image_size = ?image.size()))]
    pub fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        // Validation de l'image
        if image.empty() {
            return Err(EngineError::invalid_input("Empty image provided").into());
        }
        
        if image.channels() != 3 {
            return Err(EngineError::invalid_input("Image must have 3 channels").into());
        }
        
        // Prétraitement de l'image
        let processed = self.preprocess_image(image)
            .context("Failed to preprocess image")?;
            
        debug!("Image preprocessed successfully");
        
        // Détection parallèle avec différents détecteurs
        let mut all_detections = Vec::new();
        
        // Utilisation de rayon pour la parallélisation
        let detectors: Vec<(&str, Box<dyn WatermarkDetector + Send + Sync>)> = vec![
            ("ensemble", Box::new(self.ensemble.as_ref().clone())),
            ("frequency", Box::new(self.frequency_detector.as_ref().clone())),
            ("pattern", Box::new(self.pattern_detector.as_ref().clone())),
            ("neural", Box::new(self.neural_detector.as_ref().clone())),
        ];
        
        let detector_results: Vec<Result<Vec<Detection>>> = detectors.par_iter()
            .map(|(name, detector)| {
                debug!("Running {} detector", name);
                detector.detect(&processed)
                    .with_context(|| format!("Detection failed for {} detector", name))
            })
            .collect();
            
        // Agrégation des résultats
        for result in detector_results {
            match result {
                Ok(detections) => all_detections.extend(detections),
                Err(e) => warn!("Detector failed: {}", e),
            }
        }
        
        // Fusion des détections qui se chevauchent
        let merged = self.merge_detections(all_detections)
            .context("Failed to merge detections")?;
            
        // Filtrage par confiance
        let filtered: Vec<Detection> = merged.into_iter()
            .filter(|d| d.confidence.value >= self.config.confidence_threshold)
            .collect();
            
        info!("Found {} watermarks after filtering", filtered.len());
        
        Ok(filtered)
    }
    
    /// Détecte les filigranes dans un lot d'images
    #[instrument(skip(self, images), fields(batch_size = images.len()))]
    pub fn detect_batch(&self, images: &[Mat]) -> Result<Vec<Vec<Detection>>> {
        debug!("Starting batch detection for {} images", images.len());
        
        // Validation du batch
        if images.is_empty() {
            return Ok(vec![]);
        }
        
        // Traitement parallèle avec rayon
        let results: Vec<Result<Vec<Detection>>> = images.par_iter()
            .map(|image| self.detect(image))
            .collect();
            
        // Séparation des succès et des erreurs
        let (successes, errors): (Vec<_>, Vec<_>) = results.into_iter()
            .partition(Result::is_ok);
            
        // Log des erreurs
        for error in errors {
            if let Err(e) = error {
                warn!("Batch detection error: {}", e);
            }
        }
        
        Ok(successes.into_iter()
            .filter_map(Result::ok)
            .collect())
    }
    
    // Méthodes privées
    
    fn preprocess_image(&self, image: &Mat) -> Result<Mat> {
        let mut processed = Mat::default();
        
        // Redimensionnement si nécessaire
        let (max_width, max_height) = self.config.max_image_size;
        let size = image.size()?;
        
        if size.width > max_width as i32 || size.height > max_height as i32 {
            let scale = f64::min(
                max_width as f64 / size.width as f64,
                max_height as f64 / size.height as f64
            );
            
            let new_size = Size::new(
                (size.width as f64 * scale) as i32,
                (size.height as f64 * scale) as i32
            );
            
            imgproc::resize(
                image,
                &mut processed,
                new_size,
                0.0,
                0.0,
                imgproc::INTER_AREA
            )?;
        } else {
            image.copy_to(&mut processed)?;
        }
        
        // Prétraitement supplémentaire
        preprocess_image(&mut processed)
            .context("Failed to apply additional preprocessing")?;
            
        Ok(processed)
    }
    
    fn merge_detections(&self, detections: Vec<Detection>) -> Result<Vec<Detection>> {
        if detections.is_empty() {
            return Ok(vec![]);
        }
        
        let mut merged = Vec::new();
        let mut used = vec![false; detections.len()];
        
        for i in 0..detections.len() {
            if used[i] {
                continue;
            }
            
            let mut current = detections[i].clone();
            used[i] = true;
            
            for j in (i + 1)..detections.len() {
                if used[j] {
                    continue;
                }
                
                let other = &detections[j];
                if current.bbox.iou(&other.bbox) >= self.config.iou_threshold {
                    // Fusion des boîtes englobantes et des confiances
                    current.bbox = current.bbox.merge(&other.bbox);
                    current.confidence = Confidence::new(
                        (current.confidence.value + other.confidence.value) / 2.0
                    );
                    used[j] = true;
                }
            }
            
            merged.push(current);
        }
        
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::imgcodecs;
    use std::path::PathBuf;
    
    fn load_test_image(name: &str) -> Result<Mat> {
        let test_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("resources")
            .join("images");
            
        let path = test_dir.join(name);
        Ok(imgcodecs::imread(
            path.to_str().unwrap(),
            imgcodecs::IMREAD_COLOR
        )?)
    }
    
    #[test]
    fn test_detection_engine_creation() -> Result<()> {
        let config = DetectionConfig::default();
        let engine = DetectionEngine::new(config)?;
        Ok(())
    }
    
    #[test]
    fn test_detect_logo() -> Result<()> {
        let config = DetectionConfig::default();
        let engine = DetectionEngine::new(config)?;
        
        let image = load_test_image("logo_sample.png")?;
        let detections = engine.detect(&image)?;
        
        assert!(!detections.is_empty());
        assert!(detections.iter().any(|d| d.watermark_type == WatermarkType::Logo));
        Ok(())
    }
    
    #[test]
    fn test_detect_text() -> Result<()> {
        let config = DetectionConfig::default();
        let engine = DetectionEngine::new(config)?;
        
        let image = load_test_image("text_sample.png")?;
        let detections = engine.detect(&image)?;
        
        assert!(!detections.is_empty());
        assert!(detections.iter().any(|d| d.watermark_type == WatermarkType::Text));
        Ok(())
    }
    
    #[test]
    fn test_batch_detection() -> Result<()> {
        let config = DetectionConfig::default();
        let engine = DetectionEngine::new(config)?;
        
        let images = vec![
            load_test_image("sample1.png")?,
            load_test_image("sample2.png")?,
        ];
        
        let results = engine.detect_batch(&images)?;
        assert_eq!(results.len(), 2);
        Ok(())
    }
    
    #[test]
    fn test_invalid_image() -> Result<()> {
        let config = DetectionConfig::default();
        let engine = DetectionEngine::new(config)?;
        
        let mut invalid = Mat::new_rows_cols(100, 100, CV_8UC3)?;
        let result = engine.detect(&invalid);
        assert!(result.is_err());
        Ok(())
    }
}

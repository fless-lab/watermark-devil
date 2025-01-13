mod models;
mod frequency;
mod utils;

use std::sync::Arc;
use opencv::core::Mat;
use anyhow::Result;
use tracing::{info, debug};

use crate::types::{Detection, WatermarkType};
use crate::config::DetectionConfig;
use models::{EnsembleDetector, WatermarkDetector};

pub struct DetectionEngine {
    config: DetectionConfig,
    ensemble: Arc<EnsembleDetector>,
}

impl DetectionEngine {
    pub fn new(config: DetectionConfig) -> Result<Self> {
        Ok(Self {
            ensemble: Arc::new(EnsembleDetector::new(&config)?),
            config,
        })
    }
    
    pub fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        debug!("Starting detection on image {}x{}", image.cols(), image.rows());
        
        // Run ensemble detection
        let detections = self.ensemble.detect(image)?;
        
        info!("Found {} watermarks", detections.len());
        for detection in &detections {
            debug!(
                "Watermark: type={:?}, confidence={:.2}, bbox={:?}",
                detection.watermark_type,
                detection.confidence.value,
                detection.bbox
            );
        }
        
        Ok(detections)
    }
    
    pub fn detect_batch(&self, images: &[Mat]) -> Result<Vec<Vec<Detection>>> {
        debug!("Starting batch detection on {} images", images.len());
        
        let results: Result<Vec<_>> = images.iter()
            .map(|image| self.detect(image))
            .collect();
            
        results
    }
}

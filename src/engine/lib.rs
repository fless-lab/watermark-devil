//! Core engine for watermark removal
//! This module contains the high-performance Rust implementation
//! of watermark detection and removal algorithms.

use std::sync::Arc;
use parking_lot::RwLock;
use rayon::prelude::*;
use anyhow::{Result, Context};
use tracing::{info, debug, error, instrument};
use image::{DynamicImage, ImageBuffer};
use opencv::prelude::*;
use opencv::core::Mat;

mod detection;
mod reconstruction;
mod optimization;
mod learning;
mod error;
mod config;

pub use error::EngineError;
pub use config::EngineConfig;
use crate::types::{ProcessingResult, ImageMetadata};

/// Main engine structure that coordinates all watermark removal operations
#[derive(Clone)]
pub struct WatermarkEngine {
    /// Configuration for the engine
    config: Arc<EngineConfig>,
    /// Detection engine for identifying watermarks
    detection: Arc<detection::DetectionEngine>,
    /// Reconstruction engine for removing watermarks
    reconstruction: Arc<reconstruction::ReconstructionEngine>,
    /// Optimization engine for performance
    optimization: Arc<optimization::OptimizationEngine>,
    /// Adaptive learning system
    learning: Arc<RwLock<learning::AdaptiveLearning>>,
}

impl Default for WatermarkEngine {
    fn default() -> Self {
        Self::new(EngineConfig::default())
    }
}

impl WatermarkEngine {
    /// Creates a new instance of the watermark removal engine with custom configuration
    pub fn new(config: EngineConfig) -> Self {
        let config = Arc::new(config);
        
        Self {
            detection: Arc::new(detection::DetectionEngine::new(config.detection.clone())),
            reconstruction: Arc::new(reconstruction::ReconstructionEngine::new(config.reconstruction.clone())),
            optimization: Arc::new(optimization::OptimizationEngine::new(config.optimization.clone())),
            learning: Arc::new(RwLock::new(learning::AdaptiveLearning::new(config.learning.clone()))),
            config: config.clone(),
        }
    }

    /// Processes an image to remove watermarks
    #[instrument(skip(self, image_data), fields(image_size = image_data.len()))]
    pub async fn process_image(&self, image_data: &[u8]) -> Result<ProcessingResult> {
        debug!("Starting image processing");
        
        // Convert image data to OpenCV format
        let image = self.load_image(image_data)
            .context("Failed to load image")?;
            
        // Optimize image for processing
        let optimized = self.optimization.optimize_image(&image)
            .context("Failed to optimize image")?;
            
        // Detect watermarks
        let detections = self.detection.detect(&optimized)
            .context("Failed to detect watermarks")?;
            
        info!("Found {} watermarks", detections.len());
        
        if detections.is_empty() {
            debug!("No watermarks detected, returning original image");
            return Ok(ProcessingResult {
                image: image_data.to_vec(),
                detections: vec![],
                metadata: ImageMetadata::new(&image),
            });
        }
        
        // Reconstruct image
        let reconstructed = self.reconstruction.reconstruct(&optimized, &detections)
            .await
            .context("Failed to reconstruct image")?;
            
        // Update learning system
        if self.config.learning_enabled {
            if let Err(e) = self.learning.write().update(&detections, &reconstructed).await {
                error!("Failed to update learning system: {}", e);
                // Continue processing even if learning update fails
            }
        }
        
        // Convert back to bytes
        let final_image = self.save_image(&reconstructed)
            .context("Failed to save processed image")?;
            
        Ok(ProcessingResult {
            image: final_image,
            detections,
            metadata: ImageMetadata::new(&reconstructed),
        })
    }
    
    /// Process multiple images in parallel
    #[instrument(skip(self, images), fields(batch_size = images.len()))]
    pub async fn process_batch(&self, images: &[Vec<u8>]) -> Result<Vec<ProcessingResult>> {
        debug!("Processing batch of {} images", images.len());
        
        let results: Vec<Result<ProcessingResult>> = images.par_iter()
            .map(|img| {
                tokio::runtime::Handle::current()
                    .block_on(self.process_image(img))
            })
            .collect();
            
        // Aggregate results and errors
        let (successes, errors): (Vec<_>, Vec<_>) = results
            .into_iter()
            .partition(Result::is_ok);
            
        // Log any errors
        for error in &errors {
            if let Err(e) = error {
                error!("Batch processing error: {}", e);
            }
        }
        
        info!(
            "Batch processing complete. Successful: {}, Failed: {}", 
            successes.len(), 
            errors.len()
        );
        
        // Convert to Vec<ProcessingResult>
        Ok(successes.into_iter().filter_map(Result::ok).collect())
    }
    
    // Private helper methods
    
    fn load_image(&self, data: &[u8]) -> Result<Mat> {
        let vector = Mat::from_slice(data)?;
        opencv::imgcodecs::imdecode(&vector, opencv::imgcodecs::IMREAD_COLOR)
            .map_err(|e| EngineError::ImageLoadError(e.to_string()).into())
    }
    
    fn save_image(&self, image: &Mat) -> Result<Vec<u8>> {
        let mut buffer = Vector::new();
        opencv::imgcodecs::imencode(".png", image, &mut buffer, &Vector::new())?;
        Ok(buffer.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tokio::fs;
    
    async fn load_test_image(name: &str) -> Result<Vec<u8>> {
        let test_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("resources")
            .join("images");
        Ok(fs::read(test_dir.join(name)).await?)
    }
    
    #[tokio::test]
    async fn test_engine_creation() {
        let engine = WatermarkEngine::default();
        assert!(engine.config.learning_enabled);
    }
    
    #[tokio::test]
    async fn test_process_image() -> Result<()> {
        let engine = WatermarkEngine::default();
        let image_data = load_test_image("sample.png").await?;
        
        let result = engine.process_image(&image_data).await?;
        assert!(!result.image.is_empty());
        Ok(())
    }
    
    #[tokio::test]
    async fn test_process_batch() -> Result<()> {
        let engine = WatermarkEngine::default();
        let images = vec![
            load_test_image("sample1.png").await?,
            load_test_image("sample2.png").await?,
        ];
        
        let results = engine.process_batch(&images).await?;
        assert_eq!(results.len(), 2);
        Ok(())
    }
    
    #[tokio::test]
    async fn test_invalid_image() {
        let engine = WatermarkEngine::default();
        let invalid_data = vec![0u8; 100];
        
        let result = engine.process_image(&invalid_data).await;
        assert!(result.is_err());
    }
}

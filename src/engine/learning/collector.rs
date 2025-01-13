use std::sync::Arc;
use tokio::sync::RwLock;
use opencv::core::Mat;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use tracing::{info, debug, warn};

use crate::types::{Detection, WatermarkType};
use crate::config::LearningConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub watermark_type: WatermarkType,
    pub confidence: f32,
    pub success: bool,
    pub metrics: ExampleMetrics,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleMetrics {
    pub psnr: f32,
    pub ssim: f32,
    pub fid: Option<f32>,
    pub processing_time_ms: u64,
}

pub struct DataCollector {
    config: LearningConfig,
    storage: Arc<RwLock<DataStorage>>,
}

impl DataCollector {
    pub fn new(config: &LearningConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            storage: Arc::new(RwLock::new(DataStorage::new(config)?)),
        })
    }
    
    pub async fn collect_example(
        &self,
        image: &Mat,
        detection: &Detection,
        result: &Mat,
        metrics: ExampleMetrics,
    ) -> Result<()> {
        if !self.config.collect_data {
            return Ok(());
        }
        
        let example = TrainingExample {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            watermark_type: detection.watermark_type.clone(),
            confidence: detection.confidence.value,
            success: metrics.psnr > 35.0 && metrics.ssim > 0.95,
            metrics,
            metadata: detection.metadata.clone().unwrap_or(serde_json::json!({})),
        };
        
        // Store example
        let mut storage = self.storage.write().await;
        storage.store_example(&example, image, result).await?;
        
        // Log collection
        info!(
            "Collected training example: id={}, type={:?}, success={}",
            example.id, example.watermark_type, example.success
        );
        
        Ok(())
    }
    
    pub async fn get_training_examples(&self, limit: usize) -> Result<Vec<TrainingExample>> {
        let storage = self.storage.read().await;
        storage.get_examples(limit).await
    }
    
    pub async fn get_statistics(&self) -> Result<CollectionStatistics> {
        let storage = self.storage.read().await;
        storage.get_statistics().await
    }
}

struct DataStorage {
    config: LearningConfig,
    database: sled::Db,
}

impl DataStorage {
    fn new(config: &LearningConfig) -> Result<Self> {
        let database = sled::Config::new()
            .path(&config.data_path)
            .cache_capacity(1024 * 1024 * 1024) // 1GB cache
            .flush_every_ms(Some(1000))
            .open()?;
        
        Ok(Self {
            config: config.clone(),
            database,
        })
    }
    
    async fn store_example(
        &mut self,
        example: &TrainingExample,
        image: &Mat,
        result: &Mat,
    ) -> Result<()> {
        // Serialize example
        let example_data = bincode::serialize(example)?;
        
        // Store example metadata
        self.database.insert(
            format!("example:{}", example.id).as_bytes(),
            example_data,
        )?;
        
        // Store images if enabled
        if self.config.store_images {
            let image_path = self.config.data_path.join("images").join(&example.id);
            let result_path = self.config.data_path.join("results").join(&example.id);
            
            tokio::fs::create_dir_all(image_path.parent().unwrap()).await?;
            tokio::fs::create_dir_all(result_path.parent().unwrap()).await?;
            
            opencv::imgcodecs::imwrite(
                image_path.to_str().unwrap(),
                image,
                &opencv::core::Vector::new(),
            )?;
            
            opencv::imgcodecs::imwrite(
                result_path.to_str().unwrap(),
                result,
                &opencv::core::Vector::new(),
            )?;
        }
        
        Ok(())
    }
    
    async fn get_examples(&self, limit: usize) -> Result<Vec<TrainingExample>> {
        let mut examples = Vec::new();
        
        for item in self.database.scan_prefix(b"example:") {
            let (_, value) = item?;
            let example: TrainingExample = bincode::deserialize(&value)?;
            examples.push(example);
            
            if examples.len() >= limit {
                break;
            }
        }
        
        Ok(examples)
    }
    
    async fn get_statistics(&self) -> Result<CollectionStatistics> {
        let mut stats = CollectionStatistics::default();
        
        for item in self.database.scan_prefix(b"example:") {
            let (_, value) = item?;
            let example: TrainingExample = bincode::deserialize(&value)?;
            
            stats.total_examples += 1;
            if example.success {
                stats.successful_examples += 1;
            }
            
            match example.watermark_type {
                WatermarkType::Logo => stats.logo_examples += 1,
                WatermarkType::Text => stats.text_examples += 1,
                WatermarkType::Pattern => stats.pattern_examples += 1,
                WatermarkType::Transparent => stats.transparent_examples += 1,
            }
            
            stats.avg_psnr += example.metrics.psnr;
            stats.avg_ssim += example.metrics.ssim;
            if let Some(fid) = example.metrics.fid {
                stats.avg_fid += fid;
                stats.fid_count += 1;
            }
        }
        
        // Calculate averages
        if stats.total_examples > 0 {
            stats.avg_psnr /= stats.total_examples as f32;
            stats.avg_ssim /= stats.total_examples as f32;
            if stats.fid_count > 0 {
                stats.avg_fid /= stats.fid_count as f32;
            }
        }
        
        Ok(stats)
    }
}

#[derive(Debug, Default)]
pub struct CollectionStatistics {
    pub total_examples: usize,
    pub successful_examples: usize,
    pub logo_examples: usize,
    pub text_examples: usize,
    pub pattern_examples: usize,
    pub transparent_examples: usize,
    pub avg_psnr: f32,
    pub avg_ssim: f32,
    pub avg_fid: f32,
    fid_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_data_collection() -> Result<()> {
        let temp_dir = tempdir()?;
        
        let config = LearningConfig {
            collect_data: true,
            store_images: true,
            data_path: PathBuf::from(temp_dir.path()),
            min_confidence: 0.8,
            max_examples: 1000000,
        };
        
        let collector = DataCollector::new(&config)?;
        
        // Create test data
        let image = Mat::new_rows_cols_with_default(
            100,
            100,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::all(255.0),
        )?;
        
        let detection = Detection {
            watermark_type: WatermarkType::Logo,
            confidence: crate::types::Confidence::new(0.95),
            bbox: opencv::core::Rect::new(0, 0, 50, 50),
            metadata: None,
        };
        
        let metrics = ExampleMetrics {
            psnr: 36.0,
            ssim: 0.96,
            fid: Some(15.0),
            processing_time_ms: 100,
        };
        
        // Collect example
        collector.collect_example(&image, &detection, &image, metrics).await?;
        
        // Verify collection
        let examples = collector.get_training_examples(10).await?;
        assert_eq!(examples.len(), 1);
        
        let stats = collector.get_statistics().await?;
        assert_eq!(stats.total_examples, 1);
        assert_eq!(stats.successful_examples, 1);
        assert_eq!(stats.logo_examples, 1);
        
        Ok(())
    }
}

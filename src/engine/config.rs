use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use dotenv::dotenv;
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub processing: ProcessingConfig,
    pub detection: DetectionConfig,
    pub reconstruction: ReconstructionConfig,
    pub learning: LearningConfig,
    pub storage: StorageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConfig {
    pub use_gpu: bool,
    pub batch_size: usize,
    pub temporal_window_size: usize,
    pub quality: i32,
    pub hardware_acceleration: bool,
    pub streaming_mode: bool,
    pub buffer_size: usize,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            use_gpu: true,
            batch_size: 16,
            temporal_window_size: 5,
            quality: 28,
            hardware_acceleration: true,
            streaming_mode: false,
            buffer_size: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub gpu_enabled: bool,
    pub cuda_devices: Vec<i32>,
    pub max_batch_size: usize,
    pub processing_timeout: u64,
    pub default_image_size: u32,
    pub preserve_exif: bool,
    pub output_quality: u8,
    pub video: VideoConfig,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            gpu_enabled: true,
            cuda_devices: vec![0],
            max_batch_size: 16,
            processing_timeout: 300,
            default_image_size: 1024,
            preserve_exif: false,
            output_quality: 95,
            video: VideoConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    pub max_detections: usize,
    pub model_path: PathBuf,
    pub pattern_db_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionConfig {
    pub quality: String,
    pub method: String,
    pub model_path: PathBuf,
    pub temp_files_path: PathBuf,
    pub use_gpu: bool,
    pub preserve_details: bool,
    pub max_iterations: usize,
    pub window_size: usize,
    pub overlap: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    pub enabled: bool,
    pub min_samples_for_training: usize,
    pub training_interval: u64,
    pub model_backup_enabled: bool,
    pub model_backup_path: PathBuf,
    pub export_metrics: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub storage_type: String,
    pub local_path: PathBuf,
    pub s3_bucket: Option<String>,
    pub s3_region: Option<String>,
    pub s3_access_key: Option<String>,
    pub s3_secret_key: Option<String>,
}

pub struct ConfigManager {
    config: Arc<RwLock<EngineConfig>>,
}

impl ConfigManager {
    pub fn new() -> Result<Self> {
        dotenv().ok();
        
        let config = Self::load_config()?;
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
        })
    }

    fn load_config() -> Result<EngineConfig> {
        let processing = ProcessingConfig {
            gpu_enabled: std::env::var("GPU_ENABLED")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            cuda_devices: std::env::var("CUDA_VISIBLE_DEVICES")
                .unwrap_or_else(|_| "0".to_string())
                .split(',')
                .filter_map(|s| s.parse().ok())
                .collect(),
            max_batch_size: std::env::var("MAX_BATCH_SIZE")
                .unwrap_or_else(|_| "16".to_string())
                .parse()
                .unwrap_or(16),
            processing_timeout: std::env::var("PROCESSING_TIMEOUT")
                .unwrap_or_else(|_| "300".to_string())
                .parse()
                .unwrap_or(300),
            default_image_size: std::env::var("DEFAULT_IMAGE_SIZE")
                .unwrap_or_else(|_| "1024".to_string())
                .parse()
                .unwrap_or(1024),
            preserve_exif: std::env::var("PRESERVE_EXIF")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false),
            output_quality: std::env::var("OUTPUT_QUALITY")
                .unwrap_or_else(|_| "95".to_string())
                .parse()
                .unwrap_or(95),
            video: VideoConfig {
                use_gpu: std::env::var("VIDEO_USE_GPU")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
                batch_size: std::env::var("VIDEO_BATCH_SIZE")
                    .unwrap_or_else(|_| "16".to_string())
                    .parse()
                    .unwrap_or(16),
                temporal_window_size: std::env::var("VIDEO_TEMPORAL_WINDOW_SIZE")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()
                    .unwrap_or(5),
                quality: std::env::var("VIDEO_QUALITY")
                    .unwrap_or_else(|_| "28".to_string())
                    .parse()
                    .unwrap_or(28),
                hardware_acceleration: std::env::var("VIDEO_HARDWARE_ACCELERATION")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
                streaming_mode: std::env::var("VIDEO_STREAMING_MODE")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()
                    .unwrap_or(false),
                buffer_size: std::env::var("VIDEO_BUFFER_SIZE")
                    .unwrap_or_else(|_| "3".to_string())
                    .parse()
                    .unwrap_or(3),
            },
        };

        let detection = DetectionConfig {
            confidence_threshold: std::env::var("DETECTION_CONFIDENCE_THRESHOLD")
                .unwrap_or_else(|_| "0.5".to_string())
                .parse()
                .unwrap_or(0.5),
            iou_threshold: std::env::var("DETECTION_IOU_THRESHOLD")
                .unwrap_or_else(|_| "0.3".to_string())
                .parse()
                .unwrap_or(0.3),
            max_detections: std::env::var("DETECTION_MAX_DETECTIONS")
                .unwrap_or_else(|_| "10".to_string())
                .parse()
                .unwrap_or(10),
            model_path: PathBuf::from(
                std::env::var("DETECTION_MODEL_PATH")
                    .unwrap_or_else(|_| "models/detection".to_string())
            ),
            pattern_db_path: PathBuf::from(
                std::env::var("PATTERN_DB_PATH")
                    .unwrap_or_else(|_| "data/patterns".to_string())
            ),
        };

        let reconstruction = ReconstructionConfig {
            quality: std::env::var("RECONSTRUCTION_QUALITY")
                .unwrap_or_else(|_| "high".to_string()),
            method: std::env::var("RECONSTRUCTION_METHOD")
                .unwrap_or_else(|_| "hybrid".to_string()),
            model_path: PathBuf::from(
                std::env::var("RECONSTRUCTION_MODEL_PATH")
                    .unwrap_or_else(|_| "models/reconstruction".to_string())
            ),
            temp_files_path: PathBuf::from(
                std::env::var("TEMP_FILES_PATH")
                    .unwrap_or_else(|_| "temp".to_string())
            ),
            use_gpu: std::env::var("RECONSTRUCTION_USE_GPU")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            preserve_details: std::env::var("RECONSTRUCTION_PRESERVE_DETAILS")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            max_iterations: std::env::var("RECONSTRUCTION_MAX_ITERATIONS")
                .unwrap_or_else(|_| "100".to_string())
                .parse()
                .unwrap_or(100),
            window_size: std::env::var("RECONSTRUCTION_WINDOW_SIZE")
                .unwrap_or_else(|_| "256".to_string())
                .parse()
                .unwrap_or(256),
            overlap: std::env::var("RECONSTRUCTION_OVERLAP")
                .unwrap_or_else(|_| "128".to_string())
                .parse()
                .unwrap_or(128),
        };

        let learning = LearningConfig {
            enabled: std::env::var("LEARNING_ENABLED")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            min_samples_for_training: std::env::var("MIN_SAMPLES_FOR_TRAINING")
                .unwrap_or_else(|_| "1000".to_string())
                .parse()
                .unwrap_or(1000),
            training_interval: std::env::var("TRAINING_INTERVAL")
                .unwrap_or_else(|_| "86400".to_string())
                .parse()
                .unwrap_or(86400),
            model_backup_enabled: std::env::var("MODEL_BACKUP_ENABLED")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            model_backup_path: PathBuf::from(
                std::env::var("MODEL_BACKUP_PATH")
                    .unwrap_or_else(|_| "backups".to_string())
            ),
            export_metrics: std::env::var("EXPORT_METRICS")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
        };

        let storage = StorageConfig {
            storage_type: std::env::var("STORAGE_TYPE")
                .unwrap_or_else(|_| "local".to_string()),
            local_path: PathBuf::from(
                std::env::var("LOCAL_STORAGE_PATH")
                    .unwrap_or_else(|_| "storage".to_string())
            ),
            s3_bucket: std::env::var("S3_BUCKET").ok(),
            s3_region: std::env::var("S3_REGION").ok(),
            s3_access_key: std::env::var("S3_ACCESS_KEY").ok(),
            s3_secret_key: std::env::var("S3_SECRET_KEY").ok(),
        };

        Ok(EngineConfig {
            processing,
            detection,
            reconstruction,
            learning,
            storage,
        })
    }

    pub async fn get_config(&self) -> Arc<EngineConfig> {
        Arc::new(self.config.read().await.clone())
    }

    pub async fn update_config(&self, new_config: EngineConfig) -> Result<()> {
        let mut config = self.config.write().await;
        *config = new_config;
        Ok(())
    }

    pub async fn reload_config(&self) -> Result<()> {
        let new_config = Self::load_config()?;
        self.update_config(new_config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_config_manager() {
        // Create temporary directories for testing
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("models");
        std::fs::create_dir_all(&model_path).unwrap();

        // Set test environment variables
        std::env::set_var("DETECTION_MODEL_PATH", model_path.to_str().unwrap());
        std::env::set_var("GPU_ENABLED", "true");
        std::env::set_var("MAX_BATCH_SIZE", "32");

        // Create config manager
        let config_manager = ConfigManager::new().unwrap();
        let config = config_manager.get_config().await;

        // Verify configuration
        assert!(config.processing.gpu_enabled);
        assert_eq!(config.processing.max_batch_size, 32);
        assert_eq!(config.detection.model_path, model_path);
    }
}

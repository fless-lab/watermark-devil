use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use tracing::{info, debug, warn};

use crate::types::WatermarkType;
use super::config::LearningConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub watermark_type: WatermarkType,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub version: String,
    pub accuracy: f32,
    pub file_size: u64,
    pub metadata: serde_json::Value,
}

pub struct ModelManager {
    config: LearningConfig,
    models: Arc<RwLock<ModelRegistry>>,
}

impl ModelManager {
    pub fn new(config: &LearningConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            models: Arc::new(RwLock::new(ModelRegistry::new(config)?)),
        })
    }
    
    pub async fn load_model(&self, watermark_type: WatermarkType) -> Result<PyObject> {
        let registry = self.models.read().await;
        registry.load_model(watermark_type).await
    }
    
    pub async fn save_model(
        &self,
        watermark_type: WatermarkType,
        model: &PyObject,
        accuracy: f32,
    ) -> Result<()> {
        let mut registry = self.models.write().await;
        registry.save_model(watermark_type, model, accuracy).await
    }
    
    pub async fn get_model_info(&self, watermark_type: WatermarkType) -> Result<Option<ModelInfo>> {
        let registry = self.models.read().await;
        registry.get_model_info(watermark_type).await
    }
    
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let registry = self.models.read().await;
        registry.list_models().await
    }
    
    pub async fn cleanup_old_models(&self) -> Result<usize> {
        let mut registry = self.models.write().await;
        registry.cleanup_old_models().await
    }
}

struct ModelRegistry {
    config: LearningConfig,
    models_dir: PathBuf,
    db: sled::Db,
}

impl ModelRegistry {
    fn new(config: &LearningConfig) -> Result<Self> {
        let models_dir = config.data_path.join("models");
        std::fs::create_dir_all(&models_dir)?;
        
        let db = sled::Config::new()
            .path(config.data_path.join("models.db"))
            .cache_capacity(1024 * 1024 * 1024) // 1GB cache
            .flush_every_ms(Some(1000))
            .open()?;
        
        Ok(Self {
            config: config.clone(),
            models_dir,
            db,
        })
    }
    
    async fn load_model(&self, watermark_type: WatermarkType) -> Result<PyObject> {
        let model_path = self.get_model_path(watermark_type);
        
        if !model_path.exists() {
            return Err(anyhow::anyhow!("Model not found for {:?}", watermark_type));
        }
        
        Python::with_gil(|py| {
            let torch = PyModule::import(py, "torch")?;
            
            // Load model architecture based on type
            let model = match watermark_type {
                WatermarkType::Logo => {
                    PyModule::import(py, "ml.models.logo_detector.model")?
                        .getattr("LogoDetector")?
                        .call0()?
                }
                WatermarkType::Text => {
                    PyModule::import(py, "ml.models.text_detector.model")?
                        .getattr("TextDetector")?
                        .call0()?
                }
                WatermarkType::Pattern => {
                    PyModule::import(py, "ml.models.pattern_detector.model")?
                        .getattr("PatternDetector")?
                        .call0()?
                }
                WatermarkType::Transparent => {
                    PyModule::import(py, "ml.models.transparency_detector.model")?
                        .getattr("TransparencyDetector")?
                        .call0()?
                }
            };
            
            // Load state dict
            let state_dict = torch.getattr("load")?.call1((model_path.to_str().unwrap(),))?;
            model.call_method1("load_state_dict", (state_dict,))?;
            
            Ok(model.into())
        })
    }
    
    async fn save_model(
        &mut self,
        watermark_type: WatermarkType,
        model: &PyObject,
        accuracy: f32,
    ) -> Result<()> {
        let model_path = self.get_model_path(watermark_type);
        
        // Save model weights
        Python::with_gil(|py| {
            let torch = PyModule::import(py, "torch")?;
            let state_dict = model.call_method0(py, "state_dict")?;
            torch.getattr("save")?.call1((state_dict, model_path.to_str().unwrap()))?;
            Ok(())
        })?;
        
        // Update model info
        let info = ModelInfo {
            id: uuid::Uuid::new_v4().to_string(),
            watermark_type,
            created_at: Utc::now(),
            last_updated: Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            accuracy,
            file_size: std::fs::metadata(&model_path)?.len(),
            metadata: serde_json::json!({
                "framework": "pytorch",
                "cuda_available": Python::with_gil(|py| {
                    PyModule::import(py, "torch")?.getattr("cuda")?.getattr("is_available")?.call0()?.extract::<bool>()
                })?,
            }),
        };
        
        let info_data = bincode::serialize(&info)?;
        self.db.insert(
            format!("model:{}:{}", watermark_type as u8, info.id).as_bytes(),
            info_data,
        )?;
        
        info!(
            "Saved model for {:?}: id={}, accuracy={:.4}, size={}B",
            watermark_type, info.id, info.accuracy, info.file_size
        );
        
        Ok(())
    }
    
    async fn get_model_info(&self, watermark_type: WatermarkType) -> Result<Option<ModelInfo>> {
        let mut latest_info = None;
        let mut latest_time = DateTime::<Utc>::MIN_UTC;
        
        let prefix = format!("model:{}", watermark_type as u8);
        for item in self.db.scan_prefix(prefix.as_bytes()) {
            let (_, value) = item?;
            let info: ModelInfo = bincode::deserialize(&value)?;
            
            if info.last_updated > latest_time {
                latest_time = info.last_updated;
                latest_info = Some(info);
            }
        }
        
        Ok(latest_info)
    }
    
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let mut models = Vec::new();
        
        for item in self.db.scan_prefix(b"model:") {
            let (_, value) = item?;
            let info: ModelInfo = bincode::deserialize(&value)?;
            models.push(info);
        }
        
        models.sort_by(|a, b| b.last_updated.cmp(&a.last_updated));
        Ok(models)
    }
    
    async fn cleanup_old_models(&mut self) -> Result<usize> {
        let max_age = chrono::Duration::days(self.config.max_model_age_days as i64);
        let now = Utc::now();
        let mut removed = 0;
        
        let models = self.list_models().await?;
        for model in models {
            if now - model.last_updated > max_age {
                // Remove model file
                let path = self.get_model_path(model.watermark_type);
                if path.exists() {
                    std::fs::remove_file(path)?;
                }
                
                // Remove from database
                self.db.remove(
                    format!("model:{}:{}", model.watermark_type as u8, model.id).as_bytes()
                )?;
                
                removed += 1;
            }
        }
        
        if removed > 0 {
            info!("Removed {} old models", removed);
        }
        
        Ok(removed)
    }
    
    fn get_model_path(&self, watermark_type: WatermarkType) -> PathBuf {
        self.models_dir.join(format!("{:?}.pt", watermark_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_model_management() -> Result<()> {
        let temp_dir = tempdir()?;
        
        let config = LearningConfig {
            collect_data: true,
            store_images: true,
            data_path: PathBuf::from(temp_dir.path()),
            min_confidence: 0.8,
            max_examples: 1000000,
            batch_size: 32,
            learning_rate: 0.001,
            epochs: 100,
            early_stopping_patience: 10,
            validation_split: 0.2,
            augmentation_enabled: true,
            model_type: super::config::ModelType::Hybrid,
            hidden_layers: vec![512, 256, 128],
            dropout_rate: 0.3,
            activation_function: super::config::ActivationFunction::GELU,
            optimizer_type: super::config::OptimizerType::AdamW,
            weight_decay: 0.01,
            momentum: 0.9,
            scheduler_type: super::config::SchedulerType::OneCycleLR,
            min_examples_for_training: 1000,
            training_interval_hours: 24,
            max_model_age_days: 30,
            performance_threshold: 0.95,
        };
        
        let manager = ModelManager::new(&config)?;
        
        // Create and save a test model
        Python::with_gil(|py| {
            let model = PyModule::import(py, "ml.models.logo_detector.model")?
                .getattr("LogoDetector")?
                .call0()?;
                
            manager.save_model(WatermarkType::Logo, &model.into(), 0.95).await?;
            
            Ok::<_, anyhow::Error>(())
        })?;
        
        // List models
        let models = manager.list_models().await?;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].watermark_type, WatermarkType::Logo);
        assert!(models[0].accuracy > 0.9);
        
        // Get model info
        let info = manager.get_model_info(WatermarkType::Logo).await?;
        assert!(info.is_some());
        
        // Cleanup
        let removed = manager.cleanup_old_models().await?;
        assert_eq!(removed, 0); // Should not remove new models
        
        Ok(())
    }
}

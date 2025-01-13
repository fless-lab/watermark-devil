use std::path::PathBuf;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    // Data collection settings
    pub collect_data: bool,
    pub store_images: bool,
    pub data_path: PathBuf,
    pub min_confidence: f32,
    pub max_examples: usize,
    
    // Training settings
    pub batch_size: usize,
    pub learning_rate: f32,
    pub epochs: usize,
    pub early_stopping_patience: usize,
    pub validation_split: f32,
    pub augmentation_enabled: bool,
    
    // Model architecture settings
    pub model_type: ModelType,
    pub hidden_layers: Vec<usize>,
    pub dropout_rate: f32,
    pub activation_function: ActivationFunction,
    
    // Optimization settings
    pub optimizer_type: OptimizerType,
    pub weight_decay: f32,
    pub momentum: f32,
    pub scheduler_type: SchedulerType,
    
    // Continuous learning settings
    pub min_examples_for_training: usize,
    pub training_interval_hours: u32,
    pub max_model_age_days: u32,
    pub performance_threshold: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelType {
    CNN,
    Transformer,
    Hybrid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    LeakyReLU,
    GELU,
    Swish,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizerType {
    Adam,
    AdamW,
    SGD,
    RMSprop,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SchedulerType {
    StepLR,
    CosineAnnealing,
    ReduceLROnPlateau,
    OneCycleLR,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            // Data collection defaults
            collect_data: true,
            store_images: true,
            data_path: PathBuf::from("data/training"),
            min_confidence: 0.8,
            max_examples: 1_000_000,
            
            // Training defaults
            batch_size: 32,
            learning_rate: 0.001,
            epochs: 100,
            early_stopping_patience: 10,
            validation_split: 0.2,
            augmentation_enabled: true,
            
            // Model architecture defaults
            model_type: ModelType::Hybrid,
            hidden_layers: vec![512, 256, 128],
            dropout_rate: 0.3,
            activation_function: ActivationFunction::GELU,
            
            // Optimization defaults
            optimizer_type: OptimizerType::AdamW,
            weight_decay: 0.01,
            momentum: 0.9,
            scheduler_type: SchedulerType::OneCycleLR,
            
            // Continuous learning defaults
            min_examples_for_training: 1000,
            training_interval_hours: 24,
            max_model_age_days: 30,
            performance_threshold: 0.95,
        }
    }
}

impl LearningConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        
        if let Ok(val) = std::env::var("LEARNING_COLLECT_DATA") {
            config.collect_data = val.parse().unwrap_or(true);
        }
        if let Ok(val) = std::env::var("LEARNING_STORE_IMAGES") {
            config.store_images = val.parse().unwrap_or(true);
        }
        if let Ok(val) = std::env::var("LEARNING_DATA_PATH") {
            config.data_path = PathBuf::from(val);
        }
        if let Ok(val) = std::env::var("LEARNING_MIN_CONFIDENCE") {
            config.min_confidence = val.parse().unwrap_or(0.8);
        }
        if let Ok(val) = std::env::var("LEARNING_MAX_EXAMPLES") {
            config.max_examples = val.parse().unwrap_or(1_000_000);
        }
        if let Ok(val) = std::env::var("LEARNING_BATCH_SIZE") {
            config.batch_size = val.parse().unwrap_or(32);
        }
        if let Ok(val) = std::env::var("LEARNING_LEARNING_RATE") {
            config.learning_rate = val.parse().unwrap_or(0.001);
        }
        if let Ok(val) = std::env::var("LEARNING_EPOCHS") {
            config.epochs = val.parse().unwrap_or(100);
        }
        if let Ok(val) = std::env::var("LEARNING_EARLY_STOPPING_PATIENCE") {
            config.early_stopping_patience = val.parse().unwrap_or(10);
        }
        if let Ok(val) = std::env::var("LEARNING_VALIDATION_SPLIT") {
            config.validation_split = val.parse().unwrap_or(0.2);
        }
        if let Ok(val) = std::env::var("LEARNING_AUGMENTATION_ENABLED") {
            config.augmentation_enabled = val.parse().unwrap_or(true);
        }
        if let Ok(val) = std::env::var("LEARNING_MODEL_TYPE") {
            config.model_type = match val.to_lowercase().as_str() {
                "cnn" => ModelType::CNN,
                "transformer" => ModelType::Transformer,
                _ => ModelType::Hybrid,
            };
        }
        if let Ok(val) = std::env::var("LEARNING_HIDDEN_LAYERS") {
            config.hidden_layers = val
                .split(',')
                .filter_map(|s| s.parse().ok())
                .collect();
        }
        if let Ok(val) = std::env::var("LEARNING_DROPOUT_RATE") {
            config.dropout_rate = val.parse().unwrap_or(0.3);
        }
        if let Ok(val) = std::env::var("LEARNING_ACTIVATION_FUNCTION") {
            config.activation_function = match val.to_lowercase().as_str() {
                "relu" => ActivationFunction::ReLU,
                "leakyrelu" => ActivationFunction::LeakyReLU,
                "swish" => ActivationFunction::Swish,
                _ => ActivationFunction::GELU,
            };
        }
        if let Ok(val) = std::env::var("LEARNING_OPTIMIZER_TYPE") {
            config.optimizer_type = match val.to_lowercase().as_str() {
                "adam" => OptimizerType::Adam,
                "sgd" => OptimizerType::SGD,
                "rmsprop" => OptimizerType::RMSprop,
                _ => OptimizerType::AdamW,
            };
        }
        if let Ok(val) = std::env::var("LEARNING_WEIGHT_DECAY") {
            config.weight_decay = val.parse().unwrap_or(0.01);
        }
        if let Ok(val) = std::env::var("LEARNING_MOMENTUM") {
            config.momentum = val.parse().unwrap_or(0.9);
        }
        if let Ok(val) = std::env::var("LEARNING_SCHEDULER_TYPE") {
            config.scheduler_type = match val.to_lowercase().as_str() {
                "steplr" => SchedulerType::StepLR,
                "cosine" => SchedulerType::CosineAnnealing,
                "plateau" => SchedulerType::ReduceLROnPlateau,
                _ => SchedulerType::OneCycleLR,
            };
        }
        if let Ok(val) = std::env::var("LEARNING_MIN_EXAMPLES_FOR_TRAINING") {
            config.min_examples_for_training = val.parse().unwrap_or(1000);
        }
        if let Ok(val) = std::env::var("LEARNING_TRAINING_INTERVAL_HOURS") {
            config.training_interval_hours = val.parse().unwrap_or(24);
        }
        if let Ok(val) = std::env::var("LEARNING_MAX_MODEL_AGE_DAYS") {
            config.max_model_age_days = val.parse().unwrap_or(30);
        }
        if let Ok(val) = std::env::var("LEARNING_PERFORMANCE_THRESHOLD") {
            config.performance_threshold = val.parse().unwrap_or(0.95);
        }
        
        config
    }
    
    pub fn validate(&self) -> Result<(), String> {
        if self.min_confidence < 0.0 || self.min_confidence > 1.0 {
            return Err("min_confidence must be between 0 and 1".to_string());
        }
        if self.max_examples == 0 {
            return Err("max_examples must be greater than 0".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be greater than 0".to_string());
        }
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be greater than 0".to_string());
        }
        if self.epochs == 0 {
            return Err("epochs must be greater than 0".to_string());
        }
        if self.validation_split <= 0.0 || self.validation_split >= 1.0 {
            return Err("validation_split must be between 0 and 1".to_string());
        }
        if self.hidden_layers.is_empty() {
            return Err("hidden_layers must not be empty".to_string());
        }
        if self.dropout_rate < 0.0 || self.dropout_rate > 1.0 {
            return Err("dropout_rate must be between 0 and 1".to_string());
        }
        if self.weight_decay < 0.0 {
            return Err("weight_decay must be non-negative".to_string());
        }
        if self.momentum < 0.0 || self.momentum > 1.0 {
            return Err("momentum must be between 0 and 1".to_string());
        }
        if self.min_examples_for_training == 0 {
            return Err("min_examples_for_training must be greater than 0".to_string());
        }
        if self.training_interval_hours == 0 {
            return Err("training_interval_hours must be greater than 0".to_string());
        }
        if self.max_model_age_days == 0 {
            return Err("max_model_age_days must be greater than 0".to_string());
        }
        if self.performance_threshold <= 0.0 || self.performance_threshold > 1.0 {
            return Err("performance_threshold must be between 0 and 1".to_string());
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_validation() {
        let config = LearningConfig::default();
        assert!(config.validate().is_ok());
        
        let mut invalid_config = config.clone();
        invalid_config.min_confidence = -0.1;
        assert!(invalid_config.validate().is_err());
        
        let mut invalid_config = config.clone();
        invalid_config.batch_size = 0;
        assert!(invalid_config.validate().is_err());
        
        let mut invalid_config = config.clone();
        invalid_config.learning_rate = 0.0;
        assert!(invalid_config.validate().is_err());
    }
    
    #[test]
    fn test_config_from_env() {
        std::env::set_var("LEARNING_BATCH_SIZE", "64");
        std::env::set_var("LEARNING_LEARNING_RATE", "0.0005");
        std::env::set_var("LEARNING_MODEL_TYPE", "transformer");
        
        let config = LearningConfig::from_env();
        
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.learning_rate, 0.0005);
        matches!(config.model_type, ModelType::Transformer);
        
        std::env::remove_var("LEARNING_BATCH_SIZE");
        std::env::remove_var("LEARNING_LEARNING_RATE");
        std::env::remove_var("LEARNING_MODEL_TYPE");
    }
}

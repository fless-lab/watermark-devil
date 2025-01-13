use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use dotenv::dotenv;
use anyhow::Result;

/// Configuration principale du moteur de traitement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Configuration de la détection
    pub detection: DetectionConfig,
    /// Configuration de la reconstruction
    pub reconstruction: ReconstructionConfig,
    /// Configuration de l'optimisation
    pub optimization: OptimizationConfig,
    /// Configuration du système d'apprentissage
    pub learning: LearningConfig,
    /// Active ou désactive le système d'apprentissage
    pub learning_enabled: bool,
    /// Nombre maximum de threads pour le traitement parallèle
    pub max_threads: usize,
    /// Taille maximale de batch pour le traitement par lots
    pub max_batch_size: usize,
    /// Répertoire pour les fichiers temporaires
    pub temp_dir: PathBuf,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            detection: DetectionConfig::default(),
            reconstruction: ReconstructionConfig::default(),
            optimization: OptimizationConfig::default(),
            learning: LearningConfig::default(),
            learning_enabled: true,
            max_threads: num_cpus::get(),
            max_batch_size: 32,
            temp_dir: std::env::temp_dir().join("watermark-evil"),
        }
    }
}

/// Configuration du moteur de détection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// Seuil de confiance minimum pour les détections
    pub confidence_threshold: f32,
    /// Seuil IoU pour le NMS
    pub iou_threshold: f32,
    /// Taille maximale d'image en entrée
    pub max_image_size: (u32, u32),
    /// Utiliser le GPU si disponible
    pub use_gpu: bool,
    /// Modèles à utiliser
    pub models: Vec<DetectionModel>,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            iou_threshold: 0.45,
            max_image_size: (4096, 4096),
            use_gpu: true,
            models: vec![
                DetectionModel::Logo,
                DetectionModel::Text,
                DetectionModel::Pattern,
            ],
        }
    }
}

/// Configuration du moteur de reconstruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionConfig {
    /// Taille des patchs pour la reconstruction
    pub patch_size: u32,
    /// Chevauchement des patchs
    pub overlap: u32,
    /// Qualité minimale requise
    pub quality_threshold: f32,
    /// Utiliser le GPU si disponible
    pub use_gpu: bool,
    /// Méthode de reconstruction par défaut
    pub default_method: ReconstructionMethod,
}

impl Default for ReconstructionConfig {
    fn default() -> Self {
        Self {
            patch_size: 256,
            overlap: 32,
            quality_threshold: 0.8,
            use_gpu: true,
            default_method: ReconstructionMethod::Hybrid,
        }
    }
}

/// Configuration de l'optimisation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Utiliser CUDA si disponible
    pub use_cuda: bool,
    /// Utiliser OpenCL si disponible
    pub use_opencl: bool,
    /// Utiliser SIMD si disponible
    pub use_simd: bool,
    /// Taille du cache en mémoire (en Mo)
    pub cache_size_mb: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            use_cuda: true,
            use_opencl: true,
            use_simd: true,
            cache_size_mb: 1024,
        }
    }
}

/// Configuration du système d'apprentissage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Intervalle minimum entre les mises à jour (en secondes)
    pub update_interval: u64,
    /// Nombre minimum d'échantillons avant mise à jour
    pub min_samples: usize,
    /// Sauvegarder les modèles après chaque mise à jour
    pub save_models: bool,
    /// Répertoire pour les sauvegardes de modèles
    pub models_dir: PathBuf,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            update_interval: 3600,
            min_samples: 100,
            save_models: true,
            models_dir: PathBuf::from("models"),
        }
    }
}

/// Types de modèles de détection disponibles
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DetectionModel {
    Logo,
    Text,
    Pattern,
    Transparency,
}

/// Méthodes de reconstruction disponibles
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReconstructionMethod {
    Inpainting,
    Diffusion,
    FrequencyDomain,
    Hybrid,
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
        let detection = DetectionConfig {
            confidence_threshold: std::env::var("DETECTION_CONFIDENCE_THRESHOLD")
                .unwrap_or_else(|_| "0.5".to_string())
                .parse()
                .unwrap_or(0.5),
            iou_threshold: std::env::var("DETECTION_IOU_THRESHOLD")
                .unwrap_or_else(|_| "0.45".to_string())
                .parse()
                .unwrap_or(0.45),
            max_image_size: (
                std::env::var("DETECTION_MAX_IMAGE_SIZE_WIDTH")
                    .unwrap_or_else(|_| "4096".to_string())
                    .parse()
                    .unwrap_or(4096),
                std::env::var("DETECTION_MAX_IMAGE_SIZE_HEIGHT")
                    .unwrap_or_else(|_| "4096".to_string())
                    .parse()
                    .unwrap_or(4096),
            ),
            use_gpu: std::env::var("DETECTION_USE_GPU")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            models: vec![
                DetectionModel::Logo,
                DetectionModel::Text,
                DetectionModel::Pattern,
            ],
        };

        let reconstruction = ReconstructionConfig {
            patch_size: std::env::var("RECONSTRUCTION_PATCH_SIZE")
                .unwrap_or_else(|_| "256".to_string())
                .parse()
                .unwrap_or(256),
            overlap: std::env::var("RECONSTRUCTION_OVERLAP")
                .unwrap_or_else(|_| "32".to_string())
                .parse()
                .unwrap_or(32),
            quality_threshold: std::env::var("RECONSTRUCTION_QUALITY_THRESHOLD")
                .unwrap_or_else(|_| "0.8".to_string())
                .parse()
                .unwrap_or(0.8),
            use_gpu: std::env::var("RECONSTRUCTION_USE_GPU")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            default_method: ReconstructionMethod::Hybrid,
        };

        let optimization = OptimizationConfig {
            use_cuda: std::env::var("OPTIMIZATION_USE_CUDA")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            use_opencl: std::env::var("OPTIMIZATION_USE_OPENCL")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            use_simd: std::env::var("OPTIMIZATION_USE_SIMD")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            cache_size_mb: std::env::var("OPTIMIZATION_CACHE_SIZE_MB")
                .unwrap_or_else(|_| "1024".to_string())
                .parse()
                .unwrap_or(1024),
        };

        let learning = LearningConfig {
            update_interval: std::env::var("LEARNING_UPDATE_INTERVAL")
                .unwrap_or_else(|_| "3600".to_string())
                .parse()
                .unwrap_or(3600),
            min_samples: std::env::var("LEARNING_MIN_SAMPLES")
                .unwrap_or_else(|_| "100".to_string())
                .parse()
                .unwrap_or(100),
            save_models: std::env::var("LEARNING_SAVE_MODELS")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            models_dir: PathBuf::from(
                std::env::var("LEARNING_MODELS_DIR")
                    .unwrap_or_else(|_| "models".to_string())
            ),
        };

        let learning_enabled = std::env::var("LEARNING_ENABLED")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);

        let max_threads = std::env::var("MAX_THREADS")
            .unwrap_or_else(|_| num_cpus::get().to_string())
            .parse()
            .unwrap_or(num_cpus::get());

        let max_batch_size = std::env::var("MAX_BATCH_SIZE")
            .unwrap_or_else(|_| "32".to_string())
            .parse()
            .unwrap_or(32);

        let temp_dir = PathBuf::from(
            std::env::var("TEMP_DIR")
                .unwrap_or_else(|_| std::env::temp_dir().join("watermark-evil").to_str().unwrap().to_string())
        );

        Ok(EngineConfig {
            detection,
            reconstruction,
            optimization,
            learning,
            learning_enabled,
            max_threads,
            max_batch_size,
            temp_dir,
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
        assert!(config.detection.use_gpu);
        assert_eq!(config.optimization.cache_size_mb, 1024);
        assert_eq!(config.learning.update_interval, 3600);
    }
}

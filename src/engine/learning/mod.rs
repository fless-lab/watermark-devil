use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::Result;

mod database;
mod trainer;
mod analyzer;
mod config;
mod collector;
mod metrics;

use database::PatternDatabase;
use trainer::ModelTrainer;
use analyzer::PatternAnalyzer;
use config::LearningConfig;
use collector::{DataCollector, TrainingExample, ExampleMetrics};
use metrics::{ModelMetrics, MetricsCalculator};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningResult {
    pub pattern_type: PatternType,
    pub success_rate: f32,
    pub improvements: Vec<Improvement>,
    pub training_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Logo,
    Text,
    Repetitive,
    Complex,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Improvement {
    pub category: String,
    pub metric: String,
    pub previous_value: f32,
    pub new_value: f32,
}

pub struct AdaptiveLearning {
    pattern_db: Arc<RwLock<PatternDatabase>>,
    trainer: Arc<ModelTrainer>,
    analyzer: Arc<PatternAnalyzer>,
    config: LearningConfig,
    collector: Arc<DataCollector>,
    metrics_calculator: Arc<MetricsCalculator>,
}

#[derive(Debug, Clone)]
struct LearningConfig {
    min_samples_for_training: usize,
    training_threshold: f32,
    max_patterns_per_type: usize,
    improvement_threshold: f32,
    collect_data: bool,
    store_images: bool,
    data_path: std::path::PathBuf,
    min_confidence: f32,
    max_examples: usize,
    batch_size: usize,
    learning_rate: f32,
    epochs: usize,
    early_stopping_patience: usize,
    validation_split: f32,
    augmentation_enabled: bool,
    model_type: String,
    hidden_layers: Vec<usize>,
    dropout_rate: f32,
    activation_function: String,
    optimizer_type: String,
    weight_decay: f32,
    momentum: f32,
    scheduler_type: String,
    min_examples_for_training: usize,
    training_interval_hours: u64,
    max_model_age_days: u64,
    performance_threshold: f32,
}

impl AdaptiveLearning {
    pub fn new() -> Self {
        let config = LearningConfig {
            min_samples_for_training: 100,
            training_threshold: 0.8,
            max_patterns_per_type: 1000,
            improvement_threshold: 0.05,
            collect_data: true,
            store_images: true,
            data_path: std::path::PathBuf::from("data"),
            min_confidence: 0.8,
            max_examples: 1000000,
            batch_size: 32,
            learning_rate: 0.001,
            epochs: 100,
            early_stopping_patience: 10,
            validation_split: 0.2,
            augmentation_enabled: true,
            model_type: "Hybrid".to_string(),
            hidden_layers: vec![512, 256, 128],
            dropout_rate: 0.3,
            activation_function: "GELU".to_string(),
            optimizer_type: "AdamW".to_string(),
            weight_decay: 0.01,
            momentum: 0.9,
            scheduler_type: "OneCycleLR".to_string(),
            min_examples_for_training: 1000,
            training_interval_hours: 24,
            max_model_age_days: 30,
            performance_threshold: 0.95,
        };

        Self {
            pattern_db: Arc::new(RwLock::new(PatternDatabase::new())),
            trainer: Arc::new(ModelTrainer::new()),
            analyzer: Arc::new(PatternAnalyzer::new()),
            config,
            collector: Arc::new(DataCollector::new(&config)),
            metrics_calculator: Arc::new(MetricsCalculator::new(Vec::new())),
        }
    }

    /// Analyse un nouveau cas et met à jour la base de connaissances
    pub async fn analyze_case(
        &self,
        image_data: &[u8],
        detection_result: &crate::detection::DetectionResult,
        reconstruction_result: &crate::reconstruction::ReconstructionResult,
    ) -> Result<LearningResult> {
        // Analyser le pattern
        let pattern_info = self.analyzer.analyze_pattern(
            image_data,
            detection_result,
            reconstruction_result,
        ).await?;

        // Mettre à jour la base de données
        let mut db = self.pattern_db.write().await;
        db.add_pattern(pattern_info.clone())?;

        // Vérifier si un réentraînement est nécessaire
        let training_needed = self.check_training_needed(&db).await?;

        // Calculer les améliorations
        let improvements = self.calculate_improvements(&pattern_info).await?;

        // Process detection
        self.process_detection(detection_result, image_data, reconstruction_result).await?;

        Ok(LearningResult {
            pattern_type: pattern_info.pattern_type,
            success_rate: reconstruction_result.quality_score,
            improvements,
            training_required: training_needed,
        })
    }

    /// Déclenche le réentraînement des modèles si nécessaire
    pub async fn trigger_training(&self) -> Result<()> {
        let db = self.pattern_db.read().await;
        
        // Vérifier si nous avons assez d'échantillons
        if db.get_total_samples() < self.config.min_samples_for_training {
            return Ok(());
        }

        // Préparer les données d'entraînement
        let training_data = db.prepare_training_data().await?;

        // Entraîner les modèles
        self.trainer.train_models(training_data).await?;

        Ok(())
    }

    /// Vérifie si un réentraînement est nécessaire
    async fn check_training_needed(&self, db: &PatternDatabase) -> Result<bool> {
        // Vérifier le nombre d'échantillons
        if db.get_total_samples() < self.config.min_samples_for_training {
            return Ok(false);
        }

        // Vérifier la performance récente
        let recent_performance = db.get_recent_performance(100)?;
        if recent_performance < self.config.training_threshold {
            return Ok(true);
        }

        // Vérifier les nouveaux types de patterns
        let new_patterns = db.get_new_pattern_types()?;
        if !new_patterns.is_empty() {
            return Ok(true);
        }

        Ok(false)
    }

    /// Calcule les améliorations basées sur les résultats récents
    async fn calculate_improvements(&self, pattern_info: &analyzer::PatternInfo) -> Result<Vec<Improvement>> {
        let mut improvements = Vec::new();
        let db = self.pattern_db.read().await;

        // Calculer l'amélioration de la détection
        if let Some(prev_detection) = db.get_average_detection_rate(pattern_info.pattern_type.clone())? {
            let current_detection = pattern_info.detection_confidence;
            if current_detection > prev_detection + self.config.improvement_threshold {
                improvements.push(Improvement {
                    category: "Detection".to_string(),
                    metric: "Confidence".to_string(),
                    previous_value: prev_detection,
                    new_value: current_detection,
                });
            }
        }

        // Calculer l'amélioration de la reconstruction
        if let Some(prev_quality) = db.get_average_reconstruction_quality(pattern_info.pattern_type.clone())? {
            let current_quality = pattern_info.reconstruction_quality;
            if current_quality > prev_quality + self.config.improvement_threshold {
                improvements.push(Improvement {
                    category: "Reconstruction".to_string(),
                    metric: "Quality".to_string(),
                    previous_value: prev_quality,
                    new_value: current_quality,
                });
            }
        }

        Ok(improvements)
    }

    /// Exporte les connaissances acquises
    pub async fn export_knowledge(&self, path: &str) -> Result<()> {
        let db = self.pattern_db.read().await;
        db.export_to_file(path).await
    }

    /// Importe des connaissances externes
    pub async fn import_knowledge(&self, path: &str) -> Result<()> {
        let mut db = self.pattern_db.write().await;
        db.import_from_file(path).await
    }

    /// Process detection
    pub async fn process_detection(
        &self,
        detection: &crate::detection::DetectionResult,
        original: &[u8],
        result: &crate::reconstruction::ReconstructionResult,
        processing_time: f32,
    ) -> Result<()> {
        // Calculate metrics
        let metrics = ExampleMetrics {
            psnr: self.metrics_calculator.calculate_psnr(original, result)?,
            ssim: self.metrics_calculator.calculate_ssim(original, result)?,
            fid: self.metrics_calculator.calculate_fid(original, result)?,
            processing_time_ms: (processing_time * 1000.0) as u64,
        };
        
        // Collect training example
        self.collector.collect_example(original, detection, result, metrics).await?;
        
        // Try to trigger training if needed
        self.trainer.train_if_needed().await?;
        
        Ok(())
    }

    /// Get model metrics
    pub async fn get_model_metrics(&self) -> Result<ModelMetrics> {
        let examples = self.collector.get_training_examples(1000).await?;
        
        let mut total_confidence = 0.0;
        let mut total_psnr = 0.0;
        let mut total_ssim = 0.0;
        let mut total_fid = 0.0;
        let mut fid_count = 0;
        let mut total_time = 0.0;
        
        for example in &examples {
            total_confidence += example.confidence;
            total_psnr += example.metrics.psnr;
            total_ssim += example.metrics.ssim;
            if let Some(fid) = example.metrics.fid {
                total_fid += fid;
                fid_count += 1;
            }
            total_time += example.metrics.processing_time_ms as f32;
        }
        
        let count = examples.len() as f32;
        if count == 0.0 {
            return Ok(ModelMetrics::default());
        }
        
        Ok(ModelMetrics {
            accuracy: 0.0, // TODO: implement accuracy calculation
            precision: 0.0, // TODO: implement precision calculation
            recall: 0.0, // TODO: implement recall calculation
            f1_score: 0.0, // TODO: implement f1 calculation
            avg_confidence: total_confidence / count,
            avg_processing_time: total_time / count,
            avg_psnr: total_psnr / count,
            avg_ssim: total_ssim / count,
            avg_fid: if fid_count > 0 { Some(total_fid / fid_count as f32) } else { None },
            num_samples: examples.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::{DetectionResult, BoundingBox, WatermarkType};
    use crate::reconstruction::ReconstructionResult;
    use crate::reconstruction::ReconstructionMethod;

    #[tokio::test]
    async fn test_adaptive_learning() {
        let learning = AdaptiveLearning::new();
        
        // Créer des résultats de test
        let detection = DetectionResult {
            confidence: 0.9,
            bbox: BoundingBox {
                x: 0,
                y: 0,
                width: 100,
                height: 100,
            },
            watermark_type: WatermarkType::Logo,
            mask: None,
        };

        let reconstruction = ReconstructionResult {
            success: true,
            quality_score: 0.85,
            processing_time: std::time::Duration::from_secs(1),
            method_used: ReconstructionMethod::Inpainting,
        };

        // Tester l'analyse d'un nouveau cas
        let result = learning.analyze_case(
            &vec![0u8; 100 * 100 * 3], // Image simulée
            &detection,
            &reconstruction,
        ).await;

        assert!(result.is_ok(), "Should successfully analyze new case");
        
        let result = result.unwrap();
        assert!(!result.improvements.is_empty(), "Should identify improvements");
    }
}

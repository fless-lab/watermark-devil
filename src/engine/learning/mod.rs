use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::Result;

mod database;
mod trainer;
mod analyzer;

use database::PatternDatabase;
use trainer::ModelTrainer;
use analyzer::PatternAnalyzer;

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
}

#[derive(Debug, Clone)]
struct LearningConfig {
    min_samples_for_training: usize,
    training_threshold: f32,
    max_patterns_per_type: usize,
    improvement_threshold: f32,
}

impl AdaptiveLearning {
    pub fn new() -> Self {
        let config = LearningConfig {
            min_samples_for_training: 100,
            training_threshold: 0.8,
            max_patterns_per_type: 1000,
            improvement_threshold: 0.05,
        };

        Self {
            pattern_db: Arc::new(RwLock::new(PatternDatabase::new())),
            trainer: Arc::new(ModelTrainer::new()),
            analyzer: Arc::new(PatternAnalyzer::new()),
            config,
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

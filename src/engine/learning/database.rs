use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use tokio::fs;
use super::PatternType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternInfo {
    pub pattern_type: PatternType,
    pub features: Vec<f32>,
    pub detection_confidence: f32,
    pub reconstruction_quality: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PatternDatabase {
    patterns: HashMap<String, Vec<PatternInfo>>,
    performance_history: Vec<PerformanceRecord>,
    statistics: DatabaseStats,
}

#[derive(Debug, Serialize, Deserialize)]
struct PerformanceRecord {
    timestamp: chrono::DateTime<chrono::Utc>,
    pattern_type: PatternType,
    detection_rate: f32,
    reconstruction_quality: f32,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct DatabaseStats {
    total_samples: usize,
    pattern_type_counts: HashMap<PatternType, usize>,
    average_quality: f32,
    last_training: Option<chrono::DateTime<chrono::Utc>>,
}

impl PatternDatabase {
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            performance_history: Vec::new(),
            statistics: DatabaseStats::default(),
        }
    }

    pub fn add_pattern(&mut self, pattern: PatternInfo) -> Result<()> {
        let key = self.generate_pattern_key(&pattern);
        
        // Mettre à jour les statistiques
        self.statistics.total_samples += 1;
        *self.statistics.pattern_type_counts
            .entry(pattern.pattern_type.clone())
            .or_insert(0) += 1;

        // Mettre à jour la performance moyenne
        self.update_average_quality(pattern.reconstruction_quality);

        // Ajouter l'enregistrement de performance
        self.performance_history.push(PerformanceRecord {
            timestamp: chrono::Utc::now(),
            pattern_type: pattern.pattern_type.clone(),
            detection_rate: pattern.detection_confidence,
            reconstruction_quality: pattern.reconstruction_quality,
        });

        // Ajouter le pattern à la base
        self.patterns
            .entry(key)
            .or_insert_with(Vec::new)
            .push(pattern);

        Ok(())
    }

    pub fn get_total_samples(&self) -> usize {
        self.statistics.total_samples
    }

    pub fn get_average_detection_rate(&self, pattern_type: PatternType) -> Result<Option<f32>> {
        let records: Vec<_> = self.performance_history
            .iter()
            .filter(|r| r.pattern_type == pattern_type)
            .collect();

        if records.is_empty() {
            return Ok(None);
        }

        let sum: f32 = records.iter().map(|r| r.detection_rate).sum();
        Ok(Some(sum / records.len() as f32))
    }

    pub fn get_average_reconstruction_quality(&self, pattern_type: PatternType) -> Result<Option<f32>> {
        let records: Vec<_> = self.performance_history
            .iter()
            .filter(|r| r.pattern_type == pattern_type)
            .collect();

        if records.is_empty() {
            return Ok(None);
        }

        let sum: f32 = records.iter().map(|r| r.reconstruction_quality).sum();
        Ok(Some(sum / records.len() as f32))
    }

    pub fn get_recent_performance(&self, n_samples: usize) -> Result<f32> {
        let recent: Vec<_> = self.performance_history
            .iter()
            .rev()
            .take(n_samples)
            .collect();

        if recent.is_empty() {
            return Ok(0.0);
        }

        let sum: f32 = recent.iter().map(|r| r.reconstruction_quality).sum();
        Ok(sum / recent.len() as f32)
    }

    pub fn get_new_pattern_types(&self) -> Result<Vec<PatternType>> {
        let threshold = chrono::Utc::now() - chrono::Duration::hours(24);
        
        let mut new_types = HashSet::new();
        for record in &self.performance_history {
            if record.timestamp > threshold {
                new_types.insert(record.pattern_type.clone());
            }
        }

        Ok(new_types.into_iter().collect())
    }

    pub async fn prepare_training_data(&self) -> Result<TrainingData> {
        // Convertir les patterns en format d'entraînement
        let mut features = Vec::new();
        let mut labels = Vec::new();

        for patterns in self.patterns.values() {
            for pattern in patterns {
                features.push(pattern.features.clone());
                labels.push(pattern.pattern_type.clone());
            }
        }

        Ok(TrainingData {
            features,
            labels,
            timestamp: chrono::Utc::now(),
        })
    }

    pub async fn export_to_file(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json).await?;
        Ok(())
    }

    pub async fn import_from_file(&mut self, path: &str) -> Result<()> {
        let json = fs::read_to_string(path).await?;
        let imported: PatternDatabase = serde_json::from_str(&json)?;
        
        // Fusionner les données importées
        self.merge_database(imported)?;
        Ok(())
    }

    fn generate_pattern_key(&self, pattern: &PatternInfo) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        pattern.features.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn update_average_quality(&mut self, new_quality: f32) {
        let old_total = self.statistics.average_quality * (self.statistics.total_samples - 1) as f32;
        self.statistics.average_quality = (old_total + new_quality) / self.statistics.total_samples as f32;
    }

    fn merge_database(&mut self, other: PatternDatabase) -> Result<()> {
        // Fusionner les patterns
        for (key, patterns) in other.patterns {
            self.patterns
                .entry(key)
                .or_insert_with(Vec::new)
                .extend(patterns);
        }

        // Mettre à jour les statistiques
        self.statistics.total_samples += other.statistics.total_samples;
        for (pattern_type, count) in other.statistics.pattern_type_counts {
            *self.statistics.pattern_type_counts
                .entry(pattern_type)
                .or_insert(0) += count;
        }

        // Recalculer la qualité moyenne
        self.statistics.average_quality = (self.statistics.average_quality + other.statistics.average_quality) / 2.0;

        // Fusionner l'historique des performances
        self.performance_history.extend(other.performance_history);
        self.performance_history.sort_by_key(|r| r.timestamp);

        Ok(())
    }
}

#[derive(Debug)]
pub struct TrainingData {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<PatternType>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[tokio::test]
    async fn test_pattern_database() {
        let mut db = PatternDatabase::new();
        
        // Ajouter quelques patterns de test
        let pattern = PatternInfo {
            pattern_type: PatternType::Logo,
            features: vec![1.0, 2.0, 3.0],
            detection_confidence: 0.9,
            reconstruction_quality: 0.85,
            timestamp: chrono::Utc::now(),
        };

        db.add_pattern(pattern.clone()).unwrap();
        
        assert_eq!(db.get_total_samples(), 1);
        
        let avg_quality = db.get_average_reconstruction_quality(PatternType::Logo)
            .unwrap()
            .unwrap();
        assert!((avg_quality - 0.85).abs() < 1e-6);
    }
}

use std::sync::Arc;
use tch::{nn, Device, Tensor};
use anyhow::Result;

use super::database::TrainingData;
use super::PatternType;

pub struct ModelTrainer {
    device: Device,
    config: TrainingConfig,
}

struct TrainingConfig {
    batch_size: i64,
    learning_rate: f64,
    epochs: i64,
    early_stopping_patience: i64,
}

impl ModelTrainer {
    pub fn new() -> Self {
        Self {
            device: Device::cuda_if_available(),
            config: TrainingConfig {
                batch_size: 32,
                learning_rate: 0.001,
                epochs: 100,
                early_stopping_patience: 10,
            },
        }
    }

    pub async fn train_models(&self, data: TrainingData) -> Result<()> {
        // Préparer les données d'entraînement
        let (train_data, val_data) = self.prepare_data(data)?;

        // Créer les modèles
        let detection_model = self.create_detection_model()?;
        let reconstruction_model = self.create_reconstruction_model()?;

        // Entraîner les modèles
        self.train_detection_model(&detection_model, &train_data, &val_data).await?;
        self.train_reconstruction_model(&reconstruction_model, &train_data, &val_data).await?;

        // Sauvegarder les modèles
        self.save_models(&detection_model, &reconstruction_model).await?;

        Ok(())
    }

    fn prepare_data(&self, data: TrainingData) -> Result<(DataSet, DataSet)> {
        // Convertir les features en tensors
        let features = Tensor::of_slice2(&data.features)?;
        
        // Convertir les labels en tensors one-hot
        let labels = self.convert_labels_to_onehot(&data.labels)?;
        
        // Diviser en ensembles d'entraînement et de validation
        let n_samples = features.size()[0];
        let n_train = (n_samples as f64 * 0.8) as i64;
        
        let train_features = features.narrow(0, 0, n_train);
        let train_labels = labels.narrow(0, 0, n_train);
        let val_features = features.narrow(0, n_train, n_samples - n_train);
        let val_labels = labels.narrow(0, n_train, n_samples - n_train);
        
        Ok((
            DataSet { features: train_features, labels: train_labels },
            DataSet { features: val_features, labels: val_labels },
        ))
    }

    fn convert_labels_to_onehot(&self, labels: &[PatternType]) -> Result<Tensor> {
        let n_classes = 5; // Nombre de types de patterns
        let n_samples = labels.len();
        
        let mut onehot = Tensor::zeros(&[n_samples as i64, n_classes], (tch::Kind::Float, self.device));
        
        for (i, label) in labels.iter().enumerate() {
            let idx = match label {
                PatternType::Logo => 0,
                PatternType::Text => 1,
                PatternType::Repetitive => 2,
                PatternType::Complex => 3,
                PatternType::Unknown => 4,
            };
            onehot.get(i as i64).get(idx).fill_(1.);
        }
        
        Ok(onehot)
    }

    fn create_detection_model(&self) -> Result<nn::Sequential> {
        let vs = nn::VarStore::new(self.device);
        let p = vs.root();
        
        let seq = nn::seq()
            .add(nn::linear(&p / "fc1", 512, 256, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::dropout(&p / "dropout1", 0.5))
            .add(nn::linear(&p / "fc2", 256, 128, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::dropout(&p / "dropout2", 0.5))
            .add(nn::linear(&p / "fc3", 128, 5, Default::default()))
            .add_fn(|x| x.log_softmax(-1, tch::Kind::Float));
            
        Ok(seq)
    }

    fn create_reconstruction_model(&self) -> Result<nn::Sequential> {
        let vs = nn::VarStore::new(self.device);
        let p = vs.root();
        
        let seq = nn::seq()
            .add(nn::conv2d(&p / "conv1", 3, 64, 3, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::conv2d(&p / "conv2", 64, 64, 3, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::conv2d(&p / "conv3", 64, 3, 3, Default::default()))
            .add_fn(|x| x.tanh());
            
        Ok(seq)
    }

    async fn train_detection_model(
        &self,
        model: &nn::Sequential,
        train_data: &DataSet,
        val_data: &DataSet,
    ) -> Result<()> {
        let mut opt = nn::Adam::default().build(model.parameters(), self.config.learning_rate)?;
        
        let n_batches = train_data.features.size()[0] / self.config.batch_size;
        let mut best_val_loss = f64::INFINITY;
        let mut patience_counter = 0;
        
        for epoch in 0..self.config.epochs {
            // Entraînement
            let mut train_loss = 0.0;
            for i in 0..n_batches {
                let batch_start = i * self.config.batch_size;
                let batch_features = train_data.features.narrow(0, batch_start, self.config.batch_size);
                let batch_labels = train_data.labels.narrow(0, batch_start, self.config.batch_size);
                
                let prediction = model.forward(&batch_features);
                let loss = prediction.nll_loss(&batch_labels);
                
                opt.backward_step(&loss);
                train_loss += f64::from(&loss);
            }
            train_loss /= n_batches as f64;
            
            // Validation
            let val_prediction = model.forward(&val_data.features);
            let val_loss = f64::from(&val_prediction.nll_loss(&val_data.labels));
            
            // Early stopping
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    println!("Early stopping at epoch {}", epoch);
                    break;
                }
            }
            
            println!("Epoch {}: train_loss = {:.4}, val_loss = {:.4}", epoch, train_loss, val_loss);
        }
        
        Ok(())
    }

    async fn train_reconstruction_model(
        &self,
        model: &nn::Sequential,
        train_data: &DataSet,
        val_data: &DataSet,
    ) -> Result<()> {
        let mut opt = nn::Adam::default().build(model.parameters(), self.config.learning_rate)?;
        
        let n_batches = train_data.features.size()[0] / self.config.batch_size;
        let mut best_val_loss = f64::INFINITY;
        let mut patience_counter = 0;
        
        for epoch in 0..self.config.epochs {
            // Entraînement
            let mut train_loss = 0.0;
            for i in 0..n_batches {
                let batch_start = i * self.config.batch_size;
                let batch_features = train_data.features.narrow(0, batch_start, self.config.batch_size);
                
                let prediction = model.forward(&batch_features);
                let loss = prediction.mse_loss(&batch_features, tch::Reduction::Mean);
                
                opt.backward_step(&loss);
                train_loss += f64::from(&loss);
            }
            train_loss /= n_batches as f64;
            
            // Validation
            let val_prediction = model.forward(&val_data.features);
            let val_loss = f64::from(&val_prediction.mse_loss(&val_data.features, tch::Reduction::Mean));
            
            // Early stopping
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    println!("Early stopping at epoch {}", epoch);
                    break;
                }
            }
            
            println!("Epoch {}: train_loss = {:.4}, val_loss = {:.4}", epoch, train_loss, val_loss);
        }
        
        Ok(())
    }

    async fn save_models(
        &self,
        detection_model: &nn::Sequential,
        reconstruction_model: &nn::Sequential,
    ) -> Result<()> {
        // Sauvegarder les modèles
        detection_model.save("models/detection_model.pt")?;
        reconstruction_model.save("models/reconstruction_model.pt")?;
        Ok(())
    }
}

struct DataSet {
    features: Tensor,
    labels: Tensor,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_model_training() {
        let trainer = ModelTrainer::new();
        
        // Créer des données de test
        let features = vec![
            vec![1.0, 2.0, 3.0]; 100
        ];
        let labels = vec![
            PatternType::Logo; 100
        ];
        
        let data = TrainingData {
            features,
            labels,
            timestamp: chrono::Utc::now(),
        };
        
        let result = trainer.train_models(data).await;
        assert!(result.is_ok(), "Model training should succeed");
    }
}

pub struct ContinuousTrainer {
    config: TrainingConfig,
    collector: Arc<DataCollector>,
    state: Arc<RwLock<TrainerState>>,
}

#[derive(Debug)]
struct TrainerState {
    last_training: Option<DateTime<Utc>>,
    current_metrics: ModelMetrics,
    training_in_progress: bool,
}

impl ContinuousTrainer {
    pub fn new(config: &TrainingConfig, collector: Arc<DataCollector>) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            collector,
            state: Arc::new(RwLock::new(TrainerState {
                last_training: None,
                current_metrics: ModelMetrics::default(),
                training_in_progress: false,
            })),
        })
    }
    
    pub async fn train_if_needed(&self) -> Result<bool> {
        let mut state = self.state.write().await;
        
        if state.training_in_progress {
            debug!("Training already in progress, skipping");
            return Ok(false);
        }
        
        // Check if we have enough new examples
        let examples = self.collector.get_training_examples(1000).await?;
        if examples.len() < 100 {
            debug!("Not enough training examples ({}), skipping", examples.len());
            return Ok(false);
        }
        
        // Check if enough time has passed since last training
        if let Some(last_training) = state.last_training {
            let elapsed = Utc::now() - last_training;
            if elapsed.num_hours() < 24 {
                debug!("Not enough time since last training, skipping");
                return Ok(false);
            }
        }
        
        // Start training
        state.training_in_progress = true;
        let config = self.config.clone();
        let collector = Arc::clone(&self.collector);
        let state = Arc::clone(&self.state);
        
        tokio::spawn(async move {
            if let Err(e) = Self::train_models(config, collector, state).await {
                warn!("Training failed: {}", e);
            }
        });
        
        Ok(true)
    }
    
    async fn train_models(
        config: TrainingConfig,
        collector: Arc<DataCollector>,
        state: Arc<RwLock<TrainerState>>,
    ) -> Result<()> {
        info!("Starting continuous training");
        
        // Get training examples
        let examples = collector.get_training_examples(config.max_examples).await?;
        
        // Split examples by type
        let mut logo_examples = Vec::new();
        let mut text_examples = Vec::new();
        let mut pattern_examples = Vec::new();
        let mut transparent_examples = Vec::new();
        
        for example in examples {
            if example.confidence < config.min_confidence {
                continue;
            }
            
            match example.watermark_type {
                WatermarkType::Logo => logo_examples.push(example),
                WatermarkType::Text => text_examples.push(example),
                WatermarkType::Pattern => pattern_examples.push(example),
                WatermarkType::Transparent => transparent_examples.push(example),
            }
        }
        
        // Train models in parallel
        let mut handles = Vec::new();
        
        if !logo_examples.is_empty() {
            handles.push(Self::train_logo_model(logo_examples));
        }
        if !text_examples.is_empty() {
            handles.push(Self::train_text_model(text_examples));
        }
        if !pattern_examples.is_empty() {
            handles.push(Self::train_pattern_model(pattern_examples));
        }
        if !transparent_examples.is_empty() {
            handles.push(Self::train_transparent_model(transparent_examples));
        }
        
        // Wait for all training to complete
        for handle in handles {
            handle.await??;
        }
        
        // Update state
        let mut state = state.write().await;
        state.last_training = Some(Utc::now());
        state.training_in_progress = false;
        
        info!("Continuous training completed successfully");
        Ok(())
    }
    
    async fn train_logo_model(examples: Vec<TrainingExample>) -> Result<()> {
        debug!("Training logo model with {} examples", examples.len());
        Python::with_gil(|py| {
            let model = PyModule::import(py, "ml.models.logo_detector.train")?
                .getattr("train_model")?;
                
            let examples_dict = examples_to_dict(py, &examples)?;
            model.call1((examples_dict,))?;
            
            Ok(())
        })
    }
    
    async fn train_text_model(examples: Vec<TrainingExample>) -> Result<()> {
        debug!("Training text model with {} examples", examples.len());
        Python::with_gil(|py| {
            let model = PyModule::import(py, "ml.models.text_detector.train")?
                .getattr("train_model")?;
                
            let examples_dict = examples_to_dict(py, &examples)?;
            model.call1((examples_dict,))?;
            
            Ok(())
        })
    }
    
    async fn train_pattern_model(examples: Vec<TrainingExample>) -> Result<()> {
        debug!("Training pattern model with {} examples", examples.len());
        Python::with_gil(|py| {
            let model = PyModule::import(py, "ml.models.pattern_detector.train")?
                .getattr("train_model")?;
                
            let examples_dict = examples_to_dict(py, &examples)?;
            model.call1((examples_dict,))?;
            
            Ok(())
        })
    }
    
    async fn train_transparent_model(examples: Vec<TrainingExample>) -> Result<()> {
        debug!("Training transparent model with {} examples", examples.len());
        Python::with_gil(|py| {
            let model = PyModule::import(py, "ml.models.transparency_detector.train")?
                .getattr("train_model")?;
                
            let examples_dict = examples_to_dict(py, &examples)?;
            model.call1((examples_dict,))?;
            
            Ok(())
        })
    }
}

fn examples_to_dict<'py>(
    py: Python<'py>,
    examples: &[TrainingExample],
) -> Result<&'py PyDict> {
    let dict = PyDict::new(py);
    
    let ids: Vec<_> = examples.iter().map(|e| e.id.clone()).collect();
    let confidences: Vec<_> = examples.iter().map(|e| e.confidence).collect();
    let successes: Vec<_> = examples.iter().map(|e| e.success).collect();
    let psnrs: Vec<_> = examples.iter().map(|e| e.metrics.psnr).collect();
    let ssims: Vec<_> = examples.iter().map(|e| e.metrics.ssim).collect();
    let fids: Vec<_> = examples.iter().map(|e| e.metrics.fid.unwrap_or(0.0)).collect();
    let processing_times: Vec<_> = examples.iter().map(|e| e.metrics.processing_time_ms).collect();
    
    dict.set_item("ids", ids)?;
    dict.set_item("confidences", confidences)?;
    dict.set_item("successes", successes)?;
    dict.set_item("psnrs", psnrs)?;
    dict.set_item("ssims", ssims)?;
    dict.set_item("fids", fids)?;
    dict.set_item("processing_times", processing_times)?;
    
    Ok(dict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_continuous_training() -> Result<()> {
        let temp_dir = tempdir()?;
        
        let config = TrainingConfig {
            batch_size: 32,
            learning_rate: 0.001,
            epochs: 100,
            early_stopping_patience: 10,
            min_confidence: 0.8,
            max_examples: 1000000,
        };
        
        let collector = Arc::new(DataCollector::new(&config)?);
        let trainer = ContinuousTrainer::new(&config, Arc::clone(&collector))?;
        
        // Add some test examples
        for _ in 0..150 {
            let example = TrainingExample {
                id: uuid::Uuid::new_v4().to_string(),
                timestamp: Utc::now(),
                watermark_type: WatermarkType::Logo,
                confidence: 0.95,
                success: true,
                metrics: ExampleMetrics {
                    psnr: 36.0,
                    ssim: 0.96,
                    fid: Some(15.0),
                    processing_time_ms: 100,
                },
                metadata: serde_json::json!({}),
            };
            
            let image = Mat::new_rows_cols_with_default(
                100,
                100,
                opencv::core::CV_8UC3,
                opencv::core::Scalar::all(255.0),
            )?;
            
            collector.collect_example(&image, &example, &image, example.metrics.clone()).await?;
        }
        
        // Try to trigger training
        let training_started = trainer.train_if_needed().await?;
        assert!(training_started, "Training should have started");
        
        // Wait a bit and check state
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        
        let state = trainer.state.read().await;
        assert!(state.last_training.is_some());
        
        Ok(())
    }
}

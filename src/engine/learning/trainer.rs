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

use std::path::Path;
use tch::{
    nn::{self, ModuleT, OptimizerConfig},
    Device, Tensor,
};
use opencv::{
    core::{Mat, MatTraitConst, Size},
    imgproc::{self, INTER_LINEAR},
};
use anyhow::Result;
use serde::{Serialize, Deserialize};
use tracing::{info, debug, warn};

use crate::types::{Detection, WatermarkType, Confidence};

const INPUT_SIZE: i64 = 224;  // ResNet standard input size
const NUM_CLASSES: i64 = 3;   // Background, Text Watermark, Logo Watermark

#[derive(Debug)]
pub struct NeuralDetector {
    model: nn::Sequential,
    device: Device,
    mean: Vec<f32>,
    std: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub backbone: String,
    pub weights_path: String,
    pub use_pretrained: bool,
    pub fine_tune_layers: usize,
}

impl NeuralDetector {
    pub fn new(config: &ModelConfig) -> Result<Self> {
        info!("Initializing neural detector with backbone: {}", config.backbone);
        
        let device = Device::cuda_if_available();
        debug!("Using device: {:?}", device);
        
        // Create model architecture
        let mut vs = nn::VarStore::new(device);
        let model = Self::create_model(&config.backbone, &mut vs.root())?;
        
        // Load weights if available
        if Path::new(&config.weights_path).exists() {
            info!("Loading weights from: {}", config.weights_path);
            vs.load(&config.weights_path)?;
        } else if config.use_pretrained {
            warn!("Weights not found, using pretrained initialization");
            Self::load_pretrained_weights(&mut vs, &config.backbone)?;
        }
        
        Ok(Self {
            model,
            device,
            mean: vec![0.485, 0.456, 0.406],  // ImageNet normalization
            std: vec![0.229, 0.224, 0.225],
        })
    }

    pub fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        debug!("Running neural detection");
        
        // Preprocess image
        let tensor = self.preprocess_image(image)?;
        
        // Run inference
        let output = self.model.forward_t(&tensor, true);
        let (confidence, class) = output.softmax(-1, output.kind()).max_dim(-1, false);
        
        // Convert to CPU and get values
        let confidence = Vec::<f32>::from(confidence.to_device(Device::Cpu));
        let class = Vec::<i64>::from(class.to_device(Device::Cpu));
        
        // Create detections
        let mut detections = Vec::new();
        for (idx, &class_idx) in class.iter().enumerate() {
            if class_idx > 0 {  // Skip background class
                let watermark_type = match class_idx {
                    1 => WatermarkType::Text,
                    2 => WatermarkType::Logo,
                    _ => continue,
                };
                
                detections.push(Detection {
                    watermark_type,
                    confidence: Confidence::new(confidence[idx]),
                    bbox: opencv::core::Rect::new(0, 0, image.cols(), image.rows()),
                    metadata: Some(serde_json::json!({
                        "class_id": class_idx,
                        "raw_confidence": confidence[idx]
                    })),
                });
            }
        }
        
        Ok(detections)
    }

    fn preprocess_image(&self, image: &Mat) -> Result<Tensor> {
        // Resize
        let mut resized = Mat::default();
        imgproc::resize(
            image,
            &mut resized,
            Size::new(INPUT_SIZE as i32, INPUT_SIZE as i32),
            0.0,
            0.0,
            INTER_LINEAR,
        )?;
        
        // Convert to float and normalize
        let mut float_mat = Mat::default();
        resized.convert_to(&mut float_mat, opencv::core::CV_32F, 1.0/255.0, 0.0)?;
        
        // Convert to tensor
        let data: Vec<f32> = float_mat.data_typed()?;
        let tensor = Tensor::of_slice(&data)
            .view([1, INPUT_SIZE, INPUT_SIZE, 3])
            .permute(&[0, 3, 1, 2]);
        
        // Normalize
        let mut normalized = tensor.shallow_clone();
        for c in 0..3 {
            normalized.slice(1, c, c+1, 1).sub_(&self.mean[c]);
            normalized.slice(1, c, c+1, 1).div_(&self.std[c]);
        }
        
        Ok(normalized.to_device(self.device))
    }

    fn create_model(backbone: &str, vs: &nn::Path) -> Result<nn::Sequential> {
        let backbone = match backbone.to_lowercase().as_str() {
            "resnet18" => nn::resnet18(vs, NUM_CLASSES),
            "resnet34" => nn::resnet34(vs, NUM_CLASSES),
            "resnet50" => nn::resnet50(vs, NUM_CLASSES),
            _ => anyhow::bail!("Unsupported backbone: {}", backbone),
        };
        
        Ok(backbone)
    }

    fn load_pretrained_weights(vs: &mut nn::VarStore, backbone: &str) -> Result<()> {
        let url = match backbone.to_lowercase().as_str() {
            "resnet18" => "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            "resnet34" => "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
            "resnet50" => "https://download.pytorch.org/models/resnet50-19c8e357.pth",
            _ => anyhow::bail!("No pretrained weights for backbone: {}", backbone),
        };
        
        vs.load_partial(url)?;
        Ok(())
    }

    pub fn train(&mut self, train_data: &[(Mat, i64)], epochs: i32, learning_rate: f64) -> Result<()> {
        info!("Starting training for {} epochs", epochs);
        
        let mut opt = nn::Adam::default().build(&self.model.vs(), learning_rate)?;
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct = 0;
            let mut total = 0;
            
            for (image, label) in train_data {
                // Preprocess
                let x = self.preprocess_image(image)?;
                let y = Tensor::of_slice(&[*label]).to_device(self.device);
                
                // Forward pass
                let output = self.model.forward_t(&x, true);
                let loss = output.cross_entropy_loss::<Tensor>(&y, None, Reduction::Mean, -100, 0.0);
                
                // Backward pass
                opt.backward_step(&loss);
                
                // Statistics
                total_loss += f64::from(loss);
                let pred = output.argmax(-1, false);
                correct += i64::from(pred.eq_tensor(&y)).sum();
                total += y.size()[0];
            }
            
            let accuracy = (correct as f64) / (total as f64);
            info!(
                "Epoch {}/{}: Loss = {:.4}, Accuracy = {:.2}%",
                epoch + 1, epochs, total_loss / total as f64, accuracy * 100.0
            );
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_neural_detector() -> Result<()> {
        // Create test image
        let mut image = Mat::new_rows_cols_with_default(
            512,
            512,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::all(255.0)
        )?;
        
        // Add text-like pattern
        imgproc::put_text(
            &mut image,
            "WATERMARK",
            opencv::core::Point::new(100, 250),
            imgproc::FONT_HERSHEY_SIMPLEX,
            2.0,
            opencv::core::Scalar::new(0.0, 0.0, 0.0, 0.0),
            3,
            imgproc::LINE_8,
            false,
        )?;
        
        // Create detector
        let config = ModelConfig {
            backbone: "resnet18".to_string(),
            weights_path: "".to_string(),
            use_pretrained: true,
            fine_tune_layers: 2,
        };
        let detector = NeuralDetector::new(&config)?;
        
        // Run detection
        let detections = detector.detect(&image)?;
        
        // Basic validation
        assert!(!detections.is_empty(), "Should detect watermark");
        
        Ok(())
    }
    
    #[test]
    fn test_model_creation() -> Result<()> {
        let vs = nn::VarStore::new(Device::Cpu);
        
        // Test different backbones
        for backbone in &["resnet18", "resnet34", "resnet50"] {
            let model = NeuralDetector::create_model(backbone, &vs.root())?;
            assert!(model.forward_t(&Tensor::zeros(&[1, 3, INPUT_SIZE, INPUT_SIZE], (Kind::Float, Device::Cpu)), false).size() == &[1, NUM_CLASSES]);
        }
        
        Ok(())
    }
}

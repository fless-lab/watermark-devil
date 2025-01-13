use std::sync::Arc;
use image::{DynamicImage, ImageBuffer, Rgb};
use tch::{Device, Tensor, nn, vision};
use anyhow::Result;

use crate::detection::{DetectionResult, BoundingBox, WatermarkType};

pub struct NeuralDetector {
    model: Arc<WatermarkNet>,
    device: Device,
    input_size: (i64, i64),
    confidence_threshold: f64,
}

struct WatermarkNet {
    backbone: vision::resnet::ResNet,
    detection_head: nn::Sequential,
    segmentation_head: nn::Sequential,
}

impl NeuralDetector {
    pub fn new() -> Self {
        let device = Device::cuda_if_available();
        let model = Arc::new(WatermarkNet::new(device));
        
        Self {
            model,
            device,
            input_size: (512, 512),
            confidence_threshold: 0.7,
        }
    }

    pub async fn detect(&self, image: &DynamicImage) -> Vec<DetectionResult> {
        let tensor = self.preprocess_image(image);
        if let Ok(tensor) = tensor {
            if let Ok(detections) = self.model.forward(&tensor) {
                return self.postprocess_detections(detections, image.dimensions());
            }
        }
        Vec::new()
    }

    fn preprocess_image(&self, image: &DynamicImage) -> Result<Tensor> {
        // Redimensionner l'image
        let resized = image.resize_exact(
            self.input_size.0 as u32,
            self.input_size.1 as u32,
            image::imageops::FilterType::Lanczos3,
        );

        // Convertir en tensor PyTorch
        let tensor = vision::image::load_from_memory(&resized.to_bytes())?;
        
        // Normaliser
        let normalized = tensor.to_device(self.device)
            .to_kind(tch::Kind::Float)
            .div(255.);
        
        // Ajouter la dimension batch
        Ok(normalized.unsqueeze(0))
    }

    fn postprocess_detections(&self, detections: Tensor, original_dims: (u32, u32)) -> Vec<DetectionResult> {
        let mut results = Vec::new();
        
        let (original_width, original_height) = original_dims;
        let scale_x = original_width as f32 / self.input_size.0 as f32;
        let scale_y = original_height as f32 / self.input_size.1 as f32;

        // Extraire les prédictions
        let boxes = Vec::<Vec<f32>>::from(detections.slice(1, 0, 4, 1));
        let scores = Vec::<f32>::from(detections.slice(1, 4, 5, 1));
        let class_ids = Vec::<i64>::from(detections.slice(1, 5, 6, 1));

        for i in 0..boxes.len() {
            if scores[i] < self.confidence_threshold as f32 {
                continue;
            }

            let bbox = &boxes[i];
            results.push(DetectionResult {
                confidence: scores[i],
                bbox: BoundingBox {
                    x: (bbox[0] * scale_x) as u32,
                    y: (bbox[1] * scale_y) as u32,
                    width: ((bbox[2] - bbox[0]) * scale_x) as u32,
                    height: ((bbox[3] - bbox[1]) * scale_y) as u32,
                },
                watermark_type: self.class_id_to_type(class_ids[i]),
                mask: None,
            });
        }

        results
    }

    fn class_id_to_type(&self, class_id: i64) -> WatermarkType {
        match class_id {
            0 => WatermarkType::Logo,
            1 => WatermarkType::Text,
            2 => WatermarkType::Pattern,
            3 => WatermarkType::Transparent,
            _ => WatermarkType::Complex,
        }
    }
}

impl WatermarkNet {
    fn new(device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        
        // Utiliser ResNet50 comme backbone
        let backbone = vision::resnet::resnet50(&vs.root(), vision::resnet::ResNet50Config::default());
        
        // Tête de détection
        let detection_head = nn::seq()
            .add(nn::conv2d(
                &vs.root(),
                2048,
                512,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.relu())
            .add(nn::conv2d(
                &vs.root(),
                512,
                6, // 4 pour bbox + 1 pour score + 1 pour classe
                1,
                nn::ConvConfig::default(),
            ));

        // Tête de segmentation
        let segmentation_head = nn::seq()
            .add(nn::conv2d(
                &vs.root(),
                2048,
                256,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.relu())
            .add(nn::conv2d(
                &vs.root(),
                256,
                1,
                1,
                nn::ConvConfig::default(),
            ))
            .add_fn(|x| x.sigmoid());

        Self {
            backbone,
            detection_head,
            segmentation_head,
        }
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.backbone.forward(input);
        let detections = self.detection_head.forward(&features);
        Ok(detections.view([-1, 6])) // Reshape pour [batch_size * anchors, 6]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{RgbImage, Rgb};

    #[tokio::test]
    async fn test_neural_detector() {
        let detector = NeuralDetector::new();
        
        // Créer une image de test
        let mut test_image = RgbImage::new(512, 512);
        // Dessiner un "watermark" simple
        for y in 100..200 {
            for x in 100..300 {
                test_image.put_pixel(x, y, Rgb([200, 200, 200]));
            }
        }
        
        let dynamic_image = DynamicImage::ImageRgb8(test_image);
        let results = detector.detect(&dynamic_image).await;
        
        // Note: Ce test pourrait échouer sans modèle pré-entraîné
        // Il devrait être modifié pour utiliser un mock ou un modèle de test
        assert!(!results.is_empty(), "Should detect watermark in test image");
    }
}

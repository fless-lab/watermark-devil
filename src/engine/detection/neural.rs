use std::sync::Arc;
use opencv::{
    core::{Mat, Point, Scalar, Size, CV_32F},
    dnn,
    imgproc,
    prelude::*,
};
use anyhow::{Result, Context};
use tracing::{debug, instrument};

use crate::{
    types::{Detection, WatermarkType, BoundingBox, Confidence},
    config::DetectionConfig,
    error::EngineError,
};

use super::models::WatermarkDetector;

const INPUT_WIDTH: i32 = 640;
const INPUT_HEIGHT: i32 = 640;
const SCORE_THRESHOLD: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.45;
const CONFIDENCE_THRESHOLD: f32 = 0.45;

/// Détecteur basé sur des modèles de deep learning
#[derive(Clone)]
pub struct NeuralDetector {
    config: Arc<DetectionConfig>,
    net: Arc<dnn::Net>,
    output_layers: Vec<String>,
}

impl NeuralDetector {
    pub fn new(config: &DetectionConfig) -> Result<Self> {
        // Charger le modèle YOLOv5
        let mut net = dnn::read_net_from_onnx("models/watermark_yolov5.onnx")?;
        
        if config.use_gpu {
            net.set_preferable_backend(dnn::DNN_BACKEND_CUDA)?;
            net.set_preferable_target(dnn::DNN_TARGET_CUDA)?;
        }
        
        // Obtenir les noms des couches de sortie
        let output_layers = net.get_unconnected_out_layers_names()?;
        
        Ok(Self {
            config: Arc::new(config.clone()),
            net: Arc::new(net),
            output_layers,
        })
    }
    
    fn preprocess_image(&self, image: &Mat) -> Result<Mat> {
        let mut blob = Mat::default();
        
        // Redimensionner et normaliser l'image
        dnn::blob_from_image(
            image,
            &mut blob,
            1.0/255.0,
            Size::new(INPUT_WIDTH, INPUT_HEIGHT),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            true,
            false,
            CV_32F
        )?;
        
        Ok(blob)
    }
    
    fn post_process(&self, outputs: &[Mat], image_size: Size) -> Result<Vec<Detection>> {
        let mut detections = Vec::new();
        let scale_x = image_size.width as f32 / INPUT_WIDTH as f32;
        let scale_y = image_size.height as f32 / INPUT_HEIGHT as f32;
        
        for output in outputs {
            let total_predictions = output.rows();
            
            for i in 0..total_predictions {
                let confidence = *output.at_2d::<f32>(i, 4)?;
                
                if confidence >= CONFIDENCE_THRESHOLD {
                    let mut class_scores = Vec::new();
                    for j in 5..output.cols() {
                        class_scores.push(*output.at_2d::<f32>(i, j)?);
                    }
                    
                    let (max_score, class_id) = class_scores.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();
                        
                    let score = confidence * max_score;
                    
                    if score > SCORE_THRESHOLD {
                        let center_x = *output.at_2d::<f32>(i, 0)? * scale_x;
                        let center_y = *output.at_2d::<f32>(i, 1)? * scale_y;
                        let width = *output.at_2d::<f32>(i, 2)? * scale_x;
                        let height = *output.at_2d::<f32>(i, 3)? * scale_y;
                        
                        let left = center_x - width/2.0;
                        let top = center_y - height/2.0;
                        
                        detections.push(Detection {
                            watermark_type: match class_id {
                                0 => WatermarkType::Logo,
                                1 => WatermarkType::Text,
                                2 => WatermarkType::Pattern,
                                _ => WatermarkType::Unknown,
                            },
                            bbox: BoundingBox {
                                x: left as u32,
                                y: top as u32,
                                width: width as u32,
                                height: height as u32,
                            },
                            confidence: Confidence::new(score),
                            mask: None,
                        });
                    }
                }
            }
        }
        
        // Appliquer NMS
        let mut indices = Mat::default();
        let mut boxes = Mat::default();
        let mut scores = Mat::default();
        
        let mut box_data = Vec::new();
        let mut score_data = Vec::new();
        
        for detection in &detections {
            box_data.extend_from_slice(&[
                detection.bbox.x as f32,
                detection.bbox.y as f32,
                (detection.bbox.x + detection.bbox.width) as f32,
                (detection.bbox.y + detection.bbox.height) as f32,
            ]);
            score_data.push(detection.confidence.value);
        }
        
        let boxes_mat = Mat::from_slice(&box_data)?;
        let scores_mat = Mat::from_slice(&score_data)?;
        
        dnn::nms_boxes(
            &boxes_mat,
            &scores_mat,
            SCORE_THRESHOLD,
            NMS_THRESHOLD,
            &mut indices,
            1.0,
            0
        )?;
        
        // Filtrer les détections après NMS
        let mut filtered_detections = Vec::new();
        for i in 0..indices.rows() {
            let idx = *indices.at_2d::<i32>(i, 0)? as usize;
            filtered_detections.push(detections[idx].clone());
        }
        
        Ok(filtered_detections)
    }
}

impl WatermarkDetector for NeuralDetector {
    #[instrument(skip(self, image))]
    fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        debug!("Starting neural-based detection");
        
        // Prétraitement
        let blob = self.preprocess_image(image)
            .context("Failed to preprocess image")?;
            
        // Forward pass
        self.net.set_input(&blob, "", 1.0, Scalar::default())?;
        
        let mut outputs = Vec::new();
        for name in &self.output_layers {
            outputs.push(self.net.forward(name)?);
        }
        
        debug!("Neural network forward pass completed");
        
        // Post-traitement
        let detections = self.post_process(&outputs, image.size()?)
            .context("Failed to post-process detections")?;
            
        debug!("Found {} detections after post-processing", detections.len());
        
        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::imgcodecs;
    
    #[test]
    fn test_neural_detector_creation() -> Result<()> {
        let config = DetectionConfig::default();
        let detector = NeuralDetector::new(&config)?;
        Ok(())
    }
    
    #[test]
    fn test_preprocessing() -> Result<()> {
        let config = DetectionConfig::default();
        let detector = NeuralDetector::new(&config)?;
        
        let image = imgcodecs::imread(
            "tests/resources/images/logo_sample.png",
            imgcodecs::IMREAD_COLOR
        )?;
        
        let blob = detector.preprocess_image(&image)?;
        assert!(!blob.empty());
        assert_eq!(blob.channels(), 3);
        Ok(())
    }
    
    #[test]
    fn test_logo_detection() -> Result<()> {
        let config = DetectionConfig::default();
        let detector = NeuralDetector::new(&config)?;
        
        let image = imgcodecs::imread(
            "tests/resources/images/logo_sample.png",
            imgcodecs::IMREAD_COLOR
        )?;
        
        let detections = detector.detect(&image)?;
        
        // Vérifier la détection de logos
        assert!(!detections.is_empty());
        assert!(detections.iter().any(|d| d.watermark_type == WatermarkType::Logo));
        
        // Vérifier les propriétés des détections
        for detection in detections {
            assert!(detection.confidence.value >= CONFIDENCE_THRESHOLD);
            assert!(detection.bbox.width > 0);
            assert!(detection.bbox.height > 0);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_text_detection() -> Result<()> {
        let config = DetectionConfig::default();
        let detector = NeuralDetector::new(&config)?;
        
        let image = imgcodecs::imread(
            "tests/resources/images/text_sample.png",
            imgcodecs::IMREAD_COLOR
        )?;
        
        let detections = detector.detect(&image)?;
        
        // Vérifier la détection de texte
        assert!(!detections.is_empty());
        assert!(detections.iter().any(|d| d.watermark_type == WatermarkType::Text));
        
        Ok(())
    }
}

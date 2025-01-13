use std::sync::Arc;
use image::{DynamicImage, ImageBuffer, Rgb};
use opencv as cv;
use rayon::prelude::*;

mod patterns;
mod frequency;
mod neural;

#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub confidence: f32,
    pub bbox: BoundingBox,
    pub watermark_type: WatermarkType,
    pub mask: Option<ImageBuffer<Rgb<u8>, Vec<u8>>>,
}

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WatermarkType {
    Logo,
    Text,
    Pattern,
    Transparent,
    Complex,
}

pub struct DetectionEngine {
    pattern_detector: Arc<patterns::PatternDetector>,
    frequency_analyzer: Arc<frequency::FrequencyAnalyzer>,
    neural_detector: Arc<neural::NeuralDetector>,
}

impl DetectionEngine {
    pub fn new() -> Self {
        Self {
            pattern_detector: Arc::new(patterns::PatternDetector::new()),
            frequency_analyzer: Arc::new(frequency::FrequencyAnalyzer::new()),
            neural_detector: Arc::new(neural::NeuralDetector::new()),
        }
    }

    /// Détecte les watermarks dans une image
    pub async fn detect(&self, image: &DynamicImage) -> Vec<DetectionResult> {
        let (width, height) = image.dimensions();
        
        // Parallélisation des différentes méthodes de détection
        let mut results = vec![];
        
        // 1. Détection par pattern
        let pattern_future = tokio::spawn({
            let pattern_detector = Arc::clone(&self.pattern_detector);
            let image = image.clone();
            async move {
                pattern_detector.detect(&image).await
            }
        });

        // 2. Analyse fréquentielle
        let freq_future = tokio::spawn({
            let frequency_analyzer = Arc::clone(&self.frequency_analyzer);
            let image = image.clone();
            async move {
                frequency_analyzer.analyze(&image).await
            }
        });

        // 3. Détection neurale
        let neural_future = tokio::spawn({
            let neural_detector = Arc::clone(&self.neural_detector);
            let image = image.clone();
            async move {
                neural_detector.detect(&image).await
            }
        });

        // Collecte des résultats
        let pattern_results = pattern_future.await.unwrap_or_default();
        let freq_results = freq_future.await.unwrap_or_default();
        let neural_results = neural_future.await.unwrap_or_default();

        // Fusion des résultats avec non-maximum suppression
        results.extend(pattern_results);
        results.extend(freq_results);
        results.extend(neural_results);

        self.apply_nms(&mut results);
        results
    }

    /// Applique la non-maximum suppression pour éviter les détections redondantes
    fn apply_nms(&self, detections: &mut Vec<DetectionResult>) {
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        let mut indices_to_remove = vec![];
        for i in 0..detections.len() {
            if indices_to_remove.contains(&i) {
                continue;
            }
            
            for j in (i + 1)..detections.len() {
                if indices_to_remove.contains(&j) {
                    continue;
                }
                
                if self.calculate_iou(&detections[i].bbox, &detections[j].bbox) > 0.5 {
                    indices_to_remove.push(j);
                }
            }
        }
        
        indices_to_remove.sort_unstable();
        for index in indices_to_remove.iter().rev() {
            detections.remove(*index);
        }
    }

    /// Calcule l'IoU (Intersection over Union) entre deux bounding boxes
    fn calculate_iou(&self, bbox1: &BoundingBox, bbox2: &BoundingBox) -> f32 {
        let x1 = bbox1.x.max(bbox2.x);
        let y1 = bbox1.y.max(bbox2.y);
        let x2 = (bbox1.x + bbox1.width).min(bbox2.x + bbox2.width);
        let y2 = (bbox1.y + bbox1.height).min(bbox2.y + bbox2.height);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let area1 = bbox1.width * bbox1.height;
        let area2 = bbox2.width * bbox2.height;
        let union = area1 + area2 - intersection;

        intersection as f32 / union as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[tokio::test]
    async fn test_detection_engine() {
        let engine = DetectionEngine::new();
        let test_image = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        
        let results = engine.detect(&test_image).await;
        assert!(results.is_empty(), "Should return empty results for blank image");
    }

    #[test]
    fn test_iou_calculation() {
        let engine = DetectionEngine::new();
        
        let bbox1 = BoundingBox {
            x: 0,
            y: 0,
            width: 10,
            height: 10,
        };
        
        let bbox2 = BoundingBox {
            x: 5,
            y: 5,
            width: 10,
            height: 10,
        };
        
        let iou = engine.calculate_iou(&bbox1, &bbox2);
        assert!(iou > 0.0 && iou < 1.0, "IoU should be between 0 and 1");
    }
}

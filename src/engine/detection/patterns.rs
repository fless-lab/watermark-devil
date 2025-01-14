use std::sync::Arc;
use opencv::{
    core::{Mat, Point, Scalar, Size, CV_32F, CV_8U},
    imgproc,
    features2d,
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

/// Détecteur basé sur la reconnaissance de motifs répétitifs
#[derive(Clone)]
pub struct PatternDetector {
    config: Arc<DetectionConfig>,
    orb: Arc<features2d::ORB>,
}

impl PatternDetector {
    pub fn new(config: &DetectionConfig) -> Result<Self> {
        let orb = features2d::ORB::create(
            500, // Nombre maximum de features
            1.2, // Scale factor
            8,   // Niveaux de pyramide
            31,  // Taille du patch
            0,   // First level
            2,   // WTA_K
            features2d::ORB_ScoreType::HARRIS_SCORE as i32,
            31,  // Taille du patch
            20,  // Fast threshold
        )?;
        
        Ok(Self {
            config: Arc::new(config.clone()),
            orb: Arc::new(orb),
        })
    }
    
    fn extract_features(&self, image: &Mat) -> Result<(Mat, Mat)> {
        let mut keypoints = Mat::default();
        let mut descriptors = Mat::default();
        
        self.orb.detect_and_compute(
            image,
            &Mat::default(),
            &mut keypoints,
            &mut descriptors,
            false
        )?;
        
        Ok((keypoints, descriptors))
    }
    
    fn find_repeated_patterns(&self, keypoints: &Mat, descriptors: &Mat) -> Result<Vec<Detection>> {
        if descriptors.empty() {
            return Ok(vec![]);
        }
        
        let mut detections = Vec::new();
        let matcher = features2d::BFMatcher::create(features2d::NORM_HAMMING, false)?;
        
        // Trouver les correspondances entre les descripteurs
        let mut matches = Mat::default();
        matcher.match_(&descriptors, &descriptors, &mut matches, &Mat::default())?;
        
        // Analyser les distances entre les points correspondants
        let mut patterns = Vec::new();
        let rows = matches.rows();
        
        for i in 0..rows {
            let m = matches.at_row::<features2d::DMatch>(i)?;
            if m.query_idx != m.train_idx {
                let kp1 = keypoints.at_row::<features2d::KeyPoint>(m.query_idx as i32)?;
                let kp2 = keypoints.at_row::<features2d::KeyPoint>(m.train_idx as i32)?;
                
                let dx = kp2.pt.x - kp1.pt.x;
                let dy = kp2.pt.y - kp1.pt.y;
                let distance = (dx * dx + dy * dy).sqrt();
                
                if distance > 10.0 {
                    patterns.push((kp1.pt, kp2.pt, m.distance as f32));
                }
            }
        }
        
        // Regrouper les motifs similaires
        let mut used = vec![false; patterns.len()];
        for i in 0..patterns.len() {
            if used[i] {
                continue;
            }
            
            let (p1, p2, dist) = patterns[i];
            let mut similar_count = 1;
            let mut total_confidence = 1.0 - dist / 100.0;
            
            for j in i + 1..patterns.len() {
                if used[j] {
                    continue;
                }
                
                let (p3, p4, dist2) = patterns[j];
                let dx1 = p2.x - p1.x;
                let dy1 = p2.y - p1.y;
                let dx2 = p4.x - p3.x;
                let dy2 = p4.y - p3.y;
                
                // Vérifier si les vecteurs sont similaires
                let dot_product = dx1 * dx2 + dy1 * dy2;
                let norm1 = (dx1 * dx1 + dy1 * dy1).sqrt();
                let norm2 = (dx2 * dx2 + dy2 * dy2).sqrt();
                
                if norm1 > 0.0 && norm2 > 0.0 {
                    let angle = (dot_product / (norm1 * norm2)).acos();
                    
                    if angle.abs() < 0.2 { // ~11 degrés
                        similar_count += 1;
                        total_confidence += 1.0 - dist2 / 100.0;
                        used[j] = true;
                    }
                }
            }
            
            if similar_count >= 3 {
                let confidence = total_confidence / similar_count as f32;
                if confidence > self.config.confidence_threshold {
                    // Créer une détection pour ce groupe de motifs
                    detections.push(Detection {
                        watermark_type: WatermarkType::Pattern,
                        bbox: BoundingBox {
                            x: p1.x as u32,
                            y: p1.y as u32,
                            width: (p2.x - p1.x).abs() as u32,
                            height: (p2.y - p1.y).abs() as u32,
                        },
                        confidence: Confidence::new(confidence),
                        mask: None,
                    });
                }
            }
        }
        
        Ok(detections)
    }
}

impl WatermarkDetector for PatternDetector {
    #[instrument(skip(self, image))]
    fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        debug!("Starting pattern-based detection");
        
        // Extraction des caractéristiques
        let (keypoints, descriptors) = self.extract_features(image)
            .context("Failed to extract features")?;
            
        debug!("Extracted {} keypoints", keypoints.rows());
        
        // Recherche des motifs répétés
        let detections = self.find_repeated_patterns(&keypoints, &descriptors)
            .context("Failed to find repeated patterns")?;
            
        debug!("Found {} pattern detections", detections.len());
        
        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::imgcodecs;
    
    #[test]
    fn test_pattern_detector_creation() -> Result<()> {
        let config = DetectionConfig::default();
        let detector = PatternDetector::new(&config)?;
        Ok(())
    }
    
    #[test]
    fn test_feature_extraction() -> Result<()> {
        let config = DetectionConfig::default();
        let detector = PatternDetector::new(&config)?;
        
        let image = imgcodecs::imread(
            "tests/resources/images/pattern_sample.png",
            imgcodecs::IMREAD_COLOR
        )?;
        
        let (keypoints, descriptors) = detector.extract_features(&image)?;
        assert!(keypoints.rows() > 0);
        assert!(!descriptors.empty());
        Ok(())
    }
    
    #[test]
    fn test_pattern_detection() -> Result<()> {
        let config = DetectionConfig::default();
        let detector = PatternDetector::new(&config)?;
        
        let image = imgcodecs::imread(
            "tests/resources/images/pattern_sample.png",
            imgcodecs::IMREAD_COLOR
        )?;
        
        let detections = detector.detect(&image)?;
        
        // Vérifier que nous avons trouvé des motifs
        assert!(!detections.is_empty());
        
        // Vérifier les propriétés des détections
        for detection in detections {
            assert_eq!(detection.watermark_type, WatermarkType::Pattern);
            assert!(detection.confidence.value >= config.confidence_threshold);
            assert!(detection.bbox.width > 0);
            assert!(detection.bbox.height > 0);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_empty_image() -> Result<()> {
        let config = DetectionConfig::default();
        let detector = PatternDetector::new(&config)?;
        
        let empty_image = Mat::new_rows_cols(100, 100, CV_8U)?;
        let detections = detector.detect(&empty_image)?;
        
        assert!(detections.is_empty());
        Ok(())
    }
}

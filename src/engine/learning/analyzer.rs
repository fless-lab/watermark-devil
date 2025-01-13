use image::{DynamicImage, GenericImageView};
use opencv as cv;
use opencv::core::{Mat, Point, Scalar, Size};
use opencv::imgproc;
use anyhow::Result;

use super::{PatternType, database::PatternInfo};
use crate::detection::DetectionResult;
use crate::reconstruction::ReconstructionResult;

pub struct PatternAnalyzer {
    feature_extractor: FeatureExtractor,
}

struct FeatureExtractor {
    orb: cv::features2d::ORB,
}

impl PatternAnalyzer {
    pub fn new() -> Self {
        Self {
            feature_extractor: FeatureExtractor::new(),
        }
    }

    pub async fn analyze_pattern(
        &self,
        image_data: &[u8],
        detection: &DetectionResult,
        reconstruction: &ReconstructionResult,
    ) -> Result<PatternInfo> {
        // Charger l'image
        let image = image::load_from_memory(image_data)?;
        
        // Extraire la région du watermark
        let watermark_region = self.extract_watermark_region(&image, detection)?;
        
        // Extraire les caractéristiques
        let features = self.feature_extractor.extract_features(&watermark_region)?;
        
        // Déterminer le type de pattern
        let pattern_type = self.determine_pattern_type(
            &features,
            detection,
            reconstruction,
        )?;

        Ok(PatternInfo {
            pattern_type,
            features,
            detection_confidence: detection.confidence,
            reconstruction_quality: reconstruction.quality_score,
            timestamp: chrono::Utc::now(),
        })
    }

    fn extract_watermark_region(
        &self,
        image: &DynamicImage,
        detection: &DetectionResult,
    ) -> Result<DynamicImage> {
        let bbox = &detection.bbox;
        Ok(image.crop(
            bbox.x,
            bbox.y,
            bbox.width,
            bbox.height,
        ))
    }

    fn determine_pattern_type(
        &self,
        features: &[f32],
        detection: &DetectionResult,
        reconstruction: &ReconstructionResult,
    ) -> Result<PatternType> {
        // Analyser les caractéristiques pour déterminer le type
        let periodic = self.check_periodicity(features)?;
        let text_like = self.check_text_characteristics(features)?;
        let complex = self.check_complexity(features)?;

        // Utiliser une approche pondérée pour la décision finale
        match (periodic, text_like, complex) {
            (true, _, _) => Ok(PatternType::Repetitive),
            (_, true, _) => Ok(PatternType::Text),
            (_, _, true) => Ok(PatternType::Complex),
            _ if self.check_logo_characteristics(features)? => Ok(PatternType::Logo),
            _ => Ok(PatternType::Unknown),
        }
    }

    fn check_periodicity(&self, features: &[f32]) -> Result<bool> {
        // Analyser la périodicité dans le domaine fréquentiel
        let threshold = 0.7;
        let periodicity_score = self.calculate_periodicity_score(features);
        Ok(periodicity_score > threshold)
    }

    fn check_text_characteristics(&self, features: &[f32]) -> Result<bool> {
        // Vérifier les caractéristiques typiques du texte
        let text_score = self.calculate_text_score(features);
        Ok(text_score > 0.6)
    }

    fn check_complexity(&self, features: &[f32]) -> Result<bool> {
        // Évaluer la complexité du pattern
        let complexity_score = self.calculate_complexity_score(features);
        Ok(complexity_score > 0.8)
    }

    fn check_logo_characteristics(&self, features: &[f32]) -> Result<bool> {
        // Vérifier les caractéristiques typiques des logos
        let logo_score = self.calculate_logo_score(features);
        Ok(logo_score > 0.7)
    }

    fn calculate_periodicity_score(&self, features: &[f32]) -> f32 {
        // Utiliser la transformée de Fourier pour détecter la périodicité
        let mut fft_features = features.to_vec();
        let n = fft_features.len();
        
        // Calculer le spectre de puissance
        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let mut spectrum = vec![rustfft::num_complex::Complex::new(0.0, 0.0); n];
        
        for (i, &val) in features.iter().enumerate() {
            spectrum[i] = rustfft::num_complex::Complex::new(val, 0.0);
        }
        
        fft.process(&mut spectrum);
        
        // Analyser les pics dans le spectre
        let mut peaks = 0;
        let threshold = 0.1;
        
        for i in 1..n-1 {
            if spectrum[i].norm() > spectrum[i-1].norm() && 
               spectrum[i].norm() > spectrum[i+1].norm() &&
               spectrum[i].norm() > threshold {
                peaks += 1;
            }
        }
        
        peaks as f32 / (n as f32 / 10.0)
    }

    fn calculate_text_score(&self, features: &[f32]) -> f32 {
        // Caractéristiques typiques du texte :
        // - Distribution régulière des bords
        // - Alignement horizontal
        // - Espacement régulier
        
        let edge_distribution = self.analyze_edge_distribution(features);
        let horizontal_alignment = self.analyze_horizontal_alignment(features);
        let spacing_regularity = self.analyze_spacing_regularity(features);
        
        0.4 * edge_distribution + 0.3 * horizontal_alignment + 0.3 * spacing_regularity
    }

    fn calculate_complexity_score(&self, features: &[f32]) -> f32 {
        // Évaluer la complexité basée sur :
        // - Nombre de caractéristiques distinctes
        // - Variance des caractéristiques
        // - Distribution spatiale
        
        let feature_diversity = self.calculate_feature_diversity(features);
        let feature_variance = self.calculate_feature_variance(features);
        let spatial_distribution = self.calculate_spatial_distribution(features);
        
        0.4 * feature_diversity + 0.3 * feature_variance + 0.3 * spatial_distribution
    }

    fn calculate_logo_score(&self, features: &[f32]) -> f32 {
        // Caractéristiques typiques des logos :
        // - Forme compacte
        // - Bords nets
        // - Cohérence interne
        
        let compactness = self.calculate_compactness(features);
        let edge_sharpness = self.calculate_edge_sharpness(features);
        let internal_coherence = self.calculate_internal_coherence(features);
        
        0.4 * compactness + 0.3 * edge_sharpness + 0.3 * internal_coherence
    }

    // Méthodes d'analyse auxiliaires
    fn analyze_edge_distribution(&self, features: &[f32]) -> f32 {
        // Implémenter l'analyse de la distribution des bords
        0.5 // Valeur par défaut pour l'exemple
    }

    fn analyze_horizontal_alignment(&self, features: &[f32]) -> f32 {
        // Implémenter l'analyse de l'alignement horizontal
        0.5 // Valeur par défaut pour l'exemple
    }

    fn analyze_spacing_regularity(&self, features: &[f32]) -> f32 {
        // Implémenter l'analyse de la régularité des espacements
        0.5 // Valeur par défaut pour l'exemple
    }

    fn calculate_feature_diversity(&self, features: &[f32]) -> f32 {
        // Calculer la diversité des caractéristiques
        0.5 // Valeur par défaut pour l'exemple
    }

    fn calculate_feature_variance(&self, features: &[f32]) -> f32 {
        // Calculer la variance des caractéristiques
        if features.is_empty() {
            return 0.0;
        }

        let mean = features.iter().sum::<f32>() / features.len() as f32;
        let variance = features.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / features.len() as f32;
        
        variance.sqrt()
    }

    fn calculate_spatial_distribution(&self, features: &[f32]) -> f32 {
        // Calculer la distribution spatiale des caractéristiques
        0.5 // Valeur par défaut pour l'exemple
    }

    fn calculate_compactness(&self, features: &[f32]) -> f32 {
        // Calculer la compacité de la forme
        0.5 // Valeur par défaut pour l'exemple
    }

    fn calculate_edge_sharpness(&self, features: &[f32]) -> f32 {
        // Calculer la netteté des bords
        0.5 // Valeur par défaut pour l'exemple
    }

    fn calculate_internal_coherence(&self, features: &[f32]) -> f32 {
        // Calculer la cohérence interne
        0.5 // Valeur par défaut pour l'exemple
    }
}

impl FeatureExtractor {
    fn new() -> Self {
        Self {
            orb: cv::features2d::ORB::create(
                500, // nfeatures
                1.2, // scaleFactor
                8,   // nlevels
                31,  // edgeThreshold
                0,   // firstLevel
                2,   // WTA_K
                cv::features2d::ORB_ScoreType::HARRIS_SCORE,
                31,  // patchSize
                20,  // fastThreshold
            ).unwrap(),
        }
    }

    fn extract_features(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        // Convertir l'image en Mat OpenCV
        let img = self.dynamic_image_to_mat(image)?;
        
        // Détecter les points clés et calculer les descripteurs
        let mut keypoints = cv::core::Vector::new();
        let mut descriptors = Mat::default();
        self.orb.detect_and_compute(
            &img,
            &Mat::default(),
            &mut keypoints,
            &mut descriptors,
            false,
        )?;

        // Convertir les descripteurs en vecteur de caractéristiques
        let mut features = Vec::new();
        if !descriptors.empty()? {
            for row in 0..descriptors.rows() {
                for col in 0..descriptors.cols() {
                    features.push(descriptors.at_2d::<u8>(row, col)? as f32);
                }
            }
        }

        Ok(features)
    }

    fn dynamic_image_to_mat(&self, image: &DynamicImage) -> Result<Mat> {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        
        unsafe {
            Ok(Mat::new_rows_cols_with_data(
                height as i32,
                width as i32,
                cv::core::CV_8UC3,
                rgb.as_ptr() as *mut _,
                cv::core::Mat_AUTO_STEP,
            )?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{RgbImage, Rgb};
    use crate::detection::{BoundingBox, WatermarkType};
    use crate::reconstruction::ReconstructionMethod;

    #[tokio::test]
    async fn test_pattern_analysis() {
        let analyzer = PatternAnalyzer::new();
        
        // Créer une image de test
        let mut test_image = RgbImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                test_image.put_pixel(x, y, Rgb([200, 200, 200]));
            }
        }
        
        // Ajouter un pattern de test
        for y in 40..60 {
            for x in 40..60 {
                test_image.put_pixel(x, y, Rgb([100, 100, 100]));
            }
        }

        let detection = DetectionResult {
            confidence: 0.9,
            bbox: BoundingBox {
                x: 40,
                y: 40,
                width: 20,
                height: 20,
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

        let result = analyzer.analyze_pattern(
            &test_image.to_vec(),
            &detection,
            &reconstruction,
        ).await;
        
        assert!(result.is_ok(), "Pattern analysis should succeed");
    }
}

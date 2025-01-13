use std::sync::Arc;
use image::{DynamicImage, ImageBuffer, Rgb};
use rayon::prelude::*;

mod inpainting;
mod diffusion;
mod frequency;

use crate::detection::{DetectionResult, WatermarkType};

pub struct ReconstructionEngine {
    inpainting_engine: Arc<inpainting::InpaintingEngine>,
    diffusion_engine: Arc<diffusion::DiffusionEngine>,
    frequency_engine: Arc<frequency::FrequencyReconstructor>,
    hybrid_engine: Arc<HybridReconstructor>,
}

#[derive(Debug)]
pub struct ReconstructionResult {
    pub success: bool,
    pub quality_score: f32,
    pub processing_time: std::time::Duration,
    pub method_used: ReconstructionMethod,
}

#[derive(Debug)]
pub enum ReconstructionMethod {
    Inpainting,
    Diffusion,
    FrequencyDomain,
    Hybrid,
}

impl ReconstructionEngine {
    pub fn new(config: ReconstructionConfig) -> Self {
        Self {
            inpainting_engine: Arc::new(inpainting::InpaintingEngine::new(&config)),
            diffusion_engine: Arc::new(diffusion::DiffusionEngine::new(&config)),
            frequency_engine: Arc::new(frequency::FrequencyReconstructor::new(&config).unwrap()),
            hybrid_engine: Arc::new(HybridReconstructor::new(&config)),
        }
    }

    /// Reconstruit l'image en supprimant les watermarks détectés
    pub async fn reconstruct(
        &self,
        image: &DynamicImage,
        detections: &[DetectionResult],
    ) -> anyhow::Result<(DynamicImage, Vec<ReconstructionResult>)> {
        let mut results = Vec::new();
        let mut current_image = image.clone();

        // Trier les détections par confiance décroissante
        let mut sorted_detections = detections.to_vec();
        sorted_detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        // Traiter chaque détection
        for detection in sorted_detections {
            let start_time = std::time::Instant::now();
            
            let (reconstructed_image, method) = match detection.watermark_type {
                WatermarkType::Logo | WatermarkType::Text => {
                    // Utiliser l'inpainting pour les logos et textes
                    let result = self.inpainting_engine
                        .remove_watermark(&current_image, &detection)
                        .await?;
                    (result, ReconstructionMethod::Inpainting)
                },
                WatermarkType::Pattern => {
                    // Utiliser l'analyse fréquentielle pour les motifs répétitifs
                    let result = self.frequency_engine
                        .remove_pattern(&current_image, &detection)
                        .await?;
                    (result, ReconstructionMethod::FrequencyDomain)
                },
                WatermarkType::Transparent | WatermarkType::Complex => {
                    // Utiliser la diffusion pour les cas complexes
                    let result = self.diffusion_engine
                        .reconstruct(&current_image, &detection)
                        .await?;
                    (result, ReconstructionMethod::Diffusion)
                }
                _ => {
                    // Utiliser l'approche hybride pour les autres cas
                    let result = self.hybrid_engine
                        .reconstruct(&opencv::imgcodecs::imdecode(
                            &opencv::core::Mat::zeros(0, 0, 0),
                            opencv::imgcodecs::IMREAD_COLOR,
                            &image.to_bytes(),
                        )?, &detection)?;
                    (DynamicImage::ImageRgb8(result.image), ReconstructionMethod::Hybrid)
                }
            };

            // Évaluer la qualité de la reconstruction
            let quality_score = self.evaluate_quality(
                &current_image,
                &reconstructed_image,
                &detection,
            );

            results.push(ReconstructionResult {
                success: quality_score > 0.8,
                quality_score,
                processing_time: start_time.elapsed(),
                method_used: method,
            });

            current_image = reconstructed_image;
        }

        Ok((current_image, results))
    }

    /// Évalue la qualité de la reconstruction
    fn evaluate_quality(
        &self,
        original: &DynamicImage,
        reconstructed: &DynamicImage,
        detection: &DetectionResult,
    ) -> f32 {
        // Calculer SSIM dans la région reconstruite
        let bbox = &detection.bbox;
        let original_region = original.crop(
            bbox.x,
            bbox.y,
            bbox.width,
            bbox.height,
        );
        let reconstructed_region = reconstructed.crop(
            bbox.x,
            bbox.y,
            bbox.width,
            bbox.height,
        );

        // Convertir en niveaux de gris pour la comparaison
        let original_gray = original_region.to_luma8();
        let reconstructed_gray = reconstructed_region.to_luma8();

        // Calculer SSIM
        self.calculate_ssim(&original_gray, &reconstructed_gray)
    }

    /// Calcule l'indice SSIM (Structural Similarity Index)
    fn calculate_ssim(
        &self,
        img1: &image::GrayImage,
        img2: &image::GrayImage,
    ) -> f32 {
        const C1: f32 = 0.01 * 255.0 * 0.01 * 255.0;
        const C2: f32 = 0.03 * 255.0 * 0.03 * 255.0;

        let mut sum_ssim = 0.0;
        let mut count = 0;

        // Calculer les moyennes
        let mean1: f32 = img1.pixels().map(|p| p[0] as f32).sum::<f32>() / (img1.len() as f32);
        let mean2: f32 = img2.pixels().map(|p| p[0] as f32).sum::<f32>() / (img2.len() as f32);

        // Calculer les variances et covariance
        let mut variance1 = 0.0;
        let mut variance2 = 0.0;
        let mut covariance = 0.0;

        for (p1, p2) in img1.pixels().zip(img2.pixels()) {
            let x = p1[0] as f32 - mean1;
            let y = p2[0] as f32 - mean2;
            variance1 += x * x;
            variance2 += y * y;
            covariance += x * y;
        }

        variance1 /= img1.len() as f32;
        variance2 /= img2.len() as f32;
        covariance /= img1.len() as f32;

        // Calculer SSIM
        let numerator = (2.0 * mean1 * mean2 + C1) * (2.0 * covariance + C2);
        let denominator = (mean1 * mean1 + mean2 * mean2 + C1) * 
                         (variance1 + variance2 + C2);

        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{RgbImage, Rgb};

    #[tokio::test]
    async fn test_reconstruction() {
        let engine = ReconstructionEngine::new(ReconstructionConfig {
            method: ReconstructionMethod::Hybrid,
            quality: "high".to_string(),
            use_gpu: false,
            preserve_details: true,
            max_iterations: 1000,
        });
        
        // Créer une image de test
        let mut test_image = RgbImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                test_image.put_pixel(x, y, Rgb([200, 200, 200]));
            }
        }
        
        // Ajouter un "watermark" simulé
        for y in 40..60 {
            for x in 40..60 {
                test_image.put_pixel(x, y, Rgb([100, 100, 100]));
            }
        }

        let detection = DetectionResult {
            confidence: 0.9,
            bbox: crate::detection::BoundingBox {
                x: 40,
                y: 40,
                width: 20,
                height: 20,
            },
            watermark_type: WatermarkType::Logo,
            mask: None,
        };

        let image = DynamicImage::ImageRgb8(test_image);
        let (reconstructed, results) = engine
            .reconstruct(&image, &[detection])
            .await
            .unwrap();

        assert!(!results.is_empty(), "Should have reconstruction results");
        assert!(
            results[0].quality_score > 0.5,
            "Reconstruction quality should be reasonable"
        );
    }
}

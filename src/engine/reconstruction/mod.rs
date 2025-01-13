use std::sync::Arc;
use std::time::Instant;
use anyhow::{Result, Context};
use opencv::{
    core::{Mat, UMat, USAGE_DEFAULT},
    cudaimgproc,
    prelude::*,
};
use metrics::{counter, gauge, histogram};
use crate::config::ReconstructionConfig;
use crate::types::{Detection, ReconstructionResult, Quality};
use crate::utils::validation;

/// Métriques de performance pour la reconstruction
#[derive(Debug, Clone)]
pub struct ReconstructionMetrics {
    pub total_time_ms: f64,
    pub preprocessing_time_ms: f64,
    pub inpainting_time_ms: f64,
    pub postprocessing_time_ms: f64,
    pub memory_usage_mb: f64,
}

/// Moteur de reconstruction d'image
pub struct ReconstructionEngine {
    config: ReconstructionConfig,
    cuda_enabled: bool,
    cuda_stream: Option<opencv::core::Stream>,
}

impl ReconstructionEngine {
    pub fn new(config: ReconstructionConfig) -> Result<Self> {
        // Vérifier si CUDA est disponible
        let cuda_enabled = cudaimgproc::get_cuda_enabled_device_count()? > 0;
        let cuda_stream = if cuda_enabled {
            Some(opencv::core::Stream::new()?)
        } else {
            None
        };

        Ok(Self {
            config,
            cuda_enabled,
            cuda_stream,
        })
    }

    /// Valide les entrées de reconstruction
    fn validate_input(&self, image: &Mat, detections: &[Detection]) -> Result<()> {
        // Valider l'image
        validation::check_not_empty(image)
            .context("Image d'entrée vide")?;
            
        validation::check_dimensions(
            image,
            self.config.min_image_size,
            self.config.max_image_size
        ).context("Dimensions d'image invalides")?;
        
        validation::check_image_type(image)
            .context("Type d'image non supporté")?;
            
        // Valider les détections
        if detections.is_empty() {
            anyhow::bail!("Aucune détection fournie");
        }
        
        for detection in detections {
            validation::check_bbox(&detection.bbox, image.size()?)
                .context("Boîte englobante invalide")?;
        }
        
        Ok(())
    }

    /// Valide les résultats de reconstruction
    fn validate_result(&self, result: &Mat, original: &Mat) -> Result<Quality> {
        // Vérifier les dimensions
        if result.size()? != original.size()? {
            anyhow::bail!("Dimensions du résultat incorrectes");
        }
        
        // Vérifier le type
        if result.type_()? != original.type_()? {
            anyhow::bail!("Type du résultat incorrect");
        }
        
        // Calculer la qualité (PSNR, SSIM, etc.)
        let quality = self.compute_quality(result, original)?;
        
        // Vérifier si la qualité est acceptable
        if quality.psnr < self.config.min_psnr {
            anyhow::bail!("Qualité de reconstruction insuffisante");
        }
        
        Ok(quality)
    }

    /// Calcule les métriques de qualité
    fn compute_quality(&self, result: &Mat, original: &Mat) -> Result<Quality> {
        let psnr = opencv::core::PSNR(result, original, 255.0)?;
        // TODO: Ajouter d'autres métriques (SSIM, etc.)
        
        Ok(Quality { psnr })
    }

    /// Reconstruit l'image avec CUDA si disponible
    pub fn reconstruct(&self, image: &Mat, detections: &[Detection]) -> Result<ReconstructionResult> {
        let start = Instant::now();
        let mut metrics = ReconstructionMetrics {
            total_time_ms: 0.0,
            preprocessing_time_ms: 0.0,
            inpainting_time_ms: 0.0,
            postprocessing_time_ms: 0.0,
            memory_usage_mb: 0.0,
        };

        // Validation
        let preprocess_start = Instant::now();
        self.validate_input(image, detections)?;
        metrics.preprocessing_time_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;

        // Reconstruction
        let inpaint_start = Instant::now();
        let result = if self.cuda_enabled {
            self.reconstruct_cuda(image, detections)?
        } else {
            self.reconstruct_cpu(image, detections)?
        };
        metrics.inpainting_time_ms = inpaint_start.elapsed().as_secs_f64() * 1000.0;

        // Validation et post-traitement
        let postprocess_start = Instant::now();
        let quality = self.validate_result(&result, image)?;
        metrics.postprocessing_time_ms = postprocess_start.elapsed().as_secs_f64() * 1000.0;

        // Métriques finales
        metrics.total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        metrics.memory_usage_mb = self.get_memory_usage()?;

        // Enregistrer les métriques
        self.record_metrics(&metrics);

        Ok(ReconstructionResult {
            image: result,
            quality,
            metrics: Some(metrics),
        })
    }

    /// Reconstruction sur GPU avec CUDA
    fn reconstruct_cuda(&self, image: &Mat, detections: &[Detection]) -> Result<Mat> {
        let stream = self.cuda_stream.as_ref()
            .context("CUDA stream non initialisé")?;

        // Transférer l'image sur GPU
        let mut gpu_image = UMat::new(USAGE_DEFAULT);
        image.copy_to_umat(&mut gpu_image)?;

        // Créer le masque sur GPU
        let mut mask = self.create_mask_cuda(detections, image.size()?)?;

        // Inpainting sur GPU
        let mut result = UMat::new(USAGE_DEFAULT);
        cudaimgproc::inpaint(&gpu_image, &mask, &mut result, 
            self.config.inpaint_radius, 
            cudaimgproc::INPAINT_TELEA,
            stream)?;

        // Transférer le résultat sur CPU
        let mut cpu_result = Mat::default();
        result.copy_to(&mut cpu_result)?;

        Ok(cpu_result)
    }

    /// Reconstruction sur CPU (fallback)
    fn reconstruct_cpu(&self, image: &Mat, detections: &[Detection]) -> Result<Mat> {
        let mask = self.create_mask_cpu(detections, image.size()?)?;
        let mut result = Mat::default();
        
        opencv::photo::inpaint(
            image,
            &mask,
            &mut result,
            self.config.inpaint_radius,
            opencv::photo::INPAINT_TELEA
        )?;

        Ok(result)
    }

    /// Crée un masque sur GPU
    fn create_mask_cuda(&self, detections: &[Detection], size: opencv::core::Size) -> Result<UMat> {
        let mut mask = UMat::new(USAGE_DEFAULT);
        // TODO: Implémenter la création de masque optimisée sur GPU
        Ok(mask)
    }

    /// Crée un masque sur CPU
    fn create_mask_cpu(&self, detections: &[Detection], size: opencv::core::Size) -> Result<Mat> {
        let mut mask = Mat::zeros(size.height, size.width, opencv::core::CV_8UC1)?;
        
        for detection in detections {
            if let Some(det_mask) = &detection.mask {
                // Utiliser le masque de détection si disponible
                det_mask.copy_to(&mut mask)?;
            } else {
                // Sinon utiliser la boîte englobante
                let rect = opencv::core::Rect::new(
                    detection.bbox.x,
                    detection.bbox.y,
                    detection.bbox.width,
                    detection.bbox.height
                );
                let roi = mask.roi(rect)?;
                roi.set_to(&opencv::core::Scalar::all(255.0), &Mat::default())?;
            }
        }
        
        Ok(mask)
    }

    /// Obtient l'utilisation mémoire actuelle
    fn get_memory_usage(&self) -> Result<f64> {
        if self.cuda_enabled {
            // TODO: Implémenter la mesure de mémoire GPU
            Ok(0.0)
        } else {
            // TODO: Implémenter la mesure de mémoire CPU
            Ok(0.0)
        }
    }

    /// Enregistre les métriques de performance
    fn record_metrics(&self, metrics: &ReconstructionMetrics) {
        gauge!("reconstruction.total_time_ms", metrics.total_time_ms);
        gauge!("reconstruction.preprocessing_time_ms", metrics.preprocessing_time_ms);
        gauge!("reconstruction.inpainting_time_ms", metrics.inpainting_time_ms);
        gauge!("reconstruction.postprocessing_time_ms", metrics.postprocessing_time_ms);
        gauge!("reconstruction.memory_usage_mb", metrics.memory_usage_mb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Point, Scalar};

    fn create_test_image() -> Result<Mat> {
        let mut image = Mat::new_size_with_default(
            opencv::core::Size::new(640, 480),
            opencv::core::CV_8UC3,
            Scalar::all(255.0)
        )?;
        
        // Dessiner un rectangle noir
        opencv::imgproc::rectangle(
            &mut image,
            Point::new(100, 100),
            Point::new(200, 200),
            Scalar::all(0.0),
            -1,
            opencv::imgproc::LINE_8,
            0
        )?;
        
        Ok(image)
    }

    fn create_test_detection() -> Detection {
        Detection {
            watermark_type: crate::types::WatermarkType::Logo,
            bbox: crate::types::BoundingBox {
                x: 100,
                y: 100,
                width: 100,
                height: 100,
            },
            confidence: crate::types::Confidence::new(0.9),
            mask: None,
        }
    }

    #[test]
    fn test_validation() -> Result<()> {
        let config = ReconstructionConfig::default();
        let engine = ReconstructionEngine::new(config)?;
        
        // Image vide
        let empty_mat = Mat::default();
        let detection = create_test_detection();
        assert!(engine.validate_input(&empty_mat, &[detection]).is_err());
        
        // Image valide
        let valid_mat = create_test_image()?;
        assert!(engine.validate_input(&valid_mat, &[detection]).is_ok());
        
        Ok(())
    }

    #[test]
    fn test_reconstruction() -> Result<()> {
        let config = ReconstructionConfig::default();
        let engine = ReconstructionEngine::new(config)?;
        
        let image = create_test_image()?;
        let detection = create_test_detection();
        
        let result = engine.reconstruct(&image, &[detection])?;
        
        // Vérifier la qualité
        assert!(result.quality.psnr > 0.0);
        
        // Vérifier les métriques
        let metrics = result.metrics.unwrap();
        assert!(metrics.total_time_ms > 0.0);
        assert!(metrics.preprocessing_time_ms > 0.0);
        assert!(metrics.inpainting_time_ms > 0.0);
        assert!(metrics.postprocessing_time_ms > 0.0);
        
        Ok(())
    }

    #[test]
    fn test_cuda_fallback() -> Result<()> {
        let config = ReconstructionConfig::default();
        let engine = ReconstructionEngine::new(config)?;
        
        let image = create_test_image()?;
        let detection = create_test_detection();
        
        // La reconstruction devrait fonctionner même si CUDA n'est pas disponible
        let result = engine.reconstruct(&image, &[detection])?;
        assert!(result.quality.psnr > 0.0);
        
        Ok(())
    }
}

use std::sync::Arc;
use std::time::Instant;
use anyhow::{Result, Context};
use opencv::{
    core::{Mat, UMat, Size, USAGE_DEFAULT},
    cudaimgproc, cudawarping,
    prelude::*,
};
use metrics::{counter, gauge, histogram};
use rayon::prelude::*;

use crate::config::OptimizationConfig;
use crate::types::{OptimizationResult, Quality};
use crate::utils::validation;

/// Tailles d'images prédéfinies pour l'optimisation
const IMAGE_SIZES: &[Size] = &[
    Size::new(256, 256),
    Size::new(512, 512),
    Size::new(1024, 1024),
    Size::new(2048, 2048),
];

/// Métriques de performance pour l'optimisation
#[derive(Debug, Clone)]
pub struct OptimizationMetrics {
    pub total_time_ms: f64,
    pub preprocessing_time_ms: f64,
    pub optimization_time_ms: f64,
    pub postprocessing_time_ms: f64,
    pub gpu_memory_used_mb: f64,
    pub cpu_memory_used_mb: f64,
}

/// Moteur d'optimisation avec support CUDA
pub struct OptimizationEngine {
    config: OptimizationConfig,
    cuda_enabled: bool,
    cuda_stream: Option<opencv::core::Stream>,
    memory_pool: Arc<GpuMemoryPool>,
}

/// Gestionnaire de mémoire GPU
struct GpuMemoryPool {
    buffers: parking_lot::RwLock<Vec<UMat>>,
    max_buffers: usize,
    buffer_size: Size,
}

impl GpuMemoryPool {
    fn new(max_buffers: usize, buffer_size: Size) -> Self {
        Self {
            buffers: parking_lot::RwLock::new(Vec::with_capacity(max_buffers)),
            max_buffers,
            buffer_size,
        }
    }

    /// Obtient un buffer du pool ou en crée un nouveau
    fn get_buffer(&self) -> Result<UMat> {
        let mut buffers = self.buffers.write();
        if let Some(buffer) = buffers.pop() {
            Ok(buffer)
        } else {
            let buffer = UMat::new(USAGE_DEFAULT)?;
            buffer.create(self.buffer_size.height, self.buffer_size.width, opencv::core::CV_8UC3)?;
            Ok(buffer)
        }
    }

    /// Retourne un buffer au pool
    fn return_buffer(&self, buffer: UMat) {
        let mut buffers = self.buffers.write();
        if buffers.len() < self.max_buffers {
            buffers.push(buffer);
        }
    }
}

impl OptimizationEngine {
    pub fn new(config: OptimizationConfig) -> Result<Self> {
        // Vérifier le support CUDA
        let cuda_enabled = cudaimgproc::get_cuda_enabled_device_count()? > 0;
        let cuda_stream = if cuda_enabled {
            Some(opencv::core::Stream::new()?)
        } else {
            None
        };

        // Initialiser le pool de mémoire
        let memory_pool = Arc::new(GpuMemoryPool::new(
            config.max_gpu_buffers,
            Size::new(2048, 2048), // Taille max par défaut
        ));

        Ok(Self {
            config,
            cuda_enabled,
            cuda_stream,
            memory_pool,
        })
    }

    /// Optimise une image avec support GPU/CPU automatique
    pub fn optimize(&self, image: &Mat) -> Result<OptimizationResult> {
        let start = Instant::now();
        let mut metrics = OptimizationMetrics {
            total_time_ms: 0.0,
            preprocessing_time_ms: 0.0,
            optimization_time_ms: 0.0,
            postprocessing_time_ms: 0.0,
            gpu_memory_used_mb: 0.0,
            cpu_memory_used_mb: 0.0,
        };

        // Validation et prétraitement
        let preprocess_start = Instant::now();
        self.validate_input(image)?;
        let optimal_size = self.get_optimal_size(image.size()?);
        metrics.preprocessing_time_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;

        // Optimisation
        let optimize_start = Instant::now();
        let result = if self.cuda_enabled {
            self.optimize_cuda(image, optimal_size)?
        } else {
            self.optimize_cpu(image, optimal_size)?
        };
        metrics.optimization_time_ms = optimize_start.elapsed().as_secs_f64() * 1000.0;

        // Post-traitement et validation
        let postprocess_start = Instant::now();
        let quality = self.validate_result(&result, image)?;
        metrics.postprocessing_time_ms = postprocess_start.elapsed().as_secs_f64() * 1000.0;

        // Métriques finales
        metrics.total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        metrics.gpu_memory_used_mb = self.get_gpu_memory_usage()?;
        metrics.cpu_memory_used_mb = self.get_cpu_memory_usage()?;

        // Enregistrer les métriques
        self.record_metrics(&metrics);

        Ok(OptimizationResult {
            image: result,
            quality,
            metrics: Some(metrics),
        })
    }

    /// Optimisation sur GPU avec CUDA
    fn optimize_cuda(&self, image: &Mat, target_size: Size) -> Result<Mat> {
        let stream = self.cuda_stream.as_ref()
            .context("CUDA stream non initialisé")?;

        // Obtenir un buffer du pool
        let mut gpu_buffer = self.memory_pool.get_buffer()?;

        // Redimensionner sur GPU
        cudawarping::resize(
            image,
            &mut gpu_buffer,
            target_size,
            0.0,
            0.0,
            cudawarping::INTER_AREA,
            stream,
        )?;

        // Optimiser sur GPU
        let mut optimized = UMat::new(USAGE_DEFAULT)?;
        self.apply_cuda_optimizations(&gpu_buffer, &mut optimized, stream)?;

        // Retourner le buffer au pool
        self.memory_pool.return_buffer(gpu_buffer);

        // Transférer le résultat sur CPU
        let mut result = Mat::default();
        optimized.copy_to(&mut result)?;

        Ok(result)
    }

    /// Optimisation sur CPU (fallback)
    fn optimize_cpu(&self, image: &Mat, target_size: Size) -> Result<Mat> {
        let mut resized = Mat::default();
        opencv::imgproc::resize(
            image,
            &mut resized,
            target_size,
            0.0,
            0.0,
            opencv::imgproc::INTER_AREA,
        )?;

        let mut result = Mat::default();
        self.apply_cpu_optimizations(&resized, &mut result)?;

        Ok(result)
    }

    /// Applique les optimisations CUDA
    fn apply_cuda_optimizations(&self, input: &UMat, output: &mut UMat, stream: &opencv::core::Stream) -> Result<()> {
        // TODO: Implémenter les optimisations CUDA spécifiques
        // - Réduction du bruit
        // - Amélioration du contraste
        // - Correction des couleurs
        input.copy_to(output)?;
        Ok(())
    }

    /// Applique les optimisations CPU
    fn apply_cpu_optimizations(&self, input: &Mat, output: &mut Mat) -> Result<()> {
        // TODO: Implémenter les optimisations CPU
        // Version simplifiée des optimisations GPU
        input.copy_to(output)?;
        Ok(())
    }

    /// Valide l'image d'entrée
    fn validate_input(&self, image: &Mat) -> Result<()> {
        validation::check_not_empty(image)
            .context("Image d'entrée vide")?;
            
        validation::check_dimensions(
            image,
            self.config.min_image_size,
            self.config.max_image_size
        ).context("Dimensions d'image invalides")?;
        
        validation::check_image_type(image)
            .context("Type d'image non supporté")?;
            
        Ok(())
    }

    /// Valide le résultat et calcule la qualité
    fn validate_result(&self, result: &Mat, original: &Mat) -> Result<Quality> {
        // Vérifier le type
        if result.type_()? != original.type_()? {
            anyhow::bail!("Type du résultat incorrect");
        }
        
        // Calculer la qualité
        let quality = self.compute_quality(result, original)?;
        
        // Vérifier si la qualité est acceptable
        if quality.psnr < self.config.min_quality {
            anyhow::bail!("Qualité d'optimisation insuffisante");
        }
        
        Ok(quality)
    }

    /// Détermine la taille optimale pour l'image
    fn get_optimal_size(&self, current_size: Size) -> Size {
        // Trouver la taille prédéfinie la plus proche
        IMAGE_SIZES
            .iter()
            .min_by_key(|&&size| {
                ((size.width as i32 - current_size.width).abs()
                    + (size.height as i32 - current_size.height).abs()) as u32
            })
            .copied()
            .unwrap_or(current_size)
    }

    /// Calcule les métriques de qualité
    fn compute_quality(&self, result: &Mat, original: &Mat) -> Result<Quality> {
        let psnr = opencv::core::PSNR(result, original, 255.0)?;
        Ok(Quality { psnr })
    }

    /// Obtient l'utilisation mémoire GPU
    fn get_gpu_memory_usage(&self) -> Result<f64> {
        if self.cuda_enabled {
            // TODO: Implémenter la mesure de mémoire GPU
            Ok(0.0)
        } else {
            Ok(0.0)
        }
    }

    /// Obtient l'utilisation mémoire CPU
    fn get_cpu_memory_usage(&self) -> Result<f64> {
        // TODO: Implémenter la mesure de mémoire CPU
        Ok(0.0)
    }

    /// Enregistre les métriques de performance
    fn record_metrics(&self, metrics: &OptimizationMetrics) {
        gauge!("optimization.total_time_ms", metrics.total_time_ms);
        gauge!("optimization.preprocessing_time_ms", metrics.preprocessing_time_ms);
        gauge!("optimization.optimization_time_ms", metrics.optimization_time_ms);
        gauge!("optimization.postprocessing_time_ms", metrics.postprocessing_time_ms);
        gauge!("optimization.gpu_memory_mb", metrics.gpu_memory_used_mb);
        gauge!("optimization.cpu_memory_mb", metrics.cpu_memory_used_mb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::Scalar;

    fn create_test_image() -> Result<Mat> {
        let mut image = Mat::new_size_with_default(
            Size::new(640, 480),
            opencv::core::CV_8UC3,
            Scalar::all(255.0)
        )?;
        Ok(image)
    }

    #[test]
    fn test_validation() -> Result<()> {
        let config = OptimizationConfig::default();
        let engine = OptimizationEngine::new(config)?;
        
        // Image vide
        let empty_mat = Mat::default();
        assert!(engine.validate_input(&empty_mat).is_err());
        
        // Image valide
        let valid_mat = create_test_image()?;
        assert!(engine.validate_input(&valid_mat).is_ok());
        
        Ok(())
    }

    #[test]
    fn test_optimization() -> Result<()> {
        let config = OptimizationConfig::default();
        let engine = OptimizationEngine::new(config)?;
        
        let image = create_test_image()?;
        let result = engine.optimize(&image)?;
        
        // Vérifier la qualité
        assert!(result.quality.psnr > 0.0);
        
        // Vérifier les métriques
        let metrics = result.metrics.unwrap();
        assert!(metrics.total_time_ms > 0.0);
        assert!(metrics.preprocessing_time_ms > 0.0);
        assert!(metrics.optimization_time_ms > 0.0);
        assert!(metrics.postprocessing_time_ms > 0.0);
        
        Ok(())
    }

    #[test]
    fn test_cuda_fallback() -> Result<()> {
        let config = OptimizationConfig::default();
        let engine = OptimizationEngine::new(config)?;
        
        let image = create_test_image()?;
        
        // La reconstruction devrait fonctionner même si CUDA n'est pas disponible
        let result = engine.optimize(&image)?;
        assert!(result.quality.psnr > 0.0);
        
        Ok(())
    }

    #[test]
    fn test_optimal_size() -> Result<()> {
        let config = OptimizationConfig::default();
        let engine = OptimizationEngine::new(config)?;
        
        // Tester différentes tailles
        let sizes = [
            Size::new(200, 200),   // Devrait donner 256x256
            Size::new(600, 600),   // Devrait donner 512x512
            Size::new(1500, 1500), // Devrait donner 2048x2048
        ];
        
        for &size in &sizes {
            let optimal = engine.get_optimal_size(size);
            assert!(IMAGE_SIZES.contains(&optimal), 
                "Taille optimale {:?} non trouvée pour {:?}", optimal, size);
        }
        
        Ok(())
    }
}

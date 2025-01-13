use std::sync::Arc;
use rayon::prelude::*;
use opencv::core::Mat;
use anyhow::Result;
use tracing::{info, debug, warn};

use crate::config::OptimizationConfig;
use super::{cuda::CudaEngine, simd::SimdEngine};

pub struct ParallelEngine {
    config: OptimizationConfig,
    cuda_engine: Option<Arc<CudaEngine>>,
    simd_engine: Arc<SimdEngine>,
}

impl ParallelEngine {
    pub fn new(config: &OptimizationConfig) -> Result<Self> {
        // Initialize CUDA if available and requested
        let cuda_engine = if config.use_gpu {
            match CudaEngine::new(config) {
                Ok(engine) => {
                    info!("CUDA engine initialized successfully");
                    Some(Arc::new(engine))
                }
                Err(e) => {
                    warn!("Failed to initialize CUDA engine: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Initialize SIMD engine
        let simd_engine = Arc::new(SimdEngine::new());
        
        Ok(Self {
            config: config.clone(),
            cuda_engine,
            simd_engine,
        })
    }
    
    pub fn process_batch(&self, images: &[Mat]) -> Result<Vec<Mat>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        
        // Use GPU if available
        if let Some(cuda_engine) = &self.cuda_engine {
            debug!("Processing batch on GPU");
            return cuda_engine.process_batch(images);
        }
        
        // Otherwise use CPU with SIMD
        debug!("Processing batch on CPU with SIMD");
        self.process_cpu_parallel(images)
    }
    
    fn process_cpu_parallel(&self, images: &[Mat]) -> Result<Vec<Mat>> {
        // Configure thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.num_threads as usize)
            .build_global()?;
        
        // Process images in parallel
        let results: Result<Vec<_>> = images.par_iter()
            .map(|image| self.simd_engine.process_image(image))
            .collect();
        
        results
    }
    
    pub fn get_optimal_batch_size(&self) -> usize {
        if self.cuda_engine.is_some() {
            self.config.batch_size
        } else {
            // For CPU, use smaller batches
            (self.config.batch_size / 2).max(1)
        }
    }
    
    pub fn supports_gpu(&self) -> bool {
        self.cuda_engine.is_some()
    }
    
    pub fn get_num_threads(&self) -> u32 {
        self.config.num_threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::Scalar;
    
    #[test]
    fn test_parallel_processing() -> Result<()> {
        let config = OptimizationConfig {
            use_gpu: true,
            gpu_memory_limit: 4 * 1024 * 1024 * 1024,
            batch_size: 16,
            num_threads: 8,
        };
        
        let engine = ParallelEngine::new(&config)?;
        
        // Create test images
        let mut images = Vec::new();
        for _ in 0..4 {
            let image = Mat::new_rows_cols_with_default(
                512,
                512,
                opencv::core::CV_8UC3,
                Scalar::all(255.0),
            )?;
            images.push(image);
        }
        
        // Process batch
        let results = engine.process_batch(&images)?;
        
        // Verify results
        assert_eq!(results.len(), images.len());
        for (result, image) in results.iter().zip(images.iter()) {
            assert_eq!(result.size()?, image.size()?);
            assert_eq!(result.typ(), image.typ());
        }
        
        Ok(())
    }
    
    #[test]
    fn test_batch_size_optimization() {
        let config = OptimizationConfig {
            use_gpu: false,
            gpu_memory_limit: 4 * 1024 * 1024 * 1024,
            batch_size: 16,
            num_threads: 8,
        };
        
        let engine = ParallelEngine::new(&config).unwrap();
        
        // CPU batch size should be smaller
        assert!(engine.get_optimal_batch_size() < config.batch_size);
    }
    
    #[test]
    fn test_thread_configuration() {
        let config = OptimizationConfig {
            use_gpu: false,
            gpu_memory_limit: 4 * 1024 * 1024 * 1024,
            batch_size: 16,
            num_threads: 8,
        };
        
        let engine = ParallelEngine::new(&config).unwrap();
        
        assert_eq!(engine.get_num_threads(), config.num_threads);
    }
}

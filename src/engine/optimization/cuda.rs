use std::sync::Arc;
use opencv::core::{Mat, MatTraitConst};
use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use anyhow::Result;
use tracing::{info, debug, warn};

use crate::config::OptimizationConfig;

pub struct CudaEngine {
    context: Arc<Context>,
    stream: Stream,
    module: Module,
    config: OptimizationConfig,
}

impl CudaEngine {
    pub fn new(config: &OptimizationConfig) -> Result<Self> {
        rustacuda::init(rustacuda::CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, 
            device
        )?;
        
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
        let module = Module::load_from_string(&ptx)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        
        Ok(Self {
            context: Arc::new(context),
            stream,
            module,
            config: config.clone(),
        })
    }
    
    pub fn process_image(&self, image: &Mat) -> Result<Mat> {
        let (height, width) = (image.rows() as u32, image.cols() as u32);
        let channels = image.channels() as u32;
        let size = (width * height * channels) as usize;
        
        // Allouer la mémoire sur le GPU
        let mut input_gpu = unsafe { DeviceBuffer::uninitialized(size)? };
        let mut output_gpu = unsafe { DeviceBuffer::uninitialized(size)? };
        
        // Copier l'image vers le GPU
        input_gpu.copy_from(image.data_bytes()?.as_ref())?;
        
        // Lancer le kernel
        let block_size = 256;
        let grid_size = (size + block_size - 1) / block_size;
        
        unsafe {
            let function = self.module.get_function("process_image")?;
            let params = vec![
                input_gpu.as_device_ptr(),
                output_gpu.as_device_ptr(),
                &size,
                &width,
                &height,
                &channels,
            ];
            
            function.launch(
                LaunchConfig {
                    grid_size: (grid_size as u32, 1, 1),
                    block_size: (block_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &params,
            )?;
        }
        
        // Attendre que le kernel termine
        self.stream.synchronize()?;
        
        // Copier le résultat du GPU
        let mut output_cpu = vec![0u8; size];
        output_gpu.copy_to(&mut output_cpu)?;
        
        // Créer une nouvelle Mat avec le résultat
        let mut output_mat = unsafe {
            Mat::new_rows_cols_with_data(
                height as i32,
                width as i32,
                image.typ(),
                output_cpu.as_mut_ptr() as *mut _,
                Mat::AUTO_STEP,
            )?
        };
        
        Ok(output_mat)
    }
    
    pub fn process_batch(&self, images: &[Mat]) -> Result<Vec<Mat>> {
        debug!("Processing batch of {} images on GPU", images.len());
        
        let mut results = Vec::with_capacity(images.len());
        for image in images {
            results.push(self.process_image(image)?);
        }
        
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::imgcodecs;
    
    #[test]
    fn test_cuda_processing() -> Result<()> {
        let config = OptimizationConfig {
            use_gpu: true,
            gpu_memory_limit: 4 * 1024 * 1024 * 1024, // 4GB
            batch_size: 16,
            num_threads: 8,
        };
        
        let engine = CudaEngine::new(&config)?;
        
        // Create test image
        let mut image = Mat::new_rows_cols_with_default(
            512,
            512,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::all(255.0),
        )?;
        
        // Process image
        let result = engine.process_image(&image)?;
        
        // Verify result
        assert_eq!(result.size()?, image.size()?);
        assert_eq!(result.typ(), image.typ());
        
        Ok(())
    }
    
    #[test]
    fn test_batch_processing() -> Result<()> {
        let config = OptimizationConfig {
            use_gpu: true,
            gpu_memory_limit: 4 * 1024 * 1024 * 1024,
            batch_size: 16,
            num_threads: 8,
        };
        
        let engine = CudaEngine::new(&config)?;
        
        // Create test images
        let mut images = Vec::new();
        for _ in 0..4 {
            let image = Mat::new_rows_cols_with_default(
                512,
                512,
                opencv::core::CV_8UC3,
                opencv::core::Scalar::all(255.0),
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
}

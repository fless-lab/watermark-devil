use anyhow::Result;
use opencv::{
    core,
    prelude::*,
    videoio::{self, VideoWriter},
    cudacodec::{self, cuda_GpuMat},
};
use tokio::sync::mpsc;
use crate::types::{Frame, FrameBatch, VideoMetadata};
use crate::config::VideoConfig;

/// Advanced video encoder with hardware acceleration and quality control
pub struct VideoEncoder {
    config: VideoConfig,
    writer: Option<VideoWriter>,
    gpu_encoder: Option<cudacodec::VideoWriter>,
    metadata: Option<VideoMetadata>,
}

impl VideoEncoder {
    pub fn new(config: &VideoConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            writer: None,
            gpu_encoder: None,
            metadata: None,
        })
    }
    
    /// Configure encoder with video metadata
    pub async fn configure(
        &mut self,
        metadata: &VideoMetadata,
        output_path: &str,
    ) -> Result<()> {
        self.metadata = Some(metadata.clone());
        
        // Try hardware encoder first
        if self.config.use_gpu {
            if let Ok(encoder) = self.init_gpu_encoder(metadata, output_path) {
                self.gpu_encoder = Some(encoder);
                return Ok(());
            }
        }
        
        // Fall back to CPU encoder
        let fourcc = videoio::VideoWriter::fourcc(
            b'H' as i8,
            b'2' as i8,
            b'6' as i8,
            b'4' as i8,
        )?;
        
        let writer = VideoWriter::new(
            output_path,
            fourcc,
            metadata.fps,
            core::Size::new(metadata.width, metadata.height),
            true,
        )?;
        
        self.writer = Some(writer);
        Ok(())
    }
    
    /// Write batch of frames
    pub async fn write_batch(&mut self, batch: &FrameBatch) -> Result<()> {
        for frame in batch.iter() {
            self.write_frame(frame).await?;
        }
        Ok(())
    }
    
    /// Write single frame
    async fn write_frame(&mut self, frame: &Frame) -> Result<()> {
        if let Some(encoder) = &mut self.gpu_encoder {
            // Upload frame to GPU
            let mut gpu_frame = cuda_GpuMat::default()?;
            gpu_frame.upload(&frame.to_mat()?)?;
            
            // Encode frame
            encoder.write(&gpu_frame)?;
        } else if let Some(writer) = &mut self.writer {
            // Write frame directly
            writer.write(&frame.to_mat()?)?;
        } else {
            return Err(anyhow::anyhow!("No encoder initialized"));
        }
        
        Ok(())
    }
    
    /// Initialize GPU encoder
    fn init_gpu_encoder(
        &self,
        metadata: &VideoMetadata,
        output_path: &str,
    ) -> Result<cudacodec::VideoWriter> {
        let params = cudacodec::EncoderParams::default()?
            .preset(cudacodec::Preset::P7)
            .quality(self.config.quality)
            .target_size(core::Size::new(metadata.width, metadata.height))
            .fps(metadata.fps as f64);
        
        Ok(cudacodec::VideoWriter::new(output_path, &params)?)
    }
    
    /// Finalize encoding
    pub async fn finalize(&mut self) -> Result<()> {
        // Release encoders
        self.writer = None;
        self.gpu_encoder = None;
        
        Ok(())
    }
}

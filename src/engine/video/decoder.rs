use std::collections::VecDeque;
use anyhow::Result;
use opencv::{
    core,
    prelude::*,
    videoio::{self, VideoCapture},
    cudacodec::{self, cuda_GpuMat},
};
use tokio::sync::mpsc;
use crate::types::{Frame, FrameBatch, VideoMetadata};
use crate::config::VideoConfig;

/// Advanced video decoder with hardware acceleration support
pub struct VideoDecoder {
    config: VideoConfig,
    capture: Option<VideoCapture>,
    gpu_decoder: Option<cudacodec::VideoReader>,
    frame_queue: VecDeque<Frame>,
    metadata: Option<VideoMetadata>,
}

impl VideoDecoder {
    pub fn new(config: &VideoConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            capture: None,
            gpu_decoder: None,
            frame_queue: VecDeque::new(),
            metadata: None,
        })
    }
    
    /// Open video file and initialize decoder
    pub async fn open(&mut self, path: &str) -> Result<VideoMetadata> {
        // Try hardware decoder first
        if self.config.use_gpu {
            if let Ok(decoder) = self.init_gpu_decoder(path) {
                self.gpu_decoder = Some(decoder);
            }
        }
        
        // Fall back to CPU decoder if needed
        if self.gpu_decoder.is_none() {
            let mut capture = VideoCapture::from_file(path, videoio::CAP_ANY)?;
            
            // Configure capture
            capture.set(videoio::CAP_PROP_BUFFERSIZE, self.config.batch_size as f64)?;
            capture.set(videoio::CAP_PROP_FORMAT, core::CV_8UC3 as f64)?;
            
            self.capture = Some(capture);
        }
        
        // Get video metadata
        let metadata = self.get_metadata()?;
        self.metadata = Some(metadata.clone());
        
        Ok(metadata)
    }
    
    /// Read batch of frames
    pub async fn read_batch(&mut self) -> Result<Option<FrameBatch>> {
        let batch_size = self.config.batch_size;
        let mut batch = Vec::with_capacity(batch_size);
        
        // Read frames until batch is full or video ends
        while batch.len() < batch_size {
            if let Some(frame) = self.read_frame().await? {
                batch.push(frame);
            } else {
                break;
            }
        }
        
        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(FrameBatch::from(batch)))
        }
    }
    
    /// Read single frame
    async fn read_frame(&mut self) -> Result<Option<Frame>> {
        // Check frame queue first
        if let Some(frame) = self.frame_queue.pop_front() {
            return Ok(Some(frame));
        }
        
        // Read new frame
        if let Some(decoder) = &mut self.gpu_decoder {
            // Read from GPU decoder
            let mut gpu_frame = cuda_GpuMat::default()?;
            if decoder.next_frame(&mut gpu_frame)? {
                let mut cpu_frame = core::Mat::default()?;
                gpu_frame.download(&mut cpu_frame)?;
                Ok(Some(Frame::from(cpu_frame)))
            } else {
                Ok(None)
            }
        } else if let Some(capture) = &mut self.capture {
            // Read from CPU decoder
            let mut frame = core::Mat::default()?;
            if capture.read(&mut frame)? {
                Ok(Some(Frame::from(frame)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
    
    /// Initialize GPU decoder
    fn init_gpu_decoder(&self, path: &str) -> Result<cudacodec::VideoReader> {
        let params = cudacodec::VideoReaderInitParams::default()?
            .raw_mode(false)
            .min_num_decode_surfaces(self.config.batch_size as i32);
        
        Ok(cudacodec::VideoReader::new(path, &params)?)
    }
    
    /// Get video metadata
    fn get_metadata(&self) -> Result<VideoMetadata> {
        let (width, height, fps, frame_count) = if let Some(decoder) = &self.gpu_decoder {
            let format = decoder.format()?;
            (
                format.width,
                format.height,
                format.fps,
                format.number_of_frames,
            )
        } else if let Some(capture) = &self.capture {
            (
                capture.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32,
                capture.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32,
                capture.get(videoio::CAP_PROP_FPS)?,
                capture.get(videoio::CAP_PROP_FRAME_COUNT)? as i64,
            )
        } else {
            return Err(anyhow::anyhow!("No decoder initialized"));
        };
        
        Ok(VideoMetadata {
            width,
            height,
            fps,
            frame_count,
            processed_frames: 0,
        })
    }
}

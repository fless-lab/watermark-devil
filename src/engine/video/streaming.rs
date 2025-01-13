use std::time::{Duration, Instant};
use anyhow::Result;
use opencv::{
    core,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter},
    gapi::{self, GStreamingCompiled},
};
use tokio::sync::mpsc;
use crate::types::{Frame, VideoMetadata};
use crate::config::VideoConfig;

/// Advanced streaming processor for real-time video processing
pub struct StreamingProcessor {
    config: VideoConfig,
    capture: Option<VideoCapture>,
    writer: Option<VideoWriter>,
    pipeline: Option<GStreamingCompiled>,
    frame_times: Vec<Duration>,
    start_time: Option<Instant>,
}

impl StreamingProcessor {
    pub fn new(config: &VideoConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            capture: None,
            writer: None,
            pipeline: None,
            frame_times: Vec::new(),
            start_time: None,
        })
    }
    
    /// Initialize streaming processor
    pub async fn initialize(
        &mut self,
        input_url: &str,
        output_url: Option<&str>,
    ) -> Result<()> {
        // Initialize video capture
        let mut capture = VideoCapture::from_file(input_url, videoio::CAP_ANY)?;
        
        // Configure capture for streaming
        capture.set(videoio::CAP_PROP_BUFFERSIZE, 3.0)?;
        capture.set(videoio::CAP_PROP_FORMAT, core::CV_8UC3 as f64)?;
        
        // Enable hardware acceleration if available
        if self.config.use_gpu {
            capture.set(videoio::CAP_PROP_HW_ACCELERATION, videoio::VIDEO_ACCELERATION_ANY as f64)?;
        }
        
        self.capture = Some(capture);
        
        // Initialize video writer if output URL is provided
        if let Some(url) = output_url {
            let fourcc = videoio::VideoWriter::fourcc(
                b'H' as i8,
                b'2' as i8,
                b'6' as i8,
                b'4' as i8,
            )?;
            
            let writer = VideoWriter::new(
                url,
                fourcc,
                30.0, // FPS
                core::Size::new(1920, 1080), // Resolution
                true,
            )?;
            
            self.writer = Some(writer);
        }
        
        // Initialize timing
        self.start_time = Some(Instant::now());
        
        Ok(())
    }
    
    /// Read frame from stream
    pub async fn read_frame(&mut self) -> Result<Option<Frame>> {
        if let Some(capture) = &mut self.capture {
            let mut frame = core::Mat::default()?;
            if capture.read(&mut frame)? {
                // Update timing statistics
                if let Some(start) = self.start_time {
                    let elapsed = start.elapsed();
                    self.frame_times.push(elapsed);
                    
                    // Calculate FPS and latency
                    if self.frame_times.len() >= 30 {
                        self.update_streaming_stats()?;
                        self.frame_times.clear();
                    }
                }
                
                Ok(Some(Frame::from(frame)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
    
    /// Write frame to stream
    pub async fn write_frame(&mut self, frame: &Frame) -> Result<()> {
        if let Some(writer) = &mut self.writer {
            writer.write(&frame.to_mat()?)?;
        }
        Ok(())
    }
    
    /// Update streaming statistics
    fn update_streaming_stats(&self) -> Result<()> {
        if self.frame_times.is_empty() {
            return Ok(());
        }
        
        // Calculate FPS
        let total_time = self.frame_times.last().unwrap() - self.frame_times[0];
        let fps = self.frame_times.len() as f64 / total_time.as_secs_f64();
        
        // Calculate average latency
        let avg_latency = self.frame_times
            .windows(2)
            .map(|w| w[1] - w[0])
            .sum::<Duration>()
            .div_f64(self.frame_times.len() as f64 - 1.0);
        
        // Log statistics
        log::info!(
            "Streaming stats - FPS: {:.2}, Latency: {:.2}ms",
            fps,
            avg_latency.as_secs_f64() * 1000.0
        );
        
        Ok(())
    }
    
    /// Create optimized streaming pipeline
    pub fn create_pipeline(&mut self) -> Result<()> {
        use gapi::{GProtoArg, GKernelPackage};
        
        // Create G-API pipeline for streaming
        let g_in = gapi::GMat::new();
        
        // Add preprocessing
        let preprocessed = gapi::preprocessing(&g_in)?;
        
        // Add inference
        let features = gapi::inference(&preprocessed)?;
        
        // Add postprocessing
        let output = gapi::postprocessing(&features)?;
        
        // Compile pipeline with streaming optimizations
        let pipeline = gapi::GComputation::new(g_in, output)
            .compile(gapi::GCompileArgs::new()
                .arg(gapi::compile_args::GraphOptimizations::new()
                    .enable_all()
                    .fusion(true)
                    .inline(true))
                .arg(gapi::compile_args::GStreamingCompileArgs::new()
                    .optimize(true)
                    .num_buffers(3)))?;
        
        self.pipeline = Some(pipeline);
        Ok(())
    }
}

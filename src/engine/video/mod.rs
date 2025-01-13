use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use opencv::{
    core,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter},
    gapi::{self, GStreamingCompiled}
};
use rayon::prelude::*;
use crate::types::{Frame, FrameBatch, VideoMetadata, ProcessingMode};
use crate::config::VideoConfig;

mod decoder;
mod encoder;
mod processor;
mod temporal;
mod streaming;

pub use decoder::VideoDecoder;
pub use encoder::VideoEncoder;
pub use processor::VideoProcessor;
pub use temporal::TemporalProcessor;
pub use streaming::StreamingProcessor;

/// Advanced video processing engine with support for:
/// - Hardware-accelerated decoding/encoding (NVENC, QuickSync, VA-API)
/// - Batch processing with temporal coherence
/// - Real-time streaming capabilities
/// - Adaptive quality control
pub struct VideoEngine {
    config: VideoConfig,
    decoder: Arc<VideoDecoder>,
    encoder: Arc<VideoEncoder>,
    processor: Arc<VideoProcessor>,
    temporal: Arc<TemporalProcessor>,
    streaming: Arc<StreamingProcessor>,
    pipeline: Option<GStreamingCompiled>,
}

impl VideoEngine {
    pub fn new(config: VideoConfig) -> Result<Self> {
        let decoder = Arc::new(VideoDecoder::new(&config)?);
        let encoder = Arc::new(VideoEncoder::new(&config)?);
        let processor = Arc::new(VideoProcessor::new(&config)?);
        let temporal = Arc::new(TemporalProcessor::new(&config)?);
        let streaming = Arc::new(StreamingProcessor::new(&config)?);
        
        Ok(Self {
            config,
            decoder,
            encoder,
            processor,
            temporal,
            streaming,
            pipeline: None,
        })
    }
    
    /// Process video file with advanced features
    pub async fn process_video(
        &mut self,
        input_path: &str,
        output_path: &str,
        mode: ProcessingMode,
    ) -> Result<VideoMetadata> {
        // Open video
        let mut metadata = self.decoder.open(input_path).await?;
        
        // Configure encoder
        self.encoder.configure(&metadata, output_path).await?;
        
        // Initialize temporal processor
        self.temporal.initialize(&metadata)?;
        
        // Create processing pipeline
        self.setup_pipeline(&metadata, mode)?;
        
        // Process video in batches
        let mut processed_frames = 0;
        while let Some(batch) = self.decoder.read_batch().await? {
            // Process batch with temporal coherence
            let processed = self.process_batch(batch, mode).await?;
            
            // Encode processed frames
            self.encoder.write_batch(&processed).await?;
            
            processed_frames += processed.len();
            metadata.processed_frames = processed_frames;
        }
        
        // Finalize processing
        self.encoder.finalize().await?;
        self.temporal.finalize()?;
        
        Ok(metadata)
    }
    
    /// Process video stream (e.g., webcam, RTSP)
    pub async fn process_stream(
        &mut self,
        stream_url: &str,
        output_url: Option<&str>,
        mode: ProcessingMode,
    ) -> Result<()> {
        // Initialize streaming processor
        self.streaming.initialize(stream_url, output_url).await?;
        
        // Process stream in real-time
        while let Some(frame) = self.streaming.read_frame().await? {
            // Process frame
            let processed = self.processor.process_frame(frame, mode).await?;
            
            // Apply temporal smoothing
            let smoothed = self.temporal.process_frame(processed).await?;
            
            // Output processed frame
            if let Some(output) = output_url {
                self.streaming.write_frame(&smoothed).await?;
            }
        }
        
        Ok(())
    }
    
    /// Process batch of frames with temporal coherence
    async fn process_batch(
        &self,
        batch: FrameBatch,
        mode: ProcessingMode,
    ) -> Result<FrameBatch> {
        // Process frames in parallel
        let processed: Vec<Frame> = batch
            .into_par_iter()
            .map(|frame| {
                self.processor.process_frame(frame, mode)
            })
            .collect::<Result<Vec<_>>>()?;
        
        // Apply temporal coherence
        let smoothed = self.temporal.process_batch(&processed).await?;
        
        Ok(FrameBatch::from(smoothed))
    }
    
    /// Setup optimized processing pipeline
    fn setup_pipeline(
        &mut self,
        metadata: &VideoMetadata,
        mode: ProcessingMode,
    ) -> Result<()> {
        use gapi::{GProtoArg, GKernelPackage};
        
        // Create G-API pipeline
        let g_in = gapi::GMat::new();
        let g_out = match mode {
            ProcessingMode::Detection => {
                self.processor.create_detection_pipeline(&g_in)?
            }
            ProcessingMode::Reconstruction => {
                self.processor.create_reconstruction_pipeline(&g_in)?
            }
            ProcessingMode::Learning => {
                self.processor.create_learning_pipeline(&g_in)?
            }
        };
        
        // Compile pipeline with optimizations
        let pipeline = gapi::GComputation::new(g_in, g_out)
            .compile(gapi::GCompileArgs::new()
                .arg(gapi::compile_args::GraphOptimizations::new()
                    .enable_all()
                    .fusion(true)
                    .inline(true))
                .arg(gapi::compile_args::GStreamingCompileArgs::new()
                    .optimize(true)
                    .num_buffers(self.config.batch_size as i32)))?;
        
        self.pipeline = Some(pipeline);
        Ok(())
    }
}

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Failed to load image: {0}")]
    ImageLoadError(String),

    #[error("Failed to save image: {0}")]
    ImageSaveError(String),

    #[error("Detection error: {0}")]
    DetectionError(String),

    #[error("Reconstruction error: {0}")]
    ReconstructionError(String),

    #[error("Optimization error: {0}")]
    OptimizationError(String),

    #[error("Learning system error: {0}")]
    LearningError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("GPU error: {0}")]
    GpuError(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<opencv::Error> for EngineError {
    fn from(err: opencv::Error) -> Self {
        EngineError::Internal(err.to_string())
    }
}

impl From<image::ImageError> for EngineError {
    fn from(err: image::ImageError) -> Self {
        EngineError::ImageLoadError(err.to_string())
    }
}

impl From<std::io::Error> for EngineError {
    fn from(err: std::io::Error) -> Self {
        EngineError::Internal(err.to_string())
    }
}

// Implémentations de conversion pour d'autres types d'erreurs spécifiques
impl From<rustacuda::error::CudaError> for EngineError {
    fn from(err: rustacuda::error::CudaError) -> Self {
        EngineError::GpuError(err.to_string())
    }
}

// Helper functions for error creation
impl EngineError {
    pub fn invalid_input<T: std::fmt::Display>(msg: T) -> Self {
        EngineError::InvalidInput(msg.to_string())
    }

    pub fn internal<T: std::fmt::Display>(msg: T) -> Self {
        EngineError::Internal(msg.to_string())
    }

    pub fn detection<T: std::fmt::Display>(msg: T) -> Self {
        EngineError::DetectionError(msg.to_string())
    }

    pub fn reconstruction<T: std::fmt::Display>(msg: T) -> Self {
        EngineError::ReconstructionError(msg.to_string())
    }
}

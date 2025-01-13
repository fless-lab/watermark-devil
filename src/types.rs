use std::fmt;
use serde::{Serialize, Deserialize};
use opencv::core::{Mat, Rect};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WatermarkType {
    Text,
    Logo,
    Pattern,
    Transparent,
    Complex,
}

impl fmt::Display for WatermarkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WatermarkType::Text => write!(f, "Text"),
            WatermarkType::Logo => write!(f, "Logo"),
            WatermarkType::Pattern => write!(f, "Pattern"),
            WatermarkType::Transparent => write!(f, "Transparent"),
            WatermarkType::Complex => write!(f, "Complex"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Confidence {
    pub value: f32,
    pub source: String,
}

impl Confidence {
    pub fn new(value: f32) -> Self {
        Self {
            value,
            source: "default".to_string(),
        }
    }

    pub fn with_source(value: f32, source: &str) -> Self {
        Self {
            value,
            source: source.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub watermark_type: WatermarkType,
    pub confidence: Confidence,
    pub bbox: Rect,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ReconstructionMethod {
    Inpainting,
    Diffusion,
    Frequency,
    Hybrid,
}

impl fmt::Display for ReconstructionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReconstructionMethod::Inpainting => write!(f, "Inpainting"),
            ReconstructionMethod::Diffusion => write!(f, "Diffusion"),
            ReconstructionMethod::Frequency => write!(f, "Frequency"),
            ReconstructionMethod::Hybrid => write!(f, "Hybrid"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReconstructionResult {
    pub image: Mat,
    pub quality_score: f32,
    pub method_used: ReconstructionMethod,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub success: bool,
    pub message: String,
    pub detections: Vec<Detection>,
    pub reconstruction_method: Option<ReconstructionMethod>,
    pub quality_score: Option<f32>,
    pub processing_time_ms: u64,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct VideoMetadata {
    pub width: i32,
    pub height: i32,
    pub fps: f64,
    pub frame_count: i64,
    pub processed_frames: i64,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub data: Vec<u8>,
    pub width: i32,
    pub height: i32,
    pub channels: i32,
}

impl Frame {
    pub fn to_mat(&self) -> opencv::Result<opencv::core::Mat> {
        use opencv::{core, prelude::*};
        
        let size = core::Size::new(self.width, self.height);
        let mat_type = match self.channels {
            1 => core::CV_8UC1,
            3 => core::CV_8UC3,
            4 => core::CV_8UC4,
            _ => return Err(opencv::Error::new(
                core::StsError,
                "Unsupported number of channels".to_string()
            )),
        };
        
        unsafe {
            let mut mat = core::Mat::new_rows_cols_with_data(
                size.height,
                size.width,
                mat_type,
                self.data.as_ptr() as *mut _,
                core::Mat_AUTO_STEP,
            )?;
            mat.set_size(&[size.height, size.width])?;
            Ok(mat)
        }
    }
    
    pub fn from(mat: opencv::core::Mat) -> Self {
        let mut data = Vec::new();
        mat.data_bytes().unwrap().iter().for_each(|&b| data.push(b));
        
        Self {
            data,
            width: mat.cols(),
            height: mat.rows(),
            channels: mat.channels(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrameBatch {
    frames: Vec<Frame>,
}

impl FrameBatch {
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
        }
    }
    
    pub fn from(frames: Vec<Frame>) -> Self {
        Self { frames }
    }
    
    pub fn len(&self) -> usize {
        self.frames.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
    
    pub fn iter(&self) -> std::slice::Iter<'_, Frame> {
        self.frames.iter()
    }
}

impl IntoIterator for FrameBatch {
    type Item = Frame;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.frames.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watermark_type_display() {
        assert_eq!(WatermarkType::Text.to_string(), "Text");
        assert_eq!(WatermarkType::Logo.to_string(), "Logo");
        assert_eq!(WatermarkType::Pattern.to_string(), "Pattern");
        assert_eq!(WatermarkType::Transparent.to_string(), "Transparent");
        assert_eq!(WatermarkType::Complex.to_string(), "Complex");
    }

    #[test]
    fn test_confidence() {
        let conf = Confidence::new(0.95);
        assert_eq!(conf.value, 0.95);
        assert_eq!(conf.source, "default");

        let conf_with_source = Confidence::with_source(0.85, "neural");
        assert_eq!(conf_with_source.value, 0.85);
        assert_eq!(conf_with_source.source, "neural");
    }

    #[test]
    fn test_reconstruction_method_display() {
        assert_eq!(ReconstructionMethod::Inpainting.to_string(), "Inpainting");
        assert_eq!(ReconstructionMethod::Diffusion.to_string(), "Diffusion");
        assert_eq!(ReconstructionMethod::Frequency.to_string(), "Frequency");
        assert_eq!(ReconstructionMethod::Hybrid.to_string(), "Hybrid");
    }

    #[test]
    fn test_detection_serialization() {
        let detection = Detection {
            watermark_type: WatermarkType::Text,
            confidence: Confidence::new(0.9),
            bbox: Rect::new(10, 20, 100, 50),
            metadata: Some(serde_json::json!({
                "text": "Sample",
                "font": "Arial"
            })),
        };

        let serialized = serde_json::to_string(&detection).unwrap();
        let deserialized: Detection = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.watermark_type, detection.watermark_type);
        assert_eq!(deserialized.confidence.value, detection.confidence.value);
        assert_eq!(deserialized.bbox.x, detection.bbox.x);
        assert_eq!(deserialized.bbox.y, detection.bbox.y);
        assert_eq!(deserialized.bbox.width, detection.bbox.width);
        assert_eq!(deserialized.bbox.height, detection.bbox.height);
    }
}

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class WatermarkType(str, Enum):
    LOGO = "logo"
    TEXT = "text"
    PATTERN = "pattern"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


class ReconstructionMethod(str, Enum):
    INPAINTING = "inpainting"
    DIFFUSION = "diffusion"
    FREQUENCY = "frequency"
    HYBRID = "hybrid"


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectionResult(BaseModel):
    id: str = Field(..., description="Unique identifier for the detection")
    confidence: float = Field(..., ge=0.0, le=1.0)
    watermark_type: WatermarkType
    bbox: BoundingBox
    mask: Optional[str] = Field(None, description="Base64 encoded binary mask")
    metadata: Optional[Dict[str, Any]] = None


class DetectionResponse(BaseModel):
    success: bool
    message: str
    detections: List[DetectionResult]
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReconstructionResult(BaseModel):
    id: str = Field(..., description="Unique identifier for the reconstruction")
    success: bool
    quality_score: float = Field(..., ge=0.0, le=1.0)
    method_used: ReconstructionMethod
    processing_time: float
    image_data: str = Field(..., description="Base64 encoded output image")
    metadata: Optional[Dict[str, Any]] = None


class ReconstructionResponse(BaseModel):
    success: bool
    message: str
    reconstructions: List[ReconstructionResult]
    original_image: Optional[str] = Field(None, description="Base64 encoded original image")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ProcessingOptions(BaseModel):
    detect_multiple: bool = Field(True, description="Detect multiple watermarks")
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    preferred_method: Optional[ReconstructionMethod] = None
    use_gpu: bool = Field(True, description="Use GPU acceleration if available")
    preserve_quality: bool = Field(True, description="Preserve original image quality")
    max_processing_time: Optional[float] = Field(
        None, description="Maximum processing time in seconds"
    )


class LearningResult(BaseModel):
    pattern_type: WatermarkType
    success_rate: float = Field(..., ge=0.0, le=1.0)
    improvements: List[Dict[str, float]]
    training_required: bool


class LearningResponse(BaseModel):
    success: bool
    message: str
    result: Optional[LearningResult]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SystemStatus(BaseModel):
    status: str
    gpu_available: bool
    model_versions: Dict[str, str]
    processed_images: int
    success_rate: float
    average_processing_time: float
    last_training: datetime
    memory_usage: Dict[str, float]

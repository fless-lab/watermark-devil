"""
Configuration for ML models and training pipelines.
"""
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """Base configuration for ML models"""
    name: str
    version: str
    input_size: tuple[int, int]
    batch_size: int
    learning_rate: float
    device: str = "cuda"

@dataclass
class DetectorConfig(ModelConfig):
    """Configuration for watermark detection models"""
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100

@dataclass
class ReconstructorConfig(ModelConfig):
    """Configuration for image reconstruction models"""
    patch_size: int = 256
    overlap: int = 32
    quality_threshold: float = 0.8

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    epochs: int
    batch_size: int
    learning_rate: float
    scheduler_step_size: int
    scheduler_gamma: float
    num_workers: int
    checkpoint_dir: Path
    log_dir: Path

# Default configurations
LOGO_DETECTOR_CONFIG = DetectorConfig(
    name="logo_detector",
    version="1.0.0",
    input_size=(512, 512),
    batch_size=16,
    learning_rate=1e-4,
    confidence_threshold=0.6
)

TEXT_DETECTOR_CONFIG = DetectorConfig(
    name="text_detector",
    version="1.0.0",
    input_size=(768, 768),
    batch_size=8,
    learning_rate=1e-4,
    confidence_threshold=0.7
)

PATTERN_DETECTOR_CONFIG = DetectorConfig(
    name="pattern_detector",
    version="1.0.0",
    input_size=(1024, 1024),
    batch_size=4,
    learning_rate=1e-4,
    confidence_threshold=0.5
)

import pytest
import numpy as np
import cv2
from pathlib import Path
import asyncio
from PIL import Image
import io

from src.api.processing import WatermarkProcessor
from src.api.models import ProcessingOptions, WatermarkType, ReconstructionMethod

# Fixtures
@pytest.fixture
def test_image():
    """Create a test image with a simulated watermark."""
    # Create base image
    image = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Add a simulated watermark
    watermark = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(watermark, "TEST WATERMARK", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
    
    # Place watermark on image
    x, y = 100, 200
    image[y:y+100, x:x+300] = cv2.addWeighted(
        image[y:y+100, x:x+300], 0.7,
        watermark, 0.3, 0
    )
    
    return image

@pytest.fixture
def processor():
    """Initialize the watermark processor."""
    return WatermarkProcessor()

@pytest.fixture
def processing_options():
    """Create default processing options."""
    return ProcessingOptions(
        detect_multiple=True,
        min_confidence=0.5,
        preferred_method=ReconstructionMethod.HYBRID,
        use_gpu=True,
        preserve_quality=True,
        max_processing_time=30.0
    )

# Tests
@pytest.mark.asyncio
async def test_detection(processor, test_image, processing_options):
    """Test watermark detection."""
    # Run detection
    result = await processor.detect(test_image, processing_options)
    
    # Verify response
    assert result.success
    assert len(result.detections) > 0
    
    # Verify first detection
    detection = result.detections[0]
    assert detection.confidence > 0.5
    assert detection.watermark_type in [WatermarkType.TEXT, WatermarkType.LOGO]
    assert detection.bbox is not None

@pytest.mark.asyncio
async def test_reconstruction(processor, test_image, processing_options):
    """Test watermark reconstruction."""
    # First detect
    detection_result = await processor.detect(test_image, processing_options)
    assert detection_result.success
    
    # Then reconstruct
    reconstruction_result = await processor.reconstruct(
        test_image,
        detection_result.detections[0].id,
        processing_options
    )
    
    # Verify response
    assert reconstruction_result.success
    assert len(reconstruction_result.reconstructions) > 0
    
    # Verify reconstruction
    reconstruction = reconstruction_result.reconstructions[0]
    assert reconstruction.quality_score > 0.7
    assert reconstruction.method_used is not None
    assert reconstruction.image_data is not None

@pytest.mark.asyncio
async def test_complete_pipeline(processor, test_image, processing_options):
    """Test complete detection and reconstruction pipeline."""
    # Run complete pipeline
    result = await processor.process_complete(test_image, processing_options)
    
    # Verify response
    assert result.success
    assert len(result.reconstructions) > 0
    
    # Verify quality
    for reconstruction in result.reconstructions:
        assert reconstruction.quality_score > 0.7
        assert reconstruction.success
        assert reconstruction.method_used is not None

@pytest.mark.asyncio
async def test_learning_update(processor, test_image, processing_options):
    """Test learning system update."""
    # Run pipeline first
    pipeline_result = await processor.process_complete(test_image, processing_options)
    assert pipeline_result.success
    
    # Update learning system
    learning_result = await processor.update_learning(
        test_image,
        pipeline_result.reconstructions[0].metadata["detection_id"],
        pipeline_result.reconstructions[0].id,
        {"quality": "good"}
    )
    
    # Verify response
    assert learning_result.success
    assert learning_result.result is not None
    assert learning_result.result.success_rate > 0

@pytest.mark.asyncio
async def test_system_status(processor):
    """Test system status retrieval."""
    # Get status
    status = await processor.get_status()
    
    # Verify basic fields
    assert status["status"] == "operational"
    assert isinstance(status["gpu_available"], bool)
    assert isinstance(status["processed_images"], int)
    assert isinstance(status["success_rate"], float)
    assert isinstance(status["average_processing_time"], float)
    assert all(key in status["model_versions"] for key in ["detection", "reconstruction", "learning"])

@pytest.mark.asyncio
async def test_error_handling(processor):
    """Test error handling with invalid inputs."""
    # Test with empty image
    empty_image = np.array([])
    result = await processor.detect(empty_image, None)
    assert not result.success
    assert result.message is not None
    
    # Test with invalid detection ID
    result = await processor.reconstruct(
        np.zeros((100, 100, 3), dtype=np.uint8),
        "invalid_id",
        None
    )
    assert not result.success
    assert result.message is not None

@pytest.mark.asyncio
async def test_performance(processor, test_image, processing_options):
    """Test processing performance."""
    # Measure detection time
    start_time = asyncio.get_event_loop().time()
    detection_result = await processor.detect(test_image, processing_options)
    detection_time = asyncio.get_event_loop().time() - start_time
    
    # Verify reasonable processing time
    assert detection_time < 5.0  # seconds
    
    # Measure reconstruction time
    start_time = asyncio.get_event_loop().time()
    reconstruction_result = await processor.process_complete(test_image, processing_options)
    reconstruction_time = asyncio.get_event_loop().time() - start_time
    
    # Verify reasonable processing time
    assert reconstruction_time < 10.0  # seconds

if __name__ == "__main__":
    pytest.main([__file__])

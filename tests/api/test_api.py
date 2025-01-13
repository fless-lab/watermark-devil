import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2
import io
from PIL import Image
import base64
import json

from src.api.main import app
from src.api.models import ProcessingOptions, WatermarkType, ReconstructionMethod

client = TestClient(app)

# Fixtures
@pytest.fixture
def test_image():
    """Create a test image file."""
    # Create image
    image = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Add watermark
    cv2.putText(image, "TEST WATERMARK", (100, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 128, 128), 3)
    
    # Convert to bytes
    success, buffer = cv2.imencode('.png', image)
    assert success
    
    return io.BytesIO(buffer)

@pytest.fixture
def processing_options():
    """Create processing options."""
    return {
        "detect_multiple": True,
        "min_confidence": 0.5,
        "preferred_method": "hybrid",
        "use_gpu": True,
        "preserve_quality": True,
        "max_processing_time": 30.0
    }

# Tests
def test_detect_endpoint(test_image, processing_options):
    """Test the detection endpoint."""
    response = client.post(
        "/detect",
        files={"file": ("test.png", test_image, "image/png")},
        data={"options": json.dumps(processing_options)}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"]
    assert len(data["detections"]) > 0
    assert all(d["confidence"] > 0.5 for d in data["detections"])

def test_reconstruct_endpoint(test_image, processing_options):
    """Test the reconstruction endpoint."""
    # First detect
    detect_response = client.post(
        "/detect",
        files={"file": ("test.png", test_image, "image/png")},
        data={"options": json.dumps(processing_options)}
    )
    assert detect_response.status_code == 200
    detection_id = detect_response.json()["detections"][0]["id"]
    
    # Then reconstruct
    response = client.post(
        "/reconstruct",
        files={"file": ("test.png", test_image, "image/png")},
        data={
            "detection_id": detection_id,
            "options": json.dumps(processing_options)
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"]
    assert len(data["reconstructions"]) > 0
    assert all(r["quality_score"] > 0.7 for r in data["reconstructions"])

def test_process_endpoint(test_image, processing_options):
    """Test the complete processing endpoint."""
    response = client.post(
        "/process",
        files={"file": ("test.png", test_image, "image/png")},
        data={"options": json.dumps(processing_options)}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"]
    assert len(data["reconstructions"]) > 0
    
    # Verify reconstruction results
    for reconstruction in data["reconstructions"]:
        assert reconstruction["success"]
        assert reconstruction["quality_score"] > 0.7
        assert "image_data" in reconstruction
        
        # Verify image data is valid base64
        try:
            image_bytes = base64.b64decode(reconstruction["image_data"])
            image = Image.open(io.BytesIO(image_bytes))
            assert image.size == (512, 512)
        except Exception as e:
            pytest.fail(f"Invalid image data: {e}")

def test_learn_endpoint(test_image):
    """Test the learning endpoint."""
    # First process an image
    process_response = client.post(
        "/process",
        files={"file": ("test.png", test_image, "image/png")}
    )
    assert process_response.status_code == 200
    process_data = process_response.json()
    
    # Then update learning
    response = client.post(
        "/learn",
        files={"file": ("test.png", test_image, "image/png")},
        data={
            "detection_id": process_data["reconstructions"][0]["metadata"]["detection_id"],
            "reconstruction_id": process_data["reconstructions"][0]["id"],
            "feedback": json.dumps({"quality": "good"})
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"]
    assert data["result"] is not None
    assert data["result"]["success_rate"] > 0

def test_status_endpoint():
    """Test the status endpoint."""
    response = client.get("/status")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert isinstance(data["gpu_available"], bool)
    assert isinstance(data["processed_images"], int)
    assert isinstance(data["success_rate"], float)

def test_error_handling():
    """Test API error handling."""
    # Test with invalid image
    response = client.post(
        "/detect",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    )
    assert response.status_code == 400
    
    # Test with missing file
    response = client.post("/detect")
    assert response.status_code == 422
    
    # Test with invalid detection ID
    response = client.post(
        "/reconstruct",
        files={"file": ("test.png", io.BytesIO(), "image/png")},
        data={"detection_id": "invalid_id"}
    )
    assert response.status_code == 400

def test_large_image_handling(processing_options):
    """Test handling of large images."""
    # Create large image
    large_image = np.ones((2048, 2048, 3), dtype=np.uint8) * 255
    success, buffer = cv2.imencode('.png', large_image)
    assert success
    
    response = client.post(
        "/process",
        files={"file": ("large.png", io.BytesIO(buffer), "image/png")},
        data={"options": json.dumps(processing_options)}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"]

def test_concurrent_requests(test_image, processing_options):
    """Test handling of concurrent requests."""
    import asyncio
    import aiohttp
    
    async def make_request():
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field("file", test_image, filename="test.png", content_type="image/png")
            data.add_field("options", json.dumps(processing_options))
            
            async with session.post("http://testserver/process", data=data) as response:
                return await response.json()
    
    # Make multiple concurrent requests
    n_requests = 5
    loop = asyncio.get_event_loop()
    tasks = [make_request() for _ in range(n_requests)]
    responses = loop.run_until_complete(asyncio.gather(*tasks))
    
    # Verify all requests succeeded
    assert all(r["success"] for r in responses)

if __name__ == "__main__":
    pytest.main([__file__])

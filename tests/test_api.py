"""
Tests for API components.
"""
import pytest
import asyncio
from fastapi import FastAPI, Request, UploadFile
from fastapi.testclient import TestClient
from pathlib import Path
import time
from unittest.mock import Mock, patch, AsyncMock
import io
from PIL import Image
import numpy as np

from api.validation import (
    validate_image_file,
    ImageMetadata,
    ValidationError
)
from api.rate_limiter import RateLimiter
from api.concurrency import ConcurrencyManager
from api.errors import (
    APIError,
    ValidationError as APIValidationError,
    RateLimitError,
    ConcurrencyError
)

@pytest.fixture
def test_image():
    """Fixture for test image"""
    # Créer une image test
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

@pytest.fixture
def mock_request():
    """Fixture for mock request"""
    request = Mock()
    request.client.host = "127.0.0.1"
    request.headers = {}
    request.state.request_id = "test-123"
    return request

class TestValidation:
    """Tests for input validation"""
    
    async def test_valid_image(self, test_image):
        """Test valid image validation"""
        file = UploadFile(
            filename="test.png",
            file=test_image
        )
        
        metadata = await validate_image_file(file)
        assert isinstance(metadata, ImageMetadata)
        assert metadata.width == 100
        assert metadata.height == 100
        assert metadata.format == "png"
    
    async def test_invalid_size(self):
        """Test image size validation"""
        # Créer une grande image
        img = Image.fromarray(np.zeros((5000, 5000, 3), dtype=np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        file = UploadFile(
            filename="large.png",
            file=img_bytes
        )
        
        with pytest.raises(ValidationError):
            await validate_image_file(file)
    
    async def test_invalid_format(self):
        """Test image format validation"""
        file = UploadFile(
            filename="test.txt",
            file=io.BytesIO(b"not an image")
        )
        
        with pytest.raises(ValidationError):
            await validate_image_file(file)

class TestRateLimiter:
    """Tests for rate limiting"""
    
    def test_init(self):
        """Test rate limiter initialization"""
        limiter = RateLimiter()
        assert limiter.default_limits["anonymous"].requests == 100
        assert limiter.default_limits["authenticated"].requests == 1000
    
    async def test_rate_limit(self, mock_request):
        """Test rate limiting logic"""
        limiter = RateLimiter()
        
        # Première requête devrait passer
        await limiter.check_rate_limit(mock_request)
        
        # Simuler beaucoup de requêtes
        for _ in range(200):
            await limiter.check_rate_limit(mock_request)
        
        # La prochaine devrait être limitée
        with pytest.raises(RateLimitError):
            await limiter.check_rate_limit(mock_request)
    
    async def test_cleanup(self):
        """Test rate limiter cleanup"""
        limiter = RateLimiter()
        
        # Ajouter des données
        limiter.buckets["test"] = Mock()
        limiter.banned_ips["test"] = time.monotonic() - 3600
        
        # Nettoyer
        await limiter.cleanup()
        
        assert "test" not in limiter.banned_ips
        assert "test" not in limiter.buckets

class TestConcurrencyManager:
    """Tests for concurrency management"""
    
    def test_init(self):
        """Test concurrency manager initialization"""
        manager = ConcurrencyManager(max_concurrent_tasks=5)
        assert manager.max_tasks == 5
        assert len(manager.tasks) == 0
    
    async def test_task_management(self):
        """Test task management"""
        manager = ConcurrencyManager()
        
        async def mock_task():
            await asyncio.sleep(0.1)
            return "done"
        
        # Démarrer une tâche
        await manager.start_task("test", mock_task)
        
        # Vérifier le statut
        status = await manager.get_task_status("test")
        assert status["status"] in ["running", "completed"]
        
        # Attendre la fin
        await asyncio.sleep(0.2)
        status = await manager.get_task_status("test")
        assert status["status"] == "completed"
        assert status["result"] == "done"
    
    async def test_task_cancellation(self):
        """Test task cancellation"""
        manager = ConcurrencyManager()
        
        async def mock_task():
            await asyncio.sleep(1000)
        
        # Démarrer et annuler
        await manager.start_task("test", mock_task)
        success = await manager.cancel_task("test")
        
        assert success
        status = await manager.get_task_status("test")
        assert status["status"] == "failed"
    
    async def test_cleanup(self):
        """Test concurrency manager cleanup"""
        manager = ConcurrencyManager(task_timeout=0.1)
        
        async def mock_task():
            await asyncio.sleep(1000)
        
        # Démarrer une tâche
        await manager.start_task("test", mock_task)
        
        # Attendre et nettoyer
        await asyncio.sleep(0.2)
        await manager.cleanup()
        
        assert "test" not in manager.tasks

class TestErrorHandling:
    """Tests for error handling"""
    
    async def test_api_error(self, mock_request):
        """Test API error handling"""
        error = APIValidationError("Invalid input")
        response = await error_handler(mock_request, error)
        
        assert response.status_code == 400
        assert "code" in response.body
        assert "message" in response.body
    
    async def test_rate_limit_error(self, mock_request):
        """Test rate limit error handling"""
        error = RateLimitError("Too many requests")
        response = await error_handler(mock_request, error)
        
        assert response.status_code == 429
        assert "code" in response.body
        assert "message" in response.body
    
    async def test_concurrency_error(self, mock_request):
        """Test concurrency error handling"""
        error = ConcurrencyError("Task already running")
        response = await error_handler(mock_request, error)
        
        assert response.status_code == 409
        assert "code" in response.body
        assert "message" in response.body
    
    async def test_unknown_error(self, mock_request):
        """Test unknown error handling"""
        error = Exception("Unknown error")
        response = await error_handler(mock_request, error)
        
        assert response.status_code == 500
        assert "code" in response.body
        assert "message" in response.body

if __name__ == "__main__":
    pytest.main([__file__])

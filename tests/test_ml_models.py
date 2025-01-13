"""
Tests for ML models.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from ml.models.base import BaseModel, ModelError
from ml.models.detector import BaseDetector
from ml.config import ModelConfig, DetectorConfig

@pytest.fixture
def mock_model_config():
    """Fixture for model configuration"""
    return ModelConfig(
        name="test_model",
        version="1.0.0",
        input_size=(224, 224),
        batch_size=16,
        learning_rate=1e-4
    )

@pytest.fixture
def mock_detector_config():
    """Fixture for detector configuration"""
    return DetectorConfig(
        name="test_detector",
        version="1.0.0",
        input_size=(512, 512),
        batch_size=8,
        learning_rate=1e-4,
        confidence_threshold=0.5,
        iou_threshold=0.45
    )

class TestBaseModel:
    """Tests for BaseModel"""
    
    def test_init_validation(self, mock_model_config):
        """Test model initialization and validation"""
        class TestModel(BaseModel):
            def _validate_architecture(self):
                pass
            def _forward_impl(self, x):
                return x
        
        model = TestModel(mock_model_config)
        assert model.config == mock_model_config
        assert model.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_invalid_config(self):
        """Test invalid configuration"""
        with pytest.raises(ModelError):
            config = ModelConfig(
                name="",  # Invalid name
                version="1.0.0",
                input_size=(224, 224),
                batch_size=16,
                learning_rate=1e-4
            )
            
            class TestModel(BaseModel):
                def _validate_architecture(self):
                    pass
                def _forward_impl(self, x):
                    return x
            
            TestModel(config)
    
    def test_save_load(self, mock_model_config, tmp_path):
        """Test model save and load"""
        class TestModel(BaseModel):
            def _validate_architecture(self):
                pass
            def _forward_impl(self, x):
                return x
        
        model = TestModel(mock_model_config)
        save_path = tmp_path / "model.pt"
        
        # Save
        model.save(save_path)
        assert save_path.exists()
        
        # Load
        new_model = TestModel(mock_model_config)
        new_model.load(save_path)
        
        # Vérifier l'état
        assert new_model.batch_count == model.batch_count
        assert new_model.error_count == model.error_count
    
    def test_memory_usage(self, mock_model_config):
        """Test memory usage tracking"""
        class TestModel(BaseModel):
            def _validate_architecture(self):
                pass
            def _forward_impl(self, x):
                return x
        
        model = TestModel(mock_model_config)
        usage = model.get_memory_usage()
        
        assert "allocated" in usage
        assert "cached" in usage
        assert usage["allocated"] >= 0
        assert usage["cached"] >= 0

class TestBaseDetector:
    """Tests for BaseDetector"""
    
    def test_init_validation(self, mock_detector_config):
        """Test detector initialization and validation"""
        class TestDetector(BaseDetector):
            def _validate_architecture(self):
                pass
            def _forward_impl(self, x):
                return x
        
        detector = TestDetector(mock_detector_config)
        assert detector.config == mock_detector_config
    
    def test_process_detections(self, mock_detector_config):
        """Test detection processing"""
        class TestDetector(BaseDetector):
            def _validate_architecture(self):
                pass
            def _forward_impl(self, x):
                return x
        
        detector = TestDetector(mock_detector_config)
        
        # Créer des détections test
        raw_outputs = torch.tensor([
            [0, 0, 100, 100, 0.9],  # High confidence
            [50, 50, 150, 150, 0.8],  # High confidence, overlapping
            [0, 0, 50, 50, 0.3],  # Low confidence
        ])
        
        detections = detector.process_detections(
            raw_outputs,
            confidence_threshold=0.5,
            iou_threshold=0.5
        )
        
        assert len(detections) == 2  # Deux détections valides
        assert all(conf >= 0.5 for _, conf in detections)
    
    def test_nms(self, mock_detector_config):
        """Test non-maximum suppression"""
        class TestDetector(BaseDetector):
            def _validate_architecture(self):
                pass
            def _forward_impl(self, x):
                return x
        
        detector = TestDetector(mock_detector_config)
        
        # Créer des boîtes qui se chevauchent
        boxes = torch.tensor([
            [0, 0, 100, 100],
            [10, 10, 110, 110],  # Chevauchement important
            [200, 200, 300, 300]  # Pas de chevauchement
        ])
        
        scores = torch.tensor([0.9, 0.8, 0.7])
        
        keep = detector._nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 2  # Devrait garder deux boîtes
    
    def test_compute_ious(self, mock_detector_config):
        """Test IoU computation"""
        class TestDetector(BaseDetector):
            def _validate_architecture(self):
                pass
            def _forward_impl(self, x):
                return x
        
        detector = TestDetector(mock_detector_config)
        
        # Cas de test
        box = torch.tensor([0, 0, 100, 100])
        boxes = torch.tensor([
            [0, 0, 100, 100],  # IoU = 1.0
            [50, 50, 150, 150],  # IoU = 0.14
            [200, 200, 300, 300]  # IoU = 0.0
        ])
        
        ious = detector._compute_ious(box, boxes)
        assert len(ious) == 3
        assert torch.allclose(ious[0], torch.tensor(1.0))
        assert ious[1] > 0 and ious[1] < 1
        assert ious[2] == 0

if __name__ == "__main__":
    pytest.main([__file__])

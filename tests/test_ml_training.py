"""
Tests for ML training components.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from unittest.mock import Mock, patch
from ml.training.trainer import ModelTrainer, TrainingError
from ml.config import TrainingConfig, ModelConfig
from ml.models.base import BaseModel

@pytest.fixture
def mock_training_config(tmp_path):
    """Fixture for training configuration"""
    return TrainingConfig(
        epochs=2,
        batch_size=16,
        learning_rate=1e-4,
        scheduler_step_size=1,
        scheduler_gamma=0.1,
        num_workers=0,
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs"
    )

@pytest.fixture
def mock_model():
    """Fixture for mock model"""
    class TestModel(BaseModel):
        def __init__(self):
            config = ModelConfig(
                name="test_model",
                version="1.0.0",
                input_size=(224, 224),
                batch_size=16,
                learning_rate=1e-4
            )
            super().__init__(config)
            self.fc = nn.Linear(10, 1)
        
        def _validate_architecture(self):
            pass
        
        def _forward_impl(self, x):
            return self.fc(x)
    
    return TestModel()

@pytest.fixture
def mock_data():
    """Fixture for mock training data"""
    # Créer des données synthétiques
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=16)

class TestModelTrainer:
    """Tests for ModelTrainer"""
    
    def test_init_validation(self, mock_model, mock_training_config, mock_data):
        """Test trainer initialization and validation"""
        trainer = ModelTrainer(
            model=mock_model,
            config=mock_training_config,
            train_loader=mock_data,
            val_loader=mock_data
        )
        
        assert trainer.model == mock_model
        assert trainer.config == mock_training_config
        assert trainer.train_loader == mock_data
        assert trainer.val_loader == mock_data
    
    def test_invalid_config(self, mock_model, mock_data):
        """Test invalid training configuration"""
        with pytest.raises(TrainingError):
            config = TrainingConfig(
                epochs=-1,  # Invalid epochs
                batch_size=16,
                learning_rate=1e-4,
                scheduler_step_size=1,
                scheduler_gamma=0.1,
                num_workers=0,
                checkpoint_dir=Path("/tmp"),
                log_dir=Path("/tmp")
            )
            
            ModelTrainer(
                model=mock_model,
                config=config,
                train_loader=mock_data
            )
    
    def test_empty_dataloader(self, mock_model, mock_training_config):
        """Test empty dataloader validation"""
        empty_data = DataLoader(TensorDataset(torch.tensor([]), torch.tensor([])))
        
        with pytest.raises(TrainingError):
            ModelTrainer(
                model=mock_model,
                config=mock_training_config,
                train_loader=empty_data
            )
    
    def test_training_loop(self, mock_model, mock_training_config, mock_data):
        """Test complete training loop"""
        trainer = ModelTrainer(
            model=mock_model,
            config=mock_training_config,
            train_loader=mock_data,
            val_loader=mock_data
        )
        
        # Entraîner
        results = trainer.train()
        
        assert "train_loss" in results
        assert "val_loss" in results
        assert "best_val_loss" in results
        assert len(trainer.train_losses) == mock_training_config.epochs
    
    def test_early_stopping(self, mock_model, mock_training_config, mock_data):
        """Test early stopping behavior"""
        class MockLoss:
            def __call__(self, x, y):
                return torch.tensor(1.0)
        
        trainer = ModelTrainer(
            model=mock_model,
            config=mock_training_config,
            train_loader=mock_data,
            val_loader=mock_data,
            criterion=MockLoss()
        )
        
        results = trainer.train()
        assert results["best_val_loss"] == 1.0
    
    def test_checkpointing(self, mock_model, mock_training_config, mock_data):
        """Test model checkpointing"""
        trainer = ModelTrainer(
            model=mock_model,
            config=mock_training_config,
            train_loader=mock_data,
            val_loader=mock_data
        )
        
        # Entraîner et vérifier les checkpoints
        trainer.train()
        
        checkpoint_files = list(mock_training_config.checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0
    
    def test_callbacks(self, mock_model, mock_training_config, mock_data):
        """Test training callbacks"""
        mock_callback = Mock()
        callbacks = {
            "on_epoch_end": mock_callback
        }
        
        trainer = ModelTrainer(
            model=mock_model,
            config=mock_training_config,
            train_loader=mock_data,
            callbacks=callbacks
        )
        
        trainer.train()
        assert mock_callback.call_count == mock_training_config.epochs
    
    def test_lr_scheduling(self, mock_model, mock_training_config, mock_data):
        """Test learning rate scheduling"""
        scheduler = torch.optim.lr_scheduler.StepLR(
            mock_model.parameters(),
            step_size=1,
            gamma=0.1
        )
        
        trainer = ModelTrainer(
            model=mock_model,
            config=mock_training_config,
            train_loader=mock_data,
            scheduler=scheduler
        )
        
        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.train()
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        
        assert final_lr < initial_lr

if __name__ == "__main__":
    pytest.main([__file__])

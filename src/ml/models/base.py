"""
Base classes and utilities for ML models.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from metrics import counter, gauge, histogram
from ..config import ModelConfig

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Base exception for model errors"""
    pass

class BaseModel(nn.Module, ABC):
    """Base class for all ML models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.to(self.device)
        self._validate_model()
        self._init_metrics()
    
    def _validate_model(self) -> None:
        """Validate model configuration and architecture"""
        try:
            # Valider la configuration
            self.config.validate()
            
            # Vérifier la disponibilité du GPU si nécessaire
            if self.config.device == "cuda" and not torch.cuda.is_available():
                raise ModelError("CUDA requested but not available")
                
            # Vérifier l'architecture du modèle
            self._validate_architecture()
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise ModelError(f"Model validation failed: {e}")
    
    @abstractmethod
    def _validate_architecture(self) -> None:
        """Validate model architecture"""
        pass
    
    def _init_metrics(self) -> None:
        """Initialize model metrics"""
        self.batch_count = 0
        self.error_count = 0
        gauge(f"ml.{self.config.name}.parameters", 
              sum(p.numel() for p in self.parameters()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with error handling and metrics"""
        try:
            self.batch_count += 1
            gauge(f"ml.{self.config.name}.batch_count", self.batch_count)
            
            # Vérifier l'input
            if x.shape[0] != self.config.batch_size:
                logger.warning(f"Unexpected batch size: {x.shape[0]}")
            
            # Forward pass
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            output = self._forward_impl(x)
            end_time.record()
            
            # Calculer le temps d'inférence
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)
            histogram(f"ml.{self.config.name}.inference_time_ms", inference_time)
            
            return output
            
        except Exception as e:
            self.error_count += 1
            counter(f"ml.{self.config.name}.errors", 1)
            logger.error(f"Forward pass failed: {e}")
            raise ModelError(f"Forward pass failed: {e}")
    
    @abstractmethod
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation of the forward pass"""
        pass
    
    def save(self, path: Path) -> None:
        """Save model with error handling"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': self.config,
                'batch_count': self.batch_count,
                'error_count': self.error_count
            }, path)
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise ModelError(f"Failed to save model: {e}")
    
    def load(self, path: Path) -> None:
        """Load model with error handling"""
        try:
            if not path.exists():
                raise ModelError(f"Model file not found: {path}")
                
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.batch_count = checkpoint.get('batch_count', 0)
            self.error_count = checkpoint.get('error_count', 0)
            
            # Vérifier la compatibilité de la configuration
            saved_config = checkpoint.get('config')
            if saved_config and saved_config != self.config:
                logger.warning("Loaded model has different configuration")
                
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelError(f"Failed to load model: {e}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get model memory usage"""
        try:
            memory_stats = {
                'allocated': torch.cuda.memory_allocated(self.device) / 1024**2,
                'cached': torch.cuda.memory_reserved(self.device) / 1024**2
            }
            
            for key, value in memory_stats.items():
                gauge(f"ml.{self.config.name}.memory.{key}_mb", value)
                
            return memory_stats
            
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {'allocated': 0.0, 'cached': 0.0}

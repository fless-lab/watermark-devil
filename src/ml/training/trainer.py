"""
Model training utilities.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import logging
from pathlib import Path
from tqdm import tqdm
from metrics import gauge, counter, histogram
from ..config import TrainingConfig
from ..models.base import BaseModel, ModelError

logger = logging.getLogger(__name__)

class TrainingError(Exception):
    """Exception for training errors"""
    pass

class ModelTrainer:
    """Base trainer for all models"""
    
    def __init__(self,
                 model: BaseModel,
                 config: TrainingConfig,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 criterion: Optional[nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 callbacks: Optional[Dict[str, Callable]] = None):
        
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.MSELoss()
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        self.scheduler = scheduler
        self.callbacks = callbacks or {}
        
        self._validate_setup()
        self._init_metrics()
    
    def _validate_setup(self) -> None:
        """Validate training setup"""
        try:
            # Valider la configuration
            self.config.validate()
            
            # Vérifier les dataloaders
            if len(self.train_loader) == 0:
                raise TrainingError("Empty training dataloader")
                
            if self.val_loader is not None and len(self.val_loader) == 0:
                raise TrainingError("Empty validation dataloader")
                
            # Vérifier les callbacks
            for name, callback in self.callbacks.items():
                if not callable(callback):
                    raise TrainingError(f"Callback {name} is not callable")
                    
        except Exception as e:
            logger.error(f"Training setup validation failed: {e}")
            raise TrainingError(f"Training setup validation failed: {e}")
    
    def _init_metrics(self) -> None:
        """Initialize training metrics"""
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def train(self) -> Dict[str, float]:
        """Train the model"""
        try:
            for epoch in range(self.config.epochs):
                # Training
                train_loss = self._train_epoch(epoch)
                self.train_losses.append(train_loss)
                
                # Validation
                if self.val_loader is not None:
                    val_loss = self._validate_epoch(epoch)
                    self.val_losses.append(val_loss)
                    
                    # Early stopping
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                        self._save_checkpoint(f"best_model_epoch_{epoch}.pt")
                    else:
                        self.epochs_without_improvement += 1
                        
                # Learning rate scheduling
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                # Métriques
                self._record_epoch_metrics(epoch, train_loss, 
                                        val_loss if self.val_loader else None)
                
                # Callbacks
                if 'on_epoch_end' in self.callbacks:
                    self.callbacks['on_epoch_end'](epoch, {
                        'train_loss': train_loss,
                        'val_loss': val_loss if self.val_loader else None,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
            
            return {
                'train_loss': self.train_losses[-1],
                'val_loss': self.val_losses[-1] if self.val_losses else None,
                'best_val_loss': self.best_val_loss
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise TrainingError(f"Training failed: {e}")
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                try:
                    # Transférer les données sur le device
                    data = data.to(self.model.device)
                    target = target.to(self.model.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    # Métriques
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    
                    if 'on_batch_end' in self.callbacks:
                        self.callbacks['on_batch_end'](epoch, batch_idx, {
                            'loss': loss.item()
                        })
                        
                except Exception as e:
                    logger.error(f"Training batch {batch_idx} failed: {e}")
                    counter(f"ml.training.batch_errors", 1)
                    continue
        
        return total_loss / len(self.train_loader)
    
    def _validate_epoch(self, epoch: int) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                try:
                    # Transférer les données sur le device
                    data = data.to(self.model.device)
                    target = target.to(self.model.device)
                    
                    # Forward pass
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # Métriques
                    total_loss += loss.item()
                    
                    if 'on_validation_batch_end' in self.callbacks:
                        self.callbacks['on_validation_batch_end'](epoch, batch_idx, {
                            'loss': loss.item()
                        })
                        
                except Exception as e:
                    logger.error(f"Validation batch {batch_idx} failed: {e}")
                    counter(f"ml.validation.batch_errors", 1)
                    continue
        
        return total_loss / len(self.val_loader)
    
    def _save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint"""
        try:
            checkpoint_path = self.config.checkpoint_dir / filename
            torch.save({
                'epoch': len(self.train_losses),
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss
            }, checkpoint_path)
            
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise TrainingError(f"Failed to save checkpoint: {e}")
    
    def _record_epoch_metrics(self,
                            epoch: int,
                            train_loss: float,
                            val_loss: Optional[float]) -> None:
        """Record epoch metrics"""
        gauge(f"ml.training.epoch", epoch)
        gauge(f"ml.training.train_loss", train_loss)
        
        if val_loss is not None:
            gauge(f"ml.training.val_loss", val_loss)
            gauge(f"ml.training.best_val_loss", self.best_val_loss)
            
        gauge(f"ml.training.learning_rate", 
              self.optimizer.param_groups[0]['lr'])
        
        # Métriques mémoire
        if torch.cuda.is_available():
            gauge(f"ml.training.gpu_memory_allocated",
                  torch.cuda.memory_allocated() / 1024**2)
            gauge(f"ml.training.gpu_memory_cached",
                  torch.cuda.memory_reserved() / 1024**2)

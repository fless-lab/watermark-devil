import os
from datetime import datetime, timedelta
import torch
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging

from ..models.logo_detector import LogoDetector
from ..models.text_detector import TextDetector
from ..models.pattern_detector import PatternDetector
from ...db.models import ModelMetrics, ProcessingRequest, WatermarkDetection
from ...db.database import get_db

logger = logging.getLogger(__name__)

class AdaptiveTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_samples = config.get('MIN_SAMPLES_FOR_TRAINING', 1000)
        self.training_interval = timedelta(
            seconds=config.get('TRAINING_INTERVAL', 86400)
        )
        self.backup_enabled = config.get('MODEL_BACKUP_ENABLED', True)
        self.backup_path = config.get('MODEL_BACKUP_PATH', 'backups/')
        
        # Initialize detectors
        self.logo_detector = LogoDetector()
        self.text_detector = TextDetector()
        self.pattern_detector = PatternDetector()
        
        # Ensure backup directory exists
        if self.backup_enabled:
            os.makedirs(self.backup_path, exist_ok=True)

    def should_train(self, db: Session, model_type: str) -> bool:
        """Check if model should be retrained based on new data and time elapsed"""
        # Get latest training metrics
        latest_metrics = (
            db.query(ModelMetrics)
            .filter(ModelMetrics.model_type == model_type)
            .order_by(ModelMetrics.created_at.desc())
            .first()
        )
        
        if not latest_metrics:
            return True
            
        # Check if enough time has passed since last training
        time_elapsed = datetime.utcnow() - latest_metrics.created_at
        if time_elapsed < self.training_interval:
            return False
            
        # Count new samples since last training
        new_samples = (
            db.query(WatermarkDetection)
            .join(ProcessingRequest)
            .filter(
                WatermarkDetection.watermark_type == model_type,
                WatermarkDetection.created_at > latest_metrics.created_at
            )
            .count()
        )
        
        return new_samples >= self.min_samples

    def collect_training_data(
        self, db: Session, model_type: str, last_training: datetime
    ) -> List[Dict[str, Any]]:
        """Collect new training data from successful detections"""
        detections = (
            db.query(WatermarkDetection)
            .join(ProcessingRequest)
            .filter(
                WatermarkDetection.watermark_type == model_type,
                WatermarkDetection.created_at > last_training,
                WatermarkDetection.confidence > 0.8  # Only use high confidence detections
            )
            .all()
        )
        
        training_data = []
        for detection in detections:
            training_data.append({
                'image_path': detection.request.image_path,
                'bbox': detection.bbox,
                'confidence': detection.confidence,
                'text_content': detection.text_content
            })
        
        return training_data

    def train_model(self, model_type: str, training_data: List[Dict[str, Any]]):
        """Train the specified model with new data"""
        if model_type == 'logo':
            model = self.logo_detector
        elif model_type == 'text':
            model = self.text_detector
        elif model_type == 'pattern':
            model = self.pattern_detector
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Perform training
        metrics = model.train(training_data)
        
        if self.backup_enabled:
            self._backup_model(model, model_type)
            
        return metrics

    def _backup_model(self, model, model_type: str):
        """Create a backup of the model"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(
            self.backup_path,
            f"{model_type}_model_{timestamp}.pt"
        )
        
        torch.save(model.state_dict(), backup_file)
        logger.info(f"Created backup of {model_type} model: {backup_file}")

    def update_metrics(self, db: Session, model_type: str, metrics: Dict[str, float]):
        """Store training metrics in database"""
        new_metrics = ModelMetrics(
            model_type=model_type,
            version=datetime.utcnow().strftime('%Y%m%d_%H%M%S'),
            accuracy=metrics.get('accuracy'),
            precision=metrics.get('precision'),
            recall=metrics.get('recall'),
            f1_score=metrics.get('f1_score'),
            training_time=metrics.get('training_time'),
            sample_count=metrics.get('sample_count')
        )
        
        db.add(new_metrics)
        db.commit()
        
        logger.info(f"Updated metrics for {model_type} model: {metrics}")

    async def run_training_cycle(self):
        """Run a complete training cycle for all models"""
        model_types = ['logo', 'text', 'pattern']
        db = next(get_db())
        
        try:
            for model_type in model_types:
                if self.should_train(db, model_type):
                    logger.info(f"Starting training cycle for {model_type} model")
                    
                    # Get last training time
                    last_metrics = (
                        db.query(ModelMetrics)
                        .filter(ModelMetrics.model_type == model_type)
                        .order_by(ModelMetrics.created_at.desc())
                        .first()
                    )
                    last_training = last_metrics.created_at if last_metrics else datetime.min
                    
                    # Collect and prepare training data
                    training_data = self.collect_training_data(
                        db, model_type, last_training
                    )
                    
                    if training_data:
                        # Train model and update metrics
                        metrics = self.train_model(model_type, training_data)
                        self.update_metrics(db, model_type, metrics)
                        
                        logger.info(
                            f"Completed training cycle for {model_type} model with "
                            f"{len(training_data)} samples"
                        )
                    else:
                        logger.warning(
                            f"No valid training data found for {model_type} model"
                        )
                else:
                    logger.info(
                        f"Skipping training cycle for {model_type} model - "
                        "conditions not met"
                    )
        finally:
            db.close()

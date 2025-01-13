import os
from datetime import datetime, timedelta
import torch
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging
import numpy as np

from watermark_evil_core import PyModelTrainer
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
        
        # Initialize Rust trainer
        self.trainer = PyModelTrainer()
        
        # Initialize feature extractors
        self.logo_detector = LogoDetector()
        self.text_detector = TextDetector()
        self.pattern_detector = PatternDetector()
        
        # Ensure backup directory exists
        if self.backup_enabled:
            os.makedirs(self.backup_path, exist_ok=True)

    def should_train(self, db: Session, model_type: str) -> bool:
        """Check if model should be retrained based on new data and time elapsed"""
        if not self.trainer.should_train():
            return False
            
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
    ) -> Dict[str, Any]:
        """Collect and prepare training data for Rust core"""
        detections = (
            db.query(WatermarkDetection)
            .join(ProcessingRequest)
            .filter(
                WatermarkDetection.watermark_type == model_type,
                WatermarkDetection.created_at > last_training,
                WatermarkDetection.confidence > 0.8
            )
            .all()
        )
        
        features = []
        labels = []
        
        for detection in detections:
            # Extract features using appropriate detector
            if model_type == 'logo':
                feat = self.logo_detector.extract_features(detection.request.image_path)
            elif model_type == 'text':
                feat = self.text_detector.extract_features(detection.request.image_path)
            else:
                feat = self.pattern_detector.extract_features(detection.request.image_path)
            
            features.append(feat)
            labels.append(model_type)
        
        return {
            'features': np.array(features, dtype=np.float32).flatten().tolist(),
            'labels': labels
        }

    def train_model(self, model_type: str, training_data: Dict[str, Any]):
        """Train the model using Rust core"""
        try:
            self.trainer.train_models(training_data)
            metrics = self.trainer.get_metrics()
            return metrics
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

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
                    
                    if training_data['features']:
                        # Train model and update metrics
                        metrics = self.train_model(model_type, training_data)
                        self.update_metrics(db, model_type, metrics)
                        
                        logger.info(
                            f"Completed training cycle for {model_type} model with "
                            f"{len(training_data['features'])} samples"
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

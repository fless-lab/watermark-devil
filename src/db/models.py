from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    api_key = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    requests = relationship("ProcessingRequest", back_populates="user")

class ProcessingRequest(Base):
    __tablename__ = "processing_requests"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    image_path = Column(String, nullable=False)
    status = Column(String, nullable=False)  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    processing_time = Column(Float)
    error_message = Column(String)
    
    user = relationship("User", back_populates="requests")
    detections = relationship("WatermarkDetection", back_populates="request")

class WatermarkDetection(Base):
    __tablename__ = "watermark_detections"
    
    id = Column(Integer, primary_key=True)
    request_id = Column(Integer, ForeignKey("processing_requests.id"))
    watermark_type = Column(String, nullable=False)  # logo, text, pattern
    confidence = Column(Float, nullable=False)
    bbox = Column(JSON, nullable=False)  # [x1, y1, x2, y2]
    text_content = Column(String)  # Only for text watermarks
    created_at = Column(DateTime, default=datetime.utcnow)
    
    request = relationship("ProcessingRequest", back_populates="detections")
    removal = relationship("WatermarkRemoval", back_populates="detection", uselist=False)

class WatermarkRemoval(Base):
    __tablename__ = "watermark_removals"
    
    id = Column(Integer, primary_key=True)
    detection_id = Column(Integer, ForeignKey("watermark_detections.id"))
    method_used = Column(String, nullable=False)  # inpainting, diffusion, frequency, hybrid
    quality_score = Column(Float)  # 0-1 score of removal quality
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    detection = relationship("WatermarkDetection", back_populates="removal")

class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True)
    model_type = Column(String, nullable=False)  # detector, remover
    version = Column(String, nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_time = Column(Float)
    sample_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "model_type": self.model_type,
            "version": self.version,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "training_time": self.training_time,
            "sample_count": self.sample_count,
            "created_at": self.created_at.isoformat()
        }

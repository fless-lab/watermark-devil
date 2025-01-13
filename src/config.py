"""
Configuration management for WatermarkEvil.
"""
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class APISettings(BaseSettings):
    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    workers: int = Field(4, env="API_WORKERS")
    timeout: int = Field(30, env="API_TIMEOUT")
    api_key: str = Field(..., env="API_KEY")
    cors_origins: List[str] = Field(default_factory=list)

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

class SecuritySettings(BaseSettings):
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(60, env="RATE_LIMIT_PERIOD")
    max_file_size: int = Field(10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_file_types: List[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"]
    )

class ProcessingSettings(BaseSettings):
    gpu_enabled: bool = Field(True, env="GPU_ENABLED")
    cuda_devices: List[int] = Field(default_factory=lambda: [0])
    max_batch_size: int = Field(16, env="MAX_BATCH_SIZE")
    processing_timeout: int = Field(300, env="PROCESSING_TIMEOUT")
    default_image_size: int = Field(1024, env="DEFAULT_IMAGE_SIZE")
    preserve_exif: bool = Field(False, env="PRESERVE_EXIF")
    output_quality: int = Field(95, env="OUTPUT_QUALITY")

    @validator("cuda_devices", pre=True)
    def parse_cuda_devices(cls, v):
        if isinstance(v, str):
            return [int(x) for x in v.split(",")]
        return v

class DetectionSettings(BaseSettings):
    confidence_threshold: float = Field(0.5, env="DETECTION_CONFIDENCE_THRESHOLD")
    iou_threshold: float = Field(0.3, env="DETECTION_IOU_THRESHOLD")
    max_detections: int = Field(10, env="DETECTION_MAX_DETECTIONS")
    model_path: Path = Field(..., env="DETECTION_MODEL_PATH")
    pattern_db_path: Path = Field(..., env="PATTERN_DB_PATH")

class ReconstructionSettings(BaseSettings):
    quality: str = Field("high", env="RECONSTRUCTION_QUALITY")
    method: str = Field("hybrid", env="RECONSTRUCTION_METHOD")
    model_path: Path = Field(..., env="RECONSTRUCTION_MODEL_PATH")
    temp_files_path: Path = Field(..., env="TEMP_FILES_PATH")

class LearningSettings(BaseSettings):
    enabled: bool = Field(True, env="LEARNING_ENABLED")
    min_samples_for_training: int = Field(1000, env="MIN_SAMPLES_FOR_TRAINING")
    training_interval: int = Field(86400, env="TRAINING_INTERVAL")
    model_backup_enabled: bool = Field(True, env="MODEL_BACKUP_ENABLED")
    model_backup_path: Path = Field(..., env="MODEL_BACKUP_PATH")
    export_metrics: bool = Field(True, env="EXPORT_METRICS")

class DatabaseSettings(BaseSettings):
    type: str = Field("postgresql", env="DB_TYPE")
    host: str = Field(..., env="DB_HOST")
    port: int = Field(5432, env="DB_PORT")
    name: str = Field(..., env="DB_NAME")
    user: str = Field(..., env="DB_USER")
    password: str = Field(..., env="DB_PASSWORD")
    ssl_mode: str = Field("prefer", env="DB_SSL_MODE")

    @property
    def url(self) -> str:
        return f"{self.type}://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

class RedisSettings(BaseSettings):
    enabled: bool = Field(True, env="REDIS_ENABLED")
    host: str = Field("localhost", env="REDIS_HOST")
    port: int = Field(6379, env="REDIS_PORT")
    db: int = Field(0, env="REDIS_DB")
    password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    cache_ttl: int = Field(3600, env="CACHE_TTL")

class MonitoringSettings(BaseSettings):
    prometheus_enabled: bool = Field(True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    grafana_enabled: bool = Field(True, env="GRAFANA_ENABLED")
    grafana_port: int = Field(3000, env="GRAFANA_PORT")
    sentry_enabled: bool = Field(True, env="SENTRY_ENABLED")
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")
    healthcheck_enabled: bool = Field(True, env="HEALTHCHECK_ENABLED")
    healthcheck_interval: int = Field(60, env="HEALTHCHECK_INTERVAL")

class StorageSettings(BaseSettings):
    type: str = Field("local", env="STORAGE_TYPE")
    local_path: Path = Field(..., env="LOCAL_STORAGE_PATH")
    s3_bucket: Optional[str] = Field(None, env="S3_BUCKET")
    s3_region: Optional[str] = Field(None, env="S3_REGION")
    s3_access_key: Optional[str] = Field(None, env="S3_ACCESS_KEY")
    s3_secret_key: Optional[str] = Field(None, env="S3_SECRET_KEY")

class NotificationSettings(BaseSettings):
    smtp_enabled: bool = Field(False, env="SMTP_ENABLED")
    smtp_host: Optional[str] = Field(None, env="SMTP_HOST")
    smtp_port: Optional[int] = Field(None, env="SMTP_PORT")
    smtp_user: Optional[str] = Field(None, env="SMTP_USER")
    smtp_password: Optional[str] = Field(None, env="SMTP_PASSWORD")
    notification_email: Optional[str] = Field(None, env="NOTIFICATION_EMAIL")

class Settings(BaseSettings):
    mode: str = Field("development", env="WATERMARK_EVIL_MODE")
    debug: bool = Field(True, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    api: APISettings = APISettings()
    security: SecuritySettings = SecuritySettings()
    processing: ProcessingSettings = ProcessingSettings()
    detection: DetectionSettings = DetectionSettings()
    reconstruction: ReconstructionSettings = ReconstructionSettings()
    learning: LearningSettings = LearningSettings()
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    storage: StorageSettings = StorageSettings()
    notification: NotificationSettings = NotificationSettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings

"""
Input validation utilities for the API.
"""
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import magic
import logging
from PIL import Image
import io
from pydantic import BaseModel, validator, Field
from fastapi import HTTPException, UploadFile
from metrics import counter, gauge

logger = logging.getLogger(__name__)

# Constantes
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIME_TYPES = {
    'image/jpeg',
    'image/png',
    'image/webp',
    'image/tiff'
}
MAX_IMAGE_DIMENSION = 4096

class ValidationError(Exception):
    """Exception for validation errors"""
    pass

class ImageMetadata(BaseModel):
    """Image metadata model"""
    width: int = Field(..., gt=0, le=MAX_IMAGE_DIMENSION)
    height: int = Field(..., gt=0, le=MAX_IMAGE_DIMENSION)
    format: str
    size_bytes: int = Field(..., gt=0, le=MAX_FILE_SIZE)
    mime_type: str
    
    @validator('mime_type')
    def validate_mime_type(cls, v):
        if v not in ALLOWED_MIME_TYPES:
            raise ValueError(f"Unsupported mime type: {v}")
        return v

async def validate_image_file(file: UploadFile) -> ImageMetadata:
    """Validate uploaded image file"""
    try:
        # Lire le contenu
        content = await file.read()
        size = len(content)
        
        # Vérifier la taille
        if size > MAX_FILE_SIZE:
            counter("api.validation.errors", tags={"type": "file_too_large"})
            raise ValidationError(
                f"File too large: {size} bytes (max {MAX_FILE_SIZE} bytes)")
        
        # Vérifier le type MIME
        mime = magic.from_buffer(content, mime=True)
        if mime not in ALLOWED_MIME_TYPES:
            counter("api.validation.errors", tags={"type": "invalid_mime_type"})
            raise ValidationError(f"Invalid mime type: {mime}")
        
        # Vérifier l'image
        try:
            img = Image.open(io.BytesIO(content))
            width, height = img.size
            
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                counter("api.validation.errors", 
                       tags={"type": "image_too_large"})
                raise ValidationError(
                    f"Image dimensions too large: {width}x{height}")
            
            # Métriques
            gauge("api.image.width", width)
            gauge("api.image.height", height)
            gauge("api.image.size_bytes", size)
            
            return ImageMetadata(
                width=width,
                height=height,
                format=img.format.lower(),
                size_bytes=size,
                mime_type=mime
            )
            
        except Exception as e:
            counter("api.validation.errors", tags={"type": "invalid_image"})
            raise ValidationError(f"Invalid image: {str(e)}")
            
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        counter("api.validation.errors", tags={"type": "internal_error"})
        raise HTTPException(status_code=500, 
                          detail="Internal validation error")
    
    finally:
        # Reset file pointer
        await file.seek(0)

class ProcessingOptions(BaseModel):
    """Model for processing options"""
    detect_logos: bool = Field(default=True)
    detect_text: bool = Field(default=True)
    detect_patterns: bool = Field(default=True)
    reconstruction_quality: float = Field(
        default=0.8, ge=0.0, le=1.0)
    max_detections: int = Field(
        default=10, ge=1, le=100)
    
    @validator('reconstruction_quality')
    def validate_quality(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Quality must be between 0 and 1")
        return v

class ProcessingResult(BaseModel):
    """Model for processing results"""
    success: bool
    error: Optional[str] = None
    detections: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time: float
    image_metadata: ImageMetadata
    
    @validator('processing_time')
    def validate_time(cls, v):
        if v < 0:
            raise ValueError("Processing time cannot be negative")
        return v

def validate_output_path(path: Path) -> None:
    """Validate output file path"""
    try:
        # Vérifier si le répertoire parent existe
        if not path.parent.exists():
            raise ValidationError(
                f"Output directory does not exist: {path.parent}")
        
        # Vérifier les permissions
        if not os.access(path.parent, os.W_OK):
            raise ValidationError(
                f"No write permission for directory: {path.parent}")
        
        # Vérifier l'extension
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
            raise ValidationError(
                f"Invalid output format: {path.suffix}")
            
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Output path validation failed: {e}")
        raise HTTPException(status_code=500, 
                          detail="Internal validation error")

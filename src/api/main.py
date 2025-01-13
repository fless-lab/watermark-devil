"""
Main FastAPI application for the watermark removal service.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import uvicorn
from typing import List, Optional
import numpy as np
from PIL import Image
import io

from .models import (
    DetectionResponse,
    ReconstructionResponse,
    LearningResponse,
    ProcessingOptions,
)
from .processing import WatermarkProcessor

app = FastAPI(
    title="WatermarkEvil API",
    description="Advanced Watermark Detection and Removal API",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor
processor = WatermarkProcessor()

@app.post("/detect", response_model=DetectionResponse)
async def detect_watermark(
    file: UploadFile = File(...),
    options: Optional[ProcessingOptions] = None
):
    """
    Detect watermarks in an image or video file.
    """
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Process detection
        result = await processor.detect(image_array, options)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reconstruct", response_model=ReconstructionResponse)
async def reconstruct_image(
    file: UploadFile = File(...),
    detection_id: str,
    options: Optional[ProcessingOptions] = None
):
    """
    Remove detected watermarks and reconstruct the image.
    """
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Process reconstruction
        result = await processor.reconstruct(image_array, detection_id, options)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/process", response_model=ReconstructionResponse)
async def process_image(
    file: UploadFile = File(...),
    options: Optional[ProcessingOptions] = None
):
    """
    Complete pipeline: detect watermarks and reconstruct the image.
    """
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Process complete pipeline
        result = await processor.process_complete(image_array, options)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/learn", response_model=LearningResponse)
async def update_learning(
    file: UploadFile = File(...),
    detection_id: str,
    reconstruction_id: str,
    feedback: Optional[dict] = None
):
    """
    Update the learning system with new results and feedback.
    """
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Update learning system
        result = await processor.update_learning(
            image_array,
            detection_id,
            reconstruction_id,
            feedback
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status")
async def get_system_status():
    """
    Get the current status of the system, including model versions and statistics.
    """
    try:
        status = await processor.get_status()
        return JSONResponse(content=status)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

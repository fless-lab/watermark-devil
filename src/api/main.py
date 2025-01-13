"""
Main FastAPI application for the watermark removal service.
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .services import WatermarkService
from .models import ProcessingResponse

app = FastAPI(
    title="WatermarkEvil API",
    description="Advanced Watermark Removal API",
    version="0.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
watermark_service = WatermarkService()

@app.post("/process", response_model=ProcessingResponse)
async def process_image(file: UploadFile = File(...)):
    """
    Process an image to remove watermarks.
    """
    try:
        result = await watermark_service.process_image(file)
        return ProcessingResponse(
            success=True,
            message="Processing successful",
            data=result
        )
    except Exception as e:
        return ProcessingResponse(
            success=False,
            message=str(e),
            data=None
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

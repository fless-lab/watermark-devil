import numpy as np
import torch
import cv2
from PIL import Image
import io
import base64
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any
import asyncio
import psutil

from .models import (
    DetectionResponse,
    ReconstructionResponse,
    LearningResponse,
    ProcessingOptions,
    DetectionResult,
    ReconstructionResult,
    LearningResult,
    WatermarkType,
    ReconstructionMethod,
    BoundingBox,
)

class WatermarkProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_engine = None
        self.reconstruction_engine = None
        self.learning_system = None
        self.processed_count = 0
        self.success_count = 0
        self.total_processing_time = 0.0
        self.initialize_engines()

    def initialize_engines(self):
        """
        Initialize the detection and reconstruction engines.
        """
        # Import here to avoid circular imports
        from ..engine.detection import DetectionEngine
        from ..engine.reconstruction import ReconstructionEngine
        from ..engine.learning import AdaptiveLearning

        self.detection_engine = DetectionEngine()
        self.reconstruction_engine = ReconstructionEngine()
        self.learning_system = AdaptiveLearning()

    async def detect(
        self,
        image: np.ndarray,
        options: Optional[ProcessingOptions] = None
    ) -> DetectionResponse:
        """
        Detect watermarks in the image.
        """
        start_time = datetime.utcnow()
        
        try:
            # Apply detection
            detections = await self.detection_engine.detect(
                image,
                min_confidence=options.min_confidence if options else 0.5,
                detect_multiple=options.detect_multiple if options else True,
            )

            # Convert detections to response format
            detection_results = []
            for det in detections:
                # Create mask if available
                mask_b64 = None
                if det.mask is not None:
                    mask_bytes = cv2.imencode('.png', det.mask)[1].tobytes()
                    mask_b64 = base64.b64encode(mask_bytes).decode()

                detection_results.append(DetectionResult(
                    id=str(uuid.uuid4()),
                    confidence=det.confidence,
                    watermark_type=WatermarkType(det.watermark_type.lower()),
                    bbox=BoundingBox(
                        x=det.bbox.x,
                        y=det.bbox.y,
                        width=det.bbox.width,
                        height=det.bbox.height
                    ),
                    mask=mask_b64,
                    metadata=det.metadata
                ))

            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return DetectionResponse(
                success=True,
                message="Detection successful",
                detections=detection_results,
                processing_time=processing_time,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            return DetectionResponse(
                success=False,
                message=str(e),
                detections=[],
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                timestamp=datetime.utcnow()
            )

    async def reconstruct(
        self,
        image: np.ndarray,
        detection_id: str,
        options: Optional[ProcessingOptions] = None
    ) -> ReconstructionResponse:
        """
        Reconstruct the image by removing detected watermarks.
        """
        start_time = datetime.utcnow()
        
        try:
            # Get detection result
            detection = await self.get_detection(detection_id)
            if not detection:
                raise ValueError(f"Detection {detection_id} not found")

            # Apply reconstruction
            reconstructed, method = await self.reconstruction_engine.reconstruct(
                image,
                detection,
                preferred_method=options.preferred_method if options else None,
                use_gpu=options.use_gpu if options else True,
            )

            # Convert result to base64
            success, buffer = cv2.imencode('.png', reconstructed)
            if not success:
                raise ValueError("Failed to encode reconstructed image")
            
            image_b64 = base64.b64encode(buffer).decode()

            # Create reconstruction result
            reconstruction_result = ReconstructionResult(
                id=str(uuid.uuid4()),
                success=True,
                quality_score=0.95,  # TODO: Implement quality assessment
                method_used=ReconstructionMethod(method.lower()),
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                image_data=image_b64,
                metadata={"detection_id": detection_id}
            )

            return ReconstructionResponse(
                success=True,
                message="Reconstruction successful",
                reconstructions=[reconstruction_result],
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            return ReconstructionResponse(
                success=False,
                message=str(e),
                reconstructions=[],
                timestamp=datetime.utcnow()
            )

    async def process_complete(
        self,
        image: np.ndarray,
        options: Optional[ProcessingOptions] = None
    ) -> ReconstructionResponse:
        """
        Complete pipeline: detect and reconstruct.
        """
        try:
            # First detect
            detection_response = await self.detect(image, options)
            if not detection_response.success:
                return ReconstructionResponse(
                    success=False,
                    message=f"Detection failed: {detection_response.message}",
                    reconstructions=[],
                    timestamp=datetime.utcnow()
                )

            # Then reconstruct for each detection
            reconstructions = []
            for detection in detection_response.detections:
                reconstruction_response = await self.reconstruct(
                    image,
                    detection.id,
                    options
                )
                if reconstruction_response.success:
                    reconstructions.extend(reconstruction_response.reconstructions)

            # Create final response
            success = len(reconstructions) > 0
            message = "Processing complete" if success else "No successful reconstructions"
            
            # Update statistics
            self.processed_count += 1
            if success:
                self.success_count += 1

            return ReconstructionResponse(
                success=success,
                message=message,
                reconstructions=reconstructions,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            return ReconstructionResponse(
                success=False,
                message=str(e),
                reconstructions=[],
                timestamp=datetime.utcnow()
            )

    async def update_learning(
        self,
        image: np.ndarray,
        detection_id: str,
        reconstruction_id: str,
        feedback: Optional[Dict[str, Any]] = None
    ) -> LearningResponse:
        """
        Update the learning system with results and feedback.
        """
        try:
            # Get detection and reconstruction results
            detection = await self.get_detection(detection_id)
            reconstruction = await self.get_reconstruction(reconstruction_id)
            
            if not detection or not reconstruction:
                raise ValueError("Detection or reconstruction not found")

            # Update learning system
            learning_result = await self.learning_system.analyze_case(
                image,
                detection,
                reconstruction,
                feedback
            )

            # Convert to response format
            result = LearningResult(
                pattern_type=WatermarkType(learning_result.pattern_type.lower()),
                success_rate=learning_result.success_rate,
                improvements=learning_result.improvements,
                training_required=learning_result.training_required
            )

            return LearningResponse(
                success=True,
                message="Learning system updated successfully",
                result=result,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            return LearningResponse(
                success=False,
                message=str(e),
                result=None,
                timestamp=datetime.utcnow()
            )

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        """
        memory = psutil.Process().memory_info()
        
        return {
            "status": "operational",
            "gpu_available": torch.cuda.is_available(),
            "model_versions": {
                "detection": "1.0.0",
                "reconstruction": "1.0.0",
                "learning": "1.0.0"
            },
            "processed_images": self.processed_count,
            "success_rate": self.success_count / max(1, self.processed_count),
            "average_processing_time": self.total_processing_time / max(1, self.processed_count),
            "last_training": datetime.utcnow(),  # TODO: Get from learning system
            "memory_usage": {
                "rss": memory.rss / 1024 / 1024,  # MB
                "vms": memory.vms / 1024 / 1024,  # MB
            }
        }

    async def get_detection(self, detection_id: str) -> Optional[DetectionResult]:
        """
        Retrieve a detection result by ID.
        """
        # TODO: Implement caching/storage of results
        return None

    async def get_reconstruction(self, reconstruction_id: str) -> Optional[ReconstructionResult]:
        """
        Retrieve a reconstruction result by ID.
        """
        # TODO: Implement caching/storage of results
        return None

"""
Base detector model implementation.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import logging
from metrics import gauge, histogram
from .base import BaseModel, ModelError
from ..config import DetectorConfig

logger = logging.getLogger(__name__)

class BaseDetector(BaseModel):
    """Base class for all detector models"""
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.config: DetectorConfig = config  # Type hint pour l'IDE
        
    def _validate_architecture(self) -> None:
        """Validate detector architecture"""
        # Validation spécifique aux détecteurs à implémenter dans les sous-classes
        pass
        
    def process_detections(self, 
                         raw_outputs: torch.Tensor,
                         confidence_threshold: Optional[float] = None,
                         iou_threshold: Optional[float] = None) -> List[Tuple[torch.Tensor, float]]:
        """Process raw detector outputs into final detections"""
        try:
            # Utiliser les seuils de la config si non spécifiés
            conf_thresh = confidence_threshold or self.config.confidence_threshold
            iou_thresh = iou_threshold or self.config.iou_threshold
            
            # Filtrer par confiance
            confidences = raw_outputs[:, 4]
            keep = confidences > conf_thresh
            filtered_outputs = raw_outputs[keep]
            
            if len(filtered_outputs) == 0:
                return []
            
            # Non-maximum suppression
            keep_indices = self._nms(
                filtered_outputs[:, :4],
                filtered_outputs[:, 4],
                iou_thresh
            )
            
            # Préparer les résultats
            final_detections = []
            for idx in keep_indices:
                bbox = filtered_outputs[idx, :4]
                conf = filtered_outputs[idx, 4].item()
                final_detections.append((bbox, conf))
                
            # Métriques
            histogram(f"ml.{self.config.name}.num_detections", len(final_detections))
            if len(final_detections) > 0:
                confidences = [conf for _, conf in final_detections]
                histogram(f"ml.{self.config.name}.detection_confidence", 
                         sum(confidences) / len(confidences))
            
            return final_detections[:self.config.max_detections]
            
        except Exception as e:
            logger.error(f"Detection processing failed: {e}")
            raise ModelError(f"Detection processing failed: {e}")
    
    def _nms(self, 
             boxes: torch.Tensor,
             scores: torch.Tensor,
             iou_threshold: float) -> torch.Tensor:
        """Non-maximum suppression"""
        try:
            # Trier par score
            _, order = scores.sort(0, descending=True)
            keep = []
            
            while order.numel() > 0:
                if len(keep) >= self.config.max_detections:
                    break
                    
                # Garder le meilleur score
                if order.numel() == 1:
                    keep.append(order.item())
                    break
                    
                i = order[0]
                keep.append(i.item())
                
                # Calculer IoU avec les boîtes restantes
                ious = self._compute_ious(boxes[i], boxes[order[1:]])
                
                # Garder les boîtes avec IoU inférieur au seuil
                mask = ious <= iou_threshold
                order = order[1:][mask]
            
            return torch.tensor(keep, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"NMS failed: {e}")
            raise ModelError(f"NMS failed: {e}")
    
    def _compute_ious(self, box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU between a box and an array of boxes"""
        try:
            # Aire de la première boîte
            area1 = (box[2] - box[0]) * (box[3] - box[1])
            
            # Aires des autres boîtes
            area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            
            # Intersection
            xx1 = torch.max(box[0], boxes[:, 0])
            yy1 = torch.max(box[1], boxes[:, 1])
            xx2 = torch.min(box[2], boxes[:, 2])
            yy2 = torch.min(box[3], boxes[:, 3])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            
            intersection = w * h
            union = area1 + area2 - intersection
            
            return intersection / union
            
        except Exception as e:
            logger.error(f"IoU computation failed: {e}")
            raise ModelError(f"IoU computation failed: {e}")
    
    @property
    def num_parameters(self) -> int:
        """Get number of model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Estimate FLOPs for a given input shape"""
        try:
            from thop import profile
            input = torch.randn(1, *input_shape).to(self.device)
            flops, _ = profile(self, inputs=(input,))
            return flops
        except Exception as e:
            logger.warning(f"Failed to estimate FLOPs: {e}")
            return 0

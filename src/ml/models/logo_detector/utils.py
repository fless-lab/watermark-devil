"""
Utility functions for logo detection
"""
import numpy as np
import torch
import cv2
from typing import List, Tuple, Dict
from torchvision.ops import box_iou, soft_nms
import torchvision.transforms.functional as F

def apply_soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float = 0.5,
    score_threshold: float = 0.001
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applique Soft-NMS avec un noyau gaussien
    """
    keep_scores, keep_indices = soft_nms(
        boxes,
        scores,
        sigma=sigma,
        score_threshold=score_threshold
    )
    return boxes[keep_indices], keep_scores

def test_time_augmentation(
    model: torch.nn.Module,
    image: torch.Tensor,
    scales: List[float] = [0.8, 1.0, 1.2],
    flips: bool = True
) -> List[Tuple[List[float], float]]:
    """
    Applique Test Time Augmentation pour améliorer les prédictions
    """
    predictions = []
    
    for scale in scales:
        # Redimensionnement
        h, w = image.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled_img = F.resize(image, (new_h, new_w))
        
        # Prédiction sur l'image redimensionnée
        pred = model(scaled_img)
        predictions.extend(rescale_predictions(pred, 1/scale))
        
        if flips:
            # Prédiction sur l'image retournée horizontalement
            flipped_img = torch.flip(scaled_img, [-1])
            pred_flip = model(flipped_img)
            predictions.extend(flip_predictions(pred_flip, w, 1/scale))
    
    return combine_predictions(predictions)

def rescale_predictions(
    predictions: List[Tuple[List[float], float]],
    scale: float
) -> List[Tuple[List[float], float]]:
    """
    Redimensionne les prédictions à l'échelle originale
    """
    scaled_preds = []
    for box, score in predictions:
        scaled_box = [coord * scale for coord in box]
        scaled_preds.append((scaled_box, score))
    return scaled_preds

def flip_predictions(
    predictions: List[Tuple[List[float], float]],
    width: int,
    scale: float = 1.0
) -> List[Tuple[List[float], float]]:
    """
    Inverse les prédictions pour les images retournées
    """
    flipped_preds = []
    for box, score in predictions:
        flipped_box = [
            width - box[2] * scale,  # x1
            box[1] * scale,          # y1
            width - box[0] * scale,  # x2
            box[3] * scale,          # y2
        ]
        flipped_preds.append((flipped_box, score))
    return flipped_preds

def combine_predictions(
    predictions: List[Tuple[List[float], float]],
    iou_threshold: float = 0.5
) -> List[Tuple[List[float], float]]:
    """
    Combine les prédictions de différentes augmentations
    en utilisant weighted box fusion
    """
    if not predictions:
        return []
        
    boxes = torch.tensor([p[0] for p in predictions])
    scores = torch.tensor([p[1] for p in predictions])
    
    # Calculer IoU entre toutes les boîtes
    iou_matrix = box_iou(boxes, boxes)
    
    # Weighted Box Fusion
    final_boxes = []
    final_scores = []
    used = set()
    
    for i in range(len(boxes)):
        if i in used:
            continue
            
        # Trouver toutes les boîtes similaires
        matches = torch.where(iou_matrix[i] > iou_threshold)[0]
        used.update(matches.tolist())
        
        # Calculer la moyenne pondérée
        matched_boxes = boxes[matches]
        matched_scores = scores[matches]
        weights = matched_scores / matched_scores.sum()
        
        weighted_box = (matched_boxes * weights.unsqueeze(1)).sum(0)
        weighted_score = matched_scores.mean()
        
        final_boxes.append(weighted_box.tolist())
        final_scores.append(weighted_score.item())
    
    return list(zip(final_boxes, final_scores))

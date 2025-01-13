"""
Advanced logo detector using YOLOv8 and advanced post-processing
"""
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from ultralytics import YOLO
import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.ops import nms

class LogoDetector:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize YOLOv8 logo detector with advanced features
        Args:
            model_path: Path to custom trained model, if None uses pretrained YOLOv8x
        """
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Utiliser le modèle le plus avancé (YOLOv8x)
            self.model = YOLO('yolov8x.pt')
            
        # Configuration avancée
        self.conf_threshold = 0.35  # Plus strict que la valeur par défaut
        self.iou_threshold = 0.45
        self.max_det = 50  # Maximum détections par image
        
        # Prétraitement avancé
        self.transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            T.ColorJitter(brightness=0.1, contrast=0.1, 
                         saturation=0.1, hue=0.1),
        ])
        
        # Activer le mode d'inférence et CUDA si disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Prétraitement avancé de l'image
        """
        # Conversion en RGB si nécessaire
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Conversion en tensor
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)
        
        # Appliquer les transformations avancées
        image = self.transform(image)
        
        # Ajouter batch dimension
        image = image.unsqueeze(0)
        return image.to(self.device)

    def detect(self, image: torch.Tensor) -> List[Tuple[List[int], float]]:
        """
        Détection avec post-traitement avancé
        """
        with torch.no_grad():
            # Inférence avec le modèle
            results = self.model(image, verbose=False)
            
            # Post-traitement avancé
            boxes = []
            for result in results:
                # Extraire les boîtes et scores
                bboxes = result.boxes.xyxy.cpu()
                scores = result.boxes.conf.cpu()
                
                # Appliquer NMS avec des paramètres optimisés
                keep_idx = nms(bboxes, scores, self.iou_threshold)
                
                # Filtrer et formater les résultats
                for idx in keep_idx[:self.max_det]:
                    if scores[idx] > self.conf_threshold:
                        box = bboxes[idx].tolist()
                        boxes.append((box, scores[idx].item()))
            
            return boxes

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retourne les métadonnées du modèle
        """
        return {
            "model_type": "YOLOv8x",
            "device": str(self.device),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_det,
        }

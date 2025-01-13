"""
Advanced text detector using PaddleOCR with custom enhancements
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import cv2
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
from PIL import Image

class TextDetector:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-model text detector ensemble
        """
        # PaddleOCR - Excellent pour le texte asiatique et général
        self.paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=torch.cuda.is_available(),
            show_log=False
        )
        
        # TrOCR - Spécialisé pour le texte manuscrit et stylisé
        self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
        
        # EasyOCR - Bon pour le texte naturel et les langues multiples
        self.easy_ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # Configuration
        self.conf_threshold = config.get('conf_threshold', 0.5) if config else 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trocr_model.to(self.device)
        
        # Cache pour les résultats
        self._last_results = None

    def preprocess(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Prétraitement avancé pour les différents modèles
        """
        # Conversion en RGB si nécessaire
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Amélioration de l'image
        image_enhanced = self._enhance_image(image_rgb)
        
        # Préparer les différents formats
        pil_image = Image.fromarray(image_enhanced)
        trocr_inputs = self.trocr_processor(pil_image, return_tensors="pt").to(self.device)
        
        return {
            "paddle": image_enhanced,
            "trocr": trocr_inputs,
            "easyocr": image_enhanced,
            "original_shape": image.shape[:2]
        }

    def detect(self, inputs: Dict[str, Any]) -> List[Tuple[List[int], float, str]]:
        """
        Détection multi-modèle avec fusion des résultats
        """
        # PaddleOCR detection
        paddle_results = self.paddle_ocr.ocr(inputs["paddle"], cls=True)
        paddle_boxes = self._process_paddle_results(paddle_results)
        
        # TrOCR detection
        trocr_results = self._detect_with_trocr(inputs["trocr"])
        
        # EasyOCR detection
        easy_results = self.easy_ocr.readtext(inputs["easyocr"])
        easy_boxes = self._process_easyocr_results(easy_results)
        
        # Fusionner et filtrer les résultats
        all_detections = []
        all_detections.extend(paddle_boxes)
        all_detections.extend(trocr_results)
        all_detections.extend(easy_boxes)
        
        # Filtrer les doublons et fusionner les résultats proches
        final_detections = self._merge_overlapping_detections(all_detections)
        
        # Sauvegarder pour post-traitement
        self._last_results = final_detections
        
        return final_detections

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Amélioration de la qualité de l'image pour une meilleure détection
        """
        # Conversion en LAB pour meilleure manipulation des couleurs
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE sur le canal L
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Reconstruction de l'image
        enhanced_lab = cv2.merge([cl, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Débruitage
        enhanced_rgb = cv2.fastNlMeansDenoisingColored(enhanced_rgb)
        
        return enhanced_rgb

    def _process_paddle_results(self, results: List) -> List[Tuple[List[int], float, str]]:
        """
        Traitement des résultats PaddleOCR
        """
        boxes = []
        for result in results:
            for line in result:
                box = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                if confidence > self.conf_threshold:
                    # Convertir en format [x1, y1, x2, y2]
                    x1 = min(point[0] for point in box)
                    y1 = min(point[1] for point in box)
                    x2 = max(point[0] for point in box)
                    y2 = max(point[1] for point in box)
                    
                    boxes.append(([int(x1), int(y1), int(x2), int(y2)], confidence, text))
        
        return boxes

    def _detect_with_trocr(self, inputs: Dict) -> List[Tuple[List[int], float, str]]:
        """
        Détection avec TrOCR
        """
        with torch.no_grad():
            outputs = self.trocr_model.generate(**inputs)
        
        # Décodage du texte
        predicted_text = self.trocr_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # TrOCR ne donne pas de boîtes, on utilise toute l'image
        if predicted_text.strip():
            return [([0, 0, inputs["pixel_values"].shape[2], inputs["pixel_values"].shape[1]], 
                    0.9, predicted_text)]
        return []

    def _process_easyocr_results(self, results: List) -> List[Tuple[List[int], float, str]]:
        """
        Traitement des résultats EasyOCR
        """
        boxes = []
        for (bbox, text, confidence) in results:
            if confidence > self.conf_threshold:
                x1, y1 = map(int, bbox[0])
                x2, y2 = map(int, bbox[2])
                boxes.append(([x1, y1, x2, y2], confidence, text))
        return boxes

    def _merge_overlapping_detections(
        self,
        detections: List[Tuple[List[int], float, str]],
        iou_threshold: float = 0.5
    ) -> List[Tuple[List[int], float, str]]:
        """
        Fusion des détections qui se chevauchent
        """
        if not detections:
            return []
            
        # Trier par confiance
        detections.sort(key=lambda x: x[1], reverse=True)
        
        final_detections = []
        used = set()
        
        for i, (box1, conf1, text1) in enumerate(detections):
            if i in used:
                continue
                
            current_box = box1
            current_conf = conf1
            current_text = text1
            overlaps = []
            
            # Chercher les chevauchements
            for j, (box2, conf2, text2) in enumerate(detections):
                if i != j and j not in used:
                    iou = self._calculate_iou(box1, box2)
                    if iou > iou_threshold:
                        used.add(j)
                        overlaps.append((box2, conf2, text2))
            
            # Fusionner les détections qui se chevauchent
            if overlaps:
                merged_box = self._merge_boxes([current_box] + [b[0] for b in overlaps])
                merged_conf = max([current_conf] + [b[1] for b in overlaps])
                merged_text = self._merge_texts([current_text] + [b[2] for b in overlaps])
                final_detections.append((merged_box, merged_conf, merged_text))
            else:
                final_detections.append((current_box, current_conf, current_text))
        
        return final_detections

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calcul de l'IoU entre deux boîtes
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / float(area1 + area2 - intersection)

    def _merge_boxes(self, boxes: List[List[int]]) -> List[int]:
        """
        Fusion de plusieurs boîtes en une seule
        """
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        return [x1, y1, x2, y2]

    def _merge_texts(self, texts: List[str]) -> str:
        """
        Fusion intelligente des textes détectés
        """
        # Prendre le texte le plus long comme base
        base_text = max(texts, key=len)
        return base_text

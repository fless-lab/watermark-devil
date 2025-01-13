"""
Advanced pattern detector using frequency analysis and deep learning
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from scipy import fftpack
from scipy import signal
from sklearn.cluster import DBSCAN

class PatternDetector:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pattern detector with frequency and CNN-based analysis
        """
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paramètres de détection
        self.min_pattern_size = self.config.get('min_pattern_size', 20)
        self.max_pattern_size = self.config.get('max_pattern_size', 200)
        self.freq_threshold = self.config.get('freq_threshold', 0.1)
        self.pattern_threshold = self.config.get('pattern_threshold', 0.4)
        
        # CNN pour la validation des motifs
        self.pattern_cnn = self._build_pattern_cnn()
        self.pattern_cnn.to(self.device)

    def preprocess(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Prétraitement avancé pour l'analyse de motifs
        """
        # Conversion en niveaux de gris
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Analyse multi-échelle
        scales = [0.5, 1.0, 2.0]
        pyramids = []
        for scale in scales:
            if scale != 1.0:
                size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
                scaled = cv2.resize(enhanced, size)
            else:
                scaled = enhanced
            pyramids.append(scaled)
        
        return {
            "original": image,
            "gray": gray,
            "enhanced": enhanced,
            "pyramids": pyramids,
            "original_shape": image.shape[:2]
        }

    def detect(self, inputs: Dict[str, Any]) -> List[Tuple[List[int], float]]:
        """
        Détection avancée des motifs répétitifs
        """
        detections = []
        
        # 1. Analyse fréquentielle
        freq_patterns = self._detect_frequency_patterns(inputs["enhanced"])
        
        # 2. Analyse par auto-corrélation
        autocorr_patterns = self._detect_autocorrelation_patterns(inputs["enhanced"])
        
        # 3. Analyse multi-échelle
        for pyramid in inputs["pyramids"]:
            scale_patterns = self._detect_scale_patterns(pyramid)
            detections.extend(scale_patterns)
        
        # 4. Fusion des détections
        detections.extend(freq_patterns)
        detections.extend(autocorr_patterns)
        
        # 5. Validation par CNN
        validated_detections = self._validate_patterns(inputs["original"], detections)
        
        # 6. Post-traitement et fusion des boîtes
        final_detections = self._post_process_detections(validated_detections)
        
        return final_detections

    def _build_pattern_cnn(self) -> nn.Module:
        """
        CNN pour la validation des motifs détectés
        """
        class PatternCNN(nn.Module):
            def __init__(self):
                super().__init__()
                # Feature extraction
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                
                # Pattern analysis
                self.pattern_conv = nn.Conv2d(128, 128, 3, padding=1)
                self.pattern_pool = nn.AdaptiveAvgPool2d(1)
                
                # Classification
                self.fc = nn.Linear(128, 1)
                
            def forward(self, x):
                # Feature extraction
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv3(x))
                
                # Pattern analysis
                x = F.relu(self.pattern_conv(x))
                x = self.pattern_pool(x)
                x = x.view(x.size(0), -1)
                
                # Classification
                x = torch.sigmoid(self.fc(x))
                return x
                
        return PatternCNN()

    def _detect_frequency_patterns(self, image: np.ndarray) -> List[Tuple[List[int], float]]:
        """
        Détection des motifs par analyse de Fourier
        """
        # FFT 2D
        f_transform = fftpack.fft2(image)
        f_shift = fftpack.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Trouver les pics de fréquence
        peaks = signal.find_peaks_cwt(magnitude.ravel(), np.arange(1, 10))
        
        # Analyser les pics pour trouver les motifs
        patterns = []
        for peak in peaks:
            if magnitude.ravel()[peak] > self.freq_threshold:
                # Convertir la position du pic en coordonnées spatiales
                y, x = np.unravel_index(peak, magnitude.shape)
                period_x = image.shape[1] / (x + 1)
                period_y = image.shape[0] / (y + 1)
                
                if self.min_pattern_size < period_x < self.max_pattern_size and \
                   self.min_pattern_size < period_y < self.max_pattern_size:
                    patterns.append(([int(x), int(y), 
                                   int(x + period_x), int(y + period_y)],
                                   float(magnitude.ravel()[peak])))
        
        return patterns

    def _detect_autocorrelation_patterns(self, image: np.ndarray) -> List[Tuple[List[int], float]]:
        """
        Détection des motifs par auto-corrélation
        """
        # Calculer l'auto-corrélation 2D
        autocorr = signal.correlate2d(image, image, mode='full')
        
        # Normaliser
        autocorr = (autocorr - autocorr.min()) / (autocorr.max() - autocorr.min())
        
        # Trouver les pics locaux
        peaks = signal.peak_local_max(autocorr,
                                    min_distance=self.min_pattern_size,
                                    threshold_rel=0.5)
        
        # Convertir les pics en motifs
        patterns = []
        for peak in peaks:
            y, x = peak
            # Calculer la taille du motif basée sur la distance au centre
            center_y, center_x = autocorr.shape[0]//2, autocorr.shape[1]//2
            pattern_size_y = abs(y - center_y)
            pattern_size_x = abs(x - center_x)
            
            if self.min_pattern_size < pattern_size_x < self.max_pattern_size and \
               self.min_pattern_size < pattern_size_y < self.max_pattern_size:
                confidence = autocorr[y, x]
                patterns.append(([x - pattern_size_x//2, y - pattern_size_y//2,
                                x + pattern_size_x//2, y + pattern_size_y//2],
                               float(confidence)))
        
        return patterns

    def _detect_scale_patterns(self, image: np.ndarray) -> List[Tuple[List[int], float]]:
        """
        Détection des motifs à différentes échelles
        """
        # SIFT pour la détection de points caractéristiques
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        if descriptors is None or len(keypoints) < 2:
            return []
            
        # Clustering des descripteurs pour trouver les motifs similaires
        clustering = DBSCAN(eps=0.5, min_samples=3)
        clusters = clustering.fit_predict(descriptors)
        
        # Grouper les points par cluster
        patterns = []
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Ignorer le bruit
                continue
                
            # Points dans ce cluster
            cluster_points = [keypoints[i] for i in range(len(keypoints)) 
                            if clusters[i] == cluster_id]
            
            if len(cluster_points) >= 3:  # Au moins 3 points pour former un motif
                # Calculer la boîte englobante
                points = np.float32([[kp.pt[0], kp.pt[1]] for kp in cluster_points])
                rect = cv2.boundingRect(points)
                
                # Calculer la confiance basée sur la densité des points
                density = len(cluster_points) / (rect[2] * rect[3])
                confidence = min(density * 1000, 1.0)  # Normaliser
                
                patterns.append(([int(rect[0]), int(rect[1]),
                                int(rect[0] + rect[2]), int(rect[1] + rect[3])],
                               confidence))
        
        return patterns

    def _validate_patterns(
        self,
        image: np.ndarray,
        detections: List[Tuple[List[int], float]]
    ) -> List[Tuple[List[int], float]]:
        """
        Validation des motifs détectés avec le CNN
        """
        if not detections:
            return []
            
        validated = []
        with torch.no_grad():
            for bbox, conf in detections:
                # Extraire la région
                x1, y1, x2, y2 = bbox
                roi = image[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                # Redimensionner pour le CNN
                roi = cv2.resize(roi, (64, 64))
                roi = torch.from_numpy(roi).float().permute(2, 0, 1).unsqueeze(0)
                roi = roi.to(self.device)
                
                # Prédiction
                score = self.pattern_cnn(roi).item()
                
                # Combiner avec la confiance initiale
                final_conf = (score + conf) / 2
                if final_conf > self.pattern_threshold:
                    validated.append((bbox, final_conf))
        
        return validated

    def _post_process_detections(
        self,
        detections: List[Tuple[List[int], float]]
    ) -> List[Tuple[List[int], float]]:
        """
        Post-traitement des détections pour fusionner les boîtes qui se chevauchent
        """
        if not detections:
            return []
            
        # Trier par confiance
        detections.sort(key=lambda x: x[1], reverse=True)
        
        # Non-maximum suppression
        kept_indices = []
        for i, (box1, _) in enumerate(detections):
            keep = True
            for j in kept_indices:
                box2 = detections[j][0]
                iou = self._calculate_iou(box1, box2)
                if iou > 0.5:  # Seuil IoU
                    keep = False
                    break
            if keep:
                kept_indices.append(i)
        
        return [detections[i] for i in kept_indices]

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

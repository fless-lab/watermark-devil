"""
Advanced transparency detector using deep learning and image processing
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from scipy.ndimage import gaussian_filter
from skimage import exposure, filters, morphology

class TransparencyDetector:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize transparency detector with advanced techniques
        """
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paramètres
        self.alpha_threshold = self.config.get('alpha_threshold', 0.1)
        self.min_area = self.config.get('min_area', 100)
        self.gaussian_sigma = self.config.get('gaussian_sigma', 2.0)
        
        # CNN pour la détection avancée
        self.transparency_cnn = self._build_transparency_cnn()
        self.transparency_cnn.to(self.device)
        
        # Cache pour les résultats intermédiaires
        self._cache = {}

    def preprocess(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Prétraitement avancé pour la détection de transparence
        """
        # Extraire le canal alpha si présent
        if image.shape[-1] == 4:
            rgb = image[..., :3]
            alpha = image[..., 3]
        else:
            rgb = image
            alpha = None
            
        # Amélioration du contraste
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        
        # Calcul des gradients
        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        gradients = self._compute_gradients(gray)
        
        return {
            "original": image,
            "rgb": rgb,
            "alpha": alpha,
            "enhanced": enhanced,
            "gradients": gradients,
            "original_shape": image.shape[:2]
        }

    def detect(self, inputs: Dict[str, Any]) -> List[Tuple[List[int], float]]:
        """
        Détection avancée des zones transparentes
        """
        detections = []
        
        # 1. Détection basée sur le canal alpha
        if inputs["alpha"] is not None:
            alpha_detections = self._detect_alpha_channel(inputs["alpha"])
            detections.extend(alpha_detections)
        
        # 2. Détection basée sur les gradients
        gradient_detections = self._detect_gradient_patterns(
            inputs["gradients"],
            inputs["enhanced"]
        )
        detections.extend(gradient_detections)
        
        # 3. Détection par deep learning
        cnn_detections = self._detect_with_cnn(inputs["rgb"])
        detections.extend(cnn_detections)
        
        # 4. Fusion et post-traitement
        final_detections = self._post_process_detections(detections)
        
        # 5. Génération des masques
        final_detections = self._generate_masks(inputs["rgb"], final_detections)
        
        return final_detections

    def _build_transparency_cnn(self) -> nn.Module:
        """
        CNN spécialisé pour la détection de transparence
        """
        class TransparencyCNN(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                
                # Attention module
                self.attention = nn.Sequential(
                    nn.Conv2d(128, 1, 1),
                    nn.Sigmoid()
                )
                
                # Decoder
                self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
                self.upconv3 = nn.ConvTranspose2d(32, 1, 2, stride=2)
                
            def forward(self, x):
                # Encoding
                x1 = F.relu(self.conv1(x))
                x1_pool = F.max_pool2d(x1, 2)
                
                x2 = F.relu(self.conv2(x1_pool))
                x2_pool = F.max_pool2d(x2, 2)
                
                x3 = F.relu(self.conv3(x2_pool))
                
                # Attention
                att = self.attention(x3)
                x3_att = x3 * att
                
                # Decoding with skip connections
                x4 = F.relu(self.upconv1(x3_att))
                x4 = x4 + x2
                
                x5 = F.relu(self.upconv2(x4))
                x5 = x5 + x1
                
                x6 = torch.sigmoid(self.upconv3(x5))
                
                return x6
                
        return TransparencyCNN()

    def _compute_gradients(self, image: np.ndarray) -> np.ndarray:
        """
        Calcul avancé des gradients
        """
        # Sobel
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        
        # Magnitude et direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Normalisation
        magnitude = exposure.rescale_intensity(magnitude)
        
        return {"magnitude": magnitude, "direction": direction}

    def _detect_alpha_channel(self, alpha: np.ndarray) -> List[Tuple[List[int], float]]:
        """
        Détection basée sur le canal alpha
        """
        # Seuillage adaptatif
        thresh = filters.threshold_otsu(alpha)
        binary = alpha > thresh
        
        # Débruitage morphologique
        binary = morphology.remove_small_objects(binary)
        binary = morphology.remove_small_holes(binary)
        
        # Trouver les contours
        contours, _ = cv2.findContours(
            binary.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                confidence = np.mean(alpha[y:y+h, x:x+w]) / 255.0
                if confidence > self.alpha_threshold:
                    detections.append(([x, y, x+w, y+h], confidence))
        
        return detections

    def _detect_gradient_patterns(
        self,
        gradients: Dict[str, np.ndarray],
        image: np.ndarray
    ) -> List[Tuple[List[int], float]]:
        """
        Détection basée sur les motifs de gradient
        """
        magnitude = gradients["magnitude"]
        direction = gradients["direction"]
        
        # Calcul de l'entropie locale
        entropy = filters.rank.entropy(magnitude.astype(np.uint8), morphology.disk(5))
        
        # Détection des zones de forte entropie
        high_entropy = entropy > np.percentile(entropy, 90)
        
        # Clustering directionnel
        direction_bins = np.histogram(direction[high_entropy], bins=8)[0]
        direction_entropy = -np.sum(direction_bins * np.log2(direction_bins + 1e-10))
        
        # Trouver les régions candidates
        labeled = morphology.label(high_entropy)
        regions = morphology.regionprops(labeled)
        
        detections = []
        for region in regions:
            if region.area > self.min_area:
                # Calculer la confiance basée sur l'entropie et la cohérence directionnelle
                y1, x1, y2, x2 = region.bbox
                roi_entropy = np.mean(entropy[y1:y2, x1:x2])
                confidence = (roi_entropy / 8.0) * (1 - direction_entropy / 3.0)
                
                if confidence > self.alpha_threshold:
                    detections.append(([x1, y1, x2, y2], confidence))
        
        return detections

    def _detect_with_cnn(self, image: np.ndarray) -> List[Tuple[List[int], float]]:
        """
        Détection avec le CNN
        """
        with torch.no_grad():
            # Préparer l'image pour le CNN
            input_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
            input_tensor = input_tensor.to(self.device) / 255.0
            
            # Prédiction
            mask = self.transparency_cnn(input_tensor)[0, 0].cpu().numpy()
            
            # Post-traitement du masque
            mask = gaussian_filter(mask, self.gaussian_sigma)
            mask = (mask > self.alpha_threshold).astype(np.uint8)
            
            # Trouver les contours
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    roi_mask = mask[y:y+h, x:x+w]
                    confidence = np.mean(roi_mask)
                    detections.append(([x, y, x+w, y+h], confidence))
            
            return detections

    def _post_process_detections(
        self,
        detections: List[Tuple[List[int], float]]
    ) -> List[Tuple[List[int], float]]:
        """
        Post-traitement des détections
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
                if iou > 0.5:
                    keep = False
                    break
            if keep:
                kept_indices.append(i)
        
        return [detections[i] for i in kept_indices]

    def _generate_masks(
        self,
        image: np.ndarray,
        detections: List[Tuple[List[int], float]]
    ) -> List[Tuple[List[int], float, np.ndarray]]:
        """
        Génération des masques pour chaque détection
        """
        results = []
        for bbox, conf in detections:
            x1, y1, x2, y2 = bbox
            roi = image[y1:y2, x1:x2]
            
            # Créer un masque précis
            mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            
            # GrabCut pour affiner le masque
            bgd_model = np.zeros((1,65), np.float64)
            fgd_model = np.zeros((1,65), np.float64)
            rect = (0, 0, x2-x1, y2-y1)
            mask_rect = np.zeros(roi.shape[:2], np.uint8)
            cv2.grabCut(roi, mask_rect, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Affiner avec le CNN
            with torch.no_grad():
                roi_tensor = torch.from_numpy(roi).float().permute(2, 0, 1).unsqueeze(0)
                roi_tensor = roi_tensor.to(self.device) / 255.0
                mask_pred = self.transparency_cnn(roi_tensor)[0, 0].cpu().numpy()
                
                # Combiner GrabCut et CNN
                mask = ((mask_rect == cv2.GC_FGD) | (mask_rect == cv2.GC_PR_FGD)).astype(np.uint8)
                mask = mask & (mask_pred > self.alpha_threshold)
                
                # Nettoyage final
                mask = morphology.remove_small_objects(mask.astype(bool))
                mask = morphology.remove_small_holes(mask)
                mask = mask.astype(np.uint8) * 255
            
            results.append((bbox, conf, mask))
        
        return results

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

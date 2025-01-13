"""
Configuration for the logo detector
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class LogoDetectorConfig:
    # Modèle
    model_type: str = "yolov8x"  # Le plus avancé de la série YOLOv8
    pretrained_path: Optional[str] = None
    device: Optional[str] = None  # Auto-détection
    
    # Paramètres d'inférence
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_detections: int = 50
    
    # Prétraitement
    input_size: Tuple[int, int] = (1024, 1024)  # Haute résolution
    normalize_mean: List[float] = [0.485, 0.456, 0.406]
    normalize_std: List[float] = [0.229, 0.224, 0.225]
    
    # Augmentation pendant l'inférence
    use_tta: bool = True  # Test Time Augmentation
    tta_scales: List[float] = [0.8, 1.0, 1.2]
    tta_flips: bool = True
    
    # Post-traitement
    apply_soft_nms: bool = True
    soft_nms_sigma: float = 0.5
    min_box_area: int = 100
    
    # Performance
    use_tensorrt: bool = True  # Optimisation TensorRT si disponible
    batch_size: int = 4
    num_workers: int = 4
    
    # Logging et débogage
    save_crops: bool = True
    debug_visualization: bool = True

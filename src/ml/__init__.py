"""
Machine Learning module for watermark detection and removal
"""
import os
import torch
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configuration CUDA
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
# Configuration des chemins
MODELS_DIR = Path(__file__).parent / "models"
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Téléchargement des modèles pré-entraînés
def download_models():
    """
    Télécharge les modèles pré-entraînés nécessaires
    """
    from ultralytics import YOLO
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import easyocr
    from paddleocr import PaddleOCR
    
    try:
        # YOLOv8 pour la détection de logos
        YOLO("yolov8x.pt")
        
        # TrOCR pour le texte manuscrit
        TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
        
        # EasyOCR
        reader = easyocr.Reader(['en'])
        
        # PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())
        
        logging.info("All models downloaded successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error downloading models: {str(e)}")
        return False

# Configuration de l'environnement
def setup_environment():
    """
    Configure l'environnement pour les modèles
    """
    # Configuration CUDA
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.cuda.empty_cache()
    
    # Configuration PaddlePaddle
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
    
    # Configuration TensorRT
    if torch.cuda.is_available():
        os.environ["TRT_LOGGER_SEVERITY"] = "1"  # ERROR level
        
    logging.info(f"Environment configured. CUDA available: {torch.cuda.is_available()}")

# Initialisation automatique
setup_environment()

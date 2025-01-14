"""
Installation script for watermark detection system
"""
import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_venv():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path(__file__).parents[1] / "venv"
    if not venv_path.exists():
        logger.info("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        logger.info("Virtual environment created successfully")
    else:
        logger.info("Virtual environment already exists")

def install_requirements():
    """Install required packages"""
    logger.info("Installing requirements...")
    pip_path = "venv/Scripts/pip.exe" if sys.platform == "win32" else "venv/bin/pip"
    requirements_path = Path(__file__).parents[1] / "requirements.txt"
    
    # Mettre à jour pip d'abord
    subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    
    # Installer les dépendances
    subprocess.run([pip_path, "install", "-r", str(requirements_path)], check=True)
    logger.info("Requirements installed successfully")

def download_models():
    """Download required models"""
    # Import ici après l'installation des dépendances
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from ultralytics import YOLO
    
    logger.info("Downloading models...")
    models_dir = Path(__file__).parents[1] / "models"
    models_dir.mkdir(exist_ok=True)

    # YOLOv8 for logo detection
    logger.info("Downloading YOLOv8 model...")
    yolo_path = models_dir / "yolov8x.pt"
    if not yolo_path.exists():
        model = YOLO('yolov8x.pt')
        model.save(str(yolo_path))

    # TrOCR for text detection
    logger.info("Downloading TrOCR model...")
    trocr_path = models_dir / "trocr-large-handwritten"
    if not trocr_path.exists():
        tokenizer = AutoTokenizer.from_pretrained('microsoft/trocr-large-handwritten')
        model = AutoModelForTokenClassification.from_pretrained('microsoft/trocr-large-handwritten')
        tokenizer.save_pretrained(str(trocr_path))
        model.save_pretrained(str(trocr_path))

    logger.info("Models downloaded successfully")

def build_rust_project():
    """Build Rust project"""
    logger.info("Building Rust project...")
    try:
        subprocess.run(["cargo", "build", "--release"], check=True)
        logger.info("Rust project built successfully")
    except subprocess.CalledProcessError:
        logger.error("Failed to build Rust project")
        raise

def create_required_directories():
    """Create required directories"""
    logger.info("Creating required directories...")
    root_dir = Path(__file__).parents[1]
    
    # Create required directories
    dirs = [
        "cache",
        "logs",
        "temp",
        "data",
        "data/uploads",
        "data/processed"
    ]
    
    for dir_name in dirs:
        dir_path = root_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Required directories created successfully")

def main():
    """Main installation function"""
    try:
        # Create virtual environment
        create_venv()

        # Install requirements
        install_requirements()

        # Create directories
        create_required_directories()

        # Build Rust project
        build_rust_project()

        # Download models (après l'installation des dépendances)
        download_models()

        logger.info("Installation completed successfully!")
        
    except Exception as e:
        logger.error(f"Installation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

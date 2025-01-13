"""
Tests d'intégration pour le système de détection de filigranes
"""
import os
import pytest
import numpy as np
import cv2
import torch
from pathlib import Path

from src.ml.models.logo_detector.model import LogoDetector
from src.ml.models.text_detector.model import TextDetector
from src.ml.models.pattern_detector.model import PatternDetector
from src.ml.models.transparency_detector.model import TransparencyDetector

# Chemins des ressources de test
TEST_RESOURCES = Path(__file__).parent / "resources"
TEST_IMAGES = TEST_RESOURCES / "images"
TEST_IMAGES.mkdir(parents=True, exist_ok=True)

def create_test_image(type: str) -> np.ndarray:
    """Crée une image de test avec un filigrane spécifique"""
    image = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    if type == "logo":
        # Dessiner un logo simple
        cv2.rectangle(image, (200, 200), (300, 300), (0, 0, 0), -1)
        cv2.circle(image, (250, 250), 30, (255, 255, 255), -1)
    
    elif type == "text":
        # Ajouter du texte
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "WATERMARK", (150, 250), font, 2, (128, 128, 128), 2)
    
    elif type == "pattern":
        # Créer un motif répétitif
        pattern = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.circle(pattern, (25, 25), 10, (200, 200, 200), -1)
        for i in range(0, 512, 50):
            for j in range(0, 512, 50):
                image[i:i+50, j:j+50] = pattern
    
    elif type == "transparent":
        # Ajouter un filigrane semi-transparent
        overlay = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(overlay, "CONFIDENTIAL", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (0, 0, 0), 2)
        image = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)
    
    return image

@pytest.fixture(scope="session")
def test_images():
    """Crée et sauvegarde les images de test"""
    images = {}
    for type in ["logo", "text", "pattern", "transparent"]:
        image = create_test_image(type)
        path = TEST_IMAGES / f"{type}.png"
        cv2.imwrite(str(path), image)
        images[type] = image
    return images

@pytest.fixture(scope="session")
def cuda_available():
    """Vérifie si CUDA est disponible"""
    return torch.cuda.is_available()

def test_logo_detector(test_images, cuda_available):
    """Test du détecteur de logos"""
    detector = LogoDetector()
    image = test_images["logo"]
    
    # Test de prétraitement
    preprocessed = detector.preprocess(image)
    assert isinstance(preprocessed, torch.Tensor)
    assert preprocessed.device.type == ("cuda" if cuda_available else "cpu")
    
    # Test de détection
    detections = detector.detect(preprocessed)
    assert len(detections) > 0
    
    # Vérification du format des détections
    for bbox, conf in detections:
        assert len(bbox) == 4  # [x1, y1, x2, y2]
        assert 0 <= conf <= 1  # Score de confiance

def test_text_detector(test_images):
    """Test du détecteur de texte"""
    detector = TextDetector()
    image = test_images["text"]
    
    # Test de prétraitement
    inputs = detector.preprocess(image)
    assert isinstance(inputs, dict)
    assert "paddle" in inputs
    assert "trocr" in inputs
    assert "easyocr" in inputs
    
    # Test de détection
    detections = detector.detect(inputs)
    assert len(detections) > 0
    
    # Vérification du texte détecté
    found_watermark = False
    for bbox, conf, text in detections:
        if "WATERMARK" in text.upper():
            found_watermark = True
            break
    assert found_watermark

def test_pattern_detector(test_images):
    """Test du détecteur de motifs"""
    detector = PatternDetector()
    image = test_images["pattern"]
    
    # Test de prétraitement
    inputs = detector.preprocess(image)
    assert isinstance(inputs, dict)
    assert "gray" in inputs
    assert "enhanced" in inputs
    assert "pyramids" in inputs
    
    # Test de détection
    detections = detector.detect(inputs)
    assert len(detections) > 0
    
    # Vérification de la périodicité
    if len(detections) >= 2:
        bbox1, _ = detections[0]
        bbox2, _ = detections[1]
        # Les motifs devraient être espacés régulièrement
        distance = abs(bbox1[0] - bbox2[0])
        assert distance == pytest.approx(50, rel=0.2)  # ±20% de tolérance

def test_transparency_detector(test_images):
    """Test du détecteur de transparence"""
    detector = TransparencyDetector()
    image = test_images["transparent"]
    
    # Test de prétraitement
    inputs = detector.preprocess(image)
    assert isinstance(inputs, dict)
    assert "rgb" in inputs
    assert "enhanced" in inputs
    assert "gradients" in inputs
    
    # Test de détection
    detections = detector.detect(inputs)
    assert len(detections) > 0
    
    # Vérification des zones semi-transparentes
    for bbox, conf, mask in detections:
        roi = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # La zone détectée devrait avoir une variance de couleur
        assert np.var(roi) > 0
        # Le masque devrait correspondre à la taille de la ROI
        assert mask.shape == (bbox[3]-bbox[1], bbox[2]-bbox[0])

def test_rust_integration():
    """Test de l'intégration Rust"""
    import subprocess
    
    # Compiler et exécuter le binaire de test Rust
    result = subprocess.run(["cargo", "test", "--test", "detection", "--", "--nocapture"],
                          capture_output=True, text=True)
    
    # Vérifier que les tests Rust passent
    assert result.returncode == 0
    
    # Vérifier les messages spécifiques
    assert "Python integration successful" in result.stdout
    assert "All detectors initialized" in result.stdout
    assert "Detection pipeline complete" in result.stdout

def test_end_to_end(test_images):
    """Test de bout en bout"""
    # Charger l'image
    image_path = TEST_IMAGES / "logo.png"
    
    # Exécuter la détection via l'API Rust
    result = subprocess.run(["cargo", "run", "--release", "--", 
                           "detect", str(image_path)],
                          capture_output=True, text=True)
    
    # Vérifier le succès
    assert result.returncode == 0
    
    # Vérifier la sortie JSON
    import json
    output = json.loads(result.stdout)
    
    # Vérifier la structure
    assert "detections" in output
    assert len(output["detections"]) > 0
    
    # Vérifier les champs requis
    detection = output["detections"][0]
    assert "bbox" in detection
    assert "confidence" in detection
    assert "type" in detection

def test_watermark_detection():
    # First create test images
    from tests.integration.resources.create_test_images import main as create_test_images
    create_test_images()
    
    # Load test image
    test_image_path = Path(__file__).parent / "resources/images/combined.png"
    image = cv2.imread(str(test_image_path))
    if image is None:
        raise ValueError(f"Could not load image from {test_image_path}")
    
    print("\nTesting Logo Detection:")
    try:
        logo_detector = LogoDetector()
        # Preprocess image for logo detection
        processed_image = logo_detector.preprocess(image)
        logo_results = logo_detector.detect(processed_image)
        print(f"Found {len(logo_results)} potential logo watermarks")
        for i, (box, conf) in enumerate(logo_results):
            print(f"Logo {i+1}: Confidence {conf:.2f}, Box: {box}")
    except Exception as e:
        print(f"Logo detection failed: {str(e)}")
    
    print("\nTesting Text Detection:")
    try:
        text_detector = TextDetector()
        # Preprocess image for text detection
        processed_image = text_detector.preprocess(image)
        text_results = text_detector.detect(processed_image)
        print(f"Found {len(text_results)} text watermarks")
        for i, (box, conf, text) in enumerate(text_results):
            print(f"Text {i+1}: '{text}', Confidence {conf:.2f}, Box: {box}")
    except Exception as e:
        print(f"Text detection failed: {str(e)}")
    
    print("\nTesting Pattern Detection:")
    try:
        pattern_detector = PatternDetector()
        # Preprocess image for pattern detection
        processed_image = pattern_detector.preprocess(image)
        pattern_results = pattern_detector.detect(processed_image)
        print(f"Found {len(pattern_results)} repeating patterns")
        for i, (area, conf) in enumerate(pattern_results):
            print(f"Pattern {i+1}: Confidence {conf:.2f}, Area: {area}")
    except Exception as e:
        print(f"Pattern detection failed: {str(e)}")
    
    print("\nTesting Transparency Detection:")
    try:
        transparency_detector = TransparencyDetector()
        # Preprocess image for transparency detection
        processed_image = transparency_detector.preprocess(image)
        transparency_results = transparency_detector.detect(processed_image)
        print(f"Found {len(transparency_results)} semi-transparent watermarks")
        for i, (area, conf) in enumerate(transparency_results):
            print(f"Transparency {i+1}: Confidence {conf:.2f}, Area: {area}")
    except Exception as e:
        print(f"Transparency detection failed: {str(e)}")

def test_webp_detection():
    """Test watermark detection on WebP images"""
    # Create a test WebP image with watermark
    test_image_path = Path(__file__).parent / "resources/images/test.webp"
    
    # Create a sample image with watermark
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img.fill(255)  # White background
    
    # Add a watermark text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'TEST WATERMARK', (50, 150), font, 1, (100, 100, 100), 2)
    
    # Save as WebP
    cv2.imwrite(str(test_image_path), img, [cv2.IMWRITE_WEBP_QUALITY, 100])
    
    # Test detection
    try:
        # Test text detection
        text_detector = TextDetector()
        processed = text_detector.preprocess(img)
        text_results = text_detector.detect(processed)
        
        # Should find at least one text watermark
        assert len(text_results) > 0, "No text watermark detected in WebP image"
        
        # Check first detection
        box, conf, text = text_results[0]
        assert conf > 0.3, f"Low confidence text detection: {conf}"
        assert "WATERMARK" in text, f"Expected 'WATERMARK' in detected text, got: {text}"
        
    finally:
        # Cleanup
        if test_image_path.exists():
            test_image_path.unlink()

if __name__ == "__main__":
    test_watermark_detection()

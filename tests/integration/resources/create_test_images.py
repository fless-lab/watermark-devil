"""
Script pour créer les images de test
"""
import cv2
import numpy as np
from pathlib import Path

def create_combined_test_image():
    """Crée une image de test avec plusieurs types de filigranes"""
    image = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Ajouter un logo
    cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 0), -1)
    cv2.circle(image, (100, 100), 30, (255, 255, 255), -1)
    
    # Ajouter du texte
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "CONFIDENTIAL", (200, 250), font, 1.5, (128, 128, 128), 2)
    
    # Ajouter un motif répétitif
    pattern = np.zeros((30, 30, 3), dtype=np.uint8)
    cv2.circle(pattern, (15, 15), 5, (200, 200, 200), -1)
    for i in range(300, 482, 30):  
        for j in range(300, 482, 30):
            image[i:i+30, j:j+30] = pattern
    
    # Ajouter un filigrane semi-transparent
    overlay = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.putText(overlay, "WATERMARK", (150, 400), font, 2, (0, 0, 0), 2)
    image = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)
    
    return image

def main():
    """Fonction principale"""
    # Créer le répertoire des ressources
    resources_dir = Path(__file__).parent / "images"
    resources_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer et sauvegarder l'image combinée
    combined_image = create_combined_test_image()
    cv2.imwrite(str(resources_dir / "combined.png"), combined_image)
    
    print("Images de test créées avec succès!")

if __name__ == "__main__":
    main()

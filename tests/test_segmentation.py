import os
import cv2
import numpy as np
from PIL import Image
from src.segmentation import segment_manuscript_lines

def test_segmentation_on_dummy_image(tmp_path):
    """Vérifie que le module de segmentation extrait correctement les images des lignes."""
    # 1. Création d'une image factice blanche avec 3 bandes noires (simulant du texte)
    dummy_array = np.ones((200, 400), dtype=np.uint8) * 255
    dummy_array[40:60, 20:380] = 0   # Ligne 1
    dummy_array[90:110, 20:380] = 0  # Ligne 2
    dummy_array[140:160, 20:380] = 0 # Ligne 3

    temp_img_path = os.path.join(tmp_path, "dummy_page.png")
    cv2.imwrite(temp_img_path, dummy_array)

    # 2. Exécution de la segmentation
    extracted_lines = segment_manuscript_lines(temp_img_path)

    # 3. Vérifications : on attend une liste contenant nos 3 lignes extraites
    assert isinstance(extracted_lines, list)
    assert len(extracted_lines) == 3
    assert isinstance(extracted_lines[0], np.ndarray)
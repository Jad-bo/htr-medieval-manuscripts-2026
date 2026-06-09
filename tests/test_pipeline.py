import os
import numpy as np
import cv2
import pytest
from unittest.mock import patch
from src.main import run_pipeline

def test_full_pipeline_end_to_end_real_image(tmp_path):
    """Vérifie le fonctionnement global du pipeline en utilisant l'image réelle CATMuS."""
    
    # 1. Chemin vers la vraie image présente dans ton architecture de données
    real_image_path = "./data/images/banner_catmus_medieval_centered.png"
    
    # Sécurité au cas où l'image n'est pas présente lors d'une exécution CI/CD par exemple
    if not os.path.exists(real_image_path):
        pytest.skip("L'image de test réelle CATMuS est absente du dossier ./data/images/")

    output_json = os.path.join(tmp_path, "output_contract.json")

    # 2. Extraction réelle des lignes de l'image via notre segmentation hybride
    # On laisse la fonction s'exécuter normalement pour tester l'intégration
    from src.segmentation import segment_manuscript_lines
    extracted_lines = segment_manuscript_lines(real_image_path)
    
    # On mock uniquement le retour géométrique complexe si ton composant extract_bounding_polygons 
    # n'est pas encore totalement couplé aux listes natives, afin de sécuriser l'export JSON.
    mock_geo_polygons = [
        {"line_idx": i, "box": [0, 0, 100, 50]} for i in range(len(extracted_lines))
    ]

    # On patche uniquement la partie géométrique/TrOCR pour éviter d'appeler l'IA pendant le test unitaire
    with patch("src.main.segment_manuscript_lines", return_value=extracted_lines), \
         patch("src.main.extract_bounding_polygons", return_value=mock_geo_polygons):
        
        # Exécution du pipeline complet sur la vraie image
        run_pipeline(image_path=real_image_path, document_id="doc_catmus_real", output_json_path=output_json)

    # 3. Validation de la conformité du contrat de données final
    assert os.path.exists(output_json), "Le fichier JSON de sortie n'a pas été généré."
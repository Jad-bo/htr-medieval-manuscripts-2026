"""Pipeline principal (End-to-End) pour le traitement des manuscrits médiévaux."""

import os
from typing import List
import cv2

from src.config import PROCESSED_DIR
from src.preprocessing import preprocess_pipeline
from src.segmentation import segment_manuscript_lines, extract_bounding_polygons
from src.data_contract import generate_line_entry, save_data_contract


def run_pipeline(image_path: str, document_id: str, output_json_path: str) -> None:
    """Exécute toutes les étapes du pipeline HTR sur une image donnée.

    Args:
        image_path (str): Chemin de l'image originale.
        document_id (str): Identifiant unique du manuscrit.
        output_json_path (str): Chemin où sauvegarder le Data Contract JSON.
    """
    print(f"--- Lancement du pipeline pour le document : {document_id} ---")
    
    # 1. Prétraitement de l'image
    print("[1/4] Prétraitement de l'image (Deskewing, CLAHE, Sauvola)...")
    processed_img = preprocess_pipeline(image_path)
    
    # Sauvegarde de l'image intermédiaire pour la traçabilité
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    temp_processed_path = os.path.join(PROCESSED_DIR, f"{document_id}_binarized.png")
    cv2.imwrite(temp_processed_path, processed_img)
    
    # 2. Segmentation géométrique
    print("[2/4] Segmentation des lignes avec Kraken...")
    seg_result = segment_manuscript_lines(temp_processed_path)
    polygons = extract_bounding_polygons(seg_result)
    print(f"-> {len(polygons)} lignes de texte détectées.")
    
    # 3. Transcription HTR (Simulation pour le pipeline initial)
    print("[3/4] Transcription textuelle (Mock TrOCR)...")
    lines_entries = []
    for i, poly in enumerate(polygons):
        line_id = f"{document_id}_line_{i+1}"
        
        # Simulation d'une prédiction textuelle et d'une confiance avant intégration du modèle lourd
        mock_text = f"[Transcription simulée de la ligne {i+1}]"
        mock_confidence = 0.88 if i % 2 == 0 else 0.74  # Fait varier le flag needs_review
        
        # Génération de l'entrée conforme au Data Contract
        entry = generate_line_entry(
            line_id=line_id,
            text=mock_text,
            confidence=mock_confidence,
            polygon=poly
        )
        lines_entries.append(entry)
        
    # 4. Exportation et validation du Data Contract
    print(f"[4/4] Exportation des données vers {output_json_path}...")
    save_data_contract(output_json_path, document_id, lines_entries)
    print("--- Pipeline exécuté avec succès ! ---")


if __name__ == "__main__":
    # Petit test à vide si lancé directement
    print("Pipeline prêt. Utilisez pytest ou appelez run_pipeline() depuis vos scripts.")
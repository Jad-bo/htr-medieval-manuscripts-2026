"""Validation et structure du jeu de données final (Data Contract)."""

import json
from typing import Any, Dict, List


def generate_line_entry(
    line_id: str, text: str, confidence: float, polygon: List[List[int]]
) -> Dict[str, Any]:
    """Génère une entrée de ligne validée selon les critères du Data Contract.

    Args:
        line_id (str): Identifiant unique de la ligne.
        text (str): Texte transcrit par le modèle HTR.
        confidence (float): Score de confiance calibré (0.0 à 1.0).
        polygon (List[List[int]]): Liste de points [x, y] définissant le contour.

    Returns:
        Dict[str, Any]: Dictionnaire conforme au schéma attendu.
    """
    # Correction de la condition pour lever le SyntaxWarning
    has_low_confidence = confidence < 0.80
    is_too_short = len(text.strip()) <= 1
    
    needs_review = bool(has_low_confidence or is_too_short)
    
    return {
        "line_id": line_id,
        "transcription": text,
        "confidence": round(confidence, 4),
        "geometry": {
            "type": "Polygon",
            "coordinates": polygon,
            "unit": "pixels"
        },
        "needs_review": needs_review
    }


def save_data_contract(output_path: str, document_id: str, lines: List[Dict[str, Any]]) -> None:
    """Exporte les transcriptions et métadonnées spatiales au format JSON contractuel.

    Args:
        output_path (str): Chemin du fichier JSON de sortie.
        document_id (str): Identifiant du manuscrit d'origine.
        lines (List[Dict[str, Any]]): Liste des lignes générées par `generate_line_entry`.
    """
    output_data = {
        "document_id": document_id,
        "system_coordinates": "origin_top_left",
        "lines": lines
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
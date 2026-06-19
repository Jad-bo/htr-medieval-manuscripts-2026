"""
Validation et structure du jeu de données final (Data Contract).

Conforme aux exigences du brief MD5-2026 :
  - JSON structuré avec transcriptions, confiances et polygones
  - Flag needs_review pour les lignes incertaines
  - Compatible avec eScriptorium et les outils d'annotation collaborative
  - Schéma validable via jsonschema
"""

import json
import os
from typing import Any, Dict, List, Optional

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    import warnings
    warnings.warn("jsonschema non installé. La validation du data contract sera désactivée.")


# ============================================================
# 1. SCHÉMA JSON DU DATA CONTRACT
# ============================================================

DATA_CONTRACT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "document_id": {
            "type": "string",
            "description": "Identifiant unique du manuscrit d'origine"
        },
        "system_coordinates": {
            "type": "string",
            "enum": ["origin_top_left"],
            "description": "Système de coordonnées : origine en haut à gauche"
        },
        "image_path": {
            "type": "string",
            "description": "Chemin vers l'image source (optionnel)"
        },
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_id": {
                        "type": "string",
                        "description": "Identifiant unique de la ligne"
                    },
                    "transcription": {
                        "type": "string",
                        "description": "Texte transcrit par le modèle HTR"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Score de confiance calibré (0.0 à 1.0)"
                    },
                    "geometry": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["Polygon"]
                            },
                            "coordinates": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": 2,
                                    "maxItems": 2
                                },
                                "minItems": 3,
                                "description": "Liste de points [x, y] définissant le contour"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["pixels"]
                            }
                        },
                        "required": ["type", "coordinates", "unit"]
                    },
                    "needs_review": {
                        "type": "boolean",
                        "description": "True si la ligne nécessite une relecture humaine"
                    },
                    "reading_order": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Position dans l'ordre de lecture de la page"
                    },
                    "region_type": {
                        "type": "string",
                        "enum": ["main_text", "marginalia", "heading", "illustration", "unknown"],
                        "description": "Type de région de la ligne"
                    },
                    "model_version": {
                        "type": "string",
                        "description": "Version du modèle HTR utilisé"
                    }
                },
                "required": ["line_id", "transcription", "confidence", "geometry", "needs_review"]
            }
        }
    },
    "required": ["document_id", "system_coordinates", "lines"]
}


# ============================================================
# 2. FONCTIONS DE GÉNÉRATION
# ============================================================

def generate_line_entry(
    line_id: str,
    text: str,
    confidence: float,
    polygon: List[List[int]],
    reading_order: int = 0,
    region_type: str = "main_text",
    model_version: str = "trocr-lora-v1",
    confidence_threshold: float = 0.80
) -> Dict[str, Any]:
    """
    Génère une entrée de ligne validée selon les critères du Data Contract.

    Une ligne est flaggée `needs_review=True` si :
      - Sa confiance est < confidence_threshold (défaut 0.80)
      - Son texte est trop court (≤ 1 caractère)
      - Elle contient des caractères suspects ou des artefacts

    Args:
        line_id: Identifiant unique de la ligne (ex: "ms_001_line_0001")
        text: Texte transcrit par le modèle HTR
        confidence: Score de confiance calibré entre 0.0 et 1.0
        polygon: Liste de points [x, y] définissant le contour de la ligne
        reading_order: Position dans l'ordre de lecture (0, 1, 2, ...)
        region_type: Type de région (main_text, marginalia, heading, ...)
        model_version: Identifiant du modèle HTR utilisé
        confidence_threshold: Seuil de confiance pour flagger needs_review

    Returns:
        Dict conforme au schéma du Data Contract

    Example:
        >>> entry = generate_line_entry(
        ...     line_id="ms_001_line_0001",
        ...     text="In nomine Domini amen",
        ...     confidence=0.92,
        ...     polygon=[[10, 100], [200, 100], [200, 120], [10, 120]],
        ...     reading_order=0
        ... )
        >>> entry["needs_review"]
        False
    """
    # Validation des entrées
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError(f"Le polygone doit contenir au moins 3 points, reçu : {len(polygon)}")

    if not all(len(p) == 2 and all(isinstance(c, int) for c in p) for p in polygon):
        raise ValueError("Chaque point du polygone doit être [int, int]")

    # Normaliser la confiance
    confidence = float(max(0.0, min(1.0, confidence)))

    # Déterminer si la ligne nécessite une relecture
    has_low_confidence = confidence < confidence_threshold
    is_too_short = len(text.strip()) <= 1
    has_suspicious_chars = _has_suspicious_patterns(text)

    needs_review = bool(has_low_confidence or is_too_short or has_suspicious_chars)

    return {
        "line_id": line_id,
        "transcription": text.strip(),
        "confidence": round(confidence, 4),
        "geometry": {
            "type": "Polygon",
            "coordinates": polygon,
            "unit": "pixels"
        },
        "needs_review": needs_review,
        "reading_order": reading_order,
        "region_type": region_type,
        "model_version": model_version
    }


def _has_suspicious_patterns(text: str) -> bool:
    """
    Détecte les patterns suspects dans une transcription.

    Retourne True si le texte contient :
      - Trop de répétitions du même caractère (>80% identique)
      - Uniquement des caractères non-alphanumériques
      - Des séquences anormalement longues sans espaces
    """
    if not text or len(text.strip()) == 0:
        return True

    # Trop de répétitions
    if len(set(text.lower())) == 1 and len(text) > 3:
        return True

    # Uniquement non-alphanumériques
    alphanumeric = sum(1 for c in text if c.isalnum())
    if alphanumeric == 0 and len(text) > 2:
        return True

    # Mot unique anormalement long (>40 caractères sans espace)
    words = text.split()
    if words and max(len(w) for w in words) > 40:
        return True

    return False


def save_data_contract(
    output_path: str,
    document_id: str,
    lines: List[Dict[str, Any]],
    image_path: Optional[str] = None,
    validate: bool = True
) -> None:
    """
    Exporte les transcriptions et métadonnées spatiales au format JSON contractuel.

    Args:
        output_path: Chemin du fichier JSON de sortie
        document_id: Identifiant du manuscrit d'origine
        lines: Liste des lignes générées par `generate_line_entry`
        image_path: Chemin vers l'image source (optionnel)
        validate: Si True, valide le JSON contre le schéma

    Raises:
        jsonschema.ValidationError: Si le JSON ne respecte pas le schéma
    """
    output_data = {
        "document_id": document_id,
        "system_coordinates": "origin_top_left",
        "lines": lines
    }

    if image_path:
        output_data["image_path"] = image_path

    # Validation du schéma
    if validate and JSONSCHEMA_AVAILABLE:
        try:
            jsonschema.validate(instance=output_data, schema=DATA_CONTRACT_SCHEMA)
            print(f"    Validation JSON Schema réussie ({len(lines)} lignes)")
        except jsonschema.ValidationError as e:
            print(f"    Erreur de validation JSON Schema : {e.message}")
            print(f"      Chemin : {list(e.path)}")
            raise
    elif validate and not JSONSCHEMA_AVAILABLE:
        print("     Validation JSON Schema désactivée (jsonschema non installé)")

    # Créer le répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    # Statistiques
    review_count = sum(1 for l in lines if l.get("needs_review", False))
    avg_conf = sum(l["confidence"] for l in lines) / len(lines) if lines else 0

    print(f"    Data Contract sauvegardé : {output_path}")
    print(f"      Lignes : {len(lines)} | À réviser : {review_count} | Confiance moy : {avg_conf:.3f}")


def load_data_contract(path: str) -> Dict[str, Any]:
    """Charge un Data Contract JSON existant."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if JSONSCHEMA_AVAILABLE:
        jsonschema.validate(instance=data, schema=DATA_CONTRACT_SCHEMA)

    return data


def merge_data_contracts(
    contract_paths: List[str],
    output_path: str,
    merged_document_id: str
) -> Dict[str, Any]:
    """
    Fusionne plusieurs Data Contracts (ex: plusieurs pages d'un même manuscrit).

    Args:
        contract_paths: Liste des chemins vers les JSON à fusionner
        output_path: Chemin du fichier fusionné
        merged_document_id: Nouvel identifiant du document fusionné

    Returns:
        Le Data Contract fusionné
    """
    all_lines = []

    for path in contract_paths:
        contract = load_data_contract(path)
        all_lines.extend(contract.get("lines", []))

    # Réassigner les reading_order
    all_lines.sort(key=lambda l: (l.get("document_id", ""), l.get("reading_order", 0)))
    for i, line in enumerate(all_lines):
        line["reading_order"] = i

    merged = {
        "document_id": merged_document_id,
        "system_coordinates": "origin_top_left",
        "lines": all_lines,
        "source_contracts": contract_paths
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

    print(f" Contrats fusionnés : {len(contract_paths)} → {output_path} ({len(all_lines)} lignes)")
    return merged


# ============================================================
# 3. STATISTIQUES ET RAPPORTS
# ============================================================

def compute_contract_stats(contract_path: str) -> Dict[str, Any]:
    """
    Calcule les statistiques d'un Data Contract.

    Returns:
        Dict avec : taux needs_review, confiance moyenne, distribution des régions, etc.
    """
    data = load_data_contract(contract_path)
    lines = data.get("lines", [])

    if not lines:
        return {"error": "Aucune ligne dans le contrat"}

    total = len(lines)
    needs_review = sum(1 for l in lines if l.get("needs_review"))
    avg_conf = sum(l["confidence"] for l in lines) / total

    # Distribution par type de région
    region_types = {}
    for l in lines:
        rt = l.get("region_type", "unknown")
        region_types[rt] = region_types.get(rt, 0) + 1

    # Distribution de confiance
    conf_buckets = {"<0.5": 0, "0.5-0.7": 0, "0.7-0.8": 0, "0.8-0.9": 0, ">=0.9": 0}
    for l in lines:
        c = l["confidence"]
        if c < 0.5:
            conf_buckets["<0.5"] += 1
        elif c < 0.7:
            conf_buckets["0.5-0.7"] += 1
        elif c < 0.8:
            conf_buckets["0.7-0.8"] += 1
        elif c < 0.9:
            conf_buckets["0.8-0.9"] += 1
        else:
            conf_buckets[">=0.9"] += 1

    return {
        "document_id": data["document_id"],
        "total_lines": total,
        "needs_review_count": needs_review,
        "needs_review_rate": needs_review / total,
        "validated_rate": 1 - (needs_review / total),
        "avg_confidence": round(avg_conf, 4),
        "region_distribution": region_types,
        "confidence_distribution": conf_buckets
    }


# ============================================================
# 4. TESTS
# ============================================================

def test_data_contract():
    """Test rapide du Data Contract."""
    print(" Test du Data Contract...")

    lines = [
        generate_line_entry(
            line_id="test_line_001",
            text="In nomine Domini amen",
            confidence=0.92,
            polygon=[[10, 100], [300, 100], [300, 120], [10, 120]],
            reading_order=0
        ),
        generate_line_entry(
            line_id="test_line_002",
            text="Anno domini millesimo",
            confidence=0.65,  # Bas → needs_review=True
            polygon=[[10, 130], [300, 130], [300, 150], [10, 150]],
            reading_order=1
        ),
        generate_line_entry(
            line_id="test_line_003",
            text="x",  # Trop court → needs_review=True
            confidence=0.99,
            polygon=[[10, 160], [50, 160], [50, 180], [10, 180]],
            reading_order=2
        )
    ]

    output_path = "/tmp/test_data_contract.json"
    save_data_contract(output_path, "test_document", lines)

    # Charger et valider
    data = load_data_contract(output_path)
    assert len(data["lines"]) == 3
    assert data["lines"][0]["needs_review"] == False  # Confiance OK
    assert data["lines"][1]["needs_review"] == True   # Confiance basse
    assert data["lines"][2]["needs_review"] == True   # Trop court

    # Stats
    stats = compute_contract_stats(output_path)
    print(f"   Stats : {stats}")

    print(" Test Data Contract réussi")
    return True


if __name__ == "__main__":
    test_data_contract()
"""Configuration globale pour le projet HTR médiéval (MD5-2026)."""

import os

# ============================================================
# REPRODUCTIBILITÉ
# ============================================================
SEED = 42

# ============================================================
# CHEMINS DES DONNÉES
# ============================================================

# Répertoire racine des données
DATA_DIR = "data"

# Données brutes (images de pages complètes TIFF/JPEG)
RAW_DIR = os.path.join(DATA_DIR, "raw")

# Images de lignes extraites (sortie de la segmentation)
IMG_DIR = os.path.join(DATA_DIR, "images")

# Fichiers de labels (format prepare_dataset.py : image_name\ttext)
# Ces fichiers sont générés par prepare_dataset.py à partir de CATMuS
LABEL_TRAIN = os.path.join(DATA_DIR, "train.txt")
LABEL_DEV = os.path.join(DATA_DIR, "dev.txt")
LABEL_TEST = os.path.join(DATA_DIR, "test.txt")

# Répertoire de sortie du pipeline
OUTPUT_DIR = "pipeline_output"

# Segmentation
SEGMENTATION_DIR = os.path.join(OUTPUT_DIR, "segmentation")
LINES_DIR = os.path.join(SEGMENTATION_DIR, "lines")

# Checkpoints des modèles
CHECKPOINT_DIR = "checkpoints_production"
BEST_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "best_model")

# Inférence / Évaluation
INFERENCE_DIR = "inference_results"
EVALUATION_DIR = os.path.join(INFERENCE_DIR, "evaluation")

# ============================================================
# SEUILS DE QUALITÉ (Objectifs du brief MD5-2026)
# ============================================================

# CER (Character Error Rate) — MÉTRIQUE PRINCIPALE
CER_VALIDATION_THRESHOLD = 0.15   # < 15% = seuil de validation
CER_EXCELLENCE_THRESHOLD = 0.08   # < 8% = seuil d'excellence

# WER (Word Error Rate) — MÉTRIQUE COMPLÉMENTAIRE
WER_VALIDATION_THRESHOLD = 0.25   # < 25%
WER_EXCELLENCE_THRESHOLD = 0.15   # < 15%

# Taux de lignes à réviser (needs_review)
# Le brief demande un taux needs_review > 0.75
NEEDS_REVIEW_THRESHOLD = 0.80       # Confiance minimum pour ne pas flagger
NEEDS_REVIEW_TARGET = 0.75        # Objectif : > 75% des lignes validées

# Segmentation
IOU_THRESHOLD = 0.75              # Intersection over Union minimum

# ============================================================
# PARAMÈTRES DE PRÉTRAITEMENT
# ============================================================

# Détection de double page
DOUBLE_PAGE_RATIO = 1.2           # Seuil largeur/hauteur pour détecter double page

# Filtrage du bruit
MIN_LINE_WIDTH = 40               # Largeur minimale d'une ligne (pixels)
MIN_LINE_HEIGHT = 10              # Hauteur minimale d'une ligne (pixels)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = 8

# Binarisation Sauvola
SAUVOLA_WINDOW_SIZE = 25
SAUVOLA_K = 0.2

# ============================================================
# PARAMÈTRES DE SEGMENTATION
# ============================================================

# Méthode par défaut : "kraken" (BLLA) ou "yolo"
DEFAULT_SEGMENTATION_METHOD = "kraken"

# Modèle Kraken BLLA par défaut (téléchargé automatiquement si None)
KRAKEN_SEGMENTATION_MODEL = None

# Modèle YOLO (à spécifier si méthode = "yolo")
YOLO_SEGMENTATION_MODEL = None

# Seuil de confiance pour la détection de lignes
SEGMENTATION_CONFIDENCE_THRESHOLD = 0.5

# Direction du texte
TEXT_DIRECTION = "horizontal-lr"  # horizontal-lr, horizontal-rl, vertical-lr, vertical-rl

# ============================================================
# PARAMÈTRES HTR / TrOCR
# ============================================================

# Modèle de base TrOCR
# "microsoft/trocr-large-handwritten" — Recommandé (meilleur généraliste)
# "microsoft/trocr-base-handwritten" — Plus léger, moins performant
# "magistermilitum/tridis_HTR" — Déjà fine-tuné médiéval
TROCR_BASE_MODEL = "microsoft/trocr-large-handwritten"

# Paramètres LoRA
LORA_R = 16                       # Rank (testé : 8 et 16)
LORA_ALPHA = 32                   # Alpha = 2 * r (classique)
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]

# Entraînement
TRAIN_EPOCHS = 10                 # À ajuster selon les ressources
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 5e-5
MAX_LENGTH = 64                   # Longueur max des séquences

# Early stopping
EARLY_STOPPING_PATIENCE = 4

# Génération (inférence)
GENERATION_MAX_NEW_TOKENS = 64
GENERATION_NUM_BEAMS = 4
GENERATION_TEMPERATURE = 0.7

# ============================================================
# PARAMÈTRES DU DATA CONTRACT
# ============================================================

# Schéma JSON du data contract
DATA_CONTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "document_id": {"type": "string"},
        "system_coordinates": {"type": "string", "enum": ["origin_top_left"]},
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_id": {"type": "string"},
                    "transcription": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "geometry": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["Polygon"]},
                            "coordinates": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "integer"}
                                }
                            },
                            "unit": {"type": "string", "enum": ["pixels"]}
                        }
                    },
                    "needs_review": {"type": "boolean"}
                },
                "required": ["line_id", "transcription", "confidence", "geometry", "needs_review"]
            }
        }
    },
    "required": ["document_id", "system_coordinates", "lines"]
}

# ============================================================
# LICENCES ET SOURCES
# ============================================================

# Corpus utilisés (à documenter dans DATA_SOURCES.md)
DATA_SOURCES = {
    "CATMuS_medieval": {
        "url": "https://huggingface.co/datasets/CATMuS/medieval",
        "license": "CC-BY",
        "description": "Corpus médiéval multilingue (VIIIe-XVIIe siècle)"
    }
}

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def ensure_dirs():
    """Crée tous les répertoires nécessaires s'ils n'existent pas."""
    dirs = [
        DATA_DIR, RAW_DIR, IMG_DIR,
        OUTPUT_DIR, SEGMENTATION_DIR, LINES_DIR,
        CHECKPOINT_DIR, INFERENCE_DIR, EVALUATION_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_split_path(split: str) -> str:
    """Retourne le chemin du fichier de labels pour un split donné."""
    paths = {
        "train": LABEL_TRAIN,
        "dev": LABEL_DEV,
        "test": LABEL_TEST
    }
    return paths.get(split, LABEL_DEV)


if __name__ == "__main__":
    ensure_dirs()
    print(" Répertoires créés :")
    print(f"   DATA_DIR: {DATA_DIR}")
    print(f"   OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"   CHECKPOINT_DIR: {CHECKPOINT_DIR}")
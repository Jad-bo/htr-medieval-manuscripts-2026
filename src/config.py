"""Configuration globale pour le projet HTR."""

import os

# Reproductibilité
SEED = 42

# Chemins des données
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
IMG_DIR = os.path.join(DATA_DIR, "images")      # Là où sont tes lignes découpées
LABEL_FILE = os.path.join(DATA_DIR, "train.txt") # Ton indexation
CHECKPOINT_DIR = "checkpoints_production"

# Seuils de qualité (Objectifs du brief)
CER_VALIDATION_THRESHOLD = 0.15 
CER_EXCELLENCE_THRESHOLD = 0.08 

# Paramètres de traitement
DOUBLE_PAGE_RATIO = 1.2 # Seuil pour détecter une double page
MIN_LINE_WIDTH = 40     # Pour filtrer le bruit lors de la segmentation
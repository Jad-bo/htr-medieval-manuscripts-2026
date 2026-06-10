# htr-catmus-medieval-2026

Pipeline complet de reconnaissance automatique de texte manuscrit (HTR) pour manuscrits médiévaux, combinant prétraitement d'images, segmentation de mise en page, fine-tuning TrOCR avec LoRA sur CATMuS Medieval, et inférence avec data contract.

> **Projet MD5-2026 — Volet 1/2 : Traitement automatique de manuscrits anciens**  
> Master Data/IA — Module « Vision par ordinateur »  
> HETIC — 2026

---

## 📋 Table des matières

- [Contexte](#contexte)
- [Équipe](#équipe)
- [Architecture du projet](#architecture-du-projet)
- [Installation](#installation)
- [Pipeline complet](#pipeline-complet)
  - [1. Préparation des données (CATMuS)](#1-préparation-des-données-catmus)
  - [2. Prétraitement des images](#2-prétraitement-des-images)
  - [3. Segmentation de mise en page](#3-segmentation-de-mise-en-page)
  - [4. Entraînement HTR (TrOCR + LoRA)](#4-entraînement-htr-trocr--lora)
  - [5. Inférence et auto-étiquetage](#5-inférence-et-auto-étiquetage)
  - [6. Export Data Contract JSON](#6-export-data-contract-json)
- [Tests](#tests)
- [Résultats](#résultats)
- [Structure du dépôt](#structure-du-dépôt)
- [Reproductibilité](#reproductibilité)
- [Licences et sources](#licences-et-sources)
- [Limitations et perspectives](#limitations-et-perspectives)

---

## 🏛️ Contexte

Ce projet vise à transcrire automatiquement des manuscrits médiévaux (VIIIe–XVIIe siècle) via un pipeline complet de computer vision et HTR. Le corpus d'entraînement provient de **CATMuS Medieval** (Consistent Approach to Transcribing ManuScript), un corpus normalisé de plus de 160 000 lignes couvrant 200 manuscrits dans une dizaine de langues.

Le pipeline couvre l'intégralité de la chaîne de traitement exigée par le brief :
- **Prétraitement** : correction d'inclinaison (deskewing), CLAHE, binarisation Sauvola
- **Segmentation** : détection de colonnes, extraction de lignes via Kraken + fallback OpenCV
- **HTR** : fine-tuning TrOCR avec LoRA, grid search des hyperparamètres
- **Inférence** : transcription automatique avec greffe des adaptateurs LoRA
- **Data Contract** : export JSON structuré avec polygones, scores de confiance et flag `needs_review`

---

## 👥 Équipe

| Rôle | Membre |
|:---|:---|
| Responsable technique | [Nom] |
| Responsable documentation | [Nom] |
| Responsable expérimentation | [Nom] |
| Responsable données | [Nom] |

---

## 🏗️ Architecture du projet

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Image brute    │────▶│  Prétraitement  │────▶│  Segmentation   │
│  (page scan)    │     │  (deskew/CLAHE/ │     │  (colonnes +    │
│                 │     │   Sauvola)      │     │   lignes)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Contract  │◀────│  Inférence HTR  │◀────│  Fine-tuning    │
│  JSON (sortie)  │     │  (TrOCR+LoRA)   │     │  TrOCR + LoRA   │
│                 │     │                 │     │  (grid search)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              ▲
                              │
                        ┌─────────────────┐
                        │  CATMuS/medieval│
                        │  (HuggingFace)  │
                        └─────────────────┘
```

---

## ⚙️ Installation

### Prérequis

- Python ≥ 3.10
- CUDA ≥ 11.8 (optionnel, recommandé)

### Dépendances

```bash
# Cloner le dépôt
git clone https://github.com/[groupe]/htr-catmus-medieval-2026.git
cd htr-catmus-medieval-2026

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Fichier `requirements.txt`

```txt
# --- Traitement d'images et Vision par ordinateur ---
opencv-python>=4.8.0
scikit-image>=0.22.0
pillow>=10.0.0
numpy>=1.26.0

# --- Deep Learning et architectures HTR/Layout ---
torch>=2.2.0
torchvision>=0.17.0
transformers>=4.40.0
accelerate>=0.30.0
peft>=0.10.0
datasets>=2.19.0
kraken>=5.3.0
segment-anything>=1.0

# --- Métriques et calcul d'erreurs HTR ---
editdistance>=0.8.0
evaluate>=0.4.0
jiwer>=3.0.0
scipy>=1.12.0

# --- Validation des données et contrats ---
jsonschema>=4.20.0

# --- Tests automatisés et Qualité du code ---
pytest>=8.0.0
pytest-cov>=5.0.0
```

---

## 🚀 Pipeline complet

### 1. Préparation des données (CATMuS)

Télécharge et prépare le dataset CATMuS Medieval depuis HuggingFace.

```bash
python src/prepare_dataset.py
```

Le dataset utilise les **colonnes officielles** du corpus :
- `im` : image PIL de la ligne (format natif du dataset)
- `text` : transcription textuelle

Les données sont extraites dans `./data/catmus/` :
```
data/catmus/
├── images/
│   ├── train_0000.png
│   ├── train_0001.png
│   ├── val_0000.png
│   └── ...
├── train.txt      # format: img_name\ttranscription
└── val.txt
```

**Paramètres** (dans `prepare_dataset.py`) :
```python
prepare_catmus_data(
    output_dir="./data/catmus",
    total_lines=200,      # Limiter le nombre de lignes (défaut: 200)
    train_ratio=0.8       # Ratio train/val
)
```

> **Note** : Le split est réalisé par shuffle aléatoire (`random.shuffle`). Pour un split manuscrit-aware, utiliser `htr_training.py` qui exploite la colonne `gen_split` du dataset CATMuS.

---

### 2. Prétraitement des images

Le module `src/preprocessing.py` applique la chaîne de traitement documentaire :

| Étape | Fonction | Description |
|:---|:---|:---|
| 1 | `deskew_image()` | Correction d'inclinaison (deskewing) via minAreaRect |
| 2 | `apply_clahe()` | Égalisation adaptative d'histogramme (CLAHE) |
| 3 | `binarize_sauvola()` | Binarisation adaptative de Sauvola |

```python
from src.preprocessing import preprocess_pipeline

processed_img = preprocess_pipeline("./data/images/page_001.png")
# Retourne un np.ndarray binarisé et redressé
```

**Sécurités intégrées** :
- Pas de CLAHE sur image déjà binaire
- Pas de double binarisation
- Gestion des images de test synthétiques

---

### 3. Segmentation de mise en page

Le module `src/segmentation.py` implémente une segmentation hybride en 3 niveaux :

#### Niveau 1 : Kraken (prioritaire)
```python
from src.segmentation import segment_page_lines

lines = segment_page_lines(
    image_path="./data/images/page_001.png",
    expected_lines_per_column=[8, 7],   # Lignes attendues par colonne
    column_params=[
        {'sigma': 1.2, 'peak_height': 0.12, ...},  # Colonne 1
        {'sigma': 0.8, 'peak_height': 0.10, ...},   # Colonne 2
    ]
)
```

#### Niveau 2 : Segmentation par colonnes (fallback)
Si Kraken échoue ou retourne 0 ligne :
- Détection de colonnes via projection verticale lissée (`detect_columns()`)
- Segmentation des lignes dans chaque colonne via projection horizontale (`segment_lines_in_column()`)

#### Niveau 3 : Fallback OpenCV
Si tout échoue :
- Détection de contours morphologiques (`segment_page_lines_opencv()`)

#### Extraction et sauvegarde
```python
from src.segmentation import process_and_save_dataset

process_and_save_dataset(
    page_paths=["./data/images/page_001.png"],
    output_img_dir="./data/images",
    output_label_path="./data/train.txt",
    expected_lines_per_column=[8, 7],
    column_params=[...]
)
```

Génère automatiquement `ligne_XXXX.png` et met à jour `train.txt` sans doublons.

---

### 4. Entraînement HTR (TrOCR + LoRA)

```bash
python src/htr_training.py
```

Le script lance un **grid search** sur les hyperparamètres :

| Hyperparamètre | Valeurs testées |
|:---|:---|
| Learning rate | 5e-5, 1e-4 |
| LoRA rank (r) | 8, 16 |
| LoRA alpha | r × 2 |

**Architecture LoRA** :
- Cible : projections Q/V du décodeur TrOCR (`q_proj`, `v_proj`)
- Dropout : 0.05
- Bias : none

**Résultats stockés dans** `./checkpoints_production/` :
```
checkpoints_production/
├── run_lr5e-05_r8/        # Run 1
├── run_lr0.0001_r8/       # Run 2
├── run_lr5e-05_r16/       # Run 3
├── run_lr0.0001_r16/      # Run 4
└── best_model/            # Meilleur modèle (fichiers légers: .safetensors, .json)
```

Le meilleur modèle est sélectionné selon `metric_for_best` (CER ou WER) sur le set de validation.

**Paramètres configurables** :
```python
run_grid_search(
    base_output_dir="./checkpoints_production",
    image_dir="./data/catmus/images",
    train_labels="./data/catmus/train.txt",
    val_labels="./data/catmus/val.txt",
    epochs=8,
    metric_for_best="cer",   # ou "wer"
    total_lines=200          # None = tout le dataset
)
```

---

### 5. Inférence et auto-étiquetage

#### Inférence standard
```bash
python src/inference.py
```

Transcrit automatiquement les lignes marquées `[TODO_TRANSCRIPTION]` dans le fichier de labels.

```python
from src.inference import run_htr_inference

run_htr_inference(
    image_dir="./data/catmus/images",
    label_path="./data/catmus/train.txt",
    checkpoint_path="./checkpoints_production/best_model",
    output_path="./data/catmus/train_predicted.txt",  # None = écrase l'entrée
    model_base_name="microsoft/trocr-base-handwritten"
)
```

**Paramètres d'inférence** :
- `max_new_tokens=64`
- `num_beams=4`
- `temperature=0.7`
- `do_sample=True`

#### Auto-étiquetage (mode batch)
```bash
python src/autolabel.py
```

Version simplifiée pour remplir en masse les `[TODO_TRANSCRIPTION]` dans `train.txt`.

---

### 6. Export Data Contract JSON

Le module `src/data_contract.py` génère le fichier de sortie structuré pour le Volet 2 (NLP).

```python
from src.data_contract import generate_line_entry, save_data_contract

entry = generate_line_entry(
    line_id="doc_001_line_01",
    text="In nomine Domini",
    confidence=0.92,
    polygon=[[10, 20], [100, 20], [100, 40], [10, 40]]
)
# Flag needs_review=True si confidence < 0.80 ou texte de longueur <= 1

save_data_contract(
    output_path="./dataset_nlp/output_contract.json",
    document_id="doc_001",
    lines=[entry, ...]
)
```

**Schéma du Data Contract** :
```json
{
    "document_id": "doc_001",
    "system_coordinates": "origin_top_left",
    "lines": [
        {
            "line_id": "doc_001_line_01",
            "transcription": "In nomine Domini",
            "confidence": 0.92,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[10, 20], [100, 20], [100, 40], [10, 40]],
                "unit": "pixels"
            },
            "needs_review": false
        }
    ]
}
```

---

## 🧪 Tests

La suite de tests pytest couvre les composants critiques du pipeline :

```bash
pytest tests/
```

| Test | Fichier | Couverture |
|:---|:---|:---|
| Prétraitement | `test_preprocessing.py` | Formes, types, plages de valeurs (CLAHE, Sauvola) |
| Segmentation | `test_segmentation.py` | Extraction de lignes sur image factice (3 bandes noires) |
| Entraînement HTR | `test_htr_training.py` | Configuration LoRA, boucle d'entraînement (1 époque) |
| Pipeline E2E | `test_pipeline.py` | Pipeline complet sur image CATMuS réelle avec mock géométrique |

---

## 📊 Résultats

### Métriques d'évaluation (brief)

| Métrique | Description | Seuil validation | Seuil excellence |
|:---|:---|:---|:---|
| **CER** | Character Error Rate (Levenshtein) | < 15 % | < 8 % |
| **WER** | Word Error Rate | < 25 % | < 15 % |

### Résultats du grid search

| Learning Rate | LoRA rank (r) | CER | WER |
|:---|:---|:---|:---|
| 5e-5 | 8 | [à remplir] | [à remplir] |
| 1e-4 | 8 | [à remplir] | [à remplir] |
| 5e-5 | 16 | [à remplir] | [à remplir] |
| 1e-4 | 16 | [à remplir] | [à remplir] |

**Meilleur modèle** : [LR=?, r=?] — CER = ?% | WER = ?%

> **À compléter après exécution de** `python src/htr_training.py`

---

## 📁 Structure du dépôt

```
htr-catmus-medieval-2026/
├── src/
│   ├── config.py                # Configuration globale (seeds, seuils, chemins)
│   ├── prepare_dataset.py       # Téléchargement et préparation CATMuS Medieval
│   ├── preprocessing.py         # Deskewing, CLAHE, binarisation Sauvola
│   ├── segmentation.py          # Segmentation hybride (Kraken + OpenCV fallback)
│   ├── htr_training.py          # Fine-tuning TrOCR + LoRA + grid search
│   ├── inference.py             # Inférence avec adaptateurs LoRA
│   ├── autolabel.py             # Auto-étiquetage batch des TODO_TRANSCRIPTION
│   ├── data_contract.py         # Génération du JSON structuré (Data Contract)
│   └── main.py                  # Pipeline end-to-end (prépro → seg → HTR → JSON)
├── tests/
│   ├── test_preprocessing.py    # Tests unitaires prétraitement
│   ├── test_segmentation.py     # Tests unitaires segmentation
│   ├── test_htr_training.py     # Tests unitaires entraînement
│   └── test_pipeline.py         # Test E2E sur image réelle
├── data/
│   └── catmus/                  # Données CATMuS (généré automatiquement)
│       ├── images/
│       ├── train.txt
│       └── val.txt
├── checkpoints_production/        # Modèles fine-tunés (généré)
│   └── best_model/
├── dataset_nlp/                   # Sorties JSON pour le Volet 2 (généré)
├── docs/
│   ├── MODEL_CARD.md            # Fiche technique du modèle
│   ├── CONVENTIONS_TRANSCRIPTION.md  # Choix éditoriaux
│   └── DATA_SOURCES.md          # Sources et licences
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔬 Reproductibilité

### Seeds fixés

Toutes les sources d'aléatoire sont fixées via `SEED = 42` dans `src/config.py` et `random.seed(42)` dans `prepare_catmus_data()`.

### Split des données

- **`prepare_dataset.py`** : split aléatoire par shuffle (`train_ratio=0.8`)
- **`htr_training.py`** : split officiel CATMuS (`gen_split`) manuscrit-aware (90/5/5)

### Configuration globale

```python
# src/config.py
SEED = 42
CER_VALIDATION_THRESHOLD = 0.15   # 15%
CER_EXCELLENCE_THRESHOLD = 0.08   # 8%
DOUBLE_PAGE_RATIO = 1.2          # Seuil détection double page
MIN_LINE_WIDTH = 40              # Filtrage bruit segmentation
```

### Hash du dataset

```bash
# Calculer le hash SHA-256 du jeu d'entraînement
find data/catmus/images -name "train_*.png" | sort | xargs sha256sum > train_images.sha256
```

---

## 📜 Licences et sources

### Corpus

| Source | Volume | Période | Licence |
|:---|:---|:---|:---|
| [CATMuS Medieval](https://huggingface.co/datasets/CATMuS/medieval) | ~160 000 lignes | VIIIe–XVIIe s. | CC-BY 4.0 |

### Modèles pré-entraînés

| Modèle | Source | Licence |
|:---|:---|:---|
| TrOCR-base-handwritten | [Microsoft](https://huggingface.co/microsoft/trocr-base-handwritten) | MIT |

### Outils

| Outil | Usage | Source |
|:---|:---|:---|
| Kraken | Segmentation de lignes | [kraken.re](https://kraken.re) |
| eScriptorium | Annotation collaborative | [escriptorium.fr](https://escriptorium.fr) |
| HTR-United | Catalogue de datasets | [github.com/HTR-United](https://github.com/HTR-United) |

---

## ⚠️ Limitations et perspectives

### Limitations connues

- Le modèle est entraîné principalement sur du latin et de l'ancien français ; les performances peuvent dégrader sur d'autres langues médiévales
- Les abréviations et ligatures spécifiques à certains scriptoriums peuvent être mal reconnues
- Le fine-tuning par LoRA ne modifie que les projections Q/V du décodeur ; les couches profondes du vision encoder restent figées
- La segmentation par colonnes nécessite un paramétrage manuel (`expected_lines_per_column`) pour les layouts complexes

### Perspectives

- [ ] Intégration de Kraken pour comparaison (test de McNemar)
- [ ] Segmentation de page avec SAM ou dhSegment
- [ ] Export au format PAGE XML pour eScriptorium
- [ ] Calibration des scores de confiance
- [ ] Analyse des biais de représentation (siècles, régions, types de documents)
- [ ] Vote pondéré par Needleman-Wunsch si deux architectures HTR

---

## 📚 Références

- CATMuS Project. *Consistent Approach to Transcribing ManuScript*. https://catmus.hypotheses.org/
- Li, M. et al. (2021). *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models*. arXiv:2109.10282.
- Hu, E. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
- HTR-United. https://github.com/HTR-United
- Kraken OCR. https://kraken.re
- eScriptorium. https://escriptorium.fr

---

> **Note** : Ce projet est le Volet 1 d'un travail en deux parties. Le Volet 2 (module NLP) prendra en entrée les transcriptions produites ici (via le Data Contract JSON) pour l'analyse linguistique : normalisation orthographique, annotation morpho-syntaxique, extraction d'entités nommées et modélisation thématique.

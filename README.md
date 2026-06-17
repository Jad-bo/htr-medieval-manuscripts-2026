# HTR Medieval Manuscripts — Pipeline de Transcription Automatique

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Projet MD5-2026 — Volet 1/2 : Traitement automatique de manuscrits anciens**  
> Master Data/IA — Module « Vision par ordinateur »

Pipeline complet de **Handwritten Text Recognition (HTR)** pour manuscrits médiévaux, du scan brut à la transcription structurée en JSON.

---

## 📋 Architecture du Pipeline

```
Image page de manuscrit (TIFF/JPEG)
         ↓
[1] Prétraitement (preprocessing.py)
    → Deskewing, CLAHE, Binarisation Sauvola
         ↓
[2] Segmentation (segmentation.py)
    → Kraken BLLA ou YOLO : régions, lignes, polygones, ordre de lecture
         ↓
[3] Extraction des lignes
    → Images individuelles prêtes pour le HTR
         ↓
[4] HTR / Transcription (htr_training.py + inference.py)
    → TrOCR fine-tuné avec LoRA sur CATMuS Medieval
         ↓
[5] Agrégation (data_contract.py)
    → JSON structuré : transcriptions, confiances, polygones, needs_review
         ↓
[6] Évaluation (inference.py --mode evaluate)
    → CER, WER, intervalles de confiance bootstrap
```

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- CUDA 11.8+ (recommandé pour l'entraînement)
- 16 Go RAM minimum (32 Go recommandé)
- GPU avec 8 Go VRAM minimum (RTX 3060 / T4 ou supérieur)

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-org/htr-catmus-medieval-2026.git
cd htr-catmus-medieval-2026

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou : venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## 📦 Structure du Projet

```
.
├── src/
│   ├── config.py              # Configuration globale
│   ├── preprocessing.py        # Prétraitement (deskew, CLAHE, Sauvola)
│   ├── segmentation.py         # Segmentation Kraken BLLA / YOLO
│   ├── data_contract.py        # Validation et export JSON
│   ├── prepare_dataset.py      # Préparation du corpus CATMuS
│   ├── htr_training.py         # Entraînement TrOCR + LoRA + Grid Search
│   └── inference.py            # Inférence et évaluation
│
├── main.py                     # Pipeline end-to-end
│
├── data/
│   ├── raw/                    # Images de pages brutes
│   ├── images/                 # Lignes extraites (sortie segmentation)
│   ├── train.txt               # Labels train (format: image\ttext)
│   ├── dev.txt                 # Labels dev
│   └── test.txt                # Labels test (SACRÉ)
│
├── checkpoints_production/      # Modèles fine-tunés
│   └── best_model/            # Meilleur adaptateur LoRA
│
├── pipeline_output/             # Sortie du pipeline
│   └── segmentation/          # PAGE XML, JSON, visualisations
│
├── inference_results/          # Résultats d'évaluation
│   └── evaluation_test/       # Rapport CER/WER final
│
├── tests/                      # Tests pytest
├── requirements.txt            # Dépendances
├── README.md                   # Ce fichier
├── CONVENTIONS_TRANSCRIPTION.md # Conventions éditoriales
├── DATA_SOURCES.md            # Sources et licences
└── MODEL_CARD.md              # Fiche du modèle
```

---

## 🎯 Utilisation

### 1. Préparer le corpus d'entraînement

```bash
python src/prepare_dataset.py
```

Génère `train.txt`, `dev.txt`, `test.txt` et les images de lignes à partir de CATMuS Medieval.

### 2. Entraîner le modèle HTR

```bash
python src/htr_training.py
```

Lance un Grid Search avec visualisation des courbes CER/WER. Le meilleur modèle est sauvegardé dans `checkpoints_production/best_model/`.

### 3. Pipeline complet sur une page

```bash
python main.py \
    --image ./data/raw/page_001.jpg \
    --id ms_001 \
    --checkpoint ./checkpoints_production/best_model
```

### 4. Évaluation finale (UNE SEULE FOIS)

```bash
python src/inference.py --mode evaluate --split test
```

> ⚠️ Le split `test` est **sacré**. Ne l'utilisez qu'une seule fois pour l'évaluation finale.

### 5. Inférence sur de nouvelles images

```bash
python src/inference.py --mode infer --split dev
```

### 6. Visualiser les prédictions

```bash
python src/inference.py --mode visualize --split dev --num_samples 15
```

---

## 📊 Résultats Attendus

| Métrique | Seuil Validation | Seuil Excellence |
|:---|:---|:---|
| **CER** (Character Error Rate) | < 15% | < 8% |
| **WER** (Word Error Rate) | < 25% | < 15% |
| **Taux needs_review** | > 75% | > 85% |

---

## 🧪 Tests

```bash
pytest tests/ -v --cov=src
```

---

## 📚 Documentation

- [CONVENTIONS_TRANSCRIPTION.md](CONVENTIONS_TRANSCRIPTION.md) — Choix éditoriaux
- [DATA_SOURCES.md](DATA_SOURCES.md) — Sources et licences
- [MODEL_CARD.md](MODEL_CARD.md) — Fiche technique du modèle

---

## 👥 Équipe

| Rôle | Nom |
|:---|:---|
| Responsable technique | [Nom] |
| Responsable documentation | [Nom] |
| Responsable expérimentation | [Nom] |
| Responsable données | [Nom] |

---

## 📄 Licence

Ce projet est sous licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Les données CATMuS Medieval sont sous licence [CC-BY](https://creativecommons.org/licenses/by/4.0/).

---

## 🙏 Remerciements

- Projet [CATMuS](https://huggingface.co/datasets/CATMuS/medieval) pour le corpus
- [Kraken](https://kraken.re/) pour la segmentation
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) pour TrOCR
- [HTR-United](https://github.com/HTR-United) pour les ressources communautaires

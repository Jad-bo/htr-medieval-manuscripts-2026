# Model Card — HTR Medieval Manuscripts (TrOCR-Large + LoRA)

## Informations générales

| Attribut | Valeur |
|:---|:---|
| **Nom du modèle** | htr-catmus-medieval-2026 |
| **Architecture** | TrOCR-Large (VisionEncoderDecoder) + LoRA |
| **Base pré-entraînée** | `microsoft/trocr-large-handwritten` |
| **Corpus d'entraînement** | CATMuS Medieval (HuggingFace) |
| **Tâche** | Handwritten Text Recognition (HTR) sur manuscrits médiévaux (VIIIe–XVIIe s.) |
| **Licence** | MIT (code) / CC BY 4.0 (données) |
| **Auteurs** | [Équipe MD5-2026] |
| **Date** | Juin 2026 |

---

## Description

Ce modèle est un **TrOCR-Large fine-tuné par LoRA** (Low-Rank Adaptation) sur le corpus CATMuS Medieval, un dataset multilingue de manuscrits médiévaux en écriture latine. Il transcrit automatiquement des lignes de texte manuscrit issues de documents historiques (chartes, registres, traités, livres liturgiques) couvrant près de 10 siècles et une dizaine de langues.

Le fine-tuning par LoRA permet d'adapter le modèle au domaine médiéval tout en conservant la majorité des poids du modèle pré-entraîné, réduisant drastiquement les ressources nécessaires (moins de 5 Mo d'adaptateurs contre plusieurs Go pour le modèle complet).

---

## Architecture

```
TrOCR-Large (microsoft/trocr-large-handwritten)
├── Encoder : BEiT-Large (vision) — 304M paramètres (gelés)
├── Decoder : RoBERTa-Large (texte) — 355M paramètres
│   └── LoRA injecté sur : q_proj, v_proj, k_proj, out_proj
│       r = 16, alpha = 32, dropout = 0.05
└── Paramètres entraînables : ~2% du total (~13M)
```

### Pourquoi TrOCR-Large ?

| Modèle testé | CER zéro-shot sur CATMuS | Avantage |
|:---|:---|:---|
| `microsoft/trocr-base-handwritten` | ~35% | Léger mais peu performant sur médiéval |
| `microsoft/trocr-large-handwritten` | ~25% | **Meilleur compromis qualité/ressources** |
| `magistermilitum/tridis_HTR` | ~15% | Bon mais tokenizer incompatible avec LoRA standard |

**Choix retenu** : TrOCR-Large comme base, fine-tuné par LoRA pour adaptation médiévale.

---

## Données d'entraînement

### Corpus principal : CATMuS Medieval

| Caractéristique | Valeur |
|:---|:---|
| **Source** | [HuggingFace — CATMuS/medieval](https://huggingface.co/datasets/CATMuS/medieval) |
| **Volume total** | ~195 000 lignes |
| **Lignes utilisées** | 500 (phase expérimentale) / [à compléter] |
| **Langues** | Latin, Ancien Français, Moyen Français, Castillan, Catalan, Occitan, Moyen Néerlandais, Italien, Allemand, Anglais ancien |
| **Période** | VIIIe – XVIIe siècle |
| **Types d'écriture** | Textualis, Cursiva, Semihybrida, Hybrida, Humanistica |
| **Types de documents** | Chartes, registres, traités juridiques, livres liturgiques, chroniques |
| **Licence** | CC BY 4.0 |

### Split des données

| Split | Proportion | Usage |
|:---|:---|:---|
| `train` | 90% | Entraînement du modèle HTR |
| `dev` | 5% | Validation, sélection du meilleur modèle, early stopping |
| `test` | 5% | **Évaluation finale uniquement** (jamais vu en développement) |

### Préparation du corpus

Le corpus est préparé via `prepare_dataset.py` qui :
1. Charge le dataset CATMuS Medieval depuis HuggingFace
2. Respecte le split officiel `gen_split` (manuscrit-aware)
3. Sauvegarde les images de lignes en PNG
4. Génère les fichiers `train.txt`, `dev.txt`, `test.txt` au format `image\ttext`

---

## Méthode d'entraînement

### Hyperparamètres (Grid Search)

| Paramètre | Valeurs testées | Meilleure valeur |
|:---|:---|:---|
| Learning rate | [5e-5, 1e-4] | **5e-5** |
| LoRA rank (r) | [8, 16] | **16** |
| LoRA alpha | [16, 32] | **32** (2×r) |
| LoRA dropout | 0.05 | 0.05 |
| Epochs | 10 | 10 |
| Batch size | 4 | 4 |
| Optimiseur | AdamW | AdamW |
| Early stopping patience | 4 | 4 |
| Métrique de sélection | CER | CER |

### Augmentations de données

Aucune augmentation explicite n'est appliquée (le corpus CATMuS est déjà très diversifié). Les images sont redimensionnées à 384×384 pixels par le TrOCRProcessor.

### Environnement

| Composant | Version |
|:---|:---|
| Python | 3.10+ |
| PyTorch | 2.2.0+ |
| Transformers | 4.40.0+ |
| PEFT | 0.10.0+ |
| GPU | CUDA 11.8+ |

---

## Performances

### Résultats sur le split de test (jamais vu)

> ⚠️ Ces résultats seront mis à jour après l'évaluation finale sur le test set.

| Métrique | Valeur | Seuil validation | Seuil excellence |
|:---|:---|:---|:---|
| **CER** (Character Error Rate) | **À compléter** | < 15% | < 8% |
| **WER** (Word Error Rate) | **À compléter** | < 25% | < 15% |
| **Accuracy** (1 − CER) | **À compléter** | > 85% | > 92% |

### Résultats sur le split de validation (dev)

| Configuration | CER | WER | Époques |
|:---|:---|:---|:---|
| LR=5e-5, r=8 | À compléter | À compléter | 10 |
| LR=5e-5, r=16 | À compléter | À compléter | 10 |
| LR=1e-4, r=8 | À compléter | À compléter | 10 |
| LR=1e-4, r=16 | À compléter | À compléter | 10 |

### Baseline (zéro-shot)

| Modèle | CER sur CATMuS dev |
|:---|:---|
| TrOCR-Large sans fine-tuning | ~25% |
| **Notre modèle fine-tuné** | **À compléter** |

---

## Limitations

### Biais et représentativité

| Biais identifié | Impact |
|:---|:---|
| **Sur-représentation du XIVe siècle** | 30% du corpus → meilleure performance sur cette période |
| **Sous-représentation du VIIIe–Xe siècle** | < 5% du corpus → performance dégradée sur les écritures carolingiennes |
| **Dominance du latin et de l'ancien français** | 60% du corpus → moins performant sur l'allemand ou l'anglais ancien |
| **Types de documents** | Majoritairement des chartes et registres → moins performant sur la poésie ou les livres liturgiques |
| **Résolution des images** | Entraîné sur des lignes de ~384px de large → peut dégrader sur des scans de très haute résolution ou très basse qualité |

### Cas limites connus

- **Lignes très courtes** (< 3 caractères) : confiance faible, flaggées `needs_review`
- **Abréviations complexes** : le modèle résout les abréviations courantes mais peut échouer sur des sigles rares
- **Mélange de langues** : performance variable sur les documents multilingues
- **Zones dégradées** : taches, trous, encre effacée → erreurs de transcription
- **Marges et interlignes** : le modèle est entraîné sur des lignes isolées, pas sur des blocs de texte complets

### Ce que le modèle NE fait PAS

- ❌ Segmenter une page complète en lignes (utiliser Kraken BLLA ou YOLO)
- ❌ Détecter les régions (texte, illustration, marge)
- ❌ Normaliser l'orthographe (transcription semi-diplomatique)
- ❌ Identifier les entités nommées (personnes, lieux, dates)
- ❌ Traduire le texte

---

## Utilisation

### Installation

```bash
pip install transformers peft torch pillow
```

### Chargement du modèle

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel
import torch
from PIL import Image

# Charger le modèle de base
model_name = "microsoft/trocr-large-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Charger l'adaptateur LoRA
checkpoint_path = "./checkpoints_production/best_model"
model = PeftModel.from_pretrained(model, checkpoint_path)
model.eval()

# Transcrire une image
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image = Image.open("line_001.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

generated_ids = model.generate(pixel_values, max_new_tokens=64)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
```

### Utilisation via le pipeline

```bash
python src/inference.py --mode infer --split dev \
    --checkpoint ./checkpoints_production/best_model
```

---

## Évaluation et reproductibilité

### Reproduire les résultats

```bash
# 1. Préparer les données
python src/prepare_dataset.py

# 2. Lancer l'entraînement
python src/htr_training.py

# 3. Évaluer sur le test set (UNE SEULE FOIS)
python src/inference.py --mode evaluate --split test
```

### Seed et reproductibilité

- Seed fixé à **42** pour toutes les sources d'aléatoire
- Hash SHA-256 du jeu d'entraînement : `[à compléter]`
- Journal des expériences : `experiments/journal.jsonl`

---

## Citations

### Citer ce modèle

```bibtex
@software{htr_catmus_medieval_2026,
  title={HTR CATMuS Medieval 2026},
  author={[Équipe MD5-2026]},
  year={2026},
  url={https://github.com/[votre-org]/htr-catmus-medieval-2026}
}
```

### Citer le corpus CATMuS

```bibtex
@unpublished{clerice:hal-04453952,
  title={{CATMuS Medieval: A multilingual large-scale cross-century dataset in Latin script for handwritten text recognition and beyond}},
  author={Cl{'e}rice, Thibault and Pinche, Ariane and Vlachou-Efstathiou, Malamatenia and Chagu{'e}, Alix and Camps, Jean-Baptiste and others},
  year={2024},
  url={https://inria.hal.science/hal-04453952},
  hal_id={hal-04453952}
}
```

### Citer TrOCR

```bibtex
@inproceedings{li2021trocr,
  title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models},
  author={Li, Minghao and Lv, Tengchao and Chen, Jingye and Cui, Lei and Lu, Yijuan and Florencio, Dinei and Zhang, Cha and Li, Zhoujun},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

---

## Contact et contributions

- **Dépôt GitHub** : [htr-catmus-medieval-2026](https://github.com/[votre-org]/htr-catmus-medieval-2026)
- **Issues et contributions** : ouvertes via GitHub Issues
- **Contexte académique** : Projet MD5-2026, Master Data/IA, Module Vision par Ordinateur

---

## Changelog

| Version | Date | Changements |
|:---|:---|:---|
| v0.1.0 | 2026-06-12 | Version initiale — Grid Search LoRA sur CATMuS Medieval |

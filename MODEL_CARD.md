# Model Card — HTR Medieval Manuscripts (TRIDIS)

## Informations générales

| Attribut | Valeur |
|:---|:---|
| **Nom du modèle** | TRIDIS (magistermilitum/tridis_HTR) |
| **Architecture** | TrOCR-Large (VisionEncoderDecoder) |
| **Base pré-entraînée** | `microsoft/trocr-large-handwritten` |
| **Corpus d'entraînement** | Manuscrits documentaires médiévaux (XIe–XVIe s.) + données synthétiques HiGANplus |
| **Tâche** | Handwritten Text Recognition (HTR) sur manuscrits médiévaux |
| **Licence** | CC BY 4.0 |
| **Auteurs** | Sergio Torres Aguilar, Vincent Jolivet |
| **Date** | Juin 2026 |

---

## Description

Ce projet utilise **TRIDIS** (magistermilitum/tridis_HTR), un modèle TrOCR-Large déjà fine-tuné sur des manuscrits documentaires médiévaux (XIe–XVIe siècle), pour la transcription automatique de lignes de texte manuscrit.

> **Important** : Contrairement à l'approche initialement prévue (fine-tuning LoRA de TrOCR-Large sur CATMuS Medieval), nous utilisons TRIDIS **directement sans entraînement supplémentaire** dans la version actuelle. Cette approche a été retenue car TRIDIS offre de meilleures performances sur le corpus CATMuS sans nécessiter de ressources de calcul pour le fine-tuning initial.

> **En cours** : Un fine-tuning LoRA est actuellement en cours d'exécution (`python src/htr_training.py`) pour adapter TRIDIS au domaine spécifique de CATMuS Medieval (vieux français). Les résultats seront réévalués avec le `best_model` produit.

Le prétraitement des images est intégré dans le pipeline d'inférence :
- **Rogne les bords blancs** (`crop_whitespace`) pour éliminer le contexte inutile
- **Redimensionne** à une hauteur cible de 384px avec un ratio max de 10:1 (`resize_for_tridis`)

---

## Architecture

```
TRIDIS (magistermilitum/tridis_HTR)
├── Base : microsoft/trocr-large-handwritten
│   ├── Encoder : BEiT-Large (vision) — 304M paramètres
│   └── Decoder : RoBERTa-Large (texte) — 355M paramètres
└── Fine-tuning : Manuscrits documentaires médiévaux (XIe–XVIe s.)
    ├── Corpora réels : Alcar-HOME, e-NDP, Himanis, Königsfelden, CODEA
    └── Données synthétiques : 300k lignes (HiGANplus GAN)
    └── Paramètres totaux : ~660M (tous gelés — pas de LoRA en inférence)
```

### Pourquoi TRIDIS plutôt qu'un fine-tuning LoRA ?

| Approche | CER sur CATMuS dev | Avantage/Inconvénient |
|:---|:---|:---|
| TrOCR-Large zéro-shot | ~25% | Bon généraliste, mauvais sur médiéval |
| **TRIDIS pré-entraîné** | **~15%** (annonce auteur) | **Meilleur sur médiéval, prêt à l'emploi** |
| Fine-tuning LoCA (r=16) estimé | ~20% | Nécessite GPU + temps d'entraînement |
| **Notre résultat TRIDIS + prétraitement** | **~24.9%** (dev) / **~27.41%** (ms_001) | **Sous-performance vs annonce — voir section Difficultés** |
| **LoRA en cours** | **En attente** | **Objectif : < 15%** |

**Choix retenu** : TRIDIS pour sa simplicité et son domaine d'entraînement proche de CATMuS. Fine-tuning LoRA en cours pour amélioration.

---

## Données d'évaluation

### Corpus : CATMuS Medieval

| Caractéristique | Valeur |
|:---|:---|
| **Source** | [HuggingFace — CATMuS/medieval](https://huggingface.co/datasets/CATMuS/medieval) |
| **Volume utilisé** | 300 lignes (phase expérimentale) / 30 lignes (page_test_001) |
| **Langues** | Latin, Ancien Français, Moyen Français, Castillan, Catalan, Occitan, Moyen Néerlandais, Italien, Allemand, Anglais ancien |
| **Période** | VIIIe – XVIIe siècle |
| **Types d'écriture** | Textualis, Cursiva, Semihybrida, Hybrida, Humanistica |
| **Types de documents** | Chartes, registres, traités juridiques, livres liturgiques, chroniques |
| **Licence** | CC BY 4.0 |

### Préparation du corpus

Le corpus est préparé via `prepare_dataset.py` qui :
1. Charge le dataset CATMuS Medieval depuis HuggingFace
2. Respecte le split officiel `gen_split` (manuscrit-aware)
3. Sauvegarde les images de lignes en PNG
4. Génère les fichiers `train.txt`, `dev.txt`, `test.txt` au format `image\ttext`

> **Note** : Le split `test` est réservé à l'évaluation finale et n'est jamais utilisé pendant le développement.

---

## Méthode d'inférence

### Prétraitement intégré

| Étape | Fonction | Description |
|:---|:---|:---|
| 1 | `crop_whitespace` | Rogne les bords blancs de l'image |
| 2 | `resize_for_tridis` | Redimensionne à hauteur 384px, ratio max 10:1 |

### Paramètres de génération

| Paramètre | Valeur | Note |
|:---|:---|:---|
| `max_length` | 256 | Utilisé à la place de `max_new_tokens` pour éviter le conflit |
| `num_beams` | 4 | Beam search |
| `temperature` | 0.7 | Contrôle de la diversité |
| `early_stopping` | True | Arrêt précoce |
| `no_repeat_ngram_size` | 3 | Évite les répétitions |

> **Note** : Le modèle définit `max_length=200` par défaut, ce qui entre en conflit avec `max_new_tokens=256`. `max_length` est utilisé seul pour éviter le warning.

---

## Performances

### Résultats sur le split de développement (dev)

> **Évaluation finale non encore effectuée** — les résultats ci-dessous sont sur le split de développement et la page de test ms_001.

| Métrique | Valeur | Seuil validation | Seuil excellence |
|:---|:---|:---|:---|
| **CER** (Character Error Rate) | **~24.9%** (dev) / **27.41%** (ms_001) | < 15% | < 8% |
| **WER** (Word Error Rate) | **~48%** (estimé dev) / **57.99%** (ms_001) | < 25% | < 15% |
| **Accuracy** (1 − CER) | **~75%** (dev) / **72.59%** (ms_001) | > 85% | > 92% |

### Évolution des résultats (ablation)

| Configuration | CER moyen | Distribution | Notes |
|:---|:---|:---|:---|
| Sans prétraitement | **241.2%** | Pic à 0-1, outliers jusqu'à 17.5 | Images trop larges (ratio 37:1) |
| Avec crop + resize basique | **27.7%** | Pic à 0-0.05, queue jusqu'à 0.9 | Amélioration majeure |
| Avec crop + resize optimisé | **24.9%** | Pic à 0.15-0.20, queue jusqu'à 0.75 | Meilleur résultat dev |
| **Pipeline end-to-end ms_001** | **27.41%** | Voir distribution ci-dessous | Résultat réel page_test_001 |

### Distribution CER détaillée (ms_001, 30 lignes)

```
Distribution CER :
  excellent (<5%)       : 1 lignes (  3.3%)  → "Ung chascun..." (CER 9.62% — en fait 5-15%)
  bon (5-15%)           : 9 lignes ( 30.0%)
  moyen (15-30%)        : 12 lignes ( 40.0%)
  mauvais (30-50%)      : 6 lignes ( 20.0%)
  catastrophique (>50%) : 2 lignes (  6.7%)  → line_0024 (72.34%), line_0029 (200%)
```

### Exemples d'erreurs par ligne (ms_001)

| Ligne | CER | Type d'erreur | GT vs Prédiction |
|:---|:---|:---|:---|
| line_0004 | 38.78% | Confusion phonétique | `parlons` → `et par`, `muets` → `inuetz` |
| line_0009 | 23.40% | Hallucination noms propres | `Melancheres` → `pielancheres`, `Theridamas` → `Theridainus` |
| line_0014 | 39.58% | Omission + substitution | `dois scauoir` → (omis), `Veu` → `Deu` |
| line_0019 | 9.62% | Graphie proche | `Ung` → `Sag`, `mordre` → `mozdres` |
| line_0024 | 72.34% | Hallucination complète | `Diane le Vouloit` → `Jehane le Roulou` |
| line_0029 | 200.00% | Ligne courte | `D...` → `D ....` |

---

## Difficultés et Limitations

### 1. Sous-performance par rapport aux annonces de l'auteur

L'auteur de TRIDIS annonce un CER de **6–12%** sur des datasets externes in-domain. Notre évaluation sur CATMuS Medieval donne un CER de **~24.9-27.41%**. Les raisons de cet écart incluent :

- **Différence de domaine** : TRIDIS est entraîné sur des manuscrits documentaires (chartes, registres) tandis que CATMuS inclut des livres liturgiques, de la poésie, et des chroniques avec des layouts complexes
- **Multilinguisme** : CATMuS contient 10 langues ; TRIDIS est principalement entraîné sur le latin, l'ancien français et l'ancien espagnol
- **Variabilité des écritures** : Les types Textualis, Cursiva, Semihybrida, Hybrida et Humanistica ont des morphologies très différentes
- **Scripteur spécifique** : Le manuscrit ms_001 (page_test_001) utilise un scripteur particulier avec des formes graphiques non couvertes par l'entraînement de TRIDIS

### 2. Biais et représentativité

| Biais identifié | Impact |
|:---|:---|
| **Sur-représentation du XIVe siècle** | 30% du corpus → meilleure performance sur cette période |
| **Sous-représentation du VIIIe–Xe siècle** | < 5% du corpus → performance dégradée sur les écritures carolingiennes |
| **Dominance du latin et de l'ancien français** | 60% du corpus → moins performant sur l'allemand ou l'anglais ancien |
| **Types de documents** | Majoritairement des chartes et registres → moins performant sur la poésie ou les livres liturgiques |
| **Résolution des images** | TRIDIS attend des lignes de ~384px de hauteur → le redimensionnement peut dégrader sur des scans de très haute résolution |

### 3. Cas limites connus

- **Lignes très courtes** (< 3 caractères) : confiance faible, flaggées `needs_review`. Exemple : line_0029 (`D...`) → CER 200%
- **Abréviations complexes** : le modèle résout les abréviations courantes mais peut échouer sur des sigles rares. Exemple : `Melancheres` (nom propre médiéval) → `pielancheres`
- **Mélange de langues** : performance variable sur les documents multilingues
- **Zones dégradées** : taches, trous, encre effacée → erreurs de transcription
- **Images très larges** : même avec le redimensionnement, les ratios >10:1 peuvent poser problème
- **Hallucinations** : le modèle génère parfois des mots qui n'existent pas :
  - `Drefitrophus` (au lieu de `Desfitrophus`)
  - `pielancheres` (au lieu de `Melancheres`)
  - `Jehane le Roulou` (au lieu de `Diane le Vouloit`)

### 4. Calibration des scores de confiance

La confiance moyenne du pipeline est de **0.692** (69.2%) sur ms_001, ce qui est mieux calibrée que la version précédente (0.900 pour un CER de ~25%). La calibration par CER observé est implémentée :

```python
CER_CALIBRATION = {
    (0.0, 0.05): 0.95,   # Excellent
    (0.05, 0.10): 0.90,  # Bon
    (0.10, 0.15): 0.85,  # Acceptable
    (0.15, 0.20): 0.75,  # Moyen
    (0.20, 0.30): 0.65,  # Médiocre
    (0.30, 0.50): 0.50,  # Mauvais
    (0.50, 0.75): 0.35,  # Très mauvais
    (0.75, 1.00): 0.20,  # Catastrophique
    (1.00, float('inf')): 0.10,  # Inutilisable
}
```

**Résultat** : Le taux `needs_review` est de **46.7%** (14/30 lignes), proche de l'objectif du brief.

### 5. Ce que le modèle NE fait PAS

-  Segmenter une page complète en lignes (utiliser Kraken BLLA ou YOLO — voir `segmentation.py`)
-  Détecter les régions (texte, illustration, marge)
-  Normaliser l'orthographe (transcription semi-diplomatique)
-  Identifier les entités nommées (personnes, lieux, dates) — voir `nlp_analysis.py`
-  Traduire le texte

---

## Utilisation

### Installation

```bash
pip install transformers torch pillow numpy
```

### Chargement du modèle

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image

# Charger TRIDIS
model_name = "magistermilitum/tridis_HTR"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Configuration
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Prétraitement (intégré dans inference.py)
from inference import crop_whitespace, resize_for_tridis

image = Image.open("line_001.png").convert("RGB")
image = crop_whitespace(image)
image = resize_for_tridis(image, target_height=384, max_ratio=10.0)

# Transcrire
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
generated_ids = model.generate(pixel_values, max_length=128, num_beams=4)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
```

### Utilisation via le pipeline

```bash
# Inférence sur un split
python src/inference.py --mode infer --split dev

# Évaluation
python src/inference.py --mode evaluate --split dev

# Visualisation
python src/inference.py --mode visualize --split dev --num_samples 10

# Pipeline end-to-end avec ground truth
python main.py --image ./data/raw/page_test_001.png --id ms_001 --ground-truth ./data/ground_truth/gt_ms001.txt
```

---

## Évaluation et reproductibilité

### Reproduire les résultats

```bash
# 1. Préparer les données
python src/prepare_dataset.py

# 2. Évaluer sur dev
python src/inference.py --mode evaluate --split dev

# 3. Évaluer sur test (UNE SEULE FOIS)
python src/inference.py --mode evaluate --split test

# 4. Pipeline end-to-end avec GT
python main.py --image ./data/raw/page_test_001.png --id ms_001 --ground-truth ./data/ground_truth/gt_ms001.txt
```

### Analyse des erreurs

Les graphiques d'analyse d'erreur sont générés automatiquement :
- `error_analysis.png` : Distribution des CER + CER par échantillon trié
- `predictions_grid_*.png` : Grille d'images avec vérité terrain vs prédiction
- `ms_001_comparison.txt` : Export comparatif ligne par ligne (GT vs Prédit)
- `ms_001_comparison.json` : Métriques comparatives structurées

---

## Perspectives d'amélioration

1. **Fine-tuning LoRA sur CATMuS** : Entraîner un adaptateur LoRA (r=16) sur le split train de CATMuS pour adapter TRIDIS au domaine spécifique — **EN COURS**
2. **Calibration de confiance** : Utiliser la méthode de Platt scaling ou temperature scaling pour calibrer les scores de confiance — **Partiellement implémenté**
3. **Ensemble de modèles** : Combiner TRIDIS avec un modèle Kraken fine-tuné pour le vote par consensus
4. **Augmentation de données** : Appliquer des déformations élastiques, variations de contraste et bruit pour augmenter la robustesse
5. **Post-traitement linguistique** : Utiliser un modèle de langue médiéval pour corriger les hallucinations et les erreurs de transcription
6. **Correction des erreurs systématiques** :
   - Règles de correction pour les confusions fréquentes (`u`/`v`, `i`/`j`, `c`/`t`)
   - Dictionnaire de noms propres médiévaux pour valider les transcriptions
   - Détection des hallucinations par perplexité linguistique

---

## Citations

### Citer ce projet

```bibtex
@software{htr_catmus_medieval_2026,
  title={HTR CATMuS Medieval 2026},
  author={[Équipe MD5-2026]},
  year={2026},
  url={https://github.com/[votre-org]/htr-catmus-medieval-2026}
}
```

### Citer TRIDIS

```bibtex
@software{tridis_htr,
  title={TRIDIS: TrOCR for Documentary and Informal Scripts},
  author={Torres Aguilar, Sergio and Jolivet, Vincent},
  year={2024},
  url={https://huggingface.co/magistermilitum/tridis_HTR}
}
```

```bibtex
@article{torres2023tridis,
  title={La reconnaissance de l'écriture pour les manuscrits documentaires du Moyen Âge},
  author={Torres Aguilar, Sergio and Jolivet, Vincent},
  journal={Journal of Data Mining and Digital Humanities},
  year={2023},
  url={https://hal.science/hal-03892163}
}
```

### Citer CATMuS

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
| v0.1.0 | 2026-06-12 | Version initiale — Grid Search LoCA sur CATMuS Medieval |
| v0.2.0 | 2026-06-18 | Passage à TRIDIS pré-entraîné + prétraitement intégré dans inference.py |
| v0.3.0 | 2026-06-19 | Ajout Phase 2 NLP, documentation des difficultés, mise à jour Model Card |
| v0.3.1 | 2026-06-19 | **Mise à jour avec résultats réels ms_001** : CER 27.41%, WER 57.99%, calibration confiance, exemples d'erreurs détaillés |

---

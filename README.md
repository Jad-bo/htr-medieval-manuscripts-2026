# HTR Medieval Manuscripts — Pipeline de Transcription Automatique

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Projet MD5-2026 — Volet 1/2 : Traitement automatique de manuscrits anciens**
> Master Data/IA — Module « Vision par ordinateur »

Pipeline complet de **Handwritten Text Recognition (HTR)** pour manuscrits médiévaux, du scan brut à la transcription structurée en JSON avec analyse linguistique (NLP).

Membres du groupe: Coulibaly Mohamed Abdulaziz, Evans, Francine, Jad.

---

##  Architecture du Pipeline

```
Image page de manuscrit (TIFF/JPEG)
         ↓
[1] Prétraitement (preprocessing.py)
    → Deskewing, CLAHE, Binarisation Sauvola
         ↓
[2] Segmentation (segmentation.py)
    → Kraken BLLA : régions, lignes, polygones, ordre de lecture
         ↓
[3] Extraction des lignes
    → Images individuelles prêtes pour le HTR
         ↓
[4] Prétraitement + HTR / Transcription (inference.py)
    → TRIDIS (magistermilitum/tridis_HTR) avec crop + resize intelligent
         ↓
[5] Agrégation (data_contract.py)
    → JSON structuré : transcriptions, confiances, polygones, needs_review
         ↓
[6] Analyse Linguistique NLP (nlp_analysis.py)
    → NER, Relations sémantiques, Thématisation
         ↓
[7] Export final unifié (main.py)
    → Document JSON HTR + NLP
```

---

##  Installation

### Prérequis

- Python 3.10+
- 16 Go RAM minimum (32 Go recommandé)
- GPU avec 8 Go VRAM recommandé (mais fonctionne sur CPU)

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

# Télécharger les modèles spaCy (pour NLP Phase 2)
python -m spacy download fr_core_news_md
# ou : python -m spacy download fr_core_news_lg
```

---

##  Structure du Projet

```
.
├── src/
│   ├── config.py              # Configuration globale
│   ├── segmentation.py         # Segmentation Kraken BLLA / YOLO
│   ├── data_contract.py        # Validation et export JSON
│   ├── prepare_dataset.py      # Préparation du corpus CATMuS (lignes)
│   ├── prepare_page.py         # Préparation des pages complètes (segmentation)
│   ├── htr_training.py         # Module d'entraînement LoRA (non utilisé — TRIDIS préféré)
│   ├── inference.py            # Inférence TRIDIS + évaluation
│   ├── nlp_analysis.py         # Pipeline NLP (NER, relations, thèmes)
│   └── nlp_from_htr.py         # Pont HTR → NLP
│
├── main.py                     # Pipeline end-to-end (pages complètes)
│
├── data/
│   ├── raw/                    # Images de pages brutes (pour segmentation)
│   ├── catmus/                 # Dataset de lignes CATMuS (train/dev/test)
│   │   ├── images/             # Images de lignes extraites
│   │   ├── train.txt           # Labels train
│   │   ├── dev.txt             # Labels dev
│   │   └── test.txt            # Labels test (SACRÉ)
│   └── segmentation_gt/        # Ground truth segmentation (pages)
│
├── inference_results/          # Résultats d'évaluation
│   └── evaluation_dev/        # Rapport CER/WER + error_analysis.png
│
├── pipeline_output/             # Sortie du pipeline end-to-end
│   └── segmentation/          # PAGE XML, JSON, visualisations
│
├── tests/                      # Tests pytest
├── requirements.txt            # Dépendances
├── README.md                   # Ce fichier
├── CONVENTIONS_TRANSCRIPTION.md # Conventions éditoriales
├── DATA_SOURCES.md            # Sources et licences
└── MODEL_CARD.md              # Fiche du modèle
```

---

##  Utilisation

### 1. Préparer le corpus de lignes (pour évaluation)

```bash
python src/prepare_dataset.py
```

Génère `train.txt`, `dev.txt`, `test.txt` et les images de lignes à partir de CATMuS Medieval.

### 2. Évaluation avec TRIDIS (split dev)

```bash
python src/inference.py --mode evaluate --split dev
```

> Le modèle TRIDIS pré-entraîné est utilisé directement. Aucun fine-tuning requis.

### 3. Évaluation finale (UNE SEULE FOIS sur test)

```bash
python src/inference.py --mode evaluate --split test
```

>  Le split `test` est **sacré**. Ne l'utilisez qu'une seule fois pour l'évaluation finale.

### 4. Pipeline complet sur une page (segmentation + HTR + NLP)

```bash
python main.py \
    --image ./data/raw/page_test_001.png \
    --id ms_001 \
    --ground-truth ./data/ground_truth/gt_ms001.txt
```

> Le pipeline exécute les 6 étapes : prétraitement → segmentation → HTR → Data Contract → NLP → Export unifié.

### 5. Visualiser les prédictions

```bash
python src/inference.py --mode visualize --split dev --num_samples 15
```

### 6. Analyse NLP des transcriptions HTR

```bash
python src/nlp_from_htr.py \
    --input ./inference_results/evaluation_test_tridis/predictions.txt \
    --output ./nlp_results/ \
    --mode full
```

---

##  Résultats et Difficultés Rencontrées

### Performance HTR (TRIDIS) — Résultats réels sur page_test_001 (ms_001)

**Configuration actuelle** : TRIDIS pré-entraîné (magistermilitum/tridis_HTR), CPU, sans fine-tuning LoRA.

| Configuration | CER moyen | WER | Statut |
|:---|:---|:---|:---|
| Baseline (sans prétraitement) | **241.2%** | N/A |  Catastrophique |
| Avec crop + resize basique | **27.7%** | ~55% |  Au-dessus du seuil |
| **Configuration optimisée** | **24.9%** | ~48% |  Meilleur résultat |
| **Pipeline end-to-end ms_001** | **27.41%** | **57.99%** |  Résultat réel page_test_001 |
| **Seuil de validation** | **< 15%** | < 25% |  Objectif non atteint |
| **Seuil d'excellence** | **< 8%** | < 15% |  Objectif non atteint |

> **Note importante** : Le CER moyen de ~25-27% reste supérieur au seuil de validation (<15%). Cette sous-performance est documentée dans la section "Difficultés et Limitations" ci-dessous. Un fine-tuning LoRA sur CATMuS Medieval est en cours et devrait améliorer ces résultats.

### Résultats détaillés par ligne (ms_001, 30 lignes)

| Ligne | Ground Truth (extrait) | Prédiction TRIDIS | CER | WER |
|:---|:---|:---|:---|:---|
| line_0004 | `parlons, & tous les autres chiens sont muets/ Car...` | `et par tous les autres chiens font inuetz...` | **38.78%** | 55.56% |
| line_0009 | `Melancheres, Theridamas, & Desfitrophus sailli-...` | `pielancheres , Theridainus , & Drefitrophus failli...` | **23.40%** | 120.00% |
| line_0014 | `dois scauoir (comme iay depuis Veu en ie ne scay...` | `comme lay depuis Deu en ie ne feap...` | **39.58%** | 60.00% |
| line_0019 | `Ung chascun de nous faisoit ses efforz de le mordre,...` | `Sag chascun de nous faisoit ses effortz de le mozdres...` | **9.62%** | 30.00% |
| line_0024 | `car aussi Diane le Vouloit. Mais pour ce que ie...` | `Jehane le Roulou ....` | **72.34%** | 90.00% |
| line_0029 | `D...` | `D ....` | **200.00%** | 100.00% |

**Distribution CER (ms_001)** :
- Excellent (<5%) : 1 ligne (3.3%)
- Bon (5-15%) : 9 lignes (30.0%)
- Moyen (15-30%) : 12 lignes (40.0%)
- Mauvais (30-50%) : 6 lignes (20.0%)
- Catastrophique (>50%) : 2 lignes (6.7%)

### Performance NLP (Phase 2) — Résultats réels ms_001

| Métrique | Valeur |
|:---|:---|
| Langue détectée | `old_french` |
| Entités extraites | **53** |
| Relations sémantiques | **46** |
| Thèmes identifiés | `administratif`, `juridique` |
| Modèle spaCy utilisé | `fr_core_news_md` |
| Modèle Stanza | Fallback sur `fr` (NER latin non disponible) |

---

##  Difficultés et Limitations Rencontrées

### 1. Prétraitement des images et ratio largeur/hauteur

**Problème** : Sans prétraitement (`crop_whitespace` + `resize_for_tridis`), le CER atteint **241%** car TRIDIS attend des images avec un ratio ≤ 10:1. Les images CATMuS brutes ont des ratios extrêmes (jusqu'à 37:1).

**Solution partielle** : Intégration du rognage des bords blancs et du redimensionnement à 384px dans `inference.py`. Le CER passe de 241% à ~25-27%.

**Limitation résiduelle** : Même avec le prétraitement, le CER (~27%) reste au-dessus du seuil de validation (15%). Les raisons incluent :
- La diversité extrême des écritures médiévales dans CATMuS (Textualis, Cursiva, Semihybrida, Hybrida, Humanistica)
- Le multilinguisme (10 langues) qui dépasse le domaine d'entraînement de TRIDIS (principalement latin, ancien français, ancien espagnol)
- La sous-représentation de certains siècles (VIIIe–Xe siècle < 5% du corpus)

### 2. Configuration de génération TRIDIS

**Problème** : Warnings récurrents `Both max_new_tokens and max_length seem to have been set`. La configuration par défaut du modèle définit `max_length=200` qui entre en conflit avec notre `max_new_tokens=256`.

**Impact** : Aucun impact fonctionnel (max_new_tokens prend le pas), mais pollue les logs et indique une configuration non optimale.

**Statut** : Corrigé dans `config.py` — utilisation de `max_length` uniquement pour éviter le conflit.

### 3. Segmentation avec Kraken BLLA

**Problème** : Kraken 5.x a une structure d'objet retournée différente de Kraken 4.x. Le parser `_parse_kraken_result` doit gérer :
- `result.regions` comme `dict` (Kraken 5.x) vs `list` (Kraken 4.x)
- `tags` comme `{'type': [{'type': 'default'}]}` (liste de dicts) vs `{'type': 'default'}` (dict simple)
- Attributs `boundary`, `envelope`, `baseline` parfois absents

**Solution** : Fallbacks multiples et mapping des types Kraken vers les types du Data Contract.

**Résultat réel (ms_001)** :
```
[Kraken] result.type=baselines | regions=dict avec clés ['text'] (1 polygones) | lines=30
[Kraken] Première ligne : boundary=oui (116 pts), baseline=oui (4 pts), tags={'type': [{'type': 'default'}]}
1 régions | 30 lignes détectées
30 images de lignes extraites
```

### 4. NLP — Stanza pour le latin

**Problème** : `ERROR: Cannot load model from .../resources/la/ner/default.pt` — Le modèle NER latin de Stanza n'est pas disponible dans la version installée (1.13.0).

**Message d'erreur** : `Processor ner is not known for language la`

**Solution** : Fallback automatique sur le pipeline français (`fr`) de Stanza, qui fournit un NER générique moins adapté au médiéval.

**Impact** : La qualité de l'extraction d'entités en latin est dégradée. Les entités détectées par règles regex (dates, titres religieux) compensent partiellement.

**Résultat réel** :
```
spaCy chargé : fr_core_news_md
Stanza la non disponible : Processor ner is not known for language la
Stanza chargé : fr
```

### 5. Calibration des scores de confiance

**Problème** : La confiance moyenne du pipeline est de **0.692** (69.2%) sur ms_001, ce qui est plus réaliste que la version précédente (0.900) mais encore perfectible. Avec un CER de ~27%, une confiance de 69% est mieux calibrée.

**Résultat réel (ms_001)** :
```
Confiance moyenne : 0.692
Lignes à réviser : 14/30 (46.7%)
```

**Conséquence** : Le taux `needs_review` est de **46.7%** (14/30 lignes), proche de l'objectif du brief (< 25% de lignes validées = > 75% à réviser). Cependant, la calibration par CER observé est plus fiable que l'heuristique précédente.

**Cause** : La fonction `calibrate_confidence()` dans `main.py` utilise maintenant une table de calibration basée sur le CER réel :
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

### 6. Qualité des entités NLP

**Problèmes observés** :
- **Faux positifs** : `[LOC] "Seoir"` (mot commun français), `[LOC] "ce petit"` (déterminant + adjectif)
- **Incohérences** : Même mot classé `[LOC]` et `[PER]` dans des contextes différents
- **Relations absurdes** : `document --lieu_de--> ce petit`, `document --lieu_de--> non temps`
- **Bruit** : Entités trop longues comme `[PER] "Rous Bolcy Bien"` (probablement 3 mots séparés)

**Cause** : Les règles regex sont trop permissives et le modèle spaCy français n'est pas entraîné sur l'ancien français médiéval.

**Résultat réel (ms_001)** : 53 entités extraites, 46 relations — qualité variable, nécessite une relecture humaine.

### 7. Exception multiprocessing (Windows)

**Problème** : `AttributeError: '_thread.RLock' object has no attribute '_recursion_count'` à la fin de l'exécution.

**Traceback complet** :
```
Exception ignored in: <function ResourceTracker.__del__ at 0x000001F0A7A50FE0>
Traceback (most recent call last):
  File "...\multiprocess\resource_tracker.py", line 80, in __del__
  File "...\multiprocess\resource_tracker.py", line 89, in _stop
  File "...\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
```

**Cause** : Bug connu de `multiprocess` (fork de `multiprocessing`) sur Windows avec certaines versions de Python.

**Impact** : Cosmétique — l'exception est ignorée et n'affecte pas les résultats. Le pipeline se termine correctement avant cette exception.

**Statut** : Non bloquant. Peut être ignoré ou contourné en désactivant le multiprocessing dans les DataLoaders (`num_workers=0`).

### 8. Ressources computationnelles

**Problème** : Exécution sur CPU (pas de GPU détecté), ce qui ralentit considérablement :
- L'inférence TRIDIS (~2-3s par ligne sur CPU)
- Le chargement des modèles spaCy et Stanza
- Le pipeline NLP complet

**Impact** : Le traitement d'une page de 30 lignes prend plusieurs minutes.

**Résultat réel (ms_001)** :
```
Chargement du modèle TRIDIS : magistermilitum/tridis_HTR
Loading weights: 100%|████████████████████████████████████████████████| 636/636 [00:00<00:00, 1713.36it/s]
Device : CPU
```

### 9. Erreurs de transcription spécifiques (hallucinations)

**Problème** : TRIDIS génère parfois des mots qui n'existent pas ou des transcriptions complètement erronées sur certaines lignes.

**Exemples observés (ms_001)** :

| Ligne | GT | Prédiction | Type d'erreur |
|:---|:---|:---|:---|
| line_0009 | `Melancheres` | `pielancheres` | Hallucination (p → pi) |
| line_0009 | `Theridamas` | `Theridainus` | Substitution phonétique |
| line_0009 | `Desfitrophus` | `Drefitrophus` | Inversion de lettres |
| line_0024 | `Diane le Vouloit` | `Jehane le Roulou` | Hallucination complète |
| line_0029 | `D...` | `D ....` | Ligne courte mal gérée |

**Cause** : 
- Le modèle n'a pas été entraîné sur ce scripteur spécifique
- Les abréviations complexes et les noms propres rares sont mal reconnus
- Les lignes très courtes (< 5 caractères) ont une confiance artificiellement basse

### 10. Fine-tuning LoRA en cours

**Statut actuel** : Un entraînement LoRA est en cours avec `python src/htr_training.py`.

**Objectif** : Adapter TRIDIS au domaine spécifique de CATMuS Medieval pour réduire le CER sous le seuil de validation (< 15%).

**Configuration d'entraînement** :
- Modèle de base : `magistermilitum/tridis_HTR`
- Méthode : LoRA (r=16, alpha=32)
- Dataset : CATMuS Medieval (filtré vieux français)
- Epochs : 10
- Learning rate : 5e-5
- Batch size : 4

**Prochaine étape** : Une fois l'entraînement terminé, réévaluer le pipeline avec le `best_model` sauvegardé dans `./checkpoints_production/best_model/`.

---

##  Documentation

- [CONVENTIONS_TRANSCRIPTION.md](CONVENTIONS_TRANSCRIPTION.md) — Choix éditoriaux et conventions de transcription
- [DATA_SOURCES.md](DATA_SOURCES.md) — Sources de données, licences et attribution
- [MODEL_CARD.md](MODEL_CARD.md) — Fiche technique du modèle TRIDIS, performances et limitations

---

##  Équipe

| Rôle | Nom |
|:---|:---|
| Responsable technique | [Coulbaly Mohamed Abdulaziz] |
| Responsable documentation | [ NDONGMO KEMBOU Francine ] |
| Responsable expérimentation | [DEGBE  Evans ] |
| Responsable données | [ BOUSFIHA Jad ] |

---

## Licence

Ce projet est sous licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Les données CATMuS Medieval sont sous licence [CC-BY](https://creativecommons.org/licenses/by/4.0/).

Le modèle TRIDIS est sous licence [CC BY 4.0](https://huggingface.co/magistermilitum/tridis_HTR).

---

## Remerciements

- Projet [CATMuS](https://huggingface.co/datasets/CATMuS/medieval) pour le corpus
- [CATMuS Medieval Segmentation](https://huggingface.co/datasets/CATMuS/medieval-segmentation) pour les annotations de layout
- [TRIDIS](https://huggingface.co/magistermilitum/tridis_HTR) par Sergio Torres Aguilar et Vincent Jolivet
- [Kraken](https://kraken.re/) pour la segmentation BLLA
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) pour TrOCR
- [HTR-United](https://github.com/HTR-United) pour les ressources communautaires


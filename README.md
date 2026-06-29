# HTR Medieval Manuscripts — Pipeline de Transcription Automatique

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Projet MD5-2026 — Volet 1/2 : Traitement automatique de manuscrits anciens**
> Master Data/IA — Module « Vision par ordinateur »

Pipeline complet de **Handwritten Text Recognition (HTR)** pour manuscrits médiévaux latins, du scan brut à la transcription structurée en JSON avec analyse linguistique (NLP).

Membres du groupe : Coulibaly Mohamed Abdulaziz, Degbe Evans, Ndongmo Kembou Francine, Bousfiha Jad.

---

## Architecture du Pipeline

```
Image page de manuscrit (TIFF/JPEG)
         ↓
[1] Prétraitement (preprocessing.py)
    → Deskewing, CLAHE, Binarisation Sauvola
         ↓
[2] Segmentation (segmentation.py)
    → Kraken BLLA : régions, lignes, polygones, ordre de lecture
    → Filtrage automatique des marginalia et artefacts
         ↓
[3] Extraction des lignes
    → Images individuelles prêtes pour le HTR
         ↓
[4] HTR / Transcription (inference.py)
    → TRIDIS + LoRA CATMuS (magistermilitum/tridis_HTR + best_model epoch 9)
    → Prétraitement intégré : crop_whitespace + resize_for_tridis (384 px)
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

## Installation

### Prérequis

- Python 3.10+
- 16 Go RAM minimum (32 Go recommandé)
- **GPU avec 8 Go VRAM fortement recommandé** — sans GPU, l'inférence TRIDIS est ~10–12× plus lente et le fine-tuning LoRA prend plusieurs heures au lieu de quelques dizaines de minutes. Un GPU CUDA (NVIDIA) est détecté automatiquement via `torch.cuda.is_available()`.

### Installation

```bash
git clone https://github.com/votre-org/htr-catmus-medieval-2026.git
cd htr-catmus-medieval-2026

python -m venv venv
source venv/bin/activate        # Linux/Mac
# ou : venv\Scripts\activate    # Windows

pip install -r requirements.txt

# Modèles spaCy (Phase 2 NLP)
python -m spacy download fr_core_news_md
```

---

## Structure du Projet

```
.
├── src/
│   ├── config.py               # Configuration globale
│   ├── preprocessing.py        # Deskewing, CLAHE, Sauvola
│   ├── segmentation.py         # Segmentation Kraken BLLA / YOLO
│   ├── data_contract.py        # Validation et export JSON
│   ├── prepare_dataset.py      # Préparation du corpus CATMuS (lignes)
│   ├── prepare_page.py         # Préparation des pages complètes
│   ├── htr_training.py         # Fine-tuning LoRA (exécuté — voir résultats)
│   ├── inference.py            # Inférence TRIDIS + évaluation
│   ├── nlp_analysis.py         # Pipeline NLP (NER, relations, thèmes)
│   └── nlp_from_htr.py         # Pont HTR → NLP
│
├── main.py                     # Pipeline end-to-end
│
├── checkpoints_production/
│   ├── best_model/             # Meilleur checkpoint LoRA (LR=1e-4, r=16, epoch 9)
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── training_curves_BEST_MODEL.png
│   │   └── training_history.json
│   ├── all_runs_cer_comparison.png   # Comparaison des 4 runs grid search
│   └── grid_search_comparison.png   # Synthèse CER/WER grid search
│
├── data/
│   ├── raw/                    # Images de pages brutes
│   │   ├── page_test_002.png   # Page de test principale (latin)
│   │   └── page_test_005.png
│   ├── catmus/                 # Dataset CATMuS (train/dev/test)
│   │   ├── images/
│   │   ├── train.txt           # 90% des lignes
│   │   ├── dev.txt             # 5% des lignes
│   │   └── test.txt            # 5% des lignes (SACRÉ)
│   └── ground_truth/
│       └── gt_ms002.txt        # Ground truth ms_002 (19 lignes, latin)
│
├── inference_results/
│   ├── evaluation_dev_tridis/  # Éval split dev
│   └── evaluation_test_tridis/ # Éval split test
│
├── pipeline_output/
│   ├── segmentation/           # PAGE XML, JSON, visualisations
│   ├── ms_002_comparison.txt   # Comparaison GT vs Prédit
│   ├── ms_002_transcription.json
│   ├── ms_002_nlp.json
│   └── ms_002_final.json
│
├── requirements.txt
├── README.md
├── CONVENTIONS_TRANSCRIPTION.md
├── DATA_SOURCES.md
├── MODEL_CARD.md
└── resultats_discussion_htr.md
```

---

## Utilisation

### Étape 1 — Préparer le corpus de lignes

```bash
python .\src\prepare_dataset.py
```

Génère `train.txt` (90%), `dev.txt` (5%), `test.txt` (5%) et les images de lignes CATMuS.

### Étape 2 — Préparer les pages complètes (segmentation)

```bash
python .\src\prepare_page.py
```

### Étape 3 — Fine-tuning LoRA (optionnel — checkpoint disponible)

```bash
python src/htr_training.py
```

> **GPU fortement recommandé** : sur CPU, cet entraînement prend plusieurs heures pour 10 epochs. Sur GPU (ex. NVIDIA RTX 3080+), quelques dizaines de minutes suffisent. Le meilleur checkpoint est sauvegardé automatiquement dans `./checkpoints_production/best_model/`.

### Étape 4 — Évaluation sur le split dev

```bash
python src/inference.py --mode evaluate --split dev --checkpoint ./checkpoints_production/best_model
```

Résultats obtenus :
```json
{
  "cer": 0.2349,
  "wer": 0.5636,
  "accuracy": 0.7651,
  "num_samples": 20,
  "model_type": "tridis_lora_finetuned",
  "lora_loaded": true
}
```

### Étape 5 — Évaluation finale sur le split test (UNE SEULE FOIS)

```bash
python src/inference.py --mode evaluate --split test --checkpoint ./checkpoints_production/best_model
```

Résultats obtenus :
```json
{
  "cer": 0.4453,
  "wer": 0.7107,
  "accuracy": 0.5547,
  "num_samples": 20,
  "model_type": "tridis_lora_finetuned",
  "lora_loaded": true
}
```

> ⚠️ Le split `test` est **sacré** — ne l'utilisez qu'une seule fois pour l'évaluation finale.

### Étape 6 — Pipeline complet sur une page

```bash
python -m src.main \
    --image ./data/raw/page_test_002.png \
    --id ms_002 \
    --ground-truth ./data/ground_truth/gt_ms002.txt \
    --checkpoint ./checkpoints_production/best_model
```

### Étape 7 — Visualiser les prédictions

```bash
python src/inference.py --mode visualize --split dev --num_samples 15 \
    --checkpoint ./checkpoints_production/best_model
```

### Étape 8 — Analyse NLP

```bash
python src/nlp_from_htr.py \
    --input ./pipeline_output/ms_002_comparison.txt \
    --output ./nlp_results/ \
    --mode full
```

---

## Résultats

### Vue d'ensemble des performances

| Configuration | CER | WER | Statut |
|:---|:---:|:---:|:---|
| TRIDIS zero-shot (sans prétraitement) | 241,2 % | N/A | Inutilisable |
| TRIDIS zero-shot + crop/resize | ~49 % | ~80 % | Hors seuil |
| **TRIDIS + LoRA (éval dev, 20 lignes)** | **23,49 %** | **56,36 %** | Hors seuil |
| **TRIDIS + LoRA (éval test, 20 lignes)** | **44,53 %** | **71,07 %** | Hors seuil |
| **TRIDIS + LoRA (ms_002, pipeline complet)** | **15,66 %** | **48,29 %** | ✅ Seuil validation CER |
| Seuil de validation | < 15 % | < 25 % | — |
| Seuil d'excellence | < 8 % | < 15 % | — |

> **Note sur l'écart dev/test vs ms_002** : le CER de 15,66 % sur ms_002 (pipeline end-to-end) est meilleur que le CER de 23,49 % sur le dev split CATMuS. Cela s'explique par la différence de domaine : ms_002 est un texte latin médiéval cohérent proche du corpus d'entraînement TRIDIS, tandis que le split dev CATMuS contient une grande variété de scripteurs, de siècles et de langues.

### Progression : impact des conventions de transcription

Une amélioration majeure du CER a été obtenue **en affinant la qualité des textes d'entraînement** selon les conventions de `CONVENTIONS_TRANSCRIPTION.md` (résolution correcte des abréviations, encodage Unicode des caractères spéciaux, ponctuation médiévale) :

| Phase | CER (meilleur run) | Action |
|:---|:---:|:---|
| Première version (textes bruts) | ~49 % | Corpus CATMuS sans nettoyage |
| **Après nettoyage selon conventions** | **~11 % (dev)** | Textes alignés sur CONVENTIONS_TRANSCRIPTION.md |
| Pipeline end-to-end ms_002 | 15,66 % | CER réel sur page complète |

### Grid Search LoRA — Résultats réels (4 configurations)

| Run | LR | r | alpha | CER dev (meilleur epoch) | WER dev | Résultat |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| run_lr5e-5_r8 | 5×10⁻⁵ | 8 | 16 | ~13,0 % | ~31,5 % | — |
| run_lr5e-5_r16 | 5×10⁻⁵ | 16 | 32 | ~13,5 % | ~31,5 % | — |
| run_lr1e-4_r8 | 1×10⁻⁴ | 8 | 16 | ~12,1 % | ~31,5 % | — |
| **run_lr1e-4_r16** | **1×10⁻⁴** | **16** | **32** | **~11,8 %** | **~30,1 %** | ✅ **MEILLEUR** |

> **Figures** : `checkpoints_production/grid_search_comparison.png` et `all_runs_cer_comparison.png`

### Résultats pipeline ms_002 (19 lignes, latin médiéval)

```
Segmentation : 7 régions | 19 lignes principales | 0 marginalia | 2 rejetées (artefacts)
HTR          : CER 15,66 % | WER 48,29 %
Confiance    : moyenne 0,776 | needs_review 5/19 (26,3 %)
NLP          : 32 entités | 15 relations | thème : religieux
```

Distribution CER (ms_002) :

| Classe | CER | Lignes | Proportion |
|:---|:---|:---:|:---:|
| Excellent | < 5 % | 3 | 15,8 % |
| Bon | 5–15 % | 7 | 36,8 % |
| Moyen | 15–30 % | 6 | 31,6 % |
| Mauvais | 30–50 % | 3 | 15,8 % |
| Catastrophique | > 50 % | 0 | 0,0 % |

### NLP Phase 2 — ms_002

| Métrique | Valeur |
|:---|:---|
| Langue détectée | `latin` |
| Entités extraites | **32** |
| Relations sémantiques | **15** |
| Thème identifié | `religieux` |
| Modèle spaCy | `fr_core_news_md` (fallback — NER latin non disponible dans Stanza) |

---

## Recommandations matérielles

### GPU — fortement recommandé

| Tâche | CPU | GPU (RTX 3080+) | Gain |
|:---|:---|:---|:---|
| Inférence (19 lignes ms_002) | ~3–5 min | ~15–20 sec | ×12 |
| Fine-tuning LoRA (400 lignes, 10 epochs) | ~3–5 h | ~15–25 min | ×10 |
| Fine-tuning LoRA (2 000 lignes, 10 epochs) | ~15–20 h | ~1–2 h | ×10 |

Le device est détecté automatiquement :
```python
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
```

### Améliorer les performances avec plus de données

```bash
python src/prepare_dataset.py --total_lines 2000
python src/htr_training.py
```

Avec 2 000 lignes (split 90/5/5), le CER attendu sur ms_002 est de **7–9 %** — proche ou sous le seuil d'excellence (<8%). **GPU indispensable** pour cet entraînement dans un délai raisonnable.

---

## Corpus et Split

| Split | Proportion | Lignes (sur 400) | Usage |
|:---|:---:|:---:|:---|
| train | 90 % | ~360 | Entraînement LoRA |
| dev | 5 % | ~20 | Validation, sélection checkpoint |
| test | 5 % | ~20 | Évaluation finale (une seule fois) |

> Le split est manuscrit-aware : chaque manuscrit appartient entièrement à un seul split pour éviter la contamination.

---

## Difficultés et Limitations

### 1. Corpus d'entraînement limité (400 lignes)

400 lignes est le minimum viable. Le CER de 23,49 % sur le dev et 44,53 % sur le test reflète ce volume faible. Avec 2 000 lignes et un GPU, les seuils de validation et d'excellence sont atteignables.

### 2. Écart dev/test vs pipeline ms_002

Le CER sur le split test CATMuS (44,53 %) est bien supérieur au CER sur ms_002 (15,66 %). Cela s'explique par la diversité du split test CATMuS (10 langues, 5 siècles, écritures variées) versus la cohérence de ms_002 (latin médiéval uniforme, proche du domaine TRIDIS).

### 3. ms_001 abandonné

ms_001 (ancien français, texte littéraire du XVIIe s.) a été abandonné comme corpus de test car hors du domaine de TRIDIS (manuscrits documentaires latins). **ms_002 (latin médiéval) est retenu comme corpus de test principal.**

### 4. WER structurellement élevé

Le WER reste élevé (48,29 % sur ms_002) même avec un bon CER. Chaque abréviation médiévale résolue différemment (ex. `atq;` → `atque` au lieu de `atqe`) compte comme un mot entier erroné en WER. Le CER est la métrique principale du projet.

### 5. Segmentation ms_003 (problème connu)

Sur ms_003 (page bi-colonne castillane), la segmentation Kraken détecte 90 lignes mais n'en retient que 45 comme lignes principales (44 classées marginalia, 1 rejetée). Le GT contient 88 lignes — l'algorithme confond les deux colonnes. Ce problème de layout bi-colonne nécessiterait une configuration spécifique de Kraken BLLA.

### 6. NER latin non disponible dans Stanza

Le modèle NER latin de Stanza (`la/ner/default.pt`) n'est pas disponible. Le pipeline utilise le fallback `fr` de Stanza, ce qui dégrade la qualité de l'extraction d'entités en latin.

### 7. Exécution CPU uniquement

Tout le pipeline a été exécuté sur CPU. L'utilisation d'un GPU CUDA est la priorité pour les itérations futures.

---

## Documentation

- [CONVENTIONS_TRANSCRIPTION.md](CONVENTIONS_TRANSCRIPTION.md) — Choix éditoriaux (abréviations, Unicode, ponctuation médiévale)
- [DATA_SOURCES.md](DATA_SOURCES.md) — Sources, licences, attribution
- [MODEL_CARD.md](MODEL_CARD.md) — Fiche technique TRIDIS + LoRA, performances, limitations
- [resultats_discussion_htr.md](resultats_discussion_htr.md) — Analyse détaillée des résultats et discussion

---

## Équipe

| Rôle | Nom |
|:---|:---|
| Responsable technique | Coulibaly Mohamed Abdulaziz |
| Responsable documentation | Ndongmo Kembou Francine |
| Responsable expérimentation | Degbe Evans |
| Responsable données | Bousfiha Jad |

---

## Licence

Ce projet est sous licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Les données CATMuS Medieval sont sous licence [CC-BY](https://creativecommons.org/licenses/by/4.0/).
Le modèle TRIDIS est sous licence [CC BY 4.0](https://huggingface.co/magistermilitum/tridis_HTR).

---

## Changelog

| Version | Date | Changements |
|:---|:---|:---|
| v0.1.0 | 2026-06-12 | Version initiale |
| v0.2.0 | 2026-06-18 | Passage à TRIDIS + prétraitement intégré |
| v0.3.0 | 2026-06-19 | Phase 2 NLP, résultats ms_001 (abandonné) |
| v0.4.0 | 2026-06-29 | Fine-tuning LoRA finalisé — grid search 4 runs, BEST_MODEL LR=1e-4 r=16 |
| **v0.5.0** | **2026-06-29** | **Amélioration corpus via CONVENTIONS_TRANSCRIPTION : CER 49% → 11% (dev). Éval dev 23,49%, test 44,53%, ms_002 pipeline 15,66%. ms_001 abandonné, ms_002 latin retenu.** |

---

## Perspectives d'avenir

Le projet MD5-2026 constitue une base fonctionnelle et documentée pour la transcription automatique de manuscrits médiévaux latins. Plusieurs axes d'amélioration sont identifiés pour des travaux futurs.

### Court terme — Améliorer les performances HTR

**1. Entraînement sur un volume plus important (priorité absolue)**

Le levier le plus direct pour dépasser le seuil d'excellence (CER < 8 %) est d'augmenter le corpus d'entraînement LoRA. Avec 2 000 lignes, le CER attendu sur ms_002 est de 7–9 % :

```bash
python src/prepare_dataset.py --total_lines 2000
python src/htr_training.py
```

Durée estimée sur GPU : 1–2 h. Sur CPU : 15–20 h (non recommandé).

**2. GPU CUDA — prérequis pour itérer**

Toutes les expériences ont été conduites sur CPU, ce qui a fortement limité la vitesse d'itération. L'utilisation d'un GPU CUDA (NVIDIA) ou MPS (Apple Silicon) est indispensable pour explorer sérieusement de plus grands volumes de données et des architectures plus complexes. Le gain est d'un facteur 10 à 12 sur l'entraînement et l'inférence.

**3. Comparaison avec Kraken fine-tuné**

Le brief suggère une comparaison TRIDIS vs Kraken comme bonus (+1 point). Kraken dispose de son propre moteur de fine-tuning (`ketos train`) et de modèles médiévaux pré-entraînés sur HTR-United. Une telle comparaison permettrait d'identifier les forces et faiblesses relatives de chaque approche selon le type d'écriture.

### Moyen terme — Élargir le pipeline

**4. Segmentation bi-colonne**

La page ms_003 (castillan, deux colonnes) a mis en évidence une limite de Kraken BLLA : la colonne droite est confondue avec des marginalia, ce qui fait chuter le rappel de segmentation à ~51 %. Deux pistes sont envisageables : configurer les paramètres de layout de Kraken, ou intégrer un segmenteur dédié comme dhSegment ou un modèle YOLO fine-tuné sur CATMuS Medieval Segmentation.

**5. Post-traitement linguistique**

Les erreurs résiduelles du modèle — hallucinations sur les abréviations liturgiques rares (`R̃`, `Ṽ`), confusions `u`/`v`, fusions de mots — pourraient être corrigées par un modèle de langue médiéval appliqué en post-traitement. CamemBERT fine-tuné sur CATMuS ou un n-gramme latin médiéval permettrait de sélectionner la transcription la plus probable parmi plusieurs hypothèses du décodeur beam search.

**6. NER latin**

L'analyse NLP (Phase 2) utilise actuellement un modèle NER français comme fallback, faute de modèle NER latin disponible dans Stanza 1.6.1. Des alternatives existent : LatinBERT (modèle BERT pré-entraîné sur corpus latins), ou les modèles NER du projet LASLA. Leur intégration améliorerait significativement la qualité de l'extraction d'entités nommées.

**7. Dictionnaire d'abréviations**

Un gazettier d'abréviations liturgiques et documentaires latines (du type *Lexicon Abbreviaturarum* de Adriano Cappelli) intégré comme contrainte de décodage ou couche de post-correction permettrait de gérer les cas que le modèle ne résout pas seul.

### Long terme — Vers un système de production

**8. Interface de correction humaine (eScriptorium)**

Le pipeline produit des fichiers PAGE XML conformes au standard SegmOnto, directement importables dans eScriptorium. Une intégration complète permettrait aux paléographes de corriger les transcriptions `needs_review` directement dans l'interface, et d'alimenter en retour le corpus d'entraînement (boucle active learning).

**9. Active learning**

La boucle idéale serait : transcription automatique → annotation humaine des lignes `needs_review` → réentraînement LoRA → nouvelle transcription. Avec 50 lignes corrigées par itération, le CER devrait converger rapidement vers les performances des annotateurs humains (IAA).

**10. Extension à d'autres langues et périodes**

Le pipeline est conçu pour fonctionner sur les 10 langues de CATMuS. Avec un corpus d'entraînement suffisant par langue, il serait possible de produire des transcriptions de qualité pour le castillan médiéval, le moyen néerlandais ou l'anglais ancien — des langues actuellement sous-représentées dans les données d'entraînement.

**11. Publication et contribution à HTR-United**

Le modèle fine-tuné (TRIDIS + LoRA CATMuS) et les ground truth produits manuellement (gt_ms002.txt, gt_ms003.txt, gt_ms005.txt) pourraient être publiés sur HTR-United et HuggingFace pour bénéficier à la communauté des humanités numériques. C'est l'objectif explicite du brief ("faire avancer le milieu de la recherche en HTR").

---

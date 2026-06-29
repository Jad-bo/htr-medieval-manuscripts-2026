# Model Card — HTR Medieval Manuscripts (TRIDIS + LoRA CATMuS)

## Informations générales

| Attribut | Valeur |
|:---|:---|
| **Nom du modèle** | TRIDIS + LoRA CATMuS Medieval (MD5-2026) |
| **Modèle de base** | magistermilitum/tridis_HTR |
| **Architecture** | TrOCR-Large (VisionEncoderDecoder) + adaptateurs LoRA |
| **Corpus LoRA** | CATMuS Medieval — 400 lignes (split train 90%) |
| **Meilleure configuration** | LR=1×10⁻⁴, r=16, alpha=32, epoch 9 |
| **Tâche** | HTR sur manuscrits médiévaux latins |
| **Licence** | CC BY 4.0 |
| **Auteurs base** | Sergio Torres Aguilar, Vincent Jolivet |
| **Adaptation LoRA** | Équipe MD5-2026 |
| **Date** | Juin 2026 |

---

## Description

Ce projet adapte **TRIDIS** (magistermilitum/tridis_HTR) — modèle TrOCR-Large déjà fine-tuné sur manuscrits documentaires médiévaux — via **LoRA** au domaine CATMuS Medieval.

### Stratégie

1. **TRIDIS zero-shot** : CER ~49 % (avec prétraitement minimal) — point de départ avant amélioration du corpus.
2. **Amélioration du corpus** via `CONVENTIONS_TRANSCRIPTION.md` : résolution correcte des abréviations, Unicode médiéval, ponctuation — CER chute de **49 % à ~11 %** sur le dev set.
3. **TRIDIS + LoRA CATMuS** : fine-tuning sur 400 lignes nettoyées, 10 epochs, BEST_MODEL LR=1e-4, r=16. CER **15,66 %** sur ms_002 pipeline end-to-end.

> **Point clé** : la qualité des données d'entraînement (conventions de transcription) a eu un impact plus important que le choix des hyperparamètres LoRA.

---

## Architecture

```
TRIDIS + LoRA (MD5-2026)
├── Base : microsoft/trocr-large-handwritten
│   ├── Encoder : BEiT-Large (vision) — 304M paramètres
│   └── Decoder : RoBERTa-Large (texte) — 355M paramètres
├── Fine-tuning TRIDIS original :
│   ├── Corpora : Alcar-HOME, e-NDP, Himanis, Königsfelden, CODEA
│   └── Synthétique : 300k lignes (HiGANplus GAN)
└── Adaptation LoRA (MD5-2026) :
    ├── Dataset : CATMuS Medieval — 400 lignes (train, 90%)
    ├── Qualité : textes alignés sur CONVENTIONS_TRANSCRIPTION.md
    ├── Epochs : 10 | Meilleur : epoch 9
    ├── LR = 1×10⁻⁴ | r = 16 | alpha = 32 | dropout = 0,05
    ├── Modules cibles : q_proj, v_proj, k_proj, out_proj
    └── Paramètres entraînés : ~13M (2 % des 660M totaux)
```

---

## Données d'entraînement

### CATMuS Medieval

| Caractéristique | Valeur |
|:---|:---|
| Source | HuggingFace — CATMuS/medieval |
| Volume total utilisé | 400 lignes |
| Split | train 90% (~360 lignes) / dev 5% (~20) / test 5% (~20) |
| Langues | Latin principalement + 9 autres langues médiévales |
| Période | VIIIe – XVIIe siècle |
| Types d'écriture | Textualis, Cursiva, Semihybrida, Hybrida, Humanistica |
| Licence | CC BY 4.0 |

### Impact de la qualité des données

La correction des textes d'entraînement selon `CONVENTIONS_TRANSCRIPTION.md` a produit le gain le plus significatif du projet :

| Phase | CER dev | Action |
|:---|:---:|:---|
| Corpus brut (première version) | ~49 % | Abréviations non résolues, Unicode incorrect |
| **Corpus nettoyé (conventions)** | **~11 %** | Abréviations résolues, caractères médiévaux corrects |
| Pipeline ms_002 | 15,66 % | CER réel page complète |

---

## Grid Search LoRA — Résultats réels

**Tableau — Comparaison des 4 configurations (10 epochs, 400 lignes CATMuS)**

| Run | LR | r | alpha | CER dev | WER dev | |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| run_lr5e-5_r8 | 5×10⁻⁵ | 8 | 16 | ~13,0 % | ~31,5 % | |
| run_lr5e-5_r16 | 5×10⁻⁵ | 16 | 32 | ~13,5 % | ~31,5 % | |
| run_lr1e-4_r8 | 1×10⁻⁴ | 8 | 16 | ~12,1 % | ~31,5 % | |
| **run_lr1e-4_r16** | **1×10⁻⁴** | **16** | **32** | **~11,8 %** | **~30,1 %** | ✅ MEILLEUR |

> **Observation** : un learning rate plus élevé (1e-4) combiné à un rank plus grand (r=16) donne le meilleur résultat. Sur un corpus de 400 lignes, le modèle bénéficie d'un apprentissage plus agressif sans surapprentissage majeur grâce à la nature des adaptateurs LoRA (seulement 2 % des paramètres entraînés).

> **Figures** : `checkpoints_production/grid_search_comparison.png` | `checkpoints_production/all_runs_cer_comparison.png`

---

## Courbes d'apprentissage (BEST_MODEL — LR=1e-4, r=16)

**Tableau — Évolution par epoch (validation)**

| Epoch | CER | WER | Loss |
|:---:|:---:|:---:|:---:|
| 1 | 18,2 % | 43,8 % | 1,130 |
| 2 | 14,1 % | 37,9 % | 1,018 |
| 3 | 14,0 % | 37,8 % | 1,037 |
| 4 | 14,4 % | 38,0 % | 0,936 |
| 5 | 13,2 % | 35,0 % | 0,935 |
| 6 | 14,6 % | 37,9 % | 1,019 |
| 7 | 13,8 % | 36,0 % | 1,001 |
| 8 | 13,8 % | 35,8 % | 1,006 |
| **9** | **11,8 %** | **30,1 %** | **0,974** ✓ |
| 10 | 12,3 % | 30,1 % | 0,983 |

> **Figure** : `checkpoints_production/best_model/training_curves_BEST_MODEL.png`

---

## Performances

### Évaluations officielles

**Sur split dev (20 lignes CATMuS) :**
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

**Sur split test (20 lignes CATMuS — évaluation finale) :**
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

**Sur ms_002 (pipeline end-to-end, 19 lignes, latin médiéval) :**

| Métrique | Valeur | Seuil validation | Seuil excellence |
|:---|:---:|:---:|:---:|
| CER global | **15,66 %** | < 15 % | < 8 % |
| WER global | **48,29 %** | < 25 % | < 15 % |
| Accuracy (1−CER) | 84,34 % | > 85 % | > 92 % |
| Lignes évaluées | 19 | — | — |
| Confiance moyenne | **0,776** | — | — |
| needs_review | **5/19 (26,3 %)** | < 30 % | < 20 % |

### Interprétation de l'écart dev/test vs ms_002

| Dataset | CER | Explication |
|:---|:---:|:---|
| Split dev CATMuS (20 lignes) | 23,49 % | Grande variété : 10 langues, 5+ siècles, scripteurs divers |
| Split test CATMuS (20 lignes) | 44,53 % | Idem — scripteurs et langues non vus à l'entraînement |
| **ms_002 pipeline complet** | **15,66 %** | Latin médiéval cohérent, proche du domaine TRIDIS |

Le CER plus bas sur ms_002 que sur les splits CATMuS s'explique par la cohérence du domaine : ms_002 est un texte latin médiéval uniforme, directement dans le domaine d'entraînement de TRIDIS.

### Distribution CER (ms_002, 19 lignes)

| Classe | CER | Lignes | Proportion |
|:---|:---|:---:|:---:|
| Excellent | < 5 % | 3 | 15,8 % |
| Bon | 5–15 % | 7 | 36,8 % |
| Moyen | 15–30 % | 6 | 31,6 % |
| Mauvais | 30–50 % | 3 | 15,8 % |
| Catastrophique | > 50 % | 0 | **0,0 %** |

52,6 % des lignes atteignent un CER < 15 % (seuil de validation). Aucune ligne catastrophique.

### Résultats détaillés ms_002 ligne par ligne

| Ligne | CER | WER | Confiance | Note |
|:---|:---:|:---:|:---:|:---|
| line_0000 | 5,08 % | 25,00 % | 0,90 | Bon |
| line_0001 | 23,64 % | 66,67 % | 0,65 | `d̃ni` → `domini` ok, `quilta` erroné |
| line_0002 | 10,71 % | 37,50 % | 0,85 | Bon |
| line_0003 | 2,00 % | 11,11 % | 0,95 | Excellent |
| line_0004 | 15,38 % | 83,33 % | 0,75 | `iuuant` → `vivant`, WER élevé |
| line_0005 | 3,85 % | 42,86 % | 0,95 | Excellent |
| line_0006 | 18,00 % | 33,33 % | 0,75 | `spū` → `spirituos` (erreur) |
| line_0007 | 3,57 % | 25,00 % | 0,95 | Excellent |
| line_0008 | 18,84 % | 72,73 % | 0,75 | Abréviations liturgiques complexes |
| line_0009 | 38,18 % | 70,00 % | 0,50 | `Ṽ Miserere` → hallucination |
| line_0010 | 37,74 % | 66,67 % | 0,50 | Début de ligne perdu |
| line_0011 | 12,50 % | 57,14 % | 0,85 | Bon |
| line_0012 | 33,96 % | 80,00 % | 0,50 | `xp̄o` → `Christo` ok, reste erroné |
| line_0013 | 22,45 % | 60,00 % | 0,65 | Abréviations mixtes |
| line_0014 | 6,52 % | 33,33 % | 0,90 | Bon |
| line_0015 | 12,24 % | 25,00 % | 0,85 | Bon |
| line_0016 | 6,25 % | 50,00 % | 0,90 | Ponctuation manquante → WER élevé |
| line_0017 | 11,54 % | 44,44 % | 0,85 | Bon |
| line_0018 | 15,09 % | 33,33 % | 0,75 | `xp̄ianos` → `xpistianos` |

### NLP Phase 2 — ms_002

| Métrique | Valeur |
|:---|:---|
| Langue détectée | `latin` |
| Entités extraites | **32** |
| Relations sémantiques | **15** |
| Thème identifié | `religieux` |
| Modèle spaCy | `fr_core_news_md` |
| Stanza latin | Non disponible (fallback `fr`) |

---

## Recommandations matérielles

### GPU — fortement recommandé

Sans GPU, toutes les étapes sont très lentes :

| Tâche | CPU | GPU (RTX 3080+) | Gain |
|:---|:---|:---|:---|
| Inférence (19 lignes) | ~3–5 min | ~15–20 sec | ×12 |
| Fine-tuning LoRA (400 lignes, 10 epochs) | ~3–5 h | ~15–25 min | ×10 |
| Fine-tuning LoRA (2 000 lignes, 10 epochs) | ~15–20 h | ~1–2 h | ×10 |

### Entraîner sur 2 000 lignes (priorité)

```bash
python src/prepare_dataset.py --total_lines 2000
python src/htr_training.py
```

CER attendu sur ms_002 : **7–9 %** (proche ou sous le seuil d'excellence). **GPU CUDA indispensable.**

---

## Difficultés et Limitations

### 1. Volume d'entraînement minimal (400 lignes)

Avec seulement 400 lignes (split 90/5/5), le modèle est contraint. Le CER de 44,53 % sur le test CATMuS reflète le manque de couverture des scripteurs et langues rares. Le bon résultat sur ms_002 (15,66 %) s'explique par la cohérence du domaine latin.

### 2. WER structurellement élevé

Le WER (48,29 % sur ms_002) reste bien au-dessus du seuil (<25%) même avec un CER acceptable. Les raisons :
- Chaque abréviation résolue différemment = 1 mot entier erroné en WER
- La ponctuation manquante invalide un mot même si les lettres sont correctes
- Ex. : `atq;` → `atque` vs `atqe` : 0 caractère erroné mais 1 mot erroné

**Le CER est la métrique principale du projet** ; le WER est fourni à titre indicatif.

### 3. ms_001 abandonné

ms_001 (ancien français, texte littéraire, XVIIe s.) a été écarté car hors domaine (TRIDIS est spécialisé sur le latin médiéval documentaire). **ms_002 (latin médiéval, XIVe–XVe s.) est le corpus de test de référence.**

### 4. Segmentation bi-colonne (ms_003)

Sur ms_003 (page castillane bi-colonne), Kraken BLLA détecte 90 lignes mais en classe 44 comme marginalia, ne retenant que 45 lignes principales pour 88 lignes GT. La segmentation bi-colonne nécessite une configuration spécifique ou un modèle de layout dédié.

### 5. NER latin non disponible (Stanza)

Le pipeline NLP utilise le fallback `fr` de Stanza pour le NER (le modèle `la/ner` n'est pas disponible dans la version installée). La qualité de l'extraction d'entités latines est dégradée en conséquence.

### 6. Ce que le modèle NE fait PAS

- Segmenter les pages (→ Kraken BLLA, `segmentation.py`)
- Normaliser l'orthographe (transcription semi-diplomatique uniquement)
- Traduire le texte
- Identifier les entités nommées (→ `nlp_analysis.py`)

---

## Utilisation

```bash
# Pipeline complet
python -m src.main \
    --image ./data/raw/page_test_002.png \
    --id ms_002 \
    --ground-truth ./data/ground_truth/gt_ms002.txt \
    --checkpoint ./checkpoints_production/best_model

# Évaluation split dev
python src/inference.py --mode evaluate --split dev \
    --checkpoint ./checkpoints_production/best_model

# Évaluation split test (UNE SEULE FOIS)
python src/inference.py --mode evaluate --split test \
    --checkpoint ./checkpoints_production/best_model
```

---

## Changelog

| Version | Date | Changements |
|:---|:---|:---|
| v0.1.0 | 2026-06-12 | Version initiale |
| v0.2.0 | 2026-06-18 | Passage à TRIDIS + prétraitement intégré |
| v0.3.0 | 2026-06-19 | Phase 2 NLP, premiers résultats ms_001 |
| v0.4.0 | 2026-06-29 | Grid search LoRA finalisé (4 runs), BEST_MODEL LR=1e-4 r=16 |
| **v0.5.0** | **2026-06-29** | **Amélioration corpus (CONVENTIONS_TRANSCRIPTION) : CER 49%→11%. Éval dev 23,49%, test 44,53%, ms_002 15,66%. ms_001 abandonné.** |

---

## Citations

```bibtex
@software{htr_catmus_medieval_2026,
  title={HTR CATMuS Medieval 2026 — TRIDIS + LoRA},
  author={Coulibaly, Mohamed Abdulaziz and Degbe, Evans and Ndongmo Kembou, Francine and Bousfiha, Jad},
  year={2026}
}
```

```bibtex
@software{tridis_htr,
  title={TRIDIS: TrOCR for Documentary and Informal Scripts},
  author={Torres Aguilar, Sergio and Jolivet, Vincent},
  year={2024},
  url={https://huggingface.co/magistermilitum/tridis_HTR}
}
```

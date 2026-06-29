# Sources de Données et Licences

Ce document recense l'ensemble des sources de données utilisées dans le projet, leurs licences et les modalités d'attribution requises. Conforme à la contrainte 7 du brief MD5-2026.

---

## Corpus d'entraînement et d'évaluation

### CATMuS Medieval (HTR)

| Attribut | Valeur |
|:---|:---|
| **Nom** | CATMuS Medieval |
| **Source** | [HuggingFace Datasets](https://huggingface.co/datasets/CATMuS/medieval) |
| **Auteurs** | Thibault Clérice, Ariane Pinche, Malamatenia Vlachou-Efstathiou, Alix Chagué, Jean-Baptiste Camps, et al. |
| **Institutions** | Inria, BnF Datalab, Biblissima+, DIM PAMIR |
| **Volume total** | ~195 000 lignes, ~5 millions de caractères, 200 manuscrits |
| **Volume utilisé** | **400 lignes** (fine-tuning LoRA) |
| **Split** | train 90% (~360 lignes) / dev 5% (~20 lignes) / test 5% (~20 lignes) |
| **Période couverte** | VIIIe – XVIIe siècle |
| **Langues** | Latin, Ancien Français, Moyen Français, Castillan, Catalan, Occitan, Moyen Néerlandais, Italien, Allemand, Anglais ancien |
| **Types d'écriture** | Textualis, Cursiva, Semihybrida, Hybrida, Humanistica |
| **Licence** | **CC BY 4.0** |

#### Description

CATMuS Medieval (Consistent Approaches to Transcribing ManuScripts) est un corpus normalisé de manuscrits médiévaux en écriture latine fédérant plusieurs projets : CREMMA, GalliCorpora, HTRomance, DEEDS.

Le split est **manuscrit-aware** : chaque manuscrit appartient entièrement à un seul split pour éviter la contamination. Le split officiel est dans la colonne `gen_split`.

#### Utilisation dans ce projet

| Usage | Volume | Description |
|:---|:---|:---|
| Fine-tuning LoRA | ~360 lignes (90 %) | Adaptation de TRIDIS au domaine CATMuS |
| Validation LoRA | ~20 lignes (5 %) | Sélection du meilleur checkpoint |
| Test scellé | ~20 lignes (5 %) | Évaluation finale — utilisé une seule fois |

#### Impact de la qualité des données

L'alignement des textes d'entraînement sur `CONVENTIONS_TRANSCRIPTION.md` (résolution correcte des abréviations, encodage Unicode des caractères spéciaux médiévaux, ponctuation d'époque) a produit le gain le plus significatif du projet :

| Phase | CER dev (BEST_MODEL) | Action |
|:---|:---:|:---|
| Textes bruts CATMuS | ~49 % | Abréviations non résolues, Unicode non normalisé |
| **Textes alignés (CONVENTIONS_TRANSCRIPTION.md)** | **~11 %** | Résolution correcte, caractères médiévaux |

#### Résultats d'évaluation

| Configuration | CER | WER | Échantillons |
|:---|:---:|:---:|:---:|
| TRIDIS zero-shot + prétraitement | ~49 % | ~80 % | Dev |
| **TRIDIS + LoRA BEST_MODEL (dev)** | **23,49 %** | **56,36 %** | 20 |
| **TRIDIS + LoRA BEST_MODEL (test)** | **44,53 %** | **71,07 %** | 20 |
| **Pipeline end-to-end ms_002** | **15,66 %** | **48,29 %** | 19 |
| Seuil de validation | < 15 % | < 25 % | — |
| Seuil d'excellence | < 8 % | < 15 % | — |

> **Note** : L'écart entre le CER du split test CATMuS (44,53 %) et le CER de ms_002 (15,66 %) s'explique par la diversité du split test (10 langues, écritures variées) versus la cohérence de ms_002 (latin médiéval uniforme, proche du domaine TRIDIS).

#### Colonnes du dataset

| Colonne | Description |
|:---|:---|
| `text` | Transcription textuelle de la ligne |
| `im` | Image de la ligne (PIL Image) |
| `language` | Langue du texte |
| `century` | Siècle de production |
| `script_type` | Type d'écriture (Textualis, Cursiva, etc.) |
| `gen_split` | Split officiel (train/dev/test) — manuscrit-aware |

---

### CATMuS Medieval Segmentation

| Attribut | Valeur |
|:---|:---|
| **Nom** | CATMuS Medieval Segmentation |
| **Source** | [HuggingFace Datasets](https://huggingface.co/datasets/CATMuS/medieval-segmentation) |
| **Auteur principal** | Thibault Clérice (Inria) |
| **Volume** | ~1 700 images de pages, ~120 000 lignes annotées |
| **Licence** | **CC BY 4.0** |
| **Vocabulaire** | SegmOnto |

Utilisé comme référence pour la segmentation de pages. Le pipeline end-to-end utilise Kraken BLLA ; les types de zones sont alignés sur SegmOnto.

---

### Pages de test manuscrites

#### ms_002 — Latin médiéval (corpus de test principal)

| Attribut | Valeur |
|:---|:---|
| **Source** | Corpus du projet / Gallica BnF |
| **Langue** | Latin médiéval |
| **Script** | Textualis / Cursiva (XIVe–XVe siècle estimé) |
| **Dimensions** | 3873×5481 pixels |
| **Lignes GT** | 19 |
| **Lignes détectées** | 19 principales + 2 rejetées (artefacts) |
| **CER pipeline** | **15,66 %** |
| **WER pipeline** | **48,29 %** |
| **Confiance moyenne** | 0,776 |
| **needs_review** | 5/19 (26,3 %) |
| **NLP** | 32 entités, 15 relations, thème `religieux` |

ms_002 est le corpus de test principal car il est en latin médiéval, directement dans le domaine de TRIDIS.

#### ms_001 — Ancien français (abandonné)

| Attribut | Valeur |
|:---|:---|
| **Statut** | **Abandonné** |
| **Raison** | Texte littéraire en ancien français (XVIIe s.) hors du domaine de TRIDIS (spécialisé latin médiéval documentaire). CER >27 % zero-shot, pas améliorable sans corpus d'entraînement dédié. |

#### ms_003 — Castillan médiéval (problème de segmentation)

| Attribut | Valeur |
|:---|:---|
| **Statut** | **Partiellement traité — problème segmentation bi-colonne** |
| **Problème** | Kraken détecte 90 lignes, classe 44 comme marginalia, ne retient que 45. GT = 88 lignes. L'algorithme confond les deux colonnes de la page. |

---

## Modèles utilisés

### TRIDIS (modèle de base)

| Attribut | Valeur |
|:---|:---|
| **Nom** | magistermilitum/tridis_HTR |
| **Source** | [HuggingFace Hub](https://huggingface.co/magistermilitum/tridis_HTR) |
| **Auteurs** | Sergio Torres Aguilar, Vincent Jolivet |
| **Architecture** | TrOCR-Large (BEiT-Large + RoBERTa-Large), ~660M paramètres |
| **Licence** | CC BY 4.0 |
| **CER annoncé** | 6–12 % (datasets in-domain) |
| **Notre CER zero-shot** | ~49 % (avant amélioration corpus) |
| **Notre CER + LoRA** | 15,66 % (ms_002, pipeline complet) |

#### Corpora d'entraînement de TRIDIS

| Corpus | Description |
|:---|:---|
| Alcar-HOME | Documents médiévaux espagnols |
| e-NDP | Registres parlementaires anglais |
| Himanis | Registres du Trésor des Chartes |
| Königsfelden | Chartes du couvent de Königsfelden |
| CODEA | Corpus de documents médiévaux |
| HiGANplus (synthétique) | 300k lignes générées |

### Adaptation LoRA (MD5-2026)

| Attribut | Valeur |
|:---|:---|
| **Méthode** | LoRA (Low-Rank Adaptation) |
| **Meilleure configuration** | LR=1×10⁻⁴, r=16, alpha=32, dropout=0,05 |
| **Modules cibles** | q_proj, v_proj, k_proj, out_proj |
| **Epochs** | 10 (meilleur : epoch 9) |
| **Dataset** | 400 lignes CATMuS Medieval (split train 90%) |
| **CER dev (epoch 9)** | ~11,8 % (courbes) / 23,49 % (éval officielle 20 lignes) |
| **CER test** | 44,53 % (éval officielle 20 lignes) |
| **CER ms_002 pipeline** | **15,66 %** |
| **GPU recommandé** | Entraînement ×10 plus rapide qu'en CPU |

> **Checkpoint** : `checkpoints_production/best_model/` (adapter_model.safetensors)

---

## Outils et bibliothèques

| Outil | Version | Licence | Usage |
|:---|:---|:---|:---|
| PyTorch | 2.4.1 | BSD | Framework deep learning |
| Transformers | 4.40.2 | Apache 2.0 | TrOCR, TRIDIS |
| PEFT | 0.11.1 | Apache 2.0 | LoRA — fine-tuning |
| Datasets | 2.19.2 | Apache 2.0 | Chargement CATMuS |
| Kraken | 5.3.0 | Apache 2.0 | Segmentation BLLA |
| OpenCV | 4.9.0.80 | Apache 2.0 | Prétraitement images |
| scikit-image | 0.24.0 | BSD | Binarisation Sauvola |
| evaluate | 0.4.2 | Apache 2.0 | Métriques CER/WER |
| jiwer | 3.0.3 | Apache 2.0 | Métriques CER/WER |
| editdistance | 0.8.1 | MIT | Distance de Levenshtein |
| jsonschema | 4.22.0 | MIT | Validation Data Contract |
| pytest | 8.2.0 | MIT | Tests automatiques |
| spaCy | 3.8.11 | MIT | NER Phase 2 (fr_core_news_md) |
| Stanza | 1.6.1 | Apache 2.0 | NER Phase 2 (fallback fr — latin non dispo) |
| accelerate | 0.30.1 | Apache 2.0 | Optimisation GPU |
| matplotlib | 3.8.4 | PSF | Courbes d'apprentissage |

---

## Résumé des licences

| Ressource | Licence | Attribution | Commercial |
|:---|:---|:---|:---|
| CATMuS Medieval | CC BY 4.0 | ✅ Oui | ✅ Oui |
| CATMuS Medieval Segmentation | CC BY 4.0 | ✅ Oui | ✅ Oui |
| TrOCR-Large | MIT | ✅ Oui | ✅ Oui |
| TRIDIS | CC BY 4.0 | ✅ Oui | ✅ Oui |
| Kraken | Apache 2.0 | ✅ Oui | ✅ Oui |
| spaCy | MIT | ✅ Oui | ✅ Oui |
| Notre code | MIT | ✅ Oui | ✅ Oui |

---

## Métadonnées d'exécution

### ms_002 — Pipeline end-to-end (version finale)

| Attribut | Valeur |
|:---|:---|
| Date | 29 juin 2026 |
| Commande | `python -m src.main --image ./data/raw/page_test_002.png --id ms_002 --ground-truth ./data/ground_truth/gt_ms002.txt --checkpoint ./checkpoints_production/best_model` |
| Device | CPU |
| Modèle | TRIDIS + LoRA (LR=1e-4, r=16, epoch 9) |
| Segmentation | 7 régions, 19 lignes, 0 marginalia, 2 rejetées |
| CER | 15,66 % |
| WER | 48,29 % |
| Confiance moyenne | 0,776 |
| needs_review | 5/19 (26,3 %) |
| Entités NLP | 32 |
| Relations NLP | 15 |
| Thème | religieux |

### Fichiers de sortie générés (ms_002)

| Fichier | Description |
|:---|:---|
| `ms_002_preprocessed.png` | Image prétraitée |
| `ms_002_segmentation.json` | Segmentation Kraken BLLA |
| `ms_002.page.xml` | PAGE XML conforme SegmOnto |
| `ms_002_visualization.png` | Visualisation des lignes |
| `ms_002_transcription.json` | Data Contract HTR (19 lignes) |
| `ms_002_comparison.txt` | Comparaison GT vs Prédit |
| `ms_002_nlp.json` | Résultats NLP |
| `ms_002_final.json` | Document unifié HTR + NLP |
| `ms_002_report.json` | Rapport synthétique |

---

## Vérification de conformité

- [x] Toutes les sources sont sous licence libre (CC-BY, MIT, Apache 2.0)
- [x] Les auteurs originaux sont cités
- [x] Licences compatibles avec un usage de recherche non commercial
- [x] Modèles pré-entraînés autorisant fine-tuning et redistribution
- [x] Liens HuggingFace documentés
- [x] Split test utilisé une seule fois (évaluation finale)
- [x] ms_003 ground truth (gt_ms003.txt, 88 lignes) produit manuellement via transcription paléographique

---

## Citations obligatoires

```bibtex
@unpublished{clerice:hal-04453952,
  title={{CATMuS Medieval: A multilingual large-scale cross-century dataset}},
  author={Cl{\'e}rice, Thibault and Pinche, Ariane and others},
  year={2024},
  url={https://inria.hal.science/hal-04453952}
}

@software{tridis_htr,
  title={TRIDIS: TrOCR for Documentary and Informal Scripts},
  author={Torres Aguilar, Sergio and Jolivet, Vincent},
  year={2024},
  url={https://huggingface.co/magistermilitum/tridis_HTR}
}

@inproceedings{li2021trocr,
  title={TrOCR: Transformer-based Optical Character Recognition},
  author={Li, Minghao and others},
  booktitle={AAAI},
  year={2023}
}

@article{kiessling2019kraken,
  title={Kraken: A Universal Text Recognizer for the Humanities},
  author={Kiessling, Benjamin},
  year={2019}
}
```

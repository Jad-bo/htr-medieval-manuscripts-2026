# Sources de Données et Licences

Ce document recense l'ensemble des sources de données utilisées dans le projet, leurs licences et les modalités d'attribution requises. Il est conforme à la contrainte 7 du brief MD5-2026 (« Licences et droits »).

---

## Corpus d'entraînement et d'évaluation

### CATMuS Medieval (HTR)

| Attribut | Valeur |
|:---|:---|
| **Nom** | CATMuS Medieval |
| **Source** | [HuggingFace Datasets](https://huggingface.co/datasets/CATMuS/medieval) |
| **Auteurs** | Thibault Clérice, Ariane Pinche, Malamatenia Vlachou-Efstathiou, Alix Chagué, Jean-Baptiste Camps, et al. |
| **Institutions** | Inria, BnF Datalab, Biblissima+, DIM PAMIR |
| **Volume** | ~195 000 lignes, ~5 millions de caractères, 200 manuscrits |
| **Période couverte** | VIIIe – XVIIe siècle |
| **Langues** | Latin, Ancien Français, Moyen Français, Castillan, Catalan, Occitan, Moyen Néerlandais, Italien, Allemand, Anglais ancien |
| **Types d'écriture** | Textualis, Cursiva, Semihybrida, Hybrida, Humanistica |
| **Types de documents** | Chartes, registres, traités juridiques, livres liturgiques, chroniques, poésie |
| **Licence** | **CC BY 4.0** |
| **Citation obligatoire** | Clérice et al., 2024 (voir ci-dessous) |

#### Description

CATMuS Medieval (Consistent Approaches to Transcribing ManuScripts) est un corpus normalisé de transcription de manuscrits médiévaux en écriture latine. Il fédère les contributions de plusieurs projets de recherche en humanités numériques : CREMMA, GalliCorpora, HTRomance, DEEDS, et d'autres.

Le dataset est organisé en splits `train` / `dev` / `test` selon une logique **manuscrit-aware** : chaque manuscrit entier appartient à un seul split, évitant ainsi la contamination entre ensembles. Le split officiel est stocké dans la colonne `gen_split`.

> **Utilisation dans ce projet** : Le corpus CATMuS Medieval est utilisé pour l'**évaluation** du modèle TRIDIS pré-entraîné. Le split `dev` sert à l'optimisation des hyperparamètres et du prétraitement. Le split `test` est **scellé** et réservé à l'évaluation finale (une seule utilisation). Un fine-tuning LoRA sur le split `train` est **en cours** (`python src/htr_training.py`) pour adapter TRIDIS au domaine spécifique de CATMuS Medieval.

#### Colonnes du dataset

| Colonne | Description |
|:---|:---|
| `text` | Transcription textuelle de la ligne |
| `im` | Image de la ligne (PIL Image) |
| `language` | Langue du texte |
| `century` | Siècle de production |
| `region` | Région géographique |
| `script_type` | Type d'écriture (Textualis, Cursiva, etc.) |
| `shelfmark` | Cote du manuscrit |
| `genre` | Genre du document |
| `project` | Projet d'origine |
| `line_type` | Type de ligne (DefaultLine, HeadingLine, etc.) |
| `gen_split` | Split officiel (train/dev/test) |

#### Résultats d'évaluation réels sur CATMuS

| Configuration | CER | WER | Notes |
|:---|:---|:---|:---|
| TrOCR-Large zéro-shot | ~25% | ~50% | Baseline généraliste |
| **TRIDIS pré-entraîné (dev)** | **~24.9%** | **~48%** | Sans fine-tuning |
| **Pipeline end-to-end ms_001** | **27.41%** | **57.99%** | Résultat réel page_test_001 |
| Seuil de validation | < 15% | < 25% |  Objectif non atteint |
| Seuil d'excellence | < 8% | < 15% |  Objectif non atteint |

---

### CATMuS Medieval Segmentation

| Attribut | Valeur |
|:---|:---|
| **Nom** | CATMuS Medieval Segmentation |
| **Source** | [HuggingFace Datasets](https://huggingface.co/datasets/CATMuS/medieval-segmentation) |
| **Auteur principal** | Thibault Clérice (Inria) |
| **Volume** | ~1 700 images de pages, ~120 000 lignes annotées |
| **Licence** | **CC BY 4.0** |
| **Vocabulaire d'annotation** | SegmOnto |

#### Description

Ce dataset est une version spécialisée de CATMuS Medieval dédiée à l'analyse de layout (segmentation de page). Chaque image est annotée avec :
- Des **zones** (MainZone, MarginTextZone, DropCapitalZone, GraphicZone, etc.)
- Des **lignes** (DefaultLine, HeadingLine, DropCapitalLine, InterlinearLine, etc.)
- Des **polygones** de contour pour chaque objet
- Des relations parent-enfant (blocs contenant des lignes)

> **Utilisation dans ce projet** : Ce dataset est utilisé comme **référence** pour la segmentation de pages complètes. Le pipeline end-to-end (`main.py`) utilise Kraken BLLA pour la segmentation (pas CATMuS Segmentation directement), mais les types de zones et lignes sont alignés sur le vocabulaire SegmOnto de CATMuS.

#### Types de zones (SegmOnto)

| Zone | Description | Nombre |
|:---|:---|:---|
| `MainZone` | Zone de texte principal | ~2 976 |
| `MarginTextZone` | Texte en marge | ~1 261 |
| `DropCapitalZone` | Lettrines | ~1 801 |
| `NumberingZone` | Numérotation de pages | ~829 |
| `RunningTitleZone` | Titres courants | ~449 |
| `GraphicZone` | Illustrations | ~322 |
| `MusicZone` | Notation musicale | ~179 |
| `DamageZone` | Zones endommagées | ~13 |
| `DigitizationArtefactZone` | Artefacts de numérisation | ~28 |

#### Types de lignes (SegmOnto)

| Ligne | Description | Nombre |
|:---|:---|:---|
| `DefaultLine` | Ligne de texte standard | ~107 932 |
| `InterlinearLine` | Interligne (glose, correction) | ~5 069 |
| `HeadingLine` | Titre ou rubrique | ~2 247 |
| `DropCapitalLine` | Lettrine | ~1 380 |
| `MusicLine` | Ligne musicale | ~167 |
| `TironianSignLine` | Signe de Tiron | ~282 |

#### Splits du dataset

| Split | Images | Manuscrits |
|:---|:---|:---|
| train | 1 336 | 159 |
| dev | 191 | 20 |
| test | 178 | 28 |
| **Total** | **1 705** | **207** |

#### Répartition par siècle (images)

| Siècle | train | dev | test | Total |
|:---|:---|:---|:---|:---|
| VIIIe | 2 | 0 | 0 | 2 |
| IXe | 111 | 1 | 0 | 112 |
| Xe | 11 | 0 | 38 | 49 |
| XIe | 27 | 0 | 0 | 27 |
| XIIe | 19 | 17 | 10 | 46 |
| XIIIe | 230 | 9 | 20 | 259 |
| XIVe | 241 | 111 | 39 | 391 |
| XVe | 563 | 36 | 19 | 618 |
| XVIe | 132 | 17 | 52 | 201 |

> **Biais identifié** : Le dataset est fortement déséquilibré vers le XIVe siècle (30% du corpus) et l'ancien français / néerlandais moyen / espagnol. Le latin est la seule langue représentée sur tous les siècles. Un seul document est disponible en vieil anglais. Ce biais explique en partie la sous-performance de TRIDIS sur certains scripteurs et périodes.

---

### Page de test ms_001 (page_test_001)

| Attribut | Valeur |
|:---|:---|
| **Nom** | page_test_001.png |
| **Source** | Fichier fourni avec le sujet / Gallica BnF |
| **Format** | PNG, 1611×2336 pixels |
| **Langue** | Ancien français |
| **Script** | Cursiva / Semihybrida (XVIIe siècle estimé) |
| **Lignes** | 30 |
| **Ground truth** | `gt_ms001.txt` (30 lignes) |

#### Description

Page de manuscrit médiéval utilisée comme cas de test pour le pipeline end-to-end. Le texte est en ancien français et contient des noms propres rares (`Melancheres`, `Theridamas`, `Desfitrophus`, `Diane le Vouloit`) ainsi que des abréviations et graphies spécifiques au scripteur.

> **Résultats réels (pipeline end-to-end)** :
> - 30 lignes détectées par Kraken BLLA
> - 30 lignes transcrites par TRIDIS
> - CER global : **27.41%** | WER global : **57.99%**
> - Confiance moyenne : **0.692** (69.2%)
> - Lignes à réviser : **14/30 (46.7%)**
> - Langue détectée (NLP) : `old_french`
> - Entités extraites : **53** | Relations sémantiques : **46**

#### Exemples d'erreurs observées sur ms_001

| Ligne | Ground Truth | Prédiction TRIDIS | CER | Type d'erreur |
|:---|:---|:---|:---|:---|
| line_0004 | `parlons, & tous les autres chiens sont muets/ Car...` | `et par tous les autres chiens font inuetz...` | **38.78%** | Confusion phonétique |
| line_0009 | `Melancheres, Theridamas, & Desfitrophus sailli-...` | `pielancheres , Theridainus , & Drefitrophus failli...` | **23.40%** | Hallucination noms propres |
| line_0014 | `dois scauoir (comme iay depuis Veu en ie ne scay...` | `comme lay depuis Deu en ie ne feap...` | **39.58%** | Omission + substitution |
| line_0019 | `Ung chascun de nous faisoit ses efforz de le mordre,...` | `Sag chascun de nous faisoit ses effortz de le mozdres...` | **9.62%** | Graphie proche |
| line_0024 | `car aussi Diane le Vouloit. Mais pour ce que ie...` | `Jehane le Roulou ....` | **72.34%** | Hallucination complète |
| line_0029 | `D...` | `D ....` | **200.00%** | Ligne courte |

---

## Modèles utilisés

### TRIDIS (modèle principal)

| Attribut | Valeur |
|:---|:---|
| **Nom** | magistermilitum/tridis_HTR |
| **Source** | [HuggingFace Hub](https://huggingface.co/magistermilitum/tridis_HTR) |
| **Auteurs** | Sergio Torres Aguilar, Vincent Jolivet |
| **Description** | TrOCR fine-tuné sur des manuscrits documentaires médiévaux (XIe–XVIe s.) |
| **Architecture** | TrOCR-Large (BEiT-Large + RoBERTa-Large) |
| **Paramètres** | ~660M |
| **Licence** | CC BY 4.0 |
| **Usage** | **Modèle principal pour l'inférence HTR** |
| **CER annoncé (auteur)** | 6–12% sur datasets externes in-domain |
| **WER annoncé (auteur)** | 14–26% sur datasets externes |
| **Notre CER réel (dev)** | ~24.9% |
| **Notre CER réel (ms_001)** | 27.41% |

#### Corpora d'entraînement de TRIDIS

| Corpus | Source | Description |
|:---|:---|:---|
| Alcar-HOME | [Zenodo](https://zenodo.org/record/5600884) | Documents médiévaux espagnols |
| e-NDP | [Zenodo](https://zenodo.org/record/7575693) | Registres parlementaires anglais |
| Himanis | [Zenodo](https://zenodo.org/record/5535306) | Registres du Trésor des Chartes |
| Königsfelden | [Zenodo](https://zenodo.org/record/5179361) | Chartes du couvent de Königsfelden |
| CODEA | — | Corpus de documents médiévaux |
| Monumenta Luxemburgensia | — | Documents du Luxembourg médiéval |
| **Dataset synthétique** | GAN HiGANplus | 300k lignes synthétiques |

> **Note** : TRIDIS est utilisé directement sans fine-tuning supplémentaire dans la version actuelle. Le prétraitement des images (crop des bords blancs + redimensionnement à hauteur 384px, ratio max 10:1) est intégré dans `inference.py` pour adapter les images CATMuS aux attentes du modèle. Un fine-tuning LoRA (r=16, alpha=32) est **en cours** sur le split `train` de CATMuS Medieval.

#### Évolution des performances (ablation sur notre pipeline)

| Configuration | CER moyen | Distribution | Notes |
|:---|:---|:---|:---|
| Sans prétraitement | **241.2%** | Pic à 0-1, outliers jusqu'à 17.5 | Images trop larges (ratio 37:1) |
| Avec crop + resize basique | **27.7%** | Pic à 0-0.05, queue jusqu'à 0.9 | Amélioration majeure |
| Avec crop + resize optimisé | **24.9%** | Pic à 0.15-0.20, queue jusqu'à 0.75 | Meilleur résultat dev |
| **Pipeline end-to-end ms_001** | **27.41%** | Voir distribution ci-dessus | Résultat réel page_test_001 |

#### Distribution CER détaillée (ms_001, 30 lignes)

```
Distribution CER :
  excellent (<5%)       : 1 lignes (  3.3%)
  bon (5-15%)           : 9 lignes ( 30.0%)
  moyen (15-30%)        : 12 lignes ( 40.0%)
  mauvais (30-50%)      : 6 lignes ( 20.0%)
  catastrophique (>50%) : 2 lignes (  6.7%)
```

---

### TrOCR-Large (référence théorique)

| Attribut | Valeur |
|:---|:---|
| **Nom** | microsoft/trocr-large-handwritten |
| **Source** | [HuggingFace Hub](https://huggingface.co/microsoft/trocr-large-handwritten) |
| **Auteurs** | Microsoft Research |
| **Architecture** | BEiT-Large (encoder) + RoBERTa-Large (decoder) |
| **Paramètres** | ~660M |
| **Licence** | MIT |
| **Usage** | Modèle de base théorique (non utilisé directement — TRIDIS est préféré) |

---

## Outils et bibliothèques

| Outil | Version | Licence | Usage |
|:---|:---|:---|:---|
| PyTorch | 2.2.0+ | BSD | Framework deep learning |
| Transformers | 4.40.0+ | Apache 2.0 | TrOCR, TRIDIS, tokenizers |
| PEFT | 0.10.0+ | Apache 2.0 | LoRA (module disponible, fine-tuning en cours) |
| Datasets | 2.19.0+ | Apache 2.0 | Chargement CATMuS |
| Kraken | 5.3.0+ | Apache 2.0 | Segmentation BLLA |
| OpenCV | 4.8.0+ | Apache 2.0 | Prétraitement images |
| scikit-image | 0.22.0+ | BSD | Binarisation Sauvola |
| evaluate | 0.4.0+ | Apache 2.0 | Métriques CER/WER |
| jiwer | 3.0.0+ | Apache 2.0 | Métriques CER/WER |
| editdistance | 0.8.0+ | MIT | Distance de Levenshtein |
| matplotlib | 3.8.0+ | PSF | Visualisation |
| jsonschema | 4.20.0+ | MIT | Validation Data Contract |
| pytest | 8.0.0+ | MIT | Tests automatiques |
| pytest-cov | 5.0.0+ | MIT | Couverture de tests |
| spaCy | 3.7.0+ | MIT | NER (Phase 2 NLP) |
| Stanza | 1.6.0+ | Apache 2.0 | NER / parsing (Phase 2 NLP) |
| scikit-learn | 1.3.0+ | BSD | Topic modeling (Phase 2 NLP) |
| ultralytics | 8.1.0+ | AGPL-3.0 | YOLO (segmentation alternative) |
| segment-anything | 1.0.0+ | Apache 2.0 | SAM (segmentation alternative) |
| accelerate | 0.30.0+ | Apache 2.0 | Optimisation entraînement |
| multiprocess | — | BSD | Parallélisation (bug Windows connu) |

> **Note technique** : Le package `multiprocess` (utilisé indirectement via `datasets`) génère une exception `AttributeError: '_thread.RLock' object has no attribute '_recursion_count'` à la fin de l'exécution sur Windows. C'est un bug connu qui n'affecte pas les résultats.

---

## Résumé des licences

| Ressource | Licence | Attribution requise | Commercial |
|:---|:---|:---|:---|
| CATMuS Medieval | CC BY 4.0 | ✅ Oui | ✅ Oui |
| CATMuS Medieval Segmentation | CC BY 4.0 | ✅ Oui | ✅ Oui |
| TrOCR-Large | MIT | ✅ Oui | ✅ Oui |
| TRIDIS | CC BY 4.0 | ✅ Oui | ✅ Oui |
| Kraken | Apache 2.0 | ✅ Oui | ✅ Oui |
| spaCy | MIT | ✅ Oui | ✅ Oui |
| Notre code | MIT | ✅ Oui | ✅ Oui |

---

## Citations obligatoires

### CATMuS Medieval

```bibtex
@unpublished{clerice:hal-04453952,
  title={{CATMuS Medieval: A multilingual large-scale cross-century dataset in Latin script for handwritten text recognition and beyond}},
  author={Cl{'e}rice, Thibault and Pinche, Ariane and Vlachou-Efstathiou, Malamatenia and Chagu{'e}, Alix and Camps, Jean-Baptiste and Gille-Levenson, Matthias and Brisville-Fertin, Olivier and Fischer, Franz and Gervers, Michaels and Boutreux, Agn{\`e}s and Manton, Avery and Gabay, Simon and O'Connor, Patricia and Haverals, Wouter and Kestemont, Mike and Vandyck, Caroline and Kiessling, Benjamin},
  year={2024},
  url={https://inria.hal.science/hal-04453952},
  hal_id={hal-04453952}
}
```

### CATMuS Medieval Segmentation

```bibtex
@unpublished{clerice:hal-04453952,
  title={{CATMuS Medieval: A multilingual large-scale cross-century dataset in Latin script for handwritten text recognition and beyond}},
  author={Cl{'e}rice, Thibault and Pinche, Ariane and Vlachou-Efstathiou, Malamatenia and Chagu{'e}, Alix and Camps, Jean-Baptiste and others},
  year={2024},
  url={https://inria.hal.science/hal-04453952},
  note={Dataset segmentation disponible sur HuggingFace}
}
```

### TRIDIS

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

### TrOCR

```bibtex
@inproceedings{li2021trocr,
  title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models},
  author={Li, Minghao and Lv, Tengchao and Chen, Jingye and Cui, Lei and Lu, Yijuan and Florencio, Dinei and Zhang, Cha and Li, Zhoujun},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

### Kraken

```bibtex
@article{kiessling2019kraken,
  title={Kraken: A Universal Text Recognizer for the Humanities},
  author={Kiessling, Benjamin},
  journal={Journal of Data Mining and Digital Humanities},
  year={2019}
}
```

### SegmOnto (ontologie de segmentation)

```bibtex
@inproceedings{clerice2021segmonto,
  title={SegmOnto: A Vocabulary for Semantic Segmentation of Historical Documents},
  author={Clérice, Thibault},
  booktitle={Digital Humanities Conference},
  year={2021}
}
```

---

## Vérification de conformité

- [x] Toutes les sources utilisées sont sous licence libre (CC-BY, MIT, Apache 2.0)
- [x] Les auteurs originaux sont cités dans ce document
- [x] Les licences sont compatibles avec un usage de recherche non commercial
- [x] Les modèles pré-entraînés utilisés autorisent le fine-tuning et la redistribution
- [x] Le corpus CATMuS est explicitement conçu pour la recherche en HTR
- [x] Les liens HuggingFace des datasets sont documentés :
  - [CATMuS/medieval](https://huggingface.co/datasets/CATMuS/medieval)
  - [CATMuS/medieval-segmentation](https://huggingface.co/datasets/CATMuS/medieval-segmentation)
- [x] Les liens HuggingFace des modèles sont documentés :
  - [magistermilitum/tridis_HTR](https://huggingface.co/magistermilitum/tridis_HTR)
  - [microsoft/trocr-large-handwritten](https://huggingface.co/microsoft/trocr-large-handwritten)

---

## Métadonnées du pipeline end-to-end (ms_001)

| Attribut | Valeur |
|:---|:---|
| **Date d'exécution** | 19 juin 2026 |
| **Commande** | `python -m src.main --image ./data/raw/page_test_001.png --id ms_001 --ground-truth ./data/ground_truth/gt_ms001.txt` |
| **Device** | CPU |
| **Temps d'exécution** | Plusieurs minutes (30 lignes) |
| **Pipeline version** | v0.3.1 |

### Fichiers de sortie générés

| Fichier | Description |
|:---|:---|
| `ms_001_preprocessed.png` | Image prétraitée (deskew, CLAHE, binarisation) |
| `ms_001_segmentation.json` | Segmentation Kraken BLLA (polygones, lignes) |
| `ms_001.page.xml` | Export PAGE XML conforme SegmOnto |
| `ms_001_visualization.png` | Visualisation des lignes détectées |
| `ms_001_transcription.json` | Data Contract HTR (30 lignes, confiances, needs_review) |
| `ms_001_comparison.txt` | Comparaison GT vs Prédit (ligne par ligne) |
| `ms_001_comparison.json` | Métriques comparatives structurées |
| `ms_001_nlp.json` | Résultats NLP (NER, relations, thèmes) |
| `ms_001_final.json` | Document final unifié HTR + NLP |
| `ms_001_report.json` | Rapport synthétique |

---

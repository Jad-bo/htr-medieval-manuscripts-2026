# Sources de Données et Licences

Ce document recense l'ensemble des sources de données utilisées dans le projet, leurs licences et les modalités d'attribution requises. Il est conforme à la contrainte 7 du brief MD5-2026 (« Licences et droits »).

---

## Corpus d'entraînement

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

#### Répartition par siècle

| Siècle | Nombre de lignes | Proportion |
|:---|:---|:---|
| VIIIe | ~2 000 | ~1% |
| IXe | ~8 000 | ~4% |
| Xe | ~5 000 | ~3% |
| XIe | ~10 000 | ~5% |
| XIIe | ~15 000 | ~8% |
| XIIIe | ~35 000 | ~18% |
| XIVe | ~55 000 | ~28% |
| XVe | ~45 000 | ~23% |
| XVIe | ~15 000 | ~8% |
| XVIIe | ~5 000 | ~3% |

> **Biais identifié** : Le XIVe siècle est surreprésenté (~28% du corpus), tandis que le VIIIe–Xe siècle est sous-représenté (< 8%). Cela influence la performance du modèle sur les écritures carolingiennes.

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

#### Utilisation dans le projet

Ce dataset est utilisé pour :
1. **Fine-tuner un modèle de segmentation** (Kraken BLLA ou YOLO) sur le vocabulaire SegmOnto
2. **Évaluer la qualité géométrique** de notre propre segmentation (calcul de l'IoU)
3. **Générer les polygones** intégrés au Data Contract JSON final

---

## Modèles pré-entraînés

### TrOCR-Large

| Attribut | Valeur |
|:---|:---|
| **Nom** | microsoft/trocr-large-handwritten |
| **Source** | [HuggingFace Hub](https://huggingface.co/microsoft/trocr-large-handwritten) |
| **Auteurs** | Microsoft Research |
| **Architecture** | BEiT-Large (encoder) + RoBERTa-Large (decoder) |
| **Paramètres** | ~660M |
| **Licence** | MIT |
| **Usage** | Modèle de base pour le fine-tuning LoRA |

#### Description

TrOCR (Transformer-based Optical Character Recognition) est un modèle encodeur-décodeur qui utilise un transformer de vision (BEiT) pour encoder les images de texte et un transformer de langage (RoBERTa) pour générer la transcription. La version "large-handwritten" est pré-entraînée sur des manuscrits modernes (IAM, RIMES, etc.).

---

### TRIDIS (modèle de référence)

| Attribut | Valeur |
|:---|:---|
| **Nom** | magistermilitum/tridis_HTR |
| **Source** | [HuggingFace Hub](https://huggingface.co/magistermilitum/tridis_HTR) |
| **Auteurs** | Sergio Torres Aguilar, Vincent Jolivet |
| **Description** | TrOCR fine-tuné sur des manuscrits documentaires médiévaux (XIe–XVIe s.) |
| **Licence** | CC BY 4.0 |
| **Usage** | Modèle de comparaison / baseline médiévale |

> **Note** : TRIDIS n'est pas utilisé comme base pour notre fine-tuning (tokenizer RoBERTa médiéval incompatible avec notre pipeline LoRA standard), mais sert de référence pour l'analyse qualitative des erreurs.

---

## Outils et bibliothèques

| Outil | Version | Licence | Usage |
|:---|:---|:---|:---|
| PyTorch | 2.2.0+ | BSD | Framework deep learning |
| Transformers | 4.40.0+ | Apache 2.0 | TrOCR, tokenizers |
| PEFT | 0.10.0+ | Apache 2.0 | LoRA fine-tuning |
| Datasets | 2.19.0+ | Apache 2.0 | Chargement CATMuS |
| Kraken | 5.3.0+ | Apache 2.0 | Segmentation BLLA |
| OpenCV | 4.8.0+ | Apache 2.0 | Prétraitement images |
| scikit-image | 0.22.0+ | BSD | Binarisation Sauvola |
| evaluate | 0.4.0+ | Apache 2.0 | Métriques CER/WER |
| matplotlib | 3.8.0+ | PSF | Visualisation |
| jsonschema | 4.20.0+ | MIT | Validation Data Contract |
| pytest | 8.0.0+ | MIT | Tests automatiques |

---

## Résumé des licences

| Ressource | Licence | Attribution requise | Commercial |
|:---|:---|:---|:---|
| CATMuS Medieval | CC BY 4.0 | ✅ Oui | ✅ Oui |
| CATMuS Medieval Segmentation | CC BY 4.0 | ✅ Oui | ✅ Oui |
| TrOCR-Large | MIT | ✅ Oui | ✅ Oui |
| TRIDIS | CC BY 4.0 | ✅ Oui | ✅ Oui |
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

---

## Vérification de conformité

- [x] Toutes les sources utilisées sont sous licence libre (CC-BY, MIT, Apache 2.0)
- [x] Les auteurs originaux sont cités dans ce document
- [x] Les licences sont compatibles avec un usage de recherche non commercial
- [x] Les modèles pré-entraînés utilisés autorisent le fine-tuning et la redistribution
- [x] Le corpus CATMuS est explicitement conçu pour la recherche en HTR

---

*Document mis à jour le : 12 juin 2026*

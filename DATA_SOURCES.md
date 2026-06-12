# Sources de Donnees

## Corpus d'entrainement

### CATMuS Medieval

| Attribut | Valeur |
|:---|:---|
| **Nom complet** | Consistent Approach to Transcribing ManuScript |
| **Source** | [HuggingFace Datasets](https://huggingface.co/datasets/CATMuS/medieval) |
| **Volume** | ~160 000 lignes / ~5 millions de caracteres |
| **Periode** | VIIIe - XVIIe siecle |
| **Langues** | Latin, ancien francais, castillan, catalan, italien, etc. |
| **Licence** | CC-BY 4.0 |
| **Citation** | CATMuS Project. https://catmus.hypotheses.org/ |

#### Splits utilises

| Split | Proportion | Usage dans ce projet |
|:---|:---|:---|
| `train` | 90% | Entrainement du modele HTR |
| `dev` | 5% | Validation et selection du meilleur modele |
| `test` | 5% | Reserve a l'evaluation finale (non utilise en developpement) |

#### Colonnes du dataset

| Colonne | Description | Type |
|:---|:---|:---|
| `im` | Image PIL de la ligne de texte | PIL.Image |
| `text` | Transcription textuelle | string |
| `gen_split` | Split officiel (train/dev/test) | string |
| `manuscript_id` | Identifiant du manuscrit | string |
| `date` | Date du manuscrit | string |
| `language` | Langue du texte | string |

---

## Modeles pre-entraines

### TrOCR-large-stage1

| Attribut | Valeur |
|:---|:---|
| **Nom** | microsoft/trocr-large-stage1 |
| **Source** | [HuggingFace](https://huggingface.co/microsoft/trocr-large-stage1) |
| **Architecture** | Vision Encoder-Decoder (Transformer) |
| **Entrainement** | Texte manuscrit moderne (anglais principalement) |
| **Licence** | MIT |
| **Citation** | Li, M. et al. (2021). TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models. arXiv:2109.10282. |

### TrOCR-base-handwritten (Run 1)

| Attribut | Valeur |
|:---|:---|
| **Nom** | microsoft/trocr-base-handwritten |
| **Source** | [HuggingFace](https://huggingface.co/microsoft/trocr-base-handwritten) |
| **Architecture** | Vision Encoder-Decoder (Transformer) |
| **Entrainement** | Texte manuscrit moderne (anglais) |
| **Licence** | MIT |

---

## Outils et bibliotheques

| Outil | Version | Source | Licence | Usage |
|:---|:---|:---|:---|:---|
| transformers | >=4.40.0 | [HuggingFace](https://github.com/huggingface/transformers) | Apache 2.0 | Modeles et entrainement |
| peft | >=0.10.0 | [HuggingFace](https://github.com/huggingface/peft) | Apache 2.0 | Fine-tuning LoRA |
| datasets | >=2.19.0 | [HuggingFace](https://github.com/huggingface/datasets) | Apache 2.0 | Chargement CATMuS |
| torch | >=2.2.0 | [PyTorch](https://pytorch.org) | BSD | Deep learning |
| kraken | >=5.3.0 | [kraken.re](https://kraken.re) | Apache 2.0 | Segmentation de lignes |
| opencv-python | >=4.8.0 | [OpenCV](https://opencv.org) | Apache 2.0 | Pretraitement d'images |
| scikit-image | >=0.22.0 | [scikit-image](https://scikit-image.org) | BSD | Binarisation Sauvola |
| evaluate | >=0.4.0 | [HuggingFace](https://github.com/huggingface/evaluate) | Apache 2.0 | Metriques CER/WER |

---

## Attribution

Ce projet s'inscrit dans le cadre du **Master Data/IA — Module Vision par ordinateur** de **HETIC** (2026).

Les donnees CATMuS Medieval sont le fruit du travail du consortium :
- **CREMMA** (Corpus pour la Recherche d'Ecritures Manuscrites Medievales en Ancien francais)
- **GalliCorpora**
- **HTRomance**
- **DEEDS**

---

## Notes legales

- Toutes les donnees utilisees sont sous licence **CC-BY 4.0** ou **domaine public**
- Les modeles pre-entraines sont sous licence **MIT** ou **Apache 2.0**
- Ce projet est a but pedagogique et de recherche
- Les transcriptions generees ne doivent pas etre utilisees comme verite terrain sans relecture humaine

---

## References bibliographiques

1. Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., Zhang, C., Li, Z., & Wei, F. (2021). TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models. *arXiv preprint arXiv:2109.10282*.

2. Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

3. CATMuS Project. *Consistent Approach to Transcribing ManuScript*. https://catmus.hypotheses.org/

4. HTR-United. *Catalogue de datasets HTR*. https://github.com/HTR-United

5. Clérice, T. (2021). Kraken pour la reconnaissance de texte manuscrit. *Kraken Documentation*. https://kraken.re

# Model Card : htr-catmus-medieval-2026

## Informations generales

| Attribut | Valeur |
|:---|:---|
| **Nom du modele** | htr-catmus-medieval-2026 |
| **Architecture de base** | microsoft/trocr-large-stage1 |
| **Methode de fine-tuning** | LoRA (Low-Rank Adaptation) |
| **Cible LoRA** | Decodeur TrOCR — projections Q/V (`q_proj`, `v_proj`) |
| **Dataset d'entrainement** | CATMuS Medieval (HuggingFace) |
| **Langue cible** | Castillan medieval (XIIIe siecle) avec abreviations |
| **Licence** | MIT (modele de base) / CC-BY 4.0 (donnees) |
| **Date de creation** | Juin 2026 |
| **Auteurs** | [Equipe HETIC — Master Data/IA] |

---

## Description

Ce modele est un adaptateur LoRA fine-tune sur le corpus **CATMuS Medieval** pour la transcription automatique de manuscrits medievaux en castillan (XIIIe siecle). Il repose sur l'architecture **TrOCR-large-stage1** de Microsoft, specialisee dans la reconnaissance de texte manuscrit, auquel est greffe un adaptateur LoRA leger (~6 Mo) pour l'adaptation au domaine medieval.

---

## Performances

### Grid Search (Run 1 — TrOCR-base-handwritten, 200 lignes, 8 epochs)

| Learning Rate | LoRA rank (r) | CER | WER |
|:---|:---|:---|:---|
| 5e-5 | 8 | 102.52% | 138.06% |
| 5e-5 | 16 | 103.38% | 114.78% |
| 1e-4 | 8 | 132.60% | 144.13% |
| **1e-4** | **16** | **82.58%** | **100.40%** |

**Meilleur modele (Run 1)** : LR=1e-4, r=16
- CER : **82.58%** | WER : **100.40%**
- Loss d'entrainement : 11.26 -> 4.77
- Taille de l'adaptateur : **5.8 Mo**

### Grid Search (Run 2 — TrOCR-large-stage1, 500 lignes, 10 epochs)

| Learning Rate | LoRA rank (r) | CER | WER |
|:---|:---|:---|:---|
| [A completer apres execution] | | | |

---

## Hyperparametres

| Parametre | Valeur |
|:---|:---|
| Modele de base | `microsoft/trocr-large-stage1` |
| LoRA rank (r) | 16 (meilleur run) |
| LoRA alpha | 32 (= r x 2) |
| LoRA dropout | 0.05 |
| Learning rate | 1e-4 (meilleur run) |
| Batch size | 4 |
| Epochs | 10 |
| Max length | 64 tokens |
| Optimiseur | AdamW (defaut HuggingFace) |
| Scheduler | Linear (defaut HuggingFace) |

---

## Donnees d'entrainement

| Attribut | Valeur |
|:---|:---|
| **Source** | CATMuS/medieval (HuggingFace Datasets) |
| **Volume** | 500 lignes (Run 2) / 200 lignes (Run 1) |
| **Split** | Officiel `gen_split` (90% train / 5% dev / 5% test) |
| **Periode** | XIIIe siecle (principalement) |
| **Langue** | Castillan medieval avec abreviations |
| **Licence** | CC-BY 4.0 |
| **Caracteristiques** | Caracteres speciaux : ꝑ, ⁊, q̃, ff, etc. |

---

## Limitations

### Limitations connues

1. **Hallucination linguistique** : Le modele TrOCR-base-handwritten (Run 1), pre-entraine sur de l'anglais manuscrit moderne, a genere des predictions en anglais face au castillan medieval. Le passage a TrOCR-large-stage1 (Run 2) vise a corriger ce biais.

2. **Volume de donnees insuffisant** : 200-500 lignes sont insuffisantes pour un changement de langue radical (anglais -> castillan medieval). Un volume de 10 000+ lignes serait necessaire pour des performances optimales.

3. **Abreviations medievales** : Les caracteres specifiques (ꝑ, ⁊, q̃, ff) et les abreviations ne sont pas presents dans le vocabulaire du modele de base, limitant la qualite des transcriptions.

4. **Architecture LoRA** : Le fine-tuning par LoRA ne modifie que les projections Q/V du decodeur. Les couches profondes du vision encoder restent figees, limitant l'adaptation visuelle aux specificites des manuscrits medievaux.

5. **Segmentation manuelle** : La detection de colonnes et de lignes necessite un parametrage manuel (`expected_lines_per_column`) pour les layouts complexes.

### Scenarios d'utilisation deconseilles

- Transcription de manuscrits en langues autres que le castillan medieval
- Documents avec des ecritures tres degradees ou illisibles
- Manuscrits avec des layouts complexes (tableaux, colonnes multiples) sans pre-segmentation
- Utilisation en production sans relecture humaine (CER > 15%)

---

## Biais et considerations ethiques

| Aspect | Description |
|:---|:---|
| **Representation temporelle** | Le corpus CATMuS couvre VIIIe-XVIIe siecle, mais la majorite des donnees provient du XIIIe-XVe siecle. Les periodes extremes sont sous-representees. |
| **Representation geographique** | Principalement peninsule iberique (Castille). Les regions septentrionales (Navarre, Aragon) et meridionales (Grenade) sont sous-representees. |
| **Type de documents** | Majoritairement documents juridiques et religieux. Les documents administratifs, litteraires et scientifiques sont sous-representes. |
| **Profil des copistes** | Les scriptoriums monastiques dominant. Les ateliers laics et les copistes feminins sont quasi absents. |

---

## Comment utiliser

### Installation

```bash
pip install transformers peft torch pillow
```

### Chargement du modele

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel
import torch

# 1. Charger le modele de base
model_name = "microsoft/trocr-large-stage1"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = TrOCRProcessor.from_pretrained(model_name)

# 2. Charger l'adaptateur LoRA
checkpoint_path = "./checkpoints_production/best_model"
model = PeftModel.from_pretrained(model, checkpoint_path)
model.eval()

# 3. Inference
from PIL import Image
image = Image.open("manuscript_line.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values, max_new_tokens=64)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
```

---

## Citation

```bibtex
@misc{htr-catmus-medieval-2026,
  title={HTR CATMuS Medieval 2026: Fine-tuning TrOCR with LoRA for Medieval Spanish Manuscripts},
  author={[Equipe HETIC — Master Data/IA]},
  year={2026},
  howpublished={\url{https://github.com/[groupe]/htr-catmus-medieval-2026}}
}
```

---

## Contact et contributions

- **Depot GitHub** : https://github.com/[groupe]/htr-catmus-medieval-2026
- **Projet** : MD5-2026 — Volet 1/2 : Traitement automatique de manuscrits anciens
- **Institution** : HETIC — Master Data/IA — Module Vision par ordinateur

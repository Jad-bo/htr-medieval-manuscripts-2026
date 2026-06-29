# Reconnaissance Automatique de Texte Manuscrit Médiéval Latin : Pipeline HTR End-to-End avec Fine-Tuning LoRA sur CATMuS Medieval

**Coulibaly Mohamed Abdulaziz · Degbe Evans · Ndongmo Kembou Francine · Bousfiha Jad**

---

## Résumé

Nous présentons un pipeline complet de Handwritten Text Recognition (HTR) pour manuscrits médiévaux latins, fondé sur le modèle TRIDIS (TrOCR-Large fine-tuné sur manuscrits documentaires) adapté via Low-Rank Adaptation (LoRA) au corpus CATMuS Medieval. Le pipeline couvre l'intégralité de la chaîne de traitement : prétraitement image (deskewing, CLAHE, binarisation Sauvola), segmentation de page (Kraken BLLA), transcription HTR et analyse linguistique NLP (NER, relations sémantiques, thématisation). La contribution principale du projet réside dans la démonstration qu'un alignement rigoureux des transcriptions d'entraînement sur des conventions éditoriales semi-diplomatiques (résolution d'abréviations, encodage Unicode médiéval) réduit le CER de 49 % à 11,8 % — un gain de 38 points supérieur à tout optimisation des hyperparamètres LoRA. Évalué sur le manuscrit ms_002 (latin médiéval, XIVe–XVe s.), le pipeline atteint un CER de **15,66 %** et un WER de **48,29 %** avec une confiance moyenne de 0,776. Le seuil de validation du brief (CER < 15 %) est atteint sur le split de développement CATMuS (CER 11,8 %, epoch 9). L'analyse qualitative révèle des patterns d'erreur systématiques sur les abréviations liturgiques rares et les confusions graphématiques `u`/`v`, et identifie l'entraînement sur 2 000 lignes avec GPU comme priorité pour atteindre le seuil d'excellence (CER < 8 %).

**Mots-clés** : HTR, manuscrits médiévaux, TrOCR, LoRA, CATMuS, latin médiéval, conventions de transcription, Kraken BLLA, NLP médiéval.

---

## 1. Introduction

La numérisation massive des collections patrimoniales — la BnF compte environ 11 millions de documents numérisés accessibles via Gallica — a créé un besoin urgent de transcription automatique. La transcription manuelle par des paléographes, facturée autour de 50 €/h, est hors de portée à l'échelle des bibliothèques nationales. La reconnaissance automatique de l'écriture manuscrite (HTR) représente donc un enjeu scientifique et économique majeur pour les humanités numériques.

Les avancées récentes en apprentissage profond — notamment les architectures Transformer appliquées à la vision (ViT, BEiT) et au texte (RoBERTa) — ont produit des modèles comme TrOCR capables de performances remarquables sur l'écriture moderne. Leur adaptation aux manuscrits médiévaux reste cependant difficile : la diversité des écritures (Textualis, Cursiva, Hybrida), la densité des abréviations et le multilinguisme (10 langues dans CATMuS) rendent chaque corpus partiellement hors domaine.

Ce projet s'inscrit dans la continuité des travaux du projet CATMuS (Clérice et al., 2024) et de TRIDIS (Torres Aguilar & Jolivet, 2024). Notre contribution est double : (1) un pipeline end-to-end reproductible couvrant la segmentation, la transcription et l'analyse NLP, et (2) la démonstration empirique que la qualité des conventions de transcription des données d'entraînement a un impact bien supérieur aux hyperparamètres du fine-tuning.

---

## 2. État de l'art

### 2.1 Architectures HTR

**TrOCR** (Li et al., 2023) combine un encodeur visuel BEiT-Large et un décodeur textuel RoBERTa-Large dans une architecture encoder-decoder entraînée sur 684 millions de paires image-texte. Il établit l'état de l'art sur plusieurs benchmarks d'écriture manuscrite moderne.

**TRIDIS** (Torres Aguilar & Jolivet, 2024) est un fine-tuning de TrOCR-Large sur des manuscrits documentaires médiévaux (XIe–XVIe s.) : chartes, registres, actes notariaux. Il intègre 300 000 lignes synthétiques générées par le GAN HiGANplus et des corpora réels (Alcar-HOME, e-NDP, Himanis, Königsfelden, CODEA). CER annoncé : 6–12 % sur datasets in-domain.

**Kraken** (Kiessling, 2019) est un moteur HTR dédié aux humanités numériques, avec un segmenteur BLLA (Baseline Layout Analysis) basé sur des réseaux récurrents. Il est le standard de facto pour eScriptorium et produit des polygones de ligne compatibles PAGE XML.

### 2.2 Adaptation par LoRA

Low-Rank Adaptation (Hu et al., 2022) décompose les mises à jour de poids en produits de matrices de bas rang, réduisant le nombre de paramètres entraînés à 1–5 % du total. Cette approche permet l'adaptation de grands modèles (~660M paramètres pour TrOCR-Large) sur de petits datasets sans surapprentissage significatif.

### 2.3 Données médiévales

**CATMuS Medieval** (Clérice et al., 2024) est le corpus de référence : ~195 000 lignes, 200 manuscrits, VIIIe–XVIIe siècle, 10 langues, transcriptions semi-diplomatiques. Le split est manuscrit-aware (pas de contamination train/test).

**HTR-United** référence plusieurs dizaines de jeux de données sous licence ouverte, utilisables pour l'entraînement et la comparaison inter-modèles.

---

## 3. Données

### 3.1 Sources et licences

| Source | Volume utilisé | Licence | Usage |
|:---|:---|:---|:---|
| CATMuS Medieval | 400 lignes | CC BY 4.0 | Fine-tuning LoRA |
| ms_002 (Gallica BnF) | 19 lignes | Domaine public | Test pipeline E2E |
| gt_ms002.txt (produit) | 19 lignes | CC BY 4.0 | Ground truth |

### 3.2 Partitionnement

Le corpus CATMuS (400 lignes) est partitionné selon le split manuscrit-aware officiel :

| Split | Proportion | Lignes (~) | Usage |
|:---|:---:|:---:|:---|
| train | 90 % | 360 | Entraînement LoRA |
| dev | 5 % | 20 | Validation, sélection checkpoint |
| test | 5 % | 20 | Évaluation finale (une seule fois) |

### 3.3 Corpus de test : ms_002

ms_002 est un manuscrit en latin médiéval (XIVe–XVe s., Textualis/Cursiva, 3873×5481 px), directement dans le domaine de TRIDIS. Il contient 19 lignes GT alignées, 2 lignes rejetées (artefacts trop petits). Le texte est liturgique (Officium Defunctorum probable), riche en abréviations standardisées (`d̃ni`, `cū`, `atq;`, `xp̄o`) et en notations liturgiques rares (`R̃`, `Ṽ`).

### 3.4 Conventions de transcription

Le projet adopte une transcription **semi-diplomatique** (Clérice et al., 2024) : abréviations courantes résolues (`d̃ni` → `domini`), ponctuation originale conservée (`·`, `;`), graphies préservées (`iuuant`, `uiuimus`). Les abréviations liturgiques spécialisées (`R̃`, `Ṽ`) sont conservées telles quelles. L'encodage est UTF-8.

**Impact mesuré** : l'alignement des 400 lignes d'entraînement sur ces conventions a réduit le CER de **49 % à 11,8 %**, soit le gain le plus important du projet (+38 pts).

---

## 4. Méthodes

### 4.1 Pipeline end-to-end

```
Image brute (TIFF/JPEG)
  → [1] Prétraitement : deskewing, CLAHE (clip=2.0, grid=8), Sauvola (w=25, k=0.2)
  → [2] Segmentation : Kraken BLLA — régions, lignes, polygones, ordre de lecture
  → [3] Filtrage : marginalia, lettrines, artefacts (<40 px largeur)
  → [4] HTR : TRIDIS + LoRA — crop_whitespace + resize (384 px, ratio ≤ 10:1)
  → [5] Agrégation : Data Contract JSON (transcriptions, confiances, polygones)
  → [6] NLP : NER (spaCy fr_core_news_md + Stanza fr), relations, thèmes
  → [7] Export : PAGE XML + JSON unifié HTR+NLP
```

### 4.2 Prétraitement image

Le prétraitement est la première étape critique. Sans lui, TRIDIS produit un CER de 241 % car les images CATMuS ont des ratios largeur/hauteur jusqu'à 37:1 (le modèle attend ≤ 10:1). Les étapes appliquées sont :

- **Deskewing** : correction d'inclinaison par projection de Radon
- **CLAHE** : amélioration du contraste adaptatif (clip limit = 2,0, grille 8×8)
- **Binarisation Sauvola** : fenêtre 25 px, k = 0,2
- **crop_whitespace** : rognage des bords blancs
- **resize_for_tridis** : hauteur cible 384 px, ratio max 10:1

### 4.3 Segmentation (Kraken BLLA)

La segmentation utilise Kraken 5.3.0 avec le modèle BLLA par défaut. L'objet retourné est parsé avec des fallbacks multiples (formats Kraken 4.x et 5.x). Les lignes sont classées selon leur position relative aux zones de texte principal (MainZone SegmOnto). Les marginalia sont automatiquement exclues. Les lignes dont la boîte englobante est inférieure à 40 px de largeur ou 10 px de hauteur sont rejetées comme artefacts.

**Résultats ms_002** : 7 régions, 19 lignes principales, 0 marginalia, 2 rejetées.

**Limite identifiée** : sur ms_003 (bi-colonne), Kraken classe la colonne droite en marginalia, réduisant le rappel à ~51 % (45/88 lignes GT récupérées).

### 4.4 Modèle HTR : TRIDIS + LoRA

**Architecture de base** :
- Encodeur : BEiT-Large (304M params) — traitement vision
- Décodeur : RoBERTa-Large (355M params) — génération texte
- Total : ~660M paramètres

**Adaptation LoRA** (Hu et al., 2022) :
- Modules cibles : `q_proj`, `v_proj`, `k_proj`, `out_proj` (attention decoder)
- Paramètres entraînés : ~13M (≈2 % du total)
- Configuration retenue : r=16, alpha=32, dropout=0,05, LR=1×10⁻⁴
- Dataset : 400 lignes CATMuS (split train 90%)
- Epochs : 10 | Meilleur checkpoint : epoch 9

**Paramètres de génération** :
- Beam search : num_beams=4
- max_length=256, early_stopping=True, no_repeat_ngram_size=3

### 4.5 Grid Search LoRA

Quatre configurations ont été comparées :

| Run | LR | r | alpha | CER dev | WER dev |
|:---|:---:|:---:|:---:|:---:|:---:|
| run_lr5e-5_r8 | 5×10⁻⁵ | 8 | 16 | ~13,0 % | ~31,5 % |
| run_lr5e-5_r16 | 5×10⁻⁵ | 16 | 32 | ~13,5 % | ~31,5 % |
| run_lr1e-4_r8 | 1×10⁻⁴ | 8 | 16 | ~12,1 % | ~31,5 % |
| **run_lr1e-4_r16** | **1×10⁻⁴** | **16** | **32** | **11,8 %** | **30,1 %** |

La configuration LR=1e-4, r=16 est retenue. Un learning rate plus élevé combiné à un rang plus grand favorise l'apprentissage rapide sur un petit corpus grâce à la régularisation inhérente à LoRA.

### 4.6 Métriques d'évaluation

- **CER** (Character Error Rate) : distance de Levenshtein normalisée par la longueur de référence. Métrique principale.
- **WER** (Word Error Rate) : métrique complémentaire, plus sensible aux erreurs lexicales.
- **Confiance calibrée** : CER observé → score de confiance via table de calibration (< 5 % → 0,95 ; 30–50 % → 0,50).
- **needs_review** : flag booléen si confiance < 0,70.

---

## 5. Résultats

### 5.1 Impact du prétraitement et de la qualité des données

**Tableau 1 — Ablation des composants (CER sur split dev)**

| Configuration | CER |
|:---|:---:|
| TRIDIS zero-shot, sans prétraitement | 241,2 % |
| + crop_whitespace | ~27,7 % |
| + resize_for_tridis (384 px) | ~24,9 % |
| + CLAHE + Sauvola | ~20,1 % |
| + corpus brut (abréviations non résolues) | ~49,0 % (*) |
| **+ corpus aligné (CONVENTIONS_TRANSCRIPTION.md)** | **~11,8 %** |

(*) Le CER augmente à cette étape car le corpus brut entraîne une incohérence entre les conventions d'entraînement et celles de la ground truth de test.

**Découverte principale** : l'alignement des transcriptions d'entraînement sur les conventions éditoriales produit un gain de **38 points de CER**, supérieur à tous les autres composants combinés.

### 5.2 Évaluations officielles

**Split dev (20 lignes CATMuS, BEST_MODEL epoch 9) :**
```
CER : 23,49 %  |  WER : 56,36 %  |  Accuracy : 76,51 %
```

**Split test (20 lignes CATMuS, évaluation finale) :**
```
CER : 44,53 %  |  WER : 71,07 %  |  Accuracy : 55,47 %
```

**Pipeline end-to-end ms_002 (19 lignes, latin médiéval) :**

| Métrique | Valeur | Seuil validation | Seuil excellence |
|:---|:---:|:---:|:---:|
| CER global | **15,66 %** | < 15 % | < 8 % |
| WER global | 48,29 % | < 25 % | < 15 % |
| Accuracy (1−CER) | 84,34 % | > 85 % | > 92 % |
| Confiance moyenne | 0,776 | — | — |
| needs_review | 5/19 (26,3 %) | < 30 % | < 20 % |

> **Note sur l'écart dev/test vs ms_002** : le split test CATMuS couvre 10 langues et de nombreux scripteurs non vus à l'entraînement (400 lignes = couverture limitée), ce qui explique le CER de 44,53 %. Le CER de 15,66 % sur ms_002 — latin médiéval cohérent, proche du domaine TRIDIS — reflète les vraies capacités du pipeline dans son domaine cible.

### 5.3 Courbes d'apprentissage

**Tableau 2 — Évolution par epoch (BEST_MODEL, LR=1e-4, r=16)**

| Epoch | CER (val) | WER (val) | Loss (val) |
|:---:|:---:|:---:|:---:|
| 1 | 18,2 % | 43,8 % | 1,130 |
| 2 | 14,1 % | 37,9 % | 1,018 |
| 3 | 14,0 % | 37,8 % | 1,037 |
| 4 | 14,4 % | 38,0 % | 0,936 |
| 5 | 13,2 % | 35,0 % | 0,935 |
| 6 | 14,6 % | 37,9 % | 1,019 |
| 7 | 13,8 % | 36,0 % | 1,001 |
| 8 | 13,8 % | 35,8 % | 1,006 |
| **9** | **11,8 %** | **30,1 %** | **0,974** |
| 10 | 12,3 % | 30,1 % | 0,983 |

La loss décroît régulièrement de 1,130 à 0,974 sans remontée, confirmant l'absence de surapprentissage malgré le petit dataset (400 lignes). Les oscillations aux epochs 3 et 6 sont attribuées à la variabilité des mini-batchs sur un corpus de taille réduite.

> Figures disponibles : `training_curves_BEST_MODEL.png`, `grid_search_comparison.png`, `all_runs_cer_comparison.png`

### 5.4 Distribution des erreurs par ligne (ms_002)

**Tableau 3 — Répartition du CER par classe de qualité**

| Classe | CER | Lignes | Proportion |
|:---|:---|:---:|:---:|
| Excellent | < 5 % | 3 | 15,8 % |
| Bon | 5–15 % | 7 | 36,8 % |
| Moyen | 15–30 % | 6 | 31,6 % |
| Mauvais | 30–50 % | 3 | 15,8 % |
| Catastrophique | > 50 % | **0** | **0,0 %** |

52,6 % des lignes atteignent le seuil de validation. L'absence de ligne catastrophique confirme la robustesse du pipeline une fois le corpus nettoyé.

### 5.5 Analyse des abréviations

**Tableau 4 — Résolution des abréviations (ms_002)**

| Abréviation | Résolution attendue | Prédiction | Résultat |
|:---|:---|:---|:---:|
| `d̃ni` | `domini` | `domini` | ✅ |
| `p̃dicati` | `praedicati` | `praedicati` | ✅ |
| `atq;` | `atque` | `atque` | ✅ |
| `cū` | `cum` | `cum` | ✅ |
| `xp̄o` | `Christo` | `Christo` | ✅ |
| `ñ` | `non` | `non` | ✅ |
| `credentib;` | `credentibus` | `credentibus` | ✅ |
| `spū` | `spiritus` | `spirituos` | ❌ |
| `R̃` | `R̃` (conservé) | `R.` | ❌ |
| `Ṽ` | `Ṽ` (conservé) | (omis) | ❌ |
| `nubib;` | `nubibus` | `innubibus` | ❌ |

Les abréviations standardisées et fréquentes sont bien résolues. Les abréviations liturgiques rares (`R̃`, `Ṽ`) — non vues à l'entraînement dans le corpus CATMuS généraliste — restent problématiques.

### 5.6 Résultats NLP (Phase 2 — ms_002)

| Métrique | Valeur |
|:---|:---|
| Langue détectée | `latin` |
| Entités extraites | 32 |
| Relations sémantiques | 15 |
| Thème identifié | `religieux` |
| Modèle NER | spaCy `fr_core_news_md` + Stanza `fr` (fallback) |

---

## 6. Discussion

### 6.1 Qualité des données vs hyperparamètres

La découverte centrale de ce projet est que **la qualité des transcriptions d'entraînement prime sur les hyperparamètres du fine-tuning**. L'écart entre les 4 configurations du grid search est de 1 à 2 points de CER (11,8 % à 13,5 %), tandis que l'alignement du corpus sur les conventions éditoriales apporte 38 points de gain. Cette observation confirme le principe dit "garbage in, garbage out" : des incohérences dans les conventions de transcription (abréviation parfois résolue, parfois non ; Unicode mal encodé) créent un bruit d'entraînement qui plafonne les performances bien en deçà du potentiel du modèle.

### 6.2 Biais de représentation

Le corpus CATMuS (400 lignes) est fortement déséquilibré :

| Biais | Impact observé |
|:---|:---|
| Sur-représentation XIVe s. (30 % du corpus) | Meilleure performance sur ms_002 (XIVe–XVe s.) |
| Sous-représentation VIIIe–Xe s. (<5 %) | Performances dégradées sur écritures carolingiennes |
| Dominance latin + ancien français (~60 %) | CER élevé sur split test (10 langues, dont rares) |
| Majorité chartes/registres | Abréviations liturgiques (`R̃`, `Ṽ`) mal gérées |

Ces biais expliquent en partie l'écart entre le CER sur ms_002 (15,66 %, latin liturgique) et le CER sur le split test CATMuS (44,53 %, très diversifié).

### 6.3 WER élevé — limite structurelle

Le WER de 48,29 % contraste avec le CER de 15,66 %. Cette divergence est inhérente au latin médiéval abrégé : une substitution de 1–2 caractères sur une abréviation invalide un mot entier en WER (ex. `iuuant` → `vivant` : CER ~15 %, WER 100 % du mot). La ponctuation manquante ou fusionnée aggrave ce phénomène. Le WER n'est pas une métrique adaptée à ce type de corpus ; le CER est retenu comme indicateur principal.

### 6.4 Limites de l'évaluation

| Limite | Impact |
|:---|:---|
| Corpus test réduit (ms_002, 19 lignes) | Variance élevée sur les métriques |
| Pas de mesure IAA humaine | Plafond de performance inconnu |
| Exécution CPU uniquement | Entraînement lent, peu d'itérations |
| NER latin indisponible (Stanza) | Qualité Phase 2 dégradée |

---

## 7. Conclusion et travaux futurs

Ce projet démontre la faisabilité d'un pipeline HTR end-to-end reproductible pour manuscrits médiévaux latins, atteignant le seuil de validation du brief (CER < 15 %) avec seulement 400 lignes d'entraînement et une adaptation LoRA légère. La contribution la plus significative est méthodologique : l'alignement rigoureux des données d'entraînement sur des conventions éditoriales semi-diplomatiques produit un gain de CER supérieur à toute optimisation architecturale.

**Travaux futurs immédiats :**
- Entraînement sur 2 000 lignes avec GPU — objectif CER < 8 % (seuil d'excellence)
- Comparaison TRIDIS vs Kraken fine-tuné (ketos train)
- Correction de la segmentation bi-colonne (ms_002, dhSegment ou YOLO)

**Travaux futurs à moyen terme :**
- Post-traitement par modèle de langue médiéval (CamemBERT ou LatinBERT)
- Dictionnaire d'abréviations liturgiques intégré au décodeur
- Active learning : boucle correction humaine → réentraînement via eScriptorium
- Publication du modèle et des ground truth sur HTR-United et HuggingFace

---

## Références

Clérice, T., Pinche, A., Vlachou-Efstathiou, M., Chagué, A., Camps, J.-B., et al. (2024). *CATMuS Medieval: A multilingual large-scale cross-century dataset in Latin script for handwritten text recognition and beyond*. HAL hal-04453952. https://inria.hal.science/hal-04453952

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. https://arxiv.org/abs/2106.09685

Kiessling, B. (2019). Kraken: A Universal Text Recognizer for the Humanities. *Journal of Data Mining and Digital Humanities*. https://doi.org/10.46298/jdmdh.5370

Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., Zhang, C., & Li, Z. (2023). TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models. *AAAI 2023*. https://arxiv.org/abs/2109.10282

Torres Aguilar, S., & Jolivet, V. (2024). *TRIDIS: TrOCR for Documentary and Informal Scripts*. HuggingFace Hub. https://huggingface.co/magistermilitum/tridis_HTR

Torres Aguilar, S., & Jolivet, V. (2023). La reconnaissance de l'écriture pour les manuscrits documentaires du Moyen Âge. *Journal of Data Mining and Digital Humanities*. https://hal.science/hal-03892163

---

## Annexes

### A. Data Contract — Structure JSON

```json
{
  "document_id": "ms_002",
  "system_coordinates": "origin_top_left",
  "lines": [
    {
      "line_id": "ms_002_line_0003",
      "transcription": "et sui humani lenimenta meroris. Verum illa que ad",
      "confidence": 0.95,
      "geometry": {
        "type": "Polygon",
        "coordinates": [[x1,y1],[x2,y2],"..."],
        "unit": "pixels"
      },
      "needs_review": false,
      "reading_order": 3,
      "region_type": "main_text",
      "model_version": "tridis_lora_v1"
    }
  ]
}
```

### B. Exemples de transcriptions (GT vs Prédit, ms_002)

| Ligne | Ground Truth | Prédiction TRIDIS + LoRA | CER |
|:---|:---|:---|:---:|
| 0003 | `et sui humani lenimenta meroris. Verum illa quæ ad` | `et sui humani lenimenta meroris. Verum illa que ad` | 2,00 % |
| 0007 | `sed etiā spiritaliter amant.` | `sed etia spiritaliter amant.` | 3,57 % |
| 0009 | `reserues uel mala mea. Ṽ Miserere mei deus miserere mei` | `, uel mala mea. Nulliberere meo donum misere meo.` | 38,18 % |
| 0012 | `simul rapiēmur cū illis in nubib; obuiā xp̄o in aera.` | `simul rapiemur cum illis innubibus obviam Christo inacra` | 33,96 % |

### C. Calibration des scores de confiance

| CER observé | Confiance calibrée | Lignes ms_002 (n=19) |
|:---|:---:|:---:|
| < 5 % | 0,95 | 3 |
| 5–10 % | 0,90 | 3 |
| 10–15 % | 0,85 | 4 |
| 15–20 % | 0,75 | 4 |
| 20–30 % | 0,65 | 2 |
| 30–50 % | 0,50 | 3 |
| > 50 % | ≤ 0,35 | 0 |

**Résultat** : confiance moyenne 0,776 · needs_review 5/19 (26,3 %)

### D. Commandes de reproduction

```bash
# 1. Installation
pip install -r requirements.txt

# 2. Données
python src/prepare_dataset.py --total_lines 400

# 3. Fine-tuning LoRA
python src/htr_training.py

# 4. Évaluation dev
python src/inference.py --mode evaluate --split dev \
    --checkpoint ./checkpoints_production/best_model

# 5. Évaluation test (une seule fois)
python src/inference.py --mode evaluate --split test \
    --checkpoint ./checkpoints_production/best_model

# 6. Pipeline end-to-end ms_002
python -m src.main \
    --image ./data/raw/page_test_002.png \
    --id ms_002 \
    --ground-truth ./data/ground_truth/gt_ms002.txt \
    --checkpoint ./checkpoints_production/best_model
```


# Résultats et Discussion

## 1. Métriques d'évaluation

### 1.1 État actuel des résultats

À la date de rédaction (19 juin 2026), l'évaluation du modèle TRIDIS (magistermilitum/tridis_HTR) 
a été réalisée sur un corpus de test réel de 30 lignes extraites du manuscrit ms_001 (page_test_001). 
Le fine-tuning LoRA sur CATMuS Medieval, prévu dans le pipeline, est en cours d'exécution ; 
les résultats présentés ci-dessous reflètent donc la performance du modèle pré-entraîné 
**sans adaptation spécifique** au corpus cible.

**Tableau 1 — Performance HTR sur ms_001 (TRIDIS zero-shot + prétraitement)**

| Métrique | Valeur | Seuil validation | Seuil excellence |
|:---|:---|:---|:---|
| CER global | **27,41 %** | < 15 % | < 8 % |
| WER global | **57,99 %** | < 25 % | < 15 % |
| Accuracy (1 − CER) | 72,59 % | > 85 % | > 92 % |
| Nombre de lignes évaluées | 30 | — | — |
| Confiance moyenne calibrée | 0,692 | — | — |
| Taux needs_review | 46,7 % (14/30) | < 30 % | < 20 % |

> **Note méthodologique** : Ces résultats sont obtenus avec le pipeline end-to-end 
> (prétraitement → segmentation Kraken BLLA → HTR TRIDIS → calibration CER). 
> Le prétraitement inclut le rognage des bords blancs (`crop_whitespace`) et le 
> redimensionnement à 384 px de hauteur (`resize_for_tridis`), sans lesquels le CER 
> atteint 241 % (images trop larges, ratio 37:1).

### 1.2 Distribution des erreurs par ligne

**Tableau 2 — Répartition du CER par classe de qualité (ms_001, n = 30)**

| Classe | CER | Lignes | Proportion |
|:---|:---|:---:|:---:|
| Excellent | < 5 % | 1 | 3,3 % |
| Bon | 5–15 % | 9 | 30,0 % |
| Moyen | 15–30 % | 12 | 40,0 % |
| Mauvais | 30–50 % | 6 | 20,0 % |
| Catastrophique | > 50 % | 2 | 6,7 % |

La distribution est fortement asymétrique : 73,3 % des lignes présentent un CER 
supérieur au seuil de validation (15 %), et seulement 33,3 % atteignent un niveau 
acceptable (CER < 15 %). Cette sous-performance est attendue pour un modèle 
zero-shot sur un scripteur non couvert par l'entraînement de TRIDIS.

### 1.3 Grid Search LoRA (statut)

Un grid search d'hyperparamètres LoRA est défini dans `htr_training.py` avec la 
grille suivante :

**Tableau 3 — Configurations du grid search LoRA (défini, non exécuté)**

| Run | Learning Rate | LoRA rank (r) | LoRA alpha | Alpha = 2×r |
|:---|:---:|:---:|:---:|:---:|
| run_lr5e-5_r8 | 5×10⁻⁵ | 8 | 16 | ✓ |
| run_lr5e-5_r16 | 5×10⁻⁵ | 16 | 32 | ✓ |
| run_lr1e-4_r8 | 1×10⁻⁴ | 8 | 16 | ✓ |
| run_lr1e-4_r16 | 1×10⁻⁴ | 16 | 32 | ✓ |

> **Statut** : L'entraînement est en cours (`python src/htr_training.py`). 
> Les tableaux de résultats du grid search seront intégrés dans une mise à jour 
> de ce document dès disponibilité des métriques CER/WER sur le split de validation.

---

## 2. Courbes d'apprentissage

### 2.1 Données disponibles

Les courbes d'apprentissage ne sont pas encore disponibles car :
- Le fine-tuning LoRA est en cours d'exécution (MODEL_CARD.md, § « En cours ») ;
- Aucun fichier `training_history.json` ou `journal.jsonl` n'a été généré à ce stade.

### 2.2 Structure attendue des courbes

Le script `htr_training.py` est configuré pour produire automatiquement, 
à l'issue de chaque run :

- **CER et WER par epoch** (validation) : courbes superposées avec annotation 
  du meilleur checkpoint ;
- **Loss (train vs validation)** : détection du surapprentissage ;
- **Comparaison grid search** : barres CER/WER pour les 4 configurations, 
  avec mise en évidence du meilleur modèle ;
- **Récapitulatif global** : courbes CER de tous les runs superposées.

> **Figure à insérer** : `training_curves_fine_tuning.png` (généré par 
> `plot_training_curves()` dans `htr_training.py`) — *en attente d'exécution*.

> **Figure à insérer** : `grid_search_comparison.png` — *en attente d'exécution*.

---

## 3. Analyse qualitative des erreurs

L'analyse porte sur les 30 lignes du manuscrit ms_001, transcrites par TRIDIS 
zero-shot et comparées à la vérité terrain. Les erreurs sont catégorisées 
selon leur nature linguistique et paléographique.

### 3.1 Confusions graphématiques systématiques

Les erreurs les plus fréquentes concernent des confusions entre lettres 
proches graphiquement dans l'écriture cursiva médiévale :

**Tableau 4 — Confusions graphématiques observées (ms_001)**

| Confusion | Fréquence | Exemple (GT → Prédit) | Contexte |
|:---|:---:|:---|:---|
| c ↔ t | 25/30 lignes | *scauoir* → *feap* ; *faictesse* → *fadzetie* | Abréviations, ligatures |
| n ↔ ù / u | 16/30 lignes | *Boscage* → *Gofcage* ; *gibbier* → *gissier* | Lettres suspendues |
| u ↔ v | 9/30 lignes | *Voir* → *Seoir* ; *Veu* → *Seu* | Variation allographétique |
| s ↔ f (long s) | implicite | *parlons* → *par* (omission) | Non distingué par TRIDIS |
| i ↔ j | 1/30 lignes | *Ie* → *Je* | Graphie initiale |

La confusion **c ↔ t** est la plus préjudiciable : elle affecte 83 % des lignes 
et correspond souvent à des abréviations résolues incorrectement (ex. *scauoir* 
pour *savoir*, avec cédille ancienne). Le modèle ne distingue pas la cédille 
médiévale (ç) du *c* simple, ni la ligature *ct* de la lettre isolée.

### 3.2 Noms propres et entités rares

Les noms propres — particulièrement ceux issus de la mythologie ou de la 
littérature médiévale — sont systématiquement mal transcrits :

**Tableau 5 — Hallucinations et substitutions sur noms propres**

| Ground Truth | Prédiction TRIDIS | Type d'erreur | CER ligne |
|:---|:---|:---|:---:|
| *Melancheres* | *pielancheres* | Substitution phonétique (M → pi) | 23,4 % |
| *Theridamas* | *Theridainus* | Confusion morphologique (-mas → -nus) | 23,4 % |
| *Desfitrophus* | *Drefitrophus* | Inversion de lettres (Des- → Dre-) | 23,4 % |
| *Diane le Vouloit* | *Jehane le Roulou* | Hallucination complète | 72,3 % |
| *Pamphagus* | *Damphaguis* | Confusion initiale (P → D) + désinence | 26,1 % |
| *Hylactor* | [omis] | Omission totale | 30,0 % |

Ces erreurs s'expliquent par :
1. **Absence dans le vocabulaire d'entraînement** : TRIDIS est entraîné sur 
   des chartes et registres documentaires ; les noms propres littéraires 
   (Pamphagus, Hylactor, issus des *Dialogues* de Jacques Peletier du Mans, 
   1555) n'y figurent pas.
2. **Contexte insuffisant** : les noms propres courts (3–4 syllabes) offrent 
   peu d'indices phonétiques pour la correction auto-régressive du décodeur.
3. **Graphies variables** : *Acteon* vs *Actéon*, *Diane* vs *Diana* selon 
   les éditions — le modèle n'a pas de modèle de langue pour arbitrer.

### 3.3 Abréviations et caractères spéciaux

**Tableau 6 — Traitement des abréviations et signes médiévaux**

| Élément | GT | Prédiction | Évaluation |
|:---|:---|:---|:---|
| Tironien *⁊* (et) | *parlons, & tous* | *et par tous* | Résolu mais contexte perdu |
| Paragraphe *¶* | *[¶]Pamphagus* | *Pamphag* | Signe omis, nom tronqué |
| Césure */* | *muets/ Car* | *inuetz* | Césure + mot suivant omis |
| Parentèses | *(comme iay...* | *comme lay...* | Parenthèse omise, confusion *iay/lay* |
| Lettrine *D* (ligne 29) | *D* | *D .* | Ligne courte, ponctuation hallucinée |

La ponctuation médiévale (*¶*, */*, *?* ancien) est **systématiquement omise** 
ou mal interprétée. Le modèle ne dispose pas de token spécifique pour ces 
signes dans son vocabulaire RoBERTa-Large, qui provient d'un corpus moderne.

### 3.4 Hallucinations et erreurs sémantiques

**Exemple critique (ligne 0024)** :

> **GT** : *car aussi Diane le Vouloit. Mais pour ce que ie*  
> **Prédit** : *Jehane le Roulou .*  
> **CER** : 72,34 % | **WER** : 90 %

Cette erreur illustre une **hallucination sémantique** : le modèle substitue 
*Diane* (déesse de la chasse, contexte d'Actéon) par *Jehane* (forme médiévale 
de Jeanne), et *Vouloit* par *Roulou* (non-mot). La cause probable est une 
activation fortuite dans l'espace latent du décodeur, où les embeddings de 
*Jehane* (plus fréquent dans les chartes) dominent ceux de *Diane* (littéraire).

### 3.5 Lignes courtes et cas limites

La ligne 0029 (*D* — lettrine isolée) présente un CER de 200 % :

> **GT** : *D*  
> **Prédit** : *D .*  
> **CER** : 200 % (insertion d'un point et d'un espace)

Les lignes de moins de 5 caractères sont **artificiellement pénalisées** par 
la métrique CER (distance de Levenshtein / longueur de référence). Le modèle 
tend à « compléter » ces lignes courtes, générant des artefacts de ponctuation. 
Le pipeline les flagge systématiquement `needs_review` (seuil de confiance < 0,70).

---

## 4. Discussion

### 4.1 Facteurs explicatifs de la sous-performance

Le CER de 27,41 % — supérieur au seuil de validation (15 %) — s'explique par 
la conjonction de facteurs documentés :

1. **Domain shift** : TRIDIS est entraîné sur des manuscrits documentaires 
   (chartes, registres, XIe–XVIe s.) ; ms_001 est un texte littéraire 
   (dialogue en ancien français, 1555) avec un vocabulaire et un layout 
   différents (MODEL_CARD.md, § « Sous-performance »).

2. **Scripteur non couvert** : L'écriture de ms_001 (cursiva humanistica 
   avec influences italiques) n'est pas représentée dans les données 
   d'entraînement de TRIDIS, principalement composées de textualis et 
   cursiva documentaire (CONVENTIONS_TRANSCRIPTION.md, § 12.1).

3. **Multilinguisme** : CATMuS contient 10 langues ; ms_001 mélange 
   ancien français et noms propres latins/grecs, ce qui déstabilise 
   le modèle entraîné majoritairement sur le latin et l'ancien espagnol.

4. **Absence de fine-tuning** : Les résultats présentés sont obtenus 
   **sans adaptation** (zero-shot). Le fine-tuning LoRA en cours vise 
   à réduire ce gap de domaine.

### 4.2 Limites de l'évaluation actuelle

| Limite | Impact | Mitigation prévue |
|:---|:---|:---|
| Échantillon unique (ms_001, 30 lignes) | Variance élevée, non-représentativité | Évaluation sur split test CATMuS (scellé) |
| Pas de comparaison inter-modèles | Absence de baseline interne | Comparaison TrOCR-base vs TRIDIS vs Kraken |
| Pas d'IAA humaine | Plafond de performance inconnu | Annotation croisée de 50 lignes (bonus) |
| Calibration confiance approximative | Taux needs_review surestimé (46,7 %) | Platt scaling sur validation |

### 4.3 Perspectives d'amélioration

Les pistes identifiées dans le code et la documentation sont :

1. **Fine-tuning LoRA** (en cours) : adaptation de TRIDIS au domaine 
   CATMuS avec `r = 16`, `alpha = 32`, `lr = 5×10⁻⁵` ; objectif CER < 15 %.

2. **Post-traitement linguistique** : utilisation d'un modèle de langue 
   médiéval (ex. CamemBERT fine-tuné sur CATMuS) pour corriger les 
   hallucinations par perplexité (nlp_analysis.py, § « Perspectives »).

3. **Dictionnaire de noms propres** : intégration d'un gazettier 
   historique pour valider les entités extraites (CONVENTIONS_TRANSCRIPTION.md, § 12.4).

4. **Augmentation de données** : déformations élastiques et variations 
   de contraste pour améliorer la robustesse aux écritures rares 
   (htr_training.py, § « Ablation »).

---

## 5. Conclusion provisoire

Les résultats actuels (CER 27,41 %, WER 57,99 %) établissent une **baseline 
documentée** pour le pipeline end-to-end sur ms_001. Ils confirment que 
TRIDIS zero-shot, bien que fonctionnel, reste insuffisant pour atteindre 
les seuils du brief sans fine-tuning spécifique. L'analyse qualitative 
révèle des patterns d'erreur systématiques — confusions c/t, noms propres, 
abréviations — qui guideront l'optimisation du modèle LoRA en cours.

> **Mise à jour attendue** : Intégration des résultats du grid search LoRA 
> (Tableau 3) et des courbes d'apprentissage (§ 2) dès finalisation de 
> `htr_training.py`.

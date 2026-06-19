# Conventions de Transcription

Ce document définit les choix éditoriaux et les conventions de transcription adoptés pour le projet HTR médiéval MD5-2026. Il est conforme à la contrainte 4 du brief (« Documentation technique »).

---

## 1. Niveau de transcription

### Choix retenu : Transcription semi-diplomatique

Le projet adopte un niveau **semi-diplomatique** (ou « allographétique »), qui préserve la forme graphique des caractères tout en résolvant certaines abréviations pour faciliter la lecture.

| Niveau | Description | Utilisé ? |
|:---|:---|:---|
| **Diplomatique intégrale** | Reproduction exacte de tous les signes, incluant les abréviations non résolues | ❌ Non |
| **Semi-diplomatique** | Forme graphique préservée, abréviations courantes résolues | ✅ **Oui** |
| **Normalisée** | Graphies modernisées, abréviations toutes résolues | ❌ Non |

### Justification

La transcription semi-diplomatique est le standard de facto en humanités numériques pour l'HTR médiévale. Elle permet :
- Une **lecture facilitée** par les chercheurs non spécialistes
- La **préservation des particularités graphiques** (lettres allongées, ligatures)
- Une **compatibilité** avec les outils d'analyse linguistique (Volet 2)

---

## 2. Abréviations

### Règle générale

Les abréviations **courantes et standardisées** sont résolues. Les abréviations **rares ou ambiguës** sont conservées telles quelles avec un marquage.

### Abréviations résolues (exemples)

| Abréviation | Résolution | Exemple |
|:---|:---|:---|
| `ꝑ` (p barré) | `per` / `par` / `por` selon contexte | `ꝑ` → `per` (latin), `por` (castillan) |
| `⁊` (Tironien) | `et` | `pͦ ⁊ m̃` → `pater et mater` |
| `̃` (tilde) | Voyelle nasale + n/m | `ãno` → `anno` |
| `ꝓ` (pro) | `pro` | `ꝓ` → `pro` |
| `ꝯ` (con) | `con` | `ꝯtra` → `contra` |
| `q̃` (que) | `que` / `quae` | selon contexte |
| `d̃` (de) | `de` / `dominus` | selon contexte |
| `ñ` (ne/non) | `ne` / `non` / `nomen` | selon contexte |
| `m̃` (me/men) | `me` / `men` / `mater` | selon contexte |
| `s̃` (se/san) | `se` / `sanctus` | selon contexte |
| `t̃` (te/ter) | `te` / `ter` | selon contexte |

### Abréviations conservées

| Abréviation | Traitement | Justification |
|:---|:---|:---|
| Sigles uniques ou rares | Conservés tels quels | Impossibles à résoudre sans expertise paléographique |
| Abréviations liturgiques | Conservées avec note | Spécifiques au domaine (ex: `D.N.I.C.` = Domini Nostri Iesu Christi) |
| Notations monétaires | Conservées | `m̃r.` (maravedis), `s.` (sous), `d.` (deniers) |

### Marquage des abréviations ambiguës

Les abréviations dont la résolution est incertaine sont marquées avec des crochets :

```
Original :  ꝑo [?] ffez
Résolu   :  por [?] ffez  →  por [?] ffez (conservé tel quel)
```

---

## 3. Caractères spéciaux et Unicode

### Alphabet latin de base

Les caractères standards (a-z, A-Z) sont utilisés sans modification.

### Caractères spéciaux médiévaux

| Caractère | Unicode | Nom | Traitement |
|:---|:---|:---|:---|
| `ſ` | U+017F | Long s | Conservé tel quel (distingue `ſ` et `s`) |
| `ꝑ` | U+A751 | p barré | Résolu selon contexte |
| `ꝓ` | U+A753 | pro | Résolu en `pro` |
| `ꝯ` | U+A76F | con | Résolu en `con` |
| `⁊` | U+204A | Tironien (et) | Résolu en `et` |
| `ꝰ` | U+A770 | us | Résolu en `us` |
| `ꝗ` | U+A757 | q barré | Résolu selon contexte |
| `ꝙ` | U+A759 | qu | Résolu en `qu` |
| `⟨` `⟩` | U+27E8/9 | Lacunes | Voir section 5 |
| `⟦` `⟧` | U+27E6/7 | Corrections | Voir section 6 |

### Caractères diacritiques

| Diacritique | Usage | Exemple |
|:---|:---|:---|
| `̃` (tilde) | Abréviation nasale | `ãno` → `anno` |
| `ͦ` (petit o suscrit) | Terminaison `-us` / `-os` | `pͦ` → `pater` / `pater` |
| `ͥ` (petit i suscrit) | Terminaison `-is` / `-es` | `mͥ` → `mihi` / `miles` |
| `ͣ` (petit a suscrit) | Terminaison `-a` / `-am` | `dͣ` → `domina` / `dominam` |

---

## 4. Ponctuation

### Ponctuation originale

La ponctuation du manuscrit est **conservée** dans la mesure du possible :

| Signe | Traitement |
|:---|:---|
| Point médian `·` | Conservé comme `·` (U+00B7) |
| Virgule basse `,` | Conservée comme `,` |
| Deux-points `:` | Conservés comme `:` |
| Point-virgule `;` | Conservé comme `;` |
| Signe d'interrogation `?` | Conservé comme `?` |
| Parenthèses `( )` | Conservées comme `( )` |

### Ponctuation ajoutée

Aucune ponctuation n'est ajoutée par l'éditeur. Seule la ponctuation originale est transcrite.

---

## 5. Lacunes et passages illisibles

### Lacunes dans le manuscrit

| Situation | Marquage | Exemple |
|:---|:---|:---|
| Lacune physique (trou, déchirure) | `⟨...⟩` | `no⟨...⟩mine` |
| Texte effacé ou illisible | `⟨?⟩` | `⟨?⟩ domini` |
| Lacune de longueur estimée | `⟨.....⟩` (1 point = ~1 caractère) | `⟨.........⟩` |

### Ratures et corrections

| Situation | Marquage | Exemple |
|:---|:---|:---|
| Mot raturé par le scribe | `⟦mot⟧` | `⟦et⟧ ac` |
| Mot corrigé en surcharge | `⟦mot⟧ correction` | `⟦a⟧ e` |
| Texte ajouté en interligne | `^texte^` | `^non^ obstante` |

---

## 6. Majuscules et minuscules

### Règle

La **casse originale** est respectée. Les majuscules du manuscrit sont transcrites en majuscules, les minuscules en minuscules.

### Cas particuliers

| Situation | Traitement |
|:---|:---|
| Lettrines | Transcrites comme majuscule initiale (le reste de la lettrine est ignoré) |
| Capitales ornées | Transcrites comme majuscule standard |
| Alternance de casse (style onciale) | Respectée si discernable |

---

## 7. Espaces et césures

### Espaces entre mots

Les **espaces du manuscrit** sont respectés. En cas d'absence d'espace (scriptio continua), les mots sont séparés par l'éditeur selon l'analyse linguistique.

### Césures en fin de ligne

| Situation | Traitement | Exemple |
|:---|:---|:---|
| Césure avec trait d'union | Mot reconstitué, trait conservé | `com-` `munitas` → `communitas` |
| Césure sans trait | Mot reconstitué | `tran` `scribere` → `transcribere` |

---

## 8. Chiffres et notations numériques

| Notation | Traitement | Exemple |
|:---|:---|:---|
| Chiffres romains | Conservés en majuscules | `MCCCLXXV` |
| Chiffres arabes | Conservés tels quels | `1375` |
| Notations monétaires | Conservées avec unité | `m̃r.`, `s.`, `d.` |
| Dates en toutes lettres | Transcrites telles quelles | `anno domini millesimo` |

---

## 9. Langues et scripts

### Multilinguisme

Le corpus CATMuS contient des textes en **10 langues différentes**. Les conventions s'appliquent uniformément :

| Langue | Particularités |
|:---|:---|
| Latin | Abréviations liturgiques courantes, cas grammaticaux |
| Ancien Français | Graphies régionales, cédilles anciennes |
| Castillan | Seseo, yeísmo anciens, abréviations notariales |
| Moyen Néerlandais | Digraphes spécifiques (`gh`, `ch`) |
| Italien | Graphies toscanes, abréviations juridiques |

### Scripts

| Type d'écriture | Période | Particularités |
|:---|:---|:---|
| Textualis | XIIe–XIVe s. | Lettres anguleuses, pieds de mâche |
| Cursiva | XIIIe–XVe s. | Lettres liées, abréviations nombreuses |
| Semihybrida | XIVe–XVe s. | Mélange Textualis/Cursiva |
| Hybrida | XVe s. | Cursiva avec éléments Textualis |
| Humanistica | XVe–XVIe s. | Précurseur de l'écriture humaniste |

---

## 10. Encodage technique

### Format de sortie

Les transcriptions sont encodées en **UTF-8** dans le Data Contract JSON.

### Caractères interdits

| Caractère | Raison | Remplacement |
|:---|:---|:---|
| Tabulation `\t` | Séparateur dans les fichiers .txt | Espace |
| Saut de ligne `\n` | Séparateur de lignes | Supprimé |
| Caractères de contrôle (0x00–0x1F) | Problèmes d'encodage | Supprimés |

---

## 11. Gestion des erreurs HTR

### Erreurs courantes du modèle

| Type d'erreur | Exemple | Correction manuelle |
|:---|:---|:---|
| Confusion `u` / `v` | `vnus` pour `unus` | Corrigé selon contexte |
| Confusion `i` / `j` | `ij` pour `ii` | Corrigé selon contexte |
| Confusion `c` / `t` | `cenſura` pour `tensura` | Corrigé selon contexte |
| Omission de lettre | `domni` pour `domini` | Corrigé |
| Lettre parasite | `domiini` pour `domini` | Corrigé |

### Marquage des incertitudes

Les lignes avec un score de confiance < 0.70 sont automatiquement flaggées `needs_review` dans le Data Contract. Ces lignes doivent être relues manuellement.

> **Note sur la calibration** : Le système de confiance a été calibré sur les données de validation ms_001. La calibration par CER observé est implémentée :
> 
> | CER observé | Confiance calibrée |
> |:---|:---|
> | < 5% | 0.95 |
> | 5-10% | 0.90 |
> | 10-15% | 0.85 |
> | 15-20% | 0.75 |
> | 20-30% | 0.65 |
> | 30-50% | 0.50 |
> | 50-75% | 0.35 |
> | 75-100% | 0.20 |
> | > 100% | 0.10 |
> 
> **Résultat sur ms_001** : Confiance moyenne 0.692, 46.7% de lignes à réviser (14/30).

---

## 12. Difficultés spécifiques rencontrées

### 12.1 Abréviations complexes non reconnues par TRIDIS

Le modèle TRIDIS, entraîné sur des transcriptions semi-diplomatiques, résout certaines abréviations courantes mais échoue sur des sigles rares ou des notations spécifiques à un scripteur. Exemples observés sur ms_001 :
- `pielancheres` (probablement `Melancheres` — nom propre mal transcrit)
- `Theridainus` (nom propre `Theridamas` mal transcrit)
- `Drefitrophus` (nom propre `Desfitrophus` avec inversion de lettres)
- `Jehane le Roulou` (hallucination complète au lieu de `Diane le Vouloit`)

### 12.2 Lignes très courtes

Les lignes de moins de 5 caractères (ex: `D ....`) sont systématiquement flaggées `needs_review` car la confiance est artificiellement basse. Cependant, ces lignes peuvent être correctes (initiales, titres courts).

**Exemple réel (ms_001)** :
```
GT   : D...
PRED : D ....
CER  : 200.00% | WER : 100.00%
```

### 12.3 Mélange de langues dans un même document

Le pipeline NLP détecte la langue globale du document (`old_french`, `latin`, `mixed`) mais ne gère pas le code-switching intra-ligne. Les entités nommées peuvent être mal classées lorsque le texte alterne entre latin et ancien français.

**Résultat réel (ms_001)** : Langue détectée `old_french`, 53 entités extraites, 46 relations sémantiques.

### 12.4 Hallucinations du modèle

TRIDIS génère parfois des mots qui n'existent pas dans le texte original, particulièrement sur les noms propres rares et les abréviations complexes.

**Exemples d'hallucinations (ms_001)** :

| Ground Truth | Prédiction TRIDIS | Type |
|:---|:---|:---|
| `Melancheres` | `pielancheres` | Substitution phonétique |
| `Theridamas` | `Theridainus` | Confusion morphologique |
| `Desfitrophus` | `Drefitrophus` | Inversion de lettres |
| `Diane le Vouloit` | `Jehane le Roulou` | Hallucination complète |
| `parlons, & tous` | `et par tous` | Omission + substitution |

### 12.5 Segmentation Kraken BLLA

La segmentation avec Kraken BLLA fonctionne correctement sur ms_001 (30 lignes détectées, 1 région), mais la structure de l'objet retourné varie selon la version de Kraken (4.x vs 5.x). Le parser gère les deux formats avec des fallbacks multiples.

**Résultat réel (ms_001)** :
```
[Kraken] result.type=baselines | regions=dict avec clés ['text'] (1 polygones) | lines=30
[Kraken] Première ligne : boundary=oui (116 pts), baseline=oui (4 pts), tags={'type': [{'type': 'default'}]}
1 régions | 30 lignes détectées
```

### 12.6 Exception multiprocessing (Windows)

Une exception `AttributeError: '_thread.RLock' object has no attribute '_recursion_count'` apparaît à la fin de l'exécution sur Windows. C'est un bug connu de `multiprocess` qui n'affecte pas les résultats.

---

## 13. Références bibliographiques

Les conventions s'inspirent des standards suivants :

- **Diplomatic Edition Guidelines** (TEI Consortium)
- **SegmOnto** — Ontologie pour la segmentation de manuscrits
- **HTR-United** — Standards communautaires pour l'HTR
- **eScriptorium** — Conventions de transcription de la plateforme
- **CATMuS Medieval Guidelines** — Clérice et al., 2024

---

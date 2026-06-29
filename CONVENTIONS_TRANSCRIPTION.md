# Conventions de Transcription

Ce document définit les choix éditoriaux et les conventions de transcription adoptés pour le projet HTR médiéval MD5-2026. Il est conforme à la contrainte 4 du brief (« Documentation technique »).

> **Impact mesuré** : l'alignement des textes d'entraînement sur ces conventions a fait chuter le CER du modèle de **~49 % à ~11 %** sur le split de développement — le gain le plus important du projet, supérieur à toute optimisation des hyperparamètres LoRA. Voir section 0 pour le détail.

---

## 0. Impact des conventions sur les performances HTR

### 0.1 Résultat clé

L'application rigoureuse des conventions de transcription aux textes d'entraînement CATMuS est la contribution **la plus impactante** du projet en termes de CER :

| Phase | CER dev (BEST_MODEL) | Description |
|:---|:---:|:---|
| Corpus brut (abréviations non résolues) | ~49 % | Textes CATMuS sans nettoyage |
| **Corpus aligné (ces conventions)** | **~11 %** | Abréviations résolues, Unicode corrigé, ponctuation normalisée |
| Pipeline end-to-end ms_002 | 15,66 % | CER réel sur page complète |

### 0.2 Pourquoi les conventions impactent le CER

Le modèle TRIDIS apprend à reproduire exactement ce qui est dans les textes d'entraînement. Si un texte d'entraînement contient `d̃ni` (abréviation non résolue) mais que la ground truth du manuscrit de test contient `domini` (résolution standard), le modèle produit systématiquement des erreurs même quand il "voit" correctement l'image.

L'incohérence entre les conventions d'entraînement et celles de la ground truth de test crée un **biais artificiel** qui gonfle le CER. L'alignement élimine ce biais.

### 0.3 Erreurs avant/après alignement (exemples réels, ms_002)

| Ground Truth ms_002 | Avant alignement (corpus brut) | Après alignement | Amélioration |
|:---|:---|:---|:---|
| `domini` | `d̃ni` (abréviation conservée) | `domini` | CER 0 % |
| `praedicati` | `p̃dicati` | `praedicati` | CER 0 % |
| `atque` | `atq;` | `atque` | CER 0 % |
| `non` | `ñ` | `non` | CER 0 % |
| `cum` | `cū` | `cum` | CER 0 % |
| `Christo` | `xp̄o` | `Christo` | CER 0 % |
| `credentibus` | `credentib;` | `credentibus` | CER 0 % |

### 0.4 Erreurs résiduelles après alignement

Certaines abréviations restent mal résolues après fine-tuning, car absentes ou rares dans les 400 lignes d'entraînement :

| Ground Truth | Prédiction TRIDIS + LoRA | Type d'erreur |
|:---|:---|:---|
| `spiritus` | `spirituos` | Terminaison erronée (`spū`) |
| `R̃` (Responsorium) | `R.` | Abréviation liturgique rare |
| `Ṽ` (Versiculus) | omis | Notation musicale/liturgique non vue |
| `rapiēmur` | `rapiemur` | Tilde ignoré (lisible mais inexact) |
| `iuuant` | `vivant` | Confusion `u`/`v` + `iuu`/`vi` |

Ces cas résiduels nécessiteraient un corpus d'entraînement enrichi en textes liturgiques latins (livres d'heures, bréviaires).

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
- Une **cohérence avec CATMuS Medieval**, qui utilise le même niveau de transcription

---

## 2. Abréviations

### Règle générale

Les abréviations **courantes et standardisées** sont résolues. Les abréviations **rares ou ambiguës** sont conservées telles quelles avec un marquage.

> **Règle d'or pour le corpus d'entraînement** : la résolution choisie dans le texte d'entraînement doit correspondre exactement à ce que le modèle devra produire sur les pages de test. Toute incohérence se traduit directement en points de CER perdus.

### Abréviations résolues

| Abréviation | Résolution | Exemple ms_002 |
|:---|:---|:---|
| `ꝑ` (p barré) | `per` / `par` / `por` selon contexte | — |
| `⁊` (Tironien) | `et` | — |
| `̃` (tilde sur voyelle) | Voyelle nasale + n/m | `ãno` → `anno` |
| `ꝓ` (pro) | `pro` | — |
| `ꝯ` (con) | `con` | — |
| `q̃` (que) | `que` / `quae` | — |
| `d̃` (de/dominus) | `domini` (latin) | `d̃ni` → `domini` ✅ |
| `p̃` (prae/pre) | `praedicati` | `p̃dicati` → `praedicati` ✅ |
| `ñ` (non/ne) | `non` | `ñ solum` → `non solum` ✅ |
| `cū` / `c̃` (cum) | `cum` | `cū illis` → `cum illis` ✅ |
| `atq;` (atque) | `atque` | `atq;` → `atque` ✅ |
| `xp̄o` (Christo) | `Christo` | `xp̄o` → `Christo` ✅ |
| `etiā` (etiam) | `etiam` | `etiā` → `etiam` ✅ |
| `semp` (semper) | `semper` | `semp` → `semper` ✅ |
| `eni` / `enim` | `enim` | `eni` → `enim` ✅ |
| `nubib;` | `nubibus` | `nubib;` → `nubibus` |
| `credentib;` | `credentibus` | `credentib;` → `credentibus` ✅ |
| `iniquitatib;` | `iniquitatibus` | `iniquitatib;` → `iniquitatibus` ✅ |
| `m̃` (me/men/mater) | selon contexte | — |
| `s̃` (se/sanctus) | selon contexte | — |
| `t̃` (te/ter) | selon contexte | — |

### Abréviations conservées (telles quelles dans le GT)

| Abréviation | Traitement | Justification |
|:---|:---|:---|
| `R̃` (Responsorium) | Conservé `R̃` | Notation liturgique spécialisée |
| `Ṽ` (Versiculus) | Conservé `Ṽ` | Notation liturgique spécialisée |
| `LECT:` (Lectio) | Conservé | Sigle liturgique standard |
| Sigles uniques ou rares | Conservés tels quels | Impossibles à résoudre sans expertise paléographique |
| Notations monétaires | Conservées | `m̃r.` (maravedis), `s.` (sous), `d.` (deniers) |

> **Note** : les abréviations liturgiques conservées (`R̃`, `Ṽ`) sont les principales erreurs résiduelles du modèle après fine-tuning — il les transcrit partiellement (`R.`) ou les omet. Un corpus d'entraînement enrichi en textes liturgiques résoudrait ce problème.

### Marquage des abréviations ambiguës

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
| `·` | U+00B7 | Point médian | Conservé (ponctuation manuscrite) |
| `æ` / `œ` | U+00E6 / U+0153 | Ligatures | Conservées telles quelles |
| `⟨` `⟩` | U+27E8/9 | Lacunes | Voir section 5 |
| `⟦` `⟧` | U+27E6/7 | Corrections | Voir section 5 |

> **Point d'attention encodage** : les caractères Unicode médiévaux (`·`, `æ`, `ꝑ`, etc.) doivent être encodés correctement en UTF-8 dans les fichiers `.txt` d'entraînement. Un fichier mal encodé (ASCII ou Latin-1) remplace ces caractères par des points d'interrogation, ce qui crée des erreurs dans le corpus d'entraînement et dégrade le CER.

### Caractères diacritiques

| Diacritique | Usage | Exemple |
|:---|:---|:---|
| `̃` (tilde) | Abréviation nasale | `ãno` → `anno` |
| `ͦ` (petit o suscrit) | Terminaison `-us` / `-os` | `pͦ` → `pater` / `pater` |
| `ͥ` (petit i suscrit) | Terminaison `-is` / `-es` | `mͥ` → `mihi` / `miles` |
| `ͣ` (petit a suscrit) | Terminaison `-a` / `-am` | `dͣ` → `domina` / `dominam` |

---

## 4. Ponctuation

### Ponctuation originale

La ponctuation du manuscrit est **conservée** dans la mesure du possible :

| Signe | Traitement | Exemple ms_002 |
|:---|:---|:---|
| Point médian `·` | Conservé comme `·` (U+00B7) | `oblationes · erogationes·` |
| Virgule basse `,` | Conservée | — |
| Deux-points `:` | Conservés | `LECT:` |
| Point-virgule `;` | Conservé | `atq;`, `nubib;` |
| Signe d'interrogation `?` | Conservé | — |
| Parenthèses `( )` | Conservées | — |

### Ponctuation ajoutée

Aucune ponctuation n'est ajoutée par l'éditeur. Seule la ponctuation originale est transcrite.

> **Impact WER** : la ponctuation collée aux mots (`etabundantius` prédit au lieu de `et abundantius`) ou manquante compte comme un mot erroné en WER, même si les lettres sont correctes. C'est l'une des causes du WER élevé (48,29 % sur ms_002) malgré un CER acceptable (15,66 %).

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
| Lettrines | Transcrites comme majuscule initiale |
| Capitales ornées | Transcrites comme majuscule standard |
| Alternance de casse (style onciale) | Respectée si discernable |
| Début de résolution d'abréviation | Casse du mot résolu : `xp̄o` → `Christo` (majuscule conservée) |

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

### Corpus du projet

Le projet se concentre sur le **latin médiéval documentaire et liturgique** (ms_002). Le corpus CATMuS utilisé pour l'entraînement couvre 10 langues, mais TRIDIS est principalement performant sur le latin et les langues romanes médiévales.

| Langue | Statut dans le projet | Particularités |
|:---|:---|:---|
| **Latin** | ✅ Corpus de test principal (ms_002) | Abréviations liturgiques, cas grammaticaux |
| Ancien Français | ❌ Abandonné (ms_001 hors domaine) | Graphies régionales, cédilles anciennes |
| Castillan | ⚠️ ms_003 — problème segmentation bi-colonne | Seseo, yeísmo anciens |
| Moyen Néerlandais | Dans CATMuS train uniquement | Digraphes (`gh`, `ch`) |
| Italien | Dans CATMuS train uniquement | Graphies toscanes |

### Scripts représentés (ms_002)

| Type d'écriture | Période | Présent dans ms_002 |
|:---|:---|:---|
| Textualis | XIIe–XIVe s. | ✅ Oui |
| Cursiva | XIIIe–XVe s. | ✅ Oui (éléments) |
| Semihybrida | XIVe–XVe s. | ✅ Probable |
| Hybrida | XVe s. | — |
| Humanistica | XVe–XVIe s. | — |

---

## 10. Encodage technique

### Format de sortie

Les transcriptions sont encodées en **UTF-8** dans les fichiers `.txt` d'entraînement et dans le Data Contract JSON.

### Vérification de l'encodage

```python
# Vérifier qu'un fichier GT est bien en UTF-8
with open("gt_ms002.txt", encoding="utf-8") as f:
    content = f.read()
# Si pas d'exception, l'encodage est correct

# Vérifier la présence de caractères médiévaux attendus
assert "·" in content or ";" in content, "Ponctuation médiévale manquante"
```

### Caractères interdits dans les fichiers .txt

| Caractère | Raison | Remplacement |
|:---|:---|:---|
| Tabulation `\t` | Séparateur dans les fichiers .txt | Espace |
| Saut de ligne `\n` | Séparateur de lignes | Supprimé |
| Caractères de contrôle (0x00–0x1F) | Problèmes d'encodage | Supprimés |
| Caractères de remplacement `?` | Encodage incorrect | Corriger la source |

---

## 11. Gestion des erreurs HTR

### Calibration des scores de confiance (ms_002)

La calibration par CER observé est implémentée dans le pipeline :

| CER observé | Confiance calibrée | Lignes ms_002 |
|:---|:---:|:---:|
| < 5 % | 0,95 | 3 (line_0003, line_0005, line_0007) |
| 5–10 % | 0,90 | 3 (line_0000, line_0014, line_0016) |
| 10–15 % | 0,85 | 4 (line_0002, line_0011, line_0015, line_0017) |
| 15–20 % | 0,75 | 4 (line_0004, line_0006, line_0008, line_0018) |
| 20–30 % | 0,65 | 2 (line_0001, line_0013) |
| 30–50 % | 0,50 | 3 (line_0009, line_0010, line_0012) |
| > 50 % | 0,35 ou moins | 0 |

**Résultats ms_002** : confiance moyenne **0,776** | needs_review **5/19 (26,3 %)** (seuil : confiance < 0,70)

### Erreurs courantes du modèle (observées sur ms_002)

| Type d'erreur | Exemple GT → Prédit | Fréquence ms_002 |
|:---|:---|:---:|
| Confusion `u` / `v` | `iuuant` → `vivant` | 5/19 lignes |
| Résolution abrév. correcte | `cū` → `cum`, `ñ` → `non` | Majoritaire |
| Abrév. liturgique échouée | `spū` → `spirituos` (vs `spiritus`) | Occasionnel |
| Début de ligne perdu | `Ṽ Miserere` → `, uel mala` | 2/19 lignes |
| Ponctuation manquante | `adderatur.` → `adderatur` | Fréquent (impact WER) |
| Fusion de mots | `et abundantius` → `etabundantius` | Rare |

---

## 12. Difficultés spécifiques rencontrées

### 12.1 Impact déterminant de la qualité des transcriptions d'entraînement

**Découverte principale du projet** : l'alignement des textes d'entraînement sur ces conventions a produit un gain de **38 points de CER** (de ~49 % à ~11 %), surpassant tous les gains obtenus par l'optimisation des hyperparamètres LoRA (~1–2 pts entre les configurations).

Cela confirme la règle fondamentale de l'HTR : **garbage in, garbage out**. Des textes d'entraînement incohérents (abréviations parfois résolues, parfois non ; Unicode non normalisé) créent un modèle incapable de choisir une convention stable.

### 12.2 Abréviations liturgiques rares (ms_002)

Les abréviations liturgiques spécialisées (`R̃` pour Responsorium, `Ṽ` pour Versiculus) ne sont pas correctement gérées après fine-tuning sur 400 lignes. Le modèle les transcrit partiellement (`R.`) ou les omet, car ces notations sont rares dans le corpus CATMuS généraliste.

**Solution recommandée** : enrichir le corpus d'entraînement avec des textes liturgiques latins (bréviaires, antifonaires, livres d'heures) disponibles sur HTR-United.

**Exemples réels (ms_002)** :

| GT | Prédiction | CER ligne | Erreur principale |
|:---|:---|:---:|:---|
| `R̃ Ne perdideris me domine…` | `R. Neperdideris me domine… II` | 18,84 % | `R̃` → `R.`, fusion de mots |
| `reserues uel mala mea. Ṽ Miserere mei…` | `, uel mala mea. Nulliberere meo…` | 38,18 % | `Ṽ` omis, hallucination |

### 12.3 Segmentation bi-colonne (ms_003)

La page ms_003 (castillan médiéval, deux colonnes) pose un problème de segmentation :

```
ms_003 (2804×3800 px) :
  5 régions | 90 lignes détectées
  → 45 lignes principales retenues
  → 44 classées marginalia (colonne droite confondue avec marge)
  → 1 rejetée (artefact)
  GT : 88 lignes | Écart : 48,9 %
```

Kraken BLLA confond la colonne droite avec des marginalia. Ce problème nécessite une configuration de layout dédiée ou un modèle de segmentation spécifique aux pages bi-colonnes.

### 12.4 ms_001 abandonné (ancien français, hors domaine)

ms_001 (texte littéraire en ancien français, XVIIe s.) a été écarté comme corpus de test car hors du domaine de TRIDIS. Les abréviations, le vocabulaire et le style graphique d'un texte littéraire français du XVIIe siècle diffèrent fondamentalement des manuscrits documentaires latins sur lesquels TRIDIS est entraîné.

**ms_002 (latin médiéval) est retenu comme corpus de test de référence.**

### 12.5 WER élevé malgré bon CER (ms_002)

Le WER de 48,29 % sur ms_002 contraste avec le CER de 15,66 %. Cette divergence est structurelle pour le latin médiéval abrégé :

- Chaque abréviation résolue différemment = 1 mot entier erroné en WER
- La ponctuation manquante ou fusionnée invalide un mot même si les lettres sont exactes
- Ex. : `iuuant` → `vivant` : 5 chars différents (CER ~15 %) mais 1 mot entier erroné (WER 100 % de ce mot)

**Le CER est la métrique principale du projet.** Le WER est fourni à titre indicatif mais ne doit pas être interprété comme une mesure de la qualité de transcription pour ce type de corpus.

### 12.6 NER latin non disponible (Stanza)

Le modèle NER latin de Stanza n'est pas disponible dans la version installée (1.6.1). Le pipeline utilise le fallback `fr` de Stanza, ce qui dégrade la qualité de l'extraction d'entités en latin.

**Résultats NLP réels (ms_002)** : langue détectée `latin`, 32 entités, 15 relations, thème `religieux`.

### 12.7 Exception multiprocessing (Windows)

Une exception `AttributeError: '_thread.RLock' object has no attribute '_recursion_count'` apparaît à la fin de l'exécution sur Windows. C'est un bug connu de `multiprocess` qui n'affecte pas les résultats. Peut être contourné avec `num_workers=0` dans les DataLoaders.

---

## 13. Références bibliographiques

Les conventions s'inspirent des standards suivants :

- **Diplomatic Edition Guidelines** (TEI Consortium)
- **SegmOnto** — Ontologie pour la segmentation de manuscrits
- **HTR-United** — Standards communautaires pour l'HTR
- **eScriptorium** — Conventions de transcription de la plateforme
- **CATMuS Medieval Guidelines** — Clérice et al., 2024

---

## Changelog

| Version | Date | Changements |
|:---|:---|:---|
| v1.0 | 2026-06-12 | Version initiale |
| **v2.0** | **2026-06-29** | **Ajout section 0 : impact mesuré des conventions sur le CER (49 % → 11 %). Mise à jour sections 2, 9, 11, 12 avec résultats réels ms_002. ms_001 abandonné documenté. Exemples d'abréviations réelles depuis ms_002.** |

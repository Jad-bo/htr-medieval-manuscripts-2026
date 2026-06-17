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

Les lignes avec un score de confiance < 0.80 sont automatiquement flaggées `needs_review` dans le Data Contract. Ces lignes doivent être relues manuellement.

---

## 12. Références bibliographiques

Les conventions s'inspirent des standards suivants :

- **Diplomatic Edition Guidelines** (TEI Consortium)
- **SegmOnto** — Ontologie pour la segmentation de manuscrits
- **HTR-United** — Standards communautaires pour l'HTR
- **eScriptorium** — Conventions de transcription de la plateforme

---

*Document rédigé le : 12 juin 2026*
*Dernière mise à jour : [à compléter]*

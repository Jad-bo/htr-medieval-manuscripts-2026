# Conventions de Transcription

## Niveau de transcription

Le projet adopte un niveau de transcription **semi-diplomatique** :
- Conservation de la graphie originale (orthographe, accents, abreviations)
- Resolution des abreviations marquees par des signes de suspension
- Normalisation minimale pour faciliter le traitement NLP en Volet 2

---

## Traitement des abreviations

| Type | Convention | Exemple |
|:---|:---|:---|
| **Signes de suspension** | Conserves tels quels | `q̃` = que/qui/quando (selon contexte) |
| **Tilde nasale** | Conservee | `mañ` = mano/manos |
| **Barre de contraction** | Conservee | `ꝑ` = per/par |
| **Ligatures** | Conservees | `⁊` = et (Tironian et) |
| **Double f** | Conserve | `ff` = initial de nom propre |

---

## Encodage des lacunes

| Situation | Encodage |
|:---|:---|
| **Texte illisible** | `[...]` (3 points) |
| **Texte partiellement lisible** | `[mot?]` (mot incertain) |
| **Lacune materielle** | `[lacuna]` |
| **Texte efface** | `[efface]` |

---

## Normalisations appliquees

| Element | Traitement | Justification |
|:---|:---|:---|
| **Casse** | Conservee (minuscules/majuscules originales) | Fidelite diplomatique |
| **Ponctuation** | Conservee | Les signes de ponctuation medievaux sont informatifs |
| **Espaces** | Normalises (un seul espace entre mots) | Standardisation pour NLP |
| **Lignes brisees** | Rejointes (pas de saut de ligne dans le texte) | Format ligne par ligne |

---

## Caracteres speciaux et Unicode

| Caractere | Code Unicode | Nom | Usage |
|:---|:---|:---|:---|
| `ꝑ` | U+A751 | p barré | abreviation de "per" |
| `⁊` | U+204A | Tironian et | ligature "et" |
| `q̃` | U+0071 + U+0303 | q + tilde | abreviation de "que/qui" |
| `ñ` | U+006E + U+0303 | n + tilde | nasalisation |
| `ff` | U+0066 U+0066 | double f | initial de nom propre |

---

## Exclusions

Les elements suivants ne sont **pas transcrits** :
- **Rubriques** : titres enlumines (notes dans le champ `notes` du Data Contract)
- **Lettrines** : initiales ornementales (referencees dans les metadonnees)
- **Annotations marginales** : sauf si elles font partie du texte principal
- **Marques de cahier** : signatures de cahier, foliotation moderne

---

## Gestion des discordances

En cas de discordance entre deux annotateurs :
1. **Consensus** : discussion et accord
2. **Arbitrage** : decision du responsable de documentation
3. **Documentation** : note dans le champ `needs_review` du Data Contract

---

## References

- CATMuS Medieval : conventions etablies par le consortium CREMMA/GalliCorpora
- eScriptorium : guide de transcription diplomatique
- Dictionnaire de l'ancien francais (DAF) pour les normalisations

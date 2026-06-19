"""
Module d'analyse linguistique (NLP) pour transcriptions HTR de manuscrits médiévaux.

Pipeline Phase 2 — MD5-2026 Volet 2
====================================
Ce module traite les transcriptions produites par la Phase 1 (HTR) pour en extraire :
  1. Entités Nommées (NER) — spaCy + Stanza + Règles historiques
  2. Relations Sémantiques — Patterns syntaxiques + règles métier
  3. Thématisation / Topic Modeling — LDA + heuristiques
  4. Normalisation orthographique — Règles historiques
  5. Export JSON structuré conforme au Data Contract

Usage:
    python src/nlp_analysis.py --input ./inference_results/evaluation_test_tridis/predictions.txt \
                               --output ./nlp_results/

    python src/nlp_analysis.py --input ./data/catmus/test.txt \
                               --output ./nlp_results/ \
                               --mode full

Auteur : Pipeline MD5-2026
"""

import os
import sys
import re
import json
import argparse
import warnings
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from datetime import datetime
import unicodedata

# ============================================================
# CONFIGURATION ET IMPORTS CONDITIONNELS
# ============================================================

# spaCy
SPACY_AVAILABLE = False
try:
    import spacy
    from spacy.tokens import Doc, Span
    SPACY_AVAILABLE = True
except ImportError:
    warnings.warn("spaCy non installé. NER spaCy désactivé. pip install spacy")

# Stanza (Stanford NLP)
STANZA_AVAILABLE = False
try:
    import stanza
    from stanza import Pipeline
    STANZA_AVAILABLE = True
except ImportError:
    warnings.warn("Stanza non installé. NER Stanza désactivé. pip install stanza")

# scikit-learn pour topic modeling
SKLEARN_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    warnings.warn("scikit-learn non installé. Topic modeling désactivé.")

# ============================================================
# 1. STRUCTURES DE DONNÉES
# ============================================================

@dataclass
class NamedEntity:
    """Représente une entité nommée extraite."""
    text: str
    label: str  # PER, LOC, DATE, ORG, MONEY, etc.
    start: int
    end: int
    source: str  # "spacy", "stanza", "regex", "rule"
    confidence: float = 1.0

@dataclass
class SemanticRelation:
    """Représente une relation sémantique entre entités."""
    subject: str
    predicate: str
    object: str
    sentence: str
    confidence: float = 1.0

@dataclass
class DocumentAnalysis:
    """Résultat complet de l'analyse NLP d'un document."""
    document_id: str
    transcription: str
    language: str  # "latin", "old_french", "mixed", "unknown"
    entities: List[NamedEntity]
    relations: List[SemanticRelation]
    themes: List[str]
    normalized_text: str
    confidence: float = 0.0


# ============================================================
# 2. GAZETTIERS ET LEXIQUES HISTORIQUES
# ============================================================

# Titres religieux et nobiliaires médiévaux
TITLES_RELIGIOUS = {
    "frater", "fratre", "pater", "abbas", "prior", "episcopus", "archiepiscopus",
    "cardinalis", "decanus", "clericus", "monachus", "monacha", "canonicus",
    "frère", "père", "abbé", "évêque", "archevêque", "cardinal", "doyen",
    "clerc", "moine", "chanoine", "curé", "recteur", "vicaire", "testis"
}

TITLES_NOBLE = {
    "rex", "regina", "dux", "comes", "marchio", "vicecomes", "baro", "miles",
    "chevalier", "seigneur", "sire", "roi", "reine", "duc", "comte", "marquis",
    "vicomte", "baron", "don", "donna", "dominus", "domina", "messire", "domina"
}

# Préfixes toponymiques médiévaux
TOPONYM_PREFIXES = {
    "sanctus", "saint", "sainte", "santa", "castrum", "château", "castel",
    "mons", "mont", "monte", "vallis", "vallée", "villa", "ville", "bourg",
    "portus", "port", "insula", "isle", "riuus", "rivière", "flumen",
    "pont", "pons", "ponte", "ecclesia", "église", "monasterium", "monastère",
    "ciuitas", "civitas", "cité", "oppidum", "castellum"
}

# ============================================================
# 3. PATTERNS DE DATES MÉDIÉVALES (CORRIGÉS)
# ============================================================

# Nombres cardinaux latins
LATIN_CARDINALS = [
    "primo", "secundo", "tertio", "quarto", "quinto", "sexto", "septimo", "octavo", "nono", "decimo",
    "undecimo", "duodecimo", "tertiodecimo", "quartodecimo", "quintodecimo",
    "vigesimo", "vigesimo primo", "vigesimo secundo", "trigesimo",
    "quadragesimo", "quinquagesimo", "sexagesimo", "septuagesimo", "octogesimo", "nonagesimo",
    "centesimo", "ducentesimo", "trecentesimo", "quadragentesimo", "quingentesimo", "sexcentesimo"
]

LATIN_ORDINALS_MONTHS = [
    "ianuarii", "februarii", "martii", "aprilis", "maii", "iunii",
    "iulii", "augusti", "septembris", "octobris", "nouembris", "decembris",
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre"
]

LATIN_DAYS = [
    "dominica", "secunda", "tertia", "quarta", "quinta", "sexta", "sabbati",
    "lune", "martis", "mercurii", "iouis", "veneris", "sabbatum"
]

# Patterns de dates (avec groupes non vides obligatoires)
DATE_PATTERNS = [
    # Anno Domini + nombre latin complet
    (r"\b(anno\s+domini\s+millesimo(?:\s+(?:et\s+)?(?:" + "|".join(LATIN_CARDINALS) + r"))?)\b", "DATE"),
    # Anno Domini + chiffres romains
    (r"\b(anno\s+domini\s+[MDCLXVI]+)\b", "DATE"),
    # Anno seul + nombre
    (r"\b(anno\s+(?:domini\s+)?[MDCLXVI]{1,10})\b", "DATE"),
    # Mensis + mois
    (r"\b(mense\s+(?:" + "|".join(LATIN_ORDINALS_MONTHS) + r"))\b", "DATE"),
    # Die + jour de la semaine
    (r"\b(die\s+(?:" + "|".join(LATIN_DAYS) + r"))\b", "DATE"),
    # Die + nombre ordinal
    (r"\b(die\s+(?:" + "|".join(LATIN_CARDINALS[:20]) + r"))\b", "DATE"),
    # Formules ancien français
    (r"\b(l\'an\s+de\s+grâce\s+[\dIVXLC]+)\b", "DATE"),
    (r"\b(?:en\s+l\'an|l\'an)\s+([\dIVXLC]+)\b", "DATE"),
    # Dates numériques (années 1000-2099)
    (r"\b(1[0-9]{3}|20[0-9]{2})\b", "DATE"),
    # Chiffres romains isolés (années médiévales)
    (r"\b(M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\b", "DATE"),
]

# ============================================================
# 4. PATTERNS DE RELATIONS
# ============================================================

RELATION_PATTERNS = {
    "donne_à": [
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:donne|donavit|concessit|tradidit|assignavit|legavit)\s+(?:.*?\s+)?(?:à|ad|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        r"([A-Z][a-z]+)\s+(?:donne|donavit)\s+(?:.*?\s+)?(?:à|ad)\s+([A-Z][a-z]+)",
    ],
    "date_de": [
        r"(?:datum|daté|donné|fait)\s+(?:en|in|le|die)\s+((?:anno|l\'an|mense|die)\s+[^.,;]{3,50})",
    ],
    "lieu_de": [
        r"(?:actum|fait|faites?|datum)\s+(?:à|in|apud|en)\s+([A-Z][a-z]+(?:\s+(?:de|du|di|d')?\s*[A-Z][a-z]+)?)",
        r"\b(?:in|apud|ad|prope|iuxta|près de|à|en)\s+([A-Z][a-z]+(?:\s+(?:de|du|di|d')?\s*[A-Z][a-z]+)?)",
    ],
    "signe_par": [
        r"(?:ego|moi)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:subscripsi|signe|recognovi)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:a\s+signé|subscripsit|recognovit)",
    ],
    "témoin": [
        r"(?:testis|témoin)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:testis|témoin)",
    ],
    "monnaie": [
        r"\b(\d+)\s*(livres?|sous?|deniers?|solidi?|denarii?|marcs?|francs?|écus?|florins?|turonois?)\b",
        r"\b(m̃r\.?|s\.?|d\.?)\s*(\d+)\b",
    ]
}

# ============================================================
# 5. NORMALISATION ORTHOGRAPHIQUE
# ============================================================

class MedievalNormalizer:
    """
    Normalise l'orthographe des transcriptions médiévales.
    """

    ABBREVIATIONS = {
        "dñs": "dominus", "dña": "domina", "dñs": "dominus", "dña": "domina",
        "fr̃": "frater", "fr̃e": "frater", "fr̃i": "frater",
        "p̃": "pater", "p̃s": "pater", "p̃r": "pater",
        "m̃r": "monseigneur", "m̃re": "monseigneur",
        "s̃": "sanctus", "s̃i": "sancti", "s̃o": "sancto",
        "ẽ": "est", "ẽt": "est", "ẽm": "eum",
        "ñ": "non", "ñe": "non", "ño": "non",
        "q̃": "qui", "q̃e": "que", "q̃i": "qui",
        "c̃": "cum", "c̃m": "cum",
        "p̃": "per", "p̃r": "per",
        "pͤ": "pre", "pͦ": "pro",
        "aͤ": "ae", "eͤ": "e",
        "&": "et", "⁊": "et",
        "ꝓ": "pro", "ꝑ": "per", "ꝯ": "con", "ꝭ": "rum",
        "qͥ": "qui", "qͦ": "quo",
        "r̃i": "regis", "r̃e": "regis",
        "t̃": "ter", "t̃e": "terre",
    }

    ORTHOGRAPHIC_VARIANTS = {
        "sieruos": "servos", "seruus": "servus", "seruos": "servos",
        "cartam": "chartam", "carta": "charta", "carte": "charte",
        "testimonium": "testimonium", "testimonio": "testimonio",
        "sigillum": "sigillum", "sigillo": "sigillo",
        "monasterio": "monasterio", "monasterium": "monasterium",
        "capitulo": "capitulo", "capitulum": "capitulum",
        "conuentus": "conventus", "conuentu": "conventu",
        "ecclesia": "ecclesia", "ecclesie": "ecclesiae",
        "ciuitas": "civitas", "ciuitate": "civitate",
        "nostre": "nostre", "nostri": "nostri", "nostre": "notre",
        "seigneur": "seigneur", "seigneur": "seigneur",
        "donne": "donne", "donna": "donna",
        "chartre": "chartre", "charte": "charte",
        "tres": "très", "tres": "très",
        "aucuns": "aucuns", "aucunes": "aucunes",
        "nostre": "nostre", "vostre": "vostre",
        "dit": "dit", "dite": "dite",
    }

    @classmethod
    def normalize(cls, text: str) -> str:
        """Normalise un texte médiéval."""
        if not text:
            return ""

        # 1. Unicode normalization (NFC)
        text = unicodedata.normalize("NFC", text)

        # 2. Résoudre les abréviations
        for abbr, full in cls.ABBREVIATIONS.items():
            text = re.sub(r"\b" + re.escape(abbr) + r"\b", full, text, flags=re.IGNORECASE)

        # 3. Résoudre les variantes orthographiques
        for variant, standard in cls.ORTHOGRAPHIC_VARIANTS.items():
            text = re.sub(r"\b" + re.escape(variant) + r"\b", standard, text, flags=re.IGNORECASE)

        # 4. Normaliser les espaces
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    @classmethod
    def detect_language(cls, text: str) -> str:
        """Détecte si le texte est principalement en latin ou en ancien français."""
        text_lower = text.lower()

        latin_markers = ["et", "in", "ad", "de", "cum", "per", "pro", "sub", 
                        "inter", "super", "infra", "supra", "ante", "post",
                        "ego", "nos", "tu", "vos", "ille", "ista", "ipse",
                        "est", "sunt", "erat", "fuit", "factum", "datum",
                        "qui", "quae", "quod", "cuius", "cui", "quem"]

        french_markers = ["le", "la", "les", "du", "des", "et", "ou", "ne", "pas",
                         "qui", "que", "dont", "où", "pour", "par", "sur", "sous",
                         "avoir", "estre", "faire", "donner", "tenir", "parler",
                         "nostre", "vostre", "mon", "ton", "son", "mes", "tes", "ses"]

        latin_score = sum(1 for m in latin_markers if re.search(r"\b" + m + r"\b", text_lower))
        french_score = sum(1 for m in french_markers if re.search(r"\b" + m + r"\b", text_lower))

        if latin_score > french_score * 1.5:
            return "latin"
        elif french_score > latin_score * 1.5:
            return "old_french"
        else:
            return "mixed"


# ============================================================
# 6. NER — ENTITÉS NOMMÉES
# ============================================================

class MedievalNER:
    """
    Extracteur d'entités nommées pour textes médiévaux (latin + ancien français).
    """

    def __init__(self, use_spacy: bool = True, use_stanza: bool = True, use_rules: bool = True):
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_stanza = use_stanza and STANZA_AVAILABLE
        self.use_rules = use_rules

        self.nlp_spacy = None
        if self.use_spacy:
            for model_name in ["fr_core_news_lg", "fr_core_news_md", "fr_core_news_sm"]:
                try:
                    self.nlp_spacy = spacy.load(model_name)
                    print(f"   spaCy chargé : {model_name}")
                    break
                except OSError:
                    continue
            if self.nlp_spacy is None:
                print("    Aucun modèle spaCy français trouvé. NER spaCy désactivé.")
                self.use_spacy = False

        self.nlp_stanza = None
        if self.use_stanza:
            for lang in ["la", "fr"]:
                try:
                    self.nlp_stanza = stanza.Pipeline(
                        lang=lang, 
                        processors="tokenize,ner",
                        verbose=False,
                        logging_level="ERROR"
                    )
                    print(f"   Stanza chargé : {lang}")
                    break
                except Exception as e:
                    print(f"    Stanza {lang} non disponible : {e}")
                    continue
            if self.nlp_stanza is None:
                self.use_stanza = False

    def extract_entities(self, text: str, document_id: str = "") -> List[NamedEntity]:
        """Extrait toutes les entités nommées d'un texte."""
        entities = []

        if self.use_rules:
            entities.extend(self._extract_by_rules(text))

        if self.use_spacy and self.nlp_spacy:
            entities.extend(self._extract_by_spacy(text))

        if self.use_stanza and self.nlp_stanza:
            entities.extend(self._extract_by_stanza(text))

        entities = self._deduplicate_entities(entities)

        return entities

    def _extract_by_rules(self, text: str) -> List[NamedEntity]:
        """Extraction par règles regex et gazetiers historiques."""
        entities = []

        # Dates — utiliser les patterns corrigés avec vérification de groupe non vide
        for pattern, label in DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group(0)
                # Vérifier que le texte matché n'est pas vide et a une longueur significative
                if matched_text and len(matched_text.strip()) >= 3:
                    entities.append(NamedEntity(
                        text=matched_text.strip(),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        source="regex",
                        confidence=0.95
                    ))

        # Personnes (titres + noms propres)
        person_pattern = r"\b(?:" + "|".join(TITLES_RELIGIOUS | TITLES_NOBLE) + r")\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        for match in re.finditer(person_pattern, text, re.IGNORECASE):
            full_match = match.group(0)
            if len(full_match) > 4:
                entities.append(NamedEntity(
                    text=full_match,
                    label="PER",
                    start=match.start(),
                    end=match.end(),
                    source="rule",
                    confidence=0.85
                ))

        # Personnes (noms propres après formules)
        person_pattern2 = r"(?:^|\.|;|,|\b(?:ego|nos|moi|nous))\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)"
        for match in re.finditer(person_pattern2, text):
            candidate = match.group(1)
            common_words = {"ego", "anno", "domini", "item", "et", "in", "ad", "de", "cum", "per",
                           "pro", "sub", "super", "infra", "inter", "non", "sed", "autem", "igitur",
                           "nostre", "nostri", "vostre", "toutes", "fois", "foiz", "assises"}
            if candidate.lower() not in common_words and len(candidate) > 3:
                entities.append(NamedEntity(
                    text=candidate,
                    label="PER",
                    start=match.start(1),
                    end=match.end(1),
                    source="rule",
                    confidence=0.65
                ))

        # Lieux (préfixes toponymiques)
        loc_pattern = r"\b(?:" + "|".join(TOPONYM_PREFIXES) + r")\s+(?:de|du|des|di|d'|del|del')?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        for match in re.finditer(loc_pattern, text, re.IGNORECASE):
            full_match = match.group(0)
            if len(full_match) > 5:
                entities.append(NamedEntity(
                    text=full_match,
                    label="LOC",
                    start=match.start(),
                    end=match.end(),
                    source="rule",
                    confidence=0.80
                ))

        # Lieux (après prépositions de lieu)
        loc_pattern2 = r"\b(?:in|apud|ad|prope|iuxta|près de|à|en|vers)\s+([A-Z][a-z]+(?:\s+(?:de|du|di|d')?\s*[A-Z][a-z]+)?)"
        for match in re.finditer(loc_pattern2, text, re.IGNORECASE):
            lieu = match.group(1)
            if len(lieu) > 2 and lieu.lower() not in {"ego", "hoc", "illo", "ista", "ipso"}:
                entities.append(NamedEntity(
                    text=lieu,
                    label="LOC",
                    start=match.start(1),
                    end=match.end(1),
                    source="rule",
                    confidence=0.75
                ))

        # Monnaie
        money_pattern = r"\b(\d+)\s*(livres?|sous?|deniers?|solidi?|denarii?|marcs?|francs?|écus?|florins?|turonois?)\b"
        for match in re.finditer(money_pattern, text, re.IGNORECASE):
            entities.append(NamedEntity(
                text=match.group(0),
                label="MONEY",
                start=match.start(),
                end=match.end(),
                source="regex",
                confidence=0.90
            ))

        # Organisations
        org_pattern = r"\b(monasterio|monasterium|ecclesia|église|abbaye|couvent|chapitre|capitulum|conuentus|conventus|universitas)\s+(?:de|du|des|di|d\'|del|del\')?\s*([A-Z][a-z]+)"
        for match in re.finditer(org_pattern, text, re.IGNORECASE):
            entities.append(NamedEntity(
                text=match.group(0),
                label="ORG",
                start=match.start(),
                end=match.end(),
                source="rule",
                confidence=0.80
            ))

        return entities

    def _extract_by_spacy(self, text: str) -> List[NamedEntity]:
        """Extraction via spaCy."""
        entities = []
        doc = self.nlp_spacy(text)

        label_map = {
            "PER": "PER", "PERSON": "PER",
            "LOC": "LOC", "GPE": "LOC",
            "ORG": "ORG",
            "DATE": "DATE",
            "MONEY": "MONEY"
        }

        for ent in doc.ents:
            label = label_map.get(ent.label_, ent.label_)
            conf = 0.75 if label in ["PER", "LOC"] else 0.65
            entities.append(NamedEntity(
                text=ent.text,
                label=label,
                start=ent.start_char,
                end=ent.end_char,
                source="spacy",
                confidence=conf
            ))

        return entities

    def _extract_by_stanza(self, text: str) -> List[NamedEntity]:
        """Extraction via Stanza."""
        entities = []
        doc = self.nlp_stanza(text)

        for ent in doc.ents:
            entities.append(NamedEntity(
                text=ent.text,
                label=ent.type,
                start=ent.start_char,
                end=ent.end_char,
                source="stanza",
                confidence=0.80
            ))

        return entities

    def _deduplicate_entities(self, entities: List[NamedEntity]) -> List[NamedEntity]:
        """Dédoublonne les entités en gardant la meilleure confidence."""
        seen = {}
        for ent in entities:
            key = (ent.text.lower().strip(), ent.label, ent.start)
            if key not in seen or ent.confidence > seen[key].confidence:
                seen[key] = ent

        return sorted(seen.values(), key=lambda e: e.start)


# ============================================================
# 7. EXTRACTION DE RELATIONS SÉMANTIQUES
# ============================================================

class RelationExtractor:
    """Extrait les relations sémantiques entre entités."""

    def __init__(self, ner: MedievalNER = None):
        self.ner = ner or MedievalNER()

        self.nlp = None
        if SPACY_AVAILABLE:
            for model_name in ["fr_core_news_md", "fr_core_news_sm"]:
                try:
                    self.nlp = spacy.load(model_name)
                    break
                except:
                    continue

    def extract_relations(self, text: str, entities: List[NamedEntity]) -> List[SemanticRelation]:
        """Extrait les relations sémantiques d'un texte."""
        relations = []

        relations.extend(self._extract_by_patterns(text))

        if self.nlp:
            relations.extend(self._extract_by_syntax(text, entities))

        relations.extend(self._extract_by_proximity(text, entities))

        # Dédoublonner
        seen = set()
        unique_relations = []
        for rel in relations:
            key = (rel.subject.lower().strip(), rel.predicate, rel.object.lower().strip())
            if key not in seen and rel.subject and rel.object:
                seen.add(key)
                unique_relations.append(rel)

        return unique_relations

    def _extract_by_patterns(self, text: str) -> List[SemanticRelation]:
        """Extrait les relations par patterns regex."""
        relations = []

        # Relation : donne_à
        for pattern in RELATION_PATTERNS["donne_à"]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2 and all(g.strip() for g in groups[:2]):
                    subject = groups[0].strip()
                    obj = groups[1].strip()
                    if len(subject) > 2 and len(obj) > 2:
                        relations.append(SemanticRelation(
                            subject=subject,
                            predicate="donne_à",
                            object=obj,
                            sentence=text[max(0, match.start()-30):min(len(text), match.end()+30)].strip(),
                            confidence=0.75
                        ))

        # Relation : date_de
        for pattern in RELATION_PATTERNS["date_de"]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                date_text = match.group(1) if match.groups() else match.group(0)
                if date_text and len(date_text.strip()) > 3:
                    relations.append(SemanticRelation(
                        subject="document",
                        predicate="date_de",
                        object=date_text.strip(),
                        sentence=text[max(0, match.start()-30):min(len(text), match.end()+30)].strip(),
                        confidence=0.85
                    ))

        # Relation : lieu_de
        for pattern in RELATION_PATTERNS["lieu_de"]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.groups():
                    lieu = match.group(1).strip()
                    if len(lieu) > 2:
                        relations.append(SemanticRelation(
                            subject="document",
                            predicate="lieu_de",
                            object=lieu,
                            sentence=text[max(0, match.start()-30):min(len(text), match.end()+30)].strip(),
                            confidence=0.70
                        ))

        # Relation : signe_par
        for pattern in RELATION_PATTERNS["signe_par"]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.groups():
                    signataire = match.group(1).strip()
                    if len(signataire) > 2:
                        relations.append(SemanticRelation(
                            subject=signataire,
                            predicate="signe",
                            object="document",
                            sentence=text[max(0, match.start()-30):min(len(text), match.end()+30)].strip(),
                            confidence=0.80
                        ))

        # Relation : témoin
        for pattern in RELATION_PATTERNS["témoin"]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.groups():
                    temoin = match.group(1).strip()
                    if len(temoin) > 2:
                        relations.append(SemanticRelation(
                            subject=temoin,
                            predicate="témoin_de",
                            object="document",
                            sentence=text[max(0, match.start()-30):min(len(text), match.end()+30)].strip(),
                            confidence=0.75
                        ))

        # Relation : monnaie
        for pattern in RELATION_PATTERNS["monnaie"]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                montant = match.group(0).strip()
                if len(montant) > 1:
                    relations.append(SemanticRelation(
                        subject="transaction",
                        predicate="montant",
                        object=montant,
                        sentence=text[max(0, match.start()-30):min(len(text), match.end()+30)].strip(),
                        confidence=0.90
                    ))

        return relations

    def _extract_by_syntax(self, text: str, entities: List[NamedEntity]) -> List[SemanticRelation]:
        """Extraction par analyse syntaxique spaCy."""
        relations = []

        if not self.nlp:
            return relations

        doc = self.nlp(text)

        donation_verbs = {"donne", "donavit", "concessit", "tradidit", "assignavit",
                          "legavit", "transmisit", "concedit", "donner", "donna",
                          "donne", "donna", "concede", "concedit"}

        for token in doc:
            if token.lemma_.lower() in donation_verbs:
                subject = None
                for child in token.children:
                    if child.dep_ in {"nsubj", "nsubj:pass"}:
                        subject = child.text
                        break

                obj = None
                for child in token.children:
                    if child.dep_ in {"dobj", "obj", "iobj"}:
                        obj = child.text
                        break

                if subject and obj and len(subject) > 2 and len(obj) > 2:
                    relations.append(SemanticRelation(
                        subject=subject,
                        predicate="donne_à",
                        object=obj,
                        sentence=token.sent.text,
                        confidence=0.70
                    ))

        return relations

    def _extract_by_proximity(self, text: str, entities: List[NamedEntity]) -> List[SemanticRelation]:
        """Extrait les relations implicites par proximité."""
        relations = []

        pers = [e for e in entities if e.label == "PER"]
        dates = [e for e in entities if e.label == "DATE"]
        locs = [e for e in entities if e.label == "LOC"]

        # Personne + Date proches
        for p in pers:
            for d in dates:
                if abs(p.start - d.start) < 80 and p.text != d.text:
                    relations.append(SemanticRelation(
                        subject=p.text,
                        predicate="date_associée",
                        object=d.text,
                        sentence=text[max(0, min(p.start, d.start)-20):min(len(text), max(p.end, d.end)+20)].strip(),
                        confidence=0.55
                    ))

        # Personne + Lieu proches
        for p in pers:
            for l in locs:
                if abs(p.start - l.start) < 80 and p.text != l.text:
                    relations.append(SemanticRelation(
                        subject=p.text,
                        predicate="associé_à_lieu",
                        object=l.text,
                        sentence=text[max(0, min(p.start, l.start)-20):min(len(text), max(p.end, l.end)+20)].strip(),
                        confidence=0.50
                    ))

        return relations


# ============================================================
# 8. THÉMATISATION
# ============================================================

class MedievalThematizer:
    """Identifie les thèmes dominants d'un texte médiéval."""

    THEME_KEYWORDS = {
        "juridique": {
            "carta", "charta", "charte", "chartre", "acte", "instrumentum",
            "concessio", "donatio", "traditio", "testamentum", "contractus",
            "donne", "donna", "concedit", "concessit", "tradidit", "assignavit",
            "sigillum", "sigillo", "sceau", "témoin", "testis", "testimonium",
            "notarius", "notaire", "tabellio", "enregistré", "enregistrée",
            "assise", "assises", "justice", "jugement", "droit", "loi", "coutume",
            "appel", "cause", "partie", "défendeur", "demandeur", "procès"
        },
        "religieux": {
            "dominus", "domine", "deus", "christus", "spiritus sanctus",
            "virgo maria", "sanctus", "sancta", "benedictus", "amen",
            "ecclesia", "monasterium", "abbas", "prior", "frater", "pater",
            "prieur", "abbé", "moine", "église", "monastère", "couvent",
            "oraison", "prière", "missa", "messe", "divin", "sacré", "bénit",
            "credo", "pater noster", "ave maria", "gratia", "salutaris"
        },
        "administratif": {
            "registrum", "register", "comptum", "compte", "computus",
            "censu", "cens", "taille", "gabelle", "impôt", "taxe",
            "revenu", "dépense", "recette", "débit", "crédit",
            "bailliage", "sénéchaussée", "parlement", "conseil", "chambre",
            "receveur", "receveur", "trésorier", "trésor", "comptable"
        },
        "noblesse_féodal": {
            "rex", "regina", "dux", "comes", "marchio", "vicecomes", "baro",
            "roi", "reine", "duc", "comte", "marquis", "vicomte", "baron",
            "chevalier", "fief", "seigneurie", "domaine", "hommage", "foi",
            "vassal", "suzerain", "banneret", "banneresse", "châtelain",
            "seigneur", "sire", "dame", "maître", "messire"
        },
        "commerce": {
            "mercator", "mercatoris", "negotium", "negocium",
            "marchand", "marchandise", "commerce", "métier", "corporation",
            "guilde", "mestier", "artisan", "boutique", "marché", "foire",
            "prix", "valeur", "échange", "vente", "achat", "contrat",
            "marché", "foire", "poids", "mesure", "monnaie", "denier"
        }
    }

    OPENING_FORMULAS = {
        "juridique": [
            "in nomine", "notum sit", "sciant presentes", "omnibus presentibus",
            "a tous ceux", "sachent", "soit connu", "savoir faisons"
        ],
        "religieux": [
            "in nomine domini", "dominus vobiscum", "pater noster",
            "au nom du seigneur", "bénédiction", "oraison", "deus"
        ],
        "administratif": [
            "computus", "registrum", "memorandum", "account",
            "compte rendu", "état", "bilan", "receveur"
        ]
    }

    def identify_themes(self, text: str) -> List[Tuple[str, float]]:
        """Identifie les thèmes dominants d'un texte."""
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        scores = {}

        # 1. Scoring par mots-clés
        for theme, keywords in self.THEME_KEYWORDS.items():
            matches = words & keywords
            score = len(matches) / max(len(keywords) * 0.1, 1)
            if score > 0:
                scores[theme] = score

        # 2. Scoring par formules d'ouverture
        for theme, formulas in self.OPENING_FORMULAS.items():
            for formula in formulas:
                if formula.lower() in text_lower[:200]:
                    scores[theme] = scores.get(theme, 0) + 2.0

        # 3. Normaliser et trier
        if scores:
            max_score = max(scores.values())
            scores = {k: min(v / max_score, 1.0) for k, v in scores.items()}

        themes = [(k, round(v, 3)) for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True) if v > 0.15]

        return themes

    def topic_modeling_lda(self, texts: List[str], n_topics: int = 5) -> Dict[str, Any]:
        """Topic modeling LDA sur un corpus de transcriptions."""
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn non installé"}

        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words="french",
            max_features=1000
        )

        try:
            tf = vectorizer.fit_transform(texts)
        except:
            return {"error": "Vectorisation impossible"}

        lda = LatentDirichletAllocation(
            n_components=min(n_topics, len(texts)),
            max_iter=10,
            learning_method="online",
            random_state=42
        )
        lda.fit(tf)

        feature_names = vectorizer.get_feature_names_out()
        topics = {}

        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[f"topic_{topic_idx}"] = {
                "words": top_words,
                "weight": float(topic.sum())
            }

        return topics


# ============================================================
# 9. PIPELINE NLP COMPLET
# ============================================================

class MedievalNLPPipeline:
    """Pipeline NLP complet pour l'analyse de transcriptions médiévales."""

    def __init__(self):
        self.normalizer = MedievalNormalizer()
        self.ner = MedievalNER()
        self.relation_extractor = RelationExtractor(self.ner)
        self.thematizer = MedievalThematizer()

    def analyze_document(self, document_id: str, transcription: str) -> DocumentAnalysis:
        """Analyse complète d'un document."""
        # 1. Normalisation
        normalized = self.normalizer.normalize(transcription)

        # 2. Détection de langue
        language = self.normalizer.detect_language(transcription)

        # 3. NER
        entities = self.ner.extract_entities(normalized, document_id)

        # 4. Relations
        relations = self.relation_extractor.extract_relations(normalized, entities)

        # 5. Thèmes
        themes = self.thematizer.identify_themes(normalized)
        theme_labels = [t[0] for t in themes]

        # 6. Confiance globale
        avg_conf = sum(e.confidence for e in entities) / len(entities) if entities else 0.0

        return DocumentAnalysis(
            document_id=document_id,
            transcription=transcription,
            language=language,
            entities=entities,
            relations=relations,
            themes=theme_labels,
            normalized_text=normalized,
            confidence=round(avg_conf, 3)
        )

    def analyze_corpus(self, transcriptions: Dict[str, str]) -> List[DocumentAnalysis]:
        """Analyse un corpus complet de transcriptions."""
        results = []
        total = len(transcriptions)

        print(f"\n Analyse NLP de {total} documents...")
        print("=" * 60)

        for i, (doc_id, text) in enumerate(transcriptions.items(), 1):
            print(f"  [{i}/{total}] {doc_id}...", end=" ")
            analysis = self.analyze_document(doc_id, text)
            results.append(analysis)
            print(f"{len(analysis.entities)} entités | {len(analysis.relations)} relations | langue: {analysis.language}")

        print(f"\n Analyse terminée. {len(results)} documents traités.")
        return results


# ============================================================
# 10. EXPORT JSON ET RAPPORTS
# ============================================================

def export_to_json(analysis: DocumentAnalysis, output_path: str):
    """Exporte un DocumentAnalysis au format JSON structuré."""
    data = {
        "document_id": analysis.document_id,
        "transcription": analysis.transcription,
        "normalized_text": analysis.normalized_text,
        "language": analysis.language,
        "confidence": analysis.confidence,
        "entities": [
            {
                "text": e.text,
                "type": e.label,
                "start": e.start,
                "end": e.end,
                "source": e.source,
                "confidence": e.confidence
            }
            for e in analysis.entities
        ],
        "relations": [
            {
                "subject": r.subject,
                "predicate": r.predicate,
                "object": r.object,
                "sentence": r.sentence,
                "confidence": r.confidence
            }
            for r in analysis.relations
        ],
        "themes": analysis.themes,
        "metadata": {
            "processed_at": datetime.now().isoformat(),
            "pipeline_version": "1.0.0",
            "tools": {
                "spacy": SPACY_AVAILABLE,
                "stanza": STANZA_AVAILABLE,
                "sklearn": SKLEARN_AVAILABLE
            }
        }
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def generate_corpus_report(analyses: List[DocumentAnalysis], output_path: str):
    """Génère un rapport statistique sur le corpus analysé."""

    total_docs = len(analyses)
    total_entities = sum(len(a.entities) for a in analyses)
    total_relations = sum(len(a.relations) for a in analyses)

    entity_types = Counter()
    for a in analyses:
        for e in a.entities:
            entity_types[e.label] += 1

    relation_types = Counter()
    for a in analyses:
        for r in a.relations:
            relation_types[r.predicate] += 1

    theme_counts = Counter()
    for a in analyses:
        for t in a.themes:
            theme_counts[t] += 1

    lang_counts = Counter(a.language for a in analyses)

    report = {
        "corpus_stats": {
            "total_documents": total_docs,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "avg_entities_per_doc": round(total_entities / total_docs, 2) if total_docs else 0,
            "avg_relations_per_doc": round(total_relations / total_docs, 2) if total_docs else 0
        },
        "entity_distribution": dict(entity_types),
        "relation_distribution": dict(relation_types),
        "theme_distribution": dict(theme_counts),
        "language_distribution": dict(lang_counts),
        "documents": [
            {
                "document_id": a.document_id,
                "language": a.language,
                "num_entities": len(a.entities),
                "num_relations": len(a.relations),
                "themes": a.themes,
                "confidence": a.confidence
            }
            for a in analyses
        ]
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


# ============================================================
# 11. CHARGEMENT DES TRANSCRIPTIONS
# ============================================================

def load_transcriptions(input_path: str) -> Dict[str, str]:
    """
    Charge les transcriptions depuis un fichier.

    Formats supportés :
      - Fichier .txt (format prepare_dataset.py) : image_name\ttext\n
      - Fichier .json : liste de dicts
      - Fichier .txt simple : une transcription par ligne
    """
    transcriptions = {}

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    doc_id = item.get("document_id", item.get("img_name", f"doc_{len(transcriptions)}"))
                    transcriptions[doc_id] = item.get("transcription", item.get("text", ""))
            elif isinstance(data, dict):
                for doc_id, text in data.items():
                    transcriptions[doc_id] = text
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            if "\t" in line:
                parts = line.split("\t", 1)
                doc_id = parts[0].replace(".png", "").replace(".jpg", "")
                text = parts[1] if len(parts) > 1 else ""
            else:
                doc_id = f"doc_{i:04d}"
                text = line

            transcriptions[doc_id] = text

    return transcriptions


# ============================================================
# 12. INTERFACE EN LIGNE DE COMMANDE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline NLP pour transcriptions HTR de manuscrits médiévaux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  # Analyse d'un fichier de transcriptions HTR
  python nlp_analysis.py --input ./inference_results/evaluation_test_tridis/predictions.txt \
                         --output ./nlp_results/

  # Analyse avec rapport de corpus complet
  python nlp_analysis.py --input ./data/catmus/test.txt \
                         --output ./nlp_results/ \
                         --mode full

  # Analyse d'une seule transcription
  python nlp_analysis.py --text "In nomine Domini amen. Ego frater Johannes..." \
                         --output ./nlp_results/single.json
        """
    )

    parser.add_argument("--input", "-i", type=str, default=None,
                       help="Chemin vers le fichier de transcriptions")
    parser.add_argument("--output", "-o", type=str, default="./nlp_results",
                       help="Répertoire de sortie")
    parser.add_argument("--text", "-t", type=str, default=None,
                       help="Texte unique à analyser (alternative à --input)")
    parser.add_argument("--mode", type=str, default="standard",
                       choices=["standard", "full"],
                       help="Mode : standard (NER+relations+thèmes) ou full (+rapport corpus)")
    parser.add_argument("--no-spacy", action="store_true",
                       help="Désactiver spaCy")
    parser.add_argument("--no-stanza", action="store_true",
                       help="Désactiver Stanza")
    parser.add_argument("--no-rules", action="store_true",
                       help="Désactiver les règles historiques")

    args = parser.parse_args()

    if not args.input and not args.text:
        parser.error("Il faut spécifier --input ou --text")

    print("=" * 70)
    print("   PIPELINE NLP — Analyse Linguistique de Manuscrits Médiévaux")
    print("   Phase 2 — MD5-2026")
    print("=" * 70)

    pipeline = MedievalNLPPipeline()

    if args.no_spacy:
        pipeline.ner.use_spacy = False
    if args.no_stanza:
        pipeline.ner.use_stanza = False
    if args.no_rules:
        pipeline.ner.use_rules = False

    if args.text:
        print(f"\n Analyse du texte fourni...")
        analysis = pipeline.analyze_document("single_document", args.text)

        output_path = os.path.join(args.output, "single_analysis.json")
        export_to_json(analysis, output_path)

        print(f"\n Résultats sauvegardés : {output_path}")
        print(f"\n Résumé :")
        print(f"   Langue détectée : {analysis.language}")
        print(f"   Entités : {len(analysis.entities)}")
        for e in analysis.entities:
            print(f"      [{e.label}] {e.text} (conf: {e.confidence:.2f})")
        print(f"   Relations : {len(analysis.relations)}")
        for r in analysis.relations:
            print(f"      {r.subject} --{r.predicate}--> {r.object}")
        print(f"   Thèmes : {analysis.themes}")

    else:
        print(f"\n Chargement des transcriptions depuis : {args.input}")
        transcriptions = load_transcriptions(args.input)
        print(f"   {len(transcriptions)} documents chargés")

        analyses = pipeline.analyze_corpus(transcriptions)

        os.makedirs(args.output, exist_ok=True)
        for analysis in analyses:
            output_path = os.path.join(args.output, f"{analysis.document_id}_nlp.json")
            export_to_json(analysis, output_path)

        print(f"\n Résultats individuels sauvegardés dans : {args.output}")

        if args.mode == "full":
            report_path = os.path.join(args.output, "corpus_report.json")
            report = generate_corpus_report(analyses, report_path)

            print(f"\n" + "=" * 60)
            print("   RAPPORT DE CORPUS")
            print("=" * 60)
            print(f"   Documents analysés : {report['corpus_stats']['total_documents']}")
            print(f"   Entités totales : {report['corpus_stats']['total_entities']}")
            print(f"   Relations totales : {report['corpus_stats']['total_relations']}")
            print(f"\n   Distribution des entités :")
            for label, count in report['entity_distribution'].items():
                print(f"      {label}: {count}")
            print(f"\n   Distribution des thèmes :")
            for theme, count in report['theme_distribution'].items():
                print(f"      {theme}: {count}")
            print(f"\n   Rapport sauvegardé : {report_path}")

    print(f"\n Pipeline NLP terminé avec succès !")


# ============================================================
# 13. TESTS
# ============================================================

def test_nlp_pipeline():
    """Test rapide du pipeline NLP sur un texte médiéval factice."""
    print("=" * 70)
    print("   TEST DU PIPELINE NLP")
    print("=" * 70)

    text = """In nomine Domini amen. Ego frater Johannes, prior monasterio Sancti Dionysii,
    dono et concedo tibi frater Petrus terram meam in villa de Parisius, 
    pro precio centum solidorum. Datum anno Domini millesimo trecentesimo septuagesimo quinto,
    mense ianuarii, die sabbati. Testes : frater Martinus et frater Robertus.
    Sigillum capituli appositum est."""

    pipeline = MedievalNLPPipeline()
    analysis = pipeline.analyze_document("test_doc_001", text)

    print(f"\n Texte original :")
    print(f"   {text[:100]}...")

    print(f"\n Langue détectée : {analysis.language}")

    print(f"\n Entités nommées ({len(analysis.entities)}):")
    for e in analysis.entities:
        print(f"   [{e.label:5}] '{e.text}' (source: {e.source:6}, conf: {e.confidence:.2f})")

    print(f"\n Relations sémantiques ({len(analysis.relations)}):")
    for r in analysis.relations:
        print(f"   {r.subject} --[{r.predicate}]--> {r.object} (conf: {r.confidence:.2f})")

    print(f"\n Thèmes identifiés : {analysis.themes}")

    # Test export
    test_output = "/tmp/test_nlp_output.json"
    export_to_json(analysis, test_output)
    print(f"\n Export test : {test_output}")

    # Afficher le JSON
    with open(test_output, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n Structure JSON de sortie :")
    print(json.dumps(data, ensure_ascii=False, indent=2)[:1500] + "...")

    print("\n Test NLP réussi !")
    return True


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Aucun argument fourni. Lancement des tests...")
        test_nlp_pipeline()
    else:
        main()
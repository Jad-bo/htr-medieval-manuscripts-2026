#!/usr/bin/env python3
"""
Script de pont : Phase 1 (HTR) → Phase 2 (NLP)

Prend les transcriptions produites par inference.py et applique
l'analyse linguistique (NER, relations, thématisation).

Usage:
    # À partir du fichier predictions.txt de l'évaluation
    python nlp_from_htr.py \
        --input ./inference_results/evaluation_test_tridis/predictions.txt \
        --output ./nlp_results/

    # À partir du fichier test.txt (ground truth + HTR)
    python nlp_from_htr.py \
        --input ./data/catmus/test.txt \
        --output ./nlp_results/ \
        --mode full

    # Mode batch sur plusieurs fichiers
    python nlp_from_htr.py \
        --input ./inference_results/ \
        --output ./nlp_results/ \
        --batch
"""

import os
import sys
import argparse
import json
import glob
from typing import Dict, List
from collections import Counter

# Import du pipeline NLP
try:
    from nlp_analysis import (
        MedievalNLPPipeline,
        export_to_json as export_nlp_to_json,
        generate_corpus_report,
        load_transcriptions
    )
    NLP_AVAILABLE = True
except ImportError as e:
    print(f" Erreur : Impossible d'importer nlp_analysis.py")
    print(f"   {e}")
    print(f"   Assurez-vous que nlp_analysis.py est dans le PYTHONPATH.")
    sys.exit(1)


def load_predictions_from_file(input_path: str) -> Dict[str, str]:
    """
    Charge les prédictions HTR depuis un fichier.

    Formats supportés :
      - predictions.txt : image_name\ttranscription\n
      - test.txt : image_name\tground_truth\n (on prend la colonne texte)
      - JSON : liste de dicts
    """
    predictions = {}

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    # Détecter le format
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    doc_id = item.get("img_name", item.get("document_id", f"doc_{len(predictions)}"))
                    text = item.get("pred", item.get("transcription", item.get("text", "")))
                    predictions[doc_id] = text
            elif isinstance(data, dict):
                for doc_id, text in data.items():
                    predictions[doc_id] = text
    else:
        # Format .txt (predictions.txt ou test.txt)
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

            predictions[doc_id] = text

    return predictions


def process_single_file(input_path: str, output_dir: str, mode: str = "standard"):
    """Traite un fichier de prédictions HTR."""

    print("=" * 70)
    print("   PHASE 2 : ANALYSE LINGUISTIQUE NLP")
    print("   Entrée : Transcriptions HTR (Phase 1)")
    print("   Sortie : Entités, Relations, Thèmes")
    print("=" * 70)

    print(f"\n Chargement des transcriptions depuis : {input_path}")
    predictions = load_predictions_from_file(input_path)
    print(f"   ✓ {len(predictions)} documents chargés")

    # Initialiser le pipeline NLP
    print(f"\n Initialisation du pipeline NLP...")
    nlp_pipeline = MedievalNLPPipeline()

    # Analyser le corpus
    print(f"\n Analyse linguistique en cours...")
    analyses = nlp_pipeline.analyze_corpus(predictions)

    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)

    # Export individuel
    print(f"\n Export des résultats individuels...")
    for analysis in analyses:
        output_path = os.path.join(output_dir, f"{analysis.document_id}_nlp.json")
        export_nlp_to_json(analysis, output_path)

    print(f"   ✓ {len(analyses)} fichiers JSON sauvegardés dans {output_dir}")

    # Rapport de corpus
    if mode == "full":
        print(f"\n Génération du rapport de corpus...")
        report_path = os.path.join(output_dir, "corpus_report.json")
        report = generate_corpus_report(analyses, report_path)

        # Afficher le résumé
        print(f"\n" + "=" * 60)
        print("   RAPPORT DE CORPUS NLP")
        print("=" * 60)
        print(f"\n Statistiques globales :")
        print(f"   Documents analysés    : {report['corpus_stats']['total_documents']}")
        print(f"   Entités extraites     : {report['corpus_stats']['total_entities']}")
        print(f"   Relations extraites   : {report['corpus_stats']['total_relations']}")
        print(f"   Moy. entités/doc      : {report['corpus_stats']['avg_entities_per_doc']}")
        print(f"   Moy. relations/doc    : {report['corpus_stats']['avg_relations_per_doc']}")

        print(f"\n Distribution des entités :")
        for label, count in sorted(report['entity_distribution'].items(), key=lambda x: -x[1]):
            bar = "█" * int(count / max(report['entity_distribution'].values()) * 30)
            print(f"   {label:10} : {count:4d} {bar}")

        print(f"\n Distribution des relations :")
        for rel, count in sorted(report['relation_distribution'].items(), key=lambda x: -x[1]):
            print(f"   {rel:20} : {count}")

        print(f"\n Distribution des thèmes :")
        for theme, count in sorted(report['theme_distribution'].items(), key=lambda x: -x[1]):
            bar = "█" * int(count / max(report['theme_distribution'].values()) * 30)
            print(f"   {theme:20} : {count:3d} {bar}")

        print(f"\n Distribution des langues :")
        for lang, count in report['language_distribution'].items():
            print(f"   {lang:20} : {count}")

        print(f"\n Rapport sauvegardé : {report_path}")

    print(f"\n Phase 2 NLP terminée avec succès !")
    print(f"   Résultats dans : {output_dir}")

    return analyses


def process_batch(input_dir: str, output_dir: str):
    """Traite tous les fichiers predictions.txt d'un répertoire."""

    # Trouver tous les fichiers predictions.txt
    pattern = os.path.join(input_dir, "**", "predictions.txt")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f" Aucun fichier predictions.txt trouvé dans {input_dir}")
        return

    print(f"\n {len(files)} fichier(s) trouvé(s) :")
    for f in files:
        print(f"   • {f}")

    all_analyses = []

    for file_path in files:
        # Créer un sous-répertoire basé sur le nom du dossier parent
        rel_path = os.path.relpath(file_path, input_dir)
        sub_dir = os.path.dirname(rel_path).replace(os.sep, "_")
        if not sub_dir:
            sub_dir = "default"

        file_output_dir = os.path.join(output_dir, sub_dir)
        print(f"\n{'='*70}")
        print(f"   Traitement : {rel_path}")
        print(f"   Sortie     : {file_output_dir}")
        print(f"{'='*70}")

        analyses = process_single_file(file_path, file_output_dir, mode="full")
        all_analyses.extend(analyses)

    # Rapport global
    if all_analyses:
        global_report_path = os.path.join(output_dir, "global_corpus_report.json")
        print(f"\n Génération du rapport global...")
        generate_corpus_report(all_analyses, global_report_path)
        print(f"   ✓ Rapport global : {global_report_path}")

    return all_analyses


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 NLP : Analyse linguistique des transcriptions HTR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
═══════════════════════════════════════════════════════════════════════════════
  PHASE 2 : HTR → NLP
═══════════════════════════════════════════════════════════════════════════════

Ce script prend les transcriptions produites par la Phase 1 (HTR) et en
extrait les entités nommées, relations sémantiques et thèmes.

Exemples :

  # Analyse d'un fichier predictions.txt
  python nlp_from_htr.py \
      --input ./inference_results/evaluation_test_tridis/predictions.txt \
      --output ./nlp_results/

  # Analyse complète avec rapport de corpus
  python nlp_from_htr.py \
      --input ./data/catmus/test.txt \
      --output ./nlp_results/ \
      --mode full

  # Traitement batch de plusieurs répertoires
  python nlp_from_htr.py \
      --input ./inference_results/ \
      --output ./nlp_results/ \
      --batch

Fichiers de sortie (par document) :
  ./nlp_results/
    ├── {doc_id}_nlp.json          # Analyse complète du document
    └── corpus_report.json          # Rapport statistique global
        """
    )

    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Fichier de prédictions HTR ou répertoire (--batch)")
    parser.add_argument("--output", "-o", type=str, default="./nlp_results",
                       help="Répertoire de sortie (défaut: ./nlp_results)")
    parser.add_argument("--mode", type=str, default="standard",
                       choices=["standard", "full"],
                       help="Mode : standard ou full (avec rapport corpus)")
    parser.add_argument("--batch", action="store_true",
                       help="Mode batch : traite tous les predictions.txt du répertoire")

    args = parser.parse_args()

    if args.batch:
        if not os.path.isdir(args.input):
            print(f" Erreur : --input doit être un répertoire en mode --batch")
            sys.exit(1)
        process_batch(args.input, args.output)
    else:
        if not os.path.exists(args.input):
            print(f" Erreur : Fichier introuvable : {args.input}")
            sys.exit(1)
        process_single_file(args.input, args.output, args.mode)


if __name__ == "__main__":
    main()
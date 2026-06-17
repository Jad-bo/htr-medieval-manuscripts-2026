"""
Pipeline principal (End-to-End) pour le traitement de manuscrits médiévaux.

Architecture complète du pipeline :
    Image page de manuscrit (TIFF/JPEG)
              ↓
    [1] Prétraitement (preprocessing.py)
        → Deskewing, CLAHE, Binarisation Sauvola
              ↓
    [2] Segmentation (segmentation.py)
        → Détection régions, extraction lignes, polygones, ordre de lecture
              ↓
    [3] Extraction des images de lignes
        → Images individuelles prêtes pour le HTR
              ↓
    [4] HTR / Transcription (inference.py)
        → TrOCR fine-tuné avec LoRA
              ↓
    [5] Agrégation (data_contract.py)
        → JSON structuré avec transcriptions, confiances, polygones
              ↓
    [6] Évaluation (inference.py --mode evaluate)
        → CER, WER, intervalles de confiance

Conforme au brief MD5-2026 Volet 1/2.
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional

# Import des modules du projet
from preprocessing import preprocess_pipeline
from segmentation import segment_manuscript_page, visualize_segmentation, SegmentationResult
from data_contract import generate_line_entry, save_data_contract

# Pour l'HTR, on importe dynamiquement pour éviter les dépendances lourdes au chargement
try:
    from inference import load_htr_model, transcribe_image
    HTR_AVAILABLE = True
except ImportError:
    HTR_AVAILABLE = False
    print("  Module inference.py non disponible. Le HTR sera simulé.")


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Seuils de qualité du brief
CER_VALIDATION_THRESHOLD = 0.15   # < 15%
CER_EXCELLENCE_THRESHOLD = 0.08   # < 8%
CONFIDENCE_REVIEW_THRESHOLD = 0.80  # Flag needs_review si < 0.80


def set_seed(seed: int = 42):
    """Fixe les seeds pour la reproductibilité."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. PIPELINE PRINCIPAL
# ============================================================

def run_pipeline(
    image_path: str,
    document_id: str,
    output_dir: str = "./pipeline_output",
    checkpoint_path: str = "./checkpoints_production/best_model",
    model_base_name: str = "microsoft/trocr-large-handwritten",
    segmentation_method: str = "kraken",
    segmentation_model: str = None,
    skip_preprocessing: bool = False,
    skip_segmentation: bool = False,
    mock_htr: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Exécute le pipeline complet de traitement d'un manuscrit.

    Args:
        image_path: Chemin vers l'image de la page de manuscrit
        document_id: Identifiant unique du document
        output_dir: Répertoire de sortie
        checkpoint_path: Chemin vers le modèle HTR fine-tuné
        model_base_name: Modèle de base TrOCR
        segmentation_method: "kraken" ou "yolo"
        segmentation_model: Chemin vers un modèle de segmentation personnalisé
        skip_preprocessing: Si True, saute l'étape de prétraitement
        skip_segmentation: Si True, saute la segmentation (utilise des lignes déjà extraites)
        mock_htr: Si True, simule le HTR (pour les tests sans GPU)
        verbose: Affiche les logs détaillés

    Returns:
        Dict avec les résultats du pipeline
    """
    set_seed(42)

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*70}")
        print(f"   PIPELINE HTR — Document : {document_id}")
        print(f"{'='*70}")

    # ============================================================
    # ÉTAPE 1 : PRÉTRAITEMENT
    # ============================================================
    if not skip_preprocessing:
        if verbose:
            print(f"\n[1/5] Prétraitement de l'image...")
            print(f"      → Deskewing, CLAHE, Binarisation Sauvola")

        processed_img = preprocess_pipeline(image_path)

        processed_path = os.path.join(output_dir, f"{document_id}_preprocessed.png")
        import cv2
        cv2.imwrite(processed_path, processed_img)

        if verbose:
            print(f"       Image prétraitée sauvegardée : {processed_path}")
    else:
        processed_path = image_path
        if verbose:
            print(f"\n[1/5] Prétraitement sauté — utilisation de : {image_path}")

    # ============================================================
    # ÉTAPE 2 : SEGMENTATION
    # ============================================================
    if not skip_segmentation:
        if verbose:
            print(f"\n[2/5] Segmentation de la page...")
            print(f"      → Méthode : {segmentation_method.upper()}")
            print(f"      → Détection régions + Extraction lignes + Polygones")

        seg_output_dir = os.path.join(output_dir, "segmentation")

        segmentation = segment_manuscript_page(
            image_path=processed_path,
            output_dir=seg_output_dir,
            page_id=document_id,
            method=segmentation_method,
            model_path=segmentation_model,
            extract_lines=True,
            export_json=True,
            export_pagexml=True
        )

        # Visualisation
        viz_path = os.path.join(seg_output_dir, f"{document_id}_visualization.png")
        visualize_segmentation(segmentation, output_path=viz_path)

        if verbose:
            print(f"       {len(segmentation.all_lines)} lignes détectées")
            print(f"       PAGE XML : {os.path.join(seg_output_dir, f'{document_id}.page.xml')}")
    else:
        # Charger une segmentation existante
        seg_json = os.path.join(output_dir, "segmentation", f"{document_id}_segmentation.json")
        if os.path.exists(seg_json):
            import json
            with open(seg_json, "r") as f:
                seg_data = json.load(f)
            # Reconstruire l'objet SegmentationResult (simplifié)
            segmentation = None  # À implémenter si besoin
        else:
            raise FileNotFoundError(f"Segmentation non trouvée : {seg_json}")

        if verbose:
            print(f"\n[2/5] Segmentation sautée — chargement existant")

    # ============================================================
    # ÉTAPE 3 : HTR / TRANSCRIPTION
    # ============================================================
    if verbose:
        print(f"\n[3/5] Transcription HTR...")

    lines_dir = os.path.join(output_dir, "segmentation", "lines")

    if mock_htr or not HTR_AVAILABLE:
        # Mode simulation (pour les tests sans modèle)
        if verbose:
            print(f"      → Mode MOCK (pas de modèle HTR chargé)")

        transcriptions = _mock_transcription(segmentation, lines_dir)
    else:
        # Vrai HTR avec le modèle fine-tuné
        if verbose:
            print(f"      → Chargement du modèle : {checkpoint_path}")

        model, processor, config = load_htr_model(checkpoint_path, model_base_name)
        device = next(model.parameters()).device.type

        transcriptions = []
        lines = segmentation.get_lines_sorted()

        for i, line in enumerate(lines):
            line_img_path = os.path.join(lines_dir, f"{line.line_id}.png")

            if os.path.exists(line_img_path):
                pred_text = transcribe_image(model, processor, line_img_path, device)
                confidence = _estimate_confidence(pred_text, line.confidence)
            else:
                pred_text = "[IMAGE_MANQUANTE]"
                confidence = 0.0

            transcriptions.append({
                "line_id": line.line_id,
                "text": pred_text,
                "confidence": confidence,
                "polygon": line.polygon,
                "reading_order": line.reading_order
            })

            if verbose and (i + 1) % 5 == 0:
                print(f"      [{i+1}/{len(lines)}] {line.line_id} : {pred_text[:50]}...")

    if verbose:
        print(f"       {len(transcriptions)} lignes transcrites")

    # ============================================================
    # ÉTAPE 4 : AGRÉGATION (Data Contract)
    # ============================================================
    if verbose:
        print(f"\n[4/5] Agrégation des transcriptions...")
        print(f"      → Génération du Data Contract JSON")

    lines_entries = []
    for trans in transcriptions:
        entry = generate_line_entry(
            line_id=trans["line_id"],
            text=trans["text"],
            confidence=trans["confidence"],
            polygon=trans["polygon"]
        )
        lines_entries.append(entry)

    json_output_path = os.path.join(output_dir, f"{document_id}_transcription.json")
    save_data_contract(json_output_path, document_id, lines_entries)

    # Statistiques
    needs_review_count = sum(1 for e in lines_entries if e["needs_review"])
    review_rate = needs_review_count / len(lines_entries) if lines_entries else 0
    avg_confidence = sum(e["confidence"] for e in lines_entries) / len(lines_entries) if lines_entries else 0

    if verbose:
        print(f"       Data Contract sauvegardé : {json_output_path}")
        print(f"       Confiance moyenne : {avg_confidence:.3f}")
        print(f"       Lignes à réviser : {needs_review_count}/{len(lines_entries)} ({review_rate:.1%})")

    # ============================================================
    # ÉTAPE 5 : RAPPORT FINAL
    # ============================================================
    if verbose:
        print(f"\n[5/5] Rapport final")

    report = {
        "document_id": document_id,
        "image_path": image_path,
        "output_dir": output_dir,
        "num_lines_detected": len(segmentation.all_lines) if segmentation else 0,
        "num_lines_transcribed": len(transcriptions),
        "avg_confidence": avg_confidence,
        "needs_review_rate": review_rate,
        "needs_review_count": needs_review_count,
        "files_generated": {
            "preprocessed_image": processed_path if not skip_preprocessing else None,
            "segmentation_json": os.path.join(output_dir, "segmentation", f"{document_id}_segmentation.json"),
            "segmentation_pagexml": os.path.join(output_dir, "segmentation", f"{document_id}.page.xml"),
            "visualization": os.path.join(output_dir, "segmentation", f"{document_id}_visualization.png"),
            "lines_directory": lines_dir,
            "transcription_json": json_output_path
        }
    }

    # Sauvegarder le rapport
    report_path = os.path.join(output_dir, f"{document_id}_report.json")
    import json
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"       Rapport sauvegardé : {report_path}")
        print(f"\n{'='*70}")
        print(f"  🎉 PIPELINE TERMINÉ AVEC SUCCÈS")
        print(f"{'='*70}")

    return report


def _mock_transcription(segmentation, lines_dir: str) -> List[Dict]:
    """Simule la transcription pour les tests sans modèle HTR."""
    transcriptions = []
    lines = segmentation.get_lines_sorted()

    for line in lines:
        # Texte factice pour le test
        mock_texts = [
            "In nomine Domini amen",
            "Anno domini millesimo",
            "Ego frater Johannes",
            "testimonium perhibeo",
            "sub sigillo capituli"
        ]
        import random
        text = random.choice(mock_texts)
        confidence = random.uniform(0.70, 0.95)

        transcriptions.append({
            "line_id": line.line_id,
            "text": text,
            "confidence": confidence,
            "polygon": line.polygon,
            "reading_order": line.reading_order
        })

    return transcriptions


def _estimate_confidence(pred_text: str, seg_confidence: float) -> float:
    """
    Estime la confiance de la transcription.
    Combine la confiance de segmentation et des heuristiques sur le texte.
    """
    base_conf = seg_confidence

    # Pénaliser les textes très courts ou très longs
    text_len = len(pred_text.strip())
    if text_len <= 2:
        base_conf *= 0.5
    elif text_len > 100:
        base_conf *= 0.9

    # Pénaliser les caractères suspects (beaucoup de répétitions)
    if len(set(pred_text)) < 3 and text_len > 5:
        base_conf *= 0.3

    return min(base_conf, 1.0)


# ============================================================
# 3. INTERFACE EN LIGNE DE COMMANDE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline HTR complet pour manuscrits médiévaux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  # Pipeline complet sur une page
  python main.py --image ./data/page_001.jpg --id ms_001

  # Avec un modèle HTR personnalisé
  python main.py --image ./data/page_001.jpg --id ms_001 \
      --checkpoint ./mon_modele/best_model

  # Mode test (sans HTR, simulation)
  python main.py --image ./data/page_001.jpg --id ms_001 --mock

  # Segmentation avec YOLO au lieu de Kraken
  python main.py --image ./data/page_001.jpg --id ms_001 --seg-method yolo
        """
    )

    parser.add_argument("--image", "-i", type=str, required=True,
                       help="Chemin vers l'image de la page de manuscrit")
    parser.add_argument("--id", type=str, required=True,
                       help="Identifiant unique du document")
    parser.add_argument("--output", "-o", type=str, default="./pipeline_output",
                       help="Répertoire de sortie (défaut: ./pipeline_output)")
    parser.add_argument("--checkpoint", "-c", type=str,
                       default="./checkpoints_production/best_model",
                       help="Chemin vers le checkpoint du modèle HTR")
    parser.add_argument("--model-base", type=str,
                       default="microsoft/trocr-large-handwritten",
                       help="Modèle de base TrOCR")
    parser.add_argument("--seg-method", type=str, default="kraken",
                       choices=["kraken", "yolo"],
                       help="Méthode de segmentation")
    parser.add_argument("--seg-model", type=str, default=None,
                       help="Chemin vers un modèle de segmentation personnalisé")
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Sauter l'étape de prétraitement")
    parser.add_argument("--skip-segmentation", action="store_true",
                       help="Sauter la segmentation (utiliser des lignes existantes)")
    parser.add_argument("--mock", action="store_true",
                       help="Mode simulation (pas de vrai HTR)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Mode silencieux")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f" Image introuvable : {args.image}")
        sys.exit(1)

    try:
        report = run_pipeline(
            image_path=args.image,
            document_id=args.id,
            output_dir=args.output,
            checkpoint_path=args.checkpoint,
            model_base_name=args.model_base,
            segmentation_method=args.seg_method,
            segmentation_model=args.seg_model,
            skip_preprocessing=args.skip_preprocessing,
            skip_segmentation=args.skip_segmentation,
            mock_htr=args.mock,
            verbose=not args.quiet
        )

        # Afficher un résumé compact
        if args.quiet:
            print(f"{args.id}|{report['num_lines_transcribed']}|{report['avg_confidence']:.3f}|{report['needs_review_rate']:.2%}")

    except Exception as e:
        print(f"\n ERREUR dans le pipeline : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
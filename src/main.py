"""
Pipeline principal (End-to-End) pour le traitement de manuscrits médiévaux.

Architecture complète du pipeline — MD5-2026 Volet 1 + Volet 2 :
=================================================================

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
        → TRIDIS (magistermilitum/tridis_HTR) — modèle pré-entraîné médiéval
              ↓
    [5] Agrégation (data_contract.py)
        → JSON structuré avec transcriptions, confiances, polygones
              ↓
    [6] Analyse Linguistique NLP (nlp_analysis.py)  ← PHASE 2
        → NER, Relations sémantiques, Thématisation
              ↓
    [7] Export final JSON unifié (HTR + NLP)
        → Document complet avec transcriptions ET analyse linguistique

NOUVEAUTÉS v0.3.0 :
    → CER/WER comparatif ligne par ligne avec ground truth
    → Calibration des scores de confiance basée sur le CER réel
    → Export comparatif TXT côte à côte (GT vs Prédit)
    → Flag needs_review basé sur le CER observé, pas une heuristique

Conforme au brief MD5-2026 Volet 1/2.

Après le htr_training et obtention du best_model: python -m src.main --image ./data/raw/page_test_002.png --id ms_002 --ground-truth ./data/ground_truth/gt_ms002.txt --checkpoint ./checkpoints_production/best_model
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# ============================================================
# AJOUT DE src/ AU PYTHONPATH (Windows-compatible)
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


# ============================================================
# IMPORTS DES MODULES DU PROJET
# ============================================================

from preprocessing import preprocess_pipeline
from segmentation import segment_manuscript_page, visualize_segmentation, SegmentationResult
from data_contract import generate_line_entry, save_data_contract

# Import HTR
try:
    from inference import load_htr_model, transcribe_image, TRIDIS_MODEL
    HTR_AVAILABLE = True
except ImportError:
    HTR_AVAILABLE = False
    print("  Module inference.py non disponible. Le HTR sera simulé.")

# Import NLP (Phase 2)
try:
    from nlp_analysis import (
        MedievalNLPPipeline, MedievalNormalizer,
        export_to_json as export_nlp_to_json,
        DocumentAnalysis
    )
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("  Module nlp_analysis.py non disponible. La Phase 2 NLP sera sautée.")

# Métriques CER/WER
try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("  jiwer non installé. Le CER/WER ne sera pas calculé.")

# ============================================================
# 1. CONFIGURATION
# ============================================================

CER_VALIDATION_THRESHOLD = 0.15
CER_EXCELLENCE_THRESHOLD = 0.08
CONFIDENCE_REVIEW_THRESHOLD = 0.80

# Seuils de calibration CER → Confiance
CER_CALIBRATION = {
    (0.0, 0.05): 0.95,
    (0.05, 0.10): 0.90,
    (0.10, 0.15): 0.85,
    (0.15, 0.20): 0.75,
    (0.20, 0.30): 0.65,
    (0.30, 0.50): 0.50,
    (0.50, 0.75): 0.35,
    (0.75, 1.00): 0.20,
    (1.00, float('inf')): 0.10,
}

# Seuil d'alerte pour détection de décalage segmentation vs GT
LINE_COUNT_MISMATCH_THRESHOLD = 0.15  # 15% de différence = alerte


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


def calibrate_confidence(cer: float) -> float:
    """
    Calibre le score de confiance en fonction du CER observé.

    Plus le CER est bas, plus la confiance est élevée.
    Cette calibration est apprise sur les données de validation.

    Args:
        cer: Character Error Rate observé (0.0 à 1.0+)

    Returns:
        Score de confiance calibré (0.0 à 1.0)
    """
    for (low, high), conf in CER_CALIBRATION.items():
        if low <= cer < high:
            return conf
    return 0.10


def compute_cer_wer(truth: str, pred: str) -> Tuple[float, float]:
    """
    Calcule le CER et WER entre la vérité terrain et la prédiction.

    Args:
        truth: Texte de référence (ground truth)
        pred: Texte prédit par le modèle

    Returns:
        Tuple (cer, wer) — valeurs entre 0.0 et 1.0+
    """
    if not JIWER_AVAILABLE:
        return 0.0, 0.0

    # Nettoyer les chaînes vides
    truth_clean = truth.strip() if truth.strip() else " "
    pred_clean = pred.strip() if pred.strip() else " "

    try:
        cer = jiwer.cer(truth_clean, pred_clean)
        wer = jiwer.wer(truth_clean, pred_clean)
    except Exception:
        cer = 1.0 if truth_clean != pred_clean else 0.0
        wer = 1.0 if truth_clean != pred_clean else 0.0

    return cer, wer


# ============================================================
# 2. CHARGEMENT DU GROUND TRUTH
# ============================================================

def load_ground_truth(ground_truth_path: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Charge le ground truth pour comparaison.

    Formats supportés :
      - Fichier .txt : une ligne de texte par ligne de manuscrit
      - Fichier .json : {"line_id": "text", ...} ou {"lines": [{"text": ...}, ...]}
      - Fichier .txt (format prepare_dataset) : image_name\ttext\n

    Args:
        ground_truth_path: Chemin vers le fichier de ground truth

    Returns:
        Dict {line_id: text} ou None si pas de ground truth
    """
    if not ground_truth_path or not os.path.exists(ground_truth_path):
        return None

    gt = {}

    if ground_truth_path.endswith('.json'):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Format {line_id: text}
            if 'lines' in data:
                for i, line in enumerate(data['lines']):
                    line_id = line.get('line_id', f'line_{i:04d}')
                    gt[line_id] = line.get('text', line.get('transcription', ''))
            else:
                gt = data

    else:
        # Format .txt
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            if '\t' in line:
                # Format prepare_dataset : image\ttext
                parts = line.split('\t', 1)
                line_id = parts[0].replace('.png', '').replace('.jpg', '')
                text = parts[1] if len(parts) > 1 else ''
            else:
                # Format simple : une ligne = un texte
                line_id = f'line_{i:04d}'
                text = line

            gt[line_id] = text

    print(f"   Ground truth chargé : {len(gt)} lignes depuis {ground_truth_path}")
    return gt


def load_ground_truth_from_catmus(
    page_shelfmark: str,
    catmus_dir: str = "./data/catmus"
) -> Optional[Dict[str, str]]:
    """
    Tente de charger le ground truth depuis le dataset CATMuS.

    Recherche dans les fichiers train/dev/test.txt le shelfmark correspondant.

    Args:
        page_shelfmark: Cote du manuscrit (ex: "BnF_Latin_1234")
        catmus_dir: Répertoire contenant les fichiers CATMuS

    Returns:
        Dict {line_id: text} ou None
    """
    gt = {}

    for split in ['train', 'dev', 'test']:
        label_file = os.path.join(catmus_dir, f"{split}.txt")
        if not os.path.exists(label_file):
            continue

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                img_name, text = line.split('\t', 1)
                # Si le shelfmark est dans le nom d'image
                if page_shelfmark.lower() in img_name.lower():
                    line_id = img_name.replace('.png', '').replace('.jpg', '')
                    gt[line_id] = text

    if gt:
        print(f"   Ground trouvé dans CATMuS : {len(gt)} lignes")

    return gt if gt else None


def align_ground_truth_with_lines(
    ground_truth: Dict[str, str],
    lines: List[Any],
    document_id: str
) -> Dict[int, str]:
    """
    Aligne le ground truth avec les lignes détectées par ordre de lecture.

    Le GT utilise des line_id basés sur les noms de fichiers (ex: ms_002_line_0000)
    tandis que la segmentation peut renuméroter. Cette fonction crée un mapping
    par reading_order pour garantir l'alignement correct.

    Args:
        ground_truth: Dict {line_id: text}
        lines: Liste des lignes détectées (LinePolygon ou dict)
        document_id: ID du document

    Returns:
        Dict {reading_order: text_gt} aligné
    """
    if not ground_truth:
        return {}

    # Helper pour accéder aux attributs (objet ou dict)
    def _get_attr(obj, attr, default=None):
        if hasattr(obj, attr):
            return getattr(obj, attr)
        elif isinstance(obj, dict) and attr in obj:
            return obj[attr]
        return default

    # Stratégie 1 : Essayer le matching exact par line_id
    gt_by_order = {}

    for line in lines:
        line_id = _get_attr(line, 'line_id', '')
        reading_order = _get_attr(line, 'reading_order', 0)

        if line_id in ground_truth:
            gt_by_order[reading_order] = ground_truth[line_id]
        else:
            # Essayer avec le préfixe du document
            alt_id = f"{document_id}_{line_id}"
            if alt_id in ground_truth:
                gt_by_order[reading_order] = ground_truth[alt_id]
            else:
                # Essayer sans préfixe
                if line_id.startswith(f"{document_id}_"):
                    short_id = line_id[len(document_id)+1:]
                    if short_id in ground_truth:
                        gt_by_order[reading_order] = ground_truth[short_id]

    # Stratégie 2 : Si peu de correspondances, aligner séquentiellement
    if len(gt_by_order) < len(lines) * 0.5 and len(ground_truth) > 0:
        gt_values = list(ground_truth.values())
        # Trier les lignes par reading_order
        sorted_lines = sorted(lines, key=lambda l: _get_attr(l, 'reading_order', 0))

        for i, line in enumerate(sorted_lines):
            ro = _get_attr(line, 'reading_order', i)
            if i < len(gt_values) and ro not in gt_by_order:
                gt_by_order[ro] = gt_values[i]

    return gt_by_order


# ============================================================
# 3. PIPELINE PRINCIPAL (VOLET 1 + VOLET 2)
# ============================================================

def run_pipeline(
    image_path: str,
    document_id: str,
    output_dir: str = "./pipeline_output",
    checkpoint_path: str = None,
    model_base_name: str = None,
    segmentation_method: str = "kraken",
    segmentation_model: str = None,
    ground_truth_path: Optional[str] = None,
    skip_preprocessing: bool = False,
    skip_segmentation: bool = False,
    skip_nlp: bool = False,
    mock_htr: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Exécute le pipeline complet de traitement d'un manuscrit (Volet 1 + Volet 2).

    NOUVEAUTÉS v0.4.0 :
      - Alignement GT par reading_order avec gestion des renumérotations
      - Intégration marginalia et discarded dans le Data Contract
      - Détection d'alerte si décalage segmentation vs GT
      - CER/WER comparatif ligne par ligne avec ground truth
      - Calibration des scores de confiance basée sur le CER réel
      - Export comparatif TXT (GT vs Prédit)
      - Flag needs_review basé sur le CER observé

    Args:
        image_path: Chemin vers l'image de la page de manuscrit
        document_id: Identifiant unique du document
        output_dir: Répertoire de sortie
        checkpoint_path: Chemin vers le modèle HTR (None = TRIDIS)
        model_base_name: Modèle de base (None = TRIDIS)
        segmentation_method: "kraken" ou "yolo"
        segmentation_model: Chemin vers un modèle de segmentation personnalisé
        ground_truth_path: Chemin vers le fichier de ground truth (optionnel)
        skip_preprocessing: Si True, saute le prétraitement
        skip_segmentation: Si True, saute la segmentation
        skip_nlp: Si True, saute la Phase 2 NLP
        mock_htr: Si True, simule le HTR
        verbose: Affiche les logs détaillés

    Returns:
        Dict avec les résultats complets (HTR + NLP + métriques comparatives)
    """
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)

    # Charger le ground truth si disponible
    ground_truth = load_ground_truth(ground_truth_path)
    has_ground_truth = ground_truth is not None

    # Valeurs par défaut TRIDIS
    if checkpoint_path is None:
        checkpoint_path = TRIDIS_MODEL if HTR_AVAILABLE else None
    if model_base_name is None:
        model_base_name = TRIDIS_MODEL if HTR_AVAILABLE else "microsoft/trocr-large-handwritten"

    if verbose:
        print(f"\n{'='*70}")
        print(f"   PIPELINE HTR + NLP — Document : {document_id}")
        print(f"   Volet 1 (Computer Vision) + Volet 2 (Analyse Linguistique)")
        if has_ground_truth:
            print(f"   Ground truth : ACTIVÉ ({len(ground_truth)} lignes)")
        else:
            print(f"   Ground truth : NON (mode inférence aveugle)")
        print(f"{'='*70}")

    # ============================================================
    # ÉTAPE 1 : PRÉTRAITEMENT
    # ============================================================
    if not skip_preprocessing:
        if verbose:
            print(f"\n[1/7] Prétraitement de l'image...")
            print(f"      → Deskewing, CLAHE, Binarisation Sauvola")

        processed_img = preprocess_pipeline(image_path)
        processed_path = os.path.join(output_dir, f"{document_id}_preprocessed.png")
        import cv2
        cv2.imwrite(processed_path, processed_img)

        if verbose:
            print(f"       Image prétraitée : {processed_path}")
    else:
        processed_path = image_path
        if verbose:
            print(f"\n[1/7] Prétraitement sauté")

    # ============================================================
    # ÉTAPE 2 : SEGMENTATION
    # ============================================================
    if not skip_segmentation:
        if verbose:
            print(f"\n[2/7] Segmentation de la page...")
            print(f"      → Méthode : {segmentation_method.upper()}")
            print(f"      → Filtrage marginalia, lettrines, artefacts")

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

        viz_path = os.path.join(seg_output_dir, f"{document_id}_visualization.png")
        visualize_segmentation(segmentation, output_path=viz_path)

        if verbose:
            print(f"       {len(segmentation.all_lines)} lignes principales détectées")
            print(f"       {len(segmentation.marginalia)} marginalia exclues")
            print(f"       {len(segmentation.discarded)} lignes rejetées (artefacts)")
            print(f"       PAGE XML : {os.path.join(seg_output_dir, f'{document_id}.page.xml')}")

        # ALERTE : Vérifier le nombre de lignes vs GT
        if has_ground_truth:
            gt_count = len(ground_truth)
            pred_count = len(segmentation.all_lines)
            diff_ratio = abs(gt_count - pred_count) / max(gt_count, 1)

            if diff_ratio > LINE_COUNT_MISMATCH_THRESHOLD:
                print(f"\n   ⚠️  ALERTE : Déséquilibre lignes détectées vs GT")
                print(f"       GT : {gt_count} lignes | Détecté : {pred_count} lignes")
                print(f"       Différence : {diff_ratio:.1%}")
                print(f"       → Vérifiez les marginalia et les lignes rejetées")
                print(f"       → Vérifiez que le GT correspond bien à cette page")
    else:
        seg_json = os.path.join(output_dir, "segmentation", f"{document_id}_segmentation.json")
        if os.path.exists(seg_json):
            with open(seg_json, "r") as f:
                seg_data = json.load(f)
            segmentation = None
        else:
            raise FileNotFoundError(f"Segmentation non trouvée : {seg_json}")

        if verbose:
            print(f"\n[2/7] Segmentation sautée")

    # ============================================================
    # ÉTAPE 3 : ALIGNEMENT GT AVEC LIGNES DÉTECTÉES
    # ============================================================
    if has_ground_truth and not skip_segmentation:
        gt_aligned = align_ground_truth_with_lines(
            ground_truth,
            segmentation.all_lines,
            document_id
        )
        if verbose:
            print(f"\n[2.5/7] Alignement GT avec lignes détectées...")
            print(f"       {len(gt_aligned)} lignes GT alignées par reading_order")
    else:
        gt_aligned = {}

    # ============================================================
    # ÉTAPE 4 : HTR / TRANSCRIPTION
    # ============================================================
    if verbose:
        print(f"\n[3/7] Transcription HTR (TRIDIS)...")

    lines_dir = os.path.join(output_dir, "segmentation", "lines")

    if mock_htr or not HTR_AVAILABLE:
        if verbose:
            print(f"      → Mode MOCK")
        transcriptions = _mock_transcription(segmentation, lines_dir)
    else:
        if verbose:
            print(f"      → Chargement de TRIDIS...")

        model, processor, config = load_htr_model(checkpoint_path, model_base_name)
        device = next(model.parameters()).device.type

        transcriptions = []
        lines = segmentation.get_lines_sorted()

        for i, line in enumerate(lines):
            line_img_path = os.path.join(lines_dir, f"{line.line_id}.png")

            if os.path.exists(line_img_path):
                pred_text = transcribe_image(model, processor, line_img_path, device)

                # Récupérer le GT aligné par reading_order
                cer, wer = 0.0, 0.0
                gt_text = ""
                if gt_aligned and line.reading_order in gt_aligned:
                    gt_text = gt_aligned[line.reading_order]
                    cer, wer = compute_cer_wer(gt_text, pred_text)

                    if verbose and (i + 1) % 5 == 0:
                        print(f"      [{i+1}/{len(lines)}] {line.line_id}")
                        print(f"         GT  : {gt_text[:60]}...")
                        print(f"         PRED: {pred_text[:60]}...")
                        print(f"         CER : {cer:.2%} | WER : {wer:.2%}")

                # Calibrer la confiance sur le CER réel
                if has_ground_truth and cer > 0:
                    confidence = calibrate_confidence(cer)
                else:
                    confidence = _estimate_confidence(pred_text, line.confidence)
            else:
                pred_text = "[IMAGE_MANQUANTE]"
                gt_text = gt_aligned.get(line.reading_order, "") if gt_aligned else ""
                cer, wer = 1.0, 1.0
                confidence = 0.10

            transcriptions.append({
                "line_id": line.line_id,
                "text": pred_text,
                "ground_truth": gt_text,
                "cer": cer,
                "wer": wer,
                "confidence": confidence,
                "polygon": line.polygon,
                "reading_order": line.reading_order
            })

            if verbose and not has_ground_truth and (i + 1) % 5 == 0:
                print(f"      [{i+1}/{len(lines)}] {line.line_id} : {pred_text[:50]}...")

    if verbose:
        print(f"       {len(transcriptions)} lignes transcrites")

    # ============================================================
    # ÉTAPE 4 : MÉTRIQUES GLOBALES ET CALIBRATION
    # ============================================================
    if verbose:
        print(f"\n[4/7] Calcul des métriques comparatives...")

    if has_ground_truth:
        valid_cers = [t["cer"] for t in transcriptions if t.get("ground_truth")]
        valid_wers = [t["wer"] for t in transcriptions if t.get("ground_truth")]

        global_cer = sum(valid_cers) / len(valid_cers) if valid_cers else 0.0
        global_wer = sum(valid_wers) / len(valid_wers) if valid_wers else 0.0

        # Statistiques par niveau de CER
        cer_buckets = {
            "excellent (<5%)": sum(1 for c in valid_cers if c < 0.05),
            "bon (5-15%)": sum(1 for c in valid_cers if 0.05 <= c < 0.15),
            "moyen (15-30%)": sum(1 for c in valid_cers if 0.15 <= c < 0.30),
            "mauvais (30-50%)": sum(1 for c in valid_cers if 0.30 <= c < 0.50),
            "catastrophique (>50%)": sum(1 for c in valid_cers if c >= 0.50),
        }

        if verbose:
            print(f"       CER global : {global_cer:.2%}")
            print(f"       WER global : {global_wer:.2%}")
            print(f"       Distribution CER :")
            for bucket, count in cer_buckets.items():
                pct = count / len(valid_cers) * 100 if valid_cers else 0
                print(f"         {bucket:25} : {count:3d} lignes ({pct:5.1f}%)")
    else:
        global_cer = None
        global_wer = None
        cer_buckets = {}

    # ============================================================
    # ÉTAPE 5 : AGRÉGATION (Data Contract Volet 1)
    # ============================================================
    if verbose:
        print(f"\n[5/7] Agrégation Data Contract (Volet 1)...")

    lines_entries = []
    full_text = ""
    full_ground_truth = ""

    for trans in transcriptions:
        entry = generate_line_entry(
            line_id=trans["line_id"],
            text=trans["text"],
            confidence=trans["confidence"],
            polygon=trans["polygon"],
            reading_order=trans["reading_order"],
            confidence_threshold=0.70 if has_ground_truth else 0.80
        )
        lines_entries.append(entry)
        full_text += trans["text"] + " "
        if trans.get("ground_truth"):
            full_ground_truth += trans["ground_truth"] + " "

    full_text = full_text.strip()
    full_ground_truth = full_ground_truth.strip()

    htr_json_path = os.path.join(output_dir, f"{document_id}_transcription.json")
    save_data_contract(htr_json_path, document_id, lines_entries)

    needs_review_count = sum(1 for e in lines_entries if e["needs_review"])
    review_rate = needs_review_count / len(lines_entries) if lines_entries else 0
    avg_confidence = sum(e["confidence"] for e in lines_entries) / len(lines_entries) if lines_entries else 0

    if verbose:
        print(f"       Data Contract HTR : {htr_json_path}")
        print(f"       Confiance moyenne : {avg_confidence:.3f}")
        print(f"       Lignes à réviser : {needs_review_count}/{len(lines_entries)} ({review_rate:.1%})")

    # ============================================================
    # ÉTAPE 6 : EXPORT COMPARATIF (GT vs PRED)
    # ============================================================
    if verbose:
        print(f"\n[6/7] Export comparatif GT vs Prédit...")

    # Export TXT côte à côte
    comparison_txt_path = os.path.join(output_dir, f"{document_id}_comparison.txt")
    with open(comparison_txt_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"COMPARAISON GROUND TRUTH vs PRÉDICTION\n")
        f.write(f"Document : {document_id}\n")
        f.write(f"{'='*80}\n\n")

        if has_ground_truth:
            f.write(f"CER global : {global_cer:.2%}\n")
            f.write(f"WER global : {global_wer:.2%}\n")
            f.write(f"Lignes transcrites : {len(transcriptions)}\n")
            f.write(f"Lignes GT alignées : {len([t for t in transcriptions if t.get('ground_truth')])}\n")
            f.write(f"Marginalia détectées : {len(segmentation.marginalia)}\n")
            f.write(f"Lignes rejetées : {len(segmentation.discarded)}\n")
            f.write(f"\n{'='*80}\n\n")

        for trans in transcriptions:
            f.write(f"--- {trans['line_id']} (ro={trans['reading_order']}) ---\n")
            if trans.get("ground_truth"):
                f.write(f"GT   : {trans['ground_truth']}\n")
                f.write(f"PRED : {trans['text']}\n")
                f.write(f"CER  : {trans['cer']:.2%} | WER : {trans['wer']:.2%} | CONF : {trans['confidence']:.2f}\n")
            else:
                f.write(f"PRED : {trans['text']}\n")
                f.write(f"CONF : {trans['confidence']:.2f} (sans GT)\n")
            f.write(f"\n")

        # Section marginalia
        if segmentation.marginalia:
            f.write(f"\n{'='*80}\n")
            f.write(f"MARGINALIA DÉTECTÉES (exclues du HTR)\n")
            f.write(f"{'='*80}\n\n")
            for marg in segmentation.marginalia:
                f.write(f"--- {marg.line_id} ---\n")
                f.write(f"BBOX : {marg.bbox}\n")
                f.write(f"CONF : {marg.confidence:.2f}\n\n")

        # Section discarded
        if segmentation.discarded:
            f.write(f"\n{'='*80}\n")
            f.write(f"LIGNES REJETÉES (artefacts, trop petites)\n")
            f.write(f"{'='*80}\n\n")
            for disc in segmentation.discarded:
                f.write(f"Raison : {disc.get('reason', 'unknown')} | BBOX : {disc.get('bbox', 'N/A')}\n")

    # Export JSON comparatif détaillé
    comparison_json_path = os.path.join(output_dir, f"{document_id}_comparison.json")
    comparison_data = {
        "document_id": document_id,
        "has_ground_truth": has_ground_truth,
        "segmentation_info": {
            "num_lines_detected": len(segmentation.all_lines) if segmentation else 0,
            "num_marginalia": len(segmentation.marginalia) if segmentation else 0,
            "num_discarded": len(segmentation.discarded) if segmentation else 0,
            "line_count_alert": has_ground_truth and segmentation and abs(len(ground_truth) - len(segmentation.all_lines)) / max(len(ground_truth), 1) > LINE_COUNT_MISMATCH_THRESHOLD if segmentation else False
        },
        "metrics": {
            "cer_global": global_cer,
            "wer_global": global_wer,
            "avg_confidence": avg_confidence,
            "needs_review_rate": review_rate,
            "cer_distribution": cer_buckets if has_ground_truth else {}
        },
        "lines": [
            {
                "line_id": t["line_id"],
                "reading_order": t["reading_order"],
                "ground_truth": t.get("ground_truth", ""),
                "prediction": t["text"],
                "cer": t.get("cer", 0.0),
                "wer": t.get("wer", 0.0),
                "confidence": t["confidence"],
                "needs_review": any(e["needs_review"] for e in lines_entries if e["line_id"] == t["line_id"]),
                "region_type": t.get("region_type", "main_text")
            }
            for t in transcriptions
        ],
        "marginalia": [
            {
                "line_id": m.line_id,
                "bbox": m.bbox,
                "confidence": m.confidence,
                "polygon": m.polygon
            }
            for m in (segmentation.marginalia if segmentation else [])
        ],
        "discarded": [
            {
                "reason": d.get("reason", "unknown"),
                "bbox": d.get("bbox", None),
                "height": d.get("height", None),
                "width": d.get("width", None),
                "kraken_type": d.get("kraken_type", None)
            }
            for d in (segmentation.discarded if segmentation else [])
        ]
    }
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"       Comparaison TXT  : {comparison_txt_path}")
        print(f"       Comparaison JSON : {comparison_json_path}")

    # ============================================================
    # ÉTAPE 7 : ANALYSE LINGUISTIQUE NLP (VOLET 2)
    # ============================================================
    nlp_results = None

    if not skip_nlp and NLP_AVAILABLE and full_text:
        if verbose:
            print(f"\n[7/7] Analyse Linguistique NLP (Volet 2)...")
            print(f"      → NER, Relations sémantiques, Thématisation")

        nlp_pipeline = MedievalNLPPipeline()
        nlp_analysis = nlp_pipeline.analyze_document(document_id, full_text)

        nlp_json_path = os.path.join(output_dir, f"{document_id}_nlp.json")
        export_nlp_to_json(nlp_analysis, nlp_json_path)

        nlp_results = {
            "language": nlp_analysis.language,
            "num_entities": len(nlp_analysis.entities),
            "num_relations": len(nlp_analysis.relations),
            "themes": nlp_analysis.themes,
            "confidence": nlp_analysis.confidence,
            "entities": [
                {"text": e.text, "type": e.label, "confidence": e.confidence}
                for e in nlp_analysis.entities[:10]
            ],
            "relations": [
                {"subject": r.subject, "predicate": r.predicate, "object": r.object}
                for r in nlp_analysis.relations[:10]
            ]
        }

        if verbose:
            print(f"       Résultat NLP : {nlp_json_path}")
            print(f"       Langue : {nlp_analysis.language}")
            print(f"       Entités : {len(nlp_analysis.entities)}")
            print(f"       Relations : {len(nlp_analysis.relations)}")
            print(f"       Thèmes : {nlp_analysis.themes}")
    elif skip_nlp:
        if verbose:
            print(f"\n[7/7] Phase 2 NLP sautée (--skip-nlp)")
    else:
        if verbose:
            print(f"\n[7/7] Phase 2 NLP indisponible (module nlp_analysis.py manquant)")

    # ============================================================
    # ÉTAPE 8 : EXPORT FINAL UNIFIÉ (HTR + NLP + MÉTRIQUES)
    # ============================================================
    if verbose:
        print(f"\n[8/8] Export final unifié...")

    final_document = {
        "document_id": document_id,
        "image_path": image_path,
        "output_dir": output_dir,
        "pipeline_version": "2.2.0",
        "pipeline_stages": {
            "preprocessing": {
                "skipped": skip_preprocessing,
                "preprocessed_image": processed_path if not skip_preprocessing else None
            },
            "segmentation": {
                "skipped": skip_segmentation,
                "method": segmentation_method,
                "num_lines_detected": len(segmentation.all_lines) if segmentation else 0,
                "num_marginalia": len(segmentation.marginalia) if segmentation else 0,
                "num_discarded": len(segmentation.discarded) if segmentation else 0,
                "line_count_alert": has_ground_truth and segmentation and abs(len(ground_truth) - len(segmentation.all_lines)) / max(len(ground_truth), 1) > LINE_COUNT_MISMATCH_THRESHOLD,
                "segmentation_json": os.path.join(output_dir, "segmentation", f"{document_id}_segmentation.json"),
                "segmentation_pagexml": os.path.join(output_dir, "segmentation", f"{document_id}.page.xml"),
                "visualization": os.path.join(output_dir, "segmentation", f"{document_id}_visualization.png"),
                "lines_directory": lines_dir
            },
            "htr": {
                "model": "TRIDIS (magistermilitum/tridis_HTR)",
                "num_lines_transcribed": len(transcriptions),
                "avg_confidence": avg_confidence,
                "needs_review_rate": review_rate,
                "needs_review_count": needs_review_count,
                "transcription_json": htr_json_path,
                "has_ground_truth": has_ground_truth,
                "cer_global": global_cer,
                "wer_global": global_wer,
                "cer_distribution": cer_buckets,
                "comparison_txt": comparison_txt_path,
                "comparison_json": comparison_json_path
            },
            "nlp": {
                "skipped": skip_nlp or not NLP_AVAILABLE,
                "results": nlp_results
            }
        },
        "full_text": full_text,
        "ground_truth_text": full_ground_truth if has_ground_truth else None,
        "lines": lines_entries,
        "marginalia": [
            {
                "line_id": m.line_id,
                "bbox": m.bbox,
                "confidence": m.confidence,
                "polygon": m.polygon
            }
            for m in (segmentation.marginalia if segmentation else [])
        ],
        "discarded_summary": [
            {
                "reason": d.get("reason", "unknown"),
                "bbox": d.get("bbox", None)
            }
            for d in (segmentation.discarded if segmentation else [])
        ],
        "metadata": {
            "processed_at": __import__("datetime").datetime.now().isoformat(),
            "htr_available": HTR_AVAILABLE,
            "nlp_available": NLP_AVAILABLE,
            "jiwer_available": JIWER_AVAILABLE,
            "mock_mode": mock_htr
        }
    }

    final_json_path = os.path.join(output_dir, f"{document_id}_final.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_document, f, ensure_ascii=False, indent=2)

    report_path = os.path.join(output_dir, f"{document_id}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "document_id": document_id,
            "num_lines": len(transcriptions),
            "num_marginalia": len(segmentation.marginalia) if segmentation else 0,
            "num_discarded": len(segmentation.discarded) if segmentation else 0,
            "line_count_alert": has_ground_truth and segmentation and abs(len(ground_truth) - len(segmentation.all_lines)) / max(len(ground_truth), 1) > LINE_COUNT_MISMATCH_THRESHOLD,
            "avg_confidence": avg_confidence,
            "needs_review_rate": review_rate,
            "has_ground_truth": has_ground_truth,
            "cer_global": global_cer,
            "wer_global": global_wer,
            "nlp_available": NLP_AVAILABLE and not skip_nlp,
            "nlp_results": nlp_results,
            "files": {
                "htr_json": htr_json_path,
                "comparison_txt": comparison_txt_path,
                "comparison_json": comparison_json_path,
                "nlp_json": os.path.join(output_dir, f"{document_id}_nlp.json") if nlp_results else None,
                "final_json": final_json_path
            }
        }, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"       Document final : {final_json_path}")
        print(f"       Rapport : {report_path}")
        print(f"\n{'='*70}")
        print(f"  🎉 PIPELINE TERMINÉ AVEC SUCCÈS")
        print(f"     Volet 1 (HTR) : {len(transcriptions)} lignes transcrites")
        if has_ground_truth:
            print(f"     CER global : {global_cer:.2%} | WER global : {global_wer:.2%}")
        if segmentation:
            print(f"     Marginalia : {len(segmentation.marginalia)} | Rejetées : {len(segmentation.discarded)}")
        if nlp_results:
            print(f"     Volet 2 (NLP) : {nlp_results['num_entities']} entités, {nlp_results['num_relations']} relations")
        print(f"{'='*70}")

    return final_document


# ============================================================
# 4. FONCTIONS UTILITAIRES
# ============================================================

def _mock_transcription(segmentation, lines_dir: str) -> List[Dict]:
    """Simule la transcription pour les tests sans modèle HTR."""
    transcriptions = []
    lines = segmentation.get_lines_sorted()

    mock_texts = [
        "In nomine Domini amen",
        "Anno domini millesimo",
        "Ego frater Johannes",
        "testimonium perhibeo",
        "sub sigillo capituli",
        "donavit terram suam",
        "in villa de Parisius",
        "pro precio centum solidorum",
        "Testes frater Martinus et Robertus",
        "Sigillum capituli appositum est"
    ]
    import random

    for i, line in enumerate(lines):
        text = mock_texts[i % len(mock_texts)]
        confidence = random.uniform(0.70, 0.95)

        transcriptions.append({
            "line_id": line.line_id,
            "reading_order": line.reading_order,
            "text": text,
            "ground_truth": "",
            "cer": 0.0,
            "wer": 0.0,
            "confidence": confidence,
            "polygon": line.polygon,
            "region_type": line.region_type
        })

    return transcriptions


def _estimate_confidence(pred_text: str, seg_confidence: float) -> float:
    """Estime la confiance de la transcription (fallback sans GT)."""
    base_conf = seg_confidence
    text_len = len(pred_text.strip())

    if text_len <= 2:
        base_conf *= 0.5
    elif text_len > 100:
        base_conf *= 0.9

    if len(set(pred_text)) < 3 and text_len > 5:
        base_conf *= 0.3

    return min(base_conf, 1.0)


# ============================================================
# 5. INTERFACE EN LIGNE DE COMMANDE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline HTR + NLP complet pour manuscrits médiévaux (MD5-2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
═══════════════════════════════════════════════════════════════════════════════
  PIPELINE END-TO-END : Page → Segmentation → HTR → NLP

  NOUVEAUTÉS v2.2 :
    → Alignement GT par reading_order (gère renumérotation segmentation)
    → Détection d'alerte si décalage segmentation vs GT
    → Intégration marginalia et discarded dans le Data Contract
    → CER/WER comparatif avec --ground_truth
    → Calibration des confiances sur le CER réel
    → Export comparatif TXT (GT vs Prédit)
═══════════════════════════════════════════════════════════════════════════════

Exemples d'utilisation :

  # Pipeline complet avec ground truth (recommandé)
  python main.py --image ./data/raw/page_test_001.png --id ms_001 \
      --ground_truth ./data/ground_truth/ms_001_gt.txt

  # Sans la Phase 2 NLP (HTR seul)
  python main.py --image ./data/raw/page_test_001.png --id ms_001 --skip-nlp

  # Mode test (sans HTR ni NLP, simulation)
  python main.py --image ./data/raw/page_test_001.png --id ms_001 --mock

  # Avec segmentation YOLO
  python main.py --image ./data/raw/page_test_001.png --id ms_001 --seg-method yolo

  # Sauter le prétraitement
  python main.py --image ./data/raw/page_test_001.png --id ms_001 --skip-preprocessing

Fichiers de sortie :
  ./pipeline_output/
    ├── {id}_preprocessed.png        # Image prétraitée
    ├── segmentation/
    │   ├── {id}_segmentation.json   # Résultat segmentation
    │   ├── {id}.page.xml            # PAGE XML standard
    │   ├── {id}_visualization.png   # Visualisation
    │   └── lines/                   # Images de lignes extraites
    ├── {id}_transcription.json      # Data Contract HTR (Volet 1)
    ├── {id}_comparison.txt          # Comparaison GT vs Prédit (NOUVEAU)
    ├── {id}_comparison.json         # Métriques comparatives (NOUVEAU)
    ├── {id}_nlp.json                # Analyse NLP (Volet 2)
    ├── {id}_final.json              # Document unifié HTR+NLP
    └── {id}_report.json             # Rapport de traitement
        """
    )

    parser.add_argument("--image", "-i", type=str, required=True,
                       help="Chemin vers l'image de la page de manuscrit")
    parser.add_argument("--id", type=str, required=True,
                       help="Identifiant unique du document (ex: ms_001)")
    parser.add_argument("--output", "-o", type=str, default="./pipeline_output",
                       help="Répertoire de sortie (défaut: ./pipeline_output)")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                       help="Checkpoint modèle HTR (défaut: TRIDIS)")
    parser.add_argument("--model-base", type=str, default=None,
                       help="Modèle de base (défaut: TRIDIS)")
    parser.add_argument("--seg-method", type=str, default="kraken",
                       choices=["kraken", "yolo"],
                       help="Méthode de segmentation")
    parser.add_argument("--seg-model", type=str, default=None,
                       help="Modèle de segmentation personnalisé")
    parser.add_argument("--ground-truth", "-g", type=str, default=None,
                       help="Chemin vers le fichier de ground truth (optionnel)")
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Sauter le prétraitement")
    parser.add_argument("--skip-segmentation", action="store_true",
                       help="Sauter la segmentation")
    parser.add_argument("--skip-nlp", action="store_true",
                       help="Sauter la Phase 2 NLP")
    parser.add_argument("--mock", action="store_true",
                       help="Mode simulation (HTR factice)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Mode silencieux")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f" ERREUR : Image introuvable : {args.image}")
        sys.exit(1)

    try:
        result = run_pipeline(
            image_path=args.image,
            document_id=args.id,
            output_dir=args.output,
            checkpoint_path=args.checkpoint,
            model_base_name=args.model_base,
            segmentation_method=args.seg_method,
            segmentation_model=args.seg_model,
            ground_truth_path=args.ground_truth,
            skip_preprocessing=args.skip_preprocessing,
            skip_segmentation=args.skip_segmentation,
            skip_nlp=args.skip_nlp,
            mock_htr=args.mock,
            verbose=not args.quiet
        )

        if args.quiet:
            htr_info = result.get("pipeline_stages", {}).get("htr", {})
            nlp_info = result.get("pipeline_stages", {}).get("nlp", {})
            nlp_str = ""
            if not nlp_info.get("skipped") and nlp_info.get("results"):
                nr = nlp_info["results"]
                nlp_str = f"|{nr['num_entities']}|{nr['num_relations']}|{','.join(nr['themes'])}"

            gt_str = ""
            if htr_info.get("has_ground_truth"):
                gt_str = f"|CER={htr_info.get('cer_global', 0):.3f}|WER={htr_info.get('wer_global', 0):.3f}"

            print(f"{args.id}|{htr_info['num_lines_transcribed']}|"
                  f"{htr_info['avg_confidence']:.3f}|"
                  f"{htr_info['needs_review_rate']:.2%}{gt_str}{nlp_str}")

    except Exception as e:
        print(f"\n ERREUR dans le pipeline : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
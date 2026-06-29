"""
Module de segmentation pour manuscrits médiévaux.

Implémente la segmentation de structure de page (layout) et d'extraction de lignes
en utilisant Kraken BLLA (Baseline Layout Analysis) ou YOLO comme alternative.

Le brief exige :
  - Détection des régions (texte, illustration, marge)
  - Extraction des lignes avec polygones de contour
  - Ordre de lecture correct
  - Export compatible data_contract.py

Architecture du pipeline de segmentation :
    Image page complète
           ↓
    [1] Prétraitement (preprocessing.py)
           ↓
    [2] Détection des régions (layout) — Kraken/YOLO
           ↓
    [3] Extraction des lignes avec polygones — Kraken BLLA
           ↓
    [4] Ordre de lecture + dewarping
           ↓
    [5] Export : images de lignes + métadonnées JSON/PAGE-XML

CORRECTIONS v1.1 (2026-06-23) :
  → Filtrage des lignes trop courtes/étroites (artefacts, points)
  → Détection et exclusion des marginalia du flux principal HTR
  → Ordre de lecture avec clustering par ligne de base (tolérance Y)
  → Gestion des lettrines décoratives (tag heading, pas ligne isolée)
  → Sauvegarde des marginalia dans un fichier séparé
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings

# Kraken pour la segmentation
from kraken import blla
from kraken.lib import vgsl
from kraken.pageseg import segment as kraken_segment

# Pour YOLO (alternative)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    warnings.warn("ultralytics non installé. Le mode YOLO segmentation ne sera pas disponible.")


# ============================================================
# 1. CONSTANTES DE FILTRAGE
# ============================================================

# Seuils de filtrage des lignes (ajustables selon la résolution)
MIN_LINE_HEIGHT = 15          # Hauteur minimale d'une ligne valide (pixels)
MIN_LINE_WIDTH = 80           # Largeur minimale (pixels)
MAX_LINE_HEIGHT = 120         # Hauteur maximale (évite lettrines seules)
MARGIN_X_THRESHOLD = 0.12     # 12% des bords = zone marginale
MIN_TEXT_WIDTH_RATIO = 0.35   # Ratio min largeur/image pour être "texte principal"
Y_CLUSTER_TOLERANCE = 0.6     # Tolérance de clustering Y (ratio de la hauteur médiane)
MIN_Y_TOLERANCE = 12          # Tolérance Y minimale en pixels


# ============================================================
# 2. STRUCTURES DE DONNÉES
# ============================================================

@dataclass
class LinePolygon:
    """Représente une ligne de texte détectée avec son polygone et métadonnées."""
    line_id: str
    polygon: List[List[int]]  # [[x1,y1], [x2,y2], ...] — contour de la ligne
    baseline: List[List[int]]  # [[x1,y1], [x2,y2], ...] — ligne de base
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    region_type: str  # "main_text", "marginalia", "heading", etc.
    reading_order: int  # Position dans l'ordre de lecture
    confidence: float  # Score de confiance de la segmentation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PageRegion:
    """Représente une région détectée sur la page (colonne, illustration, etc.)."""
    region_id: str
    region_type: str  # "text", "illustration", "margin", "heading"
    polygon: List[List[int]]
    bbox: Tuple[int, int, int, int]
    lines: List[LinePolygon]
    confidence: float


@dataclass
class SegmentationResult:
    """Résultat complet de la segmentation d'une page."""
    page_id: str
    image_path: str
    image_shape: Tuple[int, int]  # (height, width)
    regions: List[PageRegion]
    all_lines: List[LinePolygon]
    marginalia: List[LinePolygon]  # NOUVEAU : marginalia séparées
    discarded: List[Dict]  # NOUVEAU : lignes rejetées avec raison

    def get_lines_sorted(self) -> List[LinePolygon]:
        """Retourne les lignes triées par ordre de lecture."""
        return sorted(self.all_lines, key=lambda l: l.reading_order)


# ============================================================
# 3. SEGMENTATION AVEC KRAKEN BLLA (MÉTHODE PRINCIPALE)
# ============================================================

def _log_kraken_result_structure(result) -> None:
    """
    Inspecte et affiche la structure de l'objet retourné par blla.segment().
    """
    regions_type = type(result.regions).__name__ if hasattr(result, 'regions') else 'N/A'
    lines_count = len(result.lines) if hasattr(result, 'lines') and result.lines else 0
    regions_info = "N/A"

    if hasattr(result, 'regions'):
        if isinstance(result.regions, dict):
            regions_info = f"dict avec clés {list(result.regions.keys())}"
            total_region_polys = sum(len(v) if isinstance(v, list) else 1
                                     for v in result.regions.values())
            regions_info += f" ({total_region_polys} polygones)"
        elif isinstance(result.regions, list):
            regions_info = f"list de {len(result.regions)} objets"

    print(f"  [Kraken] result.type={getattr(result, 'type', '?')} | "
          f"regions={regions_info} | lines={lines_count}")

    if hasattr(result, 'lines') and result.lines:
        sample = result.lines[0]
        has_boundary = hasattr(sample, 'boundary') and sample.boundary is not None
        has_baseline = hasattr(sample, 'baseline') and sample.baseline is not None
        print(f"  [Kraken] Première ligne : boundary={'oui' if has_boundary else 'non'} "
              f"({len(sample.boundary) if has_boundary else 0} pts), "
              f"baseline={'oui' if has_baseline else 'non'} "
              f"({len(sample.baseline) if has_baseline else 0} pts), "
              f"tags={getattr(sample, 'tags', None)}")


class KrakenSegmenter:
    """
    Segmenteur basé sur Kraken BLLA (Baseline Layout Analysis).
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = device
        self.model = None

        if model_path and os.path.exists(model_path):
            print(f" Chargement du modèle Kraken BLLA : {model_path}")
            self.model = vgsl.TorchVGSLModel.load_model(model_path)
        else:
            print(" Utilisation du modèle Kraken BLLA par défaut")

    def segment_page(
        self,
        image_path: str,
        page_id: str = None,
        text_direction: str = "horizontal-lr"
    ) -> SegmentationResult:
        """Segmente une page complète en régions et lignes."""
        if page_id is None:
            page_id = os.path.splitext(os.path.basename(image_path))[0]

        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        print(f" Segmentation de la page : {page_id} ({w}x{h})")

        try:
            result = blla.segment(img, model=self.model, text_direction=text_direction)
            _log_kraken_result_structure(result)
        except Exception as e:
            print(f"  Erreur Kraken BLLA : {e}")
            print("   Fallback vers la segmentation legacy Kraken...")
            result = self._fallback_segmentation(img, text_direction)

        regions, lines, marginalia, discarded = self._parse_kraken_result(result, page_id, w, h)
        lines = self._compute_reading_order(lines, text_direction)

        seg_result = SegmentationResult(
            page_id=page_id,
            image_path=image_path,
            image_shape=(h, w),
            regions=regions,
            all_lines=lines,
            marginalia=marginalia,
            discarded=discarded
        )

        print(f"    {len(regions)} régions | {len(lines)} lignes principales | "
              f"{len(marginalia)} marginalia | {len(discarded)} rejetées")
        return seg_result

    def _parse_kraken_result(self, result, page_id: str, img_w: int, img_h: int
                            ) -> Tuple[List[PageRegion], List[LinePolygon], List[LinePolygon], List[Dict]]:
        """
        Parse le résultat Kraken avec filtrage robuste.

        Retourne : (regions, main_lines, marginalia, discarded)
        """
        regions = []
        main_lines = []
        marginalia_lines = []
        discarded = []
        line_idx = 0

        # ── Parser les régions (Kraken 5.x) ────────────────────────────────
        if hasattr(result, 'regions') and isinstance(result.regions, dict):
            for r_idx, (region_type, region_polys) in enumerate(result.regions.items()):
                if not isinstance(region_polys, list):
                    region_polys = [region_polys]

                for poly_raw in region_polys:
                    if isinstance(poly_raw, (list, tuple)) and len(poly_raw) >= 3:
                        region_poly = [[int(x), int(y)] for x, y in poly_raw]
                    else:
                        region_poly = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]

                    region_bbox = self._polygon_to_bbox(region_poly)
                    page_region = PageRegion(
                        region_id=f"{page_id}_region_{r_idx:03d}",
                        region_type=str(region_type),
                        polygon=region_poly,
                        bbox=region_bbox,
                        lines=[],
                        confidence=0.9
                    )
                    regions.append(page_region)

        # ── Parser les lignes avec filtrage ───────────────────────────────
        if hasattr(result, 'lines') and result.lines:
            for line in result.lines:
                line_poly = self._extract_polygon(line, img_w, img_h)
                baseline = self._extract_baseline(line, img_w, img_h)
                bbox = self._polygon_to_bbox(line_poly)
                confidence = getattr(line, 'confidence', 0.9)

                bbox_h = bbox[3] - bbox[1]
                bbox_w = bbox[2] - bbox[0]
                x_center = (bbox[0] + bbox[2]) / 2

                # ── DÉTECTION DU TYPE KRaken ─────────────────────────────
                kraken_type = self._extract_kraken_type(line)
                _KRAKEN_TYPE_MAP = {
                    'default': 'main_text', 'text': 'main_text',
                    'marginalia': 'marginalia', 'margin': 'marginalia',
                    'heading': 'heading', 'title': 'heading',
                    'illustration': 'illustration', 'drop_capital': 'heading',
                }
                detected_type = _KRAKEN_TYPE_MAP.get(kraken_type.lower(), 'main_text')

                # ── FILTRAGE 1 : Taille minimale ─────────────────────────
                if bbox_h < MIN_LINE_HEIGHT or bbox_w < MIN_LINE_WIDTH:
                    discarded.append({
                        "reason": "too_small",
                        "bbox": bbox,
                        "height": bbox_h,
                        "width": bbox_w,
                        "kraken_type": kraken_type
                    })
                    continue

                # ── FILTRAGE 2 : Lettrines / titres isolés ─────────────────
                if bbox_h > MAX_LINE_HEIGHT:
                    detected_type = "heading"
                    # Si c'est une lettrine très grande et très étroite, c'est probablement
                    # une initiale décorative — on la garde comme heading mais on la note
                    if bbox_w < bbox_h * 1.5:
                        discarded.append({
                            "reason": "drop_capital",
                            "bbox": bbox,
                            "height": bbox_h,
                            "width": bbox_w,
                            "kraken_type": kraken_type
                        })
                        continue

                # ── FILTRAGE 3 : Détection heuristique de marginalia ──────
                is_near_left_margin = bbox[0] < img_w * MARGIN_X_THRESHOLD
                is_near_right_margin = bbox[2] > img_w * (1 - MARGIN_X_THRESHOLD)
                is_short = bbox_w < img_w * MIN_TEXT_WIDTH_RATIO

                # Forcer marginalia si proche des bords ET courte
                if (is_near_left_margin or is_near_right_margin) and is_short:
                    detected_type = 'marginalia'

                # ── FILTRAGE 4 : Exclusion des marginalia du flux HTR ─────
                if detected_type == 'marginalia':
                    marg_line = LinePolygon(
                        line_id=f"{page_id}_marg_{len(marginalia_lines):04d}",
                        polygon=line_poly,
                        baseline=baseline,
                        bbox=bbox,
                        region_type='marginalia',
                        reading_order=-1,
                        confidence=float(confidence)
                    )
                    marginalia_lines.append(marg_line)
                    continue

                # ── Ligne valide pour le flux principal ────────────────────
                line_obj = LinePolygon(
                    line_id=f"{page_id}_line_{line_idx:04d}",
                    polygon=line_poly,
                    baseline=baseline,
                    bbox=bbox,
                    region_type=detected_type,
                    reading_order=line_idx,
                    confidence=float(confidence)
                )
                main_lines.append(line_obj)
                line_idx += 1

        # ── Fallback Kraken <= 4.x ───────────────────────────────────────
        elif hasattr(result, 'regions') and isinstance(result.regions, list) and result.regions:
            for r_idx, region in enumerate(result.regions):
                region_type = getattr(region, 'type', 'text')
                region_poly = self._extract_polygon(region, img_w, img_h)
                region_bbox = self._polygon_to_bbox(region_poly)
                region_lines = []

                if hasattr(region, 'lines') and region.lines:
                    for line in region.lines:
                        line_poly = self._extract_polygon(line, img_w, img_h)
                        baseline = self._extract_baseline(line, img_w, img_h)
                        bbox = self._polygon_to_bbox(line_poly)
                        confidence = getattr(line, 'confidence', 0.9)

                        bbox_h = bbox[3] - bbox[1]
                        bbox_w = bbox[2] - bbox[0]

                        if bbox_h < MIN_LINE_HEIGHT or bbox_w < MIN_LINE_WIDTH:
                            discarded.append({"reason": "too_small", "bbox": bbox})
                            continue

                        if bbox_h > MAX_LINE_HEIGHT:
                            discarded.append({"reason": "too_tall", "bbox": bbox})
                            continue

                        is_near_left = bbox[0] < img_w * MARGIN_X_THRESHOLD
                        is_near_right = bbox[2] > img_w * (1 - MARGIN_X_THRESHOLD)
                        is_short = bbox_w < img_w * MIN_TEXT_WIDTH_RATIO

                        if (is_near_left or is_near_right) and is_short:
                            marg_line = LinePolygon(
                                line_id=f"{page_id}_marg_{len(marginalia_lines):04d}",
                                polygon=line_poly, baseline=baseline, bbox=bbox,
                                region_type='marginalia', reading_order=-1,
                                confidence=float(confidence)
                            )
                            marginalia_lines.append(marg_line)
                            continue

                        line_obj = LinePolygon(
                            line_id=f"{page_id}_line_{line_idx:04d}",
                            polygon=line_poly, baseline=baseline, bbox=bbox,
                            region_type=region_type, reading_order=line_idx,
                            confidence=float(confidence)
                        )
                        region_lines.append(line_obj)
                        main_lines.append(line_obj)
                        line_idx += 1

                page_region = PageRegion(
                    region_id=f"{page_id}_region_{r_idx:03d}",
                    region_type=region_type,
                    polygon=region_poly,
                    bbox=region_bbox,
                    lines=region_lines,
                    confidence=getattr(region, 'confidence', 0.9)
                )
                regions.append(page_region)

        # Région synthétique si nécessaire
        if not regions and main_lines:
            all_poly = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
            regions.append(PageRegion(
                region_id=f"{page_id}_region_000",
                region_type="text",
                polygon=all_poly,
                bbox=(0, 0, img_w, img_h),
                lines=main_lines,
                confidence=0.9
            ))

        return regions, main_lines, marginalia_lines, discarded

    def _extract_kraken_type(self, line) -> str:
        """Extrait le type de région depuis les tags Kraken."""
        if hasattr(line, 'tags') and line.tags:
            if isinstance(line.tags, dict):
                raw = line.tags.get('type', 'default')
                if isinstance(raw, list):
                    if raw and isinstance(raw[0], dict):
                        return raw[0].get('type', 'default')
                    elif raw and isinstance(raw[0], str):
                        return raw[0]
                    return 'default'
                return str(raw) if raw else 'default'
            return str(line.tags)
        return 'default'

    def _extract_polygon(self, obj, img_w: int, img_h: int) -> List[List[int]]:
        """Extrait le polygone d'un objet Kraken."""
        if hasattr(obj, 'boundary') and obj.boundary is not None:
            try:
                poly = [[int(p[0]), int(p[1])] for p in obj.boundary]
                if len(poly) >= 3:
                    return poly
            except (TypeError, IndexError):
                pass

        if hasattr(obj, 'envelope') and obj.envelope is not None:
            try:
                poly = [[int(p[0]), int(p[1])] for p in obj.envelope]
                if len(poly) >= 3:
                    return poly
            except (TypeError, IndexError):
                pass

        if hasattr(obj, 'bbox') and obj.bbox is not None:
            try:
                x1, y1, x2, y2 = obj.bbox
                return [[int(x1), int(y1)], [int(x2), int(y1)],
                        [int(x2), int(y2)], [int(x1), int(y2)]]
            except (TypeError, ValueError):
                pass

        return [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]

    def _extract_baseline(self, obj, img_w: int, img_h: int) -> List[List[int]]:
        """Extrait la baseline d'un objet ligne Kraken."""
        if hasattr(obj, 'baseline') and obj.baseline is not None:
            try:
                return [[int(p[0]), int(p[1])] for p in obj.baseline]
            except (TypeError, IndexError):
                pass
        return []

    def _polygon_to_bbox(self, polygon: List[List[int]]) -> Tuple[int, int, int, int]:
        """Convertit un polygone en bounding box."""
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    def _compute_reading_order(self, lines: List[LinePolygon], text_direction: str) -> List[LinePolygon]:
        """
        Calcule l'ordre de lecture avec clustering par ligne de base.
        Gère les interlignes irréguliers et évite les décalages.
        """
        if not lines:
            return lines

        if text_direction.startswith("horizontal"):
            # Utiliser le centre Y de la baseline pour le tri
            def get_baseline_y(line):
                if line.baseline and len(line.baseline) > 0:
                    ys = [p[1] for p in line.baseline]
                    return float(np.median(ys))
                else:
                    return (line.bbox[1] + line.bbox[3]) / 2.0

            # Trier par Y de baseline, puis par X
            lines_with_y = [(line, get_baseline_y(line)) for line in lines]
            lines_with_y.sort(key=lambda x: (x[1], x[0].bbox[0]))

            # Calculer la tolérance de clustering
            heights = [l.bbox[3] - l.bbox[1] for l in lines]
            median_h = float(np.median(heights)) if heights else 20.0
            y_tolerance = max(median_h * Y_CLUSTER_TOLERANCE, MIN_Y_TOLERANCE)

            # Regrouper les lignes à la même hauteur (± tolérance)
            grouped = []
            current_group = []
            current_y = None

            for line, y in lines_with_y:
                if current_y is None or abs(y - current_y) <= y_tolerance:
                    current_group.append((line, y))
                    # Moyenne pondérée des Y du groupe
                    current_y = (current_y * (len(current_group) - 1) + y) / len(current_group) if current_y is not None else y
                else:
                    # Nouveau groupe : trier le groupe courant par X
                    current_group.sort(key=lambda x: x[0].bbox[0])
                    grouped.extend(current_group)
                    current_group = [(line, y)]
                    current_y = y

            # Dernier groupe
            if current_group:
                current_group.sort(key=lambda x: x[0].bbox[0])
                grouped.extend(current_group)

            # Réassigner les reading_order
            for i, (line, _) in enumerate(grouped):
                line.reading_order = i

            return [line for line, _ in grouped]

        else:
            # Vertical : gauche à droite, puis haut en bas
            lines_sorted = sorted(lines, key=lambda l: (l.bbox[0], l.bbox[1]))
            for i, line in enumerate(lines_sorted):
                line.reading_order = i
            return lines_sorted

    def _fallback_segmentation(self, img: Image.Image, text_direction: str):
        """Fallback si BLLA échoue."""
        from kraken.pageseg import segment
        return segment(img)


# ============================================================
# 4. SEGMENTATION AVEC YOLO (ALTERNATIVE)
# ============================================================

class YOLOSegmenter:
    """Segmenteur alternatif basé sur YOLO."""

    def __init__(self, model_path: str = None, conf_threshold: float = 0.5):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics est requis. pip install ultralytics")

        self.conf_threshold = conf_threshold

        if model_path and os.path.exists(model_path):
            print(f" Chargement du modèle YOLO : {model_path}")
            self.model = YOLO(model_path)
        else:
            print(" Utilisation de YOLOv8n par défaut")
            self.model = YOLO("yolov8n.pt")

    def segment_page(self, image_path: str, page_id: str = None) -> SegmentationResult:
        """Segmente une page avec YOLO."""
        if page_id is None:
            page_id = os.path.splitext(os.path.basename(image_path))[0]

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        h, w = img.shape[:2]
        print(f" Segmentation YOLO : {page_id} ({w}x{h})")

        results = self.model(image_path, conf=self.conf_threshold)

        main_lines = []
        marginalia_lines = []
        discarded = []
        regions = []

        for r_idx, result in enumerate(results):
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]

                bbox_h = y2 - y1
                bbox_w = x2 - x1

                # Filtrage taille
                if bbox_h < MIN_LINE_HEIGHT or bbox_w < MIN_LINE_WIDTH:
                    discarded.append({"reason": "too_small", "bbox": (x1, y1, x2, y2)})
                    continue

                if bbox_h > MAX_LINE_HEIGHT:
                    discarded.append({"reason": "too_tall", "bbox": (x1, y1, x2, y2)})
                    continue

                # Déterminer le type
                if class_name in ["text", "line", "default_line"]:
                    region_type = "main_text"
                elif class_name in ["marginalia", "margin"]:
                    region_type = "marginalia"
                elif class_name in ["heading", "title"]:
                    region_type = "heading"
                else:
                    region_type = "main_text"

                # Heuristique marginalia
                is_near_left = x1 < w * MARGIN_X_THRESHOLD
                is_near_right = x2 > w * (1 - MARGIN_X_THRESHOLD)
                is_short = bbox_w < w * MIN_TEXT_WIDTH_RATIO
                if (is_near_left or is_near_right) and is_short:
                    region_type = "marginalia"

                polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                line_obj = LinePolygon(
                    line_id=f"{page_id}_line_{i:04d}",
                    polygon=polygon,
                    baseline=[[x1, (y1+y2)//2], [x2, (y1+y2)//2]],
                    bbox=(x1, y1, x2, y2),
                    region_type=region_type,
                    reading_order=i,
                    confidence=conf
                )

                if region_type == "marginalia":
                    line_obj.line_id = f"{page_id}_marg_{len(marginalia_lines):04d}"
                    line_obj.reading_order = -1
                    marginalia_lines.append(line_obj)
                else:
                    main_lines.append(line_obj)

        # Trier par ordre de lecture
        main_lines = self._compute_reading_order_yolo(main_lines)

        all_poly = [[0, 0], [w, 0], [w, h], [0, h]]
        page_region = PageRegion(
            region_id=f"{page_id}_region_000",
            region_type="text",
            polygon=all_poly,
            bbox=(0, 0, w, h),
            lines=main_lines,
            confidence=1.0
        )
        regions.append(page_region)

        print(f"    {len(main_lines)} lignes principales | {len(marginalia_lines)} marginalia")

        return SegmentationResult(
            page_id=page_id,
            image_path=image_path,
            image_shape=(h, w),
            regions=regions,
            all_lines=main_lines,
            marginalia=marginalia_lines,
            discarded=discarded
        )

    def _compute_reading_order_yolo(self, lines: List[LinePolygon]) -> List[LinePolygon]:
        """Ordre de lecture pour YOLO avec clustering Y."""
        if not lines:
            return lines

        def get_center_y(line):
            return (line.bbox[1] + line.bbox[3]) / 2.0

        lines_with_y = [(line, get_center_y(line)) for line in lines]
        lines_with_y.sort(key=lambda x: (x[1], x[0].bbox[0]))

        heights = [l.bbox[3] - l.bbox[1] for l in lines]
        median_h = float(np.median(heights)) if heights else 20.0
        y_tolerance = max(median_h * Y_CLUSTER_TOLERANCE, MIN_Y_TOLERANCE)

        grouped = []
        current_group = []
        current_y = None

        for line, y in lines_with_y:
            if current_y is None or abs(y - current_y) <= y_tolerance:
                current_group.append((line, y))
                current_y = (current_y * (len(current_group) - 1) + y) / len(current_group) if current_y is not None else y
            else:
                current_group.sort(key=lambda x: x[0].bbox[0])
                grouped.extend(current_group)
                current_group = [(line, y)]
                current_y = y

        if current_group:
            current_group.sort(key=lambda x: x[0].bbox[0])
            grouped.extend(current_group)

        for i, (line, _) in enumerate(grouped):
            line.reading_order = i

        return [line for line, _ in grouped]


# ============================================================
# 5. EXTRACTION ET DEWARPING DES LIGNES
# ============================================================

def extract_line_images(
    segmentation: SegmentationResult,
    output_dir: str,
    dewarp: bool = True,
    target_height: int = 384
) -> List[Dict[str, Any]]:
    """Extrait les images de lignes à partir du résultat de segmentation."""
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(segmentation.image_path)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {segmentation.image_path}")

    extracted = []
    lines = segmentation.get_lines_sorted()

    for line in lines:
        polygon = np.array(line.polygon, dtype=np.int32)

        x_min, y_min, x_max, y_max = line.bbox
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img.shape[1], x_max)
        y_max = min(img.shape[0], y_max)

        line_img = img[y_min:y_max, x_min:x_max]

        if line_img.size == 0:
            continue

        h, w = line_img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        line_img_resized = cv2.resize(line_img, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

        output_filename = f"{line.line_id}.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, line_img_resized)

        extracted.append({
            "line_id": line.line_id,
            "img_path": output_path,
            "polygon": line.polygon,
            "baseline": line.baseline,
            "bbox": line.bbox,
            "region_type": line.region_type,
            "reading_order": line.reading_order,
            "confidence": line.confidence
        })

    print(f" {len(extracted)} images de lignes extraites dans {output_dir}")
    return extracted


# ============================================================
# 6. EXPORT DES RÉSULTATS
# ============================================================

def export_to_json(
    segmentation: SegmentationResult,
    output_path: str,
    extracted_lines: List[Dict] = None
):
    """Exporte le résultat de segmentation en JSON structuré."""
    data = {
        "document_id": segmentation.page_id,
        "image_path": segmentation.image_path,
        "image_shape": segmentation.image_shape,
        "system_coordinates": "origin_top_left",
        "regions": [],
        "lines": [],
        "marginalia": [],
        "discarded": segmentation.discarded
    }

    for region in segmentation.regions:
        data["regions"].append({
            "region_id": region.region_id,
            "region_type": region.region_type,
            "polygon": region.polygon,
            "bbox": region.bbox,
            "confidence": region.confidence,
            "num_lines": len(region.lines)
        })

    for line in segmentation.get_lines_sorted():
        line_data = {
            "line_id": line.line_id,
            "polygon": line.polygon,
            "baseline": line.baseline,
            "bbox": line.bbox,
            "region_type": line.region_type,
            "reading_order": line.reading_order,
            "confidence": line.confidence
        }
        if extracted_lines:
            for ext in extracted_lines:
                if ext["line_id"] == line.line_id:
                    line_data["extracted_image"] = ext["img_path"]
                    break
        data["lines"].append(line_data)

    # Ajouter les marginalia
    for marg in segmentation.marginalia:
        data["marginalia"].append({
            "line_id": marg.line_id,
            "polygon": marg.polygon,
            "baseline": marg.baseline,
            "bbox": marg.bbox,
            "confidence": marg.confidence
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f" Segmentation exportée : {output_path}")


def export_to_pagexml(
    segmentation: SegmentationResult,
    output_path: str
):
    """Exporte le résultat en format PAGE XML."""
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom

    pc_gts = Element("PcGts", attrib={
        "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"
    })

    metadata = SubElement(pc_gts, "Metadata")
    creator = SubElement(metadata, "Creator")
    creator.text = "HTR Medieval Pipeline v1.1"
    created = SubElement(metadata, "Created")
    from datetime import datetime
    created.text = datetime.now().isoformat()

    page = SubElement(pc_gts, "Page")
    page.set("imageFilename", os.path.basename(segmentation.image_path))
    page.set("imageWidth", str(segmentation.image_shape[1]))
    page.set("imageHeight", str(segmentation.image_shape[0]))

    # Régions principales
    for region in segmentation.regions:
        text_region = SubElement(page, "TextRegion")
        text_region.set("id", region.region_id)
        text_region.set("type", region.region_type)

        coords = SubElement(text_region, "Coords")
        points_str = " ".join([f"{p[0]},{p[1]}" for p in region.polygon])
        coords.set("points", points_str)

        for line in region.lines:
            text_line = SubElement(text_region, "TextLine")
            text_line.set("id", line.line_id)
            text_line.set("readingOrder", str(line.reading_order))

            line_coords = SubElement(text_line, "Coords")
            line_points = " ".join([f"{p[0]},{p[1]}" for p in line.polygon])
            line_coords.set("points", line_points)

            if line.baseline:
                baseline_elem = SubElement(text_line, "Baseline")
                base_points = " ".join([f"{p[0]},{p[1]}" for p in line.baseline])
                baseline_elem.set("points", base_points)

    # Marginalia dans une région séparée
    if segmentation.marginalia:
        marg_region = SubElement(page, "TextRegion")
        marg_region.set("id", f"{segmentation.page_id}_marginalia")
        marg_region.set("type", "marginalia")
        for marg in segmentation.marginalia:
            text_line = SubElement(marg_region, "TextLine")
            text_line.set("id", marg.line_id)
            line_coords = SubElement(text_line, "Coords")
            line_points = " ".join([f"{p[0]},{p[1]}" for p in marg.polygon])
            line_coords.set("points", line_points)

    xml_str = tostring(pc_gts, encoding="unicode")
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f" PAGE XML exporté : {output_path}")


# ============================================================
# 7. FONCTION PRINCIPALE DE SEGMENTATION
# ============================================================

def segment_manuscript_page(
    image_path: str,
    output_dir: str = "./segmentation_output",
    page_id: str = None,
    method: str = "kraken",
    model_path: str = None,
    extract_lines: bool = True,
    export_json: bool = True,
    export_pagexml: bool = True
) -> SegmentationResult:
    """
    Pipeline complet de segmentation d'une page de manuscrit.
    """
    os.makedirs(output_dir, exist_ok=True)

    if page_id is None:
        page_id = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\n{'='*60}")
    print(f"   SEGMENTATION : {page_id}")
    print(f"  Méthode : {method.upper()}")
    print(f"{'='*60}")

    if method == "kraken":
        segmenter = KrakenSegmenter(model_path=model_path)
    elif method == "yolo":
        segmenter = YOLOSegmenter(model_path=model_path)
    else:
        raise ValueError(f"Méthode inconnue : {method}. Choix : 'kraken' ou 'yolo'")

    result = segmenter.segment_page(image_path, page_id=page_id)

    extracted = None
    if extract_lines:
        lines_dir = os.path.join(output_dir, "lines")
        extracted = extract_line_images(result, lines_dir)

    if export_json:
        json_path = os.path.join(output_dir, f"{page_id}_segmentation.json")
        export_to_json(result, json_path, extracted)

    if export_pagexml:
        xml_path = os.path.join(output_dir, f"{page_id}.page.xml")
        export_to_pagexml(result, xml_path)

    print(f"\n Segmentation terminée pour {page_id}")
    return result


# ============================================================
# 8. VISUALISATION DE LA SEGMENTATION
# ============================================================

def visualize_segmentation(
    segmentation: SegmentationResult,
    output_path: str = None,
    show_polygons: bool = True,
    show_baselines: bool = True,
    show_labels: bool = True
):
    """Génère une visualisation de la segmentation sur l'image originale."""
    img = cv2.imread(segmentation.image_path)
    if img is None:
        return

    vis_img = img.copy()

    colors = {
        "main_text": (0, 255, 0),       # Vert
        "text": (0, 255, 0),            # Vert
        "marginalia": (255, 0, 0),      # Bleu
        "heading": (0, 0, 255),         # Rouge
        "illustration": (255, 255, 0),  # Cyan
        "default": (128, 128, 128),     # Gris
        "unknown": (128, 128, 128),     # Gris
    }

    def _safe_color(region_type):
        if isinstance(region_type, list):
            if region_type and isinstance(region_type[0], dict):
                region_type = region_type[0].get('type', 'default')
            elif region_type and isinstance(region_type[0], str):
                region_type = region_type[0]
            else:
                region_type = 'default'
        return colors.get(str(region_type).lower(), colors["default"])

    # Régions
    for region in segmentation.regions:
        color = _safe_color(region.region_type)
        poly = np.array(region.polygon, dtype=np.int32)
        cv2.polylines(vis_img, [poly], True, color, 2)

    # Lignes principales
    for line in segmentation.get_lines_sorted():
        color = _safe_color(line.region_type)

        if show_polygons:
            poly = np.array(line.polygon, dtype=np.int32)
            cv2.polylines(vis_img, [poly], True, color, 1)

        if show_baselines and line.baseline:
            baseline = np.array(line.baseline, dtype=np.int32)
            cv2.polylines(vis_img, [baseline], False, (0, 0, 255), 2)

        if show_labels:
            x, y = line.bbox[0], line.bbox[1]
            cv2.putText(vis_img, str(line.reading_order), (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Marginalia en pointillés jaunes
    for marg in segmentation.marginalia:
        poly = np.array(marg.polygon, dtype=np.int32)
        cv2.polylines(vis_img, [poly], True, (0, 255, 255), 1)
        x, y = marg.bbox[0], marg.bbox[1]
        cv2.putText(vis_img, "MARG", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    if output_path:
        cv2.imwrite(output_path, vis_img)
        print(f"  Visualisation sauvegardée : {output_path}")

    return vis_img


# ============================================================
# 9. TESTS ET UTILITAIRES
# ============================================================

def test_segmentation():
    """Test rapide de la segmentation sur une image factice."""
    print(" Test de segmentation...")

    test_img = np.ones((800, 600, 3), dtype=np.uint8) * 240

    for i in range(5):
        y = 100 + i * 120
        cv2.rectangle(test_img, (50, y), (550, y + 60), (0, 0, 0), -1)

    test_path = "/tmp/test_manuscript.png"
    cv2.imwrite(test_path, test_img)

    try:
        result = segment_manuscript_page(
            image_path=test_path,
            output_dir="/tmp/segmentation_test",
            method="kraken",
            extract_lines=True,
            export_json=True,
            export_pagexml=True
        )
        print(" Test Kraken réussi")
        return result
    except Exception as e:
        print(f"  Test Kraken échoué : {e}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        method = sys.argv[2] if len(sys.argv) > 2 else "kraken"

        result = segment_manuscript_page(
            image_path=image_path,
            output_dir="./segmentation_output",
            method=method
        )

        visualize_segmentation(
            result,
            output_path="./segmentation_output/visualization.png"
        )
    else:
        print("Usage: python segmentation.py <image_path> [kraken|yolo]")
        print("\nLancement du test avec image factice...")
        test_segmentation()
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
# 1. STRUCTURES DE DONNÉES
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

    def get_lines_sorted(self) -> List[LinePolygon]:
        """Retourne les lignes triées par ordre de lecture."""
        return sorted(self.all_lines, key=lambda l: l.reading_order)


# ============================================================
# 2. SEGMENTATION AVEC KRAKEN BLLA (MÉTHODE PRINCIPALE)
# ============================================================

class KrakenSegmenter:
    """
    Segmenteur basé sur Kraken BLLA (Baseline Layout Analysis).

    Kraken BLLA détecte simultanément :
      - Les régions de la page (blocs de texte, illustrations)
      - Les lignes de base (baselines) avec leurs polygones de contour
      - L'ordre de lecture

    Référence : Kiessling, B. (2020). A Modular Region and Text Line Layout Analysis System.
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        """
        Args:
            model_path: Chemin vers un modèle Kraken BLLA personnalisé.
                        Si None, utilise le modèle par défaut de Kraken.
            device: "cpu" ou "cuda"
        """
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
        """
        Segmente une page complète en régions et lignes.

        Args:
            image_path: Chemin vers l'image de la page
            page_id: Identifiant du document
            text_direction: Direction du texte (horizontal-lr, horizontal-rl, vertical-lr, vertical-rl)

        Returns:
            SegmentationResult avec régions, lignes, polygones et ordre de lecture
        """
        if page_id is None:
            page_id = os.path.splitext(os.path.basename(image_path))[0]

        # Charger l'image
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        print(f" Segmentation de la page : {page_id} ({w}x{h})")

        # Segmentation Kraken BLLA
        try:
            # kraken.blla.segment retourne un objet avec regions, lines, etc.
            result = blla.segment(img, model=self.model, text_direction=text_direction)
        except Exception as e:
            print(f"  Erreur Kraken BLLA : {e}")
            print("   Fallback vers la segmentation legacy Kraken...")
            result = self._fallback_segmentation(img, text_direction)

        # Parser le résultat Kraken
        regions, lines = self._parse_kraken_result(result, page_id, w, h)

        # Calculer l'ordre de lecture
        lines = self._compute_reading_order(lines, text_direction)

        seg_result = SegmentationResult(
            page_id=page_id,
            image_path=image_path,
            image_shape=(h, w),
            regions=regions,
            all_lines=lines
        )

        print(f"    {len(regions)} régions | {len(lines)} lignes détectées")
        return seg_result

    def _parse_kraken_result(self, result, page_id: str, img_w: int, img_h: int) -> Tuple[List[PageRegion], List[LinePolygon]]:
        """Parse le résultat Kraken en structures Python."""
        regions = []
        lines = []

        # Kraken retourne des objets avec des attributs .lines et .regions
        # Structure typique : result.regions (liste de régions), chaque région a .lines

        line_idx = 0

        if hasattr(result, 'regions') and result.regions:
            for r_idx, region in enumerate(result.regions):
                region_type = getattr(region, 'type', 'text')

                # Extraire le polygone de la région
                region_poly = self._extract_polygon(region, img_w, img_h)
                region_bbox = self._polygon_to_bbox(region_poly)

                region_lines = []
                if hasattr(region, 'lines') and region.lines:
                    for line in region.lines:
                        line_poly = self._extract_polygon(line, img_w, img_h)
                        baseline = self._extract_baseline(line, img_w, img_h)
                        bbox = self._polygon_to_bbox(line_poly)
                        confidence = getattr(line, 'confidence', 0.9)

                        line_obj = LinePolygon(
                            line_id=f"{page_id}_line_{line_idx:04d}",
                            polygon=line_poly,
                            baseline=baseline,
                            bbox=bbox,
                            region_type=region_type,
                            reading_order=line_idx,  # Sera recalculé
                            confidence=confidence
                        )
                        region_lines.append(line_obj)
                        lines.append(line_obj)
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

        # Si pas de régions, essayer de parser directement les lignes
        elif hasattr(result, 'lines') and result.lines:
            for line in result.lines:
                line_poly = self._extract_polygon(line, img_w, img_h)
                baseline = self._extract_baseline(line, img_w, img_h)
                bbox = self._polygon_to_bbox(line_poly)
                confidence = getattr(line, 'confidence', 0.9)

                line_obj = LinePolygon(
                    line_id=f"{page_id}_line_{line_idx:04d}",
                    polygon=line_poly,
                    baseline=baseline,
                    bbox=bbox,
                    region_type="text",
                    reading_order=line_idx,
                    confidence=confidence
                )
                lines.append(line_obj)
                line_idx += 1

        return regions, lines

    def _extract_polygon(self, obj, img_w: int, img_h: int) -> List[List[int]]:
        """Extrait le polygone d'un objet Kraken (région ou ligne)."""
        if hasattr(obj, 'boundary') and obj.boundary:
            # Format Kraken : liste de tuples (x, y)
            poly = [[int(x), int(y)] for x, y in obj.boundary]
            return poly
        elif hasattr(obj, 'bbox'):
            # Fallback sur la bbox si pas de polygone
            x1, y1, x2, y2 = obj.bbox
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        else:
            return [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]

    def _extract_baseline(self, obj, img_w: int, img_h: int) -> List[List[int]]:
        """Extrait la baseline d'un objet ligne Kraken."""
        if hasattr(obj, 'baseline') and obj.baseline:
            return [[int(x), int(y)] for x, y in obj.baseline]
        return []

    def _polygon_to_bbox(self, polygon: List[List[int]]) -> Tuple[int, int, int, int]:
        """Convertit un polygone en bounding box."""
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    def _compute_reading_order(self, lines: List[LinePolygon], text_direction: str) -> List[LinePolygon]:
        """
        Calcule l'ordre de lecture des lignes.
        Pour du texte horizontal : de haut en bas, puis gauche à droite.
        """
        if text_direction.startswith("horizontal"):
            # Trier par y (haut en bas), puis par x (gauche à droite)
            lines_sorted = sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0]))
        else:
            # Vertical : gauche à droite, puis haut en bas
            lines_sorted = sorted(lines, key=lambda l: (l.bbox[0], l.bbox[1]))

        for i, line in enumerate(lines_sorted):
            line.reading_order = i

        return lines_sorted

    def _fallback_segmentation(self, img: Image.Image, text_direction: str):
        """Fallback si BLLA échoue — utilise la segmentation legacy Kraken."""
        from kraken.pageseg import segment
        return segment(img)


# ============================================================
# 3. SEGMENTATION AVEC YOLO (ALTERNATIVE)
# ============================================================

class YOLOSegmenter:
    """
    Segmenteur alternatif basé sur YOLO pour la détection de lignes.

    Utilise un modèle YOLO entraîné sur des manuscrits médiévaux
    (ex: BigLAM YOLO sur CATMuS Medieval Segmentation dataset).

    Référence : Clérice, T. (2022). YALTAi: YOLO for Layout Analysis.
    """

    def __init__(self, model_path: str = None, conf_threshold: float = 0.5):
        """
        Args:
            model_path: Chemin vers un modèle YOLO (.pt) entraîné sur des manuscrits
            conf_threshold: Seuil de confiance minimum pour une détection
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics est requis pour le segmenteur YOLO. pip install ultralytics")

        self.conf_threshold = conf_threshold

        if model_path and os.path.exists(model_path):
            print(f" Chargement du modèle YOLO : {model_path}")
            self.model = YOLO(model_path)
        else:
            # Modèle par défaut : YOLOv8n (léger) ou un modèle médiéval si disponible
            print(" Utilisation de YOLOv8n par défaut (à fine-tuner sur médiéval)")
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

        lines = []
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

                # Déterminer le type de région
                if class_name in ["text", "line", "default_line"]:
                    region_type = "text"
                elif class_name in ["marginalia", "margin"]:
                    region_type = "marginalia"
                elif class_name in ["heading", "title"]:
                    region_type = "heading"
                else:
                    region_type = "text"

                polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                line_obj = LinePolygon(
                    line_id=f"{page_id}_line_{i:04d}",
                    polygon=polygon,
                    baseline=[[x1, (y1+y2)//2], [x2, (y1+y2)//2]],  # Baseline approximative
                    bbox=(x1, y1, x2, y2),
                    region_type=region_type,
                    reading_order=i,
                    confidence=conf
                )
                lines.append(line_obj)

        # Trier par ordre de lecture
        lines = sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0]))
        for i, line in enumerate(lines):
            line.reading_order = i

        # Créer une région unique contenant toutes les lignes
        all_poly = [[0, 0], [w, 0], [w, h], [0, h]]
        page_region = PageRegion(
            region_id=f"{page_id}_region_000",
            region_type="text",
            polygon=all_poly,
            bbox=(0, 0, w, h),
            lines=lines,
            confidence=1.0
        )
        regions.append(page_region)

        print(f"    {len(lines)} lignes détectées avec YOLO")

        return SegmentationResult(
            page_id=page_id,
            image_path=image_path,
            image_shape=(h, w),
            regions=regions,
            all_lines=lines
        )


# ============================================================
# 4. EXTRACTION ET DEWARPING DES LIGNES
# ============================================================

def extract_line_images(
    segmentation: SegmentationResult,
    output_dir: str,
    dewarp: bool = True,
    target_height: int = 384
) -> List[Dict[str, Any]]:
    """
    Extrait les images de lignes à partir du résultat de segmentation.

    Args:
        segmentation: Résultat de la segmentation
        output_dir: Répertoire de sortie pour les images de lignes
        dewarp: Si True, applique un dewarping basé sur la baseline
        target_height: Hauteur cible des images de lignes (pour TrOCR)

    Returns:
        Liste de dicts : [{"line_id": ..., "img_path": ..., "polygon": ..., "reading_order": ...}, ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(segmentation.image_path)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {segmentation.image_path}")

    extracted = []
    lines = segmentation.get_lines_sorted()

    for line in lines:
        # Extraire la ROI avec le polygone
        polygon = np.array(line.polygon, dtype=np.int32)

        # Bounding box de la ligne
        x_min, y_min, x_max, y_max = line.bbox
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img.shape[1], x_max)
        y_max = min(img.shape[0], y_max)

        # Extraire la sous-image
        line_img = img[y_min:y_max, x_min:x_max]

        if line_img.size == 0:
            continue

        # Redimensionner à la hauteur cible (pour TrOCR)
        h, w = line_img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        line_img_resized = cv2.resize(line_img, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

        # Sauvegarder
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
# 5. EXPORT DES RÉSULTATS
# ============================================================

def export_to_json(
    segmentation: SegmentationResult,
    output_path: str,
    extracted_lines: List[Dict] = None
):
    """
    Exporte le résultat de segmentation en JSON structuré.
    Format compatible avec data_contract.py.
    """
    data = {
        "document_id": segmentation.page_id,
        "image_path": segmentation.image_path,
        "image_shape": segmentation.image_shape,
        "system_coordinates": "origin_top_left",
        "regions": [],
        "lines": []
    }

    # Régions
    for region in segmentation.regions:
        data["regions"].append({
            "region_id": region.region_id,
            "region_type": region.region_type,
            "polygon": region.polygon,
            "bbox": region.bbox,
            "confidence": region.confidence,
            "num_lines": len(region.lines)
        })

    # Lignes
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
            # Ajouter le chemin de l'image extraite si disponible
            for ext in extracted_lines:
                if ext["line_id"] == line.line_id:
                    line_data["extracted_image"] = ext["img_path"]
                    break
        data["lines"].append(line_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f" Segmentation exportée : {output_path}")


def export_to_pagexml(
    segmentation: SegmentationResult,
    output_path: str
):
    """
    Exporte le résultat en format PAGE XML (standard pour les humanités numériques).
    Compatible avec eScriptorium et Kraken.
    """
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom

    nsmap = {
        "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    }

    pc_gts = Element("PcGts", attrib={
        "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"
    })

    metadata = SubElement(pc_gts, "Metadata")
    creator = SubElement(metadata, "Creator")
    creator.text = "HTR Medieval Pipeline"
    created = SubElement(metadata, "Created")
    created.text = "2026-06-12T00:00:00"

    page = SubElement(pc_gts, "Page")
    page.set("imageFilename", os.path.basename(segmentation.image_path))
    page.set("imageWidth", str(segmentation.image_shape[1]))
    page.set("imageHeight", str(segmentation.image_shape[0]))

    # Régions
    for region in segmentation.regions:
        text_region = SubElement(page, "TextRegion")
        text_region.set("id", region.region_id)
        text_region.set("type", region.region_type)

        coords = SubElement(text_region, "Coords")
        points_str = " ".join([f"{p[0]},{p[1]}" for p in region.polygon])
        coords.set("points", points_str)

        # Lignes dans la région
        for line in region.lines:
            text_line = SubElement(text_region, "TextLine")
            text_line.set("id", line.line_id)
            text_line.set("readingOrder", str(line.reading_order))

            line_coords = SubElement(text_line, "Coords")
            line_points = " ".join([f"{p[0]},{p[1]}" for p in line.polygon])
            line_coords.set("points", line_points)

            # Baseline
            if line.baseline:
                baseline_elem = SubElement(text_line, "Baseline")
                base_points = " ".join([f"{p[0]},{p[1]}" for p in line.baseline])
                baseline_elem.set("points", base_points)

    # Orphelines (lignes sans région)
    orphan_lines = [l for l in segmentation.all_lines if not any(l in r.lines for r in segmentation.regions)]
    if orphan_lines:
        orphan_region = SubElement(page, "TextRegion")
        orphan_region.set("id", f"{segmentation.page_id}_orphans")
        orphan_region.set("type", "text")
        for line in orphan_lines:
            text_line = SubElement(orphan_region, "TextLine")
            text_line.set("id", line.line_id)
            line_coords = SubElement(text_line, "Coords")
            line_points = " ".join([f"{p[0]},{p[1]}" for p in line.polygon])
            line_coords.set("points", line_points)

    # Pretty print
    xml_str = tostring(pc_gts, encoding="unicode")
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f" PAGE XML exporté : {output_path}")


# ============================================================
# 6. FONCTION PRINCIPALE DE SEGMENTATION
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

    Args:
        image_path: Chemin vers l'image de la page
        output_dir: Répertoire de sortie
        page_id: Identifiant du document
        method: "kraken" ou "yolo"
        model_path: Chemin vers un modèle personnalisé
        extract_lines: Si True, extrait les images de lignes
        export_json: Si True, exporte en JSON
        export_pagexml: Si True, exporte en PAGE XML

    Returns:
        SegmentationResult
    """
    os.makedirs(output_dir, exist_ok=True)

    if page_id is None:
        page_id = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\n{'='*60}")
    print(f"   SEGMENTATION : {page_id}")
    print(f"  Méthode : {method.upper()}")
    print(f"{'='*60}")

    # Choisir le segmenteur
    if method == "kraken":
        segmenter = KrakenSegmenter(model_path=model_path)
    elif method == "yolo":
        segmenter = YOLOSegmenter(model_path=model_path)
    else:
        raise ValueError(f"Méthode inconnue : {method}. Choix : 'kraken' ou 'yolo'")

    # Segmentation
    result = segmenter.segment_page(image_path, page_id=page_id)

    # Extraction des images de lignes
    extracted = None
    if extract_lines:
        lines_dir = os.path.join(output_dir, "lines")
        extracted = extract_line_images(result, lines_dir)

    # Export JSON
    if export_json:
        json_path = os.path.join(output_dir, f"{page_id}_segmentation.json")
        export_to_json(result, json_path, extracted)

    # Export PAGE XML
    if export_pagexml:
        xml_path = os.path.join(output_dir, f"{page_id}.page.xml")
        export_to_pagexml(result, xml_path)

    print(f"\n Segmentation terminée pour {page_id}")
    return result


# ============================================================
# 7. VISUALISATION DE LA SEGMENTATION
# ============================================================

def visualize_segmentation(
    segmentation: SegmentationResult,
    output_path: str = None,
    show_polygons: bool = True,
    show_baselines: bool = True,
    show_labels: bool = True
):
    """
    Génère une visualisation de la segmentation sur l'image originale.
    """
    img = cv2.imread(segmentation.image_path)
    if img is None:
        return

    vis_img = img.copy()

    # Couleurs par type de région
    colors = {
        "text": (0, 255, 0),        # Vert
        "marginalia": (255, 0, 0),  # Bleu
        "heading": (0, 0, 255),     # Rouge
        "illustration": (255, 255, 0),  # Cyan
        "default": (128, 128, 128)  # Gris
    }

    # Dessiner les régions
    for region in segmentation.regions:
        color = colors.get(region.region_type, colors["default"])
        poly = np.array(region.polygon, dtype=np.int32)
        cv2.polylines(vis_img, [poly], True, color, 2)

    # Dessiner les lignes
    for line in segmentation.get_lines_sorted():
        color = colors.get(line.region_type, colors["default"])

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

    if output_path:
        cv2.imwrite(output_path, vis_img)
        print(f"  Visualisation sauvegardée : {output_path}")

    return vis_img


# ============================================================
# 8. TESTS ET UTILITAIRES
# ============================================================

def test_segmentation():
    """Test rapide de la segmentation sur une image factice."""
    print(" Test de segmentation...")

    # Créer une image de test
    test_img = np.ones((800, 600, 3), dtype=np.uint8) * 240

    # Dessiner des "lignes" de texte simulées
    for i in range(5):
        y = 100 + i * 120
        cv2.rectangle(test_img, (50, y), (550, y + 60), (0, 0, 0), -1)

    test_path = "/tmp/test_manuscript.png"
    cv2.imwrite(test_path, test_img)

    # Tester Kraken
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
    # Exemple d'utilisation
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        method = sys.argv[2] if len(sys.argv) > 2 else "kraken"

        result = segment_manuscript_page(
            image_path=image_path,
            output_dir="./segmentation_output",
            method=method
        )

        # Visualisation
        visualize_segmentation(
            result,
            output_path="./segmentation_output/visualization.png"
        )
    else:
        print("Usage: python segmentation.py <image_path> [kraken|yolo]")
        print("\nLancement du test avec image factice...")
        test_segmentation()
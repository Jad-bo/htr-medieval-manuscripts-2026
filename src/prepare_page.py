"""
Script de téléchargement et préparation de pages de manuscrits médiévaux
à partir du dataset CATMuS Medieval Segmentation (HuggingFace).

Usage:
    python src/prepare_page.py --num_pages 5
    python src/prepare_page.py --output_dir ./data --num_pages 10
"""

import os
import random
import json
import argparse
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
from typing import List, Dict, Any
from PIL import Image

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("❌ Erreur: 'datasets' n'est pas installé. pip install datasets")
    exit(1)


def prepare_test_pages(
    output_dir: str = "./data",
    num_pages: int = 5,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Télécharge des pages aléatoires depuis CATMuS Medieval Segmentation
    et sauvegarde les images + ground truth (polygones, régions).

    Args:
        output_dir: Répertoire de sortie (crée raw/ et segmentation_gt/)
        num_pages: Nombre de pages à extraire
        seed: Seed pour la reproductibilité

    Returns:
        Liste de dicts avec les métadonnées de chaque page
    """
    random.seed(seed)

    raw_dir = os.path.join(output_dir, "raw")
    gt_dir = os.path.join(output_dir, "segmentation_gt")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    print("📥 Chargement de CATMuS Medieval Segmentation...")

    ds = load_dataset("CATMuS/medieval-segmentation", split="train")

    total = len(ds)
    print(f"   Dataset chargé: {total} pages")

    # Afficher les colonnes disponibles pour debug
    print(f"   Colonnes disponibles: {list(ds.features.keys())}")

    # Sélection aléatoire
    indices = random.sample(range(total), min(num_pages, total))
    print(f"   Sélection de {len(indices)} pages aléatoires (seed={seed})")

    prepared_pages = []

    for i, idx in enumerate(indices):
        page = ds[idx]

        page_id = f"page_test_{i+1:03d}"
        img_path = os.path.join(raw_dir, f"{page_id}.png")
        gt_path = os.path.join(gt_dir, f"{page_id}_gt.json")

        # Sauvegarder l'image - colonne "image" dans le dataset
        image = page["image"]
        image.save(img_path)

        # CORRECTION: Les annotations sont dans la colonne "objects" (dict)
        # avec des clés "id", "bbox", "category", etc.
        objects = page.get("objects", {})

        # Extraire les IDs des objets (textblocks, lines, etc.)
        obj_ids = objects.get("id", [])
        obj_bboxes = objects.get("bbox", [])
        obj_categories = objects.get("category", [])
        obj_areas = objects.get("area", [])

        # Compter les types d'objets
        num_lines = sum(1 for cat in obj_categories if "line" in str(cat).lower())
        num_regions = sum(1 for cat in obj_categories if "textblock" in str(cat).lower() or "region" in str(cat).lower())

        gt_data = {
            "page_id": page_id,
            "source": "CATMuS/medieval-segmentation",
            "dataset_index": idx,
            "image_path": img_path,
            "image_size": [image.width, image.height],
            "width": page.get("width", image.width),
            "height": page.get("height", image.height),
            "shelfmark": str(page.get("shelfmark", "unknown")),
            "objects": {
                "count": len(obj_ids),
                "ids": obj_ids,
                "bboxes": obj_bboxes,
                "categories": obj_categories,
                "areas": obj_areas
            },
            "num_lines": num_lines,
            "num_regions": num_regions
        }

        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, ensure_ascii=False, indent=2)

        prepared_pages.append({
            "page_id": page_id,
            "image_path": img_path,
            "gt_path": gt_path,
            "shelfmark": gt_data["shelfmark"],
            "num_lines": num_lines,
            "num_regions": num_regions,
            "total_objects": len(obj_ids)
        })

        print(f"   ✅ {page_id}: {gt_data['shelfmark'][:50]}... | "
              f"{num_lines} lignes | {num_regions} régions | {len(obj_ids)} objets")

    # Résumé
    print(f"\n📊 Résumé:")
    print(f"   Pages téléchargées: {len(prepared_pages)}")
    print(f"   Images: {raw_dir}")
    print(f"   Ground truth: {gt_dir}")

    return prepared_pages


def main():
    parser = argparse.ArgumentParser(description="Préparer des pages de test depuis CATMuS")
    parser.add_argument("--output_dir", type=str, default="./data",
                       help="Répertoire de sortie")
    parser.add_argument("--num_pages", type=int, default=5,
                       help="Nombre de pages à extraire")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed pour la reproductibilité")

    args = parser.parse_args()

    prepare_test_pages(
        output_dir=args.output_dir,
        num_pages=args.num_pages,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
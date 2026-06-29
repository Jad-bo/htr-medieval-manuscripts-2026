"""
Module d'inférence HTR pour manuscrits médiévaux (CATMuS) - VERSION TRIDIS.

Charge le modèle TRIDIS (magistermilitum/tridis_HTR) déjà fine-tuné sur médiéval
et transcrit les images de lignes.

Usage :
    python src/inference.py --mode infer --split dev --checkpoint ./checkpoints_production/best_model
    python src/inference.py --mode evaluate --split dev --checkpoint ./checkpoints_production/best_model
    python src/inference.py --mode evaluate --split test --checkpoint ./checkpoints_production/best_model
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# PEFT / LoRA — chargement de l'adaptateur fine-tuné
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("  PEFT non installé. Le chargement de LoRA sera désactivé.")

# Import de la config pour les paramètres de génération
try:
    from config import GENERATION_MAX_LENGTH, GENERATION_NUM_BEAMS, GENERATION_TEMPERATURE
except ImportError:
    GENERATION_MAX_LENGTH = 256
    GENERATION_NUM_BEAMS = 4
    GENERATION_TEMPERATURE = 0.7

# Métriques HTR
import evaluate
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

# ============================================================
# MODÈLE TRIDIS (déjà fine-tuné médiéval)
# ============================================================
TRIDIS_MODEL = "magistermilitum/tridis_HTR"


# ============================================================
# 1. CHARGEMENT DU MODÈLE TRIDIS
# ============================================================

def load_htr_model(
    checkpoint_path: str = None,
    model_base_name: str = TRIDIS_MODEL,
    device: str = None
) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor, Dict]:
    """
    Charge le modèle TRIDIS de base, et injecte l'adaptateur LoRA
    si un checkpoint (répertoire best_model) est fourni.

    Args:
        checkpoint_path: Chemin vers le répertoire best_model (LoRA adapter)
                         ou None pour utiliser TRIDIS brut.
        model_base_name: Modèle de base HuggingFace (défaut: TRIDIS)
        device: "cuda" ou "cpu" (auto-détecté si None)

    Returns:
        (model, processor, config)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # 1. Charger le processor (depuis le checkpoint si dispo, sinon base)
    # ------------------------------------------------------------------
    if checkpoint_path and os.path.isdir(checkpoint_path):
        processor_path = checkpoint_path
        print(f"  Chargement du processor depuis : {processor_path}")
    else:
        processor_path = model_base_name
        print(f"  Chargement du processor depuis : {processor_path}")

    processor = TrOCRProcessor.from_pretrained(processor_path)

    # ------------------------------------------------------------------
    # 2. Charger le modèle de base (TRIDIS)
    # ------------------------------------------------------------------
    print(f"  Chargement du modèle de base : {model_base_name}")
    model = VisionEncoderDecoderModel.from_pretrained(model_base_name)

    # ------------------------------------------------------------------
    # 3. Injecter l'adaptateur LoRA si un checkpoint est fourni
    # ------------------------------------------------------------------
    lora_loaded = False
    if checkpoint_path and os.path.isdir(checkpoint_path):
        adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_config):
            if not PEFT_AVAILABLE:
                raise RuntimeError(
                    "PEFT n'est pas installé. Impossible de charger l'adaptateur LoRA.\n"
                    "  pip install peft"
                )
            print(f"  Injection de l'adaptateur LoRA : {checkpoint_path}")
            model = PeftModel.from_pretrained(model, checkpoint_path)
            # Fusionner les poids pour l'inférence (plus rapide, pas de overhead LoRA)
            model = model.merge_and_unload()
            print(f"   Adaptateur LoRA fusionné avec succès")
            lora_loaded = True
        else:
            print(f"   Avertissement : {checkpoint_path} ne contient pas d'adaptateur LoRA.")
            print(f"    Fichier attendu : {adapter_config}")

    # Configuration génération
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.to(device)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"     Device : {device.upper()}")
    if lora_loaded:
        print(f"     Modèle TRIDIS + LoRA fine-tuné prêt")
    else:
        print(f"     Modèle TRIDIS brut (sans fine-tuning) prêt")

    config = {
        "model": "tridis_HTR",
        "source": model_base_name,
        "description": "TrOCR fine-tuné sur manuscrits documentaires médiévaux",
        "checkpoint_path": checkpoint_path if checkpoint_path else "N/A (modèle pré-entraîné)",
        "lora_loaded": lora_loaded,
        "device": device
    }

    return model, processor, config


# ============================================================
# 2. LECTURE DES DONNÉES (format prepare_dataset.py)
# ============================================================

def load_split_data(
    image_dir: str,
    label_file: str
) -> List[Dict[str, str]]:
    """
    Charge les données d'un split (train/dev/test).

    Format attendu :
        image_name.png\ttexte transcrit\n

    Returns:
        Liste de dicts : [{"img_name": ..., "text": ..., "img_path": ...}, ...]
    """
    samples = []
    if not os.path.exists(label_file):
        print(f" Fichier introuvable : {label_file}")
        return samples

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            img_name, text = line.split("\t", 1)
            img_path = os.path.join(image_dir, img_name)
            samples.append({
                "img_name": img_name,
                "text": text,
                "img_path": img_path
            })

    print(f" {len(samples)} échantillons chargés depuis {label_file}")
    return samples


# ============================================================
# 3. TRANSCRIPTION D'UNE IMAGE
# ============================================================

def transcribe_image(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    image_path: str,
    device: str = "cuda",
    max_length: int = None,
    num_beams: int = None
) -> str:
    """
    Transcrit une seule image de manuscrit avec TRIDIS.

    Args:
        max_length: Longueur max de la séquence générée (défaut: config.GENERATION_MAX_LENGTH)
        num_beams: Nombre de beams pour la recherche (défaut: config.GENERATION_NUM_BEAMS)
    """
    # Valeurs par défaut depuis la config
    if max_length is None:
        max_length = GENERATION_MAX_LENGTH
    if num_beams is None:
        num_beams = GENERATION_NUM_BEAMS

    if not os.path.exists(image_path):
        return "[IMAGE_MANQUANTE]"

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            temperature=GENERATION_TEMPERATURE if num_beams > 1 else 1.0,
            do_sample=(num_beams > 1)
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


# ============================================================
# 4. INFÉRENCE SUR UN SPLIT COMPLET
# ============================================================

def run_inference(
    image_dir: str,
    label_file: str,
    checkpoint_path: str = None,
    output_file: str = None,
    model_base_name: str = TRIDIS_MODEL,
    transcribe_all: bool = True,
    num_beams: int = None
) -> List[Dict]:
    """
    Transcrit toutes les lignes d'un split et sauvegarde les résultats.
    """
    model, processor, config = load_htr_model(checkpoint_path, model_base_name)
    device = next(model.parameters()).device.type

    samples = load_split_data(image_dir, label_file)
    if not samples:
        return []

    results = []
    print(f"\n🚀 Inférence TRIDIS sur {len(samples)} images...")

    for i, sample in enumerate(samples):
        pred_text = transcribe_image(model, processor, sample["img_path"], device, num_beams=num_beams)

        results.append({
            "img_name": sample["img_name"],
            "truth": sample["text"],
            "pred": pred_text
        })

        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            print(f"   [{i+1}/{len(samples)}] {sample['img_name']} : {pred_text[:60]}...")

    # Sauvegarde des prédictions
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(f"{r['img_name']}\t{r['pred']}\n")
        print(f"\n💾 Prédictions sauvegardées : {output_file}")

    return results


# ============================================================
# 5. ÉVALUATION (CER/WER sur un split)
# ============================================================

def run_evaluation(
    image_dir: str,
    label_file: str,
    checkpoint_path: str = None,
    output_dir: str = "./evaluation_results",
    model_base_name: str = TRIDIS_MODEL,
    num_beams: int = None
) -> Dict[str, float]:
    """
    Évalue le modèle sur un split et calcule CER/WER.
    """
    print(f"\n{'='*60}")
    print(f"   ÉVALUATION TRIDIS")
    print(f"   Modèle : {TRIDIS_MODEL}")
    print(f"{'='*60}")

    results = run_inference(
        image_dir=image_dir,
        label_file=label_file,
        checkpoint_path=checkpoint_path,
        output_file=os.path.join(output_dir, "predictions.txt"),
        model_base_name=model_base_name,
        num_beams=num_beams
    )

    if not results:
        return {}

    truths = [r["truth"] for r in results]
    preds = [r["pred"] for r in results]

    # Nettoyage des chaînes vides
    truths = [t if t.strip() else " " for t in truths]
    preds = [p if p.strip() else " " for p in preds]

    cer = cer_metric.compute(predictions=preds, references=truths)
    wer = wer_metric.compute(predictions=preds, references=truths)

    accuracy = 1.0 - cer

    print(f"\n{'='*60}")
    print(f"   RÉSULTATS")
    print(f"{'='*60}")
    print(f"  CER (Character Error Rate) : {cer:.4f}  ({cer*100:.2f}%)")
    print(f"  WER (Word Error Rate)      : {wer:.4f}  ({wer*100:.2f}%)")
    print(f"  Accuracy (1 - CER)         : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"{'='*60}")

    # Sauvegarder le rapport
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "cer": float(cer),
        "wer": float(wer),
        "accuracy": float(accuracy),
        "num_samples": len(results),
        "model": TRIDIS_MODEL,
        "model_type": "tridis_lora_finetuned" if (checkpoint_path and os.path.isdir(checkpoint_path) and os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))) else "tridis_pretrained",
        "checkpoint": checkpoint_path if checkpoint_path else "N/A (modèle pré-entraîné HuggingFace)",
        "lora_loaded": bool(checkpoint_path and os.path.isdir(checkpoint_path) and os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")))
    }
    with open(os.path.join(output_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Graphe des erreurs par échantillon
    plot_error_analysis(results, output_dir)

    return report


def plot_error_analysis(results: List[Dict], output_dir: str):
    """Génère un graphe d'analyse des erreurs CER par échantillon."""
    from jiwer import cer as jiwer_cer

    cers = []
    for r in results:
        try:
            c = jiwer_cer(r["pred"], r["truth"])
        except:
            c = 1.0
        cers.append(c)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution des CER
    ax = axes[0]
    ax.hist(cers, bins=20, color='steelblue', edgecolor='navy', alpha=0.7)
    ax.axvline(x=np.mean(cers), color='red', linestyle='--', linewidth=2, label=f'Moyenne = {np.mean(cers):.3f}')
    ax.set_xlabel("CER par échantillon")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des erreurs CER")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CER par échantillon (trié)
    ax = axes[1]
    sorted_cers = sorted(cers)
    ax.plot(range(len(sorted_cers)), sorted_cers, 'b-', linewidth=1, alpha=0.7)
    ax.fill_between(range(len(sorted_cers)), sorted_cers, alpha=0.3)
    ax.set_xlabel("Échantillon (trié par CER)")
    ax.set_ylabel("CER")
    ax.set_title("CER par échantillon (ordre croissant)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "error_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Analyse des erreurs sauvegardée : {plot_path}")


# ============================================================
# 6. VISUALISATION DES PRÉDICTIONS
# ============================================================

def visualize_predictions(
    image_dir: str,
    label_file: str,
    checkpoint_path: str = None,
    output_dir: str = "./visualizations",
    num_samples: int = 10,
    model_base_name: str = TRIDIS_MODEL
):
    """
    Génère une grille d'images avec texte prédit vs vérité terrain.
    """
    model, processor, config = load_htr_model(checkpoint_path, model_base_name)
    device = next(model.parameters()).device.type

    samples = load_split_data(image_dir, label_file)
    if not samples:
        return

    # Sélectionner un échantillon aléatoire
    import random
    selected = random.sample(samples, min(num_samples, len(samples)))

    n_cols = 2
    n_rows = (len(selected) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    print(f"\n Génération de {len(selected)} visualisations...")

    for i, sample in enumerate(selected):
        pred_text = transcribe_image(model, processor, sample["img_path"], device)

        # Calculer le CER de cet échantillon
        from jiwer import cer as jiwer_cer
        try:
            sample_cer = jiwer_cer(pred_text, sample["text"])
        except:
            sample_cer = 1.0

        # Charger l'image
        img = Image.open(sample["img_path"]).convert("RGB")

        ax = axes[i] if len(selected) > 1 else axes[0]
        ax.imshow(img)
        ax.axis('off')

        # Couleur du titre selon la qualité
        color = 'green' if sample_cer < 0.1 else 'orange' if sample_cer < 0.3 else 'red'

        title = f"CER: {sample_cer:.2%}\n"
        title += f" Vérité : {sample['text'][:80]}\n"
        title += f" Prédit : {pred_text[:80]}"
        ax.set_title(title, fontsize=9, color=color, loc='left', wrap=True)

    # Masquer les axes vides
    for j in range(len(selected), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"predictions_grid_{num_samples}samples.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Grille de visualisation sauvegardée : {plot_path}")


# ============================================================
# 7. INTERFACE EN LIGNE DE COMMANDE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inférence et évaluation HTR avec TRIDIS pour manuscrits médiévaux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  # Inférence avec TRIDIS brut (sans fine-tuning)
  python inference.py --mode infer --split dev

  # Évaluation avec le modèle LoRA fine-tuné
  python inference.py --mode evaluate --split test --checkpoint ./checkpoints_production/best_model

  # Visualiser 15 prédictions avec le modèle fine-tuné
  python inference.py --mode visualize --split dev --num_samples 15 --checkpoint ./checkpoints_production/best_model
        """
    )

    parser.add_argument("--mode", type=str, required=True,
                       choices=["infer", "evaluate", "visualize"],
                       help="Mode : infer (transcription), evaluate (CER/WER), visualize (grille d'images)")
    parser.add_argument("--split", type=str, required=True,
                       choices=["train", "dev", "test"],
                       help="Split à utiliser (train/dev/test)")
    parser.add_argument("--data_dir", type=str, default="./data/catmus",
                       help="Répertoire contenant images/ et les fichiers .txt")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Chemin vers le répertoire best_model contenant l'adaptateur LoRA (ex: ./checkpoints_production/best_model)")
    parser.add_argument("--model_base", type=str, default=TRIDIS_MODEL,
                       help="Modèle HuggingFace (défaut: TRIDIS)")
    parser.add_argument("--output_dir", type=str, default="./inference_results",
                       help="Répertoire de sortie pour les résultats")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Nombre d'échantillons pour la visualisation")
    parser.add_argument("--num_beams", type=int, default=4,
                       help="Nombre de beams pour la génération (1=greedy, 4=meilleur)")

    args = parser.parse_args()

    # Construire les chemins automatiquement
    image_dir = os.path.join(args.data_dir, "images")
    label_file = os.path.join(args.data_dir, f"{args.split}.txt")

    # Vérifier que les fichiers existent
    if not os.path.exists(label_file):
        print(f" Fichier introuvable : {label_file}")
        print(f"   Vérifiez que prepare_dataset.py a bien été exécuté.")
        sys.exit(1)

    if args.mode == "infer":
        output_file = os.path.join(args.output_dir, f"{args.split}_predictions_tridis.txt")
        run_inference(
            image_dir=image_dir,
            label_file=label_file,
            checkpoint_path=args.checkpoint,
            output_file=output_file,
            model_base_name=args.model_base,
            num_beams=args.num_beams
        )

    elif args.mode == "evaluate":
        if args.split == "test":
            print("\n  ÉVALUATION FINALE SUR TEST SET")
            print("   Ce split n'a JAMAIS été vu pendant l'entraînement.")
            print("   Les résultats reflètent la vraie performance du modèle.\n")

        eval_dir = os.path.join(args.output_dir, f"evaluation_{args.split}_tridis")
        run_evaluation(
            image_dir=image_dir,
            label_file=label_file,
            checkpoint_path=args.checkpoint,
            output_dir=eval_dir,
            model_base_name=args.model_base,
            num_beams=args.num_beams
        )

    elif args.mode == "visualize":
        viz_dir = os.path.join(args.output_dir, f"visualizations_{args.split}_tridis")
        visualize_predictions(
            image_dir=image_dir,
            label_file=label_file,
            checkpoint_path=args.checkpoint,
            output_dir=viz_dir,
            num_samples=args.num_samples,
            model_base_name=args.model_base
        )


if __name__ == "__main__":
    main()
"""
Module d'inférence HTR pour manuscrits médiévaux (CATMuS).

Charge le meilleur modèle fine-tuné (LoRA) et transcrit les images de lignes.
Supporte :
  - Inférence sur train/dev/test
  - Évaluation finale sur le split test (jamais vu pendant l'entraînement)
  - Comparaison CER/WER entre prédiction et vérité terrain
  - Visualisation des prédictions (image + texte prédit vs réel)

Usage :
    python inference.py --mode infer --split dev
    python inference.py --mode evaluate --split test
    python inference.py --mode visualize --split dev --num_samples 10
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
from peft import PeftModel, LoraConfig, get_peft_model
import safetensors.torch

# Métriques HTR
import evaluate
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


# ============================================================
# 1. CHARGEMENT DU MODÈLE
# ============================================================

def load_htr_model(
    checkpoint_path: str,
    model_base_name: str = "microsoft/trocr-large-handwritten",
    device: str = None
) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor, Dict]:
    """
    Charge le modèle TrOCR de base + adaptateur LoRA fine-tuné.

    Args:
        checkpoint_path: Chemin vers le dossier best_model (contient adapter_model.safetensors)
        model_base_name: Modèle de base HuggingFace
        device: "cuda" ou "cpu" (auto-détecté si None)

    Returns:
        (model, processor, best_config)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f" Chargement du modèle de base : {model_base_name}")
    processor = TrOCRProcessor.from_pretrained(model_base_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_base_name)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Charger la config du meilleur modèle si disponible
    config_path = os.path.join(checkpoint_path, "best_config.json")
    best_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            best_config = json.load(f)
        print(f"   Config chargée : {best_config}")

    # Injecter LoRA
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    adapter_weights_path = os.path.join(checkpoint_path, "adapter_model.safetensors")

    if os.path.exists(adapter_weights_path):
        lora_r = best_config.get("lora_r", 16)
        lora_alpha = best_config.get("lora_alpha", 32)

        print(f" Injection LoRA (r={lora_r}, alpha={lora_alpha})...")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=None
        )
        model.decoder = get_peft_model(model.decoder, peft_config)

        state_dict = safetensors.torch.load_file(adapter_weights_path)
        model.decoder.load_state_dict(state_dict, strict=False)
        print("    Adaptateur LoRA chargé avec succès.")
    else:
        print("     Aucun adaptateur trouvé — utilisation du modèle de base pur.")

    model.to(device)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"     Device : {device.upper()}")
    return model, processor, best_config


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
    max_new_tokens: int = 64,
    num_beams: int = 4
) -> str:
    """
    Transcrit une seule image de manuscrit.

    Args:
        model: Modèle TrOCR chargé
        processor: TrOCRProcessor
        image_path: Chemin vers l'image
        device: Device PyTorch
        max_new_tokens: Longueur max de la prédiction
        num_beams: Beam search (1 = greedy, plus = meilleur mais plus lent)

    Returns:
        Texte transcrit
    """
    if not os.path.exists(image_path):
        return "[IMAGE_MANQUANTE]"

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            temperature=0.7 if num_beams > 1 else 1.0,
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
    checkpoint_path: str,
    output_file: str = None,
    model_base_name: str = "microsoft/trocr-large-handwritten",
    transcribe_all: bool = True,
    num_beams: int = 4
) -> List[Dict]:
    """
    Transcrit toutes les lignes d'un split et sauvegarde les résultats.

    Returns:
        Liste des résultats : [{"img_name": ..., "truth": ..., "pred": ...}, ...]
    """
    model, processor, config = load_htr_model(checkpoint_path, model_base_name)
    device = next(model.parameters()).device.type

    samples = load_split_data(image_dir, label_file)
    if not samples:
        return []

    results = []
    print(f"\n🚀 Inférence sur {len(samples)} images...")

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
    checkpoint_path: str,
    output_dir: str = "./evaluation_results",
    model_base_name: str = "microsoft/trocr-large-handwritten",
    num_beams: int = 4
) -> Dict[str, float]:
    """
    Évalue le modèle sur un split (idéalement test) et calcule CER/WER.
    Génère un rapport et un graphe des erreurs.

    Returns:
        {"cer": float, "wer": float, "accuracy": float}
    """
    print(f"\n{'='*60}")
    print(f"   ÉVALUATION FINALE")
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

    # Accuracy au niveau caractère (1 - CER)
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
        "model": model_base_name,
        "checkpoint": checkpoint_path
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
    checkpoint_path: str,
    output_dir: str = "./visualizations",
    num_samples: int = 10,
    model_base_name: str = "microsoft/trocr-large-handwritten"
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
        description="Inférence et évaluation HTR pour manuscrits médiévaux (CATMuS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  # Inférence sur le split dev
  python inference.py --mode infer --split dev

  # Évaluation finale sur le split test (SACRÉ)
  python inference.py --mode evaluate --split test

  # Visualiser 15 prédictions aléatoires
  python inference.py --mode visualize --split dev --num_samples 15

  # Spécifier un checkpoint personnalisé
  python inference.py --mode evaluate --split test --checkpoint ./mon_checkpoint
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
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_production/best_model",
                       help="Chemin vers le checkpoint du meilleur modèle")
    parser.add_argument("--model_base", type=str, default="microsoft/trocr-large-handwritten",
                       help="Modèle de base HuggingFace")
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
        output_file = os.path.join(args.output_dir, f"{args.split}_predictions.txt")
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

        eval_dir = os.path.join(args.output_dir, f"evaluation_{args.split}")
        run_evaluation(
            image_dir=image_dir,
            label_file=label_file,
            checkpoint_path=args.checkpoint,
            output_dir=eval_dir,
            model_base_name=args.model_base,
            num_beams=args.num_beams
        )

    elif args.mode == "visualize":
        viz_dir = os.path.join(args.output_dir, f"visualizations_{args.split}")
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
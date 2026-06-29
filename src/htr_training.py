"""
Module d'entraînement, Fine-Tuning et Grid Search du modèle HTR (TrOCR) avec LoRA
pour manuscrits médiévaux — adapté au format de prepare_dataset.py

Modèle de base recommandé : microsoft/trocr-large-handwritten
Alternative pré-entraînée médiévale : magistermilitum/tridis_HTR
"""

import os
import shutil
import json
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import evaluate
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# ============================================================
# 1. CHARGEMENT DES MÉTRIQUES
# ============================================================
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


# ============================================================
# 2. CALLBACKS PERSONNALISÉS
# ============================================================

class MetricsHistoryCallback(TrainerCallback):
    """Collecte l'historique des métriques CER/WER par epoch pour les graphes."""

    def __init__(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "eval_loss": [],
            "eval_cer": [],
            "eval_wer": [],
            "learning_rate": []
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Détecter si c'est une log d'entraînement ou d'évaluation
        if "loss" in logs and "eval_loss" not in logs:
            self.history["train_loss"].append(logs.get("loss", None))
            self.history["learning_rate"].append(logs.get("learning_rate", None))

        if "eval_loss" in logs:
            self.history["epoch"].append(state.epoch)
            self.history["eval_loss"].append(logs.get("eval_loss", None))
            self.history["eval_cer"].append(logs.get("eval_cer", None))
            self.history["eval_wer"].append(logs.get("eval_wer", None))


class SavePeftAdapterCallback(TrainerCallback):
    """Sauvegarde l'adaptateur LoRA à la fin de chaque epoch."""

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is not None and hasattr(model, "decoder"):
            adapter_dir = os.path.join(args.output_dir, f"adapter_epoch_{int(state.epoch)}")
            os.makedirs(adapter_dir, exist_ok=True)
            if hasattr(model.decoder, 'peft_config'):
                model.decoder.save_pretrained(adapter_dir)
                print(f"  -> Adaptateur LoRA sauvegardé : {adapter_dir}")
        return control


# ============================================================
# 3. DATASET PYTORCH (adapté au format prepare_dataset.py)
# ============================================================

class MedievalHTRDataset(torch.utils.data.Dataset):
    """
    Dataset PyTorch pour charger les images et transcriptions générées par prepare_dataset.py.

    Format attendu des fichiers .txt :
        image_name.png\ttexte transcrit\n
    """

    def __init__(self, image_dir: str, label_file: str, processor: TrOCRProcessor, max_length: int = 64):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.samples: List[Dict[str, str]] = []

        if os.path.exists(label_file):
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "\t" not in line:
                        continue
                    img_name, text = line.split("\t", 1)
                    # Sécurité : ignore les lignes non transcrites
                    if "[TODO_TRANSCRIPTION]" not in text:
                        self.samples.append({"img_name": img_name, "text": text})
            print(f"   {len(self.samples)} échantillons chargés depuis {label_file}")
        else:
            print(f"    Fichier {label_file} non trouvé. Mode démo activé.")
            self.samples = [{"img_name": "demo.png", "text": "In nomine Domini"}] * 4

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample["img_name"])

        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            # Image factice si le fichier n'existe pas
            fake_img = np.random.randint(200, 255, (384, 384, 3), dtype=np.uint8)
            image = Image.fromarray(fake_img)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            sample["text"],
            padding=False,
            max_length=self.max_length,
            truncation=True
        )["input_ids"]

        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        labels = torch.tensor(labels, dtype=torch.long)

        return {"pixel_values": pixel_values, "labels": labels}


# ============================================================
# 4. CONFIGURATION PEFT / LoRA
# ============================================================

def setup_peft_trocr(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float = 0.05) -> tuple:
    """
    Charge TrOCR et injecte LoRA sur le décodeur textuel.

    Pour les manuscrits médiévaux, les meilleurs modèles de base sont :
      - "microsoft/trocr-large-handwritten" (recommandé, meilleur généraliste)
      - "microsoft/trocr-base-handwritten" (plus léger, moins performant)
      - "magistermilitum/tridis_HTR" (déjà fine-tuné médiéval, excellent point de départ)
    """
    print(f"   Chargement du modèle : {model_name}")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Configuration LoRA optimisée pour HTR médiéval
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None
    )

    model.decoder = get_peft_model(model.decoder, peft_config)

    # Afficher les paramètres entraînables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres entraînables : {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model, processor


# ============================================================
# 5. FONCTION DE CALCUL DES MÉTRIQUES
# ============================================================

def build_compute_metrics(processor: TrOCRProcessor):
    """Génère la fonction de calcul du CER et WER."""
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Nettoyer les labels
        labels_ids = np.where(labels_ids != -100, labels_ids, processor.tokenizer.pad_token_id)
        pred_ids = np.where(pred_ids >= 0, pred_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        # Nettoyage des chaînes vides
        pred_str = [p if p.strip() else " " for p in pred_str]
        label_str = [l if l.strip() else " " for l in label_str]

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer, "wer": wer}
    return compute_metrics


# ============================================================
# 6. VISUALISATION DES MÉTRIQUES (GRAPHIQUES)
# ============================================================

def plot_training_curves(history: Dict[str, List], output_dir: str, run_name: str):
    """
    Génère des graphes d'évolution du CER, WER et loss pendant l'entraînement.
    Sauvegarde les figures en PNG dans output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    epochs = history.get("epoch", [])
    if len(epochs) == 0:
        print("    Pas assez de données pour générer les graphes.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Évolution de l'entraînement — {run_name}", fontsize=14, fontweight='bold')

    # 1. CER
    ax = axes[0, 0]
    ax.plot(epochs, history["eval_cer"], 'b-o', linewidth=2, markersize=6, label='CER (validation)')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CER")
    ax.set_title("Character Error Rate (CER)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    best_cer_idx = np.argmin(history["eval_cer"])
    ax.axvline(x=epochs[best_cer_idx], color='r', linestyle='--', alpha=0.5, label=f'Meilleur CER @ epoch {epochs[best_cer_idx]:.1f}')
    ax.legend()

    # 2. WER
    ax = axes[0, 1]
    ax.plot(epochs, history["eval_wer"], 'g-s', linewidth=2, markersize=6, label='WER (validation)')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("WER")
    ax.set_title("Word Error Rate (WER)")
    ax.grid(True, alpha=0.3)
    best_wer_idx = np.argmin(history["eval_wer"])
    ax.axvline(x=epochs[best_wer_idx], color='r', linestyle='--', alpha=0.5, label=f'Meilleur WER @ epoch {epochs[best_wer_idx]:.1f}')
    ax.legend()

    # 3. Loss (train vs eval)
    ax = axes[1, 0]
    ax.plot(epochs, history["eval_loss"], 'r-^', linewidth=2, markersize=6, label='Loss (validation)')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss (Validation)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Résumé comparatif CER vs WER
    ax = axes[1, 1]
    x = np.arange(len(epochs))
    width = 0.35
    ax.bar(x - width/2, history["eval_cer"], width, label='CER', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, history["eval_wer"], width, label='WER', color='seagreen', alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Taux d'erreur")
    ax.set_title("CER vs WER par Epoch")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:.0f}" for e in epochs])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"training_curves_{run_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Graphes sauvegardés : {plot_path}")


def plot_grid_search_comparison(results: List[Dict], output_dir: str):
    """
    Graphe comparatif de tous les runs du grid search (CER et WER).
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comparaison Grid Search — CER & WER", fontsize=14, fontweight='bold')

    labels = [f"LR={r['lr']:.0e}\nr={r['lora_r']}" for r in results]
    x = np.arange(len(labels))

    # CER
    ax = axes[0]
    bars = ax.bar(x, [r['cer'] for r in results], color='steelblue', alpha=0.8, edgecolor='navy')
    ax.set_ylabel("CER")
    ax.set_title("Character Error Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    # Annoter le meilleur
    best_idx = np.argmin([r['cer'] for r in results])
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkorange')
    bars[best_idx].set_linewidth(2)
    ax.annotate('MEILLEUR', xy=(best_idx, results[best_idx]['cer']), 
                xytext=(best_idx, results[best_idx]['cer'] + 0.01),
                ha='center', fontsize=9, color='darkorange', fontweight='bold')

    # WER
    ax = axes[1]
    bars = ax.bar(x, [r['wer'] for r in results], color='seagreen', alpha=0.8, edgecolor='darkgreen')
    ax.set_ylabel("WER")
    ax.set_title("Word Error Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    best_idx = np.argmin([r['wer'] for r in results])
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkorange')
    bars[best_idx].set_linewidth(2)
    ax.annotate('MEILLEUR', xy=(best_idx, results[best_idx]['wer']), 
                xytext=(best_idx, results[best_idx]['wer'] + 0.01),
                ha='center', fontsize=9, color='darkorange', fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "grid_search_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Graphe Grid Search sauvegardé : {plot_path}")


def plot_final_summary(all_histories: Dict[str, Dict], output_dir: str):
    """
    Graphe récapitulatif : courbes CER de tous les runs superposées.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))

    for i, (run_name, history) in enumerate(all_histories.items()):
        epochs = history.get("epoch", [])
        cer = history.get("eval_cer", [])
        if len(epochs) > 0 and len(cer) > 0:
            ax.plot(epochs, cer, '-o', linewidth=2, markersize=6, 
                   label=run_name, color=colors[i])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("CER")
    ax.set_title("Évolution du CER — Tous les runs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "all_runs_cer_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Graphe comparatif global sauvegardé : {plot_path}")


# ============================================================
# 7. DATA COLLATOR PERSONNALISÉ
# ============================================================

def custom_data_collator(features):
    """Colle les features en batch en paddant uniquement les labels."""
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = [f["labels"] for f in features]
    max_len = max(len(l) for l in labels)
    padded_labels = []
    for label in labels:
        padding = [-100] * (max_len - len(label))
        padded_labels.append(label.tolist() + padding)

    return {
        "pixel_values": pixel_values,
        "labels": torch.tensor(padded_labels, dtype=torch.long)
    }


# ============================================================
# 8. PRÉPARATION DES DONNÉES (adapté à prepare_dataset.py)
# ============================================================

def prepare_catmus_data(output_dir="./data/catmus", total_lines=None):
    """
    Extrait des données de CATMuS/medieval en utilisant le split officiel (gen_split).

    Le dataset CATMuS utilise un split manuscrit-aware :
        - train : 90% des manuscrits
        - dev   : 5% des manuscrits  
        - test  : 5% des manuscrits (réservé évaluation finale)

    Args:
        output_dir: Répertoire de sortie
        total_lines: Si spécifié, limite le nombre total de lignes (train+dev)
    """
    print(" Chargement du dataset CATMuS/medieval...")
    ds = load_dataset("CATMuS/medieval", split='train')

    # Séparer selon le split officiel gen_split
    train_data = [row for row in ds if row.get('gen_split') == 'train']
    dev_data = [row for row in ds if row.get('gen_split') == 'dev']
    test_data = [row for row in ds if row.get('gen_split') == 'test']

    print(f"  -> Split officiel : train={len(train_data)} | dev={len(dev_data)} | test={len(test_data)}")

    # Optionnel : limiter le nombre de lignes totales
    if total_lines is not None:
        total_available = len(train_data) + len(dev_data)
        if total_lines < total_available:
            train_ratio = len(train_data) / total_available
            n_train = int(total_lines * train_ratio)
            n_dev = total_lines - n_train
            train_data = train_data[:n_train]
            dev_data = dev_data[:n_dev]
            print(f"  -> Sous-échantillonnage : train={len(train_data)} | dev={len(dev_data)}")

    # Préparer les répertoires
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    def process_split(data, filename):
        lines = []
        for i, row in enumerate(data):
            img_name = f"{filename}_{i:04d}.png"
            img_path = os.path.join(img_dir, img_name)
            row['image'].convert("RGB").save(img_path)
            lines.append(f"{img_name}\t{row['text']}\n")

        with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"  -> {filename}.txt généré avec {len(lines)} lignes.")

    process_split(train_data, "train")
    process_split(dev_data, "dev")

    print(f"\n Dataset prêt dans : {output_dir}")
    print(f"   Images : {img_dir}")
    print(f"     Split 'test' ({len(test_data)} lignes) ignoré — réservé à l'évaluation finale.")
    return img_dir, output_dir


# ============================================================
# 9. ENTRAÎNEMENT UNIQUE (FINE-TUNING RAPIDE)
# ============================================================

def run_single_training(
    model_name: str,
    image_dir: str,
    train_labels: str,
    dev_labels: str,
    output_dir: str,
    epochs: int = 10,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 200,
    metric_for_best: str = "cer",
    early_stopping_patience: int = 4,
) -> Dict[str, Any]:
    """
    Entraînement unique (sans grid search) pour fine-tuner rapidement.

    Args:
        model_name: Modèle de base TrOCR
        image_dir: Répertoire des images de lignes
        train_labels: Fichier train.txt
        dev_labels: Fichier dev.txt
        output_dir: Répertoire de sortie
        epochs: Nombre d'epochs
        lora_r: Rank LoRA
        lora_alpha: Alpha LoRA
        batch_size: Taille du batch
        learning_rate: Learning rate
        max_length: Longueur max des séquences
        metric_for_best: Métrique pour sélectionner le meilleur modèle
        early_stopping_patience: Patience pour l'early stopping

    Returns:
        Dict avec les résultats (cer, wer, config, output_dir)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"   FINE-TUNING TRIDIS — Vieux Français")
    print(f"{'='*60}")
    print(f"  Modèle de base : {model_name}")
    print(f"  Device : {device.upper()}")
    print(f"  LoRA : r={lora_r}, alpha={lora_alpha}")
    print(f"  LR : {learning_rate:.0e} | Epochs : {epochs}")
    print(f"{'='*60}\n")

    # Vérification des fichiers
    if not os.path.exists(train_labels) or not os.path.exists(dev_labels):
        print(" Fichiers de données non trouvés !")
        print(f"   Train : {train_labels}")
        print(f"   Dev   : {dev_labels}")
        return {}

    model, processor = setup_peft_trocr(model_name, lora_r=lora_r, lora_alpha=lora_alpha)
    model.to(device)

    train_dataset = MedievalHTRDataset(image_dir, train_labels, processor, max_length=max_length)
    eval_dataset = MedievalHTRDataset(image_dir, dev_labels, processor, max_length=max_length)

    if len(train_dataset) == 0:
        print(" Dataset d'entraînement vide !")
        return {}

    print(f" Dataset : {len(train_dataset)} train | {len(eval_dataset)} dev")

    metrics_callback = MetricsHistoryCallback()

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        generation_max_length=max_length,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{metric_for_best}",
        greater_is_better=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_data_collator,
        tokenizer=processor,
        compute_metrics=build_compute_metrics(processor),
        callbacks=[
            metrics_callback,
            SavePeftAdapterCallback(),
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        ]
    )

    print("\n Début de l'entraînement...")
    trainer.train()

    eval_results = trainer.evaluate()
    final_cer = eval_results.get("eval_cer", float("inf"))
    final_wer = eval_results.get("eval_wer", float("inf"))

    print(f"\n{'='*60}")
    print(f"   RÉSULTATS FINAUX")
    print(f"{'='*60}")
    print(f"  CER : {final_cer:.4f} ({final_cer*100:.2f}%)")
    print(f"  WER : {final_wer:.4f} ({final_wer*100:.2f}%)")

    # Sauvegarder le meilleur modèle
    final_dest = os.path.join(output_dir, "best_model")
    os.makedirs(final_dest, exist_ok=True)

    if hasattr(model, 'decoder') and hasattr(model.decoder, 'peft_config'):
        model.decoder.save_pretrained(final_dest)
        processor.save_pretrained(final_dest)

        config = {
            "model_name": model_name,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "cer": final_cer,
            "wer": final_wer,
        }
        with open(os.path.join(final_dest, "best_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n Modèle sauvegardé dans : {final_dest}")

    # Graphes
    plot_training_curves(metrics_callback.history, output_dir, "fine_tuning")

    return {
        "cer": final_cer,
        "wer": final_wer,
        "config": config,
        "output_dir": output_dir
    }


# ============================================================
# 10. GRID SEARCH AVEC VISUALISATION
# ============================================================

def run_grid_search(
    base_output_dir: str = "./checkpoints_production",
    image_dir: str = "./data/catmus/images",
    train_labels: str = "./data/catmus/train.txt",
    dev_labels: str = "./data/catmus/dev.txt",
    epochs: int = 10,
    metric_for_best: str = "cer",
    total_lines: int = None,
    model_name: str = "microsoft/trocr-large-handwritten",
    early_stopping_patience: int = 3
) -> None:
    """
    Grid Search avec visualisation des métriques CER/WER.

    Args:
        metric_for_best: "cer" ou "wer" pour sélectionner le meilleur modèle
        total_lines: Limite le nombre de lignes du dataset
        model_name: Modèle TrOCR de base (recommandé: trocr-large-handwritten)
        early_stopping_patience: Arrêt précoce si pas d'amélioration après N epochs
    """
    if metric_for_best not in ["cer", "wer"]:
        raise ValueError("metric_for_best doit être 'cer' ou 'wer'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"   GRID SEARCH HTR — Manuscrits Médiévaux (CATMuS)")
    print(f"  {'='*60}")
    print(f"  Modèle de base : {model_name}")
    print(f"  Device : {device.upper()}")
    print(f"  Métrique de sélection : {metric_for_best.upper()}")
    print(f"  Early stopping patience : {early_stopping_patience}")
    print(f"{'='*60}\n")

    # Vérification des fichiers de données
    if not os.path.exists(train_labels) or not os.path.exists(dev_labels):
        print(" Fichiers de données non trouvés. Préparation automatique...")
        image_dir, _ = prepare_catmus_data(total_lines=total_lines)
        train_labels = os.path.join("./data/catmus", "train.txt")
        dev_labels = os.path.join("./data/catmus", "dev.txt")

    # GRILLE D'HYPERPARAMÈTRES (optimisée pour médiéval)
    grid = {
        "learning_rate": [5e-5, 1e-4],
        "lora_r": [8, 16],
        "lora_alpha": [16, 32]  # alpha = 2*r classiquement
    }

    results: List[Dict[str, Any]] = []
    all_histories: Dict[str, Dict] = {}
    best_score = float("inf")
    best_config = None
    best_model_ref = None
    best_history = None

    total_runs = len(grid["learning_rate"]) * len(grid["lora_r"])
    run_idx = 1

    for lr in grid["learning_rate"]:
        for r in grid["lora_r"]:
            alpha = r * 2
            run_name = f"run_lr{lr:.0e}_r{r}"
            run_dir = os.path.join(base_output_dir, run_name)

            print(f"\n{'─'*60}")
            print(f"  🔬 RUN {run_idx}/{total_runs} : LR={lr:.0e} | LoRA_r={r} | alpha={alpha}")
            print(f"{'─'*60}")

            model, processor = setup_peft_trocr(model_name, lora_r=r, lora_alpha=alpha)
            model.to(device)

            train_dataset = MedievalHTRDataset(image_dir, train_labels, processor)
            eval_dataset = MedievalHTRDataset(image_dir, dev_labels, processor)

            if len(train_dataset) == 0:
                print("   Dataset d'entraînement vide. Arrêt.")
                return

            # Callback pour l'historique des métriques
            metrics_callback = MetricsHistoryCallback()

            training_args = Seq2SeqTrainingArguments(
                output_dir=run_dir,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                predict_with_generate=True,
                generation_max_length=64,
                learning_rate=lr,
                num_train_epochs=epochs,
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model=f"eval_{metric_for_best}",
                greater_is_better=False,
                report_to="none",
                fp16=torch.cuda.is_available(),
                dataloader_num_workers=0,
                remove_unused_columns=False
            )

            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=custom_data_collator,
                tokenizer=processor,
                compute_metrics=build_compute_metrics(processor),
                callbacks=[
                    metrics_callback,
                    SavePeftAdapterCallback(),
                    EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
                ]
            )

            trainer.train()

            eval_results = trainer.evaluate()
            current_cer = eval_results.get("eval_cer", float("inf"))
            current_wer = eval_results.get("eval_wer", float("inf"))
            current_score = eval_results.get(f"eval_{metric_for_best}", float("inf"))

            print(f"\n   Résultats du Run :")
            print(f"     CER = {current_cer:.4f} | WER = {current_wer:.4f}")

            # Sauvegarder l'historique
            all_histories[run_name] = metrics_callback.history

            # Générer les graphes pour ce run
            plot_training_curves(metrics_callback.history, run_dir, run_name)

            results.append({
                "lr": lr,
                "lora_r": r,
                "lora_alpha": alpha,
                "cer": current_cer,
                "wer": current_wer,
                "score": current_score,
                "dir": run_dir,
                "history": metrics_callback.history
            })

            if current_score < best_score:
                best_score = current_score
                best_config = {
                    "lr": lr,
                    "lora_r": r,
                    "lora_alpha": alpha,
                    "cer": current_cer,
                    "wer": current_wer
                }
                best_model_ref = model
                best_history = metrics_callback.history
                print(f"   NOUVEAU MEILLEUR MODÈLE !")

            run_idx += 1

    # ============================================================
    # RAPPORT FINAL
    # ============================================================
    print(f"\n{'='*60}")
    print(f"   TABLEAU RÉCAPITULATIF DE LA GRID SEARCH")
    print(f"{'='*60}")
    print(f"{'LR':<12} | {'LoRA r':<8} | {'Alpha':<8} | {'CER':<10} | {'WER':<10}")
    print("-" * 60)
    for res in results:
        marker = " " if res["score"] == best_score else ""
        print(f"{res['lr']:<12.0e} | {res['lora_r']:<8} | {res['lora_alpha']:<8} | {res['cer']:<10.4f} | {res['wer']:<10.4f}{marker}")
    print(f"{'='*60}")
    print(f"   Meilleur modèle selon {metric_for_best.upper()} : {best_score:.4f}")
    print(f"     Config : LR={best_config['lr']:.0e}, r={best_config['lora_r']}, alpha={best_config['lora_alpha']}")

    # Graphes comparatifs
    plot_grid_search_comparison(results, base_output_dir)
    plot_final_summary(all_histories, base_output_dir)

    # ============================================================
    # DÉPLOIEMENT DU MODÈLE VAINQUEUR
    # ============================================================
    if best_model_ref is not None:
        final_dest = os.path.join(base_output_dir, "best_model")
        os.makedirs(final_dest, exist_ok=True)

        if hasattr(best_model_ref, 'decoder') and hasattr(best_model_ref.decoder, 'peft_config'):
            best_model_ref.decoder.save_pretrained(final_dest)
            # Sauvegarder aussi le processor
            processor.save_pretrained(final_dest)

            # Sauvegarder la config du meilleur modèle
            with open(os.path.join(final_dest, "best_config.json"), "w") as f:
                json.dump(best_config, f, indent=2)

            # Sauvegarder l'historique d'entraînement
            with open(os.path.join(final_dest, "training_history.json"), "w") as f:
                json.dump(best_history, f, indent=2)

            print(f"\n   Adaptateur LoRA déployé dans : {final_dest}")
            print(f"     Processor sauvegardé.")
            print(f"     Config et historique sauvegardés.")

            # Graphe final du meilleur modèle
            plot_training_curves(best_history, final_dest, "BEST_MODEL")
        else:
            print("    Impossible de sauvegarder l'adaptateur LoRA.")

    return results, best_config


# ============================================================
# 11. INTERFACE EN LIGNE DE COMMANDE
# ============================================================

def parse_args():
    """Parse les arguments CLI pour l'entraînement."""
    parser = argparse.ArgumentParser(
        description="Fine-tuning HTR TrOCR avec LoRA pour manuscrits médiévaux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :

  # Grid search par défaut (CATMuS standard)
  python htr_training.py

  # Fine-tuner TRIDIS sur du vieux français (entraînement rapide)
  python htr_training.py \\
      --model_name magistermilitum/tridis_HTR \\
      --image_dir ./data/catmus/images \\
      --train_labels ./data/catmus/train.txt \\
      --dev_labels ./data/catmus/dev.txt \\
      --epochs 10 --lora_r 16 --skip_grid_search

  # Fine-tuner TrOCR-large from scratch
  python htr_training.py \\
      --model_name microsoft/trocr-large-handwritten \\
      --epochs 15 --lora_r 32 --skip_grid_search
        """
    )

    parser.add_argument("--model_name", type=str,
                       default="magistermilitum/tridis_HTR",
                       help="Modèle de base TrOCR (défaut: TRIDIS)")
    parser.add_argument("--image_dir", type=str,
                       default="./data/catmus/images",
                       help="Répertoire des images de lignes")
    parser.add_argument("--train_labels", type=str,
                       default="./data/catmus/train.txt",
                       help="Fichier train.txt (format: image\\ttext)")
    parser.add_argument("--dev_labels", type=str,
                       default="./data/catmus/dev.txt",
                       help="Fichier dev.txt (format: image\\ttext)")
    parser.add_argument("--output_dir", type=str,
                       default="./checkpoints_production",
                       help="Répertoire de sortie des checkpoints")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Nombre d'epochs")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="Rank LoRA")
    parser.add_argument("--lora_alpha", type=int, default=None,
                       help="Alpha LoRA (défaut: 2*r)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Taille du batch")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=64,
                       help="Longueur max des séquences")
    parser.add_argument("--metric_for_best", type=str, default="cer",
                       choices=["cer", "wer"],
                       help="Métrique pour sélectionner le meilleur modèle")
    parser.add_argument("--early_stopping_patience", type=int, default=4,
                       help="Patience early stopping")
    parser.add_argument("--total_lines", type=int, default=None,
                       help="Limiter le nombre de lignes (débogage)")
    parser.add_argument("--skip_grid_search", action="store_true",
                       help="Skip grid search, entraînement unique avec les paramètres donnés")

    return parser.parse_args()


# ============================================================
# 12. SCRIPT PRINCIPAL
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    # Déterminer l'alpha si non spécifié
    lora_alpha = args.lora_alpha if args.lora_alpha else args.lora_r * 2

    if args.skip_grid_search:
        # Entraînement unique (rapide pour fine-tuning)
        run_single_training(
            model_name=args.model_name,
            image_dir=args.image_dir,
            train_labels=args.train_labels,
            dev_labels=args.dev_labels,
            output_dir=args.output_dir,
            epochs=args.epochs,
            lora_r=args.lora_r,
            lora_alpha=lora_alpha,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            metric_for_best=args.metric_for_best,
            early_stopping_patience=args.early_stopping_patience,
        )
    else:
        # Grid search complet
        run_grid_search(
            base_output_dir=args.output_dir,
            image_dir=args.image_dir,
            train_labels=args.train_labels,
            dev_labels=args.dev_labels,
            epochs=args.epochs,
            metric_for_best=args.metric_for_best,
            total_lines=args.total_lines,
            model_name=args.model_name,
            early_stopping_patience=args.early_stopping_patience
        )
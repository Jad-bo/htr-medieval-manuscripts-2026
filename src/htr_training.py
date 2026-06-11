"""Module d'entraînement, Fine-Tuning et Grid Search du modèle HTR (TrOCR) avec LoRA pour manuscrits médiévaux."""

import os
import shutil
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Any
import evaluate
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# 1. CHARGEMENT DES MÉTRIQUES DE RÉFÉRENCE (CER + WER)
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


class MedievalHTRDataset(torch.utils.data.Dataset):
    """Vrai Dataset PyTorch pour charger les images de lignes et leurs transcriptions."""

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
                    # Sécurité : On ignore les lignes non transcrites pour l'entraînement
                    if "[TODO_TRANSCRIPTION]" not in text:
                        self.samples.append({"img_name": img_name, "text": text})
        else:
            print(f" Attention : Le fichier de labels {label_file} n'existe pas. Mode démo activé.")
            self.samples = [{"img_name": "demo.png", "text": "In nomine Domini"}] * 4

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample["img_name"])

        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            fake_img = np.random.randint(200, 255, (384, 384, 3), dtype=np.uint8)
            image = Image.fromarray(fake_img)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            sample["text"],
            padding=False,  # Pas de padding ici, le DataCollator s'en charge par batch
            max_length=self.max_length,
            truncation=True,  # Tronque si dépasse max_length
            return_dict=False
        )["input_ids"]

        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        labels = torch.tensor(labels, dtype=torch.long)

        return {"pixel_values": pixel_values, "labels": labels}


class SavePeftAdapterCallback(TrainerCallback):
    """Callback personnalisé pour sauvegarder uniquement l'adaptateur LoRA (léger, ~Mo)."""
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Sauvegarde l'adaptateur LoRA à la fin de chaque epoch dans un dossier dédié."""
        if model is not None and hasattr(model, "decoder"):
            adapter_dir = os.path.join(args.output_dir, f"adapter_epoch_{int(state.epoch)}.pt")
            # Sauvegarde uniquement les poids LoRA (très léger)
            if hasattr(model.decoder, 'peft_config'):
                torch.save(model.decoder.state_dict(), adapter_dir)
                print(f"  -> Adaptateur LoRA sauvegardé : {adapter_dir}")
        return control


def setup_peft_trocr(model_name: str, lora_r: int, lora_alpha: int) -> tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
    """Charge TrOCR et injecte LoRA de manière stable uniquement sur le décodeur textuel."""
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=None
    )

    model.decoder = get_peft_model(model.decoder, peft_config)
    return model, processor


def build_compute_metrics(processor: TrOCRProcessor):
    """Génère la fonction de calcul du CER et WER adaptée au tokenizer."""
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Nettoyer les labels : remplacer -100 par pad_token_id pour le décodage
        labels_ids = np.where(labels_ids != -100, labels_ids, processor.tokenizer.pad_token_id)

        # Nettoyer les prédictions : s'assurer qu'aucune valeur n'est négative
        # (le modèle ne devrait pas générer de tokens négatifs, mais par sécurité)
        pred_ids = np.where(pred_ids >= 0, pred_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        # Nettoyage des chaînes vides pour éviter une division par zéro
        pred_str = [p if p.strip() else " " for p in pred_str]
        label_str = [l if l.strip() else " " for l in label_str]

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer, "wer": wer}
    return compute_metrics


def prepare_catmus_data(output_dir="./data/catmus", total_lines=None):
    """
    Extrait des données de CATMuS/medieval en utilisant le split officiel (gen_split).

    Le dataset CATMuS utilise un split manuscrit-aware :
        - train : 90% des manuscrits
        - dev   : 5% des manuscrits
        - test  : 5% des manuscrits

    Ici on utilise train pour l'entraînement et dev pour la validation.
    Le split test est ignoré (réservé à l'évaluation finale).

    Args:
        output_dir: Répertoire de sortie pour les images et fichiers texte
        total_lines: Si spécifié, limite le nombre total de lignes (train+dev)
    """
    # 1. Charger le dataset complet
    print("Chargement du dataset CATMuS/medieval...")
    ds = load_dataset("CATMuS/medieval", split='train')

    # 2. Séparer selon le split officiel gen_split
    train_data = [row for row in ds if row.get('gen_split') == 'train']
    dev_data = [row for row in ds if row.get('gen_split') == 'dev']
    test_data = [row for row in ds if row.get('gen_split') == 'test']

    print(f"  -> Split officiel : train={len(train_data)} | dev={len(dev_data)} | test={len(test_data)}")

    # 3. Optionnel : limiter le nombre de lignes totales
    if total_lines is not None:
        # On garde la proportion train/dev
        total_available = len(train_data) + len(dev_data)
        if total_lines < total_available:
            train_ratio = len(train_data) / total_available
            n_train = int(total_lines * train_ratio)
            n_dev = total_lines - n_train
            train_data = train_data[:n_train]
            dev_data = dev_data[:n_dev]
            print(f"  -> Sous-échantillonnage : train={len(train_data)} | dev={len(dev_data)}")

    # 4. Préparer les répertoires
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    def process_split(data, filename):
        lines = []
        for i, row in enumerate(data):
            # Nom de fichier unique basé sur le split
            img_name = f"{filename}_{i:04d}.png"
            img_path = os.path.join(img_dir, img_name)

            # Sauvegarde image
            row['image'].convert("RGB").save(img_path)

            # Préparation ligne pour le fichier texte
            lines.append(f"{img_name}\t{row['text']}\n")

        # Écriture du fichier texte correspondant
        with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"  -> {filename}.txt généré avec {len(lines)} lignes.")

    # 5. Génération des fichiers
    process_split(train_data, "train")
    process_split(dev_data, "val")

    print(f"\nDataset prêt dans : {output_dir}")
    print(f"Images stockées dans : {img_dir}")
    print(f"Note : Le split 'test' ({len(test_data)} lignes) est ignoré (réservé à l'évaluation finale).")
    return img_dir, output_dir


def run_grid_search(
    base_output_dir: str = "./checkpoints_production",
    image_dir: str = "./data/catmus/images",
    train_labels: str = "./data/catmus/train.txt",
    val_labels: str = "./data/catmus/val.txt",
    epochs: int = 3,
    metric_for_best: str = "cer",  # "cer" ou "wer" pour choisir le meilleur modèle
    total_lines: int = None
) -> None:
    """Parcourt une grille d'hyperparamètres pour extraire la configuration HTR optimale.

    Args:
        metric_for_best: Métrique utilisée pour sélectionner le meilleur modèle ("cer" ou "wer")
        total_lines: Limite le nombre total de lignes (train+dev) du dataset CATMuS
    """
    if metric_for_best not in ["cer", "wer"]:
        raise ValueError("metric_for_best doit être 'cer' ou 'wer'")

    model_name = "microsoft/trocr-base-handwritten"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Grid Search initialisée sur l'appareil : {device.upper()}")
    print(f" Métrique de sélection du meilleur modèle : {metric_for_best.upper()}")

    # Vérification des fichiers de données
    if not os.path.exists(train_labels) or not os.path.exists(val_labels):
        print("\n Fichiers de données CATMuS non trouvés. Préparation automatique avec le split officiel...")
        image_dir, _ = prepare_catmus_data(total_lines=total_lines)
        train_labels = os.path.join("./data/catmus", "train.txt")
        val_labels = os.path.join("./data/catmus", "val.txt")

    # 1. DÉFINITION DE LA GRILLE D'HYPERPARAMÈTRES
    grid = {
        "learning_rate": [5e-5, 1e-4],
        "lora_r": [8, 16]
    }

    results: List[Dict[str, Any]] = []
    best_score = float("inf")
    best_config_dir = None
    best_model_ref = None

    total_runs = len(grid["learning_rate"]) * len(grid["lora_r"])
    run_idx = 1

    # 2. BOUCLE EXPLICITE DE GRID SEARCH
    for lr in grid["learning_rate"]:
        for r in grid["lora_r"]:
            alpha = r * 2
            run_name = f"run_lr{lr}_r{r}"
            run_dir = os.path.join(base_output_dir, run_name)

            print(f"--- RUN {run_idx}/{total_runs} : LR={lr} | LoRA_r={r} ---")

            model, processor = setup_peft_trocr(model_name, lora_r=r, lora_alpha=alpha)
            model.to(device)

            train_dataset = MedievalHTRDataset(image_dir, train_labels, processor)
            eval_dataset = MedievalHTRDataset(image_dir, val_labels, processor)

            if len(train_dataset) == 0:
                print(" Erreur : Le dataset d'entraînement est vide.")
                return

            training_args = Seq2SeqTrainingArguments(
                output_dir=run_dir,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                predict_with_generate=True,
                generation_max_length=64,
                learning_rate=lr,
                num_train_epochs=epochs,
                logging_steps=5,
                eval_strategy="epoch",
                save_strategy="no",           # Désactive les checkpoints auto (économise le disque)
                save_total_limit=1,           # Ne garde qu'un checkpoint max
                load_best_model_at_end=False, # Pas besoin, on sauvegarde manuellement
                metric_for_best_model=metric_for_best,
                greater_is_better=False,
                report_to="none",
                fp16=torch.cuda.is_available(),
                dataloader_num_workers=0
            )

            # Data collator personnalisé pour VisionEncoderDecoderModel
            # Les images (pixel_values) ont déjà la même taille via le processor
            # Seuls les labels (textes) ont des tailles variables et nécessitent du padding
            def custom_data_collator(features):
                """Colle les features en batch en paddant uniquement les labels."""
                pixel_values = torch.stack([f["pixel_values"] for f in features])

                # Extraire les labels
                labels = [f["labels"] for f in features]

                # Padding des labels à la taille max du batch
                max_len = max(len(l) for l in labels)
                padded_labels = []
                for label in labels:
                    padding = [-100] * (max_len - len(label))
                    padded_labels.append(label.tolist() + padding)

                batch = {
                    "pixel_values": pixel_values,
                    "labels": torch.tensor(padded_labels, dtype=torch.long)
                }
                return batch

            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=custom_data_collator,  # <-- Collator personnalisé
                processing_class=processor,
                compute_metrics=build_compute_metrics(processor),
                callbacks=[SavePeftAdapterCallback()]
            )

            trainer.train()

            eval_results = trainer.evaluate()
            current_cer = eval_results.get("eval_cer", float("inf"))
            current_wer = eval_results.get("eval_wer", float("inf"))
            current_score = eval_results.get(f"eval_{metric_for_best}", float("inf"))

            print(f" Résultat du Run : CER = {current_cer:.4f} | WER = {current_wer:.4f}")

            results.append({
                "lr": lr,
                "lora_r": r,
                "cer": current_cer,
                "wer": current_wer,
                "score": current_score,
                "dir": run_dir
            })

            if current_score < best_score:
                best_score = current_score
                best_config_dir = run_dir
                best_model_ref = model

            run_idx += 1

    # 3. COMPILATION DU RAPPORT FINAL
    print("=========================================")
    print(" TABLEAU RÉCAPITULATIF DE LA GRID SEARCH")
    print("=========================================")
    print(f"{'Learning Rate':<15} | {'LoRA rank (r)':<15} | {'CER':<10} | {'WER':<10}")
    print("-" * 60)
    for res in results:
        print(f"{res['lr']:<15} | {res['lora_r']:<15} | {res['cer']:<10.4f} | {res['wer']:<10.4f}")
    print("=========================================")
    print(f" Meilleur modèle selon {metric_for_best.upper()} : {best_score:.4f}")

    # ============================================================
    # DÉPLOIEMENT DU MODÈLE VAINQUEUR (allégé - sauvegarde LoRA uniquement)
    # ============================================================
    if best_model_ref is not None:
        final_dest = os.path.join(base_output_dir, "best_model")
        os.makedirs(final_dest, exist_ok=True)

        # Sauvegarde uniquement l'adaptateur LoRA (fichiers légers : ~Mo)
        # Pas le modèle complet (plusieurs Go)
        if hasattr(best_model_ref, 'decoder') and hasattr(best_model_ref.decoder, 'peft_config'):
            best_model_ref.decoder.save_pretrained(final_dest)
            print(f" Adaptateur LoRA déployé dans : {final_dest}")
            print(f"   Taille estimée : ~{sum(p.numel() for p in best_model_ref.decoder.parameters() if p.requires_grad) * 4 / 1024**2:.1f} Mo")
        else:
            print(" Attention : impossible de sauvegarder l'adaptateur LoRA.")


if __name__ == "__main__":
    run_grid_search(
        base_output_dir="./checkpoints_production",
        image_dir="./data/catmus/images",
        train_labels="./data/catmus/train.txt",
        val_labels="./data/catmus/val.txt",
        epochs=8,
        metric_for_best="cer",  # Change en "wer" si tu préfères optimiser le WER
        total_lines=200  # Mettre un entier (ex: 200) pour limiter le dataset
    )
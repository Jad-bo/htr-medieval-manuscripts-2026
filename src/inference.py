"""Module d'inférence pour transcrire les lignes de manuscrits médiévaux."""

import os
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel

def run_htr_inference(
    image_dir: str = "./data/catmus/images",
    label_path: str = "./data/catmus/train.txt",
    checkpoint_path: str = "./checkpoints_production/best_model",
    output_path: str = None,
    model_base_name: str = "microsoft/trocr-base-handwritten"
) -> None:
    """
    Transcription automatique des lignes avec chargement explicite des adaptateurs LoRA.

    Args:
        image_dir: Répertoire contenant les images des lignes
        label_path: Fichier de labels (format: img_name\ttranscription)
        checkpoint_path: Chemin vers le modèle fine-tuné (adaptateurs LoRA)
        output_path: Fichier de sortie (défaut: écrase label_path)
        model_base_name: Modèle de base TrOCR
    """
    if not os.path.exists(label_path):
        print(f"Erreur : Fichier de labels introuvable : {label_path}")
        return

    # Par défaut, écrase le fichier d'entrée si pas de sortie spécifiée
    if output_path is None:
        output_path = label_path

    print(f"Chargement du modèle de base : {model_base_name}...")
    # 1. Chargement explicite du modèle de base
    model = VisionEncoderDecoderModel.from_pretrained(model_base_name)
    processor = TrOCRProcessor.from_pretrained(model_base_name)

    # 2. Greffe propre des adaptateurs LoRA
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print(f"Injection des adaptateurs LoRA depuis : {checkpoint_path}...")
        try:
            model = PeftModel.from_pretrained(model, checkpoint_path)
            model.eval()  # CRUCIAL : force le modèle en mode évaluation
            print("Succès : Adaptateurs LoRA injectés et verrouillés (mode eval).")
        except Exception as e:
            print(f"Erreur lors de l'injection LoRA : {e}")
            return
    else:
        print("Note : Aucun adaptateur LoRA trouvé, utilisation du modèle de base pur.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Lecture du fichier de labels
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_lines = []
    changes_made = 0
    errors = 0

    print(f"\nLancement de l'inférence sur [{device.upper()}]...")
    print(f"Images recherchées dans : {image_dir}\n")

    for line in lines:
        if not line.strip():
            updated_lines.append(line)
            continue

        parts = line.strip().split("\t")

        if len(parts) < 1:
            updated_lines.append(line)
            continue

        filename = parts[0]
        current_transcription = parts[1] if len(parts) > 1 else "[TODO_TRANSCRIPTION]"

        if current_transcription == "[TODO_TRANSCRIPTION]":
            img_path = os.path.join(image_dir, filename)

            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert("RGB")
                    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

                    # Paramètres d'inférence avec contraintes pour le médiéval
                    with torch.no_grad():  # Optimisation mémoire
                        generated_ids = model.generate(
                            pixel_values, 
                            max_new_tokens=64,
                            num_beams=4,
                            early_stopping=True,
                            no_repeat_ngram_size=3,
                            temperature=0.7,  # Légèrement réduit pour plus de stabilité
                            do_sample=True
                        )

                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    print(f" -> {filename} : {generated_text}")
                    updated_lines.append(f"{filename}\t{generated_text}\n")
                    changes_made += 1
                except Exception as e:
                    print(f" Erreur sur {filename} : {e}")
                    updated_lines.append(line)
                    errors += 1
            else:
                print(f" Image manquante : {filename}")
                updated_lines.append(line)
                errors += 1
        else:
            updated_lines.append(line)

    # Écriture des résultats
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)

    print(f"\n{'='*50}")
    print(f"Inférence terminée")
    print(f"{'='*50}")
    print(f"  Lignes transcrites : {changes_made}")
    print(f"  Erreurs / manquants : {errors}")
    print(f"  Fichier de sortie : {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    # Exemple d'utilisation avec les chemins CATMuS par défaut
    run_htr_inference(
        image_dir="./data/catmus/images",
        label_path="./data/catmus/train.txt",
        checkpoint_path="./checkpoints_production/best_model"
    )
import os
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel

def auto_label_dataset(label_path: str, image_dir: str):
    """Parcourt train.txt et remplit les [TODO_TRANSCRIPTION] avec le modèle."""
    model_base = "microsoft/trocr-base-handwritten"
    adapter_path = "./checkpoints_production/best_model"
    
    # Chargement
    processor = TrOCRProcessor.from_pretrained(model_base)
    model = VisionEncoderDecoderModel.from_pretrained(model_base)
    model = PeftModel.from_pretrained(model, adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    print(" Début de l'auto-étiquetage...")
    
    for line in lines:
        if "[TODO_TRANSCRIPTION]" in line:
            filename = line.split("\t")[0].strip()
            img_path = os.path.join(image_dir, filename)
            
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(pixel_values, max_new_tokens=64)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                print(f" {filename} : {text}")
                new_lines.append(f"{filename}\t{text}\n")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    with open(label_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(" Terminé ! Relis les lignes ajoutées et corrige les erreurs.")

if __name__ == "__main__":
    auto_label_dataset("./data/train.txt", "./data/images")
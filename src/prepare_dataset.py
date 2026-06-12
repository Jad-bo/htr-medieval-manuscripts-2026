import os
import random
from datasets import load_dataset
from PIL import Image

def prepare_catmus_data(output_dir="./data/catmus", total_lines=500, train_ratio=0.8):
    """
    Extrait des données de CATMuS/medieval et les sépare en train.txt et val.txt.
    """
    # 1. Charger le dataset (split train par défaut)
    print("Chargement du dataset CATMuS/medieval...")
    ds = load_dataset("CATMuS/medieval", split='train')
    
    # 2. Préparer les répertoires
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # On limite aux 'total_lines' demandées
    subset = ds.take(total_lines)
    
    # Conversion en liste pour mélanger et splitter
    data_list = list(subset)
    random.shuffle(data_list)
    
    split_idx = int(len(data_list) * train_ratio)
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]
    
    def process_split(data, filename):
        lines = []
        for i, row in enumerate(data):
            img_name = f"{filename}_{i:04d}.png"
            img_path = os.path.join(img_dir, img_name)
            
            # Ici on utilise les bonnes colonnes : 'im' pour l'image et 'text' pour le texte
            image = row['im']
            text = row['text']
            
            # Sauvegarde de l'image
            image.convert("RGB").save(img_path)
            
            # Préparation ligne pour le fichier texte
            lines.append(f"{img_name}\t{text}\n")
        
        # Écriture du fichier texte
        with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"-> {filename}.txt généré avec {len(lines)} lignes.")

    # 3. Génération des fichiers
    process_split(train_data, "train")
    process_split(val_data, "val")
            
    print(f"\nDataset prêt dans : {output_dir}")
    print(f"Images stockées dans : {img_dir}")

    # Charger le dataset
    ds = load_dataset("CATMuS/medieval", split='train')

if __name__ == "__main__":
    prepare_catmus_data()
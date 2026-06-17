import os
import random
from datasets import load_dataset
from PIL import Image

def prepare_catmus_data(output_dir="./data/catmus", total_lines=300, 
                          train_ratio=0.90, dev_ratio=0.05, test_ratio=0.05):
    """
    Extrait des données de CATMuS/medieval et les sépare en train/dev/test.

    Splits:
      - train: 90% -> entraînement du modèle HTR
      - dev:   5%  -> validation et sélection du meilleur modèle
      - test:  5%  -> évaluation finale (jamais utilisé en développement)
    """
    # Vérification des ratios
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        "Les ratios doivent sommer à 1.0"

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

    n = len(data_list)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    train_data = data_list[:train_end]
    dev_data = data_list[train_end:dev_end]
    test_data = data_list[dev_end:]

    print(f"\n Répartition des données ({n} lignes au total):")
    print(f"   - train: {len(train_data)} ({len(train_data)/n*100:.1f}%)")
    print(f"   - dev:   {len(dev_data)} ({len(dev_data)/n*100:.1f}%)")
    print(f"   - test:  {len(test_data)} ({len(test_data)/n*100:.1f}%)")

    def process_split(data, filename):
        lines = []
        for i, row in enumerate(data):
            img_name = f"{filename}_{i:04d}.png"
            img_path = os.path.join(img_dir, img_name)

            # Colonnes : 'im' pour l'image et 'text' pour le texte
            image = row['im']
            text = row['text']

            # Sauvegarde de l'image
            image.convert("RGB").save(img_path)

            # Préparation ligne pour le fichier texte
            lines.append(f"{img_name}\t{text}\n")

        # Écriture du fichier texte
        with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"   -> {filename}.txt généré avec {len(lines)} lignes.")

    # 3. Génération des fichiers
    process_split(train_data, "train")
    process_split(dev_data, "dev")
    process_split(test_data, "test")

    print(f"\n Dataset prêt dans : {output_dir}")
    print(f"   Images stockées dans : {img_dir}")
    print(f"\n  IMPORTANT: Le split 'test' est réservé à l'évaluation finale.")
    print(f"   Ne l'utilisez JAMAIS pour l'entraînement ou la validation !")

if __name__ == "__main__":
    prepare_catmus_data()
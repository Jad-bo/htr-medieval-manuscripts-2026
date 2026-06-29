import os
import random
from datasets import load_dataset
from PIL import Image


# ============================================================
# FILTRAGE LINGUISTIQUE
# ============================================================

def is_old_french(text: str) -> bool:
    """
    Détecte si un texte est en vieux français (vs latin médiéval).
    Heuristique basée sur les marqueurs morphologiques et lexicaux.
    """
    if not text:
        return False

    text_lower = text.lower()

    # Marqueurs typiques du vieux français (XIIe–XIVe s.)
    french_markers = [
        'le ', 'la ', 'les ', 'et ', 'ou ', 'ne ', 'pas ', 'qui ', 'que ', 'dont',
        'pour ', 'par ', 'sur ', 'sous ', 'nostre ', 'vostre ', 'mon ', 'ton ', 'son ',
        'mes ', 'tes ', 'ses ', 'faire ', 'donner ', 'tenir ', 'parler ', 'estre ', 'avoir ',
        'faisoit', 'donna', 'donne', 'tenoit', 'parla', 'estoient', 'estoient',
        'chevalier', 'seigneur', 'sire', 'dame', 'roi', 'reine', 'françois', 'france',
        'chascun', 'aucuns', 'toutes', 'fois', 'foiz', 'assises', 'court', 'parlement',
        'bailliage', 'sénéchaussée', 'chastel', 'prouvince', 'conté', 'duché',
        'homme', 'femme', 'enfant', 'jour', 'an', 'temps', 'terre', 'ville', 'champagne',
        'bourg', 'marché', 'foire', 'prix', 'denier', 'livre', 'sou', 'marc',
        'conte', 'duc', 'vicomte', 'châtelain', 'bailli', 'sénéchal', 'prévôt',
        'maire', 'échevin', 'juré', 'sergent', 'messire', 'damoisel', 'escuier',
        'clerc', 'prevost', 'maistre', 'docteur', 'notaire', 'tabellion',
    ]

    # Marqueurs typiques du latin médiéval
    latin_markers = [
        ' in ', ' ad ', ' de ', ' cum ', ' per ', ' pro ', ' sub ', ' inter ', ' super ',
        ' est ', ' sunt ', ' erat ', ' fuit ', ' fuerunt ', ' esse ', ' fuit ',
        ' ego ', ' nos ', ' tu ', ' vos ', ' ille ', ' iste ', ' ipse ', ' idem ',
        ' dominus ', ' domine ', ' deus ', ' christus ', ' amen ', ' anno ', ' millesimo ',
        ' die ', ' mense ', ' indictione ', ' pontificatus ', ' regni ', ' imperii ',
        ' frater ', ' pater ', ' abbas ', ' prior ', ' episcopus ', ' archiepiscopus ',
        ' carta ', ' charta ', ' testamentum ', ' privilegium ', ' bulla ',
        ' sigillum ', ' sigilli ', ' subscripsi ', ' recognovi ', ' tradidit ',
        ' concessit ', ' donavit ', ' assignavit ', ' legavit ', ' confirmavit ',
        ' notum ', ' sciant ', ' presentes ', ' futuri ',
    ]

    french_score = sum(1 for m in french_markers if m in text_lower)
    latin_score = sum(1 for m in latin_markers if m in text_lower)

    # C'est du vieux français si le score français > latin
    # ET présence d'au moins 2 marqueurs français
    return french_score > latin_score and french_score >= 2


def detect_language(text: str) -> str:
    """Retourne 'old_french', 'latin', ou 'unknown'."""
    if not text:
        return 'unknown'

    text_lower = text.lower()

    french_markers = [
        'le ', 'la ', 'les ', 'nostre ', 'vostre ', 'faisoit', 'donna', 'chevalier',
        'seigneur', 'chascun', 'aucuns', 'fois', 'foiz', 'assises', 'parlement',
    ]
    latin_markers = [
        ' in ', ' ad ', ' cum ', ' est ', ' sunt ', ' dominus ', ' anno ', ' millesimo ',
        ' frater ', ' pater ', ' carta ', ' sigillum ', ' subscripsi ',
    ]

    french_score = sum(1 for m in french_markers if m in text_lower)
    latin_score = sum(1 for m in latin_markers if m in text_lower)

    if french_score > latin_score:
        return 'old_french'
    elif latin_score > french_score:
        return 'latin'
    else:
        return 'unknown'


# ============================================================
# PRÉPARATION DU DATASET
# ============================================================

def prepare_catmus_data(
    output_dir="./data/catmus",
    total_lines=400,  # ← MODIFIÉ : 150 images par défaut
    train_ratio=0.90,
    dev_ratio=0.05,
    test_ratio=0.05,
    filter_old_french=False,
):
    """
    Extrait des données de CATMuS/medieval et les sépare en train/dev/test.

    NOUVEAUTÉ :
      → Filtrage optionnel du vieux français (--filter-old-french)

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

    # 2. Filtrage linguistique optionnel
    if filter_old_french:
        print("\n Filtrage linguistique (vieux français uniquement)...")
        ds_filtered = [row for row in ds if is_old_french(row.get('text', ''))]
        print(f"   Total CATMuS     : {len(ds)} lignes")
        print(f"   Vieux français   : {len(ds_filtered)} lignes ({len(ds_filtered)/len(ds)*100:.1f}%)")
        print(f"   Latin/autres     : {len(ds) - len(ds_filtered)} lignes")
        ds = ds_filtered

    # 3. Préparer les répertoires
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # On limite aux 'total_lines' demandées
    subset = ds[:total_lines] if isinstance(ds, list) else ds.take(total_lines)

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

    # 4. Génération des fichiers
    process_split(train_data, "train")
    process_split(dev_data, "dev")
    process_split(test_data, "test")

    print(f"\n Dataset prêt dans : {output_dir}")
    print(f"   Images stockées dans : {img_dir}")
    print(f"\n  IMPORTANT: Le split 'test' est réservé à l'évaluation finale.")
    print(f"   Ne l'utilisez JAMAIS pour l'entraînement ou la validation !")

    # Résumé par langue si filtrage activé
    if filter_old_french:
        lang_counts = {}
        for row in data_list:
            lang = detect_language(row.get('text', ''))
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        print(f"\n Distribution linguistique du dataset :")
        for lang, count in sorted(lang_counts.items()):
            print(f"   - {lang:15} : {count} lignes ({count/n*100:.1f}%)")

    return output_dir, img_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Préparer le dataset CATMuS pour l'entraînement HTR"
    )
    parser.add_argument("--output_dir", type=str, default="./data/catmus",
                       help="Répertoire de sortie")
    parser.add_argument("--total_lines", type=int, default=400,  # ← MODIFIÉ : 150 par défaut
                       help="Nombre total de lignes")
    parser.add_argument("--filter_old_french", action="store_true",
                       help="Filtrer uniquement le vieux français")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed pour la reproductibilité")

    args = parser.parse_args()

    random.seed(args.seed)

    prepare_catmus_data(
        output_dir=args.output_dir,
        total_lines=args.total_lines,
        filter_old_french=args.filter_old_french,
    )
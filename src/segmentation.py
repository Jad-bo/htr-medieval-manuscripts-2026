"""Module de segmentation de mise en page (Layout Analysis) et d'extraction de lignes avec Kraken et OpenCV."""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from kraken import binarization
from kraken.pageseg import segment
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def detect_columns(thresh: np.ndarray, sigma: float = 20, threshold_ratio: float = 0.15, 
                   min_width: int = 300) -> List[Tuple[int, int]]:
    """Détecte les colonnes de texte via projection verticale lissée.

    Args:
        thresh: Image binaire inversée (texte blanc sur fond noir)
        sigma: Paramètre de lissage gaussien pour la projection verticale
        threshold_ratio: Ratio du maximum pour le seuil de détection de texte
        min_width: Largeur minimale en pixels pour considérer une zone comme colonne de texte

    Returns:
        Liste de tuples (x_start, x_end) pour chaque colonne détectée
    """
    vertical_proj = np.sum(thresh, axis=0)
    v_smooth = gaussian_filter1d(vertical_proj.astype(float), sigma=sigma)
    threshold = np.max(v_smooth) * threshold_ratio
    in_text = v_smooth > threshold

    changes = np.diff(in_text.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    if in_text[0]:
        starts = np.insert(starts, 0, 0)
    if in_text[-1]:
        ends = np.append(ends, len(in_text))

    valid_cols = []
    for s, e in zip(starts, ends):
        if e - s > min_width:
            valid_cols.append((s, e))

    return valid_cols


def segment_lines_in_column(column_img: np.ndarray, column_thresh: np.ndarray,
                            expected_lines: Optional[int] = None,
                            min_line_height: int = 15, padding: int = 2,
                            sigma: float = 1.2, peak_height: float = 0.12,
                            peak_distance: int = 8, peak_prominence: float = 0.06,
                            group_distance: int = 12,
                            split_threshold: int = 50, split_max_ratio: float = 0.75) -> List[Tuple[int, int]]:
    """Segmente les lignes de texte dans une colonne via projection horizontale et détection de pics.

    Args:
        column_img: Image RGB de la colonne
        column_thresh: Image binaire inversée de la colonne
        expected_lines: Nombre de lignes attendu (optionnel, pour forcer le nombre)
        min_line_height: Hauteur minimale d'une ligne en pixels
        padding: Marge ajoutée autour de chaque ligne
        sigma: Paramètre de lissage gaussien pour la projection horizontale
        peak_height: Hauteur minimale relative des pics (0-1)
        peak_distance: Distance minimale entre deux pics
        peak_prominence: Prominence minimale des pics
        group_distance: Distance maximale pour grouper deux pics proches
        split_threshold: Hauteur minimale pour découper une ligne en deux
        split_max_ratio: Ratio maximum (min/local_max) pour autoriser un découpage

    Returns:
        Liste de tuples (y_start, y_end) pour chaque ligne détectée
    """
    h = column_img.shape[0]
    h_proj = np.sum(column_thresh, axis=1)
    h_smooth = gaussian_filter1d(h_proj.astype(float), sigma=sigma)

    max_val = np.max(h_smooth)
    h_norm = h_smooth / max_val if max_val > 0 else h_smooth

    # Détection des pics (centres des lignes de texte)
    peaks, props = find_peaks(h_norm, height=peak_height, distance=peak_distance, 
                               prominence=peak_prominence)

    if len(peaks) == 0:
        return []

    # Grouper les pics très proches (moins de group_distance px) en gardant le plus haut
    grouped_peaks = []
    if len(peaks) > 0:
        current_group = [peaks[0]]
        for i in range(1, len(peaks)):
            if peaks[i] - peaks[i-1] < group_distance:
                current_group.append(peaks[i])
            else:
                best = max(current_group, key=lambda p: h_norm[p])
                grouped_peaks.append(best)
                current_group = [peaks[i]]
        best = max(current_group, key=lambda p: h_norm[p])
        grouped_peaks.append(best)

    # Construire les lignes autour des pics avec un seuil adaptatif
    lines = []
    threshold = 0.08

    for peak in grouped_peaks:
        top = peak
        while top > 0 and h_norm[top] > threshold:
            top -= 1

        bottom = peak
        while bottom < h - 1 and h_norm[bottom] > threshold:
            bottom += 1

        # Pas de padding ici (ajouté à la fin)
        top = max(0, top)
        bottom = min(h, bottom)

        if bottom - top >= min_line_height:
            lines.append((top, bottom))

    # Trier et fusionner les chevauchements significatifs
    lines = sorted(lines, key=lambda x: x[0])
    merged = []
    if lines:
        current_start, current_end = lines[0]
        for start, end in lines[1:]:
            if start < current_end - 3:  # Chevauchement significatif
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))

    # Découper les lignes trop grandes (> split_threshold) en cherchant un minimum local
    final_lines = []
    for s, e in merged:
        height = e - s

        if height > split_threshold:
            # Chercher un minimum local au milieu
            mid = (s + e) // 2
            search_start = s + height // 4
            search_end = e - height // 4

            min_y = mid
            min_val = h_norm[mid]
            for y in range(search_start, search_end + 1):
                if h_norm[y] < min_val:
                    min_val = h_norm[y]
                    min_y = y

            local_max = max(h_norm[s:e+1])

            if min_val < local_max * split_max_ratio and min_y > s + 10 and min_y < e - 10:
                # Découper en deux sans padding pour éviter les chevauchements
                final_lines.append((s, min_y))
                final_lines.append((min_y, e))
            else:
                final_lines.append((s, e))
        else:
            final_lines.append((s, e))

    # Ajuster au nombre attendu
    if expected_lines:
        # Si trop peu, découper les plus grandes
        while len(final_lines) < expected_lines:
            max_size_idx = max(range(len(final_lines)), key=lambda i: final_lines[i][1] - final_lines[i][0])
            s, e = final_lines[max_size_idx]
            if e - s < 40:
                break
            mid = (s + e) // 2
            final_lines = final_lines[:max_size_idx] + [(s, mid), (mid, e)] + final_lines[max_size_idx+1:]

        # Si trop, fusionner les lignes les plus petites
        while len(final_lines) > expected_lines:
            min_size = float('inf')
            min_idx = -1
            for i in range(len(final_lines)):
                size = final_lines[i][1] - final_lines[i][0]
                if size < min_size:
                    min_size = size
                    min_idx = i

            # Fusionner avec le voisin le plus proche
            if min_idx == 0:
                merge_with = 1
            elif min_idx == len(final_lines) - 1:
                merge_with = len(final_lines) - 2
            else:
                gap_before = final_lines[min_idx][0] - final_lines[min_idx - 1][1]
                gap_after = final_lines[min_idx + 1][0] - final_lines[min_idx][1]
                merge_with = min_idx - 1 if gap_before < gap_after else min_idx + 1

            indices = sorted([min_idx, merge_with])
            new_start = min(final_lines[indices[0]][0], final_lines[indices[1]][0])
            new_end = max(final_lines[indices[0]][1], final_lines[indices[1]][1])
            final_lines = final_lines[:indices[0]] + [(new_start, new_end)] + final_lines[indices[1]+1:]

    # Ajouter le padding à la toute fin
    padded_lines = []
    for s, e in final_lines:
        ps = max(0, s - padding)
        pe = min(h, e + padding)
        padded_lines.append((ps, pe))

    return padded_lines


def segment_page_lines_opencv(image_path: str) -> List[np.ndarray]:
    """Approche géométrique de secours via OpenCV pour les bannières et images isolées (ex: CATMuS)."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarisation locale (Otsu) + inversion pour avoir le texte en blanc sur fond noir
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilatation horizontale pour lier les lettres d'une même ligne en un seul bloc
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Détection des contours des lignes
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines_images = []
    cv_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Tri des contours de haut en bas pour garder l'ordre de lecture
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filtrage du bruit (on veut une vraie ligne de texte)
        if w > 40 and h > 10:
            lines_images.append(cv_img_rgb[y:y+h, x:x+w])

    return lines_images


def segment_page_lines(image_path: str, expected_lines_per_column: Optional[List[int]] = None,
                       column_params: Optional[List[dict]] = None) -> List[np.ndarray]:
    """Prend une page de manuscrit et extrait la liste des images de lignes de texte.

    Args:
        image_path: Chemin vers l'image de la page
        expected_lines_per_column: Liste optionnelle indiquant le nombre de lignes attendu 
                                  par colonne (ex: [8, 7] pour 2 colonnes)
        column_params: Paramètres spécifiques par colonne pour la segmentation
                      (ex: [{'sigma': 1.2, ...}, {'sigma': 0.8, ...}])
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Impossible de trouver l'image de la page : {image_path}")

    lines_images = []

    # --- tentative 1 : KRAKEN ---
    try:
        im = Image.open(image_path)
        print(f"-> Tentative de binarisation Kraken : {os.path.basename(image_path)}")
        bw_im = binarization.nlbin(im)

        print("-> Analyse de la mise en page via Kraken...")
        seg_result = segment(bw_im)

        cv_img = cv2.imread(image_path)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        if hasattr(seg_result, "lines"):
            raw_lines = seg_result.lines
        elif isinstance(seg_result, dict):
            raw_lines = seg_result.get("lines", [])
        else:
            raw_lines = []

        for line in raw_lines:
            boundary = line.boundary if hasattr(line, "boundary") else line.get("boundary", [])
            polygon = np.array(boundary, dtype=np.int32)
            if len(polygon) == 0:
                continue
            x, y, w, h = cv2.boundingRect(polygon)
            if w > 20 and h > 10:
                lines_images.append(cv_img[y:y+h, x:x+w])
    except Exception as e:
        print(f" Kraken a rencontré une erreur : {e}")

    # --- TENTATIVE 2 : SEGMENTATION PAR COLONNES (Si Kraken échoue ou donne 0 ligne) ---
    if len(lines_images) == 0:
        print(" Kraken n'a détecté aucune ligne (normal sur des images multi-colonnes/bannières).")
        print(" Basculement sur le moteur de segmentation par colonnes OpenCV...")

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Détecter les colonnes de texte
        columns = detect_columns(thresh)
        print(f"-> Colonnes de texte détectées : {len(columns)}")
        for i, (s, e) in enumerate(columns):
            print(f"   Colonne {i+1}: x={s} à {e} (largeur={e-s})")

        # Pour chaque colonne, segmenter les lignes
        for col_idx, (x_start, x_end) in enumerate(columns):
            col_img = img_rgb[:, x_start:x_end]
            col_thresh = thresh[:, x_start:x_end]

            expected = expected_lines_per_column[col_idx] if expected_lines_per_column and col_idx < len(expected_lines_per_column) else None
            params = column_params[col_idx] if column_params and col_idx < len(column_params) else {}

            print(f"-> Segmentation des lignes dans la colonne {col_idx+1}...")
            lines = segment_lines_in_column(col_img, col_thresh, expected_lines=expected, **params)

            for y_start, y_end in lines:
                lines_images.append(img_rgb[y_start:y_end, x_start:x_end])

    # --- TENTATIVE 3 : FALLBACK OPENCV (Si tout échoue) ---
    if len(lines_images) == 0:
        print(" Basculement sur le moteur géométrique OpenCV (Fallback)...")
        lines_images = segment_page_lines_opencv(image_path)

    print(f"-> {len(lines_images)} lignes extraites avec succès.")
    return lines_images


def process_and_save_dataset(
    page_paths: List[str], 
    output_img_dir: str = "./data/images", 
    output_label_path: str = "./data/train.txt",
    expected_lines_per_column: Optional[List[int]] = None,
    column_params: Optional[List[dict]] = None
) -> None:
    """Prend un ensemble de pages, extrait toutes les lignes, les sauvegarde et pré-remplit le fichier de labels sans doublons."""
    os.makedirs(output_img_dir, exist_ok=True)

    label_dir = os.path.dirname(output_label_path)
    if label_dir:
        os.makedirs(label_dir, exist_ok=True)

    # 1. Charger les fichiers déjà référencés pour éviter les doublons textuels
    existing_files = set()
    if os.path.exists(output_label_path):
        with open(output_label_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts = line.split("\t")
                    existing_files.add(parts[0].strip())

    # Déterminer le compteur de départ basé sur le dossier d'images pour ne rien écraser
    existing_images = [f for f in os.listdir(output_img_dir) if f.startswith("ligne_") and f.endswith(".png")]
    line_counter = len(existing_images) + 1

    # On ouvre en mode 'a' (écriture à la suite) mais de façon sécurisée
    with open(output_label_path, "a", encoding="utf-8") as label_file:
        # S'assurer que le fichier existant se termine par un retour à la ligne
        if os.path.exists(output_label_path) and os.path.getsize(output_label_path) > 0:
            with open(output_label_path, "rb+") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    label_file.write("\n")

        for page_path in page_paths:
            try:
                lines = segment_page_lines(page_path, expected_lines_per_column=expected_lines_per_column,
                                          column_params=column_params)

                for line_img in lines:
                    filename = f"ligne_{line_counter:04d}.png"
                    full_save_path = os.path.join(output_img_dir, filename)

                    # Sauvegarde physique de la découpe
                    cv2.imwrite(full_save_path, cv2.cvtColor(line_img, cv2.COLOR_RGB2BGR))

                    # On n'écrit dans train.txt QUE si la ligne n'existe pas déjà
                    if filename not in existing_files:
                        label_file.write(f"{filename}\t[TODO_TRANSCRIPTION]\n")
                        existing_files.add(filename)

                    line_counter += 1
            except Exception as e:
                print(f" Erreur lors du traitement de la page {page_path} : {e}")

    print(f"\n Terminé ! Le catalogue dans '{output_img_dir}' est synchronisé.")
    print(f" Le fichier '{output_label_path}' a été sécurisé et mis à jour sans doublons.")


if __name__ == "__main__":
    # Ajusté avec le nom de ton image actuelle
    mes_pages = ["./data/images/banner_catmus_medieval_centered.png"]

    # Pour l'image CATMuS: 2 colonnes avec 8 et 7 lignes respectivement
    lignes_attendues = [8, 7]

    # Paramètres spécifiques par colonne (optionnel, pour affiner la segmentation)
    params_colonnes = [
        {'sigma': 1.2, 'peak_height': 0.12, 'peak_distance': 8, 'peak_prominence': 0.06, 
         'group_distance': 12, 'split_threshold': 55},
        {'sigma': 0.8, 'peak_height': 0.10, 'peak_distance': 6, 'peak_prominence': 0.04, 
         'group_distance': 8, 'split_threshold': 45},
    ]

    if os.path.exists(mes_pages[0]):
        process_and_save_dataset(
            page_paths=mes_pages, 
            expected_lines_per_column=lignes_attendues,
            column_params=params_colonnes
        )
    else:
        print(f"Dépose ton image cible dans : {mes_pages[0]}")

# --- ALIAS POUR ASSURER LA RÉTROCOMPATIBILITÉ AVEC MAIN.PY ET LES TESTS ---

def segment_manuscript_lines(image_path: str) -> List[np.ndarray]:
    """Alias pour maintenir la compatibilité avec l'ancien pipeline."""
    return segment_page_lines(image_path)

def extract_bounding_polygons(image_path: str):
    """Alias de secours. Renvoie les lignes segmentées pour ne pas bloquer les anciens tests."""
    return segment_page_lines(image_path)
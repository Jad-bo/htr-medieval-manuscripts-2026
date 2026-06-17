"""
Module de prétraitement des images de manuscrits anciens.

Pipeline de prétraitement requis par le brief MD5-2026 :
  1. Correction d'inclinaison (Deskewing)
  2. Égalisation adaptative d'histogramme (CLAHE)
  3. Binarisation adaptative (Sauvola)

Ces étapes améliorent significativement la qualité de la segmentation
et réduisent le CER de 5 à 10 points selon la littérature.
"""

import os
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from typing import Tuple, Optional


# ============================================================
# 1. CORRECTION D'INCLINAISON (DESKEWING)
# ============================================================

def deskew_image(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Corrige l'inclinaison (deskewing) d'une image de manuscrit.

    Utilise la méthode de la boîte englobante minimale sur les pixels
    de texte pour déterminer l'angle de rotation optimal.

    Args:
        image: Image en niveaux de gris (np.ndarray, dtype=uint8)
        max_angle: Angle maximum de correction en degrés (sécurité)

    Returns:
        Image redressée (np.ndarray)

    Raises:
        ValueError: Si l'image est vide ou invalide
    """
    if image is None or image.size == 0:
        raise ValueError("Image vide ou invalide pour le deskewing")

    # Binariser temporairement pour isoler le texte
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Coordonnées des pixels de texte
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return image  # Pas de texte détecté

    # Boîte englobante minimale
    angle = cv2.minAreaRect(coords)[-1]

    # Normaliser l'angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Limiter l'angle de correction
    if abs(angle) > max_angle:
        angle = np.sign(angle) * max_angle

    # Appliquer la rotation
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def detect_skew_angle(image: np.ndarray) -> float:
    """
    Détecte l'angle d'inclinaison sans appliquer la correction.

    Returns:
        Angle en degrés (positif = sens horaire)
    """
    if image is None or image.size == 0:
        return 0.0

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))

    if len(coords) == 0:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    return angle


# ============================================================
# 2. ÉGALISATION ADAPTATIVE (CLAHE)
# ============================================================

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8
) -> np.ndarray:
    """
    Applique l'égalisation adaptative d'histogramme (CLAHE) pour améliorer le contraste.

    Le CLAHE est particulièrement efficace sur les manuscrits dégradés
    où le contraste entre l'encre et le parchemin est faible.

    Args:
        image: Image en niveaux de gris (np.ndarray, dtype=uint8)
        clip_limit: Seuil de clipping (2.0 est un bon compromis)
        grid_size: Taille de la grille de tuiles (8x8 par défaut)

    Returns:
        Image avec contraste amélioré
    """
    # Si l'image est déjà binaire, ne pas appliquer CLAHE
    unique_values = np.unique(image)
    if len(unique_values) <= 2:
        return image

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size)
    )
    return clahe.apply(image)


# ============================================================
# 3. BINARISATION ADAPTATIVE (SAUVOLA)
# ============================================================

def binarize_sauvola(
    image: np.ndarray,
    window_size: int = 25,
    k: float = 0.2
) -> np.ndarray:
    """
    Applique une binarisation adaptative selon la méthode de Sauvola.

    La méthode de Sauvola est optimale pour les documents anciens car
    elle s'adapte localement à l'éclairage et à la dégradation du support.

    Args:
        image: Image en niveaux de gris (np.ndarray, dtype=uint8)
        window_size: Taille de la fenêtre locale (doit être impair)
        k: Paramètre de sensibilité (0.2 = standard, 0.3 = plus permissif)

    Returns:
        Image binarisée (0 = fond, 255 = texte)
    """
    # Si l'image est déjà binaire, la retourner telle quelle
    if len(np.unique(image)) <= 2:
        return image

    # Assurer que window_size est impair
    if window_size % 2 == 0:
        window_size += 1

    # Sauvola retourne un masque booléen : True = texte (fond clair → texte foncé)
    thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=k)

    # Inverser : texte = 255, fond = 0
    binarized = (image <= thresh_sauvola).astype(np.uint8) * 255

    return binarized


# ============================================================
# 4. PIPELINE COMPLET
# ============================================================

def preprocess_pipeline(
    image_path: str,
    output_path: Optional[str] = None,
    deskew: bool = True,
    apply_clahe_step: bool = True,
    binarize: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: int = 8,
    sauvola_window: int = 25,
    sauvola_k: float = 0.2
) -> np.ndarray:
    """
    Pipeline complet de prétraitement d'une image brute de manuscrit.

    Ordre des opérations :
      1. Chargement en niveaux de gris
      2. Deskewing (correction d'inclinaison)
      3. CLAHE (amélioration du contraste)
      4. Binarisation Sauvola

    Args:
        image_path: Chemin d'accès vers l'image source
        output_path: Si spécifié, sauvegarde l'image prétraitée
        deskew: Si True, applique la correction d'inclinaison
        apply_clahe_step: Si True, applique le CLAHE
        binarize: Si True, applique la binarisation
        clahe_clip_limit: Paramètre CLAHE
        clahe_grid_size: Paramètre CLAHE
        sauvola_window: Paramètre Sauvola
        sauvola_k: Paramètre Sauvola

    Returns:
        Image prétraitée (np.ndarray, dtype=uint8)

    Raises:
        FileNotFoundError: Si l'image n'existe pas
        ValueError: Si l'image est corrompue ou invalide
    """
    # 1. Chargement
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

    result = img.copy()

    # 2. Deskewing
    if deskew:
        angle = detect_skew_angle(result)
        if abs(angle) > 0.5:  # Ne corriger que si l'angle est significatif
            result = deskew_image(result)

    # 3. CLAHE
    if apply_clahe_step:
        result = apply_clahe(result, clip_limit=clahe_clip_limit, grid_size=clahe_grid_size)

    # 4. Binarisation
    if binarize:
        result = binarize_sauvola(result, window_size=sauvola_window, k=sauvola_k)

    # Sauvegarde optionnelle
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        cv2.imwrite(output_path, result)

    return result


def preprocess_for_segmentation(image_path: str, output_path: Optional[str] = None) -> np.ndarray:
    """
    Version allégée du prétraitement optimisée pour la segmentation.

    Conserve plus d'information que la version complète (pas de binarisation)
    car les segmenteurs modernes (Kraken BLLA, YOLO) préfèrent les images
    en niveaux de gris ou RGB.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

    result = deskew_image(img)
    result = apply_clahe(result)

    # Convertir en RGB pour les segmenteurs
    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        cv2.imwrite(output_path, result_rgb)

    return result_rgb


# ============================================================
# 5. UTILITAIRES
# ============================================================

def compute_image_stats(image: np.ndarray) -> dict:
    """
    Calcule les statistiques de base d'une image.
    Utile pour le diagnostic et le choix des paramètres de prétraitement.
    """
    return {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "min": int(np.min(image)),
        "max": int(np.max(image)),
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
        "median": float(np.median(image)),
        "is_binary": len(np.unique(image)) <= 2,
        "skew_angle": detect_skew_angle(image) if len(image.shape) == 2 else 0.0
    }


def is_double_page(image: np.ndarray, ratio_threshold: float = 1.2) -> bool:
    """
    Détecte si une image contient une double page.

    Args:
        image: Image (np.ndarray)
        ratio_threshold: Seuil largeur/hauteur (typiquement > 1.2 pour double page)

    Returns:
        True si l'image est probablement une double page
    """
    if len(image.shape) == 3:
        h, w = image.shape[:2]
    else:
        h, w = image.shape

    return (w / h) > ratio_threshold


# ============================================================
# 6. TESTS
# ============================================================

def test_preprocessing():
    """Test rapide du pipeline de prétraitement."""
    print(" Test du prétraitement...")

    # Créer une image de test simulant un manuscrit
    test_img = np.ones((600, 800), dtype=np.uint8) * 240  # Fond clair

    # Ajouter du "texte" simulé (lignes horizontales)
    for i in range(5):
        y = 100 + i * 100
        cv2.line(test_img, (50, y), (750, y), 30, 3)  # Lignes sombres

    # Incliner l'image
    center = (400, 300)
    M = cv2.getRotationMatrix2D(center, 5, 1.0)  # 5 degrés d'inclinaison
    test_img = cv2.warpAffine(test_img, M, (800, 600), borderMode=cv2.BORDER_REPLICATE)

    # Sauvegarder temporairement
    test_path = "/tmp/test_manuscript_raw.png"
    cv2.imwrite(test_path, test_img)

    # Tester le pipeline
    result = preprocess_pipeline(test_path)

    stats = compute_image_stats(result)
    print(f"   Stats : {stats}")

    # Vérifications
    assert result.shape == test_img.shape, "La taille ne doit pas changer"
    assert len(np.unique(result)) <= 2, "L'image doit être binarisée"

    print(" Test prétraitement réussi")
    return True


if __name__ == "__main__":
    test_preprocessing()
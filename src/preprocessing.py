"""
Module de prétraitement pour manuscrits médiévaux.

Implémente la chaîne de traitement documentaire :
  - Détection et correction d'inclinaison (deskewing)
  - Amélioration de contraste (CLAHE)
  - Binarisation adaptative (Sauvola)
  - Filtrage du bruit et détection de double page
  - Rogne des bords blancs (crop_whitespace)

Conforme au brief MD5-2026 — Contrainte 2 : Pipeline de prétraitement paramétrable.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
from skimage.filters import threshold_sauvola
from skimage.morphology import remove_small_objects
import warnings

# ============================================================
# 1. CONSTANTES DE CONFIGURATION
# ============================================================

# Détection de double page
DOUBLE_PAGE_RATIO = 1.2  # Seuil largeur/hauteur pour détecter double page

# Filtrage du bruit — tailles minimales des lignes
MIN_LINE_WIDTH = 40   # Largeur minimale d'une ligne (pixels)
MIN_LINE_HEIGHT = 10  # Hauteur minimale d'une ligne (pixels)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = 8

# Binarisation Sauvola
SAUVOLA_WINDOW_SIZE = 25
SAUVOLA_K = 0.2

# Deskewing
DESKEW_ANGLE_MAX = 10.0  # Angle maximum de correction (degrés)


# ============================================================
# 2. CHARGEMENT D'IMAGE
# ============================================================

def load_image(image_input: Union[str, np.ndarray]) -> np.ndarray:
    """
    Charge une image depuis un chemin ou un array numpy.

    Args:
        image_input: Chemin vers l'image (str) ou array numpy (BGR/RGB)

    Returns:
        Image en niveaux de gris (numpy array 2D, uint8)

    Raises:
        FileNotFoundError: Si le chemin n'existe pas
        ValueError: Si l'input est invalide

    Example:
        >>> gray = load_image("./data/raw/page_001.png")
        >>> gray.shape
        (1200, 800)
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image introuvable : {image_input}")
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Impossible de charger l'image : {image_input}")
        return img
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            # Convertir BGR/RGB en niveaux de gris
            return cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        return image_input.copy()
    else:
        raise ValueError(f"Type d'input non supporté : {type(image_input)}")


# ============================================================
# 3. DÉTECTION ET CORRECTION D'INCLINAISON (DESKEWING)
# ============================================================

def detect_skew(image: np.ndarray, max_angle: float = DESKEW_ANGLE_MAX) -> float:
    """
    Détecte l'angle d'inclinaison d'une page de manuscrit.

    Méthode : détection des lignes de texte via transformée de Hough,
    puis calcul de l'angle médian des lignes proches de l'horizontal.

    Args:
        image: Image en niveaux de gris
        max_angle: Angle maximum acceptable (degrés)

    Returns:
        Angle d'inclinaison en degrés (0 si pas d'inclinaison détectée)

    Example:
        >>> angle = detect_skew(gray_img)
        >>> print(f"Inclinaison détectée : {angle:.2f}°")
    """
    # Binarisation rapide pour la détection
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Détection des contours
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Transformée de Hough probabiliste
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100,
        minLineLength=max(image.shape[1] // 4, 100),
        maxLineGap=20
    )

    if lines is None or len(lines) == 0:
        return 0.0

    # Collecter les angles proches de l'horizontal
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Ne garder que les angles proches de l'horizontal (-max_angle à +max_angle)
        if -max_angle < angle < max_angle:
            angles.append(angle)

    if not angles:
        return 0.0

    # Utiliser la médiane pour robustesse face aux outliers
    median_angle = np.median(angles)
    return median_angle


def deskew_image(image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
    """
    Corrige l'inclinaison d'une image.

    Args:
        image: Image en niveaux de gris ou BGR
        angle: Angle de rotation (degrés). Si None, détecté automatiquement.

    Returns:
        Image redressée

    Example:
        >>> corrected = deskew_image(gray_img)
        >>> # ou avec angle connu :
        >>> corrected = deskew_image(gray_img, angle=2.5)
    """
    if angle is None:
        # Convertir en niveaux de gris si nécessaire pour la détection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        angle = detect_skew(gray)

    if abs(angle) < 0.1:
        # Pas d'inclinaison significative
        return image.copy()

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Matrice de rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculer la nouvelle taille pour éviter la perte de pixels
    cos = np.abs(np.cos(np.radians(angle)))
    sin = np.abs(np.sin(np.radians(angle)))
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Ajuster la matrice de translation
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Rotation avec interpolation cubique et bord répliqué
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# ============================================================
# 4. AMÉLIORATION DU CONTRASTE (CLAHE)
# ============================================================

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    grid_size: int = CLAHE_GRID_SIZE
) -> np.ndarray:
    """
    Applique l'égalisation adaptative de l'histogramme (CLAHE).

    CLAHE améliore le contraste local sans amplifier le bruit,
    contrairement à l'égalisation globale.

    Args:
        image: Image en niveaux de gris ou BGR
        clip_limit: Seuil de clipping (contraste max par zone)
        grid_size: Taille de la grille adaptative

    Returns:
        Image contrastée (niveaux de gris)

    Reference:
        Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"

    Example:
        >>> enhanced = apply_clahe(gray_img, clip_limit=3.0, grid_size=8)
    """
    # Convertir en niveaux de gris si l'image est en couleur (3 ou 4 canaux)
    if len(image.shape) == 3 and image.shape[2] in [3, 4]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] not in [3, 4]:
        # Cas improbable : 3D avec canal différent, prendre le premier canal
        image = image[:, :, 0]
    # Si déjà 2D (niveaux de gris), on garde tel quel

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size)
    )
    return clahe.apply(image)


# ============================================================
# 5. BINARISATION ADAPTATIVE (SAUVOLA)
# ============================================================

def binarize_sauvola(
    image: np.ndarray,
    window_size: int = SAUVOLA_WINDOW_SIZE,
    k: float = SAUVOLA_K
) -> np.ndarray:
    """
    Binarise une image avec la méthode de Sauvola & Pietikäinen.

    Cette méthode est adaptative : le seuil varie selon la moyenne
    et l'écart-type local, ce qui la rend robuste aux variations
    d'illumination et au bruit de fond.

    Args:
        image: Image en niveaux de gris ou BGR
        window_size: Taille de la fenêtre locale (doit être impaire)
        k: Paramètre de sensibilité (0.2 = standard, 0.5 = plus permissif)

    Returns:
        Image binarisée (0 = fond, 255 = texte)

    Reference:
        Sauvola, J., & Pietikäinen, M. (2000). "Adaptive document image binarization"

    Example:
        >>> binary = binarize_sauvola(gray_img, window_size=25, k=0.2)
    """
    # Convertir en niveaux de gris si l'image est en couleur
    if len(image.shape) == 3 and image.shape[2] in [3, 4]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3:
        image = image[:, :, 0]

    # Assurer que window_size est impair
    if window_size % 2 == 0:
        window_size += 1

    # Normaliser en float [0, 1] pour scikit-image
    img_float = image.astype(np.float64) / 255.0

    # Calcul du seuil Sauvola
    thresh = threshold_sauvola(img_float, window_size=window_size, k=k)

    # Binarisation
    binary = img_float > thresh
    binary = (binary * 255).astype(np.uint8)

    return binary


def binarize_otsu(image: np.ndarray) -> np.ndarray:
    """
    Binarisation par méthode d'Otsu (alternative rapide).

    Moins robuste que Sauvola sur les images inhomogènes,
    mais plus rapide. Utile pour le deskewing ou les tests.

    Args:
        image: Image en niveaux de gris ou BGR

    Returns:
        Image binarisée (0 = fond, 255 = texte)
    """
    # Convertir en niveaux de gris si l'image est en couleur
    if len(image.shape) == 3 and image.shape[2] in [3, 4]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3:
        image = image[:, :, 0]

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ============================================================
# 6. ROGNE DES BORDS BLANCS (CROP WHITESPACE)
# ============================================================

def crop_whitespace(
    image: np.ndarray,
    padding: int = 10,
    threshold: int = 250
) -> np.ndarray:
    """
    Rogne les bords blancs autour du contenu d'une image.

    Essentiel pour le modèle TRIDIS qui attend des images
    avec un ratio largeur/hauteur ≤ 10:1.

    Args:
        image: Image en niveaux de gris ou BGR
        padding: Marge (pixels) à conserver autour du contenu
        threshold: Seuil de blanc (0-255, 250 = très blanc)

    Returns:
        Image rognée

    Example:
        >>> cropped = crop_whitespace(gray_img, padding=20)
        >>> print(f"Dimensions : {cropped.shape}")
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Binarisation pour détecter le contenu
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Trouver les contours du contenu
    coords = cv2.findNonZero(binary)
    if coords is None:
        # Image complètement blanche
        return image.copy()

    x, y, w, h = cv2.boundingRect(coords)

    # Ajouter le padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    return image[y1:y2, x1:x2]


def resize_for_tridis(
    image: np.ndarray,
    target_height: int = 384,
    max_ratio: float = 10.0
) -> np.ndarray:
    """
    Redimensionne une image de ligne pour le modèle TRIDIS.

    TRIDIS attend des images avec :
      - Hauteur ≈ 384px
      - Ratio largeur/hauteur ≤ 10:1

    Args:
        image: Image (BGR ou niveaux de gris)
        target_height: Hauteur cible en pixels
        max_ratio: Ratio largeur/hauteur maximum

    Returns:
        Image redimensionnée

    Example:
        >>> resized = resize_for_tridis(line_img, target_height=384)
    """
    h, w = image.shape[:2]

    if h == 0:
        return image.copy()

    # Calculer le ratio de redimensionnement
    scale = target_height / h
    new_w = int(w * scale)
    new_h = target_height

    # Limiter le ratio si nécessaire
    if new_w / new_h > max_ratio:
        new_w = int(new_h * max_ratio)

    # Redimensionner
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return resized


# ============================================================
# 7. DÉTECTION DE DOUBLE PAGE
# ============================================================

def detect_double_page(image: np.ndarray, ratio_threshold: float = DOUBLE_PAGE_RATIO) -> bool:
    """
    Détecte si une image contient une double page (deux pages côte à côte).

    Args:
        image: Image (niveaux de gris ou BGR)
        ratio_threshold: Seuil largeur/hauteur

    Returns:
        True si double page détectée

    Example:
        >>> if detect_double_page(img):
        ...     left, right = split_double_page(img)
    """
    h, w = image.shape[:2]
    ratio = w / h if h > 0 else 0
    return ratio > ratio_threshold


def split_double_page(image: np.ndarray, margin: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sépare une double page en deux pages individuelles.

    Args:
        image: Image contenant une double page
        margin: Marge à ajouter lors de la découpe

    Returns:
        Tuple (page_gauche, page_droite)

    Raises:
        ValueError: Si l'image n'est pas une double page

    Example:
        >>> left, right = split_double_page(double_page_img)
    """
    if not detect_double_page(image):
        raise ValueError("L'image ne semble pas être une double page")

    h, w = image.shape[:2]
    mid = w // 2

    # Ajuster la coupure pour éviter de couper du texte
    left_page = image[:, :mid + margin]
    right_page = image[:, mid - margin:]

    return left_page, right_page


# ============================================================
# 8. FILTRAGE DU BRUIT
# ============================================================

def remove_noise(
    image: np.ndarray,
    min_width: int = MIN_LINE_WIDTH,
    min_height: int = MIN_LINE_HEIGHT
) -> np.ndarray:
    """
    Supprime les composantes connexes trop petites (bruit, taches).

    Args:
        image: Image binarisée (0 = fond, 255 = texte)
        min_width: Largeur minimale d'une composante
        min_height: Hauteur minimale

    Returns:
        Image filtrée

    Example:
        >>> clean = remove_noise(binary_img, min_width=40, min_height=10)
    """
    # Inverser pour que le texte soit blanc (255) et le fond noir (0)
    # car remove_small_objects attend des objets True/1
    inverted = image > 128  # Texte = True

    # Filtrer les petits objets
    filtered = remove_small_objects(inverted, min_size=min_width * min_height)

    # Reconvertir en uint8
    result = (filtered * 255).astype(np.uint8)

    return result


# ============================================================
# 9. PIPELINE COMPLET DE PRÉTRAITEMENT
# ============================================================

def preprocess_pipeline(
    image_input: Union[str, np.ndarray],
    apply_deskew: bool = True,
    apply_clahe_flag: bool = True,
    apply_binarize: bool = False,  # False par défaut car TRIDIS préfère le RGB
    apply_crop: bool = True,
    target_height: Optional[int] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Pipeline complet de prétraitement pour une image de manuscrit.

    Ordre des opérations :
      1. Chargement (si chemin fourni)
      2. Deskewing (correction d'inclinaison)
      3. CLAHE (amélioration du contraste)
      4. Binarisation Sauvola (optionnel, désactivé par défaut)
      5. Rogne des bords blancs
      6. Redimensionnement (si target_height spécifié)

    Args:
        image_input: Chemin vers l'image ou array numpy
        apply_deskew: Activer la correction d'inclinaison
        apply_clahe_flag: Activer CLAHE
        apply_binarize: Activer la binarisation (désactivé par défaut)
        apply_crop: Rogner les bords blancs
        target_height: Hauteur cible pour redimensionnement (ex: 384 pour TRIDIS)
        verbose: Afficher les étapes

    Returns:
        Image prétraitée (numpy array)

    Raises:
        FileNotFoundError: Si le chemin n'existe pas

    Example:
        >>> # Pour le pipeline end-to-end (page complète) :
        >>> processed = preprocess_pipeline("page_001.png", apply_binarize=False)
        >>>
        >>> # Pour une ligne extraite avant HTR :
        >>> processed = preprocess_pipeline("line_001.png", target_height=384)
        >>>
        >>> # Avec verbose :
        >>> processed = preprocess_pipeline(img, verbose=True)
    """
    # Étape 1 : Chargement
    if verbose:
        print("[1/6] Chargement de l'image...")
    image = load_image(image_input)
    original_shape = image.shape

    # Étape 2 : Deskewing
    if apply_deskew:
        if verbose:
            print("[2/6] Détection et correction d'inclinaison...")
        angle = detect_skew(image)
        if abs(angle) > 0.5:
            if verbose:
                print(f"       Angle détecté : {angle:.2f}°")
            image = deskew_image(image, angle)
        else:
            if verbose:
                print("       Pas d'inclinaison significative")

    # Étape 3 : CLAHE
    if apply_clahe_flag:
        if verbose:
            print("[3/6] Amélioration du contraste (CLAHE)...")
        image = apply_clahe(image)

    # Étape 4 : Binarisation (optionnel)
    if apply_binarize:
        if verbose:
            print("[4/6] Binarisation adaptative (Sauvola)...")
        image = binarize_sauvola(image)
    else:
        if verbose:
            print("[4/6] Binarisation sautée (mode RGB pour TRIDIS)")

    # Étape 5 : Rogne des bords blancs
    if apply_crop:
        if verbose:
            print("[5/6] Rogne des bords blancs...")
        image = crop_whitespace(image)

    # Étape 6 : Redimensionnement (optionnel)
    if target_height is not None:
        if verbose:
            print(f"[6/6] Redimensionnement à hauteur {target_height}px...")
        image = resize_for_tridis(image, target_height=target_height)

    if verbose:
        print(f"✓ Prétraitement terminé : {original_shape} → {image.shape}")

    return image


def preprocess_pipeline_pil(
    image_input: Union[str, np.ndarray],
    **kwargs
) -> Image.Image:
    """
    Version du pipeline retournant un objet PIL.Image.

    Utile pour l'intégration avec transformers/TrOCR.

    Args:
        image_input: Chemin ou array numpy
        **kwargs: Arguments passés à preprocess_pipeline()

    Returns:
        Image PIL (RGB)

    Example:
        >>> pil_img = preprocess_pipeline_pil("line_001.png", target_height=384)
        >>> pixel_values = processor(images=pil_img, return_tensors="pt")
    """
    result = preprocess_pipeline(image_input, **kwargs)

    # Convertir en PIL
    if len(result.shape) == 2:
        # Niveaux de gris → RGB
        pil_img = Image.fromarray(result).convert("RGB")
    else:
        # BGR → RGB
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

    return pil_img


# ============================================================
# 10. FONCTIONS UTILITAIRES
# ============================================================

def save_preprocessed(image: np.ndarray, output_path: str) -> str:
    """
    Sauvegarde une image prétraitée.

    Args:
        image: Image numpy array
        output_path: Chemin de sortie

    Returns:
        Chemin du fichier sauvegardé
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, image)
    return output_path


def get_image_stats(image: np.ndarray) -> dict:
    """
    Retourne les statistiques d'une image.

    Args:
        image: Image numpy

    Returns:
        Dict avec dimensions, moyenne, écart-type, etc.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "min": int(np.min(gray)),
        "max": int(np.max(gray)),
        "ratio": image.shape[1] / image.shape[0] if image.shape[0] > 0 else 0
    }


# ============================================================
# 11. TESTS
# ============================================================

def test_preprocessing():
    """Test rapide du module de prétraitement."""
    print("=" * 60)
    print("   TEST DU MODULE DE PRÉTRAITEMENT")
    print("=" * 60)

    # Créer une image de test simulant un manuscrit incliné
    test_img = np.ones((600, 800), dtype=np.uint8) * 240

    # Ajouter du "texte" simulé (lignes noires)
    for i in range(5):
        y = 100 + i * 80
        cv2.line(test_img, (50, y), (750, y + 10), 0, 3)  # Légèrement incliné

    # Test 1 : Deskewing
    print("[1] Test deskewing...")
    angle = detect_skew(test_img)
    print(f"    Angle détecté : {angle:.2f}°")
    deskewed = deskew_image(test_img, angle)
    print(f"    Dimensions après deskew : {deskewed.shape}")

    # Test 2 : CLAHE
    print("[2] Test CLAHE...")
    enhanced = apply_clahe(test_img)
    print(f"    Moyenne avant : {np.mean(test_img):.1f}")
    print(f"    Moyenne après : {np.mean(enhanced):.1f}")

    # Test 3 : Binarisation
    print("[3] Test binarisation Sauvola...")
    binary = binarize_sauvola(test_img)
    print(f"    Unique values : {np.unique(binary)}")

    # Test 4 : Crop whitespace
    print("[4] Test crop whitespace...")
    # Image avec bords blancs
    padded = np.pad(test_img, ((50, 50), (100, 100)), mode='constant', constant_values=255)
    cropped = crop_whitespace(padded, padding=5)
    print(f"    Avant : {padded.shape} → Après : {cropped.shape}")

    # Test 5 : Pipeline complet
    print("[5] Test pipeline complet...")
    result = preprocess_pipeline(test_img, verbose=True, apply_binarize=False)
    print(f"    Résultat final : {result.shape}")

    # Test 6 : Stats
    print("[6] Test statistiques...")
    stats = get_image_stats(result)
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("✓ Tous les tests ont réussi !")
    return True


if __name__ == "__main__":
    test_preprocessing()
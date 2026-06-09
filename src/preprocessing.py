"""Module de prétraitement des images de manuscrits anciens."""

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Corrige l'inclinaison (deskewing) d'une image de manuscrit.

    Args:
        image (np.ndarray): Image en niveaux de gris.

    Returns:
        np.ndarray: Image corrigée.
    """
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Applique l'égalisation adaptative d'histogramme (CLAHE) pour améliorer le contraste.

    Args:
        image (np.ndarray): Image en niveaux de gris.

    Returns:
        np.ndarray: Image avec contraste amélioré.
    """
    # Si l'image est déjà purement binaire (0 ou 255), on n'applique pas la CLAHE
    if len(np.unique(image)) <= 2:
        return image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(image)


def binarize_sauvola(image: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    """Applique une binarisation adaptative selon la méthode de Sauvola.

    Args:
        image (np.ndarray): Image en niveaux de gris.

    Returns:
        np.ndarray: Image binarisée.
    """
    # Sécurité pour les images de test synthétiques ou déjà binaires
    if len(np.unique(image)) <= 2:
        return image
        
    thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=k)
    binarized = (image > thresh_sauvola) * 255
    return binarized.astype(np.uint8)


def preprocess_pipeline(image_path: str) -> np.ndarray:
    """Pipeline complet de prétraitement d'une image brute.

    Args:
        image_path (str): Chemin d'accès vers l'image source.

    Returns:
        np.ndarray: Image finale binarisée et redressée.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")
        
    img_skewed = deskew_image(img)
    img_clahe = apply_clahe(img_skewed)
    img_bin = binarize_sauvola(img_clahe)
    
    return img_bin
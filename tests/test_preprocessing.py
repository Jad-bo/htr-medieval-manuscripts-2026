"""Tests unitaires pour valider les transformations de prétraitement."""

import numpy as np
from src.preprocessing import apply_clahe, binarize_sauvola


def test_preprocessing_shapes_and_ranges():
    """Vérifie que les sorties respectent les types et les plages de valeurs."""
    # Simulation d'une image factice (niveaux de gris, 100x100)
    fake_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # Test de la CLAHE
    clahe_res = apply_clahe(fake_image)
    assert clahe_res.shape == fake_image.shape
    assert clahe_res.dtype == np.uint8
    
    # Test de la binarisation de Sauvola
    bin_res = binarize_sauvola(clahe_res, window_size=15, k=0.2)
    assert bin_res.shape == fake_image.shape
    
    # Vérifie que l'image en sortie est binarisée (uniquement des 0 et des 255)
    assert set(np.unique(bin_res)).issubset({0, 255})
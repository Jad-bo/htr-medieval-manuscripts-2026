"""Tests unitaires pour la boucle d'entraînement et la configuration LoRA."""

import os
from src.htr_training import train_htr_model, setup_peft_trocr


def test_peft_trocr_setup():
    """Vérifie que la configuration LoRA s'applique correctement sur TrOCR."""
    model_name = "microsoft/trocr-base-handwritten"
    model, processor = setup_peft_trocr(model_name)
    
    # On s'assure que le modèle a bien été converti en modèle PEFT
    assert hasattr(model, "base_model")
    # On vérifie que les paramètres entraînables ont été réduits (grâce à LoRA)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Les paramètres entraînables avec LoRA doivent représenter une infime fraction du total
    assert trainable_params < total_params


def test_htr_training_pipeline_execution(tmp_path):
    """S'assure que le chargeur de données et l'entraîneur bouclent correctement."""
    output_dir = os.path.join(tmp_path, "checkpoints_test")
    
    # Ce test exécute une boucle complète à blanc sur 1 époque
    # Si une dimension de tenseur est mauvaise, cela lèvera une exception et échouera ici
    try:
        train_htr_model(output_dir=output_dir, epochs=1)
        execution_success = True
    except Exception as e:
        execution_success = False
        print(f"Erreur détectée pendant la boucle d'entraînement : {e}")
        
    assert execution_success is True
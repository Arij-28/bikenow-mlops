from pathlib import Path

import joblib


def test_production_model_exists():
    model_path = Path("registry/production/model.joblib")
    assert model_path.exists(), "Le modèle de production est introuvable."


def test_production_model_can_be_loaded():
    model_path = Path("registry/production/model.joblib")
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Le modèle chargé ne possède pas de méthode predict."
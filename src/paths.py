from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"


def project_path(relative_path: str | Path) -> Path:
    """Retourne un chemin absolu à partir de la racine du projet."""
    return ROOT_DIR / relative_path


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Charge le fichier config.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_project_dirs() -> None:
    """Crée les dossiers standards du projet si nécessaire."""
    folders = [
        "data/raw",
        "data/processed",
        "data/input",
        "data/output",
        "models",
        "reports/figures",
        "runs",
        "registry/candidate",
        "registry/production",
        "logs",
    ]
    for folder in folders:
        project_path(folder).mkdir(parents=True, exist_ok=True)

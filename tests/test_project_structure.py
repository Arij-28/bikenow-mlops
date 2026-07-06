from pathlib import Path


REQUIRED_PATHS = [
    "data/raw",
    "data/processed",
    "data/input",
    "data/output",
    "models",
    "reports",
    "reports/figures",
    "runs",
    "registry/candidate",
    "registry/production",
    "logs",
    "src",
    "tests",
    "config.yaml",
    "requirements.txt",
    "README.md",
]


def test_required_paths_exist():
    root = Path(__file__).resolve().parents[1]
    missing = [path for path in REQUIRED_PATHS if not (root / path).exists()]
    assert not missing, f"Chemins manquants : {missing}"

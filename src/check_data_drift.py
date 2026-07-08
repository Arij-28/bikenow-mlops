import json
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOG_PATH = PROJECT_ROOT / "reports" / "api_predictions_log.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "data_drift_report.json"


DRIFT_COLUMNS = [
    "hr",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "weathersit",
]


RELATIVE_THRESHOLD = 0.20


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=4, ensure_ascii=False)


def relative_difference(reference: float, current: float) -> float:
    denominator = abs(reference)

    if denominator < 1e-9:
        denominator = 1.0

    return abs(current - reference) / denominator


def main() -> None:
    config = load_config()

    train_path = PROJECT_ROOT / config["data"]["train_path"]

    if not train_path.exists():
        raise FileNotFoundError(
            "Fichier train introuvable. Lancez : python src/prepare_data.py"
        )

    if not LOG_PATH.exists():
        raise FileNotFoundError(
            "Fichier de log introuvable : reports/api_predictions_log.csv. "
            "Appelez d'abord l'endpoint /predict plusieurs fois."
        )

    train_df = pd.read_csv(train_path)
    log_df = pd.read_csv(LOG_PATH)

    log_df = log_df[log_df["status"] == "success"]

    if log_df.empty:
        raise ValueError("Aucune prédiction réussie dans les logs.")

    checks = {}
    drifted_columns = []

    for column in DRIFT_COLUMNS:
        if column not in train_df.columns or column not in log_df.columns:
            checks[column] = {
                "status": "missing_column",
                "drift_detected": None,
            }
            continue

        train_mean = float(train_df[column].mean())
        production_mean = float(log_df[column].mean())
        diff = relative_difference(train_mean, production_mean)

        drift_detected = diff > RELATIVE_THRESHOLD

        if drift_detected:
            drifted_columns.append(column)

        checks[column] = {
            "train_mean": train_mean,
            "production_mean": production_mean,
            "relative_difference": diff,
            "threshold": RELATIVE_THRESHOLD,
            "drift_detected": drift_detected,
        }

    report = {
        "status": "ok",
        "drift_detected": len(drifted_columns) > 0,
        "drifted_columns": drifted_columns,
        "threshold": RELATIVE_THRESHOLD,
        "train_path": config["data"]["train_path"],
        "production_log": "reports/api_predictions_log.csv",
        "checks": checks,
    }

    save_json(OUTPUT_PATH, report)

    print("Analyse du data drift terminée.")
    print("Rapport : reports/data_drift_report.json")
    print(f"Drift détecté : {report['drift_detected']}")

    if drifted_columns:
        print("Variables concernées : " + ", ".join(drifted_columns))
    else:
        print("Aucune variable en drift selon le seuil choisi.")


if __name__ == "__main__":
    main()
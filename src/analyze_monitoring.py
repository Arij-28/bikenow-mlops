import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "reports" / "api_predictions_log.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "monitoring_summary.json"


def save_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=4, ensure_ascii=False)


def main() -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            "Fichier de log introuvable : reports/api_predictions_log.csv. "
            "Appelez d'abord l'endpoint /predict plusieurs fois."
        )

    df = pd.read_csv(LOG_PATH)

    if df.empty:
        raise ValueError("Le fichier de log est vide.")

    success_df = df[df["status"] == "success"]

    summary = {
        "status": "ok",
        "log_path": "reports/api_predictions_log.csv",
        "total_requests": int(len(df)),
        "successful_predictions": int(len(success_df)),
        "errors": int((df["status"] != "success").sum()),
        "mean_prediction": float(success_df["prediction"].mean()),
        "min_prediction": float(success_df["prediction"].min()),
        "max_prediction": float(success_df["prediction"].max()),
        "mean_response_time_ms": float(df["response_time_ms"].mean()),
        "max_response_time_ms": float(df["response_time_ms"].max()),
        "most_common_hour": int(success_df["hr"].mode()[0]),
        "last_prediction_at": str(df["timestamp"].iloc[-1]),
    }

    save_json(OUTPUT_PATH, summary)

    print("Analyse monitoring terminée.")
    print("Rapport : reports/monitoring_summary.json")
    print(f"Nombre total de requêtes : {summary['total_requests']}")
    print(f"Prédictions réussies : {summary['successful_predictions']}")
    print(f"Erreurs : {summary['errors']}")
    print(f"Prédiction moyenne : {summary['mean_prediction']:.2f}")
    print(f"Temps de réponse moyen : {summary['mean_response_time_ms']:.2f} ms")


if __name__ == "__main__":
    main()
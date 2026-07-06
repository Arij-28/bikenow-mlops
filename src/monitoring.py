import json

import pandas as pd

from paths import ensure_project_dirs, load_config, project_path


def main() -> None:
    ensure_project_dirs()
    config = load_config()

    log_path = project_path(config["paths"]["prediction_log"])
    report_path = project_path("reports/monitoring_report.json")

    if not log_path.exists():
        raise FileNotFoundError(
            "Aucun log de prédiction trouvé. Lancez d'abord : python src/predict_batch.py"
        )

    df = pd.read_csv(log_path)

    prediction_col = "prediction_cnt"
    if prediction_col not in df.columns:
        raise ValueError(f"Colonne introuvable dans les logs : {prediction_col}")

    report = {
        "number_of_logged_predictions": int(len(df)),
        "prediction_min": float(df[prediction_col].min()),
        "prediction_max": float(df[prediction_col].max()),
        "prediction_mean": float(df[prediction_col].mean()),
        "prediction_std": float(df[prediction_col].std()),
        "negative_predictions": int((df[prediction_col] < 0).sum()),
        "very_high_predictions_over_1000": int((df[prediction_col] > 1000).sum()),
    }

    report["alert"] = bool(
        report["negative_predictions"] > 0 or report["very_high_predictions_over_1000"] > 0
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("Monitoring terminé.")
    print(f"Rapport : {report_path.relative_to(project_path('.'))}")
    print(f"Alerte : {report['alert']}")


if __name__ == "__main__":
    main()

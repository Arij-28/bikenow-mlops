import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from features import add_cyclical_time_features
from paths import ensure_project_dirs, load_config, project_path


def main() -> None:
    ensure_project_dirs()
    config = load_config()

    model_path = project_path(config["paths"]["registry_production"])
    input_path = project_path(config["data"]["input_path"])
    output_path = project_path(config["data"]["output_path"])
    report_path = project_path("reports/batch_report.json")
    log_path = project_path(config["paths"]["prediction_log"])

    if not model_path.exists():
        raise FileNotFoundError(
            "Modèle production introuvable. Lancez d'abord : python src/promote_model.py"
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier d'entrée introuvable : {input_path}. "
            "Placez un fichier new_hours.csv dans data/input/."
        )

    pipeline = joblib.load(model_path)
    df = pd.read_csv(input_path)
    df_features = add_cyclical_time_features(df)

    target = config["features"]["target"]
    drop_columns = config["features"]["drop_columns"] + [target]
    X = df_features.drop(columns=[col for col in drop_columns if col in df_features.columns])

    predictions = pipeline.predict(X)

    output_df = df.copy()
    output_df["prediction_cnt"] = predictions
    output_df.to_csv(output_path, index=False)

    batch_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "number_of_predictions": int(len(output_df)),
        "prediction_min": float(output_df["prediction_cnt"].min()),
        "prediction_max": float(output_df["prediction_cnt"].max()),
        "prediction_mean": float(output_df["prediction_cnt"].mean()),
    }

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(batch_report, file, indent=4, ensure_ascii=False)

    log_df = output_df.copy()
    log_df["timestamp_utc"] = batch_report["timestamp_utc"]
    log_df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    print("Prédiction batch terminée.")
    print(f"Prédictions : {output_path.relative_to(project_path('.'))}")
    print(f"Rapport : {report_path.relative_to(project_path('.'))}")
    print(f"Log : {log_path.relative_to(project_path('.'))}")


if __name__ == "__main__":
    main()

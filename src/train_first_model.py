import json
import numpy as np

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import build_feature_target
from paths import ensure_project_dirs, load_config, project_path


def main() -> None:
    ensure_project_dirs()
    config = load_config()

    train_path = project_path(config["data"]["train_path"])
    test_path = project_path(config["data"]["test_path"])

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Fichiers train/test introuvables. Lancez d'abord : python src/prepare_data.py"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target = config["features"]["target"]
    drop_columns = config["features"]["drop_columns"]

    X_train, y_train = build_feature_target(train_df, target, drop_columns)
    X_test, y_test = build_feature_target(test_df, target, drop_columns)

    # Pour un premier modèle simple, on garde seulement les colonnes numériques.
    X_train = X_train.select_dtypes(include=["number"])
    X_test = X_test[X_train.columns]

    # Baseline simple : prédire toujours la moyenne du train.
    baseline_prediction = [y_train.mean()] * len(y_test)
    baseline_mae = mean_absolute_error(y_test, baseline_prediction)

    model = RandomForestRegressor(
        n_estimators=int(config["model"]["n_estimators"]),
        max_depth=int(config["model"]["max_depth"]),
        min_samples_leaf=int(config["model"]["min_samples_leaf"]),
        random_state=int(config["project"]["random_state"]),
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    metrics = {
        "model": "RandomForestRegressor_first_model",
        "baseline_mae": float(baseline_mae),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2_score(y_test, predictions)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": list(X_train.columns),
    }

    model_path = project_path("models/first_model.joblib")
    metrics_path = project_path("reports/first_model_metrics.json")

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    print("Premier modèle entraîné.")
    print(f"Modèle : {model_path.relative_to(project_path('.'))}")
    print(f"Métriques : {metrics_path.relative_to(project_path('.'))}")
    print(f"MAE modèle : {metrics['mae']:.2f}")
    print(f"RMSE modèle : {metrics['rmse']:.2f}")
    print(f"R2 modèle : {metrics['r2']:.3f}")
    print(f"MAE baseline : {metrics['baseline_mae']:.2f}")


if __name__ == "__main__":
    main()
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import build_feature_target
from paths import ensure_project_dirs, load_config, project_path


def compute_metrics(y_true, predictions) -> dict:
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)

    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, predictions)),
    }


def evaluate_baseline_global(y_train, y_test) -> dict:
    prediction = np.full(shape=len(y_test), fill_value=y_train.mean())
    metrics = compute_metrics(y_test, prediction)
    metrics["model"] = "baseline_global_mean"
    metrics["model_path"] = ""
    return metrics


def evaluate_baseline_by_hour(train_df, test_df, target: str) -> dict:
    global_mean = train_df[target].mean()

    mean_by_hour = train_df.groupby("hr")[target].mean()

    predictions = test_df["hr"].map(mean_by_hour).fillna(global_mean)

    metrics = compute_metrics(test_df[target], predictions)
    metrics["model"] = "baseline_mean_by_hour"
    metrics["model_path"] = ""
    return metrics


def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = compute_metrics(y_test, predictions)
    metrics["model"] = model_name

    model_path = project_path(f"models/{model_name}.joblib")
    joblib.dump(model, model_path)

    metrics["model_path"] = str(model_path.relative_to(project_path(".")))

    return metrics


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

    # TP3 : on garde une version simple et lisible.
    # Le vrai pipeline sklearn complet sera construit au TP4.
    X_train = X_train.select_dtypes(include=["number"])
    X_test = X_test[X_train.columns]

    results = []

    baseline_global = evaluate_baseline_global(y_train, y_test)
    results.append(baseline_global)

    baseline_hour = evaluate_baseline_by_hour(train_df, test_df, target)
    results.append(baseline_hour)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=int(config["model"]["n_estimators"]),
            max_depth=int(config["model"]["max_depth"]),
            min_samples_leaf=int(config["model"]["min_samples_leaf"]),
            random_state=int(config["project"]["random_state"]),
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=int(config["project"]["random_state"])
        ),
    }

    for model_name, model in models.items():
        metrics = train_and_evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        results.append(metrics)

    comparison_df = pd.DataFrame(results)

    baseline_mae = comparison_df.loc[
        comparison_df["model"] == "baseline_global_mean", "mae"
    ].iloc[0]

    baseline_hour_mae = comparison_df.loc[
        comparison_df["model"] == "baseline_mean_by_hour", "mae"
    ].iloc[0]

    comparison_df["beats_global_baseline"] = comparison_df["mae"] < baseline_mae
    comparison_df["beats_hour_baseline"] = comparison_df["mae"] < baseline_hour_mae

    comparison_df = comparison_df[
        [
            "model",
            "mae",
            "mse",
            "rmse",
            "r2",
            "beats_global_baseline",
            "beats_hour_baseline",
            "model_path",
        ]
    ]

    output_path = project_path("reports/model_comparison.csv")
    comparison_df.to_csv(output_path, index=False)

    json_path = project_path("reports/model_comparison.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("Comparaison des modèles terminée.")
    print(f"Rapport CSV : {output_path.relative_to(project_path('.'))}")
    print(f"Rapport JSON : {json_path.relative_to(project_path('.'))}")
    print()
    print(comparison_df.sort_values("mae")[["model", "mae", "rmse", "r2"]])


if __name__ == "__main__":
    main()
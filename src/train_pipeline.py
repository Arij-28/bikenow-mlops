import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import build_feature_target
from paths import ensure_project_dirs, load_config, project_path


def make_one_hot_encoder():
    """
    Compatible avec plusieurs versions de scikit-learn.
    Les versions récentes utilisent sparse_output.
    Les anciennes versions utilisent sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def compute_metrics(y_true, predictions) -> dict:
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    errors = predictions - y_true
    absolute_errors = np.abs(errors)

    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, predictions)),
        "mean_error": float(errors.mean()),
        "median_absolute_error": float(np.median(absolute_errors)),
        "max_absolute_error": float(absolute_errors.max()),
        "p90_absolute_error": float(np.percentile(absolute_errors, 90)),
    }


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in X_train.columns if column not in numeric_features
    ]

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )


def save_predictions(y_true, predictions) -> None:
    prediction_df = pd.DataFrame(
        {
            "y_true": y_true,
            "prediction": predictions,
            "residual": predictions - y_true,
            "absolute_error": np.abs(predictions - y_true),
        }
    )

    output_path = project_path("reports/pipeline_predictions.csv")
    prediction_df.to_csv(output_path, index=False)


def save_figures(y_true, predictions) -> None:
    figures_dir = project_path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    residuals = predictions - y_true

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, predictions, alpha=0.4)
    plt.title("Prédictions vs valeurs réelles")
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "pipeline_predictions_vs_actual.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=40)
    plt.title("Distribution des erreurs du pipeline")
    plt.xlabel("Erreur de prédiction")
    plt.ylabel("Nombre d'observations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "pipeline_residuals.png")
    plt.close()


def main() -> None:
    ensure_project_dirs()
    config = load_config()

    train_path = project_path(config["data"]["train_path"])
    test_path = project_path(config["data"]["test_path"])

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Fichiers train/test introuvables. Lancez : python src/prepare_data.py"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target = config["features"]["target"]
    drop_columns = config["features"]["drop_columns"]

    X_train, y_train = build_feature_target(train_df, target, drop_columns)
    X_test, y_test = build_feature_target(test_df, target, drop_columns)

    preprocessor = build_preprocessor(X_train)

    model = HistGradientBoostingRegressor(
        random_state=int(config["project"]["random_state"])
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = compute_metrics(y_test, predictions)

    model_path = project_path("models/sklearn_pipeline.joblib")
    joblib.dump(pipeline, model_path)

    metrics_path = project_path("reports/pipeline_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    metrics_csv_path = project_path("reports/pipeline_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)

    save_predictions(y_test, predictions)
    save_figures(y_test, predictions)

    print("Pipeline sklearn entraîné avec succès.")
    print(f"Pipeline : {model_path.relative_to(project_path('.'))}")
    print(f"Métriques JSON : {metrics_path.relative_to(project_path('.'))}")
    print(f"Métriques CSV : {metrics_csv_path.relative_to(project_path('.'))}")
    print("Prédictions : reports/pipeline_predictions.csv")
    print("Figures : reports/figures/")
    print()
    print(f"MAE : {metrics['mae']:.2f}")
    print(f"RMSE : {metrics['rmse']:.2f}")
    print(f"R2 : {metrics['r2']:.3f}")
    print(f"Erreur médiane absolue : {metrics['median_absolute_error']:.2f}")
    print(f"Erreur absolue P90 : {metrics['p90_absolute_error']:.2f}")


if __name__ == "__main__":
    main()
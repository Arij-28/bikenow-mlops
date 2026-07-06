import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from features import build_feature_target
from paths import ensure_project_dirs, load_config, project_path
from train_pipeline import build_preprocessor, compute_metrics, save_figures, save_predictions


def setup_mlflow() -> None:
    """
    Configure MLflow avec un backend SQLite.

    Pourquoi SQLite ?
    Les versions récentes de MLflow déconseillent le backend fichier.
    SQLite est plus propre pour stocker les expériences localement.
    """
    mlflow_db = project_path("runs/mlflow.db")
    artifact_dir = project_path("runs/mlflow_artifacts")

    mlflow_db.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{mlflow_db.resolve().as_posix()}"
    artifact_uri = artifact_dir.resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "BikeNow-TP5-MLflow"

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_uri,
        )

    mlflow.set_experiment(experiment_name)


def main() -> None:
    ensure_project_dirs()
    config = load_config()

    setup_mlflow()

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

    params = {
        "model_type": "HistGradientBoostingRegressor",
        "target": target,
        "random_state": int(config["project"]["random_state"]),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "features_count": X_train.shape[1],
    }

    with mlflow.start_run(run_name="tp5_sklearn_pipeline_tracking") as run:
        mlflow.log_params(params)

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = compute_metrics(y_test, predictions)

        mlflow.log_metrics(metrics)

        model_path = project_path("models/mlflow_pipeline.joblib")
        joblib.dump(pipeline, model_path)

        metrics_path = project_path("reports/mlflow_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4, ensure_ascii=False)

        run_summary_path = project_path("reports/mlflow_last_run.json")
        run_summary = {
            "run_id": run.info.run_id,
            "experiment_name": "BikeNow-TP5-MLflow",
            "tracking_uri": mlflow.get_tracking_uri(),
            "params": params,
            "metrics": metrics,
        }

        with open(run_summary_path, "w", encoding="utf-8") as file:
            json.dump(run_summary, file, indent=4, ensure_ascii=False)

        save_predictions(y_test, predictions)
        save_figures(y_test, predictions)

        #mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")
        mlflow.log_artifact(str(run_summary_path), artifact_path="reports")
        mlflow.log_artifact(
            str(project_path("reports/pipeline_predictions.csv")),
            artifact_path="reports",
        )

        figures_dir = project_path("reports/figures")
        for figure_path in figures_dir.glob("pipeline_*.png"):
            mlflow.log_artifact(str(figure_path), artifact_path="figures")

        print("Expérience MLflow terminée.")
        print(f"Run ID : {run.info.run_id}")
        print(f"Tracking URI : {mlflow.get_tracking_uri()}")
        print("Base SQLite : runs/mlflow.db")
        print("Artefacts MLflow : runs/mlflow_artifacts")
        print("Modèle local : models/mlflow_pipeline.joblib")
        print("Métriques : reports/mlflow_metrics.json")
        print("Résumé du run : reports/mlflow_last_run.json")
        print()
        print(f"MAE : {metrics['mae']:.2f}")
        print(f"RMSE : {metrics['rmse']:.2f}")
        print(f"R2 : {metrics['r2']:.3f}")


if __name__ == "__main__":
    main()
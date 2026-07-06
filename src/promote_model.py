import json
import shutil
from datetime import datetime, timezone

from paths import ensure_project_dirs, project_path


THRESHOLDS = {
    "max_mae": 60.0,
    "max_rmse": 90.0,
    "min_r2": 0.85,
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def validate_metrics(metrics):
    errors = []

    mae = float(metrics["mae"])
    rmse = float(metrics["rmse"])
    r2 = float(metrics["r2"])

    if mae > THRESHOLDS["max_mae"]:
        errors.append(
            f"MAE trop élevée : {mae:.2f} > {THRESHOLDS['max_mae']:.2f}"
        )

    if rmse > THRESHOLDS["max_rmse"]:
        errors.append(
            f"RMSE trop élevée : {rmse:.2f} > {THRESHOLDS['max_rmse']:.2f}"
        )

    if r2 < THRESHOLDS["min_r2"]:
        errors.append(
            f"R2 trop faible : {r2:.3f} < {THRESHOLDS['min_r2']:.3f}"
        )

    if errors:
        return "FAILED", errors

    return "PASSED", []


def copy_model_package(source_model_path, destination_dir, metrics, status, run_summary):
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination_model_path = destination_dir / "model.joblib"
    shutil.copy2(source_model_path, destination_model_path)

    model_card = {
        "model_name": "BikeNow demand forecasting pipeline",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model": str(source_model_path.relative_to(project_path("."))),
        "model_path": str(destination_model_path.relative_to(project_path("."))),
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "run_id": run_summary.get("run_id"),
        "experiment_name": run_summary.get("experiment_name"),
        "tracking_uri": run_summary.get("tracking_uri"),
    }

    save_json(destination_dir / "model_card.json", model_card)

    return model_card


def main():
    ensure_project_dirs()

    source_model_path = project_path("models/mlflow_pipeline.joblib")
    metrics_path = project_path("reports/mlflow_metrics.json")
    run_summary_path = project_path("reports/mlflow_last_run.json")

    if not source_model_path.exists():
        raise FileNotFoundError(
            "Modèle introuvable : models/mlflow_pipeline.joblib. Lancez d'abord le TP5."
        )

    if not metrics_path.exists():
        raise FileNotFoundError(
            "Métriques introuvables : reports/mlflow_metrics.json. Lancez d'abord le TP5."
        )

    metrics = load_json(metrics_path)

    if run_summary_path.exists():
        run_summary = load_json(run_summary_path)
    else:
        run_summary = {}

    validation_status, validation_errors = validate_metrics(metrics)

    registry_dir = project_path("registry")
    candidate_dir = registry_dir / "candidate"
    production_dir = registry_dir / "production"

    candidate_card = copy_model_package(
        source_model_path=source_model_path,
        destination_dir=candidate_dir,
        metrics=metrics,
        status="CANDIDATE_" + validation_status,
        run_summary=run_summary,
    )

    if validation_status == "PASSED":
        production_card = copy_model_package(
            source_model_path=source_model_path,
            destination_dir=production_dir,
            metrics=metrics,
            status="PRODUCTION",
            run_summary=run_summary,
        )
        decision = "PROMOTED_TO_PRODUCTION"
    else:
        production_card = None
        decision = "REJECTED"

    promotion_report = {
        "decision": decision,
        "validation_status": validation_status,
        "validation_errors": validation_errors,
        "thresholds": THRESHOLDS,
        "candidate": candidate_card,
        "production": production_card,
    }

    report_path = project_path("reports/registry_promotion_report.json")
    save_json(report_path, promotion_report)

    print("Validation du modèle terminée.")
    print(f"Statut validation : {validation_status}")
    print(f"Décision : {decision}")
    print(f"Rapport : {report_path.relative_to(project_path('.'))}")

    print()
    print(f"MAE : {metrics['mae']:.2f}")
    print(f"RMSE : {metrics['rmse']:.2f}")
    print(f"R2 : {metrics['r2']:.3f}")

    if validation_errors:
        print()
        print("Erreurs de validation :")
        for error in validation_errors:
            print(f"- {error}")

    if decision == "PROMOTED_TO_PRODUCTION":
        print()
        print("Modèle promu en production.")
        print("Production : registry/production/model.joblib")
    else:
        print()
        print("Modèle non promu en production.")


if __name__ == "__main__":
    main()
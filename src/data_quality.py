from pathlib import Path
import json

import pandas as pd

from paths import ensure_project_dirs, load_config, project_path

EXPECTED_COLUMNS = [
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
]


def build_quality_report(data_path: Path) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {data_path}. "
            "Placez hour.csv dans data/raw/."
        )

    df = pd.read_csv(data_path)

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    unexpected_columns = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    report = {
        "dataset_path": str(data_path),
        "number_of_rows": int(df.shape[0]),
        "number_of_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "missing_values": {col: int(value) for col, value in df.isna().sum().items()},
        "duplicated_rows": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "target_min": float(df["cnt"].min()) if "cnt" in df.columns else None,
        "target_max": float(df["cnt"].max()) if "cnt" in df.columns else None,
        "target_mean": float(df["cnt"].mean()) if "cnt" in df.columns else None,
    }

    return report


def main() -> None:
    ensure_project_dirs()
    config = load_config()

    data_path = project_path(config["data"]["raw_path"])
    report_path = project_path("reports/data_quality.json")

    report = build_quality_report(data_path)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("Contrôle qualité terminé.")
    print(f"Rapport : {report_path.relative_to(project_path('.'))}")
    print(f"Lignes : {report['number_of_rows']}")
    print(f"Colonnes manquantes : {report['missing_columns']}")
    print(f"Doublons : {report['duplicated_rows']}")


if __name__ == "__main__":
    main()

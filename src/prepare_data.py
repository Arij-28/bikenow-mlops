import json

import pandas as pd

from paths import ensure_project_dirs, load_config, project_path


def temporal_split(df: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["dteday"] = pd.to_datetime(df["dteday"])
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)

    split_index = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def main() -> None:
    ensure_project_dirs()
    config = load_config()

    raw_path = project_path(config["data"]["raw_path"])
    train_path = project_path(config["data"]["train_path"])
    test_path = project_path(config["data"]["test_path"])
    report_path = project_path("reports/split_summary.json")
    test_ratio = float(config["data"]["test_ratio"])

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {raw_path}. "
            "Placez hour.csv dans data/raw/."
        )

    df = pd.read_csv(raw_path)
    train_df, test_df = temporal_split(df, test_ratio)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    split_report = {
        "total_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_ratio": test_ratio,
        "train_start_date": str(pd.to_datetime(train_df["dteday"]).min().date()),
        "train_end_date": str(pd.to_datetime(train_df["dteday"]).max().date()),
        "test_start_date": str(pd.to_datetime(test_df["dteday"]).min().date()),
        "test_end_date": str(pd.to_datetime(test_df["dteday"]).max().date()),
        "temporal_leakage_check": bool(train_df["instant"].max() < test_df["instant"].min()),
    }

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(split_report, file, indent=4, ensure_ascii=False)

    print("Préparation terminée.")
    print(f"Train : {train_path.relative_to(project_path('.'))}")
    print(f"Test  : {test_path.relative_to(project_path('.'))}")
    print(f"Rapport : {report_path.relative_to(project_path('.'))}")


if __name__ == "__main__":
    main()

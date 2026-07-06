from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paths import ensure_project_dirs, load_config, project_path


def main() -> None:
    ensure_project_dirs()
    config = load_config()

    data_path = project_path(config["data"]["raw_path"])
    reports_dir = project_path("reports")
    figures_dir = project_path("reports/figures")

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {data_path}. Placez hour.csv dans data/raw/."
        )

    df = pd.read_csv(data_path)

    eda_summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "target_mean",
                "target_min",
                "target_max",
                "missing_values_total",
                "duplicated_rows",
            ],
            "value": [
                df.shape[0],
                df.shape[1],
                df["cnt"].mean(),
                df["cnt"].min(),
                df["cnt"].max(),
                df.isna().sum().sum(),
                df.duplicated().sum(),
            ],
        }
    )

    eda_summary.to_csv(reports_dir / "eda_summary.csv", index=False)

    demand_by_hour = (
        df.groupby("hr")["cnt"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )
    demand_by_hour.to_csv(reports_dir / "demand_by_hour.csv", index=False)

    demand_by_season = (
        df.groupby("season")["cnt"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )
    demand_by_season.to_csv(reports_dir / "demand_by_season.csv", index=False)

    demand_by_weather = (
        df.groupby("weathersit")["cnt"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )
    demand_by_weather.to_csv(reports_dir / "demand_by_weather.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(demand_by_hour["hr"], demand_by_hour["mean"], marker="o")
    plt.title("Demande moyenne de vélos par heure")
    plt.xlabel("Heure")
    plt.ylabel("Nombre moyen de locations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "demand_by_hour.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(demand_by_season["season"], demand_by_season["mean"])
    plt.title("Demande moyenne de vélos par saison")
    plt.xlabel("Saison")
    plt.ylabel("Nombre moyen de locations")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(figures_dir / "demand_by_season.png")
    plt.close()

    print("EDA terminée.")
    print("Résumé : reports/eda_summary.csv")
    print("Demande par heure : reports/demand_by_hour.csv")
    print("Demande par saison : reports/demand_by_season.csv")
    print("Demande par météo : reports/demand_by_weather.csv")
    print("Figures : reports/figures/")


if __name__ == "__main__":
    main()
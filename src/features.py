import numpy as np
import pandas as pd


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des variables cycliques pour l'heure et le mois."""
    df = df.copy()

    if "hr" in df.columns:
        df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
        df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)

    if "mnth" in df.columns:
        df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
        df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)

    return df


def build_feature_target(
    df: pd.DataFrame,
    target: str,
    drop_columns: list[str],
    use_cyclical_features: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Construit X et y à partir d'un DataFrame BikeNow."""
    df = df.copy()

    if use_cyclical_features:
        df = add_cyclical_time_features(df)

    if target not in df.columns:
        raise ValueError(f"Target introuvable : {target}")

    y = df[target]
    columns_to_drop = [col for col in drop_columns + [target] if col in df.columns]
    X = df.drop(columns=columns_to_drop)

    return X, y

import joblib
import pandas as pd

from features import build_feature_target
from paths import load_config, project_path


def main():
    config = load_config()

    model_path = project_path("registry/production/model.joblib")
    test_path = project_path(config["data"]["test_path"])

    if not model_path.exists():
        raise FileNotFoundError(
            "Modèle de production introuvable : registry/production/model.joblib"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            "Fichier de test introuvable. Lancez d'abord : python src/prepare_data.py"
        )

    model = joblib.load(model_path)

    test_df = pd.read_csv(test_path).head(5)

    target = config["features"]["target"]
    drop_columns = config["features"]["drop_columns"]

    X_test, y_test = build_feature_target(test_df, target, drop_columns)

    predictions = model.predict(X_test)

    result = pd.DataFrame(
        {
            "real_value": y_test.values,
            "prediction": predictions,
            "absolute_error": abs(predictions - y_test.values),
        }
    )

    output_path = project_path("reports/registry_smoke_test.csv")
    result.to_csv(output_path, index=False)

    print("Test du modèle de production réussi.")
    print("Modèle chargé depuis : registry/production/model.joblib")
    print("Rapport : reports/registry_smoke_test.csv")
    print()
    print(result)


if __name__ == "__main__":
    main()
import csv
import os
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "registry" / "production" / "model.joblib"
LOG_PATH = PROJECT_ROOT / "reports" / "api_predictions_log.csv"


DEFAULT_FEATURE_COLUMNS = [
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
]


class BikeDemandInput(BaseModel):
    season: int
    yr: int
    mnth: int
    hr: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int
    temp: float
    atemp: float
    hum: float
    windspeed: float

    instant: Optional[int] = 0
    dteday: Optional[str] = "2012-06-01"
    casual: Optional[int] = 0
    registered: Optional[int] = 0


class PredictionOutput(BaseModel):
    prediction: float
    model_path: str
    status: str


@lru_cache
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modèle de production introuvable : registry/production/model.joblib"
        )

    return joblib.load(MODEL_PATH)


def to_dict(input_data: BikeDemandInput) -> dict:
    if hasattr(input_data, "model_dump"):
        return input_data.model_dump()

    return input_data.dict()


def get_expected_columns(model) -> list:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return DEFAULT_FEATURE_COLUMNS


def check_api_key(api_key: Optional[str]) -> None:
    expected_key = os.getenv("BIKENOW_API_KEY")

    if not expected_key:
        return

    if api_key != expected_key:
        raise HTTPException(
            status_code=403,
            detail="Clé API invalide ou manquante.",
        )


def append_prediction_log(
    row: dict,
    prediction: float,
    status: str,
    response_time_ms: float,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = LOG_PATH.exists()

    fieldnames = [
        "timestamp",
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
        "prediction",
        "status",
        "response_time_ms",
    ]

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "season": row.get("season"),
        "yr": row.get("yr"),
        "mnth": row.get("mnth"),
        "hr": row.get("hr"),
        "holiday": row.get("holiday"),
        "weekday": row.get("weekday"),
        "workingday": row.get("workingday"),
        "weathersit": row.get("weathersit"),
        "temp": row.get("temp"),
        "atemp": row.get("atemp"),
        "hum": row.get("hum"),
        "windspeed": row.get("windspeed"),
        "prediction": prediction,
        "status": status,
        "response_time_ms": round(response_time_ms, 2),
    }

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(log_row)


app = FastAPI(
    title="BikeNow Demand Forecasting API",
    description="API de prédiction de la demande horaire de vélos avec monitoring.",
    version="2.0.0",
)


@app.get("/")
def root():
    return {
        "message": "BikeNow API is running",
        "model": "registry/production/model.joblib",
        "docs": "/docs",
        "monitoring": "/monitoring-summary",
    }


@app.get("/health")
def health():
    model_exists = MODEL_PATH.exists()

    return {
        "status": "ok" if model_exists else "model_missing",
        "model_path": "registry/production/model.joblib",
        "model_exists": model_exists,
        "monitoring_log": "reports/api_predictions_log.csv",
        "api_key_enabled": bool(os.getenv("BIKENOW_API_KEY")),
    }


@app.get("/model-info")
def model_info():
    try:
        model = load_model()
        expected_columns = get_expected_columns(model)

        return {
            "status": "ok",
            "model_path": "registry/production/model.joblib",
            "expected_columns": expected_columns,
            "expected_columns_count": len(expected_columns),
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/monitoring-summary")
def monitoring_summary():
    if not LOG_PATH.exists():
        return {
            "status": "no_data",
            "message": "Aucun log de prédiction disponible.",
            "log_path": "reports/api_predictions_log.csv",
        }

    df = pd.read_csv(LOG_PATH)

    if df.empty:
        return {
            "status": "no_data",
            "message": "Le fichier de log est vide.",
            "log_path": "reports/api_predictions_log.csv",
        }

    return {
        "status": "ok",
        "total_predictions": int(len(df)),
        "mean_prediction": float(df["prediction"].mean()),
        "min_prediction": float(df["prediction"].min()),
        "max_prediction": float(df["prediction"].max()),
        "mean_response_time_ms": float(df["response_time_ms"].mean()),
        "most_common_hour": int(df["hr"].mode()[0]),
        "last_prediction_at": str(df["timestamp"].iloc[-1]),
        "log_path": "reports/api_predictions_log.csv",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(
    input_data: BikeDemandInput,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    start_time = time.time()

    check_api_key(x_api_key)

    try:
        model = load_model()

        row = to_dict(input_data)
        expected_columns = get_expected_columns(model)

        input_row = {}
        for column in expected_columns:
            input_row[column] = row.get(column, 0)

        input_df = pd.DataFrame([input_row], columns=expected_columns)

        prediction = float(model.predict(input_df)[0])

        response_time_ms = (time.time() - start_time) * 1000

        append_prediction_log(
            row=row,
            prediction=prediction,
            status="success",
            response_time_ms=response_time_ms,
        )

        return PredictionOutput(
            prediction=prediction,
            model_path="registry/production/model.joblib",
            status="success",
        )

    except Exception as error:
        response_time_ms = (time.time() - start_time) * 1000

        try:
            append_prediction_log(
                row=to_dict(input_data),
                prediction=-1,
                status="error",
                response_time_ms=response_time_ms,
            )
        except Exception:
            pass

        raise HTTPException(status_code=500, detail=str(error))
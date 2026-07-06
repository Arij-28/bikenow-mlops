from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "registry" / "production" / "model.joblib"


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


app = FastAPI(
    title="BikeNow Demand Forecasting API",
    description="API de prédiction de la demande horaire de vélos.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "BikeNow API is running",
        "model": "registry/production/model.joblib",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    model_exists = MODEL_PATH.exists()

    return {
        "status": "ok" if model_exists else "model_missing",
        "model_path": "registry/production/model.joblib",
        "model_exists": model_exists,
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


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: BikeDemandInput):
    try:
        model = load_model()

        row = to_dict(input_data)
        expected_columns = get_expected_columns(model)

        input_row = {}
        for column in expected_columns:
            input_row[column] = row.get(column, 0)

        input_df = pd.DataFrame([input_row], columns=expected_columns)

        prediction = model.predict(input_df)[0]

        return PredictionOutput(
            prediction=float(prediction),
            model_path="registry/production/model.joblib",
            status="success",
        )

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
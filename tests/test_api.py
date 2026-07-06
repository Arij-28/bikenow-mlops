from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_exists"] is True


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["expected_columns_count"] > 0


def test_predict_endpoint():
    payload = {
        "season": 1,
        "yr": 1,
        "mnth": 6,
        "hr": 17,
        "holiday": 0,
        "weekday": 2,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.62,
        "atemp": 0.60,
        "hum": 0.45,
        "windspeed": 0.20,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
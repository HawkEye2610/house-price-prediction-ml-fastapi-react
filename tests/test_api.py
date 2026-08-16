from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "Random Forest Regressor"
    assert "mae" in data
    assert "rmse" in data
    assert "r2" in data


def test_predict_valid_input():
    payload = {
        "MedInc": 5,
        "HouseAge": 20,
        "AveRooms": 6,
        "AveBedrms": 1,
        "Population": 1000,
        "AveOccup": 3,
        "Latitude": 37,
        "Longitude": -122
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_price_usd" in data
    assert "estimated_error_usd" in data
    assert "prediction_range" in data
    assert "model" in data

    assert data["predicted_price_usd"] > 0


def test_predict_invalid_input():
    payload = {
        "MedInc": 5,
        "HouseAge": -10,
        "AveRooms": 6,
        "AveBedrms": 1,
        "Population": 1000,
        "AveOccup": 3,
        "Latitude": 37,
        "Longitude": -122
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_file():
    csv_content = """MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude
5,20,6,1,1000,3,37,-122
3,30,5,1.2,1200,3.5,34,-118.25
"""

    response = client.post(
        "/predict_file",
        files={
            "file": (
                "test.csv",
                csv_content,
                "text/csv"
            )
        }
    )

    assert response.status_code == 200

    content = response.text

    assert "predicted_price_usd" in content
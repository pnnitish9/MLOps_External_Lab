from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
VALID = {"motor_current": 30, "rated_motor_current": 40, "vibration_level": 4.0,
         "belt_age_months": 30, "parcels_per_min": 40, "dust_index": 40,
         "days_since_maintenance": 30, "humidity": 55}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_prediction_executes():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert r.json()["prediction"] in ("Jam Risk", "Normal Operation")

def test_high_stress_input_is_valid():
    r = client.post("/predict", json={**VALID, "motor_current": 44, "vibration_level": 10,
                                      "dust_index": 90, "days_since_maintenance": 120})
    assert r.status_code == 200

def test_rejects_negative_current():
    assert client.post("/predict", json={**VALID, "motor_current": -5}).status_code == 422

def test_rejects_zero_rated_current():
    assert client.post("/predict", json={**VALID, "rated_motor_current": 0}).status_code == 422

def test_rejects_out_of_range_dust():
    assert client.post("/predict", json={**VALID, "dust_index": 150}).status_code == 422

def test_rejects_missing_field():
    bad = {k: v for k, v in VALID.items() if k != "vibration_level"}
    assert client.post("/predict", json=bad).status_code == 422

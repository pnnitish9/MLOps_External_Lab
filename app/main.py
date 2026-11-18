import os, joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.build_features import engineer, FEATURES

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
app = FastAPI(title="Conveyor Jam Predictor")
_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

class SensorInput(BaseModel):
    motor_current: float = Field(..., gt=0, le=200)
    rated_motor_current: float = Field(..., gt=0, le=200)
    vibration_level: float = Field(..., ge=0, le=50)
    belt_age_months: float = Field(..., ge=0, le=240)
    parcels_per_min: float = Field(..., ge=0, le=300)
    dust_index: float = Field(..., ge=0, le=100)
    days_since_maintenance: float = Field(..., ge=0, le=1000)
    humidity: float = Field(..., ge=0, le=100)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}

@app.post("/predict")
def predict(x: SensorInput):
    df = engineer(pd.DataFrame([x.model_dump()]))
    prob = float(get_model().predict_proba(df[FEATURES])[0][1])
    label = "Jam Risk" if prob >= 0.5 else "Normal Operation"
    return {"prediction": label, "jam_probability": round(prob, 4)}

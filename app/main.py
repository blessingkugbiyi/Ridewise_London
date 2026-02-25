from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import json
import pandas as pd

app = FastAPI(title="RideWise Churn API")

# Load model and feature list
model = joblib.load("models/churn_model.joblib")

with open("models/feature_cols.json", "r") as f:
    FEATURE_COLS = json.load(f)


class PredictionRequest(BaseModel):
    features: Dict[str, Any]


@app.get("/")
def root():
    return {"message": "RideWise Churn API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    # Create dataframe with correct feature order
    input_data = {col: request.features.get(col, 0) for col in FEATURE_COLS}
    df = pd.DataFrame([input_data])

    probability = float(model.predict_proba(df)[:, 1][0])
    prediction = int(model.predict(df)[0])

    return {
        "churn_probability": probability,
        "churn_prediction": prediction
    }
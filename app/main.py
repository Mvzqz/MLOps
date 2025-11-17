# app/main.py
from fastapi import FastAPI
from app.schemas.predict import PredictRequest, PredictionResponse
from app.services.predictor import Predictor
from app.core.config import settings

app = FastAPI(
    title="Seoul bikesharing ML Prediction API",
    version="0.0.1",
    description="An api for predicting bikesharing demand in Seoul using a pre-trained ML model.",
)

predictor = Predictor(model_path=settings.MODEL_PATH)

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "message": "Seoul bikesharing ML Prediction API is running."}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: PredictRequest):
    """Return prediction based on input features."""
    result = predictor.predict(data)
    print("RESULT:", result)
    return PredictionResponse(**result)

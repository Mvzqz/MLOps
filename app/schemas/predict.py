# app/schemas/predict.py
from pydantic import BaseModel, Field
from typing import List, Union

class PredictionRequest(BaseModel):
    features: List[Union[int, float, str]]

class PredictionResponse(BaseModel):
    prediction: float
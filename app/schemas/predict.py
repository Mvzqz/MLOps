# app/schemas/predict.py
from pydantic import BaseModel, Field
from typing import List, Union

class FeatureRow(BaseModel):
    Date: str
    Hour: float
    Temperature_C: float
    Humidity: float
    Wind_speed_m_s: float
    Visibility_10m: float
    Dew_point_temperature_c: float
    solar_radiation_mj_m2: float
    rainfall_mm: str
    snowfall_cm: str
    Seasons: str
    Holiday: str
    Functioning_Day: str
    mixed_type_col: str

class PredictRequest(BaseModel):
    Features: List[FeatureRow]



class PredictionResponse(BaseModel):
    Prediction: list[float]


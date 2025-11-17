# app/schemas/predict.py
from pydantic import BaseModel, Field
from typing import List, Union

class FeatureRow(BaseModel):
    Date: str = Field(..., example="01/12/2017")
    Hour: float = Field(..., example=0.0)
    Temperature_C: float = Field(..., example=-5.2)
    Humidity: float = Field(..., example=37.0)
    Wind_speed_m_s: float = Field(..., example=2.2)
    Visibility_10m: float = Field(..., example=2000.0)
    Dew_point_temperature_c: float = Field(..., example=-17.6)
    solar_radiation_mj_m2: float = Field(..., example=0.0)
    rainfall_mm: float = Field(..., example=0.0)
    snowfall_cm: float = Field(..., example=0.0)
    Seasons: str = Field(..., example="Winter")
    Holiday: str = Field(..., example="No Holiday")
    Functioning_Day: str = Field(..., example="Yes")
    mixed_type_col: str = Field(..., example="873") 
class PredictRequest(BaseModel):
    Features: List[FeatureRow]



class PredictionResponse(BaseModel):
    Prediction: list[float]


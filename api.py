"""
FastAPI service that serves the trained house price model.

Run: uvicorn api:app --reload --port 8000
Docs (auto-generated): http://localhost:8000/docs

Serves whichever model version is currently marked "production" in the
model registry (model_registry.py), and logs every prediction to
SQLite via monitoring.py so /stats has something to aggregate.
"""

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import model_registry
import monitoring
from features import ALL_FEATURES, engineer_features

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts median house value for a California census district.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def get_model():
    global _model, _metadata
    if _model is None:
        try:
            _model, _metadata = model_registry.load_production_model()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _model, _metadata


class HouseFeatures(BaseModel):
    """Raw district-level inputs. Pydantic validates types and ranges
    automatically -- a request with a negative population, or a string
    where a number is expected, gets rejected with a clear 422 error
    before it ever reaches the model.
    """
    longitude: float = Field(..., ge=-125, le=-113, description="District longitude (California range)")
    latitude: float = Field(..., ge=32, le=42, description="District latitude (California range)")
    housing_median_age: float = Field(..., ge=0, le=60, description="Median age of houses in the district (years)")
    total_rooms: float = Field(..., gt=0, description="Total rooms across all houses in the district")
    total_bedrooms: float = Field(..., gt=0, description="Total bedrooms across all houses in the district")
    population: float = Field(..., gt=0, description="District population")
    households: float = Field(..., gt=0, description="Number of households in the district")
    median_income: float = Field(..., gt=0, description="Median income, in tens of thousands of USD (e.g. 5.0 = ₹50,000)")
    ocean_proximity: str = Field(..., description="One of: NEAR BAY, <1H OCEAN, INLAND, NEAR OCEAN, ISLAND")

    class Config:
        json_schema_extra = {
            "example": {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY",
            }
        }


class PredictionResponse(BaseModel):
    predicted_price: float
    model_name: str
    model_version: str
    latency_ms: float
    engineered_features: dict


@app.get("/")
def root():
    return {"service": "House Price Prediction API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    prod = model_registry.get_production_info()
    return {"status": "ok" if prod else "no production model", "production": prod}


@app.get("/metadata")
def metadata():
    _, meta = get_model()
    return meta


@app.get("/versions")
def versions():
    """Full version history from the registry -- every model ever
    trained, its metrics, and whether it was promoted to production.
    """
    return {
        "production": model_registry.get_production_info(),
        "all_versions": model_registry.list_versions(),
    }


@app.get("/stats")
def stats():
    """Aggregate monitoring stats from logged predictions: volume,
    latency, and predicted-price distribution. In a real deployment
    this is what you'd wire a dashboard or alerting to.
    """
    return monitoring.get_stats()


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    model, meta = get_model()

    with monitoring.Timer() as t:
        raw_df = pd.DataFrame([features.model_dump()])
        feats = engineer_features(raw_df)[ALL_FEATURES]
        prediction = float(model.predict(feats)[0])

    monitoring.log_prediction(
        model_version=meta["version_id"],
        latency_ms=t.elapsed_ms,
        predicted_price=prediction,
        input_dict=features.model_dump(),
    )

    return PredictionResponse(
        predicted_price=round(prediction, 2),
        model_name=meta["model_name"],
        model_version=meta["version_id"],
        latency_ms=round(t.elapsed_ms, 2),
        engineered_features={
            "rooms_per_household": round(float(feats["rooms_per_household"].iloc[0]), 3),
            "bedrooms_per_room": round(float(feats["bedrooms_per_room"].iloc[0]), 3),
            "population_per_household": round(float(feats["population_per_household"].iloc[0]), 3),
        },
    )

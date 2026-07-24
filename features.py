"""
Shared feature engineering for the house price model.

Imported by both train.py (offline training) and api.py (online serving)
so both sides transform raw inputs identically -- the same train/serve
skew fix applied in the fraud detection project.
"""

import numpy as np
import pandas as pd

RAW_NUMERIC = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]

# Engineered ratio features. Raw counts like total_rooms or total_bedrooms
# mean very little on their own -- a district with 2000 rooms could be a
# few large apartment blocks or hundreds of small houses. Expressing them
# as per-household ratios turns raw counts into meaningful signal.
ENGINEERED_NUMERIC = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]

NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "median_house_value"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds engineered ratio features to the raw housing dataframe.

    Expects raw columns: longitude, latitude, housing_median_age,
    total_rooms, total_bedrooms, population, households, median_income,
    ocean_proximity (+ median_house_value if present, for training).
    """
    out = df.copy()

    out["rooms_per_household"] = out["total_rooms"] / out["households"]
    out["bedrooms_per_room"] = out["total_bedrooms"] / out["total_rooms"]
    out["population_per_household"] = out["population"] / out["households"]

    # Guard against divide-by-zero edge cases in raw/adversarial input.
    out[ENGINEERED_NUMERIC] = out[ENGINEERED_NUMERIC].replace([np.inf, -np.inf], np.nan)

    cols = ALL_FEATURES + ([TARGET] if TARGET in out.columns else [])
    return out[cols]

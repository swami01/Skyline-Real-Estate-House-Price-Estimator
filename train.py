"""
Trains and evaluates the house price model, then saves the best one.

Run: python3 train.py

Real dataset: California Housing Prices, 1990 US Census block-group data
(Pace & Barry, 1997) -- 20,640 rows, legitimately and openly documented.
This replaces the original project's dataset, which was the deprecated
Boston Housing dataset with columns renamed and a fabricated source
("house of hirnandani") -- see the conversation notes / README for why
that had to go before this went on a resume.
"""

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import model_registry
from features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, engineer_features

DATA_PATH = "data/housing.csv"
RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    # SimpleImputer handles the ~200 real missing values in total_bedrooms
    # baked into this dataset -- median is robust to the right-skewed
    # distribution of bedroom counts.
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    metrics = {
        "rmse": rmse,
        "mae": mean_absolute_error(y_test, pred),
        "r2": r2_score(y_test, pred),
    }
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"  {k:6s}: {v:,.4f}")
    return metrics


def main():
    print("Loading data...")
    raw = pd.read_csv(DATA_PATH)
    df = engineer_features(raw)
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    candidates = {}

    # --- Linear Regression: simple baseline, establishes a floor. ---
    lr_pipe = Pipeline([("prep", build_preprocessor()), ("reg", LinearRegression())])
    lr_pipe.fit(X_train, y_train)
    candidates["linear_regression"] = lr_pipe

    # --- Random Forest: tuned via CV, not an arbitrary max_depth. ---
    rf_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("reg", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
    ])
    # min_samples_leaf is floored at 2 (not 1) deliberately: unconstrained
    # depth-20 trees with leaf=1 on this data produce a 160MB+ pickled
    # model, which exceeds GitHub's 100MB file limit. leaf>=2 keeps trees
    # smaller with a negligible accuracy cost, and the model ends up
    # portfolio/deploy-friendly instead of an oversized artifact.
    rf_grid = {
        "reg__n_estimators": [80, 150],
        "reg__max_depth": [10, 20],
        "reg__min_samples_leaf": [2, 5],
    }
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_grid, n_iter=3, scoring="neg_root_mean_squared_error",
        cv=cv, random_state=RANDOM_STATE, n_jobs=1,
    )
    t0 = time.time()
    rf_search.fit(X_train, y_train)
    print(f"RandomForest search done in {time.time()-t0:.1f}s, "
          f"best CV RMSE={-rf_search.best_score_:,.0f}, params={rf_search.best_params_}")
    candidates["random_forest"] = rf_search.best_estimator_

    # --- Gradient Boosting: usually strong on tabular regression. ---
    gb_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("reg", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])
    gb_grid = {
        "reg__n_estimators": [80, 150],
        "reg__max_depth": [3, 4],
        "reg__learning_rate": [0.05, 0.1],
    }
    gb_search = RandomizedSearchCV(
        gb_pipe, gb_grid, n_iter=3, scoring="neg_root_mean_squared_error",
        cv=cv, random_state=RANDOM_STATE, n_jobs=1,
    )
    t0 = time.time()
    gb_search.fit(X_train, y_train)
    print(f"GradientBoosting search done in {time.time()-t0:.1f}s, "
          f"best CV RMSE={-gb_search.best_score_:,.0f}, params={gb_search.best_params_}")
    candidates["gradient_boosting"] = gb_search.best_estimator_

    results = {name: evaluate(name, model, X_test, y_test) for name, model in candidates.items()}

    best_name = min(results, key=lambda n: results[n]["rmse"])
    best_model = candidates[best_name]
    print(f"\nSelected model: {best_name} (lowest RMSE on held-out test set)")

    joblib_free_metadata = {
        "model_name": best_name,
        "features": ALL_FEATURES,
        "target": TARGET,
        "test_metrics": results[best_name],
        "all_candidate_metrics": results,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    entry = model_registry.register_model(best_model, joblib_free_metadata)

    print(f"\nRegistered as version: {entry['version_id']}")
    if entry["promoted"]:
        print("-> Promoted to PRODUCTION (beat or matched previous best RMSE, or no prior production model existed)")
    else:
        prod = model_registry.get_production_info()
        print(f"-> NOT promoted. Current production version ({prod['active_version']}) "
              f"still has a lower/equal RMSE. Use model_registry.rollback_to() to force-promote if needed.")


if __name__ == "__main__":
    main()

"""
train_experiments.py
--------------------
Controlled comparative experiments for infographic aesthetics prediction.

This script:
1. Loads infographic dataset
2. Splits features into groups
3. Trains separate models
4. Evaluates performance
5. Saves trained models

Project 47 — Assignment 3
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)

from config import (
    IMAGE_FEATURES,
    INFOGRAPHIC_FEATURES,
    ALL_FEATURES,
    TARGET_COL,
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(csv_path="data.csv"):

    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()

    df = df[
        pd.to_numeric(df["Stimulus ID"], errors="coerce").notna()
    ].copy()

    df = df.reset_index(drop=True)

    all_needed_cols = ALL_FEATURES + [TARGET_COL]

    for col in all_needed_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=all_needed_cols)

    print(f"Loaded {len(df)} infographics after cleaning.")

    return df


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_experiment(
    df,
    feature_group,
    feature_group_name,
    output_filename,
):

    print("\n" + "=" * 52)
    print("Project 47 - Assignment 3 Experimental Training")
    print("=" * 52)

    print("\nUsing feature group:")

    for feature in feature_group:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # FEATURE MATRIX
    # --------------------------------------------------------

    X = df[feature_group].values
    y = df[TARGET_COL].values

    print(f"\nDataset size: {len(df)} infographics")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape : {y.shape}")

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\nTrain/test split complete.")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1)),
    ])

    print("\nTraining Ridge Regression model...")

    pipeline.fit(X_train, y_train)

    print("Training complete.")

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("\n--- Experiment Results ---")
    print(f"Feature group : {feature_group_name}")
    print("Model         : Ridge(alpha=1)")
    print(f"R²            : {r2:.3f}")
    print(f"MAE           : {mae:.3f}")
    print(f"RMSE          : {rmse:.3f}")

    # --------------------------------------------------------
    # FEATURE COEFFICIENTS
    # --------------------------------------------------------

    inner_model = pipeline.named_steps["model"]

    print("\nFeature coefficients:")

    for feature, coef in sorted(
        zip(feature_group, inner_model.coef_),
        key=lambda x: abs(x[1]),
        reverse=True,
    ):

        direction = "↑ appeal" if coef > 0 else "↓ appeal"

        print(
            f"  {feature:20s} "
            f"coef={coef:+.4f} {direction}"
        )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    os.makedirs("models", exist_ok=True)

    bundle = {
        "pipeline": pipeline,
        "feature_cols": feature_group,
        "feature_group_name": feature_group_name,
        "model_name": "Ridge(alpha=1)",
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "num_samples": int(len(df)),
    }

    output_path = f"models/{output_filename}"

    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nModel saved to: {output_path}")

    print("\nExperiment completed successfully.")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    # --------------------------------------------------------
    # IMAGE FEATURES ONLY
    # --------------------------------------------------------

    run_experiment(
        df=df,
        feature_group=IMAGE_FEATURES,
        feature_group_name="IMAGE_FEATURES",
        output_filename="ridge_image.pkl",
    )

    # --------------------------------------------------------
    # INFOGRAPHIC FEATURES ONLY
    # --------------------------------------------------------

    run_experiment(
        df=df,
        feature_group=INFOGRAPHIC_FEATURES,
        feature_group_name="INFOGRAPHIC_FEATURES",
        output_filename="ridge_infographic.pkl",
    )

    # --------------------------------------------------------
    # COMBINED FEATURES
    # --------------------------------------------------------

    run_experiment(
        df=df,
        feature_group=ALL_FEATURES,
        feature_group_name="ALL_FEATURES",
        output_filename="ridge_combined.pkl",
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
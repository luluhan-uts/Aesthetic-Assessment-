"""
train_experiments.py
--------------------
Assignment 3 experimental training pipeline for Project 47.

This script extends the original train_model.py workflow into a
controlled comparative experiment framework.

Current version:
    - Ridge Regression only
    - Image-aesthetic feature group only
    - Single experiment pathway
    - Saves trained model bundle into models/

Goal:
    Build stable experimental infrastructure BEFORE adding:
        • multiple feature groups
        • multiple regressors
        • automated experiment loops
        • explainability analysis

Usage:
    python train_experiments.py
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from config import (
    DATA_CSV,
    TARGET_COL,
    IMAGE_FEATURES,
    INFOGRAPHIC_FEATURES,
    ALL_FEATURES,
)

from train_model import load_data


def run_experiment(feature_list, feature_group_name, output_filename):

    print("\n====================================================")
    print("Project 47 - Assignment 3 Experimental Training")
    print("====================================================")

    # ------------------------------------------------------------------
    # Load cleaned dataset
    # ------------------------------------------------------------------
    df = load_data(DATA_CSV)

    print("\nUsing feature group:")
    for feature in feature_list:
        print(f"  - {feature}")

    # ------------------------------------------------------------------
    # Select features and target
    # ------------------------------------------------------------------
    X = df[feature_list].values
    y = df[TARGET_COL].values

    print(f"\nDataset size: {len(df)} infographics")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape : {y.shape}")

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\nTrain/test split complete.")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    # ------------------------------------------------------------------
    # Build Ridge regression pipeline
    # ------------------------------------------------------------------
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])

    print("\nTraining Ridge Regression model...")

    # ------------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------------
    pipeline.fit(X_train, y_train)

    print("Training complete.")

    # ------------------------------------------------------------------
    # Predict on test set
    # ------------------------------------------------------------------
    y_pred = pipeline.predict(X_test)

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Experiment Results ---")
    print(f"Feature group : {feature_group_name}")
    print(f"Model         : Ridge(alpha=1)")
    print(f"R²            : {r2:.3f}")
    print(f"MAE           : {mae:.3f}")
    print(f"RMSE          : {rmse:.3f}")

    # ------------------------------------------------------------------
    # Feature coefficients
    # ------------------------------------------------------------------
    inner_model = pipeline.named_steps["model"]

    if hasattr(inner_model, "coef_"):

        print("\nFeature coefficients:")

        coefficients = inner_model.coef_

        for feature, coef in sorted(
            zip(feature_list, coefficients),
            key=lambda x: abs(x[1]),
            reverse=True
        ):

            direction = "↑ appeal" if coef > 0 else "↓ appeal"

            print(
                f"  {feature:20s} "
                f"coef={coef:+.4f} "
                f"{direction}"
            )

    # ------------------------------------------------------------------
    # Prepare output folder
    # ------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)

    # ------------------------------------------------------------------
    # Save model bundle
    # ------------------------------------------------------------------
    bundle = {
        "pipeline": pipeline,
        "feature_cols": feature_list,
        "target_col": TARGET_COL,
        "model_name": "Ridge(alpha=1)",
        "feature_group": feature_group_name,
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


if __name__ == "__main__":

    run_experiment(
        IMAGE_FEATURES,
        "IMAGE_FEATURES",
        "ridge_image.pkl"
    )

    run_experiment(
        INFOGRAPHIC_FEATURES,
        "INFOGRAPHIC_FEATURES",
        "ridge_infographic.pkl"
    )

    run_experiment(
        ALL_FEATURES,
        "ALL_FEATURES",
        "ridge_combined.pkl"
    )
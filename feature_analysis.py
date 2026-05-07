"""
feature_analysis.py
-------------------
Loads trained Ridge regression experiment models and extracts
feature coefficients for explainability analysis.

Outputs:
    outputs/feature_analysis/feature_coefficients.csv

This script focuses on:
    • feature importance
    • coefficient direction
    • explainability analysis

Usage:
    python feature_analysis.py
"""

import os
import pickle
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
MODELS_DIR = "models"
OUTPUT_DIR = "outputs/feature_analysis"

MASTER_OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "feature_coefficients.csv"
)


def load_model_bundle(filepath):
    """
    Load a saved model bundle (.pkl).
    """

    with open(filepath, "rb") as f:
        bundle = pickle.load(f)

    return bundle


def extract_coefficients(bundle, filename):
    """
    Extract Ridge coefficients and related metadata.
    """

    pipeline = bundle["pipeline"]

    model = pipeline.named_steps["model"]

    feature_cols = bundle["feature_cols"]

    feature_group = bundle.get("feature_group", "Unknown")

    model_name = bundle.get("model_name", "Unknown")

    # ------------------------------------------------------------------
    # Ensure model has coefficients
    # ------------------------------------------------------------------
    if not hasattr(model, "coef_"):

        print(f"Skipping {filename} (no coefficients available).")

        return []

    coefficients = model.coef_

    rows = []

    for feature, coef in zip(feature_cols, coefficients):

        row = {
            "model_file": filename,
            "model_name": model_name,
            "feature_group": feature_group,
            "feature": feature,
            "coefficient": float(coef),
            "abs_coefficient": float(abs(coef)),
            "direction": (
                "positive"
                if coef > 0
                else "negative"
            ),
        }

        rows.append(row)

    return rows


def main():

    print("\n====================================================")
    print("Project 47 - Feature Explainability Analysis")
    print("====================================================")

    # ------------------------------------------------------------------
    # Create output folder
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Find model files
    # ------------------------------------------------------------------
    model_files = [
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".pkl")
    ]

    if len(model_files) == 0:

        print("\nNo model files found.")

        return

    print(f"\nFound {len(model_files)} model files.")

    all_rows = []

    # ------------------------------------------------------------------
    # Process each model
    # ------------------------------------------------------------------
    for filename in sorted(model_files):

        filepath = os.path.join(MODELS_DIR, filename)

        print(f"\nLoading: {filename}")

        try:

            bundle = load_model_bundle(filepath)

            rows = extract_coefficients(bundle, filename)

            all_rows.extend(rows)

            print(f"  Extracted {len(rows)} coefficients.")

        except Exception as e:

            print("  Failed to analyse model.")
            print(f"  Error: {e}")

    # ------------------------------------------------------------------
    # Build dataframe
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_rows)

    if len(df) == 0:

        print("\nNo coefficient data extracted.")

        return

    # ------------------------------------------------------------------
    # Sort by importance
    # ------------------------------------------------------------------
    df = df.sort_values(
        by="abs_coefficient",
        ascending=False
    )

    df = df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n====================================================")
    print("TOP FEATURE COEFFICIENTS")
    print("====================================================")

    print(
        df[
            [
                "feature_group",
                "feature",
                "coefficient",
                "direction",
            ]
        ].head(20).to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Save master CSV
    # ------------------------------------------------------------------
    df.to_csv(MASTER_OUTPUT_CSV, index=False)

    print("\nFeature coefficient table saved to:")
    print(f"  {MASTER_OUTPUT_CSV}")

    # ------------------------------------------------------------------
    # Save individual CSVs
    # ------------------------------------------------------------------
    for model_file in df["model_file"].unique():

        sub_df = df[df["model_file"] == model_file]

        output_name = (
            model_file.replace(".pkl", "_coefficients.csv")
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        sub_df.to_csv(output_path, index=False)

        print(f"  Saved: {output_name}")

    print("\nExplainability analysis completed successfully.")


if __name__ == "__main__":
    main()
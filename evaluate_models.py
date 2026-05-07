"""
evaluate_models.py
------------------
Loads all trained experiment models from the models/ folder,
extracts stored evaluation metrics and metadata, then builds
a structured comparison table for Assignment 3 analysis.

Outputs:
    outputs/metrics/model_comparison.csv

This script does NOT retrain models.
It only evaluates previously saved experiment results.

Usage:
    python evaluate_models.py
"""

import os
import pickle
import pandas as pd


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
MODELS_DIR = "models"
OUTPUT_DIR = "outputs/metrics"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "model_comparison.csv")


def load_model_bundle(filepath):
    """
    Load a saved experiment bundle (.pkl file).
    """

    with open(filepath, "rb") as f:
        bundle = pickle.load(f)

    return bundle


def extract_metrics(bundle, filename):
    """
    Extract key evaluation information from a model bundle.
    """

    result = {
        "model_file": filename,
        "model_name": bundle.get("model_name", "Unknown"),
        "feature_group": bundle.get("feature_group", "Unknown"),
        "r2": bundle.get("r2", None),
        "mae": bundle.get("mae", None),
        "rmse": bundle.get("rmse", None),
        "num_samples": bundle.get("num_samples", None),
        "num_features": len(bundle.get("feature_cols", [])),
    }

    return result


def main():

    print("\n====================================================")
    print("Project 47 - Model Evaluation Summary")
    print("====================================================")

    # ------------------------------------------------------------------
    # Create output folder if needed
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Find all .pkl model files
    # ------------------------------------------------------------------
    model_files = [
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".pkl")
    ]

    if len(model_files) == 0:
        print("\nNo model files found.")
        return

    print(f"\nFound {len(model_files)} model files.")

    # ------------------------------------------------------------------
    # Load and extract results
    # ------------------------------------------------------------------
    results = []

    for filename in sorted(model_files):

        filepath = os.path.join(MODELS_DIR, filename)

        print(f"\nLoading: {filename}")

        try:
            bundle = load_model_bundle(filepath)

            result = extract_metrics(bundle, filename)

            results.append(result)

            print("  Successfully loaded.")

        except Exception as e:

            print(f"  Failed to load model.")
            print(f"  Error: {e}")

    # ------------------------------------------------------------------
    # Build comparison dataframe
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)

    # Sort by R² descending
    if "r2" in df.columns:
        df = df.sort_values(by="r2", ascending=False)

    # Reset index
    df = df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("\n====================================================")
    print("MODEL COMPARISON TABLE")
    print("====================================================")

    print(df.to_string(index=False))

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nComparison table saved to:")
    print(f"  {OUTPUT_CSV}")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
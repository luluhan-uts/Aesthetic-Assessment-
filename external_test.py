"""
external_test.py
----------------
Runs the trained infographic aesthetic model on the curated
external infographic dataset.

The script:
1. Loads a trained model (.pkl)
2. Iterates through all external infographic folders
3. Extracts features from each infographic
4. Predicts aesthetic appeal score
5. Merges predictions with metadata CSV
6. Saves detailed results to CSV
7. Produces grouped category summaries

Project 47 — Assignment 3
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd

from feature_extractor import extract_features
from config import ALL_FEATURES


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/ridge_combined.pkl"

EXTERNAL_DATASET_DIR = "external_infographics"
METADATA_PATH = (
    "external_infographics/metadata/"
    "external_infographics_metadata.csv"
)

OUTPUT_DIR = "outputs/external_testing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.webp"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def load_model(model_path):
    """Load trained model bundle."""

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    print(f"Loaded model: {bundle['model_name']}")
    return bundle



def find_all_images(root_dir):
    """Recursively find infographic images."""

    image_paths = []

    for ext in VALID_EXTENSIONS:
        image_paths.extend(
            glob.glob(
                os.path.join(root_dir, "**", ext),
                recursive=True,
            )
        )

    return sorted(image_paths)



def build_feature_vector(feature_dict, feature_cols):
    """Convert extracted feature dictionary into model vector."""

    values = []

    for col in feature_cols:
        value = feature_dict.get(col, 0)

        if value is None:
            value = 0

        values.append(float(value))

    return np.array(values).reshape(1, -1)


# ============================================================
# MAIN TESTING PIPELINE
# ============================================================


def main():

    print("\n" + "=" * 60)
    print("Project 47 - External Infographic Testing")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    bundle = load_model(MODEL_PATH)

    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]

    print("\nUsing features:")
    for feature in feature_cols:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    metadata_df = pd.read_csv(METADATA_PATH)

    print(
        f"\nLoaded metadata for "
        f"{len(metadata_df)} external infographics."
    )

    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------

    image_paths = find_all_images(EXTERNAL_DATASET_DIR)

    print(f"Found {len(image_paths)} infographic images.")

    results = []

    # --------------------------------------------------------
    # PROCESS EACH IMAGE
    # --------------------------------------------------------

    for i, image_path in enumerate(image_paths):

        filename = os.path.basename(image_path)

        print("\n" + "-" * 50)
        print(f"[{i+1}/{len(image_paths)}] {filename}")

        try:
            # ------------------------------------------------
            # FEATURE EXTRACTION
            # ------------------------------------------------

            features = extract_features(image_path)

            print("Feature extraction complete.")

            # ------------------------------------------------
            # BUILD FEATURE VECTOR
            # ------------------------------------------------

            X = build_feature_vector(features, feature_cols)

            # ------------------------------------------------
            # PREDICT APPEAL
            # ------------------------------------------------

            predicted_score = pipeline.predict(X)[0]

            predicted_score = float(predicted_score)

            print(f"Predicted appeal score: {predicted_score:.3f}")

            # ------------------------------------------------
            # FIND METADATA ROW
            # ------------------------------------------------

            metadata_match = metadata_df[
                metadata_df["filename"] == filename
            ]

            if len(metadata_match) > 0:
                row = metadata_match.iloc[0]

                category = row["category"]
                subcategory = row["subcategory"]
                dominant_features = row["dominant_features"]
                visual_characteristics = row[
                    "visual_characteristics"
                ]
                expected_behavior = row[
                    "expected_model_behavior"
                ]

            else:
                category = "unknown"
                subcategory = "unknown"
                dominant_features = ""
                visual_characteristics = ""
                expected_behavior = ""

            # ------------------------------------------------
            # SAVE RESULT ROW
            # ------------------------------------------------

            result = {
                "filename": filename,
                "category": category,
                "subcategory": subcategory,
                "predicted_score": predicted_score,
                "dominant_features": dominant_features,
                "visual_characteristics": visual_characteristics,
                "expected_model_behavior": expected_behavior,
            }

            # Add extracted features
            for feature_name, feature_value in features.items():
                result[feature_name] = feature_value

            results.append(result)

        except Exception as e:

            print(f"Failed processing: {filename}")
            print(f"Error: {e}")

    # ========================================================
    # BUILD RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # SORT BY SCORE
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        by="predicted_score",
        ascending=False,
    )

    # --------------------------------------------------------
    # SAVE FULL RESULTS
    # --------------------------------------------------------

    results_csv_path = os.path.join(
        OUTPUT_DIR,
        "external_predictions.csv",
    )

    results_df.to_csv(results_csv_path, index=False)

    print("\n" + "=" * 60)
    print("EXTERNAL TESTING RESULTS")
    print("=" * 60)

    display_cols = [
        "filename",
        "category",
        "predicted_score",
    ]

    print(results_df[display_cols].to_string(index=False))

    print(f"\nSaved full results to:")
    print(f"  {results_csv_path}")

    # ========================================================
    # CATEGORY SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("CATEGORY SUMMARY")
    print("=" * 60)

    category_summary = (
        results_df
        .groupby("category")
        ["predicted_score"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    category_summary = category_summary.sort_values(
        by="mean",
        ascending=False,
    )

    print(category_summary.to_string(index=False))

    summary_csv_path = os.path.join(
        OUTPUT_DIR,
        "category_summary.csv",
    )

    category_summary.to_csv(summary_csv_path, index=False)

    print(f"\nSaved category summary to:")
    print(f"  {summary_csv_path}")

    # ========================================================
    # TOP / BOTTOM EXAMPLES
    # ========================================================

    print("\n" + "=" * 60)
    print("TOP 5 PREDICTED INFOGRAPHICS")
    print("=" * 60)

    top5 = results_df.head(5)

    print(
        top5[
            [
                "filename",
                "category",
                "predicted_score",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("BOTTOM 5 PREDICTED INFOGRAPHICS")
    print("=" * 60)

    bottom5 = results_df.tail(5)

    print(
        bottom5[
            [
                "filename",
                "category",
                "predicted_score",
            ]
        ].to_string(index=False)
    )

    print("\nExternal testing completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

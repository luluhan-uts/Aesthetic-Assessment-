"""
predict.py
----------
Given a trained model (model.pkl) and an infographic image, extracts
features and predicts the aesthetic appeal score (1–9 scale).

Usage:
    python predict.py --image my_infographic.png --model model.pkl
"""

import argparse
import pickle
import numpy as np
from feature_extractor import extract_features
from config import FEATURE_COLS, MODEL_PKL


def predict(image_path: str, model_path: str) -> dict:
    """
    Predict aesthetic appeal of an infographic.

    Parameters
    ----------
    image_path : str
        Path to infographic image.
    model_path : str
        Path to saved model bundle (.pkl).

    Returns
    -------
    dict with predicted score, extracted features, and model metadata.
    """
    # Load model
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    pipeline = bundle["pipeline"]

    # Extract features
    print(f"Extracting features from: {image_path}")
    features = extract_features(image_path)

    print("\nExtracted features:")
    for col in FEATURE_COLS:
        val = features.get(col, 0)
        print(f"  {col:20s}: {val:.4f}" if isinstance(val, float) else
              f"  {col:20s}: {val}")

    # Build feature vector in correct column order
    X = np.array([[features.get(col, 0) for col in FEATURE_COLS]])

    # Predict
    score = float(pipeline.predict(X)[0])
    # Clamp to valid Likert range
    score = float(np.clip(score, 1.0, 9.0))

    # Interpretation
    if score < 3:
        label = "low appeal"
    elif score < 4.5:
        label = "below average"
    elif score < 5.5:
        label = "average"
    elif score < 7:
        label = "above average"
    else:
        label = "high appeal"

    result = {
        "predicted_appeal": round(score, 2),
        "label": label,
        "features": features,
        "model_cv_r2": bundle.get("cv_r2"),
        "model_name": bundle.get("model_name"),
    }

    print(f"\n{'='*40}")
    print(f"Predicted appeal score: {score:.2f} / 9.0")
    print(f"Interpretation:         {label}")
    print(f"(Model CV R²={bundle.get('cv_r2', '?'):.3f})")
    print(f"{'='*40}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to infographic image")
    parser.add_argument("--model", default=MODEL_PKL, help="Path to trained model")
    args = parser.parse_args()
    predict(args.image, args.model)
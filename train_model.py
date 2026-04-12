"""
train_model.py
--------------
Trains a Ridge regression model to predict infographic aesthetic appeal
using the dataset from:
  Harrison, Reinecke & Chang (CHI 2015) "Infographic Aesthetics"

The CSV (data.csv) contains pre-computed features + mean appeal ratings
for 326 infographics. We train on these and save the model to disk.

Usage:
    python train_model.py --csv data.csv --output model.pkl
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from config import FEATURE_COLS, TARGET_COL
from feature_extractor import normalize_features


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and clean the Harrison et al. CSV."""
    df = pd.read_csv(csv_path, header=0)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Keep only numeric rows (drop summary rows at the bottom)
    df = df[pd.to_numeric(df["Stimulus ID"], errors="coerce").notna()].copy()
    df = df.reset_index(drop=True)

    # Cast all feature and target columns to float
    for col in FEATURE_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with any missing values in features or target
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    # Normalize scale-sensitive features so data.csv and extracted features
    # are on the same scale (viewport crops vs. full-page screenshots).
    total_area = df['NonTextArea'] + df['TextArea']
    df['NonTextArea'] = df['NonTextArea'] / total_area.clip(lower=1)
    df['TextArea']    = df['TextArea']    / total_area.clip(lower=1)
    df['QuadTree']    = np.log(df['QuadTree'] + 1)

    print(f"Loaded {len(df)} infographics after cleaning.")
    return df


def train(csv_path: str, output_path: str):
    df = load_data(csv_path)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    print(f"\nFeature summary:")
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {col:20s}  mean={X[:, i].mean():.3f}  std={X[:, i].std():.3f}")
    print(f"\nTarget (appeal) mean={y.mean():.3f}  std={y.std():.3f}  "
          f"range=[{y.min():.2f}, {y.max():.2f}]")

    # -----------------------------------------------------------------------
    # Compare three models with 5-fold cross-validation
    # -----------------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Ridge (alpha=1)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Ridge (alpha=10)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=200, max_depth=3,
                learning_rate=0.05, random_state=42
            )),
        ]),
    }

    print("\n--- Cross-validation results (5-fold) ---")
    best_name, best_score, best_pipeline = None, -np.inf, None

    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, X, y, cv=kf,
                                 scoring="r2", n_jobs=-1)
        mean_r2 = scores.mean()
        print(f"  {name:30s}  R²={mean_r2:.3f}  (±{scores.std():.3f})")
        if mean_r2 > best_score:
            best_score = mean_r2
            best_name = name
            best_pipeline = pipeline

    print(f"\nBest model: {best_name}  (CV R²={best_score:.3f})")

    # -----------------------------------------------------------------------
    # Fit best model on full dataset
    # -----------------------------------------------------------------------
    best_pipeline.fit(X, y)
    y_pred = best_pipeline.predict(X)
    train_r2 = r2_score(y, y_pred)
    train_mae = mean_absolute_error(y, y_pred)
    print(f"Train R²={train_r2:.3f}  MAE={train_mae:.3f}")

    # Feature importances (for Ridge: coefficients after scaling)
    scaler = best_pipeline.named_steps["scaler"]
    inner_model = best_pipeline.named_steps["model"]

    print("\nFeature importances / coefficients:")
    if hasattr(inner_model, "coef_"):
        coefs = inner_model.coef_
        for col, coef in sorted(zip(FEATURE_COLS, coefs),
                                key=lambda x: abs(x[1]), reverse=True):
            direction = "↑ appeal" if coef > 0 else "↓ appeal"
            print(f"  {col:20s}  coef={coef:+.4f}  {direction}")
    elif hasattr(inner_model, "feature_importances_"):
        imps = inner_model.feature_importances_
        for col, imp in sorted(zip(FEATURE_COLS, imps),
                               key=lambda x: x[1], reverse=True):
            print(f"  {col:20s}  importance={imp:.4f}")

    # -----------------------------------------------------------------------
    # Save model + metadata
    # -----------------------------------------------------------------------
    bundle = {
        "pipeline": best_pipeline,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "model_name": best_name,
        "cv_r2": best_score,
        "train_r2": train_r2,
        "train_mae": train_mae,
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
    }

    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nModel saved to: {output_path}")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data.csv")
    parser.add_argument("--output", default="model.pkl")
    args = parser.parse_args()
    train(args.csv, args.output)

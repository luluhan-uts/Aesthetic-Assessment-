"""
batch_analyse.py
----------------
Processes a folder of infographic images, extracts features from each,
aligns them with the ground-truth data in data.csv by Stimulus ID,
and produces:
  1. A comparison of extracted vs. paper's pre-computed features
  2. Train/test split evaluation of the regression model
  3. A results CSV with predicted vs. actual appeal scores

Usage:
    python batch_analyse.py --images ./infographics --csv data.csv --output results.csv

The image filenames must contain the Stimulus ID from the CSV.
Accepted formats:
    0.png, 1.jpg, 42.png, infographic_7.png, img_042.jpg
(the script extracts the first number it finds in the filename)
"""

import argparse
import os
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from feature_extractor import extract_features, normalize_features
from config import FEATURE_COLS, TARGET_COL, DATA_CSV, IMAGES_FOLDER, MODEL_PKL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_stimulus_id(filename: str):
    """Pull the first integer out of a filename. E.g. 'img_042.png' -> 42."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"\d+", stem)
    return int(match.group()) if match else None


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=0)
    df.columns = df.columns.str.strip()

    # Drop unnamed/extra columns from Excel export
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Keep only rows where Stimulus ID is a valid number
    df = df[pd.to_numeric(df["Stimulus ID"], errors="coerce").notna()].copy()
    df["Stimulus ID"] = df["Stimulus ID"].astype(int)

    # Detect whether this is individual-level or per-infographic data
    if "Final Score" in df.columns:
        # Per-infographic summary CSV (data.csv from paper website)
        print("  Detected: per-infographic summary CSV")
        df["Final Score"] = pd.to_numeric(df["Final Score"], errors="coerce")
    elif "Phase 1 Response" in df.columns and "Phase 2 Response" in df.columns:
        # Individual-level CSV with participant ratings
        print("  Detected: individual participant ratings CSV")
        print("  Computing Final Score as mean of Phase 1 and Phase 2 responses...")
        df["Phase 1 Response"] = pd.to_numeric(df["Phase 1 Response"], errors="coerce")
        df["Phase 2 Response"] = pd.to_numeric(df["Phase 2 Response"], errors="coerce")
        df["Final Score"] = df[["Phase 1 Response", "Phase 2 Response"]].mean(axis=1)

        # Also parse demographic columns if present
        for col in ["Gender", "Age", "Education Level"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Aggregate to per-infographic means (average across all participants)
        # keeping feature columns and demographics for reference
        agg_dict = {col: "first" for col in FEATURE_COLS if col in df.columns}
        agg_dict["Final Score"] = "mean"
        agg_dict["Phase 1 Response"] = "mean"
        agg_dict["Phase 2 Response"] = "mean"
        if "Image Link" in df.columns:
            agg_dict["Image Link"] = "first"

        df = df.groupby("Stimulus ID").agg(agg_dict).reset_index()
        print(f"  Aggregated to {len(df)} unique infographics.")
    else:
        raise ValueError(
            "CSV must contain either 'Final Score' or both "
            "'Phase 1 Response' and 'Phase 2 Response' columns."
        )

    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns] + ["Final Score"])
    df = df.set_index("Stimulus ID")
    return df


def find_images(folder: str) -> dict:
    """Returns {stimulus_id: filepath} for all images in folder."""
    supported = {".png", ".jpg", ".jpeg", ".webp"}
    images = {}
    for fname in os.listdir(folder):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in supported:
            continue
        sid = extract_stimulus_id(fname)
        if sid is not None:
            images[sid] = os.path.join(folder, fname)
    return images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(images_folder: str, csv_path: str, output_path: str):

    print("Loading CSV...")
    df_csv = load_csv(csv_path)
    print(f"  {len(df_csv)} infographics in CSV.")

    print(f"\nScanning image folder: {images_folder}")
    image_map = find_images(images_folder)
    print(f"  {len(image_map)} images found.")

    # Find which stimulus IDs exist in both
    common_ids = sorted(set(image_map.keys()) & set(df_csv.index))
    print(f"  {len(common_ids)} images matched to CSV rows by Stimulus ID.")

    if len(common_ids) == 0:
        print("\nERROR: No images matched. Check that filenames contain the "
              "Stimulus ID number (e.g. '42.png' or 'infographic_42.jpg').")
        return

    # ---------------------------------------------------------------------------
    # Extract features from all matched images
    # ---------------------------------------------------------------------------
    print(f"\nExtracting features from {len(common_ids)} images...")
    records = []
    failed = []

    for i, sid in enumerate(common_ids):
        path = image_map[sid]
        try:
            feats = normalize_features(extract_features(path))
            feats["Stimulus ID"] = sid
            feats["image_path"] = path
            feats[TARGET_COL] = df_csv.loc[sid, TARGET_COL]
            records.append(feats)
            print(f"  [{i+1}/{len(common_ids)}] ID {sid:4d} — appeal={feats[TARGET_COL]:.2f} ✓")
        except Exception as e:
            print(f"  [{i+1}/{len(common_ids)}] ID {sid:4d} — FAILED: {e}")
            failed.append(sid)

    if not records:
        print("\nNo features extracted successfully. Exiting.")
        return

    df_extracted = pd.DataFrame(records).set_index("Stimulus ID")
    print(f"\nSuccessfully extracted: {len(records)}  |  Failed: {len(failed)}")

    # ---------------------------------------------------------------------------
    # Compare extracted features vs. paper's pre-computed values
    # ---------------------------------------------------------------------------
    print("\n--- Feature comparison: extracted vs. CSV (mean absolute difference) ---")
    comparable = [c for c in FEATURE_COLS if c in df_extracted.columns and c in df_csv.columns]
    if not comparable:
        print("  Skipped: the CSV does not contain the paper's pre-computed feature columns.")
        print("  (This is expected when using an individual-ratings CSV.)")
    else:
        # Normalize CSV features the same way as extracted features for a fair comparison
        df_csv_norm = df_csv.copy()
        total_area = df_csv_norm['NonTextArea'] + df_csv_norm['TextArea']
        df_csv_norm['NonTextArea'] = df_csv_norm['NonTextArea'] / total_area.clip(lower=1)
        df_csv_norm['TextArea']    = df_csv_norm['TextArea']    / total_area.clip(lower=1)
        df_csv_norm['QuadTree']    = np.log(df_csv_norm['QuadTree'] + 1)

        for col in comparable:
            extracted_vals = df_extracted[col]
            csv_vals = df_csv_norm.loc[df_extracted.index, col]
            csv_mean = csv_vals.mean()
            if csv_mean == 0:
                continue
            rel_diff = (extracted_vals - csv_vals).abs().mean() / csv_mean * 100
            corr = extracted_vals.corr(csv_vals)
            print(f"  {col:20s}  correlation={corr:+.3f}  mean_rel_diff={rel_diff:.1f}%")

    # ---------------------------------------------------------------------------
    # Train / test split using EXTRACTED features → predict appeal
    # ---------------------------------------------------------------------------
    print("\n--- Model evaluation using extracted features ---")

    X = df_extracted[FEATURE_COLS].values
    y = df_extracted[TARGET_COL].values

    if len(X) < 10:
        print("  Too few samples for a train/test split. Need at least 10 matched images.")
    else:
        indices = df_extracted.index.values
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, indices, test_size=0.2, random_state=42
        )
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred = np.clip(y_pred, 1.0, 9.0)

        r2  = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        errors = np.abs(y_pred - y_test)
        within_05 = (errors <= 0.5).mean() * 100
        within_10 = (errors <= 1.0).mean() * 100

        print(f"\n  {'─'*42}")
        print(f"  {'ACCURACY SUMMARY':^42}")
        print(f"  {'─'*42}")
        print(f"  Train size      : {len(X_train)}  |  Test size: {len(X_test)}")
        print(f"  R²  (test)      : {r2:.3f}  (1.0 = perfect)")
        print(f"  MAE (test)      : {mae:.3f}  (on 1–9 scale)")
        print(f"  Within ±0.5 pts : {within_05:.1f}%")
        print(f"  Within ±1.0 pts : {within_10:.1f}%")
        print(f"  {'─'*42}")

        print("\n  Predicted vs. actual (test set):")
        print(f"  {'Stimulus ID':>12}  {'Actual':>8}  {'Predicted':>10}  {'Error':>8}")
        for sid, actual, pred in zip(idx_test, y_test, y_pred):
            err = pred - actual
            flag = "  ✓" if abs(err) <= 1.0 else "  ✗"
            print(f"  {sid:>12}  {actual:>8.2f}  {pred:>10.2f}  {err:>+8.2f}{flag}")

        # Save model trained on full extracted data
        pipeline_full = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ])
        pipeline_full.fit(X, y)
        with open("model_from_images.pkl", "wb") as f:
            pickle.dump({
                "pipeline": pipeline_full,
                "feature_cols": FEATURE_COLS,
                "test_r2": r2,
                "test_mae": mae,
            }, f)
        print("\n  Model trained on extracted features saved to: model_from_images.pkl")

    # ---------------------------------------------------------------------------
    # Save full results CSV
    # ---------------------------------------------------------------------------
    output_df = df_extracted[FEATURE_COLS + [TARGET_COL, "image_path"]].copy()

    # Add CSV's original feature values alongside for comparison
    for col in FEATURE_COLS:
        output_df[f"{col}_csv"] = df_csv.loc[df_extracted.index, col]

    output_df.to_csv(output_path)
    print(f"\nResults saved to: {output_path}")
    print("Columns: extracted features, original CSV features, appeal score, image path.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=IMAGES_FOLDER)
    parser.add_argument("--csv",    default=DATA_CSV)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv"))
    args = parser.parse_args()

    print()
    print("=" * 50)
    print("INFOGRAPHIC AESTHETICS BATCH ANALYSER")
    print("=" * 50)
    print(f"  Images : {args.images}")
    print(f"  CSV    : {args.csv}")
    print(f"  Output : {args.output}")
    print()

    run(args.images, args.csv, args.output)
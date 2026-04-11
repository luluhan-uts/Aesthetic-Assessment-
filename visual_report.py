"""
visual_report.py
----------------
Generates a visual analysis report for a single infographic image:

  Left panel  — original image with detected text regions (blue boxes) and
                graphic regions (green boxes) overlaid
  Right panel — feature profile: each of the 9 features compared to the
                dataset mean, coloured by whether the value boosts or
                reduces predicted appeal (requires data.csv for z-scoring)
  Bottom      — predicted appeal score on a 1–9 colour gradient gauge

Output: a single PNG saved to disk.

Usage (interactive file picker — recommended):
    python visual_report.py

Usage (command-line):
    python visual_report.py --image my_infographic.png
    python visual_report.py --image my_infographic.png --model model.pkl \\
                            --csv data.csv --output report.png
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save to file without opening a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from config import FEATURE_COLS, DATA_CSV, MODEL_FROM_IMAGES_PKL
from feature_extractor import extract_features


# ---------------------------------------------------------------------------
# Region detection
# Mirrors the classification logic in feature_extractor.compute_space_based
# so we can obtain bounding boxes for the visual overlay.
# ---------------------------------------------------------------------------

def detect_regions(img_bgr: np.ndarray):
    """
    Returns (text_boxes, image_boxes) — each a list of (x, y, w, h) tuples.
    Uses the same connected-component + edge-density heuristic as
    feature_extractor.compute_space_based.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    _, binary = cv2.threshold(inv, 15, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)

    text_boxes, image_boxes = [], []
    min_area = 100

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        patch = img_bgr[y:y + bh, x:x + bw]
        if patch.size == 0:
            continue
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_patch, 50, 150)
        edge_density = float(edges.mean())
        color_std = float(patch.std(axis=(0, 1)).mean())
        # High edges + low colour variance → text; otherwise → graphic region
        if edge_density > 8 and color_std < 60:
            text_boxes.append((x, y, bw, bh))
        else:
            image_boxes.append((x, y, bw, bh))

    return text_boxes, image_boxes


def annotate_image(img_bgr: np.ndarray,
                   text_boxes: list, image_boxes: list) -> np.ndarray:
    """Return an RGB copy of the image with coloured bounding boxes drawn."""
    out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    for (x, y, w, h) in image_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (34, 139, 34), 2)   # green
    for (x, y, w, h) in text_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (70, 130, 180), 2)  # blue
    return out


def score_label(score: float) -> str:
    if score < 3.0:  return "Low Appeal"
    if score < 4.5:  return "Below Average"
    if score < 5.5:  return "Average"
    if score < 7.0:  return "Above Average"
    return "High Appeal"


# ---------------------------------------------------------------------------
# Dataset statistics  (enables z-score comparison in the feature chart)
# ---------------------------------------------------------------------------

def load_dataset_stats(csv_path: str):
    """
    Loads data.csv and returns (mean_series, std_series) for FEATURE_COLS.
    Returns (None, None) if the file is missing or lacks feature columns.
    """
    if not csv_path or not os.path.isfile(csv_path):
        return None, None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, header=0)
        df.columns = df.columns.str.strip()
        df = df[pd.to_numeric(df["Stimulus ID"], errors="coerce").notna()]
        present = [c for c in FEATURE_COLS if c in df.columns]
        if not present:
            return None, None
        for c in present:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[present].mean(), df[present].std()
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(image_path: str, model_path: str,
                    csv_path: str, output_path: str) -> None:

    # ── Load model ──────────────────────────────────────────────────────────
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    pipeline   = bundle["pipeline"]
    cv_r2      = bundle.get("cv_r2")
    model_name = bundle.get("model_name", "model")

    # ── Load & resize image (match feature_extractor behaviour) ─────────────
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    # ── Extract features & predict ──────────────────────────────────────────
    print("Extracting features...")
    features = extract_features(image_path)
    X = np.array([[features.get(col, 0) for col in FEATURE_COLS]])
    score = float(np.clip(pipeline.predict(X)[0], 1.0, 9.0))
    print(f"  Predicted appeal: {score:.2f} / 9.0  ({score_label(score)})")

    # ── Region detection for annotation overlay ──────────────────────────────
    print("Detecting regions...")
    text_boxes, image_boxes = detect_regions(img_bgr)
    annotated_rgb = annotate_image(img_bgr, text_boxes, image_boxes)
    print(f"  Graphic regions: {len(image_boxes)}  |  Text regions: {len(text_boxes)}")

    # ── Dataset statistics for z-score comparison ────────────────────────────
    ds_mean, ds_std = load_dataset_stats(csv_path)
    if ds_mean is not None:
        print("  Dataset stats loaded — feature chart will show z-scores.")
    else:
        print("  No dataset stats — feature chart will show raw values.")

    # ── Model coefficients (determine good/bad bar colours) ──────────────────
    inner = pipeline.named_steps.get("model")
    if hasattr(inner, "coef_"):
        coefs = dict(zip(FEATURE_COLS, inner.coef_))
        has_direction = True
    else:
        # GradientBoosting: importances are unsigned — can't infer direction
        coefs = {}
        has_direction = False

    # ── Theme ────────────────────────────────────────────────────────────────
    BG    = "#1a1a2e"
    PANEL = "#16213e"
    FG    = "#eaeaea"
    GREEN = "#4ade80"
    RED   = "#f87171"
    BLUE  = "#60a5fa"

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[0.06, 1, 0.10],
        width_ratios=[1.1, 0.9],
        hspace=0.22, wspace=0.06,
        left=0.03, right=0.98, top=0.97, bottom=0.04,
    )

    # ── Title bar ────────────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(BG)
    ax_title.axis("off")
    r2_str = f"   ·   Model CV R²={cv_r2:.3f}" if cv_r2 is not None else ""
    ax_title.text(
        0.5, 0.5,
        f"{os.path.basename(image_path)}   │   "
        f"Predicted appeal: {score:.2f} / 9.0   ·   {score_label(score)}{r2_str}",
        transform=ax_title.transAxes,
        ha="center", va="center",
        fontsize=13, color=FG, fontweight="bold",
    )

    # ── Annotated image (left) ───────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[1, 0])
    ax_img.set_facecolor(PANEL)
    ax_img.imshow(annotated_rgb)
    ax_img.axis("off")
    ax_img.set_title("Detected regions", color=FG, fontsize=10, pad=6)
    ax_img.legend(
        handles=[
            mpatches.Patch(color=np.array([34, 139, 34]) / 255,
                           label=f"Graphic regions  ({len(image_boxes)})"),
            mpatches.Patch(color=np.array([70, 130, 180]) / 255,
                           label=f"Text regions  ({len(text_boxes)})"),
        ],
        loc="lower left", fontsize=8, framealpha=0.75,
        facecolor=PANEL, labelcolor=FG,
    )

    # ── Feature chart (right) ────────────────────────────────────────────────
    ax_feat = fig.add_subplot(gs[1, 1])
    ax_feat.set_facecolor(PANEL)
    for sp in ax_feat.spines.values():
        sp.set_color("#2a2a4a")

    feat_vals = [features.get(col, 0) for col in FEATURE_COLS]

    if ds_mean is not None:
        bar_vals = []
        for col, val in zip(FEATURE_COLS, feat_vals):
            if col in ds_mean.index and ds_std[col] > 0:
                bar_vals.append((val - ds_mean[col]) / ds_std[col])
            else:
                bar_vals.append(0.0)
        xlabel = "Standard deviations from dataset mean  (0 = average infographic)"
        ax_feat.axvline(0, color="#888", linewidth=0.8, linestyle="--", zorder=0)
    else:
        bar_vals = feat_vals
        xlabel = "Raw feature value  (provide data.csv for z-score comparison)"

    # Colour each bar: green if the value is in the "good for appeal" direction
    if has_direction:
        bar_colors = [
            GREEN if (coefs.get(col, 0) > 0) == (val >= 0) else RED
            for col, val in zip(FEATURE_COLS, bar_vals)
        ]
    else:
        bar_colors = [BLUE] * len(FEATURE_COLS)

    # Draw bars top-to-bottom (reverse so index 0 is at the top)
    labels_rev = FEATURE_COLS[::-1]
    vals_rev   = bar_vals[::-1]
    colors_rev = bar_colors[::-1]

    bars = ax_feat.barh(labels_rev, vals_rev,
                        color=colors_rev, edgecolor="none", height=0.55, zorder=2)

    # Value label on each bar
    x_range = max(abs(v) for v in bar_vals) if bar_vals else 1
    for bar, val in zip(bars, vals_rev):
        bw = bar.get_width()
        offset = (x_range or 1) * 0.04
        ax_feat.text(
            bw + (offset if bw >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}" if ds_mean is not None else f"{val:.1f}",
            va="center", ha="left" if bw >= 0 else "right",
            fontsize=7.5, color=FG,
        )

    ax_feat.set_xlabel(xlabel, color=FG, fontsize=7.5)
    ax_feat.set_title("Feature profile vs. dataset", color=FG, fontsize=10, pad=6)
    ax_feat.tick_params(colors=FG, labelsize=9)
    ax_feat.xaxis.label.set_color(FG)

    if has_direction:
        ax_feat.legend(
            handles=[
                mpatches.Patch(color=GREEN, label="Boosts appeal"),
                mpatches.Patch(color=RED,   label="Reduces appeal"),
            ],
            loc="lower right", fontsize=8, framealpha=0.75,
            facecolor=PANEL, labelcolor=FG,
        )

    # ── Score gauge (full width, bottom) ─────────────────────────────────────
    ax_gauge = fig.add_subplot(gs[2, :])
    ax_gauge.set_facecolor(BG)
    for sp in ax_gauge.spines.values():
        sp.set_visible(False)

    gradient = np.linspace(0, 1, 512).reshape(1, -1)
    ax_gauge.imshow(gradient, aspect="auto", cmap="RdYlGn",
                    extent=[1, 9, 0, 1])

    # Marker line + score label
    ax_gauge.axvline(score, color="white", linewidth=3, zorder=3)
    label_ha  = "left"  if score < 7.5 else "right"
    label_off = 0.1     if score < 7.5 else -0.1
    ax_gauge.text(
        score + label_off, 0.5,
        f"{score:.2f} / 9  ·  {score_label(score)}",
        ha=label_ha, va="center",
        fontsize=10, color="white", fontweight="bold", zorder=4,
        bbox=dict(facecolor="#00000070", edgecolor="none",
                  boxstyle="round,pad=0.35"),
    )

    ax_gauge.set_xlim(1, 9)
    ax_gauge.set_ylim(0, 1)
    ax_gauge.set_yticks([])
    ax_gauge.set_xticks(range(1, 10))
    ax_gauge.tick_params(colors=FG, labelsize=9)
    ax_gauge.set_xlabel("Appeal score  (1 = lowest  ·  9 = highest)",
                         color=FG, fontsize=9)

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"\nReport saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a visual appeal report for an infographic image."
    )
    parser.add_argument("--image",  help="Path to infographic image")
    parser.add_argument("--model",  default=MODEL_FROM_IMAGES_PKL,
                        help="Path to trained model (default: model_from_images.pkl)")
    parser.add_argument("--csv",    default=DATA_CSV,
                        help="Path to data.csv for z-score comparison")
    parser.add_argument("--output", help="Output PNG path (default: <image>_report.png)")
    args = parser.parse_args()

    if args.image:
        # Command-line mode
        image_path  = args.image
        model_path  = args.model
        csv_path    = args.csv
        stem        = os.path.splitext(os.path.basename(image_path))[0]
        output_path = args.output or f"{stem}_report.png"
    else:
        # Interactive: prompt for image only — model and CSV are automatic
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.focus_force()

        print()
        print("=" * 50)
        print("INFOGRAPHIC VISUAL REPORT GENERATOR")
        print("=" * 50)
        print(f"  Model : {args.model}")
        print(f"  CSV   : {args.csv}")
        print()

        input("Press Enter to select your INFOGRAPHIC IMAGE...")
        image_path = filedialog.askopenfilename(
            title="Select infographic image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp")],
        )
        if not image_path:
            print("No image selected. Exiting.")
            exit()
        print(f"  Selected: {image_path}")

        model_path  = args.model
        csv_path    = args.csv
        stem        = os.path.splitext(os.path.basename(image_path))[0]
        output_path = args.output or os.path.join(
            os.path.dirname(image_path), f"{stem}_report.png"
        )
        print(f"  Report will be saved to: {output_path}")

        print()
        print("=" * 50)
        print("Generating report...")
        print("=" * 50)
        print()

    generate_report(image_path, model_path, csv_path, output_path)

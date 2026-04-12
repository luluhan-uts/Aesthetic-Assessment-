"""
batch_compare.py
----------------
Batch-processes a folder of infographic images and for each one saves a
side-by-side comparison PNG:

  LEFT  — visual report using the paper's pre-computed features (from data.csv)
  RIGHT — visual report using features freshly extracted from the image

Each panel contains:
  · Annotated image (text regions = blue, graphic regions = green)
  · Feature profile chart (z-scores vs. dataset mean, coloured by appeal impact)
  · Predicted appeal score gauge (1–9)

All PNGs are saved to a single output folder named
    {stimulus_id:04d}_comparison.png  (e.g. 0042_comparison.png)

Usage (interactive file picker — recommended):
    python batch_compare.py

Usage (command-line):
    python batch_compare.py --images ./infographics --csv data.csv \\
                            --model model_from_images.pkl --output ./comparisons

NOTE: data.csv must be the per-infographic summary CSV that contains the
paper's pre-computed feature columns (ImageArea, TextGroup, …).
The individual-ratings CSV does not have these and cannot be used here.
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

from config import FEATURE_COLS, TARGET_COL, DATA_CSV, IMAGES_FOLDER, MODEL_FROM_IMAGES_PKL
from feature_extractor import extract_features, normalize_features
from batch_analyse import load_csv, find_images   # reuse existing helpers


# ---------------------------------------------------------------------------
# Theme  (shared across all panels)
# ---------------------------------------------------------------------------

_BG    = "#1a1a2e"
_PANEL = "#16213e"
_FG    = "#eaeaea"
_GREEN = "#4ade80"
_RED   = "#f87171"
_BLUE  = "#60a5fa"
_SEP   = "#2a2a4a"


# ---------------------------------------------------------------------------
# Region detection
# Mirrors the classification logic in feature_extractor.compute_space_based
# so we can obtain bounding boxes for the annotation overlay.
# ---------------------------------------------------------------------------

def detect_regions(img_bgr: np.ndarray):
    """Returns (text_boxes, image_boxes) — each a list of (x, y, w, h)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    _, binary = cv2.threshold(inv, 15, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)

    text_boxes, image_boxes = [], []
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < 100:
            continue
        patch = img_bgr[y:y + bh, x:x + bw]
        if patch.size == 0:
            continue
        gray_p       = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edge_density = float(cv2.Canny(gray_p, 50, 150).mean())
        color_std    = float(patch.std(axis=(0, 1)).mean())
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
# draw_side — renders one complete panel into its three axes
# ---------------------------------------------------------------------------

def draw_side(ax_img, ax_feat, ax_gauge,
              annotated_rgb: np.ndarray,
              feat_dict: dict,
              ds_mean, ds_std,
              coefs: dict, has_direction: bool,
              score: float,
              side_title: str,
              text_boxes: list, image_boxes: list) -> None:
    """
    Draws a single visual-report panel (annotated image + feature chart +
    score gauge) into the three supplied axes.

    Parameters
    ----------
    ax_img, ax_feat, ax_gauge
        Matplotlib Axes for the three sub-panels.
    annotated_rgb
        Pre-computed annotated image (RGB numpy array).
    feat_dict
        Mapping of feature name → value for this side
        (either paper's CSV values or freshly extracted values).
    ds_mean, ds_std
        Dataset mean/std pandas Series for z-score computation,
        or None to show raw values.
    coefs
        Model coefficient dict {feature_name: coef} for bar colouring.
    has_direction
        True if coefs carry meaningful sign (Ridge), False otherwise (GBM).
    score
        Predicted appeal score (already clipped to [1, 9]).
    side_title
        Label shown above the annotated image.
    text_boxes, image_boxes
        Region bounding boxes for the legend counts.
    """

    # — Annotated image ———————————————————————————————————————————————————
    ax_img.set_facecolor(_PANEL)
    ax_img.imshow(annotated_rgb)
    ax_img.axis("off")
    ax_img.set_title(side_title, color=_FG, fontsize=10, pad=6, fontweight="bold")
    ax_img.legend(
        handles=[
            mpatches.Patch(color=np.array([34, 139, 34]) / 255,
                           label=f"Graphic regions  ({len(image_boxes)})"),
            mpatches.Patch(color=np.array([70, 130, 180]) / 255,
                           label=f"Text regions  ({len(text_boxes)})"),
        ],
        loc="lower left", fontsize=7, framealpha=0.75,
        facecolor=_PANEL, labelcolor=_FG,
    )

    # — Feature chart —————————————————————————————————————————————————————
    ax_feat.set_facecolor(_PANEL)
    for sp in ax_feat.spines.values():
        sp.set_color(_SEP)

    feat_vals = [float(feat_dict.get(col, 0)) for col in FEATURE_COLS]

    if ds_mean is not None:
        bar_vals = [
            (val - ds_mean[col]) / ds_std[col]
            if col in ds_mean.index and ds_std[col] > 0 else 0.0
            for col, val in zip(FEATURE_COLS, feat_vals)
        ]
        xlabel = "Std deviations from dataset mean"
        ax_feat.axvline(0, color="#888", linewidth=0.8, linestyle="--", zorder=0)
    else:
        bar_vals = feat_vals
        xlabel = "Feature value"

    if has_direction:
        bar_colors = [
            _GREEN if (coefs.get(col, 0) > 0) == (v >= 0) else _RED
            for col, v in zip(FEATURE_COLS, bar_vals)
        ]
    else:
        bar_colors = [_BLUE] * len(FEATURE_COLS)

    bars = ax_feat.barh(
        FEATURE_COLS[::-1], bar_vals[::-1],
        color=bar_colors[::-1], edgecolor="none", height=0.55, zorder=2,
    )

    x_range = max(abs(v) for v in bar_vals) if bar_vals else 1
    for bar, val in zip(bars, bar_vals[::-1]):
        bw  = bar.get_width()
        off = (x_range or 1) * 0.05
        ax_feat.text(
            bw + (off if bw >= 0 else -off),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}" if ds_mean is not None else f"{val:.1f}",
            va="center", ha="left" if bw >= 0 else "right",
            fontsize=7, color=_FG,
        )

    ax_feat.set_xlabel(xlabel, color=_FG, fontsize=7)
    ax_feat.tick_params(colors=_FG, labelsize=8)
    ax_feat.xaxis.label.set_color(_FG)

    if has_direction:
        ax_feat.legend(
            handles=[
                mpatches.Patch(color=_GREEN, label="Boosts appeal"),
                mpatches.Patch(color=_RED,   label="Reduces appeal"),
            ],
            loc="lower right", fontsize=7, framealpha=0.75,
            facecolor=_PANEL, labelcolor=_FG,
        )

    # — Score gauge ————————————————————————————————————————————————————————
    ax_gauge.set_facecolor(_BG)
    for sp in ax_gauge.spines.values():
        sp.set_visible(False)

    gradient = np.linspace(0, 1, 512).reshape(1, -1)
    ax_gauge.imshow(gradient, aspect="auto", cmap="RdYlGn",
                    extent=[1, 9, 0, 1])
    ax_gauge.axvline(score, color="white", linewidth=3, zorder=3)

    label_ha  = "left"  if score < 7.5 else "right"
    label_off = 0.12    if score < 7.5 else -0.12
    ax_gauge.text(
        score + label_off, 0.5,
        f"{score:.2f}  ·  {score_label(score)}",
        ha=label_ha, va="center",
        fontsize=9, color="white", fontweight="bold", zorder=4,
        bbox=dict(facecolor="#00000070", edgecolor="none",
                  boxstyle="round,pad=0.3"),
    )
    ax_gauge.set_xlim(1, 9)
    ax_gauge.set_ylim(0, 1)
    ax_gauge.set_yticks([])
    ax_gauge.set_xticks(range(1, 10))
    ax_gauge.tick_params(colors=_FG, labelsize=8)
    ax_gauge.set_xlabel("Appeal score  (1 = lowest  ·  9 = highest)",
                         color=_FG, fontsize=8)


# ---------------------------------------------------------------------------
# generate_comparison — builds the full figure for one image
# ---------------------------------------------------------------------------

def generate_comparison(image_path: str,
                        stimulus_id: int,
                        paper_feats: dict,
                        bundle: dict,
                        ds_mean, ds_std,
                        output_path: str) -> tuple:
    """
    Generates and saves a side-by-side comparison PNG for a single image.

    Parameters
    ----------
    image_path   : path to the infographic image file
    stimulus_id  : integer Stimulus ID (used in title)
    paper_feats  : dict of paper's pre-computed feature values for this ID
    bundle       : loaded model bundle dict (from pickle)
    ds_mean      : dataset mean Series for z-scoring (or None)
    ds_std       : dataset std Series for z-scoring (or None)
    output_path  : where to save the PNG

    Returns
    -------
    (score_paper, score_extracted) — predicted scores for both panels
    """
    pipeline = bundle["pipeline"]
    cv_r2    = bundle.get("cv_r2")

    # — Load & resize image ————————————————————————————————————————————————
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    # — Extract features ———————————————————————————————————————————————————
    extracted_feats = normalize_features(extract_features(image_path))

    # — Predict scores ————————————————————————————————————————————————————
    def predict(feat_dict):
        X = np.array([[float(feat_dict.get(col, 0)) for col in FEATURE_COLS]])
        return float(np.clip(pipeline.predict(X)[0], 1.0, 9.0))

    score_paper     = predict(paper_feats)
    score_extracted = predict(extracted_feats)

    # — Region detection (identical for both sides — same image) ———————————
    text_boxes, image_boxes = detect_regions(img_bgr)
    annotated_rgb = annotate_image(img_bgr, text_boxes, image_boxes)

    # — Model coefficients ————————————————————————————————————————————————
    inner = pipeline.named_steps.get("model")
    if hasattr(inner, "coef_"):
        coefs         = dict(zip(FEATURE_COLS, inner.coef_))
        has_direction = True
    else:
        coefs         = {}
        has_direction = False

    # ── Figure layout ─────────────────────────────────────────────────────
    # 3 rows × 5 columns:
    #   col 0-1  = left panel  (paper features)
    #   col 2    = vertical separator
    #   col 3-4  = right panel (extracted features)
    fig = plt.figure(figsize=(30, 12), facecolor=_BG)
    gs = GridSpec(
        3, 5, figure=fig,
        height_ratios=[0.07, 1, 0.10],
        width_ratios=[1.1, 0.9, 0.015, 1.1, 0.9],
        hspace=0.25, wspace=0.28,
        left=0.02, right=0.99, top=0.97, bottom=0.04,
    )

    # — Overall title ——————————————————————————————————————————————————————
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(_BG)
    ax_title.axis("off")
    r2_str = f"   ·   Model CV R²={cv_r2:.3f}" if cv_r2 is not None else ""
    diff   = score_extracted - score_paper
    ax_title.text(
        0.5, 0.5,
        f"Stimulus ID {stimulus_id}  —  {os.path.basename(image_path)}{r2_str}\n"
        f"Paper score: {score_paper:.2f}  ({score_label(score_paper)})     "
        f"Extracted score: {score_extracted:.2f}  ({score_label(score_extracted)})     "
        f"Δ = {diff:+.2f}",
        transform=ax_title.transAxes,
        ha="center", va="center",
        fontsize=11, color=_FG, fontweight="bold",
        linespacing=1.7,
    )

    # — Vertical separator ————————————————————————————————————————————————
    ax_sep = fig.add_subplot(gs[1:, 2])
    ax_sep.set_facecolor(_SEP)
    ax_sep.axis("off")

    # — Left panel: paper features ————————————————————————————————————————
    draw_side(
        fig.add_subplot(gs[1, 0]),    # annotated image
        fig.add_subplot(gs[1, 1]),    # feature chart
        fig.add_subplot(gs[2, 0:2]),  # gauge spanning both left columns
        annotated_rgb, paper_feats,
        ds_mean, ds_std, coefs, has_direction,
        score_paper,
        "Paper features  (data.csv)",
        text_boxes, image_boxes,
    )

    # — Right panel: extracted features ———————————————————————————————————
    draw_side(
        fig.add_subplot(gs[1, 3]),    # annotated image
        fig.add_subplot(gs[1, 4]),    # feature chart
        fig.add_subplot(gs[2, 3:5]), # gauge spanning both right columns
        annotated_rgb, extracted_feats,
        ds_mean, ds_std, coefs, has_direction,
        score_extracted,
        "Extracted features  (from image)",
        text_boxes, image_boxes,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close()
    return score_paper, score_extracted


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run(images_folder: str, csv_path: str,
        model_path: str, output_folder: str) -> None:

    os.makedirs(output_folder, exist_ok=True)

    # — Load and validate CSV ——————————————————————————————————————————————
    print("Loading CSV...")
    df_csv = load_csv(csv_path)
    print(f"  {len(df_csv)} infographics in CSV.")

    missing_cols = [c for c in FEATURE_COLS if c not in df_csv.columns]
    if missing_cols:
        print(f"\nERROR: data.csv is missing feature columns: {missing_cols}")
        print("This tool requires the per-infographic summary CSV with pre-computed features.")
        print("The individual-ratings CSV does not have these columns.")
        return

    ds_mean = df_csv[FEATURE_COLS].mean()
    ds_std  = df_csv[FEATURE_COLS].std()

    # — Load model ————————————————————————————————————————————————————————
    print("Loading model...")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    cv_r2 = bundle.get("cv_r2")
    if cv_r2 is not None:
        print(f"  Model CV R² = {cv_r2:.3f}")

    # — Find and match images ——————————————————————————————————————————————
    print(f"\nScanning: {images_folder}")
    image_map  = find_images(images_folder)
    print(f"  {len(image_map)} images found.")

    common_ids = sorted(set(image_map.keys()) & set(df_csv.index))
    print(f"  {len(common_ids)} matched to CSV rows.")

    if not common_ids:
        print("\nERROR: No images matched. Check that filenames contain the Stimulus ID.")
        return

    # — Generate reports ———————————————————————————————————————————————————
    n         = len(common_ids)
    width     = len(str(n))
    succeeded = []
    failed    = []

    print(f"\nGenerating {n} comparison reports → {output_folder}")
    print("-" * 60)

    actuals          = []
    preds_extracted  = []

    for i, sid in enumerate(common_ids):
        path        = image_map[sid]
        out_path    = os.path.join(output_folder, f"{sid:04d}_comparison.png")
        paper_feats = normalize_features({col: float(df_csv.loc[sid, col]) for col in FEATURE_COLS})

        try:
            _, score_e = generate_comparison(path, sid, paper_feats, bundle,
                                                   ds_mean, ds_std, out_path)
            actual = float(df_csv.loc[sid, TARGET_COL])
            actuals.append(actual)
            preds_extracted.append(score_e)
            print(f"  [{i+1:>{width}}/{n}]  ID {sid:4d}  actual={actual:.2f}  pred={score_e:.2f}  ✓")
            succeeded.append(sid)
        except Exception as e:
            print(f"  [{i+1:>{width}}/{n}]  ID {sid:4d}  ✗  {e}")
            failed.append(sid)

    print("-" * 60)
    print(f"Complete.  Saved: {len(succeeded)}  |  Failed: {len(failed)}")
    if failed:
        print(f"Failed IDs: {failed}")
    print(f"\nAll reports saved to: {output_folder}")

    # ── Accuracy summary ──────────────────────────────────────────────────────
    if len(actuals) >= 2:
        import numpy as np
        from sklearn.metrics import r2_score, mean_absolute_error

        y_true  = np.array(actuals)
        y_pred  = np.array(preds_extracted)
        errors  = np.abs(y_pred - y_true)
        r2      = r2_score(y_true, y_pred)
        mae     = mean_absolute_error(y_true, y_pred)
        w05     = (errors <= 0.5).mean() * 100
        w10     = (errors <= 1.0).mean() * 100

        print()
        print(f"  {'─'*44}")
        print(f"  {'ACCURACY SUMMARY (extracted features)':^44}")
        print(f"  {'─'*44}")
        print(f"  Images scored   : {len(actuals)}")
        print(f"  R²              : {r2:.3f}  (1.0 = perfect)")
        print(f"  MAE             : {mae:.3f}  (on 1–9 scale)")
        print(f"  Within ±0.5 pts : {w05:.1f}%")
        print(f"  Within ±1.0 pts : {w10:.1f}%")
        print(f"  {'─'*44}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Generate side-by-side paper vs. extracted feature comparison reports."
    )
    parser.add_argument("--images", default=IMAGES_FOLDER)
    parser.add_argument("--csv",    default=DATA_CSV)
    parser.add_argument("--model",  default=MODEL_FROM_IMAGES_PKL)
    parser.add_argument("--output", default=os.path.join(_here, "comparisons"))
    args = parser.parse_args()

    print()
    print("=" * 55)
    print("INFOGRAPHIC COMPARISON REPORT — BATCH GENERATOR")
    print("=" * 55)
    print(f"  Images : {args.images}")
    print(f"  CSV    : {args.csv}")
    print(f"  Model  : {args.model}")
    print(f"  Output : {args.output}")
    print()

    run(args.images, args.csv, args.model, args.output)

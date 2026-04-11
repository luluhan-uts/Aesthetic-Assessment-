"""
pipeline_diagram.py
-------------------
Generates a visual flowchart of the infographic aesthetics prediction pipeline,
annotating the CV techniques used in feature extraction and the ML models used
in training.

Usage:
    python pipeline_diagram.py
    python pipeline_diagram.py --output pipeline.png
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Colour palette ─────────────────────────────────────────────────────────────
BG      = "#0d0d1f"
FG      = "#e8e8f8"

C_IN    = "#0c2540"   # data inputs
C_PROC  = "#1a0b3b"   # processing scripts
C_FEAT  = "#062030"   # feature vector
C_MODEL = "#2a0a2a"   # trained model
C_APP   = "#181200"   # application scripts
C_OUT   = "#081508"   # output artefacts

E_IN    = "#4a9ede"
E_PROC  = "#9b59d6"
E_FEAT  = "#22d3ee"
E_MODEL = "#e040d0"
E_APP   = "#f5c518"
E_OUT   = "#4ade80"

ARROW   = "#8888aa"


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_box(ax, cx, cy, w, h, title,
             face, edge,
             subtitle=None, fs_title=9, fs_sub=7.0):
    """Draw a rounded rectangle with an optional subtitle below a divider."""
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.012",
        facecolor=face, edgecolor=edge,
        linewidth=1.3, zorder=3,
    ))
    if subtitle:
        ax.text(cx, cy + h * 0.23, title,
                ha="center", va="center",
                fontsize=fs_title, fontweight="bold", color=FG, zorder=4)
        ax.plot([cx - w * 0.38, cx + w * 0.38],
                [cy + h * 0.06,  cy + h * 0.06],
                color=edge, lw=0.6, alpha=0.5, zorder=4)
        ax.text(cx, cy - h * 0.20, subtitle,
                ha="center", va="center",
                fontsize=fs_sub, color=FG, alpha=0.88,
                linespacing=1.5, zorder=4, style="italic")
    else:
        ax.text(cx, cy, title,
                ha="center", va="center",
                fontsize=fs_title, fontweight="bold", color=FG, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=ARROW,
            lw=1.5, mutation_scale=12,
        ),
        zorder=2,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main(output_path: str) -> None:

    fig, ax = plt.subplots(figsize=(15, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Heading ───────────────────────────────────────────────────────────────
    ax.text(0.5, 0.967, "Infographic Aesthetics Prediction Pipeline",
            ha="center", va="center",
            fontsize=15, fontweight="bold", color=FG)
    ax.text(0.5, 0.944, "Harrison, Reinecke & Chang (CHI 2015) — Python replication",
            ha="center", va="center",
            fontsize=9, color=FG, alpha=0.55, style="italic")

    # ── Layout constants ──────────────────────────────────────────────────────
    XL, XR, XM = 0.22, 0.78, 0.50   # left / right / mid columns

    Y0 = 0.880   # inputs
    Y1 = 0.730   # processing layer 1
    Y2 = 0.555   # processing layer 2
    Y3 = 0.375   # trained model
    Y4 = 0.195   # application scripts
    Y5 = 0.055   # output artefacts

    H0, H1, H2, H3 = 0.080, 0.112, 0.133, 0.098
    W_MAIN = 0.27
    W_APP  = 0.185
    H_APP  = 0.080

    # ────────────────────────────────────────────────────────────────────────
    #  NODES
    # ────────────────────────────────────────────────────────────────────────

    # Row 0 — inputs
    draw_box(ax, XL, Y0, W_MAIN, H0,
             "Infographic Images",
             C_IN, E_IN,
             subtitle="PNG / JPG folder")

    draw_box(ax, XR, Y0, W_MAIN, H0,
             "data.csv",
             C_IN, E_IN,
             subtitle="326 infographics\npaper's features + appeal ratings (1–9)")

    # Row 1 — loading / first processing
    draw_box(ax, XL, Y1, W_MAIN, H1,
             "feature_extractor.py",
             C_PROC, E_PROC,
             subtitle="XY-Cut page segmentation\nQuadtree entropy decomposition\nHSV · CIELab chroma · RGB opponent channels")

    draw_box(ax, XR, Y1, W_MAIN, H1,
             "load_csv()",
             C_PROC, E_PROC,
             subtitle="Auto-detects CSV format\n(summary or individual ratings)\nAggregates to per-infographic means")

    # Row 2 — feature vector  +  model training
    draw_box(ax, XL, Y2, W_MAIN, H2,
             "9 Visual Features",
             C_FEAT, E_FEAT,
             subtitle="ImageArea  ·  TextGroup\nNonTextArea  ·  TextArea  ·  Complexity\nQuadTree  ·  Saturation\nColorfulness1 (Yendrikhovskij 1998)\nColorfulness2 (Hasler & Suesstrunk 2003)",
             fs_sub=6.7)

    draw_box(ax, XR, Y2, W_MAIN, H2,
             "train_model.py",
             C_PROC, E_PROC,
             subtitle="Ridge Regression  α = 1\nRidge Regression  α = 10\nGradient Boosting  (200 trees, depth 3)\n──────────────────────────\n5-fold cross-validation  →  best CV R² wins",
             fs_sub=6.7)

    # Row 3 — trained model (centre)
    draw_box(ax, XM, Y3, 0.32, H3,
             "model.pkl",
             C_MODEL, E_MODEL,
             subtitle="Best pipeline: StandardScaler + winning model\nSaved with feature list, CV R², train MAE\nCV R² ≈ 0.02  (paper: 0.34 — see README)")

    # Row 4 — application scripts
    X_APPS   = [0.105, 0.365, 0.635, 0.895]
    APPS     = [
        ("predict.py",        "Single image\n→ appeal score (1–9)\n+ feature printout"),
        ("batch_analyse.py",  "Image folder + data.csv\n→ feature comparison\n→ retrain + results.csv"),
        ("visual_report.py",  "Single image\n→ annotated PNG\n(regions + chart + gauge)"),
        ("batch_compare.py",  "Image folder + data.csv\n→ side-by-side PNGs\n(paper vs. extracted)"),
    ]
    for xc, (label, sub) in zip(X_APPS, APPS):
        draw_box(ax, xc, Y4, W_APP, H_APP,
                 label, C_APP, E_APP,
                 subtitle=sub, fs_title=8.2, fs_sub=6.5)

    # Row 5 — output artefacts
    OUTS = [
        "score  1–9\n+ label",
        "results.csv\nmodel_from_images.pkl",
        "<image>_report.png",
        "comparisons/\n{id}_comparison.png\n(one per image)",
    ]
    for xc, lbl in zip(X_APPS, OUTS):
        ax.add_patch(FancyBboxPatch(
            (xc - W_APP/2, Y5 - 0.034), W_APP, 0.068,
            boxstyle="round,pad=0.010",
            facecolor=C_OUT, edgecolor=E_OUT,
            linewidth=0.9, zorder=3,
        ))
        ax.text(xc, Y5, lbl,
                ha="center", va="center",
                fontsize=6.5, color=E_OUT, linespacing=1.4, zorder=4)

    # ────────────────────────────────────────────────────────────────────────
    #  ARROWS
    # ────────────────────────────────────────────────────────────────────────

    # Inputs → layer 1
    draw_arrow(ax, XL, Y0 - H0/2,       XL, Y1 + H1/2)
    draw_arrow(ax, XR, Y0 - H0/2,       XR, Y1 + H1/2)

    # Layer 1 → layer 2
    draw_arrow(ax, XL, Y1 - H1/2,       XL, Y2 + H2/2)
    draw_arrow(ax, XR, Y1 - H1/2,       XR, Y2 + H2/2)

    # Layer 2 → model.pkl  (diagonal, converging to centre)
    draw_arrow(ax, XL,       Y2 - H2/2, XM - 0.075, Y3 + H3/2)
    draw_arrow(ax, XR,       Y2 - H2/2, XM + 0.075, Y3 + H3/2)

    # model.pkl → fan line → app scripts
    Y_FAN = Y3 - H3/2 - 0.022
    ax.plot([XM, XM],            [Y3 - H3/2, Y_FAN], color=ARROW, lw=1.5, zorder=2)
    ax.plot([X_APPS[0], X_APPS[-1]], [Y_FAN, Y_FAN],  color=ARROW, lw=1.5, zorder=2)
    for xc in X_APPS:
        draw_arrow(ax, xc, Y_FAN, xc, Y4 + H_APP/2)

    # App scripts → outputs
    for xc in X_APPS:
        draw_arrow(ax, xc, Y4 - H_APP/2, xc, Y5 + 0.035)

    # ────────────────────────────────────────────────────────────────────────
    #  ANNOTATIONS — floating labels beside key nodes
    # ────────────────────────────────────────────────────────────────────────

    # "CV techniques" label beside feature_extractor arrow
    ax.annotate(
        "CV techniques",
        xy=(XL + W_MAIN/2, Y1),
        xytext=(XL + W_MAIN/2 + 0.13, Y1 + 0.04),
        fontsize=7.5, color=E_PROC, style="italic",
        arrowprops=dict(arrowstyle="-", color=E_PROC, lw=0.8),
        zorder=5,
    )

    # "5-fold CV" label beside train_model arrow
    ax.annotate(
        "model selection",
        xy=(XR - W_MAIN/2, Y2),
        xytext=(XR - W_MAIN/2 - 0.17, Y2 + 0.05),
        fontsize=7.5, color=E_PROC, style="italic",
        arrowprops=dict(arrowstyle="-", color=E_PROC, lw=0.8),
        zorder=5,
    )

    # "feature vector" label on left converging arrow
    ax.text(XL + 0.02, (Y2 - H2/2 + Y3 + H3/2) / 2 + 0.01,
            "feature\nvector",
            ha="left", va="center", fontsize=7, color=E_FEAT,
            style="italic", linespacing=1.3)

    # "paper features" label on right converging arrow
    ax.text(XR - 0.02, (Y2 - H2/2 + Y3 + H3/2) / 2 + 0.01,
            "paper\nfeatures",
            ha="right", va="center", fontsize=7, color=E_PROC,
            style="italic", linespacing=1.3)

    # ────────────────────────────────────────────────────────────────────────
    #  COLUMN HEADERS
    # ────────────────────────────────────────────────────────────────────────
    ax.text(XL, 0.920, "EXTRACTION  PIPELINE",
            ha="center", va="center", fontsize=7.5,
            color=E_FEAT, fontweight="bold", alpha=0.75,
            style="italic")
    ax.text(XR, 0.920, "TRAINING  PIPELINE",
            ha="center", va="center", fontsize=7.5,
            color=E_PROC, fontweight="bold", alpha=0.75,
            style="italic")

    # ────────────────────────────────────────────────────────────────────────
    #  LEGEND
    # ────────────────────────────────────────────────────────────────────────
    legend_items = [
        (C_IN,    E_IN,    "Data input"),
        (C_PROC,  E_PROC,  "Processing / script"),
        (C_FEAT,  E_FEAT,  "Feature vector"),
        (C_MODEL, E_MODEL, "Trained model"),
        (C_APP,   E_APP,   "Application script"),
        (C_OUT,   E_OUT,   "Output artefact"),
    ]
    lx, ly = 0.682, 0.455
    ax.text(lx, ly, "Legend",
            ha="left", va="center",
            fontsize=8, color=FG, fontweight="bold")
    for i, (fc, ec, label) in enumerate(legend_items):
        yy = ly - 0.030 * (i + 1)
        ax.add_patch(FancyBboxPatch(
            (lx, yy - 0.009), 0.030, 0.018,
            boxstyle="round,pad=0.005",
            facecolor=fc, edgecolor=ec,
            linewidth=0.9, zorder=5,
        ))
        ax.text(lx + 0.040, yy, label,
                ha="left", va="center",
                fontsize=7.5, color=FG)

    # ────────────────────────────────────────────────────────────────────────
    #  FOOTNOTE
    # ────────────────────────────────────────────────────────────────────────
    ax.text(0.01, 0.010,
            "CV R² gap (0.02 vs paper's 0.34): paper uses individual ratings with "
            "demographic interactions (Gender × Colorfulness, Age × Complexity) "
            "in a mixed-effects model — not replicable from per-infographic means alone.",
            ha="left", va="center",
            fontsize=6.2, color=FG, alpha=0.50, style="italic")

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"Pipeline diagram saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a visual flowchart of the prediction pipeline."
    )
    parser.add_argument("--output", default="pipeline.png",
                        help="Output PNG path (default: pipeline.png)")
    args = parser.parse_args()
    main(args.output)

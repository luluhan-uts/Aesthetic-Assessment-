"""
pipeline_flow.py  —  simple two-path pipeline diagram.

Usage:
    python pipeline_flow.py
    python pipeline_flow.py --output flow.png
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG  = "#0d0d1f"
FG  = "#e8e8f8"
DIM = "#667799"

A_F = "#0a1d35";  A_E = "#4a9ede"   # Path A  blue
B_F = "#211100";  B_E = "#f5a623"   # Path B  amber
M_F = "#180020";  M_E = "#dd44cc"   # Model   magenta
O_F = "#071406";  O_E = "#4ade80"   # Output  green


def box(ax, cx, cy, w, h, face, edge, title, sub=None, ft=9, fs=7):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.016",
        facecolor=face, edgecolor=edge, linewidth=1.6, zorder=3,
    ))
    if sub:
        ax.text(cx, cy + h * 0.19, title,
                ha="center", va="center",
                fontsize=ft, fontweight="bold", color=FG, zorder=4)
        ax.plot([cx - w*0.38, cx + w*0.38],
                [cy + h*0.04, cy + h*0.04],
                color=edge, lw=0.5, alpha=0.35, zorder=4)
        ax.text(cx, cy - h * 0.20, sub,
                ha="center", va="center",
                fontsize=fs, color=FG, alpha=0.78,
                linespacing=1.45, zorder=4, style="italic")
    else:
        ax.text(cx, cy, title,
                ha="center", va="center",
                fontsize=ft, fontweight="bold", color=FG, zorder=4)


def arr(ax, cx, y1, y2, color, lw=1.6):
    ax.annotate("", xy=(cx, y2), xytext=(cx, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=13), zorder=5)


def main(output_path):
    fig, ax = plt.subplots(figsize=(14, 13))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.967, "Infographic Aesthetics — Pipeline Flow",
            ha="center", fontsize=15, fontweight="bold", color=FG, zorder=6)

    # ── Lane backgrounds ──────────────────────────────────────────────────────
    ax.add_patch(plt.Rectangle((0.03, 0.03), 0.44, 0.910,
        facecolor=A_F, edgecolor=A_E, lw=1.1, alpha=0.20, zorder=0, ls="--"))
    ax.add_patch(plt.Rectangle((0.53, 0.03), 0.44, 0.910,
        facecolor=B_F, edgecolor=B_E, lw=1.1, alpha=0.20, zorder=0, ls="--"))

    # ── Lane headers ──────────────────────────────────────────────────────────
    ax.text(0.25, 0.932, "PATH A  —  EXISTING INFOGRAPHICS",
            ha="center", fontsize=11, fontweight="bold", color=A_E, zorder=6)
    ax.text(0.75, 0.932, "PATH B  —  NEW IMAGE",
            ha="center", fontsize=11, fontweight="bold", color=B_E, zorder=6)

    # ── Dimensions ───────────────────────────────────────────────────────────
    W  = 0.34    # full-width box
    Wh = 0.155   # half-width box
    H0 = 0.060   # input / output height
    H1 = 0.085   # process height
    H2 = 0.060   # model height

    # ══════════════════════════════════════════════════════════════════════════
    #  PATH A
    # ══════════════════════════════════════════════════════════════════════════
    L  = 0.25    # Path A centre x
    BL = 0.130   # left branch x  (results)
    BR = 0.375   # right branch x (batch_compare)

    # Row positions
    Ay = [0.875, 0.762, 0.655, 0.538, 0.422, 0.295, 0.170]
    #     inp   train  mdl   batch  mfim  fork   out

    # — Inputs (side by side) —
    box(ax, L - 0.095, Ay[0], Wh, H0, A_F, A_E,
        "training_images/", "326 PNG / JPG infographics", ft=8.5, fs=6.5)
    box(ax, L + 0.095, Ay[0], Wh, H0, A_F, A_E,
        "data.csv", "paper features + ratings (1–9)", ft=8.5, fs=6.5)

    # — train_model.py —
    box(ax, L, Ay[1], W, H1, A_F, A_E,
        "train_model.py",
        "Ridge · Gradient Boosting · 5-fold CV")

    # — model.pkl —
    box(ax, L, Ay[2], W, H2, M_F, M_E,
        "model.pkl", "Best trained model")

    # — batch_analyse.py —
    box(ax, L, Ay[3], W, H1, A_F, A_E,
        "batch_analyse.py",
        "Extract & compare features · evaluate accuracy")

    # — model_from_images.pkl —
    box(ax, L, Ay[4], W, H2, M_F, M_E,
        "model_from_images.pkl",
        "Model trained on extracted image features")

    # — Fork: results (left) | batch_compare (right) —
    box(ax, BL, Ay[5], Wh, H1, A_F, A_E,
        "results.csv",
        "Feature comparison\n+ accuracy summary", ft=8.5, fs=6.5)
    box(ax, BR, Ay[5], Wh, H1, A_F, A_E,
        "batch_compare.py",
        "Side-by-side visual\nreport per image", ft=8.5, fs=6.5)

    # — Output (right branch only) —
    box(ax, BR, Ay[6], Wh, H0, O_F, O_E,
        "Visual comparison results", ft=8)

    # ── Path A arrows ─────────────────────────────────────────────────────────
    # inputs fan → train_model
    ax.plot([L - 0.095, L - 0.095, L],
            [Ay[0] - H0/2, Ay[0] - H0/2 - 0.030, Ay[0] - H0/2 - 0.030],
            color=A_E, lw=1.5, zorder=2)
    ax.plot([L + 0.095, L + 0.095, L],
            [Ay[0] - H0/2, Ay[0] - H0/2 - 0.030, Ay[0] - H0/2 - 0.030],
            color=A_E, lw=1.5, zorder=2)
    arr(ax, L, Ay[0] - H0/2 - 0.030, Ay[1] + H1/2, A_E)

    arr(ax, L, Ay[1] - H1/2, Ay[2] + H2/2, M_E)   # train → model.pkl
    arr(ax, L, Ay[2] - H2/2, Ay[3] + H1/2, A_E)   # model → batch_analyse
    arr(ax, L, Ay[3] - H1/2, Ay[4] + H2/2, M_E)   # batch_analyse → model_from_images

    # model_from_images → fork (horizontal spread then down)
    FORK_Y = Ay[4] - H2/2 - 0.030
    ax.plot([L, L], [Ay[4] - H2/2, FORK_Y], color=A_E, lw=1.5, zorder=2)
    ax.plot([BL, BR], [FORK_Y, FORK_Y], color=A_E, lw=1.5, zorder=2)
    arr(ax, BL, FORK_Y, Ay[5] + H1/2, A_E)
    arr(ax, BR, FORK_Y, Ay[5] + H1/2, A_E)

    arr(ax, BR, Ay[5] - H1/2, Ay[6] + H0/2, O_E)  # batch_compare → output

    # ── Cross-lane: Path A model_from_images → Path B model_from_images ───────
    # Path A mfim centre: (L=0.25, Ay[4]=0.422), Path B mfim centre: (0.75, 0.690)
    x_start = L + W/2     # 0.25 + 0.17 = 0.42
    x_mid   = 0.500
    x_end   = 0.75 - W/2  # 0.75 - 0.17 = 0.58
    y_a     = Ay[4]       # 0.422
    y_b     = 0.690
    ax.plot([x_start, x_mid, x_mid, x_end],
            [y_a,    y_a,   y_b,   y_b],
            color=M_E, lw=1.4, ls="--", zorder=2)
    ax.annotate("", xy=(x_end, y_b), xytext=(x_end - 0.001, y_b),
                arrowprops=dict(arrowstyle="-|>", color=M_E,
                                lw=1.4, mutation_scale=12), zorder=5)

    # ══════════════════════════════════════════════════════════════════════════
    #  PATH B
    # ══════════════════════════════════════════════════════════════════════════
    R  = 0.75    # Path B centre x

    By = [0.875, 0.690, 0.500, 0.310]
    #     img   mfim   vr    out

    # — New image —
    box(ax, R, By[0], W, H0, B_F, B_E,
        "New Infographic Image", "PNG / JPG / WEBP", fs=6.8)

    # — model_from_images.pkl —
    box(ax, R, By[1], W, H2, M_F, M_E,
        "model_from_images.pkl",
        "Load model trained on extracted features")

    # — visual_report.py —
    box(ax, R, By[2], W, H1 + 0.020, B_F, B_E,
        "visual_report.py",
        "Annotated image overlay\nFeature bar chart · Appeal gauge  1–9")

    # — Output —
    box(ax, R, By[3], W, H0, O_F, O_E,
        "Image Aesthetic Report  (.png)", ft=8.5)

    # ── Path B arrows ─────────────────────────────────────────────────────────
    arr(ax, R, By[0] - H0/2, By[1] + H2/2, B_E)
    arr(ax, R, By[1] - H2/2, By[2] + (H1+0.020)/2, B_E)
    arr(ax, R, By[2] - (H1+0.020)/2, By[3] + H0/2, O_E)

    # ── Divider ───────────────────────────────────────────────────────────────
    ax.plot([0.5, 0.5], [0.05, 0.915],
            color="#2a2a55", lw=1.2, ls=":", zorder=1)

    # ── Footnote ──────────────────────────────────────────────────────────────
    ax.text(0.5, 0.048,
            "Path B requires model_from_images.pkl — run batch_analyse.py (Path A) first",
            ha="center", fontsize=7, color=DIM, style="italic", zorder=6)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="pipeline_flow.png")
    args = parser.parse_args()
    main(args.output)

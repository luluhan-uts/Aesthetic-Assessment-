"""
model_comparison_visualisation.py
---------------------------------
Visualises comparative model performance for Assignment 3.

Creates:
1. R² comparison chart
2. MAE comparison chart
3. RMSE comparison chart
4. Combined overview chart

Outputs saved to:
outputs/presentation_visualisations/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD RESULTS
# ============================================================

csv_path = "outputs/metrics/model_comparison.csv"

df = pd.read_csv(csv_path)

print("Loaded model comparison results:")
print(df)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("outputs/presentation_visualisations", exist_ok=True)


# ============================================================
# STYLE
# ============================================================

BG = "#0f172a"
PANEL = "#1e293b"
FG = "#e2e8f0"

plt.rcParams["figure.facecolor"] = BG
plt.rcParams["axes.facecolor"] = PANEL
plt.rcParams["savefig.facecolor"] = BG
plt.rcParams["text.color"] = FG
plt.rcParams["axes.labelcolor"] = FG
plt.rcParams["xtick.color"] = FG
plt.rcParams["ytick.color"] = FG
plt.rcParams["axes.edgecolor"] = FG


# ============================================================
# CLEAN LABELS
# ============================================================

label_map = {
    "IMAGE_FEATURES": "Image Features",
    "INFOGRAPHIC_FEATURES": "Infographic Features",
    "ALL_FEATURES": "Combined Features",
}

df["display_name"] = df["feature_group"].map(label_map)


# ============================================================
# R² CHART
# ============================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(
    df["display_name"],
    df["r2"],
)

plt.title("Model Comparison — R²")
plt.ylabel("R² Score")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

plt.savefig(
    "outputs/presentation_visualisations/r2_comparison.png",
    dpi=300,
)

plt.close()


# ============================================================
# MAE CHART
# ============================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(
    df["display_name"],
    df["mae"],
)

plt.title("Model Comparison — MAE")
plt.ylabel("Mean Absolute Error")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

plt.savefig(
    "outputs/presentation_visualisations/mae_comparison.png",
    dpi=300,
)

plt.close()


# ============================================================
# RMSE CHART
# ============================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(
    df["display_name"],
    df["rmse"],
)

plt.title("Model Comparison — RMSE")
plt.ylabel("Root Mean Squared Error")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

plt.savefig(
    "outputs/presentation_visualisations/rmse_comparison.png",
    dpi=300,
)

plt.close()


# ============================================================
# COMBINED PERFORMANCE CHART
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(df))

width = 0.25

ax.bar(
    [i - width for i in x],
    df["r2"],
    width=width,
    label="R²",
)

ax.bar(
    x,
    df["mae"],
    width=width,
    label="MAE",
)

ax.bar(
    [i + width for i in x],
    df["rmse"],
    width=width,
    label="RMSE",
)

ax.set_xticks(list(x))
ax.set_xticklabels(df["display_name"])

ax.set_title("Comparative Model Performance")
ax.set_ylabel("Metric Value")

ax.legend()

plt.tight_layout()

plt.savefig(
    "outputs/presentation_visualisations/combined_model_comparison.png",
    dpi=300,
)

plt.close()


# ============================================================
# FINISH
# ============================================================

print("\nVisualisations saved to:")
print("outputs/presentation_visualisations/")
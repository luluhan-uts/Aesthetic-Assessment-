"""
visualize_failure_analysis.py
-----------------------------
Creates visualisations for the Assignment 3 external testing
and failure-case analysis outputs.

Generates:
1. Category average score bar chart
2. Prediction distribution histogram
3. Top/bottom infographic comparison chart
4. Category variability chart

Project 47 — Assignment 3
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PREDICTIONS_PATH = (
    "outputs/external_testing/external_predictions.csv"
)

CATEGORY_PATH = (
    "outputs/external_testing/category_summary.csv"
)

OUTPUT_DIR = "outputs/visualisations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

pred_df = pd.read_csv(PREDICTIONS_PATH)
cat_df = pd.read_csv(CATEGORY_PATH)


# ============================================================
# CHART 1 — CATEGORY MEAN SCORES
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    cat_df["category"],
    cat_df["mean"],
)

plt.ylabel("Average Predicted Score")
plt.xlabel("Infographic Category")
plt.title("Average Predicted Aesthetic Score by Category")
plt.xticks(rotation=20)

plt.tight_layout()

chart1_path = os.path.join(
    OUTPUT_DIR,
    "category_mean_scores.png",
)

plt.savefig(chart1_path, dpi=300)
plt.close()

print(f"Saved: {chart1_path}")


# ============================================================
# CHART 2 — PREDICTION DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    pred_df["predicted_score"],
    bins=8,
)

plt.xlabel("Predicted Aesthetic Score")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Scores")

plt.tight_layout()

chart2_path = os.path.join(
    OUTPUT_DIR,
    "prediction_distribution.png",
)

plt.savefig(chart2_path, dpi=300)
plt.close()

print(f"Saved: {chart2_path}")


# ============================================================
# CHART 3 — TOP VS BOTTOM PREDICTIONS
# ============================================================

sorted_df = pred_df.sort_values(
    by="predicted_score",
    ascending=False,
)

comparison_df = pd.concat([
    sorted_df.head(5),
    sorted_df.tail(5),
])

plt.figure(figsize=(12, 6))

plt.bar(
    comparison_df["filename"],
    comparison_df["predicted_score"],
)

plt.ylabel("Predicted Score")
plt.xlabel("Infographic")
plt.title("Top and Bottom Predicted Infographics")
plt.xticks(rotation=45)

plt.tight_layout()

chart3_path = os.path.join(
    OUTPUT_DIR,
    "top_bottom_predictions.png",
)

plt.savefig(chart3_path, dpi=300)
plt.close()

print(f"Saved: {chart3_path}")


# ============================================================
# CHART 4 — CATEGORY VARIABILITY
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    cat_df["category"],
    cat_df["std"],
)

plt.ylabel("Standard Deviation")
plt.xlabel("Infographic Category")
plt.title("Prediction Variability Across Categories")
plt.xticks(rotation=20)

plt.tight_layout()

chart4_path = os.path.join(
    OUTPUT_DIR,
    "category_variability.png",
)

plt.savefig(chart4_path, dpi=300)
plt.close()

print(f"Saved: {chart4_path}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("VISUALISATION GENERATION COMPLETE")
print("=" * 60)

print("\nGenerated figures:")
print("- category_mean_scores.png")
print("- prediction_distribution.png")
print("- top_bottom_predictions.png")
print("- category_variability.png")

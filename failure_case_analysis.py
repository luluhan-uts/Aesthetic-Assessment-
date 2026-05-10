"""
failure_case_analysis.py
------------------------
Analyses interesting/surprising behaviours from the external
infographic testing stage.

This script:
1. Loads external prediction results
2. Identifies highest and lowest predictions
3. Computes category-level ranking
4. Detects compressed prediction range
5. Generates lightweight discussion outputs
6. Saves CSV summaries for reporting/presentation

Project 47 — Assignment 3
"""

import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PREDICTIONS_PATH = (
    "outputs/external_testing/external_predictions.csv"
)

OUTPUT_DIR = "outputs/failure_case_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MAIN
# ============================================================


def main():

    print("\n" + "=" * 60)
    print("Project 47 - Failure Case Analysis")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD RESULTS
    # --------------------------------------------------------

    df = pd.read_csv(PREDICTIONS_PATH)

    print(f"\nLoaded {len(df)} external predictions.")

    # --------------------------------------------------------
    # SCORE RANGE ANALYSIS
    # --------------------------------------------------------

    score_min = df["predicted_score"].min()
    score_max = df["predicted_score"].max()
    score_mean = df["predicted_score"].mean()
    score_std = df["predicted_score"].std()

    print("\nPrediction Range Summary")
    print("-" * 40)
    print(f"Minimum score : {score_min:.3f}")
    print(f"Maximum score : {score_max:.3f}")
    print(f"Mean score    : {score_mean:.3f}")
    print(f"Std deviation : {score_std:.3f}")

    # --------------------------------------------------------
    # TOP PREDICTIONS
    # --------------------------------------------------------

    top_predictions = df.sort_values(
        by="predicted_score",
        ascending=False,
    ).head(5)

    print("\n" + "=" * 60)
    print("TOP 5 PREDICTIONS")
    print("=" * 60)

    print(
        top_predictions[
            [
                "filename",
                "category",
                "predicted_score",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # LOWEST PREDICTIONS
    # --------------------------------------------------------

    low_predictions = df.sort_values(
        by="predicted_score",
        ascending=True,
    ).head(5)

    print("\n" + "=" * 60)
    print("LOWEST 5 PREDICTIONS")
    print("=" * 60)

    print(
        low_predictions[
            [
                "filename",
                "category",
                "predicted_score",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # CATEGORY ANALYSIS
    # --------------------------------------------------------

    category_summary = (
        df.groupby("category")
        ["predicted_score"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    category_summary = category_summary.sort_values(
        by="mean",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("CATEGORY-LEVEL ANALYSIS")
    print("=" * 60)

    print(category_summary.to_string(index=False))

    # --------------------------------------------------------
    # SURPRISING CASES
    # --------------------------------------------------------

    surprising_cases = df[
        (
            (df["category"] == "cluttered")
            & (df["predicted_score"] > score_mean)
        )
        |
        (
            (df["category"] == "award_winning")
            & (df["predicted_score"] < score_mean)
        )
    ]

    print("\n" + "=" * 60)
    print("POTENTIAL FAILURE / SURPRISING CASES")
    print("=" * 60)

    if len(surprising_cases) > 0:

        print(
            surprising_cases[
                [
                    "filename",
                    "category",
                    "predicted_score",
                    "expected_model_behavior",
                ]
            ].to_string(index=False)
        )

    else:
        print("No surprising cases identified.")

    # --------------------------------------------------------
    # SAVE CSV OUTPUTS
    # --------------------------------------------------------

    top_predictions.to_csv(
        os.path.join(OUTPUT_DIR, "top_predictions.csv"),
        index=False,
    )

    low_predictions.to_csv(
        os.path.join(OUTPUT_DIR, "low_predictions.csv"),
        index=False,
    )

    category_summary.to_csv(
        os.path.join(OUTPUT_DIR, "category_analysis.csv"),
        index=False,
    )

    surprising_cases.to_csv(
        os.path.join(OUTPUT_DIR, "surprising_cases.csv"),
        index=False,
    )

    print("\nSaved outputs to:")
    print(f"  {OUTPUT_DIR}")

    # --------------------------------------------------------
    # INTERPRETATION SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INTERPRETATION SUMMARY")
    print("=" * 60)

    print(
        "The prediction scores occupy a relatively narrow range, "
        "suggesting that the handcrafted feature set may have "
        "limited discriminative power for infographic aesthetics."
    )

    print()

    print(
        "Minimalist infographics generally received higher scores, "
        "while some cluttered infographics also scored surprisingly "
        "well, indicating that the model may reward structured "
        "density rather than penalising clutter consistently."
    )

    print()

    print(
        "Award-winning and experimental infographics were not "
        "always strongly rewarded, suggesting that the current "
        "feature set struggles to capture deeper semantic or "
        "communicative qualities."
    )

    print("\nFailure-case analysis completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

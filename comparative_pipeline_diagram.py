"""
comparative_pipeline_diagram.py
--------------------------------
Generates a pipeline diagram for the current
Assignment 3 experimental framework.

Project 47 — Assignment 3
"""

from graphviz import Digraph


# ============================================================
# CREATE DIAGRAM
# ============================================================

dot = Digraph(
    "Assignment3Pipeline",
    format="png"
)

dot.attr(
    rankdir="LR",
    splines="ortho",
    nodesep="0.6",
    ranksep="1.0",
    bgcolor="#0d1021"
)

dot.attr(
    "node",
    shape="box",
    style="rounded,filled",
    fontname="Helvetica",
    fontsize="14",
    fontcolor="white",
    color="white",
    margin="0.3,0.2"
)

dot.attr(
    "edge",
    color="white",
    arrowsize="0.8"
)


# ============================================================
# TRAINING PIPELINE
# ============================================================

with dot.subgraph(name="cluster_training") as c:

    c.attr(
        label="ASSIGNMENT 3 — TRAINING & EXPERIMENTS",
        color="#3a7bd5",
        fontcolor="#3a7bd5",
        fontsize="20",
        style="dashed"
    )

    c.node(
        "data",
        "data.csv\n\nTufts infographic ratings + features",
        fillcolor="#143d6b"
    )

    c.node(
        "config",
        "config.py\n\nIMAGE_FEATURES\nINFOGRAPHIC_FEATURES\nALL_FEATURES",
        fillcolor="#143d6b"
    )

    c.node(
        "train",
        "train_experiments.py\n\nControlled Ridge experiments\nComparative feature analysis",
        fillcolor="#145a86"
    )

    c.node(
        "models",
        "models/\n\nridge_image.pkl\nridge_infographic.pkl\nridge_combined.pkl",
        fillcolor="#5a189a"
    )

    c.node(
        "metrics",
        "outputs/metrics/\n\nmodel_comparison.csv",
        fillcolor="#0f766e"
    )

    c.node(
        "feature_analysis",
        "feature_analysis.py\n\nCoefficient extraction\nExplainability analysis",
        fillcolor="#145a86"
    )

    c.node(
        "feature_outputs",
        "outputs/feature_analysis/\n\nfeature_coefficients.csv",
        fillcolor="#0f766e"
    )

    c.edge("data", "train")
    c.edge("config", "train")
    c.edge("train", "models")
    c.edge("train", "metrics")
    c.edge("models", "feature_analysis")
    c.edge("feature_analysis", "feature_outputs")


# ============================================================
# EXTERNAL TESTING
# ============================================================

with dot.subgraph(name="cluster_external") as c:

    c.attr(
        label="EXTERNAL INFOGRAPHIC TESTING",
        color="#f59e0b",
        fontcolor="#f59e0b",
        fontsize="20",
        style="dashed"
    )

    c.node(
        "external_data",
        "external_infographics/\n\nMinimalist\nCluttered\nExperimental\nMisleading\nAward-winning\nText-heavy",
        fillcolor="#5b3200"
    )

    c.node(
        "metadata",
        "external_infographics_metadata.csv\n\nCategory labels\nVisual characteristics",
        fillcolor="#5b3200"
    )

    c.node(
        "external_test",
        "external_test.py\n\nFeature extraction\nPrediction generation",
        fillcolor="#7c3aed"
    )

    c.node(
        "external_outputs",
        "outputs/external_testing/\n\nexternal_predictions.csv\ncategory_summary.csv",
        fillcolor="#0f766e"
    )

    c.edge("external_data", "external_test")
    c.edge("metadata", "external_test")
    c.edge("models", "external_test")
    c.edge("external_test", "external_outputs")


# ============================================================
# FAILURE CASE ANALYSIS
# ============================================================

with dot.subgraph(name="cluster_failure") as c:

    c.attr(
        label="FAILURE CASE & BEHAVIOURAL ANALYSIS",
        color="#ec4899",
        fontcolor="#ec4899",
        fontsize="20",
        style="dashed"
    )

    c.node(
        "failure",
        "failure_case_analysis.py\n\nBehaviour analysis\nSurprising predictions\nCategory interpretation",
        fillcolor="#831843"
    )

    c.node(
        "failure_outputs",
        "outputs/failure_case_analysis/\n\nTop/bottom predictions\nFailure summaries",
        fillcolor="#0f766e"
    )

    c.edge("external_outputs", "failure")
    c.edge("failure", "failure_outputs")


# ============================================================
# VISUALISATION PIPELINE
# ============================================================

with dot.subgraph(name="cluster_visuals") as c:

    c.attr(
        label="PRESENTATION VISUALISATIONS",
        color="#22c55e",
        fontcolor="#22c55e",
        fontsize="20",
        style="dashed"
    )

    c.node(
        "visuals",
        "advanced_visualisations.py\n\nCharts\nComparisons\nBehaviour plots",
        fillcolor="#14532d"
    )

    c.node(
        "visual_outputs",
        "outputs/presentation_visualisations/\n\nR² comparison\nFeature importance\nCategory scores\nPrediction distributions",
        fillcolor="#166534"
    )

    c.edge("metrics", "visuals")
    c.edge("feature_outputs", "visuals")
    c.edge("external_outputs", "visuals")
    c.edge("failure_outputs", "visuals")
    c.edge("visuals", "visual_outputs")


# ============================================================
# FINAL OUTPUT
# ============================================================

dot.node(
    "final",
    "FINAL REPORT & PRESENTATION\n\nComparative experiments\nExplainability\nExternal testing\nFailure-case discussion",
    fillcolor="#15803d",
    fontsize="16"
)

dot.edge("visual_outputs", "final")


# ============================================================
# SAVE
# ============================================================

dot.render(
    "assignment3_pipeline",
    cleanup=True
)

print("=" * 60)
print("ASSIGNMENT 3 PIPELINE GENERATED")
print("=" * 60)

print("\nSaved files:")
print("  assignment3_pipeline.png")
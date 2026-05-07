"""
config.py
---------
Shared constants and default paths used across the pipeline.
All paths are resolved relative to this file so scripts work regardless
of the working directory they are launched from.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Default data paths ────────────────────────────────────────────────────────
DATA_CSV              = os.path.join(_HERE, "data.csv")
IMAGES_FOLDER         = os.path.join(_HERE, "training_images")
MODEL_PKL             = os.path.join(_HERE, "model.pkl")
MODEL_FROM_IMAGES_PKL = os.path.join(_HERE, "model_from_images.pkl")

# ── Feature / target definitions ─────────────────────────────────────────────
FEATURE_COLS = [
    "ImageArea",
    "TextGroup",
    "NonTextArea",
    "TextArea",
    "QuadTree",
    "Saturation",
    "Colorfulness2",
    "Colorfulness1",
    "Complexity",
]

# ── Feature groups for Assignment 3 comparative experiments ──────────────────

# Traditional image-aesthetic-related features
IMAGE_FEATURES = [
    "QuadTree",
    "Saturation",
    "Colorfulness2",
    "Colorfulness1",
]

# Infographic-specific structural/layout features
INFOGRAPHIC_FEATURES = [
    "ImageArea",
    "TextGroup",
    "NonTextArea",
    "TextArea",
    "Complexity",
]

# Combined feature set
ALL_FEATURES = IMAGE_FEATURES + INFOGRAPHIC_FEATURES

# Safety check
assert set(ALL_FEATURES) == set(FEATURE_COLS)

TARGET_COL = "Final Score"

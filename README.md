# Infographic Aesthetics Predictor
## Based on Harrison, Reinecke & Chang (CHI 2015)

---

## What this pipeline does

Extracts 9 visual features from infographic images and trains a regression model
to predict aesthetic appeal (1–9 Likert scale). The pipeline replicates the
methodology from the CHI 2015 paper using Python and scikit-learn.

---

## Project structure

```
Python Pipeline/
├── config.py               # Shared constants and default file paths
├── feature_extractor.py    # CV feature extraction (all 9 features)
├── train_model.py          # Train and compare models on data.csv
├── predict.py              # Predict appeal for a single image
├── batch_analyse.py        # Batch extract + compare + retrain on a folder
├── visual_report.py        # Generate an annotated visual report for one image
├── batch_compare.py        # Side-by-side paper vs. extracted reports for a folder
├── pipeline_diagram.py     # Generate a flowchart of the full pipeline
├── data.csv                # Ground-truth dataset (326 infographics, paper features + ratings)
├── training_images/        # Infographic image files (named by Stimulus ID)
├── model.pkl               # Model trained on paper's pre-computed features
└── model_from_images.pkl   # Model trained on freshly extracted features
```

---

## Quick start

All scripts auto-detect `data.csv`, `training_images/`, `model.pkl`, and
`model_from_images.pkl` from the project folder — no file picking needed.

### Predict appeal for a new infographic

```bash
python visual_report.py
```

A file picker asks for one image. The report is saved as
`<image_name>_report.png` next to the image. Uses `model_from_images.pkl`
automatically (calibrated to the extractor's output scale).

### Run the full batch analysis

```bash
python batch_analyse.py
```

Processes all images in `training_images/`, compares extracted vs. paper
features, evaluates a train/test split, and saves `results.csv`.

### Generate side-by-side comparison reports

```bash
python batch_compare.py
```

For every matched image, produces a PNG showing the paper's features (left)
alongside freshly extracted features (right), including annotated image,
feature chart, and score gauge. Saves to `comparisons/`.

### Predict a single image from the command line

```bash
python predict.py --image path/to/image.png
```

### Regenerate the pipeline diagram

```bash
python pipeline_diagram.py
```

Saves `pipeline.png` showing the full data flow, CV techniques, and ML models.

---

## All scripts accept CLI overrides

Defaults can be overridden with flags on any script:

```bash
python batch_analyse.py --images ./other_folder --csv other.csv --output out.csv
python batch_compare.py --images ./other_folder --output ./my_comparisons
python visual_report.py --image x.png --model model.pkl   # use paper model instead
python predict.py --image x.png --model model_from_images.pkl
```

---

## Features extracted per image

| Feature | Description | Method |
|---|---|---|
| `ImageArea` | Number of distinct graphic/image regions | XY-Cut segmentation |
| `TextGroup` | Number of horizontal text clusters | XY-Cut segmentation |
| `NonTextArea` | Total pixel area of non-text content | Connected components |
| `TextArea` | Total pixel area of text content | Connected components |
| `QuadTree` | Information density (recursive quadrant splits) | Quadtree decomposition |
| `Saturation` | Mean HSV saturation across all pixels | Colour model |
| `Colorfulness1` | Yendrikhovskij et al. (1998) — CIELab chroma | Colour model |
| `Colorfulness2` | Hasler & Suesstrunk (2003) — RGB opponent channels | Colour model |
| `Complexity` | Composite score from space-based leaf count | XY-Cut decomposition |

---

## Models

`train_model.py` compares three models via 5-fold cross-validation and saves
the best one to `model.pkl`:

| Model | Notes |
|---|---|
| Ridge Regression (α=1) | L2 regularised linear model |
| Ridge Regression (α=10) | Stronger regularisation — typically wins on this dataset |
| Gradient Boosting | 200 trees, max depth 3, learning rate 0.05 |

### Two saved models

| File | Trained on | Use for |
|---|---|---|
| `model.pkl` | Paper's pre-computed features from `data.csv` | Replicating paper results; comparing against ground truth |
| `model_from_images.pkl` | Features freshly extracted by `feature_extractor.py` | Predicting appeal for new images (calibrated to extractor scale) |

Use `model_from_images.pkl` when scoring new infographics — it is calibrated
to the same feature scale that `feature_extractor.py` produces.
Use `model.pkl` when comparing against the paper's reported numbers.

---

## Understanding the R² score

The pipeline achieves **CV R² ≈ 0.02** on features alone.
The paper reports **R² = 0.34** — the gap has three causes:

1. **Demographic interactions** — the paper uses Gender, Age, and
   EducationLevel as fixed effects interacting with colorfulness and
   complexity. `data.csv` contains per-infographic means, not individual
   ratings with demographics, so these terms cannot be reproduced.

2. **Random effects** — ParticipantID and InfographicID are modelled as
   random effects using `lme4` in R. Standard sklearn regression cannot
   partition variance this way.

3. **Individual-level data** — the paper analyses ~83,000 individual ratings.
   Averaging to 326 per-infographic means removes most of the variance the
   model was designed to explain.

The features are still directionally correct. Key findings from model coefficients:

| Feature | Effect on appeal |
|---|---|
| NonTextArea ↑ | Higher appeal — more visual space, less text-heavy |
| Complexity ↑ | Lower appeal — matches paper finding |
| TextArea ↑ | Lower appeal — text-heavy designs score lower |
| Colorfulness1 ↑ | Higher appeal — more chromatic variation |

---

## Extending the pipeline

### Adding CLIP embeddings as additional features

```python
import clip, torch

model_clip, preprocess = clip.load("ViT-L/14")

probes = ["colorful infographic", "complex cluttered infographic",
          "simple clean infographic", "visually appealing design"]

def clip_features(image_path):
    img = preprocess(Image.open(image_path)).unsqueeze(0)
    text = clip.tokenize(probes)
    with torch.no_grad():
        img_feat = model_clip.encode_image(img)
        txt_feat = model_clip.encode_text(text)
    return (img_feat @ txt_feat.T).squeeze().numpy()
```

Concatenate CLIP features with the 9 paper features before training.

### Using a mixed-effects model in Python

```python
import statsmodels.formula.api as smf

# Requires individual-level data (one row per rating, not per infographic)
model = smf.mixedlm(
    "appeal ~ colorfulness2 * gender + complexity * age_group + education",
    df,
    groups=df["participant_id"],
    re_formula="~infographic_id"
)
result = model.fit()
print(result.summary())
```

### Adjusting predictions for a known audience

```python
def adjust_for_demographics(base_score, colorfulness2, complexity,
                             audience_age="young", audience_gender="mixed",
                             education="college"):
    adjustment = 0.0
    if audience_age == "older":       # 45+: prefer lower complexity
        adjustment -= complexity * 0.05
    if audience_gender == "female":   # prefer more colourful, less complex
        adjustment += colorfulness2 * 0.01
        adjustment -= complexity * 0.02
    if education == "phd":            # sensitive to excessive colourfulness
        adjustment -= colorfulness2 * 0.015
    return float(np.clip(base_score + adjustment, 1.0, 9.0))
```

---

## Key findings from the paper

| Finding | Design implication |
|---|---|
| Colorfulness > complexity for appeal | Prioritise colour over simplification |
| Higher colorfulness → higher appeal | Use saturated, varied colours |
| Females prefer more colourful, less complex | Adjust palette if targeting female audiences |
| Males largely unaffected by complexity | Text-heavy designs are lower risk for male audiences |
| Older viewers prefer lower complexity | Simplify for 45+ audiences |
| Higher education → dislike excessive colour/complexity | More restrained palette for academic audiences |
| Aim for low-to-medium complexity | Limit distinct text and image regions |
| Aim for medium-to-high colorfulness | Increase saturation and colour contrast |

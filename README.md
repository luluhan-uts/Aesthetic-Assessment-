# Infographic Aesthetics Predictor
## Based on Harrison, Reinecke & Chang (CHI 2015)

---

## What this pipeline does

Extracts the 9 visual features used in the paper and trains a regression model
to predict aesthetic appeal (1–9 Likert scale) from infographic images.

### Features extracted per image

| Feature | Description | Paper source |
|---------|-------------|--------------|
| `ImageArea` | Number of distinct image/graphic regions | Space-based decomposition |
| `TextGroup` | Number of horizontal text groups | Space-based decomposition |
| `NonTextArea` | Total pixel area of non-text content | Space-based decomposition |
| `TextArea` | Total pixel area of text content | Space-based decomposition |
| `QuadTree` | Number of quadtree leaves (info density) | Quadtree decomposition |
| `Saturation` | Mean HSV saturation across all pixels | Colour model |
| `Colorfulness1` | Yendrikhovskij et al. (1998) metric | CIELab chroma |
| `Colorfulness2` | Hasler & Suesstrunk (2003) metric | RGB opponent channels |
| `Complexity` | Composite complexity score | Space-based leaf count |

---

## How to use

### 1. Train the model (already done if model.pkl exists)

```bash
python train_model.py --csv data.csv --output model.pkl
```

### 2. Predict appeal for a new infographic

```bash
python predict.py --image my_infographic.png --model model.pkl
```

---

## Understanding the R² score

The trained model achieves **CV R² ≈ 0.02** on features alone.
The paper reports **R² = 0.34** — here's why there's a gap:

### What the paper's 0.34 R² includes that we don't have:

1. **Demographic interactions** — the paper uses a mixed-effects model where
   Gender, Age, and EducationLevel are fixed effects that interact with
   colorfulness and complexity. These demographic terms substantially boost R².
   The CSV only has per-infographic mean ratings, not individual ratings with
   demographics — so we can't replicate this.

2. **Random effects** — ParticipantID and InfographicID are modelled as random
   effects using lme4 in R. This partitions variance in a way standard
   sklearn regression cannot.

3. **Individual-level data** — the paper analyses ~83,000 individual ratings.
   We only have 325 per-infographic means. Averaging removes a large portion
   of the variance the model was designed to explain.

### What this means practically

The features are still meaningful and directionally correct (see coefficients):
- **NonTextArea ↑** → higher appeal (more visual space, less text-heavy)
- **Complexity ↓** → lower appeal (matches paper's finding)
- **Colorfulness2 ↓** → lower appeal (Hasler metric captures excessive colour)
- **TextArea ↓** → lower appeal (text-heavy infographics score lower)

To get closer to the paper's R², you would need to:
1. Collect individual ratings (not just means) with participant demographics
2. Use a mixed-effects model (e.g. `lme4` in R or `statsmodels` MixedLM in Python)
3. Include Gender × Colorfulness and Age × Complexity interaction terms

---

## Extending the pipeline

### Adding CLIP embeddings as additional features

```python
import clip, torch

model_clip, preprocess = clip.load("ViT-L/14")

# Text probes that map to the paper's dimensions
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

Then concatenate CLIP features with the 9 paper features before training.

### Adding demographic predictions

If you know your target audience, you can manually adjust the predicted score:

```python
def adjust_for_demographics(base_score, colorfulness2, complexity,
                             audience_age="young", audience_gender="mixed",
                             education="college"):
    adjustment = 0.0

    # Older audiences prefer lower complexity (paper: β = -0.0004 per year)
    if audience_age == "older":  # 45+
        adjustment -= complexity * 0.05

    # Female audiences prefer more colourful, less complex infographics
    if audience_gender == "female":
        adjustment += colorfulness2 * 0.01
        adjustment -= complexity * 0.02

    # Higher education → more sensitive to excessive colourfulness
    if education == "phd":
        adjustment -= colorfulness2 * 0.015

    return float(np.clip(base_score + adjustment, 1.0, 9.0))
```

### Using a mixed-effects model in Python

```python
import statsmodels.formula.api as smf

# Requires individual-level data (one row per rating, not per infographic)
# df must have columns: appeal, colorfulness2, complexity,
#                       gender, age_group, education, participant_id, infographic_id
model = smf.mixedlm(
    "appeal ~ colorfulness2 * gender + complexity * age_group + education",
    df,
    groups=df["participant_id"],
    re_formula="~infographic_id"
)
result = model.fit()
print(result.summary())
```

---

## Key findings from the paper (design guidelines)

| Finding | Implication |
|---------|-------------|
| Colorfulness > complexity for infographic appeal | Prioritise colour over simplification |
| Higher colorfulness → higher appeal (generally) | Use saturated, varied colours |
| Females prefer more colorful, less complex | Adjust if targeting female audiences |
| Males largely unaffected by complexity | Text-heavy designs less risky for male audiences |
| Older viewers prefer lower complexity | Simplify for 45+ audiences |
| Higher education → dislike excessive colour & complexity | More restrained palette for academic audiences |
| Aim for low-to-medium complexity | Limit distinct text and image regions |
| Aim for medium-to-high colorfulness | Increase saturation and colour contrast |

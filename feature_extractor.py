"""
feature_extractor.py
--------------------
Extracts the same visual features used in:
  Harrison, Reinecke & Chang (CHI 2015) "Infographic Aesthetics"
  Reinecke et al. (CHI 2013) "Predicting Users' First Impressions of Website Aesthetics"

Features extracted per image:
  - ImageArea      : number of image/graphic regions (space-based decomposition)
  - TextGroup      : number of horizontal text groups
  - NonTextArea    : total pixel area of non-text regions
  - TextArea       : total pixel area of text regions
  - QuadTree       : number of leaves from quadtree decomposition (complexity proxy)
  - Saturation     : mean HSV saturation across all pixels
  - Colorfulness1  : Yendrikhovskij et al. (1998) metric
  - Colorfulness2  : Hasler & Suesstrunk (2003) metric
  - Complexity     : composite complexity score (space-based leaf count normalised)
"""

import numpy as np
import cv2
from PIL import Image
from skimage import color as skcolor
from skimage import filters, measure
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Colour metrics
# ---------------------------------------------------------------------------

def compute_saturation(img_bgr: np.ndarray) -> float:
    """Mean HSV saturation across all pixels (0–255 scale)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return float(hsv[:, :, 1].mean())


def compute_colorfulness1(img_bgr: np.ndarray) -> float:
    """
    Yendrikhovskij et al. (1998): colorfulness = mean_chroma + std_chroma
    where chroma is computed in CIELab as sqrt(a^2 + b^2) / L  (approx saturation in Lab).
    Paper uses: sum of mean saturation and its std dev in CIELab.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
    lab = skcolor.rgb2lab(img_rgb)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    chroma = np.sqrt(a ** 2 + b ** 2)
    return float(chroma.mean() + chroma.std())


def compute_colorfulness2(img_bgr: np.ndarray) -> float:
    """
    Hasler & Suesstrunk (2003): colorfulness = sqrt(std_rg^2 + std_yb^2)
                                              + 0.3 * sqrt(mean_rg^2 + mean_yb^2)
    where rg = R - G,  yb = 0.5*(R+G) - B.
    This correlates at r=.95 with human colorfulness ratings.
    """
    img_rgb = img_bgr[:, :, ::-1].astype(np.float32)
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    colorfulness = (np.sqrt(std_rg ** 2 + std_yb ** 2)
                    + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))
    return float(colorfulness)


# ---------------------------------------------------------------------------
# Quadtree decomposition  (complexity proxy)
# ---------------------------------------------------------------------------

def _quadtree_split(region: np.ndarray, min_size: int = 8,
                    entropy_threshold: float = 0.5) -> int:
    """
    Recursively split region into quadrants until entropy falls below threshold
    or region is too small. Returns number of leaf quadrants.
    """
    h, w = region.shape[:2]
    if h < min_size or w < min_size:
        return 1

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    # entropy via histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))

    if entropy < entropy_threshold:
        return 1

    mh, mw = h // 2, w // 2
    quadrants = [
        region[:mh, :mw],
        region[:mh, mw:],
        region[mh:, :mw],
        region[mh:, mw:],
    ]
    return sum(_quadtree_split(q, min_size, entropy_threshold) for q in quadrants)


def compute_quadtree(img_bgr: np.ndarray) -> int:
    """Number of quadtree leaves — higher = more complex."""
    return _quadtree_split(img_bgr)


# ---------------------------------------------------------------------------
# Space-based decomposition  (X-Y cut variant)
# ---------------------------------------------------------------------------

def _find_dividers(projection: np.ndarray, threshold: float) -> list:
    """Find positions where projection falls below threshold (whitespace gaps)."""
    below = projection < threshold
    dividers = []
    in_gap = False
    start = 0
    for i, val in enumerate(below):
        if val and not in_gap:
            in_gap = True
            start = i
        elif not val and in_gap:
            in_gap = False
            dividers.append((start + i) // 2)
    return dividers


def _xy_cut(region: np.ndarray, depth: int = 0, max_depth: int = 6,
            min_size: int = 20) -> dict:
    """
    Recursive X-Y cut page segmentation.
    Returns a tree of nodes; leaves represent individual content blocks.
    """
    h, w = region.shape[:2]
    if h < min_size or w < min_size or depth >= max_depth:
        return {"type": "leaf", "h": h, "w": w, "area": h * w}

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # invert so content = high values, whitespace = low
    inv = 255 - gray

    # horizontal projection (sum across columns → row profile)
    h_proj = inv.mean(axis=1)
    # vertical projection (sum across rows → col profile)
    v_proj = inv.mean(axis=0)

    h_thresh = h_proj.max() * 0.05
    v_thresh = v_proj.max() * 0.05

    h_divs = _find_dividers(h_proj, h_thresh)
    v_divs = _find_dividers(v_proj, v_thresh)

    # prefer the axis with more dividers; fall back to horizontal
    if not h_divs and not v_divs:
        return {"type": "leaf", "h": h, "w": w, "area": h * w}

    children = []
    if len(h_divs) >= len(v_divs):
        boundaries = [0] + h_divs + [h]
        for i in range(len(boundaries) - 1):
            sub = region[boundaries[i]:boundaries[i + 1], :]
            children.append(_xy_cut(sub, depth + 1, max_depth, min_size))
    else:
        boundaries = [0] + v_divs + [w]
        for i in range(len(boundaries) - 1):
            sub = region[:, boundaries[i]:boundaries[i + 1]]
            children.append(_xy_cut(sub, depth + 1, max_depth, min_size))

    return {"type": "node", "children": children}


def _collect_leaves(tree: dict) -> list:
    if tree["type"] == "leaf":
        return [tree]
    leaves = []
    for child in tree["children"]:
        leaves.extend(_collect_leaves(child))
    return leaves


def _is_text_region(region_bgr: np.ndarray) -> bool:
    """
    Heuristic: a region is 'text' if it has high edge density
    but low colour variance (text is typically dark marks on light bg).
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean()
    color_std = region_bgr.std(axis=(0, 1)).mean()
    # high edges + low colour std → likely text
    return edge_density > 8 and color_std < 60


def compute_space_based(img_bgr: np.ndarray):
    """
    Returns (ImageArea, TextGroup, NonTextArea, TextArea, Complexity)
    matching the column definitions in the Harrison et al. dataset.
    """
    tree = _xy_cut(img_bgr)
    leaves = _collect_leaves(tree)
    total_leaves = len(leaves)

    h_total, w_total = img_bgr.shape[:2]

    text_area = 0
    non_text_area = 0
    text_groups = 0
    image_areas = 0

    # reconstruct leaf bounding boxes approximately for classification
    # (since we lost exact coordinates during recursion, re-do a flat pass)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    _, binary = cv2.threshold(inv, 15, 255, cv2.THRESH_BINARY)

    # connected component analysis to find content blocks
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    min_area = 100  # ignore tiny noise
    text_group_count = 0
    image_area_count = 0
    text_pixel_total = 0
    non_text_pixel_total = 0

    for i in range(1, num_labels):  # skip background
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        patch = img_bgr[y:y + bh, x:x + bw]
        if patch.size == 0:
            continue
        if _is_text_region(patch):
            text_group_count += 1
            text_pixel_total += area
        else:
            image_area_count += 1
            non_text_pixel_total += area

    # complexity: normalised leaf count (paper uses raw count ~0-28 range)
    complexity = total_leaves / (h_total * w_total) * 1e6

    return {
        "ImageArea": image_area_count,
        "TextGroup": text_group_count,
        "NonTextArea": non_text_pixel_total,
        "TextArea": text_pixel_total,
        "Complexity": complexity,
    }


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_features(image_path: str) -> dict:
    """
    Extract all features from an infographic image.

    Parameters
    ----------
    image_path : str
        Path to the image file (PNG, JPG, etc.)

    Returns
    -------
    dict with keys:
        ImageArea, TextGroup, NonTextArea, TextArea, QuadTree,
        Saturation, Colorfulness1, Colorfulness2, Complexity
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Resize very large images to speed up processing (preserving aspect ratio)
    max_dim = 1024
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    features = {}
    features["Saturation"] = compute_saturation(img_bgr)
    features["Colorfulness1"] = compute_colorfulness1(img_bgr)
    features["Colorfulness2"] = compute_colorfulness2(img_bgr)
    features["QuadTree"] = compute_quadtree(img_bgr)

    space = compute_space_based(img_bgr)
    features.update(space)

    return features


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_extractor.py <image_path>")
        sys.exit(1)
    feats = extract_features(sys.argv[1])
    for k, v in feats.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")

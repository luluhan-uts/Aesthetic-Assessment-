"""
feature_extractor.py
--------------------
Faithful Python replication of the vizweb feature extractor used in:
  Harrison, Reinecke & Chang (CHI 2015) "Infographic Aesthetics"

Reverse-engineered from vizweb-master Java source.

Features extracted:
  ImageArea     : # parent blocks containing a non-text leaf child (area >= 800 px)
  TextGroup     : # parent blocks containing a text-classified child
  NonTextArea   : sum pixel area of non-text leaf blocks
  TextArea      : sum pixel area of text leaf blocks
  QuadTree      : # quadtree leaves (2-D H×S color entropy)
  Saturation    : mean HSV S-channel value (0-255)
  Colorfulness1 : Hasler & Suesstrunk (2003)  [Java: computeColorfulness()]
  Colorfulness2 : HSV chroma/hue metric        [Java: computeColorfulness2()]
  Complexity    : total XY-cut leaf count
"""

import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 1.  COLOUR FEATURES
# =============================================================================

def compute_saturation(img_bgr: np.ndarray) -> float:
    """Mean HSV S-channel (0-255). Java: ColorAnalyzer.computeAverageHueSaturationValue()."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].astype(np.float64).mean())


def compute_colorfulness1(img_bgr: np.ndarray) -> float:
    """
    Hasler & Suesstrunk (2003). Java: ColorAnalyzer.computeColorfulness().
    alpha = R-G,  beta = 0.5*(R+G)-B
    metric = sqrt(sigma_a^2 + sigma_b^2) + 0.3 * sqrt(mu_a^2 + mu_b^2)
    """
    f = img_bgr[:, :, ::-1].astype(np.float64)   # BGR -> RGB
    R, G, B = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    alpha = R - G
    beta  = 0.5 * (R + G) - B
    return float(np.sqrt(alpha.std()**2 + beta.std()**2)
                 + 0.3 * np.sqrt(alpha.mean()**2 + beta.mean()**2))


def compute_colorfulness2(img_bgr: np.ndarray) -> float:
    """
    Java: ColorAnalyzer.computeColorfulness2().
    Despite the variable being named 'luv', the code calls CV_BGR2HSV.
    Channels after split: H (0-180), S (0-255), V (0-255).
    chroma = sqrt(S^2 + V^2)
    sat_map = chroma / H   (H==0 pixels -> 0, matching OpenCV cvDiv behaviour)
    Returns mean + stddev of sat_map.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    chroma = np.sqrt(S * S + V * V)
    with np.errstate(divide='ignore', invalid='ignore'):
        sat = np.where(H > 0, chroma / H, 0.0)
    return float(sat.mean() + sat.std())


# =============================================================================
# 2.  QUADTREE  (ColorEntropyDecompositionStrategy + QuadTreeDecomposer)
# =============================================================================

def _color_entropy(hsv_region: np.ndarray) -> float:
    """
    2-D H×S histogram (30×32 bins, H in [0,180], S in [0,256]).
    Normalised to sum=100. Returns sum(p * ln(p)).
    Matches EntropyComputer.computeColorEntropy() — note: NO minus sign.
    Zero bins contribute 0 (limit of p*ln(p) as p->0 is 0).
    """
    H = hsv_region[:, :, 0].ravel().astype(np.float32)
    S = hsv_region[:, :, 1].ravel().astype(np.float32)
    # Java OpenCV range [0,255) for S — exclusive upper bound, matches cvCalcHist
    hist, _, _ = np.histogram2d(H, S, bins=[30, 32],
                                range=[[0, 180], [0, 255]])
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist * (100.0 / total)          # normalise so sum == 100
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask])))


def _qt_leaves(hsv: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """
    Recursive quadtree matching QuadTreeDecomposer.decompose().
    Returns number of leaf slots (null children in Java each count as 1).
    """
    # Too small -> null in Java -> counts as 1 at the parent level
    if w < 10 and h < 10:
        return 1

    region = hsv[y:y + h, x:x + w]
    entropy = _color_entropy(region)

    # Force split when wide (>500) or entropy indicates diversity (<300)
    if w > 500 or entropy < 300:
        hw = w // 2          # Java uses integer division for all four quadrants
        hh = h // 2
        quadrants = [
            (x + hw, y,      hw, hh),   # q1 top-right
            (x,      y,      hw, hh),   # q2 top-left
            (x,      y + hh, hw, hh),   # q3 bottom-left
            (x + hw, y + hh, hw, hh),   # q4 bottom-right
        ]
        return sum(_qt_leaves(hsv, qx, qy, qw, qh)
                   for qx, qy, qw, qh in quadrants)
    else:
        return 1  # leaf (not decomposed)


def compute_quadtree(img_bgr: np.ndarray) -> int:
    """Quadtree leaf count via 2-D H×S color entropy."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img_bgr.shape[:2]
    return _qt_leaves(hsv, 0, 0, w, h)


# =============================================================================
# 3.  BLOCK STRUCTURE  (XY-cut + text detection)
# =============================================================================

class _Block:
    """Minimal Python equivalent of vizweb Block.java."""
    __slots__ = ('x', 'y', 'w', 'h', 'children', 'is_text')

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.children: list = []
        self.is_text: bool = False

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ---------------------------------------------------------------------------
# Separator utilities
# ---------------------------------------------------------------------------

def _segment(mask) -> list:
    """
    BooleanListSegmenter.segment(): find contiguous True runs.
    Returns list of (position, size) tuples.
    """
    segs, in_seg, start = [], False, 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            in_seg, start = True, i
        elif not v and in_seg:
            in_seg = False
            segs.append((start, i - start))
    if in_seg:
        segs.append((start, len(mask) - start))
    return segs


def _find_content_bounds(gray: np.ndarray,
                         rx: int, ry: int, rw: int, rh: int):
    """
    Approximate Space.findContentBounds(): trim rows/cols whose pixel
    stddev < 10 (same threshold as SpaceSeparatorExtractor) from each border.
    Returns (bx, by, bw, bh) in absolute image coordinates.
    """
    region = gray[ry:ry + rh, rx:rx + rw].astype(np.float32)
    row_content = region.std(axis=1) >= 10
    col_content = region.std(axis=0) >= 10

    rows = np.where(row_content)[0]
    cols = np.where(col_content)[0]
    if len(rows) == 0 or len(cols) == 0:
        return rx, ry, rw, rh

    top, bot   = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    return rx + left, ry + top, right - left, bot - top


def _space_separators(gray: np.ndarray,
                      rx: int, ry: int, rw: int, rh: int) -> list:
    """SpaceSeparatorExtractor: per-row/col stddev < 10 -> space separator."""
    region = gray[ry:ry + rh, rx:rx + rw].astype(np.float32)
    seps = []

    h_mask = (region.std(axis=1) < 10).tolist()
    for pos, size in _segment(h_mask):
        seps.append({'x': rx, 'y': ry + pos, 'w': rw, 'h': size,
                     'horiz': True, 'length': rw, 'thick': size})

    v_mask = (region.std(axis=0) < 10).tolist()
    for pos, size in _segment(v_mask):
        seps.append({'x': rx + pos, 'y': ry, 'w': size, 'h': rh,
                     'horiz': False, 'length': rh, 'thick': size})

    return seps


def _line_separators(edge_map: np.ndarray,
                     rx: int, ry: int, rw: int, rh: int) -> list:
    """
    LineSeparatorExtractor: Canny edge density > 0.85 per row/col,
    with before/after verification (responses < 20% of line response).
    """
    region = edge_map[ry:ry + rh, rx:rx + rw]
    THRESH = 0.85
    seps = []

    h_counts = (region > 0).sum(axis=1).astype(np.float32)
    h_mask = (h_counts / rw > THRESH).tolist()
    for pos, size in _segment(h_mask):
        bi = max(pos - 1, 0)
        ai = min(pos + size, len(h_counts) - 1)
        lr = float(h_counts[pos])
        if lr > 0 and h_counts[bi] < 0.2 * lr and h_counts[ai] < 0.2 * lr:
            seps.append({'x': rx, 'y': ry + pos, 'w': rw, 'h': size,
                         'horiz': True, 'length': rw, 'thick': size})

    v_counts = (region > 0).sum(axis=0).astype(np.float32)
    v_mask = (v_counts / rh > THRESH).tolist()
    for pos, size in _segment(v_mask):
        bi = max(pos - 1, 0)
        ai = min(pos + size, len(v_counts) - 1)
        lr = float(v_counts[pos])
        if lr > 0 and v_counts[bi] < 0.2 * lr and v_counts[ai] < 0.2 * lr:
            seps.append({'x': rx + pos, 'y': ry, 'w': size, 'h': rh,
                         'horiz': False, 'length': rh, 'thick': size})

    return seps


def _select_best(seps: list) -> list:
    """
    DefaultXYDecompositionStrategy.selectTheLargestAndSimilarOnes:
    keep the separator with the largest area plus any with similar
    thickness (within 50%) and length (within 5 px).
    """
    if not seps:
        return []
    best = max(seps, key=lambda s: s['w'] * s['h'])
    out = [best]
    bt, bl = best['thick'], best['length']
    for s in seps:
        if s is best:
            continue
        st, sl = s['thick'], s['length']
        if (max(bt, st) > 0
                and min(bt, st) / max(bt, st) > 0.5
                and abs(sl - bl) < 5):
            out.append(s)
    return out


def _split_two(rx, ry, rw, rh, sep):
    """splitRectangleIntoTwoRegions: returns [top_or_left, bottom_or_right]."""
    if sep['horiz']:
        y, s = sep['y'], sep['h']
        return [(rx, ry, rw, y - ry),
                (rx, y + s, rw, rh - (y - ry) - s)]
    else:
        x, s = sep['x'], sep['w']
        return [(rx, ry, x - rx, rh),
                (x + s, ry, rw - (x - rx) - s, rh)]


def _split_many(rx, ry, rw, rh, seps) -> list:
    """
    splitRectangleIntoRegions: sort descending by position, then peel off
    the 'far' region one separator at a time (matches Java logic exactly).
    """
    if not seps:
        return []
    horiz = seps[0]['horiz']
    ordered = sorted(seps, key=lambda s: s['y'] if horiz else s['x'],
                     reverse=True)
    children, remaining = [], (rx, ry, rw, rh)
    for i, sep in enumerate(ordered):
        two = _split_two(*remaining, sep)
        children.append(two[1])           # bottom / right part
        if i == len(ordered) - 1:
            children.append(two[0])       # last: also keep top / left
        else:
            remaining = two[0]
    return children


# ---------------------------------------------------------------------------
# Recursive XY-cut decomposition
# ---------------------------------------------------------------------------

_MIN_AREA  = 50
_MIN_DIM   = 5
_MAX_LEVEL = 10


def _xy_decompose(gray: np.ndarray, edge_map: np.ndarray,
                  rx: int, ry: int, rw: int, rh: int,
                  level: int) -> _Block:
    """
    XYDecomposer.decomposeRecursively() faithfully translated.
    DefaultXYDecompositionStrategy: minArea=50, minDim=5, maxLevel=10,
    isRemovingBorder=True.
    """
    node = _Block(rx, ry, rw, rh)

    # --- stop conditions ---
    if (rw * rh < _MIN_AREA or rw < _MIN_DIM or rh < _MIN_DIM
            or level > _MAX_LEVEL):
        return node

    # --- border removal (isRemovingBorder=True) ---
    bx, by, bw, bh = _find_content_bounds(gray, rx, ry, rw, rh)
    if bw < rw or bh < rh:
        if bw >= _MIN_DIM and bh >= _MIN_DIM:
            if level == 0:
                inner = _xy_decompose(gray, edge_map, bx, by, bw, bh, 1)
                node.children.append(inner)
                return node
            else:
                # Java discards the node and returns the trimmed block directly
                return _xy_decompose(gray, edge_map, bx, by, bw, bh, level)

    # --- separator detection ---
    # For regions > 100x100: try line separators first, then space separators.
    # For smaller regions: space separators only.
    attempts = []
    if rw > 100 and rh > 100:
        lseps = _line_separators(edge_map, rx, ry, rw, rh)
        # LineSeparatorExtractor filter: length > 100 and roi > 100x100
        lseps = [s for s in lseps if s['length'] > 100]
        attempts.append(('line', lseps))
    attempts.append(('space', None))   # space seps computed lazily

    for kind, presep in attempts:
        if kind == 'line':
            selected = presep
        else:
            sseps = _space_separators(gray, rx, ry, rw, rh)
            selected = _select_best(sseps)

        if not selected:
            continue

        child_rois = _split_many(rx, ry, rw, rh, selected)
        for crx, cry, crw, crh in child_rois:
            if crw <= 0 or crh <= 0:
                continue
            if crw * crh < _MIN_AREA or crw < _MIN_DIM or crh < _MIN_DIM:
                continue
            child = _xy_decompose(gray, edge_map, crx, cry, crw, crh,
                                  level + 1)
            node.children.append(child)
        return node     # first type of separator that yields results wins

    return node         # no separator found -> leaf


# ---------------------------------------------------------------------------
# Post-processing: filter tiny blocks
# ---------------------------------------------------------------------------

def _filter_small(node: _Block):
    """Block.filterOutSmallBlocks(): drop children with area<10 or dim<2."""
    node.children = [c for c in node.children
                     if c.area >= 10 and c.w >= 2 and c.h >= 2]
    for c in node.children:
        _filter_small(c)


# ---------------------------------------------------------------------------
# Text detection  (BlockTextDetector + XYTextDetector)
# ---------------------------------------------------------------------------

def _statistics(node: _Block) -> dict:
    """BlockAnalysis.computeStatistics(): children areas normalised by parent."""
    ch = node.children
    if not ch:
        return {'mean': 0.0, 'stdev': 0.0, 'min': 0, 'max': 0}
    pa = node.area
    norms = [c.area / pa for c in ch]
    mean = sum(norms) / len(norms)
    stdev = (sum((v - mean) ** 2 for v in norms) / len(norms)) ** 0.5
    return {'mean': mean, 'stdev': stdev,
            'min': int(min(norms)), 'max': int(max(norms))}


def _avg_spacing(blocks: list) -> float:
    """BlockAnalysis.getAverageSpacing(): mean horizontal gap sorted by maxX."""
    if len(blocks) < 2:
        return 0.0
    sb = sorted(blocks, key=lambda b: b.x + b.w)
    gaps = [sb[i + 1].x - (sb[i].x + sb[i].w) for i in range(len(sb) - 1)]
    return max(0.0, sum(gaps) / len(gaps))


def _is_reasonable_text(node: _Block) -> bool:
    """isReasonableToBeText(): height, child size, and vertical-overlap checks."""
    if node.h > 200:
        return False
    for c in node.children:
        if c.area > 50000:
            return False
    # Vertical overlap: children sorted by maxX must share Y-band (overlap >= 3 px)
    ch = sorted(node.children, key=lambda b: b.x + b.w)
    for i in range(len(ch) - 1):
        a, b = ch[i], ch[i + 1]
        if a.y < b.y:
            if (a.y + a.h) - b.y < 3:
                return False
        else:
            if (b.y + b.h) - a.y < 3:
                return False
    return True


def _detect_text(node: _Block, img_bgr: np.ndarray):
    """
    BlockTextDetector.detect() applied recursively (XYTextDetector.detect()).
    Processes node first (pre-order), then recurses to children.
    """
    if node.is_leaf:
        node.is_text = False
        return

    stat  = _statistics(node)
    ratio = node.h / node.w if node.w > 0 else float('inf')
    ch    = node.children

    detected = False
    if ratio < 0.5 and _is_reasonable_text(node) and stat['stdev'] < 0.5:
        n = len(ch)
        if n >= 5:
            detected = True
        elif n >= 3:
            detected = (_avg_spacing(ch) <= 25 and stat['max'] < 5000)
        elif n >= 2:
            if all(_is_reasonable_text(c) for c in ch):
                grandkids = [gc for c in ch for gc in c.children]
                if len(grandkids) > 1:
                    detected = _avg_spacing(grandkids) <= 15

    # Re-evaluate large blocks using a quadtree complexity check
    if detected and node.area > 8000 and node.h > 50:
        roi_img = img_bgr[node.y:node.y + node.h, node.x:node.x + node.w]
        if roi_img.size > 0:
            qt_count = compute_quadtree(roi_img)
            if qt_count > 10:
                detected = False

    node.is_text = detected
    for c in ch:
        _detect_text(c, img_bgr)


def _remove_text_children(node: _Block):
    """Block.removeChildrenOfTextBlocks(): text blocks become leaves."""
    if node.is_leaf:
        return
    if node.is_text:
        node.children.clear()
    else:
        for c in node.children:
            _remove_text_children(c)


# ---------------------------------------------------------------------------
# Feature counting (XYFeatureComputer)
# ---------------------------------------------------------------------------

def _collect_leaf_depths(node: _Block, depth: int) -> list:
    """Return list of depths for all leaf nodes in the subtree."""
    if node.is_leaf:
        return [depth]
    return [d for c in node.children
            for d in _collect_leaf_depths(c, depth + 1)]


def _avg_decomposition_level(root: _Block) -> float:
    """
    XYFeatureComputer.computeAverageDecompositionLevel():
    average depth of all leaf nodes across the block tree.
    This is the paper's 'Complexity' feature (float, typically 5-10).
    """
    depths = _collect_leaf_depths(root, 0)
    if not depths:
        return 0.0
    return sum(depths) / len(depths)


def _count_leaves(node: _Block) -> int:
    if node.is_leaf:
        return 1
    return sum(_count_leaves(c) for c in node.children)


def _count_text_group(node: _Block) -> int:
    """countNumberOfTextGroup: non-leaf nodes having >= 1 text child."""
    if node.is_leaf:
        return 0
    count = 1 if any(c.is_text for c in node.children) else 0
    return count + sum(_count_text_group(c) for c in node.children)


def _count_image_area(node: _Block) -> int:
    """countNumberOfImageArea: non-leaf nodes with a non-text leaf child (area>=800)."""
    if node.is_leaf:
        return 0
    has_img = any(c.is_leaf and not c.is_text and c.area >= 800
                  for c in node.children)
    count = 1 if has_img else 0
    return count + sum(_count_image_area(c) for c in node.children)


def _text_area(node: _Block) -> int:
    """computeTextArea: sum areas of text leaf blocks."""
    if node.is_leaf:
        return node.area if node.is_text else 0
    return sum(_text_area(c) for c in node.children)


def _non_text_area(node: _Block) -> int:
    """computeNonTextLeavesArea: sum areas of non-text leaf blocks."""
    if node.is_leaf:
        return node.area if not node.is_text else 0
    return sum(_non_text_area(c) for c in node.children)


def compute_space_based(img_bgr: np.ndarray) -> dict:
    """
    Run XY-cut decomposition + text detection and return the 5 block features.
    Matches XYFeatureComputer.getXYBlockStructure() + individual counters.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # LineSeparatorExtractor edge map: Canny -> dilate -> erode (3x3 rect)
    edge = cv2.Canny(gray, 33, 67, apertureSize=3)
    k3   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge = cv2.dilate(edge, k3, iterations=1)
    edge = cv2.erode(edge,  k3, iterations=1)

    h, w = img_bgr.shape[:2]
    root = _xy_decompose(gray, edge, 0, 0, w, h, 0)

    _filter_small(root)
    _detect_text(root, img_bgr)
    _remove_text_children(root)

    return {
        'ImageArea':   _count_image_area(root),
        'TextGroup':   _count_text_group(root),
        'NonTextArea': _non_text_area(root),
        'TextArea':    _text_area(root),
        'Complexity':  _avg_decomposition_level(root),
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def extract_features(image_path: str) -> dict:
    """
    Extract all 9 features from an infographic image.

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
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    features = {
        'Saturation':    compute_saturation(img_bgr),
        'Colorfulness1': compute_colorfulness1(img_bgr),
        'Colorfulness2': compute_colorfulness2(img_bgr),
        'QuadTree':      compute_quadtree(img_bgr),
    }
    features.update(compute_space_based(img_bgr))
    return features


def normalize_features(features: dict) -> dict:
    """
    Normalize scale-sensitive features so values are comparable regardless
    of whether the source image is a viewport crop or a full-page screenshot.

    Applied transforms:
      NonTextArea, TextArea → proportion of total leaf-block area  [0, 1]
      QuadTree              → log(QuadTree + 1)

    All other features (Saturation, Colorfulness1/2, TextGroup, ImageArea,
    Complexity) are returned unchanged — they are already scale-robust.
    """
    import math
    out = dict(features)
    total = out.get('NonTextArea', 0) + out.get('TextArea', 0)
    if total > 0:
        out['NonTextArea'] = out['NonTextArea'] / total
        out['TextArea']    = out['TextArea']    / total
    else:
        out['NonTextArea'] = 0.0
        out['TextArea']    = 0.0
    out['QuadTree'] = math.log(out.get('QuadTree', 0) + 1)
    return out


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python feature_extractor.py <image_path>')
        sys.exit(1)
    feats = extract_features(sys.argv[1])
    for k, v in feats.items():
        print(f'  {k:20s}: {v:.4f}' if isinstance(v, float) else
              f'  {k:20s}: {v}')

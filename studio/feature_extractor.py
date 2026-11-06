"""
LianLianKan Feature Extractor
Extracts HSV color histogram, dominant tile body color, and central icon shape features.
Distinguishes empty background cells and obstacle stone blocks.
"""

import cv2
import numpy as np

def _cell_metrics(cell_img):
    """Colour-independent descriptors of one cell (all relative to cell size)."""
    h, w = cell_img.shape[:2]
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gm = (np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)) +
          np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)))
    y0, y1 = int(h * 0.25), max(int(h * 0.75), int(h * 0.25) + 2)
    x0, x1 = int(w * 0.25), max(int(w * 0.75), int(w * 0.25) + 2)
    core = cell_img[y0:y1, x0:x1]
    core_mean = np.mean(core.reshape(-1, 3), axis=0)
    core_std = float(np.mean(np.std(core.reshape(-1, 3), axis=0)))
    ring_mask = np.ones((h, w), bool)
    ring_mask[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)] = False
    return core_mean, core_std, float(gm[ring_mask].mean())

def extract_features(cell_img):
    """
    Extract comprehensive visual features from a sliced cell image.
    Returns a dictionary of features and a *provisional* type classification;
    call classify_cells() on the whole board for the final empty/obstacle split.
    """
    h, w, _ = cell_img.shape
    cy, cx = h // 2, w // 2
    core_mean_bgr, core_std_bgr, ring_grad = _cell_metrics(cell_img)

    # Flat centre (no icon) => empty ground or obstacle; decided board-wide later.
    if core_std_bgr < 10.0:
        return {
            "type": "flat",
            "core_mean_bgr": core_mean_bgr.tolist(),
            "core_std_bgr": float(core_std_bgr),
            "ring_grad": ring_grad
        }

    # 3. Matchable tile feature extraction
    # Tile body color (outer frame ring, avoiding central icon)
    margin = max(2, int(round(min(h, w) * 0.09)))
    bw = max(2, margin)
    border_pixels = np.concatenate([
        cell_img[margin:margin+bw, margin:w-margin].reshape(-1, 3),
        cell_img[h-margin-bw:h-margin, margin:w-margin].reshape(-1, 3),
        cell_img[margin:h-margin, margin:margin+bw].reshape(-1, 3),
        cell_img[margin:h-margin, w-margin-bw:w-margin].reshape(-1, 3)
    ])
    body_bgr = np.median(border_pixels, axis=0)
    body_hsv = cv2.cvtColor(np.uint8([[body_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    
    # Determine Color Family based on HSV
    h_val, s_val, v_val = int(body_hsv[0]), int(body_hsv[1]), int(body_hsv[2])
    if s_val < 30:
        color_family = "GREY"
    elif 10 <= h_val <= 35:
        color_family = "YELLOW"
    elif 80 <= h_val <= 115:
        color_family = "CYAN_BLUE"
    elif 116 <= h_val <= 145:
        color_family = "VIOLET"
    else:
        color_family = "MAGENTA"
        
    # HSV Histogram (8 H bins, 4 S bins, 4 V bins)
    hsv_cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_cell], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_feat = hist.flatten().tolist()
    
    # Icon shape / perceptual features (center 28x28)
    ih, iw = max(4, int(h * 0.31)), max(4, int(w * 0.31))
    icon_roi = cell_img[max(0, cy-ih):cy+ih, max(0, cx-iw):cx+iw]
    icon_gray = cv2.cvtColor(icon_roi, cv2.COLOR_BGR2GRAY)
    
    # Icon contour / mask (color diff from body)
    diff = np.linalg.norm(icon_roi.astype(float) - body_bgr.astype(float), axis=2)
    icon_mask = np.uint8((diff > 25) * 255)
    
    # Hu moments
    moments = cv2.moments(icon_mask)
    hu = cv2.HuMoments(moments).flatten()
    log_hu = [-float(np.sign(val) * np.log10(abs(val) + 1e-10)) for val in hu]
    
    return {
        "type": "matchable",
        "color_family": color_family,
        "body_bgr": [float(x) for x in body_bgr],
        "body_hsv": [int(x) for x in body_hsv],
        "core_mean_bgr": [float(x) for x in core_mean_bgr],
        "core_std_bgr": float(core_std_bgr),
        "icon_area": int(moments['m00']),
        "hu_moments": log_hu,
        "hist": hist_feat
    }


def classify_cells(cells):
    """
    Board-wide, colour-independent split of flat cells into EMPTY ground and
    OBSTACLE blocks.  Works on any skin (tan, blue swirl, black ...):
      * flat cell   : centre texture is far below the icon tiles (extract_features)
      * obstacle    : flat centre but a tile-like outline (ring gradient close to
                      real tiles), or - when a clear majority of flat cells share one
                      colour - a flat cell whose colour clearly differs from that
                      majority and which still shows a moderate outline.
      * empty       : every other flat cell.
    Mutates each cell's 'tile_type' / 'features["type"]' and returns the cells.
    """
    tiles = [c for c in cells if c["features"].get("type") == "matchable"]
    flats = [c for c in cells if c["features"].get("type") == "flat"]
    if not flats:
        return cells
    if tiles:
        ref_ring = float(np.median([_cell_metrics(c["image"])[2] for c in tiles]))
    else:
        ref_ring = max(f["features"]["ring_grad"] for f in flats) or 1.0
    ref_ring = max(ref_ring, 1e-6)

    # majority colour of flat cells (Lab distance, robust median)
    labs = np.array([cv2.cvtColor(np.uint8([[f["features"]["core_mean_bgr"]]]),
                                  cv2.COLOR_BGR2LAB)[0][0] for f in flats], float)
    majority = np.median(labs, axis=0)

    for f in flats:
        ratio = f["features"]["ring_grad"] / ref_ring
        lab = cv2.cvtColor(np.uint8([[f["features"]["core_mean_bgr"]]]),
                           cv2.COLOR_BGR2LAB)[0][0].astype(float)
        dcol = float(np.linalg.norm(lab - majority))
        is_obstacle = ratio >= 0.45 or (ratio >= 0.30 and dcol > 25)
        f["features"]["type"] = "obstacle" if is_obstacle else "empty"
        f["features"]["id"] = -1 if is_obstacle else 0
        f["tile_type"] = f["features"]["type"]
    return cells

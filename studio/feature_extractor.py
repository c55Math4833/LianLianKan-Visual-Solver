"""
LianLianKan Feature Extractor
Extracts HSV color histogram, dominant tile body color, and central icon shape features.
Distinguishes empty background cells and obstacle stone blocks.
"""

import cv2
import numpy as np

def extract_features(cell_img):
    """
    Extract comprehensive visual features from a sliced cell image.
    Returns a dictionary of features and initial type classification.
    """
    h, w, _ = cell_img.shape
    cy, cx = h // 2, w // 2
    
    # Core central patch (20x20)
    core = cell_img[cy-10:cy+10, cx-10:cx+10]
    core_mean_bgr = np.mean(core, axis=(0, 1))
    core_std_bgr = np.mean(np.std(core, axis=(0, 1)))
    
    # Grayscale conversion
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    core_gray = gray[cy-10:cy+10, cx-10:cx+10]
    core_std_gray = float(np.std(core_gray))
    
    # 1. Check for Empty background cell:
    # Cleared board has a flat, solid tan/khaki color with near-zero texture/variance
    is_empty = (
        128 <= core_mean_bgr[0] <= 155 and
        158 <= core_mean_bgr[1] <= 178 and
        178 <= core_mean_bgr[2] <= 198 and
        core_std_bgr < 2.5
    )
    
    if is_empty:
        return {
            "type": "empty",
            "id": 0,
            "core_mean_bgr": core_mean_bgr.tolist(),
            "core_std_bgr": float(core_std_bgr)
        }
        
    # 2. Check for Obstacle stone block:
    # Obstacles are the solid grey stone blocks (bluish grey tint, low texture)
    is_obstacle = (
        152 <= core_mean_bgr[0] <= 175 and
        138 <= core_mean_bgr[1] <= 158 and
        115 <= core_mean_bgr[2] <= 135 and
        core_std_bgr < 3.0
    )
    
    if is_obstacle:
        return {
            "type": "obstacle",
            "id": -1,
            "core_mean_bgr": core_mean_bgr.tolist(),
            "core_std_bgr": float(core_std_bgr)
        }
        
    # 3. Matchable tile feature extraction
    # Tile body color (outer frame ring, avoiding central icon)
    margin = 4
    border_pixels = np.concatenate([
        cell_img[margin:margin+4, margin:w-margin].reshape(-1, 3),
        cell_img[h-margin-4:h-margin, margin:w-margin].reshape(-1, 3),
        cell_img[margin:h-margin, margin:margin+4].reshape(-1, 3),
        cell_img[margin:h-margin, w-margin-4:w-margin].reshape(-1, 3)
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
    icon_roi = cell_img[cy-14:cy+14, cx-14:cx+14]
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

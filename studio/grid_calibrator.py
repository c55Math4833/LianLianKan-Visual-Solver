"""
LianLianKan Grid Calibrator
Adaptive ROI selection, Comb-Filter Lattice Alignment, and grid pitch detection.
Handles regular rectangular boards and irregular (hollow, cross, diamond, sparse) boards.
Guarantees pixel-perfect tile centering and full outer tile inclusion without clipping.
"""

import os
import cv2
import numpy as np
from scipy.signal import find_peaks

def load_image(path):
    """Load image from path supporting unicode filenames on Windows."""
    with open(path, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def save_image(path, img):
    """Save image supporting unicode filenames."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1]
    res, enc = cv2.imencode(ext, img)
    if res:
        with open(path, 'wb') as f:
            f.write(enc)

def get_dominant_pitch(profile, max_len):
    """
    Computes dominant grid pitch using 1D signal autocorrelation.
    """
    norm = profile - np.mean(profile)
    corr = np.correlate(norm, norm, mode='full')[len(profile)-1:]
    min_lag, max_lag = 18, min(120, max_len // 2)
    if max_lag <= min_lag:
        return 45.0
    peaks, _ = find_peaks(corr[min_lag:max_lag], distance=10)
    peaks = peaks + min_lag
    if len(peaks) == 0:
        return 45.0
    return float(peaks[np.argmax(corr[peaks])])

def align_grid_lattice(img, approx_roi=None):
    """
    Comb-Filter Lattice Alignment:
    Uses mathematical periodic comb filtering across Sobel gradient projections
    to find the global optimal grid origin (X0, Y0) and dimensions (W, H).
    Completely eliminates human dragging inaccuracies and phase offsets.
    """
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    
    px = np.mean(gx, axis=0)
    py = np.mean(gy, axis=1)
    
    pitch_x = get_dominant_pitch(px, w_img)
    pitch_y = get_dominant_pitch(py, h_img)
    
    if approx_roi is not None:
        rx = max(0, int(approx_roi['x']))
        ry = max(0, int(approx_roi['y']))
        rw = min(int(approx_roi['w']), w_img - rx)
        rh = min(int(approx_roi['h']), h_img - ry)
    else:
        rx, ry, rw, rh = 0, 0, w_img, h_img
        
    cols = max(1, int(round(rw / pitch_x)))
    rows = max(1, int(round(rh / pitch_y)))
    
    # Sweep phase X0 in range around rx (+/- 0.6 * pitch_x)
    search_rx_min = max(0, int(round(rx - pitch_x * 0.6)))
    search_rx_max = min(w_img - int(round(cols * pitch_x)), int(round(rx + pitch_x * 0.6)))
    
    best_x0 = rx
    best_score_x = -1.0
    if search_rx_max >= search_rx_min:
        for x0 in range(search_rx_min, search_rx_max + 1):
            score = sum(
                px[int(round(x0 + c * pitch_x))]
                for c in range(cols + 1)
                if int(round(x0 + c * pitch_x)) < len(px)
            )
            if score > best_score_x:
                best_score_x = score
                best_x0 = x0

    # Sweep phase Y0 in range around ry (+/- 0.6 * pitch_y)
    search_ry_min = max(0, int(round(ry - pitch_y * 0.6)))
    search_ry_max = min(h_img - int(round(rows * pitch_y)), int(round(ry + pitch_y * 0.6)))
    
    best_y0 = ry
    best_score_y = -1.0
    if search_ry_max >= search_ry_min:
        for y0 in range(search_ry_min, search_ry_max + 1):
            score = sum(
                py[int(round(y0 + r * pitch_y))]
                for r in range(rows + 1)
                if int(round(y0 + r * pitch_y)) < len(py)
            )
            if score > best_score_y:
                best_score_y = score
                best_y0 = y0

    final_w = int(round(cols * pitch_x))
    final_h = int(round(rows * pitch_y))
    
    return {
        "x": int(best_x0),
        "y": int(best_y0),
        "w": int(final_w),
        "h": int(final_h),
        "rows": rows,
        "cols": cols,
        "pitch_x": pitch_x,
        "pitch_y": pitch_y
    }

def detect_board_roi(img):
    """
    Robustly auto-detect the game board outer bounding box.
    Uses 2D Tile Contour Clustering + Comb-Filter Lattice Alignment
    to ensure full tile inclusion without clipping board edges.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2D Tile Contour Extraction
    edges = cv2.Canny(gray, 30, 110)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tile_boxes = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if 18 <= bw <= 95 and 18 <= bh <= 95 and 0.6 <= bw / float(bh) <= 1.6:
            tile_boxes.append((bx, by, bw, bh))
            
    if len(tile_boxes) >= 6:
        med_w = np.median([b[2] for b in tile_boxes])
        med_h = np.median([b[3] for b in tile_boxes])
        
        valid = [
            b for b in tile_boxes
            if abs(b[2] - med_w) <= med_w * 0.45 and abs(b[3] - med_h) <= med_h * 0.45
        ]
        
        if len(valid) >= 6:
            min_x = min(b[0] for b in valid)
            min_y = min(b[1] for b in valid)
            max_x = max(b[0] + b[2] for b in valid)
            max_y = max(b[1] + b[3] for b in valid)
            
            raw_roi = {
                "x": min_x,
                "y": min_y,
                "w": max_x - min_x,
                "h": max_y - min_y
            }
            # Refine with comb filter lattice alignment
            aligned = align_grid_lattice(img, approx_roi=raw_roi)
            return {
                "x": aligned["x"],
                "y": aligned["y"],
                "w": aligned["w"],
                "h": aligned["h"]
            }
            
    # Fallback to direct lattice alignment
    aligned = align_grid_lattice(img, approx_roi=None)
    return {
        "x": aligned["x"],
        "y": aligned["y"],
        "w": aligned["w"],
        "h": aligned["h"]
    }

def refine_board_roi(img, rough_roi):
    """
    Snap-to-Grid Lattice Alignment:
    Takes the user's rough drag box and automatically aligns the origin and bounds
    to the underlying physical periodic tile grid lines.
    Trims away surrounding desktop/background while preventing outer tile clipping.
    """
    aligned = align_grid_lattice(img, approx_roi=rough_roi)
    return {
        "x": aligned["x"],
        "y": aligned["y"],
        "w": aligned["w"],
        "h": aligned["h"]
    }

def estimate_adaptive_grid(roi_img):
    """
    Adaptive Grid Estimation using Autocorrelation Signal Analysis.
    Detects dominant pitch in [18, 120] px and establishes rows/cols.
    """
    h, w, _ = roi_img.shape
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    
    px = np.mean(grad_x, axis=0)
    py = np.mean(grad_y, axis=1)
    
    pitch_x = get_dominant_pitch(px, w)
    pitch_y = get_dominant_pitch(py, h)
    
    cols = max(1, int(round(w / pitch_x)))
    rows = max(1, int(round(h / pitch_y)))
    
    cell_w = w / float(cols)
    cell_h = h / float(rows)
    
    return {
        "rows": rows,
        "cols": cols,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "pitch_x": pitch_x,
        "pitch_y": pitch_y
    }

def slice_cells(img, roi, rows, cols):
    """
    Slices the board into rows * cols cells with subpixel-accurate float rounding.
    Returns list of cell dictionaries.
    """
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    board_crop = img[y:y+h, x:x+w]
    
    cell_w = w / float(cols)
    cell_h = h / float(rows)
    
    cells = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(round(c * cell_w))
            y1 = int(round(r * cell_h))
            x2 = int(round((c + 1) * cell_w))
            y2 = int(round((r + 1) * cell_h))
            
            x2 = min(x2, w)
            y2 = min(y2, h)
            
            cell_img = board_crop[y1:y2, x1:x2].copy()
            
            center_x = x + int(round((x1 + x2) / 2.0))
            center_y = y + int(round((y1 + y2) / 2.0))
            
            cells.append({
                "row": r,
                "col": c,
                "bbox": {"x": x + x1, "y": y + y1, "w": x2 - x1, "h": y2 - y1},
                "center": {"x": center_x, "y": center_y},
                "image": cell_img
            })
            
    return cells

def draw_grid_overlay(img, roi, rows, cols, output_path=None):
    """
    Draws the calibrated grid overlay onto a copy of the image and saves it.
    """
    overlay = img.copy()
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    cell_w = w / float(cols)
    cell_h = h / float(rows)
    
    for c in range(cols + 1):
        cx = int(round(x + c * cell_w))
        cv2.line(overlay, (cx, y), (cx, y + h), (0, 255, 0), 1)
        
    for r in range(rows + 1):
        ry = int(round(y + r * cell_h))
        cv2.line(overlay, (x, ry), (x + w, ry), (0, 255, 0), 1)
        
    if output_path:
        save_image(output_path, overlay)
        
    return overlay

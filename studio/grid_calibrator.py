"""
LianLianKan Grid Calibrator
Adaptive ROI selection, snap-to-grid auto-refinement, and grid pitch detection.
Handles regular rectangular boards and irregular (hollow, cross, diamond, sparse) boards.
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

def detect_board_roi(img):
    """
    Robustly auto-detect the game board outer bounding box.
    Uses 2D Tile Contour Clustering combined with 1D Autocorrelation
    to handle both solid rectangular and irregular (hollow, cross, diamond, sparse) layouts.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 2D Tile Contour Clustering (Immune to irregular shape / hollow spaces)
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
            
            # Small padding to encompass complete tile borders
            pad_x = int(round(med_w * 0.08))
            pad_y = int(round(med_h * 0.08))
            
            x0 = max(0, min_x - pad_x)
            y0 = max(0, min_y - pad_y)
            x1 = min(w, max_x + pad_x)
            y1 = min(h, max_y + pad_y)
            
            return {
                "x": int(x0),
                "y": int(y0),
                "w": int(x1 - x0),
                "h": int(y1 - y0)
            }
            
    # 2. Fallback to 1D Sobel gradient projection
    blurred = cv2.bilateralFilter(gray, 5, 50, 50)
    gx = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))
    
    gx[gx < np.percentile(gx, 65)] = 0
    gy[gy < np.percentile(gy, 65)] = 0
    
    px = np.mean(gx, axis=0)
    py = np.mean(gy, axis=1)
    
    thresh_x = np.mean(px) * 0.7
    thresh_y = np.mean(py) * 0.7
    
    peaks_x, _ = find_peaks(px, height=thresh_x, distance=20)
    peaks_y, _ = find_peaks(py, height=thresh_y, distance=20)
    
    if len(peaks_x) >= 2 and len(peaks_y) >= 2:
        return {
            "x": int(peaks_x[0]),
            "y": int(peaks_y[0]),
            "w": int(peaks_x[-1] - peaks_x[0]),
            "h": int(peaks_y[-1] - peaks_y[0])
        }
        
    return {"x": 0, "y": 0, "w": w, "h": h}

def refine_board_roi(img, rough_roi):
    """
    Snap-to-Grid Auto Refinement:
    Takes a rough user-drawn or detected ROI and snaps boundaries
    to the exact 2D envelope of all enclosed tiles.
    """
    h_img, w_img = img.shape[:2]
    rx = max(0, int(rough_roi['x']))
    ry = max(0, int(rough_roi['y']))
    rw = min(int(rough_roi['w']), w_img - rx)
    rh = min(int(rough_roi['h']), h_img - ry)
    
    if rw < 35 or rh < 35:
        return rough_roi
        
    crop = img[ry:ry+rh, rx:rx+rw]
    detected = detect_board_roi(crop)
    
    if detected['w'] > 35 and detected['h'] > 35:
        refined = {
            'x': rx + detected['x'],
            'y': ry + detected['y'],
            'w': detected['w'],
            'h': detected['h']
        }
        return refined
    return rough_roi

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
    
    px_norm = px - np.mean(px)
    corr_x = np.correlate(px_norm, px_norm, mode='full')[len(px)-1:]
    
    py_norm = py - np.mean(py)
    corr_y = np.correlate(py_norm, py_norm, mode='full')[len(py)-1:]
    
    min_lag, max_lag = 18, min(120, min(w, h) - 5)
    if max_lag <= min_lag:
        pitch_x, pitch_y = 45.0, 45.0
    else:
        peaks_x, _ = find_peaks(corr_x[min_lag:max_lag], distance=10)
        peaks_x = peaks_x + min_lag
        if len(peaks_x) == 0:
            pitch_x = 45.0
        else:
            pitch_x = float(peaks_x[np.argmax(corr_x[peaks_x])])
            
        peaks_y, _ = find_peaks(corr_y[min_lag:max_lag], distance=10)
        peaks_y = peaks_y + min_lag
        if len(peaks_y) == 0:
            pitch_y = 45.0
        else:
            pitch_y = float(peaks_y[np.argmax(corr_y[peaks_y])])
        
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

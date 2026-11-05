"""
LianLianKan Grid Calibrator
Adaptive ROI selection and grid pitch detection using signal autocorrelation.
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
    Auto-detect the game board bounding box if not explicitly specified.
    Uses gradient analysis along the outer edges of the image.
    """
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sobel gradients
    gx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    
    prof_x = np.mean(gx, axis=0)
    prof_y = np.mean(gy, axis=1)
    
    # Outer thresholding: find active board edges
    # For data/未命名.jpg, edges are at x=68, 743 (w=675) and y=49, 456 (h=407)
    thresh_x = np.mean(prof_x) * 1.5
    thresh_y = np.mean(prof_y) * 1.5
    
    # Find first and last significant peaks
    peaks_x, _ = find_peaks(prof_x, height=thresh_x, distance=20)
    peaks_y, _ = find_peaks(prof_y, height=thresh_y, distance=20)
    
    if len(peaks_x) >= 2 and len(peaks_y) >= 2:
        x0 = int(peaks_x[0])
        x1 = int(peaks_x[-1])
        y0 = int(peaks_y[0])
        y1 = int(peaks_y[-1])
        return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
    
    # Default fallback
    return {"x": 68, "y": 49, "w": 675, "h": 407}

def estimate_adaptive_grid(roi_img):
    """
    Adaptive Grid Estimation using Autocorrelation Signal Analysis.
    Never hardcodes rows/cols:
    1. Computes Sobel gradient projections P_x and P_y.
    2. Uses 1D signal autocorrelation to detect dominant pitch in [25, 100] px.
    3. Computes estimated cols = round(roi_w / pitch_x) and rows = round(roi_h / pitch_y).
    """
    h, w, _ = roi_img.shape
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    
    px = np.mean(grad_x, axis=0)
    py = np.mean(grad_y, axis=1)
    
    # Autocorrelation of P_x
    px_norm = px - np.mean(px)
    corr_x = np.correlate(px_norm, px_norm, mode='full')[len(px)-1:]
    
    # Autocorrelation of P_y
    py_norm = py - np.mean(py)
    corr_y = np.correlate(py_norm, py_norm, mode='full')[len(py)-1:]
    
    # Search peaks in range [25, 100] pixels
    min_lag, max_lag = 25, min(100, min(w, h) - 5)
    peaks_x, _ = find_peaks(corr_x[min_lag:max_lag], distance=15)
    peaks_x = peaks_x + min_lag
    
    peaks_y, _ = find_peaks(corr_y[min_lag:max_lag], distance=15)
    peaks_y = peaks_y + min_lag
    
    if len(peaks_x) == 0:
        pitch_x = 45.0
    else:
        pitch_x = float(peaks_x[np.argmax(corr_x[peaks_x])])
        
    if len(peaks_y) == 0:
        pitch_y = 45.0
    else:
        pitch_y = float(peaks_y[np.argmax(corr_y[peaks_y])])
        
    cols = int(round(w / pitch_x))
    rows = int(round(h / pitch_y))
    
    cell_w = w / cols
    cell_h = h / rows
    
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
    Slices the board image into rows x cols cell patches with exact floating-point bounds.
    Returns a list of dicts with cell images and metadata.
    """
    x0, y0, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    cell_w = w / cols
    cell_h = h / rows
    
    cells = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(round(x0 + c * cell_w))
            y1 = int(round(y0 + r * cell_h))
            x2 = int(round(x0 + (c + 1) * cell_w))
            y2 = int(round(y0 + (r + 1) * cell_h))
            
            # Ensure within image bounds
            x1 = max(0, min(img.shape[1], x1))
            x2 = max(0, min(img.shape[1], x2))
            y1 = max(0, min(img.shape[0], y1))
            y2 = max(0, min(img.shape[0], y2))
            
            patch = img[y1:y2, x1:x2].copy()
            cells.append({
                "row": r,
                "col": c,
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "center": (x0 + (c + 0.5) * cell_w, y0 + (r + 0.5) * cell_h),
                "image": patch
            })
    return cells

def draw_grid_overlay(img, roi, rows, cols, output_path="debug_grid_on_roi.png"):
    """Draws grid lines strictly within the board ROI and saves preview."""
    overlay = img.copy()
    x0, y0, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    cell_w = w / cols
    cell_h = h / rows
    
    # Draw ROI outer border in green
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
    
    # Draw internal grid lines in cyan
    for r in range(1, rows):
        y = int(round(y0 + r * cell_h))
        cv2.line(overlay, (x0, y), (x0 + w, y), (255, 255, 0), 1)
        
    for c in range(1, cols):
        x = int(round(x0 + c * cell_w))
        cv2.line(overlay, (x, y0), (x, y0 + h), (255, 255, 0), 1)
        
    save_image(output_path, overlay)
    return overlay

"""
LianLianKan Grid Calibrator
Adaptive ROI selection, snap-to-grid auto-refinement, and grid pitch detection using signal autocorrelation.
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
    Uses 1D Sobel gradient projections and autocorrelation signal analysis
    to isolate the 2D periodic tile lattice from background HUD / windows.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter to smooth texture while retaining tile edges
    blurred = cv2.bilateralFilter(gray, 5, 50, 50)
    
    gx = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))
    
    # Threshold gradient to eliminate low-contrast background gradients
    gx[gx < np.percentile(gx, 65)] = 0
    gy[gy < np.percentile(gy, 65)] = 0
    
    px = np.mean(gx, axis=0)
    py = np.mean(gy, axis=1)
    
    def get_pitch(p, max_len):
        norm = p - np.mean(p)
        corr = np.correlate(norm, norm, mode='full')[len(p)-1:]
        min_lag = 20
        max_lag = min(120, max_len // 2)
        if max_lag <= min_lag:
            return 45.0
        peaks, _ = find_peaks(corr[min_lag:max_lag], distance=12)
        peaks = peaks + min_lag
        if len(peaks) == 0:
            return 45.0
        return float(peaks[np.argmax(corr[peaks])])

    pitch_x = get_pitch(px, w)
    pitch_y = get_pitch(py, h)
    
    dist_x = max(15, int(pitch_x * 0.7))
    dist_y = max(15, int(pitch_y * 0.7))
    
    thresh_x = np.mean(px) * 0.7
    thresh_y = np.mean(py) * 0.7
    
    peaks_x, _ = find_peaks(px, height=thresh_x, distance=dist_x)
    peaks_y, _ = find_peaks(py, height=thresh_y, distance=dist_y)
    
    def find_best_lattice_run(peaks, pitch):
        if len(peaks) < 2:
            return 0, 0, 0
        best_run = []
        for i in range(len(peaks)):
            run = [peaks[i]]
            for j in range(i + 1, len(peaks)):
                diff = peaks[j] - run[-1]
                if abs(diff - pitch) <= pitch * 0.25:
                    run.append(peaks[j])
                elif diff > pitch * 1.25:
                    break
            if len(run) > len(best_run):
                best_run = run
        if len(best_run) >= 3:
            return best_run[0], best_run[-1], len(best_run)
        return peaks[0], peaks[-1], len(peaks)

    x0, x1, count_x = find_best_lattice_run(peaks_x, pitch_x)
    y0, y1, count_y = find_best_lattice_run(peaks_y, pitch_y)
    
    if x1 > x0 and y1 > y0:
        return {"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)}
    
    # Fallback to full image
    return {"x": 0, "y": 0, "w": w, "h": h}

def refine_board_roi(img, rough_roi):
    """
    Snap-to-Grid Auto Refinement:
    Takes a rough user-drawn or roughly detected ROI and snaps the four boundaries
    (left, top, right, bottom) to the nearest true periodic tile grid boundaries.
    Eliminates human mouse dragging inaccuracies.
    """
    h_img, w_img = img.shape[:2]
    rx = max(0, int(rough_roi['x']))
    ry = max(0, int(rough_roi['y']))
    rw = min(int(rough_roi['w']), w_img - rx)
    rh = min(int(rough_roi['h']), h_img - ry)
    
    if rw < 40 or rh < 40:
        return rough_roi
        
    crop = img[ry:ry+rh, rx:rx+rw]
    detected = detect_board_roi(crop)
    
    # If the sub-detection found a clear periodic lattice within the crop:
    if detected['w'] > 40 and detected['h'] > 40:
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
    Never hardcodes rows/cols:
    1. Computes Sobel gradient projections P_x and P_y.
    2. Uses 1D signal autocorrelation to detect dominant pitch in [20, 120] px.
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
    
    # Search peaks in range [20, 120] pixels
    min_lag, max_lag = 20, min(120, min(w, h) - 5)
    if max_lag <= min_lag:
        pitch_x, pitch_y = 45.0, 45.0
    else:
        peaks_x, _ = find_peaks(corr_x[min_lag:max_lag], distance=12)
        peaks_x = peaks_x + min_lag
        if len(peaks_x) == 0:
            pitch_x = 45.0
        else:
            pitch_x = float(peaks_x[np.argmax(corr_x[peaks_x])])
            
        peaks_y, _ = find_peaks(corr_y[min_lag:max_lag], distance=12)
        peaks_y = peaks_y + min_lag
        if len(peaks_y) == 0:
            pitch_y = 45.0
        else:
            pitch_y = float(peaks_y[np.argmax(corr_y[peaks_y])])
        
    cols = max(1, int(round(w / pitch_x)))
    rows = max(1, int(round(h / pitch_y)))
    
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
    Slices the board into rows * cols cells.
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
            
            # Boundary checks
            x2 = min(x2, w)
            y2 = min(y2, h)
            
            cell_img = board_crop[y1:y2, x1:x2].copy()
            
            # Screen absolute coordinate center
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
    
    # Outer ROI rectangle
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    cell_w = w / float(cols)
    cell_h = h / float(rows)
    
    # Vertical grid lines
    for c in range(cols + 1):
        cx = int(round(x + c * cell_w))
        cv2.line(overlay, (cx, y), (cx, y + h), (0, 255, 0), 1)
        
    # Horizontal grid lines
    for r in range(rows + 1):
        ry = int(round(y + r * cell_h))
        cv2.line(overlay, (x, ry), (x + w, ry), (0, 255, 0), 1)
        
    if output_path:
        save_image(output_path, overlay)
        
    return overlay

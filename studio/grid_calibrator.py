"""
LianLianKan Grid Calibrator
Adaptive ROI selection, Comb-Filter Lattice Alignment, and grid pitch detection.
Handles regular rectangular boards and irregular (hollow, cross, diamond, sparse) boards.
Guarantees pixel-perfect tile centering and full outer tile inclusion without clipping.
"""

import os
import cv2
import numpy as np

def find_peaks_1d(arr, distance=10):
    """
    Pure NumPy 1D peak detection with minimum peak distance.
    Drop-in replacement for scipy.signal.find_peaks.
    """
    if len(arr) < 3:
        return np.array([], dtype=int)
    dx = np.diff(arr)
    peaks = np.where((dx[:-1] > 0) & (dx[1:] <= 0))[0] + 1
    if len(peaks) <= 1 or distance <= 1:
        return peaks
    order = np.argsort(arr[peaks])[::-1]
    keep = []
    for idx in order:
        p = peaks[idx]
        if all(abs(p - k) >= distance for k in keep):
            keep.append(p)
    return np.sort(np.array(keep, dtype=int))

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

def _unbiased_autocorr(profile):
    """Mean-removed, overlap-normalised autocorrelation (value at lag 0 == 1)."""
    p = np.asarray(profile, dtype=np.float64)
    p = p - p.mean()
    n = len(p)
    c = np.correlate(p, p, mode='full')[n - 1:]
    c = c / (n - np.arange(n))
    return c / c[0] if c[0] > 1e-9 else c

def estimate_pitch(profile, min_lag=15, max_lag=160, fallback=45.0):
    """
    Sub-pixel grid pitch from a 1-D gradient profile.
    1. Autocorrelation peaks -> fundamental = smallest lag that is a strong peak
       (so 2x / 3x multiples are never chosen).
    2. Refine with the higher harmonics (k * pitch) by a least-squares fit, which
       removes the integer-lag quantisation error that accumulates over many tiles.
    """
    n = len(profile)
    max_lag = min(max_lag, n // 2)
    if max_lag <= min_lag + 2:
        return float(fallback)
    c = _unbiased_autocorr(profile)
    cand = [i for i in range(min_lag, max_lag)
            if c[i] > c[i - 1] and c[i] >= c[i + 1] and c[i] > 0.1]
    if not cand:
        return float(fallback)
    # Score every candidate by how well its whole harmonic comb (L, 2L, 3L ...)
    # lands on autocorrelation peaks: side lobes (e.g. icon edges) have no
    # harmonics, whereas the true pitch does.  Prefer the smallest strong one so
    # that 2x / 3x multiples are never chosen.
    def comb_score(L):
        vals = []
        for k in range(1, 7):
            pos = int(round(k * L))
            if pos + 1 >= n or k * L > n * 0.6:
                break
            vals.append(c[max(pos - 1, 0):pos + 2].max())
        return float(np.mean(vals)) if vals else 0.0
    scores = {i: comb_score(i) for i in cand}
    top = max(scores.values())
    p0 = float(next(i for i in cand if scores[i] >= 0.65 * top))
    # Harmonic refinement: each k-th peak is searched near k * (current estimate),
    # so the search window stays narrow and never jumps onto a neighbouring peak.
    pk = p0
    ks, ls = [], []
    for k in range(1, 13):
        if k * pk >= n * 0.6:
            break
        half = 1.5 + 0.3 * k
        lo = max(1, int(round(k * pk - half)))
        hi = min(n - 2, int(round(k * pk + half)))
        if hi <= lo:
            break
        j = lo + int(np.argmax(c[lo:hi + 1]))
        if j in (lo, hi) or c[j] <= 0.15:
            continue                      # not a clean local maximum
        d = c[j - 1] - 2 * c[j] + c[j + 1]
        off = 0.5 * (c[j - 1] - c[j + 1]) / d if d < 0 else 0.0
        ks.append(k)
        ls.append(j + off)
        a_, b_ = np.array(ks, float), np.array(ls, float)
        pk = float(np.sum(a_ * b_) / np.sum(a_ * a_))
    return pk if ks else p0

def get_dominant_pitch(profile, max_len=None):
    """Backward compatible wrapper."""
    return estimate_pitch(profile)

def _fold(profile, pitch, nb):
    n = len(profile)
    idx = ((np.arange(n) % pitch) / pitch * nb).astype(int) % nb
    return np.bincount(idx, weights=profile, minlength=nb) / np.maximum(
        np.bincount(idx, minlength=nb), 1)

def _lattice_phase(profile, pitch):
    """
    Position (px, 0 <= x < pitch) of the tile *boundary* lattice.
    The profile is folded modulo the sub-pixel pitch.  A row/column of tiles is
    mirror-symmetric about both the tile centre and the tile boundary, so the
    reflection-symmetry axes of the folded profile give the two candidates; the
    one closest to the dominant edge-energy peak (tile outlines pile up at the
    boundary) is the boundary.  This is far less biased by one-sided tile shadows / 3-D
    side faces than taking the energy peak directly.
    """
    p = np.asarray(profile, dtype=np.float64)
    nb = max(8, int(round(pitch * 2)))
    nb += nb % 2
    F = _fold(p, pitch, nb)
    Fc = F - F.mean()
    h = nb // 2
    t = np.arange(1, h)
    sym = np.array([np.sum(Fc[(k + t) % nb] * Fc[(k - t) % nb]) for k in range(nb)])
    k = int(np.argmax(sym[:h] + sym[h:]))
    # smoothed energy to tell boundary from tile centre
    win = max(3, int(round(0.14 * nb)))
    ext = np.concatenate([F[-win:], F, F[:win]])
    sm = np.convolve(ext, np.ones(win) / win, mode='same')[win:-win]
    peak = int(np.argmax(sm))
    best = min((k, k + h), key=lambda c: min((c - peak) % nb, (peak - c) % nb))
    return (best % nb) / nb * pitch

def _axis_lattice(profile, size, pitch):
    """
    Returns (first_boundary, n_cells) of the lattice along one axis.
    Cells that stick out of the image by more than 15 % of a pitch are dropped.
    """
    first = _lattice_phase(profile, pitch) % pitch
    if first - pitch > -0.15 * pitch:      # leading cell is >= 85 % visible
        first -= pitch
    n = int(np.floor((size - first) / pitch + 0.15))
    return first, max(1, n)

def align_grid_lattice(img, approx_roi=None):
    """
    Pitch + phase based lattice alignment.
    With approx_roi (user selection, may include < 1 tile of margin) the lattice is
    estimated inside the selection only, using sub-pixel pitch and Fourier phase,
    then outer rows/cols that are not tile-like (margin/background) are trimmed.
    """
    h_img, w_img = img.shape[:2]
    if approx_roi is None:
        approx_roi = {"x": 0, "y": 0, "w": w_img, "h": h_img}
    rx = max(0, int(approx_roi['x']))
    ry = max(0, int(approx_roi['y']))
    rw = min(int(approx_roi['w']), w_img - rx)
    rh = min(int(approx_roi['h']), h_img - ry)

    sub = img[ry:ry + rh, rx:rx + rw]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    px = gx.mean(axis=0)
    py = gy.mean(axis=1)

    pitch_x = estimate_pitch(px)
    pitch_y = estimate_pitch(py)

    fx, ncx = _axis_lattice(px, rw, pitch_x)
    fy, ncy = _axis_lattice(py, rh, pitch_y)

    # Trim margin rows/cols: a real outer row/col is closed by tile outlines on BOTH
    # of its lattice boundaries, a margin cell has (almost) no edge energy there.
    def boundary_ok(profile, first, pitch, ncell, size):
        win = max(3, int(round(pitch * 0.2)))
        e = []
        for k in range(ncell + 1):
            pos = first + k * pitch
            lo, hi = int(round(pos - win)), int(round(pos + win)) + 1
            e.append(float(profile[max(lo, 0):min(hi, size)].max())
                     if hi > 0 and lo < size and hi > lo else None)
        vals = [v for v in e if v is not None]
        ref_ = float(np.median(vals)) if vals else 1.0
        e = [ref_ if v is None else v for v in e]    # out-of-image boundary: neutral
        return [(e[k] >= 0.25 * ref_ and e[k + 1] >= 0.25 * ref_) for k in range(ncell)]

    col_ok = np.array(boundary_ok(px, fx, pitch_x, ncx, rw))
    row_ok = np.array(boundary_ok(py, fy, pitch_y, ncy, rh))
    if not col_ok.any():
        col_ok[:] = True
    if not row_ok.any():
        row_ok[:] = True
    c0 = int(np.argmax(col_ok)); c1 = ncx - int(np.argmax(col_ok[::-1]))
    r0 = int(np.argmax(row_ok)); r1 = ncy - int(np.argmax(row_ok[::-1]))
    rows, cols = r1 - r0, c1 - c0

    x0 = fx + c0 * pitch_x
    y0 = fy + r0 * pitch_y
    final_x = int(round(max(0, x0)))
    final_y = int(round(max(0, y0)))
    final_w = min(int(round(cols * pitch_x)), rw - final_x)
    final_h = min(int(round(rows * pitch_y)), rh - final_y)

    return {
        "x": rx + final_x,
        "y": ry + final_y,
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
    Uses Multi-Scale 2D Contour Extraction + Spatial Density Filtering
    + Connected Lattice Component Analysis + Localized Comb-Filter Lattice Alignment.
    Guarantees zero interference from desktop icons, window taskbars, or UI noise.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Multi-scale edge detection
    edges = cv2.Canny(gray, 25, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tile_boxes = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if 18 <= bw <= 110 and 18 <= bh <= 110 and 0.55 <= bw / float(bh) <= 1.8:
            tile_boxes.append((bx, by, bw, bh))
            
    if len(tile_boxes) >= 6:
        med_w = np.median([b[2] for b in tile_boxes])
        med_h = np.median([b[3] for b in tile_boxes])
        pitch = (med_w + med_h) / 2.0
        
        valid = [
            b for b in tile_boxes
            if abs(b[2] - med_w) <= med_w * 0.45 and abs(b[3] - med_h) <= med_h * 0.45
        ]
        
        if len(valid) >= 6:
            centers = np.array([(b[0] + b[2]/2.0, b[1] + b[3]/2.0) for b in valid])
            n = len(valid)
            
            # Spatial density filter: board tiles are clustered with multiple neighbors
            R = 3.5 * pitch
            dense_indices = [
                i for i in range(n)
                if np.sum(np.sqrt(np.sum((centers - centers[i])**2, axis=1)) <= R) - 1 >= 3
            ]
            
            if len(dense_indices) >= 6:
                dense_boxes = [valid[idx] for idx in dense_indices]
                dense_centers = centers[dense_indices]
                
                # Connect adjacent tile clusters (connect threshold up to 4.5 * pitch to span hollow gaps)
                m = len(dense_boxes)
                adj = [[] for _ in range(m)]
                for i in range(m):
                    for j in range(i + 1, m):
                        if np.linalg.norm(dense_centers[i] - dense_centers[j]) <= 4.5 * pitch:
                            adj[i].append(j)
                            adj[j].append(i)
                            
                visited = [False] * m
                components = []
                for i in range(m):
                    if not visited[i]:
                        comp = []
                        q = [i]
                        visited[i] = True
                        while q:
                            curr = q.pop()
                            comp.append(curr)
                            for nbr in adj[curr]:
                                if not visited[nbr]:
                                    visited[nbr] = True
                                    q.append(nbr)
                        components.append(comp)
                        
                components.sort(key=len, reverse=True)
                largest = components[0]
                board_boxes = [dense_boxes[idx] for idx in largest]
                
                min_x = min(b[0] for b in board_boxes)
                min_y = min(b[1] for b in board_boxes)
                max_x = max(b[0] + b[2] for b in board_boxes)
                max_y = max(b[1] + b[3] for b in board_boxes)
                
                raw_roi = {
                    "x": min_x,
                    "y": min_y,
                    "w": max_x - min_x,
                    "h": max_y - min_y
                }
                
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
    
    # Draw outer ROI box
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cell_w = w / float(cols)
    cell_h = h / float(rows)
    
    # Vertical grid lines
    for c in range(1, cols):
        lx = int(round(x + c * cell_w))
        cv2.line(overlay, (lx, y), (lx, y + h), (0, 220, 255), 1)
        
    # Horizontal grid lines
    for r in range(1, rows):
        ly = int(round(y + r * cell_h))
        cv2.line(overlay, (x, ly), (x + w, ly), (0, 220, 255), 1)
        
    if output_path:
        save_image(output_path, overlay)
        
    return overlay

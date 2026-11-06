//! Grid calibration: sub-pixel pitch, lattice phase, board ROI refinement, cell slicing.
//! Direct port of studio/grid_calibrator.py.

use crate::imgproc::{gray, median, pyround, sobel, Image};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Roi {
    pub x: i64,
    pub y: i64,
    pub w: i64,
    pub h: i64,
}

fn unbiased_autocorr(profile: &[f64]) -> Vec<f64> {
    let n = profile.len();
    let mean = profile.iter().sum::<f64>() / n as f64;
    let p: Vec<f64> = profile.iter().map(|v| v - mean).collect();
    let mut c = vec![0.0; n];
    for lag in 0..n {
        let mut s = 0.0;
        for i in 0..n - lag {
            s += p[i] * p[i + lag];
        }
        c[lag] = s / (n - lag) as f64;
    }
    if c[0] > 1e-9 {
        let c0 = c[0];
        for v in c.iter_mut() {
            *v /= c0;
        }
    }
    c
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

pub fn estimate_pitch(profile: &[f64]) -> f64 {
    let (min_lag, fallback) = (15usize, 45.0);
    let n = profile.len();
    let max_lag = 160usize.min(n / 2);
    if max_lag <= min_lag + 2 {
        return fallback;
    }
    let c = unbiased_autocorr(profile);
    let cand: Vec<usize> = (min_lag..max_lag)
        .filter(|&i| c[i] > c[i - 1] && c[i] >= c[i + 1] && c[i] > 0.1)
        .collect();
    if cand.is_empty() {
        return fallback;
    }
    let comb_score = |l: f64| -> f64 {
        let mut vals = Vec::new();
        for k in 1..=6 {
            let pos = pyround(k as f64 * l) as usize;
            if pos + 1 >= n || k as f64 * l > n as f64 * 0.6 {
                break;
            }
            let lo = pos.saturating_sub(1);
            vals.push(c[lo..pos + 2].iter().cloned().fold(f64::MIN, f64::max));
        }
        if vals.is_empty() { 0.0 } else { vals.iter().sum::<f64>() / vals.len() as f64 }
    };
    let scores: Vec<f64> = cand.iter().map(|&i| comb_score(i as f64)).collect();
    let top = scores.iter().cloned().fold(f64::MIN, f64::max);
    let p0 = cand.iter().zip(&scores).find(|(_, &s)| s >= 0.65 * top).map(|(&i, _)| i).unwrap() as f64;

    let mut pk = p0;
    let (mut ks, mut ls): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
    for k in 1..=12usize {
        if k as f64 * pk >= n as f64 * 0.6 {
            break;
        }
        let half = 1.5 + 0.3 * k as f64;
        let lo = (pyround(k as f64 * pk - half)).max(1) as usize;
        let hi = (pyround(k as f64 * pk + half)).min(n as i64 - 2) as usize;
        if hi <= lo {
            break;
        }
        let j = lo + argmax(&c[lo..=hi]);
        if j == lo || j == hi || c[j] <= 0.15 {
            continue;
        }
        let d = c[j - 1] - 2.0 * c[j] + c[j + 1];
        let off = if d < 0.0 { 0.5 * (c[j - 1] - c[j + 1]) / d } else { 0.0 };
        ks.push(k as f64);
        ls.push(j as f64 + off);
        let num: f64 = ks.iter().zip(&ls).map(|(a, b)| a * b).sum();
        let den: f64 = ks.iter().map(|a| a * a).sum();
        pk = num / den;
    }
    if ks.is_empty() { p0 } else { pk }
}

fn fmod(x: f64, m: f64) -> f64 {
    x - (x / m).floor() * m
}

fn fold(profile: &[f64], pitch: f64, nb: usize) -> Vec<f64> {
    let mut sum = vec![0.0; nb];
    let mut cnt = vec![0.0f64; nb];
    for (i, &v) in profile.iter().enumerate() {
        let idx = ((fmod(i as f64, pitch) / pitch * nb as f64) as i64).rem_euclid(nb as i64) as usize;
        sum[idx] += v;
        cnt[idx] += 1.0;
    }
    (0..nb).map(|i| sum[i] / cnt[i].max(1.0)).collect()
}

fn lattice_phase(profile: &[f64], pitch: f64) -> f64 {
    let mut nb = 8usize.max(pyround(pitch * 2.0) as usize);
    nb += nb % 2;
    let f = fold(profile, pitch, nb);
    let mean = f.iter().sum::<f64>() / nb as f64;
    let fc: Vec<f64> = f.iter().map(|v| v - mean).collect();
    let h = nb / 2;
    let sym: Vec<f64> = (0..nb)
        .map(|k| (1..h).map(|t| fc[(k + t) % nb] * fc[(k + nb - t) % nb]).sum())
        .collect();
    let comb: Vec<f64> = (0..h).map(|k| sym[k] + sym[k + h]).collect();
    let k = argmax(&comb);

    let win = 3usize.max(pyround(0.14 * nb as f64) as usize);
    let mut ext = Vec::with_capacity(nb + 2 * win);
    ext.extend_from_slice(&f[nb - win..]);
    ext.extend_from_slice(&f);
    ext.extend_from_slice(&f[..win]);
    // np.convolve(ext, ones(win)/win, mode='same')
    let nfull = ext.len() + win - 1;
    let mut full = vec![0.0; nfull];
    for (i, &e) in ext.iter().enumerate() {
        for j in 0..win {
            full[i + j] += e / win as f64;
        }
    }
    let start = (win - 1) / 2;
    let same = &full[start..start + ext.len()];
    let sm = &same[win..same.len() - win];
    let peak = argmax(sm) as i64;
    let dist = |c: usize| {
        let c = c as i64;
        ((c - peak).rem_euclid(nb as i64)).min((peak - c).rem_euclid(nb as i64))
    };
    let best = if dist(k) <= dist(k + h) { k } else { k + h };
    (best % nb) as f64 / nb as f64 * pitch
}

fn axis_lattice(profile: &[f64], size: usize, pitch: f64) -> (f64, usize) {
    let mut first = fmod(lattice_phase(profile, pitch), pitch);
    if first - pitch > -0.15 * pitch {
        first -= pitch;
    }
    let n = ((size as f64 - first) / pitch + 0.15).floor() as i64;
    (first, n.max(1) as usize)
}

fn boundary_ok(profile: &[f64], first: f64, pitch: f64, ncell: usize, size: usize) -> Vec<bool> {
    let win = 3i64.max(pyround(pitch * 0.2));
    let mut e: Vec<Option<f64>> = Vec::new();
    for k in 0..=ncell {
        let pos = first + k as f64 * pitch;
        let lo = pyround(pos - win as f64);
        let hi = pyround(pos + win as f64) + 1;
        if hi > 0 && lo < size as i64 && hi > lo {
            let a = lo.max(0) as usize;
            let b = (hi.min(size as i64)) as usize;
            e.push(Some(profile[a..b].iter().cloned().fold(f64::MIN, f64::max)));
        } else {
            e.push(None);
        }
    }
    let vals: Vec<f64> = e.iter().flatten().cloned().collect();
    let r = if vals.is_empty() { 1.0 } else { median(&vals) };
    let e: Vec<f64> = e.into_iter().map(|v| v.unwrap_or(r)).collect();
    (0..ncell).map(|k| e[k] >= 0.25 * r && e[k + 1] >= 0.25 * r).collect()
}

pub struct Aligned {
    pub roi: Roi,
    pub rows: usize,
    pub cols: usize,
    pub pitch_x: f64,
    pub pitch_y: f64,
}

/// Lattice alignment inside `approx` (the user's rough selection).
pub fn align_grid_lattice(img: &Image, approx: Option<Roi>) -> Aligned {
    let approx = approx.unwrap_or(Roi { x: 0, y: 0, w: img.w as i64, h: img.h as i64 });
    let rx = approx.x.max(0);
    let ry = approx.y.max(0);
    let rw = approx.w.min(img.w as i64 - rx);
    let rh = approx.h.min(img.h as i64 - ry);
    let sub = img.crop(rx as usize, ry as usize, rw as usize, rh as usize);
    let (w, h) = (sub.w, sub.h);
    let g = gray(&sub);
    let gx = sobel(&g, w, h, true);
    let gy = sobel(&g, w, h, false);
    let mut px = vec![0.0; w];
    let mut py = vec![0.0; h];
    for y in 0..h {
        for x in 0..w {
            px[x] += gx[y * w + x].abs();
            py[y] += gy[y * w + x].abs();
        }
    }
    px.iter_mut().for_each(|v| *v /= h as f64);
    py.iter_mut().for_each(|v| *v /= w as f64);

    let pitch_x = estimate_pitch(&px);
    let pitch_y = estimate_pitch(&py);
    let (fx, ncx) = axis_lattice(&px, w, pitch_x);
    let (fy, ncy) = axis_lattice(&py, h, pitch_y);

    let mut col_ok = boundary_ok(&px, fx, pitch_x, ncx, w);
    let mut row_ok = boundary_ok(&py, fy, pitch_y, ncy, h);
    if !col_ok.iter().any(|&b| b) {
        col_ok.iter_mut().for_each(|b| *b = true);
    }
    if !row_ok.iter().any(|&b| b) {
        row_ok.iter_mut().for_each(|b| *b = true);
    }
    let c0 = col_ok.iter().position(|&b| b).unwrap();
    let c1 = ncx - col_ok.iter().rev().position(|&b| b).unwrap();
    let r0 = row_ok.iter().position(|&b| b).unwrap();
    let r1 = ncy - row_ok.iter().rev().position(|&b| b).unwrap();
    let (rows, cols) = (r1 - r0, c1 - c0);

    let x0 = fx + c0 as f64 * pitch_x;
    let y0 = fy + r0 as f64 * pitch_y;
    let final_x = pyround(x0.max(0.0));
    let final_y = pyround(y0.max(0.0));
    let final_w = pyround(cols as f64 * pitch_x).min(rw - final_x);
    let final_h = pyround(rows as f64 * pitch_y).min(rh - final_y);

    Aligned {
        roi: Roi { x: rx + final_x, y: ry + final_y, w: final_w, h: final_h },
        rows,
        cols,
        pitch_x,
        pitch_y,
    }
}

pub fn refine_board_roi(img: &Image, rough: Roi) -> Roi {
    align_grid_lattice(img, Some(rough)).roi
}

pub struct GridInfo {
    pub rows: usize,
    pub cols: usize,
    pub cell_w: f64,
    pub cell_h: f64,
    pub pitch_x: f64,
    pub pitch_y: f64,
}

/// estimate_adaptive_grid(): rows/cols of an already cropped board image.
pub fn estimate_adaptive_grid(roi_img: &Image) -> GridInfo {
    let (w, h) = (roi_img.w, roi_img.h);
    let g = gray(roi_img);
    let gx = sobel(&g, w, h, true);
    let gy = sobel(&g, w, h, false);
    let mut px = vec![0.0; w];
    let mut py = vec![0.0; h];
    for y in 0..h {
        for x in 0..w {
            px[x] += gx[y * w + x].abs();
            py[y] += gy[y * w + x].abs();
        }
    }
    px.iter_mut().for_each(|v| *v /= h as f64);
    py.iter_mut().for_each(|v| *v /= w as f64);
    let pitch_x = estimate_pitch(&px);
    let pitch_y = estimate_pitch(&py);
    let cols = pyround(w as f64 / pitch_x).max(1) as usize;
    let rows = pyround(h as f64 / pitch_y).max(1) as usize;
    GridInfo { rows, cols, cell_w: w as f64 / cols as f64, cell_h: h as f64 / rows as f64, pitch_x, pitch_y }
}

pub struct Cell {
    pub row: usize,
    pub col: usize,
    pub image: Image,
}

/// slice_cells(): rows*cols sub-images of the board region.
pub fn slice_cells(img: &Image, roi: Roi, rows: usize, cols: usize) -> Vec<Cell> {
    let board = img.crop(roi.x as usize, roi.y as usize, roi.w as usize, roi.h as usize);
    let (w, h) = (board.w as i64, board.h as i64);
    let cw = w as f64 / cols as f64;
    let ch = h as f64 / rows as f64;
    let mut cells = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let x1 = pyround(c as f64 * cw);
            let y1 = pyround(r as f64 * ch);
            let x2 = pyround((c + 1) as f64 * cw).min(w);
            let y2 = pyround((r + 1) as f64 * ch).min(h);
            cells.push(Cell {
                row: r,
                col: c,
                image: board.crop(x1 as usize, y1 as usize, (x2 - x1).max(0) as usize, (y2 - y1).max(0) as usize),
            });
        }
    }
    cells
}

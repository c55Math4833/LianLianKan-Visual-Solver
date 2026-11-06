//! Cell classification: matchable tile vs. empty ground vs. obstacle block.
//! Port of studio/feature_extractor.py (colour-independent rules).

use crate::calibrate::Cell;
use crate::imgproc::{gray, median, rgb_to_lab, sobel, Image};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellType {
    Matchable,
    Flat, // provisional: empty or obstacle, decided board-wide
    Empty,
    Obstacle,
}

pub struct Classified {
    pub cell: Cell,
    pub ty: CellType,
    pub core_mean_rgb: [f64; 3],
    pub ring_grad: f64,
}

/// (core mean RGB, core std (avg per channel), ring gradient)
pub fn cell_metrics(img: &Image) -> ([f64; 3], f64, f64) {
    let (w, h) = (img.w, img.h);
    let g = gray(img);
    let gx = sobel(&g, w, h, true);
    let gy = sobel(&g, w, h, false);

    let y0 = (h as f64 * 0.25) as usize;
    let y1 = ((h as f64 * 0.75) as usize).max(y0 + 2);
    let x0 = (w as f64 * 0.25) as usize;
    let x1 = ((w as f64 * 0.75) as usize).max(x0 + 2);
    let (y1, x1) = (y1.min(h), x1.min(w));
    let mut sum = [0.0f64; 3];
    let mut n = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            let p = img.get(x, y);
            for c in 0..3 {
                sum[c] += p[c] as f64;
            }
            n += 1.0;
        }
    }
    let mean = [sum[0] / n, sum[1] / n, sum[2] / n];
    let mut var = [0.0f64; 3];
    for y in y0..y1 {
        for x in x0..x1 {
            let p = img.get(x, y);
            for c in 0..3 {
                var[c] += (p[c] as f64 - mean[c]).powi(2);
            }
        }
    }
    let std = (0..3).map(|c| (var[c] / n).sqrt()).sum::<f64>() / 3.0;

    // ring = everything outside the central 40 % box (ring_mask in Python)
    let ry0 = (h as f64 * 0.3) as usize;
    let ry1 = (h as f64 * 0.7) as usize;
    let rx0 = (w as f64 * 0.3) as usize;
    let rx1 = (w as f64 * 0.7) as usize;
    let mut s = 0.0;
    let mut cnt = 0.0;
    for y in 0..h {
        for x in 0..w {
            if y >= ry0 && y < ry1 && x >= rx0 && x < rx1 {
                continue;
            }
            s += gx[y * w + x].abs() + gy[y * w + x].abs();
            cnt += 1.0;
        }
    }
    (mean, std, if cnt > 0.0 { s / cnt } else { 0.0 })
}

pub fn classify_cells(cells: Vec<Cell>) -> Vec<Classified> {
    let mut out: Vec<Classified> = cells
        .into_iter()
        .map(|cell| {
            let (mean, std, ring) = cell_metrics(&cell.image);
            let ty = if std < 10.0 { CellType::Flat } else { CellType::Matchable };
            Classified { cell, ty, core_mean_rgb: mean, ring_grad: ring }
        })
        .collect();

    let flats: Vec<usize> = (0..out.len()).filter(|&i| out[i].ty == CellType::Flat).collect();
    if flats.is_empty() {
        return out;
    }
    let tile_rings: Vec<f64> =
        out.iter().filter(|c| c.ty == CellType::Matchable).map(|c| c.ring_grad).collect();
    let mut ref_ring = if !tile_rings.is_empty() {
        median(&tile_rings)
    } else {
        let m = flats.iter().map(|&i| out[i].ring_grad).fold(0.0, f64::max);
        if m == 0.0 { 1.0 } else { m }
    };
    ref_ring = ref_ring.max(1e-6);

    // np.uint8([[mean]]) truncates; then BGR2LAB
    let lab_of = |m: &[f64; 3]| rgb_to_lab(m[0] as u8, m[1] as u8, m[2] as u8);
    let labs: Vec<[f64; 3]> = flats.iter().map(|&i| lab_of(&out[i].core_mean_rgb)).collect();
    let majority = [
        median(&labs.iter().map(|l| l[0]).collect::<Vec<_>>()),
        median(&labs.iter().map(|l| l[1]).collect::<Vec<_>>()),
        median(&labs.iter().map(|l| l[2]).collect::<Vec<_>>()),
    ];
    for (k, &i) in flats.iter().enumerate() {
        let ratio = out[i].ring_grad / ref_ring;
        let l = labs[k];
        let dcol = ((l[0] - majority[0]).powi(2) + (l[1] - majority[1]).powi(2) + (l[2] - majority[2]).powi(2)).sqrt();
        let obstacle = ratio >= 0.45 || (ratio >= 0.30 && dcol > 25.0);
        out[i].ty = if obstacle { CellType::Obstacle } else { CellType::Empty };
    }
    out
}

use crate::board::{Board, EMPTY, OBSTACLE, UNKNOWN};
use anyhow::{bail, Result};
use image::{imageops::FilterType, DynamicImage, RgbaImage};
use std::path::Path;

/// Max translation (in pixels of the normalised cell) tolerated when comparing a cell to a
/// template. Real boards are never sliced perfectly on the tile edge.
const SHIFT: i32 = 3;
/// A non-flat cell this close (mean abs. RGB diff, 0..255) to the obstacle/empty sample is
/// still accepted as that sample.
const SPECIAL_MAX_DIFF: f32 = 30.0;

#[derive(Debug, Clone)]
pub struct Template {
    pub id: i32,
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Full resized RGB image, row-major, 3 values per pixel.
    pub rgb: Vec<f32>,
    /// Centre patch (outer 15 % removed) used for correlation.
    pub core: Vec<f32>,
    pub core_mean: f32,
    pub core_norm: f32, // sqrt(sum((x - mean)^2))
}

fn margin_of(w: usize, h: usize) -> usize {
    ((w.min(h) as f32) * 0.15).round() as usize
}

/// Copy a `(w-2m) x (h-2m)` window of an RGB buffer, shifted by (dx, dy).
fn core_patch(rgb: &[f32], w: usize, h: usize, m: usize, dx: i32, dy: i32) -> Vec<f32> {
    let cw = w - 2 * m;
    let ch = h - 2 * m;
    let mut out = Vec::with_capacity(cw * ch * 3);
    for y in 0..ch {
        let sy = (m as i32 + dy + y as i32).clamp(0, h as i32 - 1) as usize;
        for x in 0..cw {
            let sx = (m as i32 + dx + x as i32).clamp(0, w as i32 - 1) as usize;
            let i = (sy * w + sx) * 3;
            out.extend_from_slice(&rgb[i..i + 3]);
        }
    }
    out
}

fn mean_norm(v: &[f32]) -> (f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let ss: f32 = v.iter().map(|x| (x - mean) * (x - mean)).sum();
    (mean, ss.sqrt())
}

/// Colour-aware normalised cross-correlation: ONE scalar mean over all R,G,B values, so
/// the offsets between channels (the colour of the tile) stay part of the signal.
/// A yellow moon and a grey moon therefore no longer look alike.
fn ncc(a: &[f32], a_mean: f32, a_norm: f32, b: &[f32], b_mean: f32, b_norm: f32) -> f32 {
    if a_norm < 1e-4 || b_norm < 1e-4 {
        return -1.0;
    }
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot += (a[i] - a_mean) * (b[i] - b_mean);
    }
    dot / (a_norm * b_norm)
}

/// Mean absolute RGB difference between two equally sized buffers (0..255).
fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return f32::MAX;
    }
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

/// Average per-channel standard deviation of an RGB buffer.
fn colour_std(v: &[f32]) -> f32 {
    let n = v.len() / 3;
    if n == 0 {
        return 0.0;
    }
    let mut total = 0.0f32;
    for ch in 0..3 {
        let mean = (0..n).map(|i| v[i * 3 + ch]).sum::<f32>() / n as f32;
        let var = (0..n).map(|i| (v[i * 3 + ch] - mean).powi(2)).sum::<f32>() / n as f32;
        total += var.sqrt();
    }
    total / 3.0
}

fn to_rgb_vec(img: &DynamicImage, w: u32, h: u32) -> Vec<f32> {
    let resized = img.resize_exact(w, h, FilterType::Triangle).to_rgb8();
    resized.as_raw().iter().map(|&v| v as f32).collect()
}

impl Template {
    pub fn from_rgb(id: i32, name: &str, w: u32, h: u32, rgb: Vec<f32>) -> Self {
        let m = margin_of(w as usize, h as usize);
        let core = core_patch(&rgb, w as usize, h as usize, m, 0, 0);
        let (core_mean, core_norm) = mean_norm(&core);
        Self {
            id,
            name: name.to_string(),
            width: w,
            height: h,
            rgb,
            core,
            core_mean,
            core_norm,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatcherConfig {
    pub threshold: f32,
    /// A cell whose centre has an (average per-channel) colour standard deviation below this
    /// value is "flat": blank ground or an obstacle block (decided against the samples).
    /// 0..255 scale; studio-made profiles use 15.
    pub empty_variance_threshold: f32,
    pub cell_target_width: u32,
    pub cell_target_height: u32,
}

impl Default for MatcherConfig {
    fn default() -> Self {
        Self {
            threshold: 0.75,
            empty_variance_threshold: 15.0,
            cell_target_width: 32,
            cell_target_height: 32,
        }
    }
}

pub struct Matcher {
    pub config: MatcherConfig,
    pub templates: Vec<Template>,
}

impl Matcher {
    pub fn new(config: MatcherConfig) -> Self {
        Self {
            config,
            templates: Vec::new(),
        }
    }

    pub fn add_template(&mut self, id: i32, name: &str, img: &DynamicImage) {
        let (w, h) = (self.config.cell_target_width, self.config.cell_target_height);
        let rgb = to_rgb_vec(img, w, h);
        self.templates.push(Template::from_rgb(id, name, w, h, rgb));
    }

    /// Load template images from directory. Files named `1.png`, `2.png`, etc. or `obstacle.png`, `empty.png`
    pub fn load_templates_from_dir<P: AsRef<Path>>(&mut self, dir: P) -> Result<usize> {
        let dir = dir.as_ref();
        if !dir.exists() || !dir.is_dir() {
            bail!("Template directory {:?} does not exist", dir);
        }

        let mut loaded = 0;
        let entries = std::fs::read_dir(dir)?;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            if ext != "png" && ext != "jpg" && ext != "jpeg" && ext != "bmp" {
                continue;
            }

            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();

            let id = if stem == "empty" {
                EMPTY
            } else if stem == "obstacle" || stem == "block" {
                OBSTACLE
            } else if let Ok(parsed_id) = stem.parse::<i32>() {
                parsed_id
            } else {
                continue;
            };

            if let Ok(img) = image::open(&path) {
                self.add_template(id, &stem, &img);
                loaded += 1;
            }
        }

        Ok(loaded)
    }

    /// Nearest of the EMPTY / OBSTACLE samples by mean absolute colour difference.
    fn nearest_special(&self, cell_rgb: &[f32]) -> Option<(i32, f32)> {
        self.templates
            .iter()
            .filter(|t| t.id <= 0)
            .map(|t| (t.id, mean_abs_diff(cell_rgb, &t.rgb)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Match a single cropped cell image against the templates.
    /// Returns (id, confidence). `EMPTY` / `OBSTACLE` for blank ground / blocks,
    /// a positive tile id for a recognised tile, `UNKNOWN` when it is none of those.
    pub fn match_cell(&self, cell_img: &DynamicImage) -> (i32, f32) {
        if self.templates.is_empty() {
            return (EMPTY, 0.0);
        }

        let w = self.config.cell_target_width;
        let h = self.config.cell_target_height;
        let (wu, hu) = (w as usize, h as usize);
        let m = margin_of(wu, hu);
        let cell_rgb = to_rgb_vec(cell_img, w, h);

        // 1. Flat centre => blank ground or obstacle block.
        let core0 = core_patch(&cell_rgb, wu, hu, m, 0, 0);
        if colour_std(&core0) < self.config.empty_variance_threshold {
            return match self.nearest_special(&cell_rgb) {
                Some((id, diff)) => (id, (1.0 - diff / 255.0).max(0.0)),
                None => (EMPTY, 1.0),
            };
        }

        // 2. Tile templates: coarse pass without shift, then refine the best few with shifts.
        let (c_mean, c_norm) = mean_norm(&core0);
        let mut coarse: Vec<(usize, f32)> = self
            .templates
            .iter()
            .enumerate()
            .filter(|(_, t)| t.id > 0 && t.core_norm >= 1e-4)
            .map(|(i, t)| (i, ncc(&core0, c_mean, c_norm, &t.core, t.core_mean, t.core_norm)))
            .collect();
        coarse.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut best_id = UNKNOWN;
        let mut best = -1.0f32;
        for &(ti, base) in coarse.iter().take(4) {
            let t = &self.templates[ti];
            let mut score = base;
            for dy in -SHIFT..=SHIFT {
                for dx in -SHIFT..=SHIFT {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let patch = core_patch(&cell_rgb, wu, hu, m, dx, dy);
                    let (pm, pn) = mean_norm(&patch);
                    let s = ncc(&patch, pm, pn, &t.core, t.core_mean, t.core_norm);
                    if s > score {
                        score = s;
                    }
                }
            }
            if score > best {
                best = score;
                best_id = t.id;
            }
        }

        if best >= self.config.threshold {
            return (best_id, best);
        }

        // 3. Textured but looks like an obstacle / blank sample (e.g. a bevelled stone).
        if let Some((id, diff)) = self.nearest_special(&cell_rgb) {
            if diff < SPECIAL_MAX_DIFF {
                return (id, (1.0 - diff / 255.0).max(0.0));
            }
        }

        // Not blank, not a known tile: do NOT pretend it is empty (the solver would route
        // through a real tile). Report UNKNOWN.
        (UNKNOWN, best)
    }

    /// Converts a board image into a Board matrix by slicing into cells and matching
    pub fn match_board_image(
        &self,
        board_img: &RgbaImage,
        rows: usize,
        cols: usize,
    ) -> Result<Board> {
        let (w, h) = board_img.dimensions();
        if rows == 0 || cols == 0 {
            bail!("Invalid grid dimensions: rows={}, cols={}", rows, cols);
        }

        let cell_w_f = w as f64 / cols as f64;
        let cell_h_f = h as f64 / rows as f64;

        if cell_w_f < 1.0 || cell_h_f < 1.0 {
            bail!("Board dimensions too small for grid size {}x{}", rows, cols);
        }

        let mut board = Board::new(rows, cols);

        for r in 0..rows {
            for c in 0..cols {
                let x = (c as f64 * cell_w_f).round() as u32;
                let y = (r as f64 * cell_h_f).round() as u32;
                let next_x = ((c + 1) as f64 * cell_w_f).round() as u32;
                let next_y = ((r + 1) as f64 * cell_h_f).round() as u32;

                let cur_w = (next_x - x).min(w.saturating_sub(x));
                let cur_h = (next_y - y).min(h.saturating_sub(y));

                if cur_w == 0 || cur_h == 0 {
                    continue;
                }

                let sub_view = image::imageops::crop_imm(board_img, x, y, cur_w, cur_h);
                let dyn_cell = DynamicImage::ImageRgba8(sub_view.to_image());

                let (id, _conf) = self.match_cell(&dyn_cell);
                board.set(r as i32, c as i32, id);
            }
        }

        Ok(board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn test_template_exact_ncc_match() {
        let mut tmpl_img = GrayImage::new(16, 16);
        for x in 0..16 {
            for y in 0..16 {
                let val = if (x + y) % 2 == 0 { 200u8 } else { 50u8 };
                tmpl_img.put_pixel(x, y, Luma([val]));
            }
        }

        let mut matcher = Matcher::new(MatcherConfig {
            threshold: 0.8,
            cell_target_width: 16,
            cell_target_height: 16,
            ..Default::default()
        });

        matcher.add_template(42, "tile_42", &DynamicImage::ImageLuma8(tmpl_img.clone()));

        // Match against exact same image
        let (id, ncc) = matcher.match_cell(&DynamicImage::ImageLuma8(tmpl_img));
        assert_eq!(id, 42);
        assert!((ncc - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_empty_variance_detection() {
        // Flat gray image (empty cell)
        let flat_img = GrayImage::from_pixel(16, 16, Luma([128u8]));

        let mut matcher = Matcher::new(MatcherConfig {
            threshold: 0.8,
            empty_variance_threshold: 5.0,
            cell_target_width: 16,
            cell_target_height: 16,
        });

        // Add dummy template with pattern
        let mut pat_img = GrayImage::new(16, 16);
        for i in 0..16 {
            pat_img.put_pixel(i, i, Luma([255]));
        }
        matcher.add_template(1, "pattern", &DynamicImage::ImageLuma8(pat_img));

        let (id, _) = matcher.match_cell(&DynamicImage::ImageLuma8(flat_img));
        assert_eq!(id, EMPTY);
    }

    #[test]
    fn test_unrecognised_cell_is_unknown_not_empty() {
        let mut matcher = Matcher::new(MatcherConfig {
            threshold: 0.8,
            empty_variance_threshold: 5.0,
            cell_target_width: 16,
            cell_target_height: 16,
        });
        let mut diag = GrayImage::new(16, 16);
        for i in 0..16 {
            diag.put_pixel(i, i, Luma([255]));
        }
        matcher.add_template(1, "diag", &DynamicImage::ImageLuma8(diag));

        // Textured cell that looks nothing like the template (vertical stripes)
        let mut stripes = GrayImage::new(16, 16);
        for y in 0..16 {
            for x in (0..16).step_by(2) {
                stripes.put_pixel(x, y, Luma([255]));
            }
        }
        let (id, _) = matcher.match_cell(&DynamicImage::ImageLuma8(stripes));
        assert_eq!(id, UNKNOWN);
    }
}

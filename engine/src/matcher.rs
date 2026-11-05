use crate::board::{Board, EMPTY, OBSTACLE};
use anyhow::{bail, Result};
use image::{imageops::FilterType, DynamicImage, GrayImage, RgbaImage};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Template {
    pub id: i32,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub gray: Vec<f32>,
    pub mean: f32,
    pub std_norm: f32, // sqrt(sum((x - mean)^2))
}

impl Template {
    pub fn from_image(id: i32, name: &str, img: &GrayImage) -> Self {
        let (w, h) = img.dimensions();
        let total = (w * h) as f32;
        let mut gray = Vec::with_capacity((w * h) as usize);

        let mut sum = 0.0f32;
        for p in img.pixels() {
            let val = p.0[0] as f32;
            gray.push(val);
            sum += val;
        }

        let mean = if total > 0.0 { sum / total } else { 0.0 };

        let mut sum_sq_diff = 0.0f32;
        for val in &gray {
            let diff = val - mean;
            sum_sq_diff += diff * diff;
        }

        let std_norm = sum_sq_diff.sqrt();

        Self {
            id,
            name: name.to_string(),
            width: w,
            height: h,
            gray,
            mean,
            std_norm,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatcherConfig {
    pub threshold: f32,
    pub empty_variance_threshold: f32,
    pub cell_target_width: u32,
    pub cell_target_height: u32,
}

impl Default for MatcherConfig {
    fn default() -> Self {
        Self {
            threshold: 0.82,
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
        let resized = img.resize_exact(
            self.config.cell_target_width,
            self.config.cell_target_height,
            FilterType::Triangle,
        );
        let gray = resized.to_luma8();
        let tmpl = Template::from_image(id, name, &gray);
        self.templates.push(tmpl);
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

    /// Match a single cropped cell image against templates using NCC (Normalized Cross-Correlation)
    /// Returns (best_id, confidence_score)
    pub fn match_cell(&self, cell_img: &DynamicImage) -> (i32, f32) {
        if self.templates.is_empty() {
            return (EMPTY, 0.0);
        }

        let resized = cell_img.resize_exact(
            self.config.cell_target_width,
            self.config.cell_target_height,
            FilterType::Nearest,
        );
        let gray = resized.to_luma8();
        let (w, h) = gray.dimensions();
        let total = (w * h) as f32;

        let mut cell_data = Vec::with_capacity((w * h) as usize);
        let mut sum = 0.0f32;
        for p in gray.pixels() {
            let val = p.0[0] as f32;
            cell_data.push(val);
            sum += val;
        }

        let mean = if total > 0.0 { sum / total } else { 0.0 };

        let mut sum_sq_diff = 0.0f32;
        for val in &cell_data {
            let diff = val - mean;
            sum_sq_diff += diff * diff;
        }

        let variance = if total > 0.0 { sum_sq_diff / total } else { 0.0 };
        let std_norm = sum_sq_diff.sqrt();

        // Check if cell is essentially uniform / flat background
        if variance < self.config.empty_variance_threshold || std_norm < 1e-4 {
            return (EMPTY, 1.0);
        }

        let mut best_id = EMPTY;
        let mut best_ncc = -1.0f32;

        for tmpl in &self.templates {
            if tmpl.std_norm < 1e-4 {
                continue;
            }

            let mut dot = 0.0f32;
            for i in 0..cell_data.len() {
                dot += (cell_data[i] - mean) * (tmpl.gray[i] - tmpl.mean);
            }

            let ncc = dot / (std_norm * tmpl.std_norm);
            if ncc > best_ncc {
                best_ncc = ncc;
                best_id = tmpl.id;
            }
        }

        // If best NCC is below confidence threshold, treat as EMPTY
        if best_ncc < self.config.threshold {
            (EMPTY, best_ncc)
        } else {
            (best_id, best_ncc)
        }
    }

    /// Converts a board image into a Board matrix by slicing into cells and matching
    pub fn match_board_image(
        &self,
        board_img: &RgbaImage,
        rows: usize,
        cols: usize,
    ) -> Result<Board> {
        let (w, h) = board_img.dimensions();
        let cell_w = w / cols as u32;
        let cell_h = h / rows as u32;

        if cell_w == 0 || cell_h == 0 {
            bail!("Board dimensions too small for grid size {}x{}", rows, cols);
        }

        let mut board = Board::new(rows, cols);

        for r in 0..rows {
            for c in 0..cols {
                let x = c as u32 * cell_w;
                let y = r as u32 * cell_h;

                let sub_view = image::imageops::crop_imm(board_img, x, y, cell_w, cell_h);
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
}

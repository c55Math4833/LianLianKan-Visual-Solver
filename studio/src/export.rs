//! Writes templates (<id>.png), game_profile.json and the editable preview.html.
//! Output layout is identical to studio/exporter.py so the engine and the HTML editor
//! keep working unchanged.

use crate::calibrate::{GridInfo, Roi};
use crate::imgproc::{resize, Image};
use anyhow::Result;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const EDITOR_HTML: &str = include_str!("editor.html");

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for ch in data.chunks(3) {
        let b = [ch[0], *ch.get(1).unwrap_or(&0), *ch.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if ch.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if ch.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

pub fn png_bytes(img: &Image) -> Vec<u8> {
    let mut buf = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img.to_rgb())
        .write_to(&mut buf, image::ImageFormat::Png)
        .expect("png encode");
    buf.into_inner()
}

fn data_uri(img: &Image) -> String {
    format!("data:image/png;base64,{}", base64(&png_bytes(img)))
}

fn save_png(path: &Path, img: &Image) -> Result<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    std::fs::write(path, png_bytes(img))?;
    Ok(())
}

pub struct ExportCell {
    pub row: usize,
    pub col: usize,
    pub kind: &'static str, // "matchable" | "empty" | "obstacle"
    pub id: i64,
    pub image: Image,
}

pub struct ExportTemplate {
    pub id: usize,
    pub image: Image, // already 45x45
}

/// Path of `target` relative to the folder holding `profile` (best effort, forward slashes).
fn relative_template_dir(templates: &Path, profile: &Path) -> String {
    let abs = |p: &Path| std::fs::canonicalize(p).ok();
    let pdir = profile.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    if let (Some(t), Some(p)) = (abs(templates), abs(pdir)) {
        if let Ok(rel) = t.strip_prefix(&p) {
            let s = rel.to_string_lossy().replace('\\', "/");
            return if s.is_empty() { ".".into() } else { s };
        }
    }
    templates.to_string_lossy().replace('\\', "/")
}

#[allow(clippy::too_many_arguments)]
pub fn export_all(
    roi: Roi,
    grid: &GridInfo,
    cells: &[ExportCell],
    templates: &[ExportTemplate],
    templates_dir: &Path,
    profile_path: &Path,
    preview_path: &Path,
) -> Result<Value> {
    std::fs::create_dir_all(templates_dir)?;
    let norm = |p: PathBuf| p.to_string_lossy().replace('\\', "/");

    let sample_of = |kind: &str, gray: u8| -> Image {
        match cells.iter().find(|c| c.kind == kind) {
            Some(c) => resize(&c.image, 45, 45),
            None => Image { w: 45, h: 45, data: vec![gray; 45 * 45 * 3] },
        }
    };
    let empty_sample = sample_of("empty", 180);
    let obstacle_sample = sample_of("obstacle", 120);

    let empty_file = norm(templates_dir.join("empty.png"));
    let obstacle_file = norm(templates_dir.join("obstacle.png"));
    save_png(Path::new(&empty_file), &empty_sample)?;
    save_png(Path::new(&obstacle_file), &obstacle_sample)?;

    let mut tile_mapping = vec![
        json!({"id": 0, "type": "empty", "file": empty_file}),
        json!({"id": -1, "type": "obstacle", "file": obstacle_file}),
    ];
    let mut tpl_json = Vec::new();
    for t in templates {
        let file = norm(templates_dir.join(format!("{}.png", t.id)));
        save_png(Path::new(&file), &t.image)?;
        tile_mapping.push(json!({"id": t.id, "type": "matchable", "file": file}));
        tpl_json.push(json!({"id": t.id, "file": file, "img": data_uri(&t.image)}));
    }

    let r2 = |v: f64| (v * 100.0).round() / 100.0;
    let profile = json!({
        "name": "LianLianKan Studio Profile",
        "version": "1.0",
        "board": {
            "rows": grid.rows,
            "cols": grid.cols,
            "roi": {"x": roi.x, "y": roi.y, "width": roi.w, "height": roi.h},
            "cell_size": {"w": r2(grid.cell_w), "h": r2(grid.cell_h)},
            "offset": {"x": roi.x, "y": roi.y}
        },
        "matcher": {
            "threshold": 0.75,
            "empty_variance_threshold": 15.0,
            "template_dir": relative_template_dir(templates_dir, profile_path)
        },
        "rules": {"allow_outer_path": true},
        "timing": {"click_interval_ms": 35, "animation_delay_ms": 160, "verify_timeout_ms": 600},
        "tile_mapping": tile_mapping
    });
    std::fs::write(profile_path, serde_json::to_string_pretty(&profile)?)?;

    write_editor_html(&profile, grid.rows, grid.cols, false, cells, tpl_json, preview_path)?;
    Ok(profile)
}

/// Writes the editable preview page. `augment` = page for an "add objects" run (shows only the
/// objects of this screenshot, no parity warnings).
pub fn write_editor_html(
    profile: &Value,
    rows: usize,
    cols: usize,
    augment: bool,
    cells: &[ExportCell],
    templates: Vec<Value>,
    preview_path: &Path,
) -> Result<()> {
    let cell_json: Vec<Value> = cells
        .iter()
        .map(|c| json!({"r": c.row, "c": c.col, "type": c.kind, "id": c.id, "img": data_uri(&c.image)}))
        .collect();
    let data = json!({
        "rows": rows,
        "cols": cols,
        "augment": augment,
        "profile": profile,
        "cells": cell_json,
        "templates": templates,
    });
    write_page(&data, rows.max(cols), preview_path)
}

/// Settings-only page (no board / classes): edits timing, matcher and rules of an existing profile.
pub fn write_settings_html(profile: &Value, templates: Vec<Value>, merge: Value, obstacle: Value, generic_obstacle: String, preview_path: &Path) -> Result<()> {
    let n = |k: &str| profile["board"][k].as_u64().unwrap_or(0) as usize;
    let data = json!({
        "rows": n("rows"),
        "cols": n("cols"),
        "augment": false,
        "settingsOnly": true,
        "profile": profile,
        "cells": [],
        "templates": templates,
        "genericObstacle": generic_obstacle,
        "initState": {"merge": merge, "obstacle": obstacle},
    });
    write_page(&data, 0, preview_path)
}

fn write_page(data: &Value, max_dim: usize, preview_path: &Path) -> Result<()> {
    let payload = serde_json::to_string(data)?.replace("</", "<\\/");
    let cell_px = if max_dim <= 20 { 48 } else { 32 };
    let html = EDITOR_HTML.replace("__DATA__", &payload).replace("__CELL__", &cell_px.to_string());
    std::fs::write(preview_path, html)?;
    Ok(())
}

pub fn image_data_uri(img: &Image) -> String {
    data_uri(img)
}

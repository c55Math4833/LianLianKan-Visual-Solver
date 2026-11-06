//! "Add objects from another screenshot": later levels bring new objects. The board of the
//! new screenshot is compared with every existing template; only objects that match nothing
//! are clustered and appended with new ids (max existing id + 1, ...). Existing ids are never
//! renumbered. Port of studio/augment.py.

use crate::calibrate::{estimate_adaptive_grid, slice_cells, Roi};
use crate::cluster::{agglomerative, prepare, similarity, Linkage, Prepared};
use crate::export::{image_data_uri, write_editor_html, ExportCell};
use crate::features::{classify_cells, CellType};
use crate::imgproc::{resize, resize_area, Image};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Cell vs. template similarity needed to call it "already known".
const KNOWN_SIM: f64 = 0.80;
/// Unmatched cells closer than this (1 - similarity) are the same new object.
const NEW_CLUSTER_DIST: f64 = 0.25;

pub struct Summary {
    pub rows: usize,
    pub cols: usize,
    pub tiles: usize,
    pub known: usize,
    pub new_ids: Vec<i64>,
    pub new_counts: Vec<usize>,
}

pub struct LibEntry {
    pub id: i64,
    pub kind: String, // "matchable" | "obstacle"
    images: Vec<Image>,
    prepared: Vec<Prepared>,
    paths: Vec<PathBuf>,
    file: String,
}

/// Templates are stored at 45x45 while cells are e.g. 29x29: bring both to a common size
/// first, otherwise the cropped cores shift by a pixel and small icons stop correlating.
fn prep(img: &Image) -> Prepared {
    prepare(&resize_area(img, 40, 40))
}

fn load_png(path: &Path) -> Option<Image> {
    image::open(path).ok().map(|i| Image::from_rgb(&i.to_rgb8()))
}

fn resolve(file: &str, profile_dir: &Path, templates_dir: &Path) -> Option<PathBuf> {
    let base = Path::new(file).file_name()?.to_owned();
    [
        PathBuf::from(file),
        profile_dir.join(file),
        templates_dir.join(&base),
        profile_dir.join(templates_dir).join(&base),
    ]
    .into_iter()
    .find(|p| p.is_file())
}

/// Read every non-generic template of the profile (all files of merged classes) into memory.
/// Returns (entries, highest id in use).
pub fn load_library(profile: &Value, profile_dir: &Path, templates_dir: &Path) -> Result<(Vec<LibEntry>, i64)> {
    let mut max_id = 0i64;
    let mut library: Vec<LibEntry> = Vec::new();
    for item in profile["tile_mapping"].as_array().cloned().unwrap_or_default() {
        let id = item["id"].as_i64().unwrap_or(0);
        max_id = max_id.max(id);
        if let Some(m) = item["merged_from"].as_array() {
            for v in m.iter().filter_map(|v| v.as_i64()) {
                max_id = max_id.max(v);
            }
        }
        // id <= 0 are the generic empty / obstacle samples (flat images): never compare tiles to them.
        let kind = item["type"].as_str().unwrap_or("");
        if id <= 0 || (kind != "matchable" && kind != "obstacle") {
            continue;
        }
        let files: Vec<String> = match item["files"].as_array() {
            Some(a) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
            None => item["file"].as_str().map(|s| vec![s.to_string()]).unwrap_or_default(),
        };
        let mut images: Vec<Image> = Vec::new();
        let mut paths: Vec<PathBuf> = Vec::new();
        for f in &files {
            if let Some(p) = resolve(f, profile_dir, templates_dir) {
                if let Some(img) = load_png(&p) {
                    images.push(img);
                    paths.push(p);
                }
            }
        }
        if images.is_empty() {
            continue;
        }
        let prepared = images.iter().map(prep).collect();
        library.push(LibEntry {
            id,
            kind: kind.to_string(),
            images,
            prepared,
            paths,
            file: item["file"].as_str().unwrap_or("").to_string(),
        });
    }
    if library.is_empty() {
        bail!("找不到既有的磚塊模板：請先建立第一份 game_profile.json 與 templates/。");
    }
    Ok((library, max_id))
}


pub fn augment_from_board(board: &Image, profile_path: &Path, templates_dir: &Path, preview_path: &Path) -> Result<Summary> {
    let mut profile: Value = serde_json::from_str(
        &std::fs::read_to_string(profile_path).with_context(|| format!("cannot read {profile_path:?}"))?,
    )?;
    let profile_dir = std::fs::canonicalize(profile_path)
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let mut templates_dir = templates_dir.to_path_buf();
    if !templates_dir.is_dir() {
        if let Some(alt) = profile["matcher"]["template_dir"].as_str() {
            templates_dir = profile_dir.join(alt);
        }
    }

    // ---- existing library
    let (library, max_id) = load_library(&profile, &profile_dir, &templates_dir)?;

    // ---- slice + classify the new board
    let grid = estimate_adaptive_grid(board);
    let roi = Roi { x: 0, y: 0, w: board.w as i64, h: board.h as i64 };
    let cells = classify_cells(slice_cells(board, roi, grid.rows, grid.cols));
    let tile_idx: Vec<usize> = (0..cells.len()).filter(|&i| cells[i].ty == CellType::Matchable).collect();

    // ---- compare with the library
    let mut kinds: Vec<&'static str> = cells
        .iter()
        .map(|c| match c.ty {
            CellType::Obstacle => "obstacle",
            CellType::Matchable => "matchable",
            _ => "empty",
        })
        .collect();
    let mut assigned: Vec<i64> = vec![0; cells.len()];
    let mut unmatched: Vec<usize> = Vec::new();
    for &i in &tile_idx {
        let p = prep(&cells[i].cell.image);
        let (mut best_id, mut best_sim, mut best_kind) = (0i64, -1.0f64, "");
        for e in &library {
            for lp in &e.prepared {
                let s = similarity(&p, lp);
                if s > best_sim {
                    best_sim = s;
                    best_id = e.id;
                    best_kind = &e.kind;
                }
            }
        }
        if best_sim >= KNOWN_SIM {
            if best_kind == "obstacle" {
                kinds[i] = "obstacle";
                assigned[i] = -1;
            } else {
                assigned[i] = best_id;
            }
        } else {
            unmatched.push(i);
        }
    }

    // ---- cluster the unknown cells -> new objects
    struct Group {
        members: Vec<usize>,
        medoid: usize,
    }
    let mut groups: Vec<Group> = Vec::new();
    if !unmatched.is_empty() {
        let prepared: Vec<Prepared> = unmatched.iter().map(|&i| prep(&cells[i].cell.image)).collect();
        let n = unmatched.len();
        let mut dist = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = (1.0 - similarity(&prepared[i], &prepared[j])).max(0.0);
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }
        let labels = agglomerative(&dist, NEW_CLUSTER_DIST, Linkage::Average);
        let mut uniq = labels.clone();
        uniq.sort();
        uniq.dedup();
        for l in uniq {
            let idx: Vec<usize> = (0..n).filter(|&i| labels[i] == l).collect();
            let mut best = (f64::INFINITY, idx[0]);
            for &i in &idx {
                let s: f64 = idx.iter().map(|&j| dist[i][j]).sum();
                if s < best.0 {
                    best = (s, i);
                }
            }
            groups.push(Group { members: idx.iter().map(|&k| unmatched[k]).collect(), medoid: unmatched[best.1] });
        }
        groups.sort_by_key(|g| std::cmp::Reverse(g.members.len())); // stable: most frequent first
    }

    // ---- save new templates + update the profile (existing ids untouched)
    std::fs::create_dir_all(&templates_dir)?;
    let tdir_str = templates_dir.to_string_lossy().replace('\\', "/");
    let mut new_ids = Vec::new();
    let mut new_counts = Vec::new();
    let mut new_templates: Vec<(i64, String, Image)> = Vec::new();
    let mut next_id = max_id + 1;
    for g in &groups {
        let nid = next_id;
        next_id += 1;
        let tpl = resize(&cells[g.medoid].cell.image, 45, 45);
        let file = format!("{}/{}.png", tdir_str.trim_end_matches('/'), nid);
        std::fs::write(&file, crate::export::png_bytes(&tpl))?;
        profile["tile_mapping"].as_array_mut().unwrap().push(json!({"id": nid, "type": "matchable", "file": file}));
        for &m in &g.members {
            assigned[m] = nid;
        }
        new_ids.push(nid);
        new_counts.push(g.members.len());
        new_templates.push((nid, file, tpl));
    }
    std::fs::write(profile_path, serde_json::to_string_pretty(&profile)?)?;

    // ---- editable preview for THIS screenshot (all known ids, counts from this board)
    let mut tpl_json: Vec<Value> = Vec::new();
    for item in profile["tile_mapping"].as_array().unwrap() {
        let id = item["id"].as_i64().unwrap_or(0);
        if item["type"].as_str() != Some("matchable") || id <= 0 {
            continue;
        }
        let img = if let Some(e) = library.iter().find(|e| e.id == id) {
            resize(&e.images[0], 45, 45)
        } else if let Some((_, _, t)) = new_templates.iter().find(|(nid, _, _)| *nid == id) {
            t.clone()
        } else {
            continue;
        };
        let _ = library.iter().find(|e| e.id == id).map(|e| &e.file);
        tpl_json.push(json!({"id": id, "file": item["file"], "img": image_data_uri(&img)}));
    }
    let export_cells: Vec<ExportCell> = cells
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let (kind, id) = match kinds[i] {
                "matchable" => ("matchable", assigned[i]),
                "obstacle" => ("obstacle", -1),
                _ => ("empty", 0),
            };
            ExportCell { row: c.cell.row, col: c.cell.col, kind, id, image: c.cell.image.clone() }
        })
        .collect();
    write_editor_html(&profile, grid.rows, grid.cols, true, &export_cells, tpl_json, preview_path)?;

    Ok(Summary {
        rows: grid.rows,
        cols: grid.cols,
        tiles: tile_idx.len(),
        known: tile_idx.len() - unmatched.len(),
        new_ids,
        new_counts,
    })
}


// ---------------------------------------------------------------------------------------------
// Re-aiming a profile on a re-scaled game window
// ---------------------------------------------------------------------------------------------

/// Best similarity of one cell image to any template of the library.
fn best_match(img: &Image, library: &[LibEntry]) -> (i64, f64) {
    let p = prep(img);
    let (mut id, mut best) = (0i64, -1.0f64);
    for e in library {
        for lp in &e.prepared {
            let s = similarity(&p, lp);
            if s > best {
                best = s;
                id = e.id;
            }
        }
    }
    (id, best)
}

fn roi_fits(screen: &Image, r: Roi) -> bool {
    r.x >= 0 && r.y >= 0 && r.w > 8 && r.h > 8 && r.x + r.w <= screen.w as i64 && r.y + r.h <= screen.h as i64
}

/// How well a candidate frame explains the screen: sum of (similarity - 0.5) of every tile cell
/// against the known templates. A misaligned frame cuts tiles in half and scores low.
fn frame_score(screen: &Image, roi: Roi, rows: usize, cols: usize, library: &[LibEntry]) -> f64 {
    if !roi_fits(screen, roi) {
        return f64::NEG_INFINITY;
    }
    let cells = classify_cells(slice_cells(screen, roi, rows, cols));
    cells
        .iter()
        .filter(|c| c.ty == CellType::Matchable)
        .map(|c| (best_match(&c.cell.image, library).1 - 0.5).max(0.0))
        .sum()
}

fn best_of(screen: &Image, cands: &[Roi], rows: usize, cols: usize, library: &[LibEntry]) -> (Roi, f64) {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(cands.len().max(1));
    let chunk = cands.len().div_ceil(n.max(1)).max(1);
    let mut best = (cands[0], f64::NEG_INFINITY);
    std::thread::scope(|sc| {
        let hs: Vec<_> = cands
            .chunks(chunk)
            .map(|ch| {
                sc.spawn(move || {
                    let mut b = (ch[0], f64::NEG_INFINITY);
                    for &r in ch {
                        let s = frame_score(screen, r, rows, cols, library);
                        if s > b.1 {
                            b = (r, s);
                        }
                    }
                    b
                })
            })
            .collect();
        for h in hs {
            let b = h.join().unwrap();
            if b.1 > best.1 {
                best = b;
            }
        }
    });
    best
}

/// Polish a fitted frame by what actually matters: how many tiles match the known templates.
/// Searches small position / size changes around `start` (coarse to fine).
pub fn refine_frame(screen: &Image, start: Roi, rows: usize, cols: usize, library: &[LibEntry]) -> Roi {
    let pitch = (start.w as f64 / cols as f64).min(start.h as f64 / rows as f64);
    let r = (pitch * 0.12).round().max(2.0) as i64;
    let st = ((pitch * 0.04).round() as i64).max(1);
    let mut cur = start;

    // 1) position
    let mut c = Vec::new();
    let mut dy = -r;
    while dy <= r {
        let mut dx = -r;
        while dx <= r {
            c.push(Roi { x: start.x + dx, y: start.y + dy, ..start });
            dx += st;
        }
        dy += st;
    }
    cur = best_of(screen, &c, rows, cols, library).0.max_or(cur);

    // 2) size (cell pitch), keeping the frame centre
    let mut c = Vec::new();
    for dw in -2..=2i64 {
        for dh in -2..=2i64 {
            let (w, h) = (cur.w + dw * st * 2, cur.h + dh * st * 2);
            c.push(Roi { x: cur.x - (w - cur.w) / 2, y: cur.y - (h - cur.h) / 2, w, h });
        }
    }
    cur = best_of(screen, &c, rows, cols, library).0.max_or(cur);

    // 3) final +-1 px
    let mut c = Vec::new();
    for dy in -1..=1i64 {
        for dx in -1..=1i64 {
            c.push(Roi { x: cur.x + dx, y: cur.y + dy, ..cur });
        }
    }
    best_of(screen, &c, rows, cols, library).0.max_or(cur)
}

trait MaxOr {
    fn max_or(self, fallback: Roi) -> Roi;
}
impl MaxOr for Roi {
    fn max_or(self, fallback: Roi) -> Roi {
        if self.w > 0 { self } else { fallback }
    }
}

pub struct Resample {
    pub tiles: usize,
    pub known: usize,
    pub updated_ids: Vec<i64>,
    pub avg_sim: f64,
}

/// Replace the reference image of every known object with a sample taken from the LIVE board, so
/// the engine compares like with like (same scale, same anti-aliasing). Only cells that already
/// match an existing object (similarity >= KNOWN_SIM) are used; nothing new is invented.
pub fn resample_templates(board: &Image, rows: usize, cols: usize, library: &[LibEntry], templates_dir: &Path) -> Result<Resample> {
    let roi = Roi { x: 0, y: 0, w: board.w as i64, h: board.h as i64 };
    let cells = classify_cells(slice_cells(board, roi, rows, cols));
    let mut by_id: std::collections::BTreeMap<i64, Vec<usize>> = std::collections::BTreeMap::new();
    let (mut tiles, mut known, mut sum) = (0usize, 0usize, 0.0f64);
    for (i, c) in cells.iter().enumerate() {
        if c.ty != CellType::Matchable {
            continue;
        }
        tiles += 1;
        let (id, s) = best_match(&c.cell.image, library);
        sum += s.max(0.0);
        if s >= KNOWN_SIM {
            known += 1;
            by_id.entry(id).or_default().push(i);
        }
    }
    let mut updated = Vec::new();
    for (id, members) in by_id {
        let Some(entry) = library.iter().find(|e| e.id == id) else { continue };
        if entry.kind != "matchable" {
            continue;
        }
        // medoid: the member most similar to the others (robust against a hover highlight etc.)
        let prepared: Vec<Prepared> = members.iter().map(|&i| prep(&cells[i].cell.image)).collect();
        let mut best = (f64::NEG_INFINITY, 0usize);
        for a in 0..members.len() {
            let s: f64 = (0..members.len()).filter(|&b| b != a).map(|b| similarity(&prepared[a], &prepared[b])).sum();
            if s > best.0 {
                best = (s, a);
            }
        }
        let img = &cells[members[best.1]].cell.image;
        let target = entry.paths.first().cloned().unwrap_or_else(|| templates_dir.join(format!("{id}.png")));
        std::fs::write(&target, crate::export::png_bytes(img))?;
        updated.push(id);
    }
    Ok(Resample { tiles, known, updated_ids: updated, avg_sim: if tiles > 0 { sum / tiles as f64 } else { 0.0 } })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs LLK_BASE (dir with game_profile.json + templates), LLK_IMG (scaled screenshot) and
    /// LLK_SCALE; skipped otherwise.
    #[test]
    fn reaim_on_scaled_screenshot() {
        let (Ok(base), Ok(img), Ok(sc)) = (std::env::var("LLK_BASE"), std::env::var("LLK_IMG"), std::env::var("LLK_SCALE")) else { return };
        let f: f64 = sc.parse().unwrap();
        let base = PathBuf::from(base);
        let prof: Value = serde_json::from_str(&std::fs::read_to_string(base.join("game_profile.json")).unwrap()).unwrap();
        let (rows, cols) = (prof["board"]["rows"].as_u64().unwrap() as usize, prof["board"]["cols"].as_u64().unwrap() as usize);
        let g = |k: &str| prof["board"]["roi"][k].as_f64().unwrap();
        let exp = Roi { x: (g("x") * f) as i64, y: (g("y") * f) as i64, w: (g("width") * f) as i64, h: (g("height") * f) as i64 };
        let screen = Image::from_rgb(&image::open(img).unwrap().to_rgb8());
        let (lib, _) = load_library(&prof, &base, &base.join("templates")).unwrap();
        let cell = exp.w as f64 / cols as f64;
        let rough = Roi { x: exp.x - (0.4 * cell) as i64, y: exp.y + (0.3 * cell) as i64, w: exp.w + (0.8 * cell) as i64, h: exp.h + (0.7 * cell) as i64 };
        let fit = crate::calibrate::fit_known_grid(&screen, rough, rows, cols);
        let fin = refine_frame(&screen, fit.roi, rows, cols, &lib);
        println!("expected {:?}\nfit      {:?}\nrefined  {:?}", exp, fit.roi, fin);
        let board = screen.crop(fin.x as usize, fin.y as usize, fin.w as usize, fin.h as usize);
        let rs = resample_templates(&board, rows, cols, &lib, &base.join("templates")).unwrap();
        println!("tiles {} known {} avg {:.3} updated {:?}", rs.tiles, rs.known, rs.avg_sim, rs.updated_ids);
        let tol = (cell * 0.12) as i64 + 2;
        assert!((fin.x - exp.x).abs() <= tol && (fin.y - exp.y).abs() <= tol);
        assert!(rs.known as f64 >= 0.9 * rs.tiles as f64);
    }
}

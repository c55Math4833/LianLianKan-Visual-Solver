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

struct LibEntry {
    id: i64,
    kind: String, // "matchable" | "obstacle"
    images: Vec<Image>,
    prepared: Vec<Prepared>,
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
        let images: Vec<Image> =
            files.iter().filter_map(|f| resolve(f, &profile_dir, &templates_dir)).filter_map(|p| load_png(&p)).collect();
        if images.is_empty() {
            continue;
        }
        let prepared = images.iter().map(prep).collect();
        library.push(LibEntry {
            id,
            kind: kind.to_string(),
            images,
            prepared,
            file: item["file"].as_str().unwrap_or("").to_string(),
        });
    }
    if library.is_empty() {
        bail!("找不到既有的磚塊模板：請先建立第一份 game_profile.json 與 templates/。");
    }

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

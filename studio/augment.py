"""
LianLianKan Studio - Add objects from another screenshot.

Later levels usually introduce objects that were not on the first board.  Instead of
rebuilding everything, this module takes an already calibrated board image plus an
existing game_profile.json / templates folder and:

  1. slices the new board and separates empty / obstacle / tile cells,
  2. compares every tile cell with ALL existing templates,
  3. clusters the cells that match nothing (= new objects) and appends them as new
     template ids (max existing id + 1, ...) - existing ids are never renumbered,
  4. writes the new <id>.png files, updates game_profile.json and regenerates the
     editable preview.html for the new screenshot (merge / obstacle editing still works).
"""

import os
import json

import cv2
import numpy as np

from grid_calibrator import (
    load_image,
    estimate_adaptive_grid,
    slice_cells,
)
from feature_extractor import extract_features, classify_cells
from clusterer import compute_tile_similarity as _tile_sim, agglomerative_clustering_precomputed
from exporter import save_image_unicode, generate_html_report

# Cell vs. template similarity needed to call it "already known" (0..1, same metric the
# clusterer uses).  Cells of the SAME object score well above this, different objects below.
KNOWN_SIM = 0.80
# Unmatched cells closer than this (1 - similarity) are the same new object.
NEW_CLUSTER_DIST = 0.25


def compute_tile_similarity(a, b):
    """
    Same metric as the clusterer, but both images are first brought to one common size.
    Templates are stored at 45x45 while board cells are e.g. 29x29: comparing them at their
    native sizes shifts the cropped core by a pixel or two, which is enough to wreck the
    correlation of small icons (own-class similarity dropped to ~0.3).
    """
    a = cv2.resize(a, (40, 40), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (40, 40), interpolation=cv2.INTER_AREA)
    return _tile_sim(a, b)


def _resolve_template_path(entry_file, profile_dir, templates_dir):
    """Find a template file named in tile_mapping (paths may be relative to anything)."""
    cands = [
        entry_file,
        os.path.join(profile_dir, entry_file),
        os.path.join(templates_dir, os.path.basename(entry_file)),
        os.path.join(profile_dir, templates_dir, os.path.basename(entry_file)),
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def load_existing_library(profile_path, templates_dir):
    """
    Returns (profile, library) where library is a list of
    {"id", "type", "images": [bgr, ...]} for every matchable / obstacle class, plus
    the highest id ever used (including ids absorbed by a merge).
    """
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    profile_dir = os.path.dirname(os.path.abspath(profile_path))
    if not os.path.isdir(templates_dir):
        alt = profile.get("matcher", {}).get("template_dir")
        if alt:
            templates_dir = os.path.join(profile_dir, alt)

    library = []
    max_id = 0
    for item in profile.get("tile_mapping", []):
        ids = [item.get("id", 0)] + list(item.get("merged_from", []))
        max_id = max([max_id] + [i for i in ids if isinstance(i, int)])
        # id <= 0 are the generic empty / obstacle samples (flat images) - never compare tiles to them.
        # Obstacle *classes* marked in the HTML editor keep their positive id.
        if item.get("type") not in ("matchable", "obstacle") or item.get("id", 0) <= 0:
            continue
        files = item.get("files") or [item.get("file")]
        imgs = []
        for fpath in files:
            real = _resolve_template_path(fpath, profile_dir, templates_dir)
            if real:
                im = load_image(real)
                if im is not None:
                    imgs.append(im)
        if imgs:
            library.append({"id": item["id"], "type": item["type"], "images": imgs})
    return profile, library, templates_dir, max_id


def augment_from_board(
    board_img,
    profile_path="game_profile.json",
    templates_dir="templates",
    preview_path="preview.html",
    known_sim=KNOWN_SIM,
    new_cluster_dist=NEW_CLUSTER_DIST,
):
    """
    board_img: BGR image that contains exactly the board (already ROI-selected/snapped).
    Returns a summary dict.
    """
    profile, library, templates_dir, max_id = load_existing_library(profile_path, templates_dir)
    if not library:
        raise RuntimeError(
            "找不到既有的磚塊模板：請先用模式 1 或 2 建立 game_profile.json 與 templates/。")

    h, w = board_img.shape[:2]
    grid = estimate_adaptive_grid(board_img)
    rows, cols = grid["rows"], grid["cols"]
    roi = {"x": 0, "y": 0, "w": w, "h": h}
    cells = slice_cells(board_img, roi, rows, cols)
    for c in cells:
        c["features"] = extract_features(c["image"])
        c["tile_type"] = c["features"].get("type", "matchable")
    classify_cells(cells)

    tile_cells = [c for c in cells if c["tile_type"] == "matchable"]
    empty_cells = [c for c in cells if c["tile_type"] == "empty"]
    obstacle_cells = [c for c in cells if c["tile_type"] == "obstacle"]

    # 1. compare with the existing library
    assigned = {}          # (row, col) -> existing id
    unmatched = []
    for c in tile_cells:
        best_id, best_sim, best_type = None, -1.0, None
        for entry in library:
            for tpl in entry["images"]:
                s = compute_tile_similarity(c["image"], tpl)
                if s > best_sim:
                    best_id, best_sim, best_type = entry["id"], s, entry["type"]
        c["match_sim"] = best_sim
        if best_sim >= known_sim:
            if best_type == "obstacle":
                c["tile_type"] = "obstacle"
                obstacle_cells.append(c)
            else:
                assigned[(c["row"], c["col"])] = best_id
        else:
            unmatched.append(c)

    # 2. cluster the unknown cells -> new objects
    new_groups = []
    if unmatched:
        n = len(unmatched)
        sim = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                s = compute_tile_similarity(unmatched[i]["image"], unmatched[j]["image"])
                sim[i, j] = sim[j, i] = s
        dist = np.maximum(0.0, 1.0 - sim)
        labels = agglomerative_clustering_precomputed(dist, new_cluster_dist, linkage="average")
        for lbl in sorted(set(labels.tolist())):
            idx = [i for i in range(n) if labels[i] == lbl]
            sub = dist[np.ix_(idx, idx)]
            medoid = unmatched[idx[int(np.argmin(sub.sum(axis=1)))]]
            new_groups.append({"members": [unmatched[i] for i in idx], "medoid": medoid})
        # most frequent first, so ids are stable and meaningful
        new_groups.sort(key=lambda g: -len(g["members"]))

    # 3. save new templates + update profile
    os.makedirs(templates_dir, exist_ok=True)
    tile_mapping = profile.setdefault("tile_mapping", [])
    new_ids = []
    next_id = max_id + 1
    for g in new_groups:
        nid = next_id
        next_id += 1
        tpl = cv2.resize(g["medoid"]["image"], (45, 45))
        fpath = os.path.join(templates_dir, f"{nid}.png").replace("\\", "/")
        save_image_unicode(fpath, tpl)
        tile_mapping.append({"id": nid, "type": "matchable", "file": fpath})
        g["id"] = nid
        g["template"] = tpl
        new_ids.append(nid)
        for m in g["members"]:
            assigned[(m["row"], m["col"])] = nid

    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    # 4. editable preview for this screenshot (all known ids, coordinates from this board)
    coords = {}
    for (r, c), cid in assigned.items():
        coords.setdefault(cid, []).append((r, c))
    tpl_images = {}
    for entry in library:
        tpl_images[entry["id"]] = cv2.resize(entry["images"][0], (45, 45))
    for g in new_groups:
        tpl_images[g["id"]] = g["template"]

    templates_meta = []
    for item in tile_mapping:
        cid = item.get("id", 0)
        if item.get("type") != "matchable" or cid not in tpl_images:
            continue
        templates_meta.append({
            "id": cid,
            "file": item.get("file"),
            "color_family": "auto",
            "count": len(coords.get(cid, [])),
            "coords": sorted(coords.get(cid, [])),
            "image": tpl_images[cid],
        })

    def _sample(name, fallback_cells, gray):
        path = os.path.join(templates_dir, name)
        if os.path.isfile(path):
            im = load_image(path)
            if im is not None:
                return cv2.resize(im, (45, 45))
        if fallback_cells:
            return cv2.resize(fallback_cells[0]["image"], (45, 45))
        return np.full((45, 45, 3), gray, dtype=np.uint8)

    all_cells = {(c["row"], c["col"]): c for c in cells}
    generate_html_report(
        profile=profile,
        empty_sample=_sample("empty.png", empty_cells, 180),
        obstacle_sample=_sample("obstacle.png", obstacle_cells, 120),
        templates_meta=templates_meta,
        empty_cells=empty_cells,
        obstacle_cells=obstacle_cells,
        all_cells_matrix=all_cells,
        output_html=preview_path,
        rows=rows,
        cols=cols,
        augment=True,
    )

    return {
        "rows": rows,
        "cols": cols,
        "tiles": len(tile_cells),
        "known": len(tile_cells) - len(unmatched),
        "new_ids": new_ids,
        "new_counts": [len(g["members"]) for g in new_groups],
        "empty": len(empty_cells),
        "obstacle": len(obstacle_cells),
    }

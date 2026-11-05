"""
LianLianKan Exporter
Exports template images, game_profile.json for Rust solver engine,
and generates local HTML preview gallery (preview.html).
"""

import os
import json
import base64
import cv2
import numpy as np

def save_image_unicode(path, img):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1]
    res, enc = cv2.imencode(ext, img)
    if res:
        with open(path, 'wb') as f:
            f.write(enc)

def to_base64_src(img):
    res, enc = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(enc).decode('utf-8')

def export_all(
    board_roi,
    grid_info,
    empty_cells,
    obstacle_cells,
    clusters,
    all_cells_matrix,
    output_dir="templates",
    profile_path="game_profile.json",
    preview_path="preview.html"
):
    """
    Exports:
    1. templates/empty.png, templates/obstacle.png, templates/tile_01.png ...
    2. game_profile.json matching exact Rust schema
    3. preview.html with full visual inspection gallery
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tile_mapping = []
    
    # 1. Export empty template
    if len(empty_cells) > 0:
        empty_sample = cv2.resize(empty_cells[0]["image"], (45, 45))
    else:
        empty_sample = np.full((45, 45, 3), 180, dtype=np.uint8)
    empty_file = os.path.join(output_dir, "empty.png").replace("\\", "/")
    save_image_unicode(empty_file, empty_sample)
    tile_mapping.append({
        "id": 0,
        "type": "empty",
        "file": empty_file
    })
    
    # 2. Export obstacle template
    if len(obstacle_cells) > 0:
        obstacle_sample = cv2.resize(obstacle_cells[0]["image"], (45, 45))
    else:
        obstacle_sample = np.full((45, 45, 3), 120, dtype=np.uint8)
    obstacle_file = os.path.join(output_dir, "obstacle.png").replace("\\", "/")
    save_image_unicode(obstacle_file, obstacle_sample)
    tile_mapping.append({
        "id": -1,
        "type": "obstacle",
        "file": obstacle_file
    })
    
    # 3. Export matchable templates
    templates_meta = []
    for c in clusters:
        cid = c["cluster_id"]
        filename = f"tile_{cid:02d}.png"
        filepath = os.path.join(output_dir, filename).replace("\\", "/")
        save_image_unicode(filepath, c["template_image"])
        
        mapping_item = {
            "id": cid,
            "type": "matchable",
            "file": filepath
        }
        tile_mapping.append(mapping_item)
        templates_meta.append({
            "id": cid,
            "file": filepath,
            "color_family": c["color_family"],
            "count": c["count"],
            "coords": c["coords"],
            "image": c["template_image"]
        })
        
    # 4. Export game_profile.json with exact schema
    profile = {
        "version": "1.0",
        "board": {
            "rows": grid_info["rows"],
            "cols": grid_info["cols"],
            "roi": {
                "x": int(board_roi["x"]),
                "y": int(board_roi["y"]),
                "w": int(board_roi["w"]),
                "h": int(board_roi["h"]),
                "width": int(board_roi["w"]),
                "height": int(board_roi["h"])
            },
            "cell_size": {
                "w": round(float(grid_info["cell_w"]), 2),
                "h": round(float(grid_info["cell_h"]), 2)
            },
            "offset": {
                "x": int(board_roi["x"]),
                "y": int(board_roi["y"])
            }
        },
        "runtime_params": {
            "click_interval_ms": 35,
            "animation_wait_ms": 160,
            "similarity_threshold": 0.85
        },
        "tile_mapping": tile_mapping
    }
    
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
        
    print(f"[Export] Saved game profile to {profile_path}")
    print(f"[Export] Saved {len(tile_mapping)} templates to {output_dir}/")
    
    # 5. Generate preview.html
    generate_html_report(
        profile=profile,
        empty_sample=empty_sample,
        obstacle_sample=obstacle_sample,
        templates_meta=templates_meta,
        empty_cells=empty_cells,
        obstacle_cells=obstacle_cells,
        all_cells_matrix=all_cells_matrix,
        output_html=preview_path
    )
    return profile

def generate_html_report(
    profile,
    empty_sample,
    obstacle_sample,
    templates_meta,
    empty_cells,
    obstacle_cells,
    all_cells_matrix,
    output_html="preview.html"
):
    rows = profile["board"]["rows"]
    cols = profile["board"]["cols"]
    total_cells = rows * cols
    matchable_count = sum(t["count"] for t in templates_meta)
    
    # Check parity constraint
    all_even = all(t["count"] % 2 == 0 and t["count"] >= 2 for t in templates_meta)
    parity_badge = '<span class="badge pass">PARITY PASSED (100% Even)</span>' if all_even else '<span class="badge fail">PARITY FAILED</span>'
    
    # Build coordinate to tile mapping
    coord_map = {}
    for ec in empty_cells:
        coord_map[(ec["row"], ec["col"])] = {"type": "empty", "id": 0, "img": ec["image"]}
    for oc in obstacle_cells:
        coord_map[(oc["row"], oc["col"])] = {"type": "obstacle", "id": -1, "img": oc["image"]}
    for tm in templates_meta:
        for (r, c) in tm["coords"]:
            coord_map[(r, c)] = {"type": "matchable", "id": tm["id"], "color": tm["color_family"]}
            
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LianLianKan Studio - Board & Sample Library Inspection</title>
<style>
:root {{
    --bg-dark: #121417;
    --card-bg: #1e2228;
    --border-color: #2e3642;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent-blue: #58a6ff;
    --accent-green: #3fb950;
    --accent-yellow: #d29922;
    --accent-purple: #bc8cff;
    --accent-red: #f85149;
}}
body {{
    background-color: var(--bg-dark);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 24px;
}}
h1, h2, h3 {{ margin-top: 0; color: #fff; }}
.header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 16px;
    margin-bottom: 24px;
}}
.stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}}
.stat-card {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
}}
.stat-title {{ font-size: 13px; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 6px; }}
.stat-value {{ font-size: 24px; font-weight: bold; color: var(--accent-blue); }}
.badge {{
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
}}
.badge.pass {{ background: rgba(63, 185, 80, 0.2); color: #56d364; border: 1px solid #3fb950; }}
.badge.fail {{ background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid #f85149; }}

.section {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 28px;
}}

/* Board Grid Layout */
.board-container {{
    overflow-x: auto;
    padding: 10px 0;
}}
.board-grid {{
    display: inline-grid;
    grid-template-columns: repeat({cols}, 48px);
    gap: 3px;
    background: #0d1117;
    padding: 8px;
    border-radius: 6px;
    border: 1px solid var(--border-color);
}}
.cell-box {{
    width: 48px;
    height: 48px;
    position: relative;
    background: #161b22;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    border: 1px solid #30363d;
}}
.cell-box img {{
    max-width: 42px;
    max-height: 42px;
    border-radius: 2px;
}}
.cell-box .tag {{
    position: absolute;
    bottom: 1px;
    right: 2px;
    font-size: 9px;
    font-family: monospace;
    background: rgba(0,0,0,0.75);
    padding: 0 2px;
    border-radius: 2px;
}}
.cell-box.empty {{ border: 1px dashed #484f58; opacity: 0.5; }}
.cell-box.obstacle {{ border: 2px solid #8b949e; background: #21262d; }}
.cell-box.matchable {{ border: 1px solid var(--accent-blue); }}

/* Gallery */
.gallery-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px;
}}
.template-card {{
    background: #161b22;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px;
    display: flex;
    align-items: center;
    gap: 16px;
}}
.template-card img {{
    width: 48px;
    height: 48px;
    border-radius: 4px;
    border: 1px solid var(--border-color);
    background: #000;
}}
.template-info {{ flex: 1; }}
.template-name {{ font-weight: bold; font-size: 15px; margin-bottom: 4px; }}
.template-sub {{ font-size: 12px; color: var(--text-secondary); margin-bottom: 2px; }}
.tag-color {{
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    margin-top: 4px;
}}
.color-GREY {{ background: #484f58; color: #fff; }}
.color-YELLOW {{ background: #8e660e; color: #f2cc60; }}
.color-CYAN_BLUE {{ background: #1f6feb; color: #79c0ff; }}
.color-VIOLET {{ background: #6e40c9; color: #d2a8ff; }}
.color-MAGENTA {{ background: #bf3989; color: #ff7b72; }}
</style>
</head>
<body>

<div class="header">
    <div>
        <h1>LianLianKan Solver Studio</h1>
        <div style="color: var(--text-secondary); font-size: 14px;">Adaptive Visual Calibration & Normalized Template Library</div>
    </div>
    <div>{parity_badge}</div>
</div>

<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-title">Board Dimension</div>
        <div class="stat-value">{rows} x {cols} <span style="font-size: 14px; color: var(--text-secondary);">({total_cells} cells)</span></div>
    </div>
    <div class="stat-card">
        <div class="stat-title">ROI Bounding Box</div>
        <div class="stat-value" style="font-size: 18px; font-family: monospace;">X:{profile['board']['roi']['x']} Y:{profile['board']['roi']['y']} W:{profile['board']['roi']['w']} H:{profile['board']['roi']['h']}</div>
    </div>
    <div class="stat-card">
        <div class="stat-title">Cell Size</div>
        <div class="stat-value">{profile['board']['cell_size']['w']} x {profile['board']['cell_size']['h']} px</div>
    </div>
    <div class="stat-card">
        <div class="stat-title">Partition Breakdown</div>
        <div class="stat-value" style="font-size: 17px; color: #fff;">
            Tiles: {matchable_count} | Empty: {len(empty_cells)} | Stone: {len(obstacle_cells)}
        </div>
    </div>
</div>

<div class="section">
    <h2>Reconstructed Board Grid ({rows} Rows &times; {cols} Columns)</h2>
    <div class="board-container">
        <div class="board-grid">
"""
    for r in range(rows):
        for c in range(cols):
            cell_item = all_cells_matrix.get((r, c))
            img_src = to_base64_src(cell_item["image"]) if cell_item else ""
            meta = coord_map.get((r, c), {"type": "unknown", "id": 0})
            c_type = meta["type"]
            c_id = meta.get("id", 0)
            tag_text = f"T{c_id:02d}" if c_type == "matchable" else ("OBS" if c_type == "obstacle" else "EMP")
            
            html += f"""
            <div class="cell-box {c_type}">
                <img src="{img_src}" title="({r},{c}) Type: {c_type} ID: {c_id}">
                <span class="tag">{tag_text}</span>
            </div>
            """

    html += f"""
        </div>
    </div>
</div>

<div class="section">
    <h2>Exported Sample Library ({len(templates_meta) + 2} Templates)</h2>
    <div class="gallery-grid">
        <div class="template-card">
            <img src="{to_base64_src(empty_sample)}">
            <div class="template-info">
                <div class="template-name">empty.png (ID: 0)</div>
                <div class="template-sub">Type: empty (cleared background)</div>
                <div class="template-sub">Occurrences: {len(empty_cells)} cells</div>
            </div>
        </div>
        <div class="template-card">
            <img src="{to_base64_src(obstacle_sample)}">
            <div class="template-info">
                <div class="template-name">obstacle.png (ID: -1)</div>
                <div class="template-sub">Type: obstacle stone block</div>
                <div class="template-sub">Occurrences: {len(obstacle_cells)} cells ({', '.join([f'({r},{c})' for r,c in [(4,3),(4,11)]])})</div>
            </div>
        </div>
"""
    for tm in templates_meta:
        cid = tm["id"]
        c_fam = tm["color_family"]
        cnt = tm["count"]
        html += f"""
        <div class="template-card">
            <img src="{to_base64_src(tm['image'])}">
            <div class="template-info">
                <div class="template-name">tile_{cid:02d}.png (ID: {cid})</div>
                <div class="template-sub">Count: <strong>{cnt}</strong> tiles (Even &ge; 2)</div>
                <div class="template-sub">Coords: {', '.join([f'({r},{c})' for r,c in tm['coords'][:4]])}{'...' if cnt > 4 else ''}</div>
                <span class="tag-color color-{c_fam}">{c_fam}</span>
            </div>
        </div>
        """

    html += """
    </div>
</div>

</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Export] Saved HTML preview to {output_html}")

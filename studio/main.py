"""
LianLianKan Visual Studio CLI
Main entry point for calibration, slicing, clustering, exporting, and previewing.
"""

import os
import sys
import argparse
import webbrowser
import cv2

from grid_calibrator import (
    load_image,
    save_image,
    detect_board_roi,
    estimate_adaptive_grid,
    slice_cells,
    draw_grid_overlay
)
from feature_extractor import extract_features
from clusterer import cluster_matchable_tiles
from exporter import export_all

def run_studio_pipeline(
    image_path,
    roi_override=None,
    interactive_roi=False,
    open_browser=False,
    output_templates="templates",
    output_profile="game_profile.json",
    output_preview="preview.html"
):
    print("=" * 60)
    print("LianLianKan Studio - Visual Slicer & Profile Generator")
    print("=" * 60)
    
    # 1. Load Image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    img = load_image(image_path)
    h, w, c = img.shape
    print(f"[1/6] Loaded image: {image_path} (W={w}, H={h})")
    
    # 2. ROI Determination
    if interactive_roi:
        print("[ROI] Opening OpenCV window for interactive ROI selection...")
        print("[ROI] Drag to select board ROI, then press SPACE or ENTER. Press 'c' to cancel.")
        cv_roi = cv2.selectROI("Select Board ROI", img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        if cv_roi[2] > 0 and cv_roi[3] > 0:
            roi = {"x": int(cv_roi[0]), "y": int(cv_roi[1]), "w": int(cv_roi[2]), "h": int(cv_roi[3])}
        else:
            print("[ROI] Selection cancelled. Falling back to default/auto-detection.")
            roi = detect_board_roi(img)
    elif roi_override:
        roi = roi_override
    else:
        # Default verified ROI on 未命名.jpg or auto-detect
        roi = {"x": 68, "y": 49, "w": 675, "h": 407}
        
    print(f"[2/6] Board ROI: X={roi['x']}, Y={roi['y']}, W={roi['w']}, H={roi['h']}")
    
    # 3. Adaptive Grid Estimation (Autocorrelation Signal Analysis)
    roi_img = img[roi["y"]:roi["y"]+roi["h"], roi["x"]:roi["x"]+roi["w"]]
    grid_info = estimate_adaptive_grid(roi_img)
    rows = grid_info["rows"]
    cols = grid_info["cols"]
    print(f"[3/6] Adaptive Grid Detection: {rows} Rows x {cols} Columns")
    print(f"      Pitch: dx={grid_info['pitch_x']:.2f}px, dy={grid_info['pitch_y']:.2f}px")
    print(f"      Cell Size: {grid_info['cell_w']:.2f} x {grid_info['cell_h']:.2f} px")
    
    # Draw and save grid overlay
    draw_grid_overlay(img, roi, rows, cols, output_path="debug_grid_on_roi.png")
    print("      Saved grid overlay visualization to debug_grid_on_roi.png")
    
    # 4. Slicing and Feature Extraction
    print(f"[4/6] Slicing {rows * cols} cells and extracting visual features...")
    raw_cells = slice_cells(img, roi, rows, cols)
    
    empty_cells = []
    obstacle_cells = []
    matchable_tiles = []
    all_cells_matrix = {}
    
    for c in raw_cells:
        feat = extract_features(c["image"])
        c["features"] = feat
        all_cells_matrix[(c["row"], c["col"])] = c
        
        if feat["type"] == "empty":
            empty_cells.append(c)
        elif feat["type"] == "obstacle":
            obstacle_cells.append(c)
        else:
            matchable_tiles.append(c)
            
    print(f"      Classification results:")
    print(f"      - Empty background cells: {len(empty_cells)}")
    print(f"      - Obstacle stone blocks:  {len(obstacle_cells)} (coords: {[(oc['row'], oc['col']) for oc in obstacle_cells]})")
    print(f"      - Matchable tiles:        {len(matchable_tiles)}")
    
    # 5. Parity-Constrained Clustering
    print(f"[5/6] Clustering {len(matchable_tiles)} matchable tiles with Parity Constraint...")
    clusters, parity_valid = cluster_matchable_tiles(matchable_tiles)
    print(f"      Formed {len(clusters)} distinct tile clusters.")
    for cl in clusters:
        print(f"      - Tile {cl['cluster_id']:02d} [{cl['color_family']}]: count={cl['count']} tiles (Parity OK: {cl['count'] % 2 == 0})")
        
    if parity_valid:
        print("      [PASSED] All clusters satisfy the Parity Constraint (even count >= 2)!")
    else:
        print("      [WARNING] One or more clusters violated the parity constraint!")
        
    # 6. Exporter
    print(f"[6/6] Exporting templates, game profile, and HTML preview...")
    profile = export_all(
        board_roi=roi,
        grid_info=grid_info,
        empty_cells=empty_cells,
        obstacle_cells=obstacle_cells,
        clusters=clusters,
        all_cells_matrix=all_cells_matrix,
        output_dir=output_templates,
        profile_path=output_profile,
        preview_path=output_preview
    )
    
    if open_browser:
        preview_abs = os.path.abspath(output_preview)
        print(f"[Preview] Opening browser to: {preview_abs}")
        webbrowser.open(f"file:///{preview_abs}")
        
    print("=" * 60)
    print(f"Pipeline successfully completed!")
    print(f"Output Game Profile: {os.path.abspath(output_profile)}")
    print(f"Output Templates Dir: {os.path.abspath(output_templates)}")
    print(f"Output HTML Preview: {os.path.abspath(output_preview)}")
    print("=" * 60)
    return profile

def main():
    parser = argparse.ArgumentParser(description="LianLianKan Visual Studio")
    parser.add_argument("--image", "-i", type=str, default=r"..\data\未命名.jpg", help="Path to input board image")
    parser.add_argument("--roi", nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'), help="Manual ROI bounding box")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive cv2.selectROI")
    parser.add_argument("--open", action="store_true", help="Automatically open preview.html in browser")
    parser.add_argument("--templates", type=str, default="templates", help="Output templates directory")
    parser.add_argument("--profile", type=str, default="game_profile.json", help="Output game_profile.json path")
    parser.add_argument("--preview", type=str, default="preview.html", help="Output preview.html path")
    
    args = parser.parse_args()
    
    roi_override = None
    if args.roi:
        roi_override = {"x": args.roi[0], "y": args.roi[1], "w": args.roi[2], "h": args.roi[3]}
        
    run_studio_pipeline(
        image_path=args.image,
        roi_override=roi_override,
        interactive_roi=args.interactive,
        open_browser=args.open,
        output_templates=args.templates,
        output_profile=args.profile,
        output_preview=args.preview
    )

if __name__ == "__main__":
    main()

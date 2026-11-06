"""
LianLianKan Visual Studio CLI
Supports:
1. Live Screen Snip & Snap-to-Grid Auto-Refinement (Multi-monitor DPI-aware)
2. Existing Screenshot Image File Analysis (drag-and-drop, user-selected ROI)
"""

import os
import sys
import argparse
import webbrowser
import cv2
import numpy as np

from grid_calibrator import (
    load_image,
    save_image,
    refine_board_roi,
    estimate_adaptive_grid,
    slice_cells,
    draw_grid_overlay
)
from feature_extractor import extract_features, classify_cells
from clusterer import cluster_matchable_tiles
from exporter import export_all
from screen_capturer import (
    get_system_monitors,
    capture_screen
)

def find_default_image():
    candidates = [
        "screenshot_board.png",
        "data/未命名.jpg",
        "data/未命名(3).jpg",
        "data/未命名(2).png",
        "data/未命名(4).jpg",
        "../data/未命名.jpg",
        "../data/未命名(3).jpg",
        "未命名.jpg",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def run_studio_pipeline(
    image_input,
    roi_override=None,
    interactive_roi=False,
    open_browser=False,
    output_templates="templates",
    output_profile="game_profile.json",
    output_preview="preview.html",
    export_roi_override=None
):
    print("=" * 60)
    print("LianLianKan Studio - Visual Slicer & Profile Generator")
    print("=" * 60)
    
    # 1. Load Image (from path or numpy array)
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Input image file not found: {image_input}")
        img = load_image(image_input)
        img_label = os.path.basename(image_input)
    elif isinstance(image_input, np.ndarray):
        img = image_input
        img_label = "<In-Memory Buffer>"
    else:
        raise ValueError("Invalid image input type: must be path str or numpy ndarray")
        
    if img is None:
        raise ValueError(f"Failed to decode image from: {image_input}")
        
    h, w, _ = img.shape
    print(f"[1/6] Loaded image: {img_label} (W={w}, H={h})")
    
    # 2. ROI Determination & Auto-Refinement
    if roi_override:
        roi = roi_override
        print(f"[ROI] 使用指定盤面範圍: X={roi['x']}, Y={roi['y']}, W={roi['w']}, H={roi['h']}")
    elif interactive_roi:
        print("\n" + "=" * 60)
        print(" [ROI 互動框選提示]")
        print(" 已開啟視窗【Select Board ROI】：")
        print(" 1. 請使用滑鼠左鍵【拖曳框選】遊戲棋盤大略區域（系統會自動磁吸對齊磚塊外緣）")
        print(" 2. 框選完成後請按鍵盤【SPACE】或【ENTER】確認")
        print(" 3. 框選範圍超出棋盤外緣請勿大於 1 個磚塊")
        print("=" * 60 + "\n")
        cv_roi = cv2.selectROI("Select Board ROI", img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        if cv_roi[2] > 20 and cv_roi[3] > 20:
            rough_roi = {"x": int(cv_roi[0]), "y": int(cv_roi[1]), "w": int(cv_roi[2]), "h": int(cv_roi[3])}
            print(f"[ROI 手動框選] 取得大略範圍: X={rough_roi['x']}, Y={rough_roi['y']}, W={rough_roi['w']}, H={rough_roi['h']}")
            roi = refine_board_roi(img, rough_roi)
            if roi != rough_roi:
                print(f"[網格對齊] 已自動校準至物理網格晶格: X={roi['x']}, Y={roi['y']}, W={roi['w']}, H={roi['h']}")
        else:
            raise RuntimeError("未框選棋盤（需用滑鼠拖曳框選並按 SPACE/ENTER 確認），已取消。")
    else:
        raise RuntimeError("需要使用者手動框選棋盤：請勿使用 --no-interactive，或以 --roi X Y W H 指定範圍。")

    print(f"[2/6] Board ROI: X={roi['x']}, Y={roi['y']}, W={roi['w']}, H={roi['h']}")
    
    # 3. Adaptive Grid Estimation
    rx, ry, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
    roi_crop = img[ry:ry+rh, rx:rx+rw]
    
    grid_info = estimate_adaptive_grid(roi_crop)
    rows = grid_info["rows"]
    cols = grid_info["cols"]
    print(f"[3/6] Adaptive Grid Detection: {rows} Rows x {cols} Columns")
    print(f"      Pitch: dx={grid_info['pitch_x']:.2f}px, dy={grid_info['pitch_y']:.2f}px")
    print(f"      Cell Size: {grid_info['cell_w']:.2f} x {grid_info['cell_h']:.2f} px")
    
    # Save visualization of calibrated grid
    draw_grid_overlay(img, roi, rows, cols, output_path="debug_grid_on_roi.png")
    print("      Saved grid overlay visualization to debug_grid_on_roi.png")
    
    # 4. Slicing & Visual Feature Extraction
    cells = slice_cells(img, roi, rows, cols)
    print(f"[4/6] Slicing {len(cells)} cells and extracting visual features...")
    
    empty_cells = []
    obstacle_cells = []
    matchable_tiles = []
    all_cells_matrix = {}
    
    for c in cells:
        c["features"] = extract_features(c["image"])
        c["tile_type"] = c["features"].get("type", "matchable")
    classify_cells(cells)

    for c in cells:
        r_idx = c["row"]
        c_idx = c["col"]
        tile_type = c["tile_type"]
        all_cells_matrix[(r_idx, c_idx)] = c

        if tile_type == "empty":
            empty_cells.append(c)
        elif tile_type == "obstacle":
            obstacle_cells.append(c)
        else:
            matchable_tiles.append(c)

    print(f"      Classification results:")
    print(f"      - Empty background cells: {len(empty_cells)}")
    print(f"      - Obstacle stone blocks:  {len(obstacle_cells)} (coords: {[(ob['row'], ob['col']) for ob in obstacle_cells]})")
    print(f"      - Matchable tiles:        {len(matchable_tiles)}")
    
    # 5. Clusterer with Parity Constraint
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
    exported_roi = export_roi_override if export_roi_override else roi
    print(f"[6/6] Exporting templates, game profile, and HTML preview...")
    profile = export_all(
        board_roi=exported_roi,
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

def run_screen_snip_mode(
    monitor_index=None,
    output_screenshot="screenshot_board.png",
    open_browser=True,
    output_templates="templates",
    output_profile="game_profile.json",
    output_preview="preview.html"
):
    """
    Mode 3: Live Screen Snip with Snap-to-Grid Alignment
    """
    print("\n" + "=" * 60)
    print("      [模式 1：螢幕直接框選截圖 (多螢幕 & 高 DPI 支援)]")
    print("=" * 60)
    
    monitors = get_system_monitors()
    chosen_monitor = None
    
    if len(monitors) > 1 and monitor_index is None:
        print(f"系統偵測到 {len(monitors)} 個顯示器：")
        for m in monitors:
            p_str = " (主要螢幕)" if m["is_primary"] else ""
            r = m["rect"]
            print(f"  [{m['index']}] {m['device']}{p_str}: 解析度={r[2]}x{r[3]}, 縮放={m['scale']}% (X={r[0]}, Y={r[1]})")
        print("=" * 60)
        try:
            choice = input(f"請選擇遊戲所在的螢幕 [1-{len(monitors)}] (預設 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            return None
        try:
            idx = int(choice) if choice else 1
            if 1 <= idx <= len(monitors):
                chosen_monitor = monitors[idx - 1]
        except ValueError:
            chosen_monitor = monitors[0]
            
    if chosen_monitor is None:
        if monitor_index is not None and 1 <= monitor_index <= len(monitors):
            chosen_monitor = monitors[monitor_index - 1]
        elif monitors:
            chosen_monitor = monitors[0]
            
    print(f"\n[擷取] 正在擷取顯示器: {chosen_monitor['device']} (實體像素: {chosen_monitor['rect'][2]}x{chosen_monitor['rect'][3]}, 縮放: {chosen_monitor['scale']}%)")
    screen_img, chosen_monitor = capture_screen(monitor=chosen_monitor)
    h, w, _ = screen_img.shape
    
    print("=" * 60)
    print(" [ROI 框選提示]")
    print(f" 已開啟視窗【Screen Snip - {chosen_monitor['device']}】：")
    print(" 1. 使用滑鼠左鍵【拖曳框選】棋盤（系統會自動向內/向外磁吸對齊磚塊外緣）")
    print(" 2. 框選完畢後請按【SPACE】或【ENTER】確認")
    print(" 3. 框選範圍超出棋盤外緣請勿大於 1 個磚塊")
    print("=" * 60 + "\n")

    win_title = f"Screen Snip - {chosen_monitor['device']} (SPACE/ENTER: Confirm)"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    try:
        cv2.moveWindow(win_title, chosen_monitor['rect'][0], chosen_monitor['rect'][1])
        cv2.resizeWindow(win_title, chosen_monitor['rect'][2], chosen_monitor['rect'][3])
        cv2.setWindowProperty(win_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass

    cv_roi = cv2.selectROI(win_title, screen_img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    rx, ry, rw, rh = int(cv_roi[0]), int(cv_roi[1]), int(cv_roi[2]), int(cv_roi[3])
    
    if not (rw > 20 and rh > 20):
        print("\n[取消] 未手動框選棋盤，已結束。")
        return None
    rough_roi = {"x": rx, "y": ry, "w": rw, "h": rh}
    print(f"\n[手動框選] 取得大略範圍: X={rx}, Y={ry}, 寬={rw}, 高={rh}")
    final_roi = refine_board_roi(screen_img, rough_roi)
    if final_roi != rough_roi:
        print(f"[網格對齊] 已自動校準至物理網格晶格: X={final_roi['x']}, Y={final_roi['y']}, 寬={final_roi['w']}, 高={final_roi['h']}")

    fx, fy, fw, fh = final_roi["x"], final_roi["y"], final_roi["w"], final_roi["h"]
    cropped_board = screen_img[fy:fy+fh, fx:fx+fw]
    save_image(output_screenshot, cropped_board)
    print(f"[儲存] 已將盤面截圖存至: {os.path.abspath(output_screenshot)}")

    phys_x = chosen_monitor['rect'][0] + fx
    phys_y = chosen_monitor['rect'][1] + fy
    print(f"[實體座標] 虛擬桌面絕對實體座標: X={phys_x}, Y={phys_y}, W={fw}, H={fh}")

    return run_studio_pipeline(
        image_input=cropped_board,
        roi_override={"x": 0, "y": 0, "w": fw, "h": fh},
        interactive_roi=False,
        open_browser=open_browser,
        output_templates=output_templates,
        output_profile=output_profile,
        output_preview=output_preview,
        export_roi_override={"x": phys_x, "y": phys_y, "w": fw, "h": fh}
    )

def run_existing_image_mode(
    image_path=None,
    roi_override=None,
    interactive_roi=True,
    open_browser=True,
    output_templates="templates",
    output_profile="game_profile.json",
    output_preview="preview.html"
):
    """
    Mode 4: Existing Image File Analysis
    """
    if not image_path:
        default_img = find_default_image()
        print("\n" + "=" * 60)
        print("        [模式 2：讀取現有截圖檔案]")
        print("=" * 60)
        print(" 請提供遊戲截圖檔案：")
        print(" 1. 直接將圖片檔案【拖曳進此視窗】後按 Enter")
        if default_img:
            print(f" 2. 或直接按 Enter 使用預設圖片: {default_img}")
        print("=" * 60)
        while True:
            try:
                user_input = input("請輸入或拖曳圖片路徑: ").strip().strip('"').strip("'")
            except (EOFError, KeyboardInterrupt):
                print("\n操作已取消。")
                return None
            if not user_input:
                if default_img and os.path.exists(default_img):
                    image_path = default_img
                    print(f"[提示] 選用預設圖片: {image_path}")
                    break
                else:
                    print("[錯誤] 請輸入有效的圖片路徑！")
                    continue
            elif os.path.exists(user_input):
                image_path = user_input
                break
            else:
                print(f"[錯誤] 找不到檔案: {user_input}，請重新輸入！")

    return run_studio_pipeline(
        image_input=image_path,
        roi_override=roi_override,
        interactive_roi=interactive_roi,
        open_browser=open_browser,
        output_templates=output_templates,
        output_profile=output_profile,
        output_preview=output_preview
    )

def main():
    parser = argparse.ArgumentParser(description="LianLianKan Visual Studio")
    parser.add_argument("--snip", "-s", action="store_true", help="Launch interactive screen snip mode")
    parser.add_argument("--monitor", "-m", type=int, default=None, help="Target monitor index (1-based)")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input screenshot image")
    parser.add_argument("--roi", nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'), help="Manual ROI bounding box")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive ROI selection GUI")
    parser.add_argument("--no-browser", action="store_true", help="Do not open preview.html in browser")
    parser.add_argument("--templates", type=str, default="templates", help="Output templates directory")
    parser.add_argument("--profile", type=str, default="game_profile.json", help="Output game_profile.json path")
    parser.add_argument("--preview", type=str, default="preview.html", help="Output preview.html path")
    
    args, unknown = parser.parse_known_args()
    
    dragged_file = None
    if unknown:
        candidate = unknown[0].strip('"').strip("'")
        if os.path.exists(candidate) and os.path.isfile(candidate):
            dragged_file = candidate

    # 1. Screen Snip Mode
    if args.snip:
        run_screen_snip_mode(
            monitor_index=args.monitor,
            open_browser=not args.no_browser,
            output_templates=args.templates,
            output_profile=args.profile,
            output_preview=args.preview
        )
    # 2. Image Mode
    elif args.image or dragged_file:
        target_img = args.image if args.image else dragged_file
        roi_override = None
        if args.roi:
            roi_override = {"x": args.roi[0], "y": args.roi[1], "w": args.roi[2], "h": args.roi[3]}
            interactive = False
        else:
            interactive = not args.no_interactive

        run_existing_image_mode(
            image_path=target_img,
            roi_override=roi_override,
            interactive_roi=interactive,
            open_browser=not args.no_browser,
            output_templates=args.templates,
            output_profile=args.profile,
            output_preview=args.preview
        )
    # 3. Interactive Menu Mode
    else:
        print()
        print("=" * 60)
        print("       LianLianKan Studio 盤面分析與特徵建模工具")
        print("=" * 60)
        print("請選擇分析模式：")
        print("  [1] 螢幕框選截圖 (滑鼠手動框選棋盤，框選範圍超出棋盤勿大於 1 個磚塊)")
        print("  [2] 讀取現有截圖檔案 (支援拖曳圖片，並手動框選棋盤)")
        print("=" * 60)

        try:
            choice = input("請選擇模式 [1/2] (預設 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("已取消。")
            return

        if choice == "2":
            run_existing_image_mode(
                open_browser=not args.no_browser,
                output_templates=args.templates,
                output_profile=args.profile,
                output_preview=args.preview
            )
        else:
            run_screen_snip_mode(
                monitor_index=args.monitor,
                open_browser=not args.no_browser,
                output_templates=args.templates,
                output_profile=args.profile,
                output_preview=args.preview
            )

if __name__ == "__main__":
    main()

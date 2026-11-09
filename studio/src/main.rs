//! LianLianKan Studio (Rust port).
//!
//!   studio-rs --snip [--monitor N]                  live screen capture + mouse selection
//!   studio-rs --image FILE --select                 mouse selection on a screenshot file
//!   studio-rs --image FILE [--rough-roi X Y W H | --roi X Y W H]   no window (scripts / tests)
//!   studio-rs --add (--snip | --image FILE [--select])   add NEW objects from another screenshot
//!                                                       to the existing profile/templates
//!
//! Common: [--templates DIR] [--profile FILE] [--preview FILE] [--dump FILE]
//!         [--screenshot FILE] [--open | --no-browser]
//!
//! The board is located inside the user's selection exactly like the Python studio:
//! pitch/phase calibration -> cell slicing -> empty/obstacle/tile classification ->
//! parity-constrained clustering -> templates + game_profile.json + editable preview.html.

mod augment;
mod calibrate;
mod capture;
mod cluster;
mod export;
mod features;
mod imgproc;
mod select;
mod serve;

use anyhow::{bail, Context, Result};
use calibrate::{estimate_adaptive_grid, refine_board_roi, slice_cells, Roi};
use export::{ExportCell, ExportTemplate};
use features::{classify_cells, CellType};
use imgproc::{resize, Image};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone)]
struct Args {
    /// started without a mode: show the menu and come back to it after every action
    interactive: bool,
    image: Option<PathBuf>,
    snip: bool,
    add: bool,
    reroi: bool,
    settings: bool,
    select: bool,
    selftest: bool,
    monitor: Option<usize>,
    roi: Option<Roi>,
    rough: Option<Roi>,
    templates: PathBuf,
    profile: PathBuf,
    preview: PathBuf,
    dump: Option<PathBuf>,
    screenshot: PathBuf,
    open: Option<bool>,
    no_serve: bool,
}

fn parse_roi(a: &[String], i: usize) -> Result<Roi> {
    let v: Vec<i64> = a
        .get(i + 1..i + 5)
        .context("expected 4 numbers: X Y W H")?
        .iter()
        .map(|s| s.parse::<i64>().context("ROI values must be integers"))
        .collect::<Result<_>>()?;
    Ok(Roi { x: v[0], y: v[1], w: v[2], h: v[3] })
}

fn parse_args() -> Result<Args> {
    let mut a: Vec<String> = std::env::args().collect();
    // Dropping an image onto the exe passes its path as the only argument.
    if a.len() == 2 && !a[1].starts_with('-') && Path::new(&a[1]).is_file() {
        let f = a[1].clone();
        a = vec![a[0].clone(), "--image".into(), f, "--select".into()];
    }
    let mut args = Args {
        interactive: false,
        image: None,
        snip: false,
        add: false,
        reroi: false,
        settings: false,
        select: false,
        selftest: false,
        monitor: None,
        roi: None,
        rough: None,
        templates: PathBuf::from("templates"),
        profile: PathBuf::from("game_profile.json"),
        preview: PathBuf::from("preview.html"),
        dump: None,
        screenshot: PathBuf::from("screenshot_board.png"),
        open: None,
        no_serve: false,
    };
    let mut i = 1;
    while i < a.len() {
        let next = |i: usize| a.get(i + 1).cloned().context("missing value");
        match a[i].as_str() {
            "--image" | "-i" => { args.image = Some(PathBuf::from(next(i)?)); i += 1; }
            "--snip" | "-s" => args.snip = true,
            "--add" => args.add = true,
            "--reroi" => args.reroi = true,
            "--settings" => args.settings = true,
            "--select" => args.select = true,
            "--self-test-window" => args.selftest = true,
            "--monitor" | "-m" => { args.monitor = Some(next(i)?.parse().context("--monitor needs a number")?); i += 1; }
            "--templates" => { args.templates = PathBuf::from(next(i)?); i += 1; }
            "--profile" => { args.profile = PathBuf::from(next(i)?); i += 1; }
            "--preview" => { args.preview = PathBuf::from(next(i)?); i += 1; }
            "--screenshot" => { args.screenshot = PathBuf::from(next(i)?); i += 1; }
            "--dump" => { args.dump = Some(PathBuf::from(next(i)?)); i += 1; }
            "--open" => args.open = Some(true),
            "--no-browser" => args.open = Some(false),
            "--no-serve" => args.no_serve = true,
            "--roi" => { args.roi = Some(parse_roi(&a, i)?); i += 4; }
            "--rough-roi" => { args.rough = Some(parse_roi(&a, i)?); i += 4; }
            "-h" | "--help" => {
                println!("usage:\n  studio-rs --add (--snip | --image FILE [--select])   (add new objects to an existing profile)\n  studio-rs --snip [--monitor N]\n  studio-rs --image FILE --select\n  studio-rs --image FILE [--rough-roi X Y W H | --roi X Y W H]\n\ncommon: [--templates DIR] [--profile FILE] [--preview FILE] [--screenshot FILE] [--dump FILE] [--open | --no-browser] [--no-serve]");
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
        i += 1;
    }
    args.interactive = !args.snip && !args.reroi && !args.settings && !args.selftest && args.image.is_none();
    Ok(args)
}

/// Interactive mode picker (used when the exe is started without arguments).
/// Fills `args` with the chosen mode; Ok(false) = the user chose to quit.
fn menu(args: &mut Args) -> Result<bool> {
    println!("{}", "=".repeat(60));
    println!("       LianLianKan Studio 盤面分析與特徵建模工具");
    println!("{}", "=".repeat(60));
    println!("請選擇分析模式：");
    println!("  [1] 螢幕框選截圖 (滑鼠手動框選棋盤，框選範圍超出棋盤勿大於 1 個磚塊)");
    println!("  [2] 追加截圖 (新關卡出現新物件時，從螢幕框選，把新物件補進現有模板庫)");
    println!("  [3] 調整既有 profile 的框選區域 (遊戲視窗移動／縮放後，重新對準 ROI)");
    println!("  [4] 開啟設定頁面 (編輯既有 profile 的時間／辨識／規則設定，不需截圖)");
    println!("  [0] 離開");
    println!("  （讀取截圖檔案請改用命令列：studio --image FILE --select）");
    println!("{}", "=".repeat(60));
    loop {
        print!("請選擇模式 [0/1/2/3/4] (預設 1): ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            return Ok(false); // stdin closed
        }
        match line.trim() {
            "" | "1" => args.snip = true,
            "2" => {
                args.add = true;
                args.snip = true;
            }
            "3" => args.reroi = true,
            "4" => args.settings = true,
            "0" | "q" | "Q" => return Ok(false),
            other => {
                println!("[錯誤] 無效的選項: {other}");
                continue;
            }
        }
        return Ok(true);
    }
}

fn open_in_browser(path: &Path) {
    if let Ok(abs) = std::fs::canonicalize(path) {
        let s = abs.to_string_lossy().trim_start_matches(r"\\?\").to_string();
        let _ = std::process::Command::new("cmd").args(["/C", "start", "", &s]).spawn();
    }
}

fn print_selection_help() {
    println!("{}", "=".repeat(60));
    println!(" [ROI 框選提示]");
    println!(" 1. 用滑鼠左鍵【拖曳框選】棋盤（系統會自動磁吸對齊磚塊外緣）");
    println!(" 2. 框好後可再調整：拖曳【框內】移動；拖曳【邊或角的方塊】縮放；");
    println!("    方向鍵微調位置（Shift = 10 px），Ctrl+方向鍵調整右／下邊；框外拖曳 = 重畫");
    println!(" 3. 按【Enter】或【Space】確認；【Esc】取消");
    println!(" 4. 框選範圍超出棋盤外緣請勿大於 1 個磚塊");
    println!("{}", "=".repeat(60));
}

/// Steps 3-6: grid -> classify -> cluster -> export. `board` is the cropped board image,
/// `profile_roi` the rectangle written into game_profile.json (screen coordinates for snips).
fn process_board(board: &Image, profile_roi: Roi, args: &Args) -> Result<()> {
    let grid = estimate_adaptive_grid(board);
    println!("[3/6] Adaptive Grid Detection: {} Rows x {} Columns (pitch {:.2} x {:.2})", grid.rows, grid.cols, grid.pitch_x, grid.pitch_y);

    let board_roi = Roi { x: 0, y: 0, w: board.w as i64, h: board.h as i64 };
    let cells = classify_cells(slice_cells(board, board_roi, grid.rows, grid.cols));
    let n_empty = cells.iter().filter(|c| c.ty == CellType::Empty).count();
    let n_obst = cells.iter().filter(|c| c.ty == CellType::Obstacle).count();
    let tiles: Vec<usize> = (0..cells.len()).filter(|&i| cells[i].ty == CellType::Matchable).collect();
    println!("[4/6] Cells: {} empty, {} obstacle, {} matchable", n_empty, n_obst, tiles.len());

    let tile_images: Vec<&Image> = tiles.iter().map(|&i| &cells[i].cell.image).collect();
    let (clusters, parity_ok) = cluster::cluster_tiles(&tile_images);
    println!("[5/6] {} tile clusters (parity {})", clusters.len(), if parity_ok { "OK" } else { "VIOLATED" });
    for c in &clusters {
        println!("      - Tile {:02}: count={}", c.id, c.members.len());
    }

    let mut ids = vec![0i64; cells.len()];
    for cl in &clusters {
        for &m in &cl.members {
            ids[tiles[m]] = cl.id as i64;
        }
    }
    let export_cells: Vec<ExportCell> = cells
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let (kind, id) = match c.ty {
                CellType::Matchable => ("matchable", ids[i]),
                CellType::Obstacle => ("obstacle", -1),
                _ => ("empty", 0),
            };
            ExportCell { row: c.cell.row, col: c.cell.col, kind, id, image: c.cell.image.clone() }
        })
        .collect();
    let templates: Vec<ExportTemplate> = clusters
        .iter()
        .map(|cl| ExportTemplate { id: cl.id, image: resize(tile_images[cl.medoid], 45, 45) })
        .collect();

    export::export_all(profile_roi, &grid, &export_cells, &templates, &args.templates, &args.profile, &args.preview)?;
    println!("[6/6] Exported {:?}, {:?}, {:?}", args.profile, args.templates, args.preview);

    if let Some(d) = &args.dump {
        let cells_json: Vec<_> = export_cells.iter().map(|c| serde_json::json!([c.row, c.col, c.kind, c.id])).collect();
        let dump = serde_json::json!({
            "roi": {"x": profile_roi.x, "y": profile_roi.y, "w": profile_roi.w, "h": profile_roi.h},
            "rows": grid.rows, "cols": grid.cols, "parity_ok": parity_ok, "cells": cells_json
        });
        std::fs::write(d, serde_json::to_string(&dump)?)?;
    }
    Ok(())
}

fn finish(args: &Args, interactive: bool) {
    let open = args.open.unwrap_or(interactive);
    if open {
        if !args.no_serve {
            let cfg = serve::Cfg { preview: args.preview.clone(), profile: args.profile.clone(), templates: args.templates.clone() };
            match serve::serve(&cfg) {
                Ok(_) => return,
                Err(e) => println!("[Preview] 無法啟動編輯伺服器 ({e})，改為直接開啟檔案（儲存需用下載按鈕）。"),
            }
        }
        println!("[Preview] 開啟編輯頁面: {:?}", args.preview);
        open_in_browser(&args.preview);
    }
}

fn choose_monitor(args: &Args, mons: &[capture::Monitor], default_idx: usize) -> Result<capture::Monitor> {
    if let Some(n) = args.monitor {
        return mons.iter().find(|m| m.index == n).cloned().with_context(|| format!("no monitor #{n}"));
    }
    if mons.len() > 1 {
        println!("系統偵測到 {} 個顯示器：", mons.len());
        for m in mons {
            println!(
                "  [{}] {}{}: 解析度={}x{}, 縮放={}% (X={}, Y={})",
                m.index, m.device, if m.primary { " (主要螢幕)" } else { "" }, m.w, m.h, m.scale_percent, m.x, m.y
            );
        }
        print!("請選擇遊戲現在所在的螢幕 [1-{}] (預設 {}): ", mons.len(), default_idx);
        std::io::stdout().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        let n: usize = line.trim().parse().unwrap_or(default_idx);
        return Ok(mons.iter().find(|m| m.index == n).or_else(|| mons.iter().find(|m| m.index == default_idx)).unwrap_or(&mons[0]).clone());
    }
    Ok(mons[0].clone())
}

/// Shared by every flow (screen snip, image file, re-aim): shows `img` in the selection window and
/// returns the rough rectangle in `img` pixels, or None if cancelled / too small.
fn pick_rough_roi(img: &Image, title: &str, place: &select::Placement, initial: Option<Roi>) -> Result<Option<Roi>> {
    Ok(select::select_roi(img, title, place, initial)?.filter(|r| r.w > 20 && r.h > 20))
}

/// Shared by the screen and image flows: snap the rough rectangle to the board's grid lattice
/// and cut the board out. Returns (board crop, ROI in `img` pixels).
fn refine_and_crop(img: &Image, rough: Roi) -> (Image, Roi) {
    let roi = refine_board_roi(img, rough);
    (img.crop(roi.x as usize, roi.y as usize, roi.w as usize, roi.h as usize), roi)
}

/// Live screen capture + mouse selection. Returns (board crop, board rect in virtual-desktop px).
fn pick_board_from_screen(args: &Args) -> Result<Option<(Image, Roi)>> {
    println!("{}", "=".repeat(60));
    println!("      [螢幕直接框選截圖 (多螢幕 & 高 DPI 支援)]");
    println!("{}", "=".repeat(60));
    let mons = capture::monitors();
    if mons.is_empty() {
        bail!("no monitor found");
    }
    let mon = choose_monitor(args, &mons, 1)?;
    println!("[擷取] 顯示器 {}: {}x{} 實體像素, 縮放 {}%", mon.device, mon.w, mon.h, mon.scale_percent);
    let screen = capture::capture_rect(mon.x, mon.y, mon.w, mon.h)?;

    print_selection_help();
    let place = select::Placement { at: Some((mon.x as isize, mon.y as isize)), size: (mon.w as usize, mon.h as usize), borderless: true };
    let Some(rough) = pick_rough_roi(&screen, "Screen Snip - drag to select the board, Enter = OK, Esc = cancel", &place, None)? else {
        println!("[取消] 未手動框選棋盤，已結束。");
        return Ok(None);
    };
    println!("[手動框選] 取得大略範圍: X={}, Y={}, 寬={}, 高={}", rough.x, rough.y, rough.w, rough.h);
    let (board, roi) = refine_and_crop(&screen, rough);
    println!("[網格對齊] 已自動校準至物理網格晶格: X={}, Y={}, 寬={}, 高={}", roi.x, roi.y, roi.w, roi.h);

    std::fs::write(&args.screenshot, export::png_bytes(&board))?;
    println!("[儲存] 已將盤面截圖存至: {:?}", args.screenshot);
    let phys = Roi { x: mon.x as i64 + roi.x, y: mon.y as i64 + roi.y, w: roi.w, h: roi.h };
    println!("[實體座標] 虛擬桌面絕對實體座標: X={}, Y={}, W={}, H={}", phys.x, phys.y, phys.w, phys.h);
    Ok(Some((board, phys)))
}

/// A screenshot file (+ optional mouse selection). Returns (board crop, board rect in image px).
fn pick_board_from_image(args: &Args) -> Result<Option<(Image, Roi)>> {
    let path = args.image.as_ref().context("--image is required")?;
    let img = Image::from_rgb(&image::open(path).with_context(|| format!("cannot read {path:?}"))?.to_rgb8());
    println!("[1/6] Loaded image: {:?} (W={}, H={})", path, img.w, img.h);

    let (board, roi) = if let Some(r) = args.roi {
        // exact ROI given: no snapping
        (img.crop(r.x as usize, r.y as usize, r.w as usize, r.h as usize), r)
    } else {
        let rough = if args.select {
            print_selection_help();
            // fit the image on the primary monitor (the window shows it scaled; coordinates map back)
            let mons = capture::monitors();
            let (mw, mh) = mons.first().map(|m| (m.w as f64, m.h as f64)).unwrap_or((1280.0, 720.0));
            let sc = (mw * 0.9 / img.w as f64).min(mh * 0.9 / img.h as f64).min(1.0);
            let size = (((img.w as f64 * sc) as usize).max(100), ((img.h as f64 * sc) as usize).max(100));
            let place = select::Placement { at: None, size, borderless: false };
            let Some(r) = pick_rough_roi(&img, "Select Board ROI - drag, Enter = OK, Esc = cancel", &place, None)? else {
                println!("[取消] 未手動框選棋盤。");
                return Ok(None);
            };
            println!("[ROI 手動框選] 取得大略範圍: X={}, Y={}, W={}, H={}", r.x, r.y, r.w, r.h);
            r
        } else {
            args.rough.unwrap_or(Roi { x: 0, y: 0, w: img.w as i64, h: img.h as i64 })
        };
        refine_and_crop(&img, rough)
    };
    println!("[2/6] Board ROI: X={}, Y={}, W={}, H={}", roi.x, roi.y, roi.w, roi.h);
    Ok(Some((board, roi)))
}

/// Rescale every template image in `dir` by (fx, fy). Returns how many files were rewritten.
fn rescale_templates(dir: &Path, fx: f64, fy: f64) -> Result<usize> {
    let mut n = 0;
    for entry in std::fs::read_dir(dir)?.flatten() {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()) != Some("png".into()) {
            continue;
        }
        let Ok(img) = image::open(&path) else { continue };
        let img = Image::from_rgb(&img.to_rgb8());
        let nw = ((img.w as f64 * fx).round() as usize).max(8);
        let nh = ((img.h as f64 * fy).round() as usize).max(8);
        if (nw, nh) == (img.w, img.h) {
            continue;
        }
        // area averaging when shrinking (no aliasing), smooth interpolation when enlarging
        let out = if nw <= img.w && nh <= img.h { imgproc::resize_area(&img, nw, nh) } else { resize(&img, nw, nh) };
        std::fs::write(&path, export::png_bytes(&out))?;
        n += 1;
    }
    Ok(n)
}

/// Re-aim an existing profile after the game window was moved / scaled.
/// Rows and columns stay exactly as saved; the frame position and the cell size are re-fitted to
/// the grid lines on screen near the user's rectangle, then the template images are rescaled to
/// the new cell size. Everything else in the profile (rules, merged / obstacle classes) is kept.
fn run_reroi(args: &Args) -> Result<()> {
    use serde_json::{json, Value};
    println!("{}", "=".repeat(60));
    println!("      [重新對準既有 profile（視窗移動／縮放後）]");
    println!("{}", "=".repeat(60));
    if !args.profile.is_file() {
        bail!("找不到既有設定檔 {:?}。請先建立 profile。", args.profile);
    }
    let mut prof: Value = serde_json::from_slice(&std::fs::read(&args.profile)?)?;
    let b = &prof["board"];
    let rows = b["rows"].as_i64().context("profile 缺少 board.rows")? as usize;
    let cols = b["cols"].as_i64().context("profile 缺少 board.cols")? as usize;
    let g = |k: &str| b["roi"][k].as_i64().with_context(|| format!("profile 缺少 board.roi.{k}"));
    let old = Roi { x: g("x")?, y: g("y")?, w: g("width")?, h: g("height")? };
    let (ocw, och) = (old.w as f64 / cols as f64, old.h as f64 / rows as f64);
    println!("[保留] 網格 {} 列 x {} 欄；目前 ROI: X={}, Y={}, W={}, H={}（格 {:.2} x {:.2}）", rows, cols, old.x, old.y, old.w, old.h, ocw, och);

    let mons = capture::monitors();
    if mons.is_empty() {
        bail!("no monitor found");
    }
    // default to the monitor holding the old ROI's centre
    let (cx, cy) = (old.x + old.w / 2, old.y + old.h / 2);
    let holder = mons
        .iter()
        .find(|m| cx >= m.x as i64 && cx < m.x as i64 + m.w as i64 && cy >= m.y as i64 && cy < m.y as i64 + m.h as i64)
        .cloned();
    // The game may have been moved to ANOTHER monitor: always let the user pick the current one
    // (default = where the saved ROI was).
    let def = holder.as_ref().map(|m| m.index).unwrap_or(1);
    let mon = choose_monitor(args, &mons, def)?;
    println!("[擷取] 顯示器 {}: {}x{} 實體像素, 縮放 {}%", mon.device, mon.w, mon.h, mon.scale_percent);
    let screen = capture::capture_rect(mon.x, mon.y, mon.w, mon.h)?;
    println!("紅黃框是舊位置。請拖到新的棋盤上，大小只需大略框住（可比棋盤外緣大不超過 1 個磚塊），");
    println!("程式會依原本的 {} 列 x {} 欄自動算出精準外框與新的格子大小。", rows, cols);
    print_selection_help();
    let on_this = holder.as_ref().map(|m| m.index == mon.index).unwrap_or(false);
    let initial = if on_this {
        Roi { x: old.x - mon.x as i64, y: old.y - mon.y as i64, w: old.w, h: old.h }
    } else {
        // different monitor: same physical size corrected for the DPI difference, centred as a starting point
        let from = holder.as_ref().map(|m| m.scale_percent as f64).unwrap_or(mon.scale_percent as f64);
        let k = mon.scale_percent as f64 / from.max(1.0);
        let (w, h) = (((old.w as f64 * k) as i64).min(mon.w as i64 - 20), ((old.h as f64 * k) as i64).min(mon.h as i64 - 20));
        println!("[提示] 舊 ROI 不在這個螢幕上，已用相近大小（縮放 x{:.2}）放在畫面中央，請拖到棋盤上。", k);
        Roi { x: (mon.w as i64 - w) / 2, y: (mon.h as i64 - h) / 2, w, h }
    };
    let place = select::Placement {
        at: Some((mon.x as isize, mon.y as isize)),
        size: (mon.w as usize, mon.h as usize),
        borderless: true,
    };
    let Some(rough) = pick_rough_roi(&screen, "Re-aim ROI - move / resize roughly, Enter = OK, Esc = cancel", &place, Some(initial))? else {
        println!("[取消] 未更動 profile。");
        return Ok(());
    };
    println!("[大略範圍] X={}, Y={}, W={}, H={}", rough.x, rough.y, rough.w, rough.h);

    let fit = calibrate::fit_known_grid(&screen, rough, rows, cols);
    println!("[格線貼合] 第一步（依格線）: X={}, Y={}, W={}, H={}（可信度 {:.0}%）", fit.roi.x, fit.roi.y, fit.roi.w, fit.roi.h, fit.confidence * 100.0);
    if fit.confidence < 0.35 {
        println!("[警告] 貼合可信度偏低：畫面上的格線不明顯，或框選位置離棋盤太遠。建議重新執行並把框拖得更貼近棋盤。");
    }

    // template folder + the existing object library (loaded BEFORE rescaling: it is the reference for "does this tile match")
    let tdir = {
        let rel = prof["matcher"]["template_dir"].as_str().unwrap_or("templates").to_string();
        let base = args.profile.parent().filter(|p| !p.as_os_str().is_empty()).map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
        let cand = base.join(&rel);
        if cand.is_dir() { cand } else { args.templates.clone() }
    };
    let pdir = std::fs::canonicalize(&args.profile).ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(|| PathBuf::from("."));
    let library = if tdir.is_dir() { augment::load_library(&prof, &pdir, &tdir).ok().map(|l| l.0) } else { None };

    // second step: choose, among small shifts / size changes, the frame under which the tiles match the known objects best
    let mut froi = fit.roi;
    if let Some(lib) = &library {
        froi = augment::refine_frame(&screen, fit.roi, rows, cols, lib, fit.confidence >= 0.6);
        println!("[比對微調] 以磚塊比對結果微調: X={}, Y={}, W={}, H={}", froi.x, froi.y, froi.w, froi.h);
    }
    let nroi = Roi { x: mon.x as i64 + froi.x, y: mon.y as i64 + froi.y, w: froi.w, h: froi.h };
    let (ncw, nch) = (froi.w as f64 / cols as f64, froi.h as f64 / rows as f64);
    println!("[最終] ROI: X={}, Y={}, W={}, H={}（格 {:.2} x {:.2}）", nroi.x, nroi.y, nroi.w, nroi.h, ncw, nch);
    let (fx, fy) = (ncw / ocw, nch / och);
    println!("[縮放比例] 水平 x{:.3}，垂直 x{:.3}（相對原本的格子大小）", fx, fy);

    // preview crop of the fitted board so the user can check it
    let board = screen.crop(froi.x as usize, froi.y as usize, froi.w as usize, froi.h as usize);
    std::fs::write(&args.screenshot, export::png_bytes(&board))?;
    println!("[儲存] 貼合後的盤面截圖存至 {:?}（可開啟確認外框是否正確）", args.screenshot);

    let r2 = |v: f64| (v * 100.0).round() / 100.0;
    let patch = |v: &mut Value| {
        let b = &mut v["board"];
        b["roi"] = json!({"x": nroi.x, "y": nroi.y, "width": nroi.w, "height": nroi.h});
        b["cell_size"] = json!({"w": r2(ncw), "h": r2(nch)});
        b["offset"] = json!({"x": nroi.x, "y": nroi.y});
    };

    // template images: scale to the new cell size, then overwrite known objects with samples of the live board
    if tdir.is_dir() {
        let n = rescale_templates(&tdir, fx, fy)?;
        println!("[模板] 已依比例縮放 {} 張參照圖：{:?}", n, tdir);
        if let Some(lib) = &library {
            let rs = augment::resample_templates(&board, rows, cols, lib, &tdir)?;
            println!(
                "[即時取樣] 磚塊 {} 格中有 {} 格比對到既有物件（平均相似度 {:.2}）；{} 種物件的參照圖已換成目前畫面的樣本",
                rs.tiles, rs.known, rs.avg_sim, rs.updated_ids.len()
            );
            if rs.tiles > 0 && (rs.known as f64) < 0.8 * rs.tiles as f64 {
                println!("[警告] 只有 {:.0}% 的磚塊比對得到既有物件：外框可能沒對準棋盤，或這個關卡出現新物件（可再用「追加截圖」補入）。", 100.0 * rs.known as f64 / rs.tiles as f64);
            }
        }
    } else {
        println!("[模板] 找不到模板資料夾 {:?}，略過縮放。", tdir);
    }

    patch(&mut prof);
    std::fs::write(&args.profile, serde_json::to_string_pretty(&prof)?)?;
    println!("[儲存] 已更新 {:?}（列／欄數與其餘設定不變）", args.profile);

    // keep the editable preview in sync, otherwise its "download profile" button would bring the old ROI back
    if args.preview.is_file() {
        let html = std::fs::read_to_string(&args.preview)?;
        if let (Some(a), Some(e)) = (html.find("const D = "), html.find(";\nconst KEY")) {
            let a = a + "const D = ".len();
            if a < e {
                if let Ok(mut d) = serde_json::from_str::<Value>(&html[a..e]) {
                    patch(&mut d["profile"]);
                    let out = format!("{}{}{}", &html[..a], serde_json::to_string(&d)?, &html[e..]);
                    std::fs::write(&args.preview, out)?;
                    println!("[同步] 已同步 {:?} 內的 ROI", args.preview);
                }
            }
        }
    }
    println!("{}", "=".repeat(60));
    println!("完成。engine 下次啟動就會使用新的 ROI 與縮放後的模板。");
    Ok(())
}

/// Open the settings page for the existing profile (timing / matcher / rules) in the editor server.
fn run_settings(args: &Args) -> Result<()> {
    println!("{}", "=".repeat(60));
    println!("      [設定頁面（時間／辨識／規則）]");
    println!("{}", "=".repeat(60));
    if !args.profile.is_file() {
        bail!("找不到既有設定檔 {:?}。請先用螢幕框選建立第一份 profile。", args.profile);
    }
    let prof: serde_json::Value = serde_json::from_slice(&std::fs::read(&args.profile)?).with_context(|| format!("{:?} 不是有效的 JSON", args.profile))?;
    // its own file: must not overwrite (or trigger a reload of) the board editor's preview.html
    let page = args.preview.with_file_name("settings_preview.html");
    // class list (images, merges, obstacles) rebuilt from the profile + template files
    let pdir = args.profile.parent().filter(|p| !p.as_os_str().is_empty()).map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let tdir = {
        let rel = prof["matcher"]["template_dir"].as_str().unwrap_or("templates").to_string();
        let cand = pdir.join(&rel);
        if cand.is_dir() { cand } else { args.templates.clone() }
    };
    let (tpls, merge, obstacle, generic) = augment::editor_classes(&prof, &pdir, &tdir);
    export::write_settings_html(&prof, tpls, merge, obstacle, generic, &page)?;
    let cfg = serve::Cfg { preview: page.clone(), profile: args.profile.clone(), templates: args.templates.clone() };
    let res = serve::serve(&cfg);
    // the page is a throw-away: remove it when this run's server stops (not if another server still uses it)
    if !matches!(res, Ok(false)) {
        let _ = std::fs::remove_file(&page);
    }
    res.map(|_| ())
}

fn run_snip(args: &Args) -> Result<()> {
    if let Some((board, phys)) = pick_board_from_screen(args)? {
        process_board(&board, phys, args)?;
        finish(args, true);
    }
    Ok(())
}

fn run_image(args: &Args) -> Result<()> {
    if let Some((board, roi)) = pick_board_from_image(args)? {
        process_board(&board, roi, args)?;
        finish(args, args.select);
    }
    Ok(())
}

/// Add objects from another screenshot to the existing profile / templates.
fn run_add(args: &Args) -> Result<()> {
    println!("{}", "=".repeat(60));
    println!("      [追加截圖（補充新關卡出現的新物件）]");
    println!("{}", "=".repeat(60));
    if !args.profile.is_file() {
        bail!("找不到既有設定檔 {:?}。請先用螢幕框選或讀取截圖建立第一份 profile 與 templates。", args.profile);
    }
    let picked = if args.snip { pick_board_from_screen(args)? } else { pick_board_from_image(args)? };
    let Some((board, _)) = picked else { return Ok(()) };

    let sum = augment::augment_from_board(&board, &args.profile, &args.templates, &args.preview)?;
    println!("{}", "=".repeat(60));
    println!("本張截圖：{} 列 x {} 欄，磚塊 {} 格", sum.rows, sum.cols, sum.tiles);
    println!("  已知物件：{} 格（沿用原本編號）", sum.known);
    if sum.new_ids.is_empty() {
        println!("  沒有發現新物件，模板庫維持不變。");
    } else {
        println!("  新增物件：{} 種", sum.new_ids.len());
        let tdir = args.templates.to_string_lossy().replace('\\', "/");
        for (id, n) in sum.new_ids.iter().zip(&sum.new_counts) {
            println!("    - 新編號 {}: 本張出現 {} 格 -> {}/{}.png", id, n, tdir, id);
        }
        println!("已更新 {:?}；請開啟 {:?} 檢查，若新物件被誤分成多類，可在網頁中合併。", args.profile, args.preview);
    }
    println!("{}", "=".repeat(60));
    finish(args, args.snip || args.select);
    Ok(())
}

/// Opens the selection window for a moment (no input needed) to prove the GUI stack works.
fn self_test_window() -> Result<()> {
    let mons = capture::monitors();
    for m in &mons {
        println!("monitor {} {} {}x{} at ({},{}) scale {}%{}", m.index, m.device, m.w, m.h, m.x, m.y, m.scale_percent, if m.primary { " primary" } else { "" });
    }
    if let Some(m) = mons.first() {
        let cap = capture::capture_rect(m.x, m.y, m.w.min(400), m.h.min(300))?;
        let distinct: std::collections::HashSet<[u8; 3]> = (0..cap.w * cap.h).map(|i| [cap.data[i * 3], cap.data[i * 3 + 1], cap.data[i * 3 + 2]]).collect();
        println!("capture {}x{}: {} distinct colours (>1 means a real picture)", cap.w, cap.h, distinct.len());
    }
    let img = Image { w: 320, h: 200, data: vec![90; 320 * 200 * 3] };
    let mut win = minifb::Window::new("studio-rs self test", 320, 200, minifb::WindowOptions::default())?;
    let buf = vec![0x2040A0u32; 320 * 200];
    let t0 = std::time::Instant::now();
    let mut frames = 0;
    while win.is_open() && t0.elapsed().as_millis() < 700 {
        win.update_with_buffer(&buf, 320, 200)?;
        frames += 1;
    }
    println!("window opened, size={:?}, frames={}, image={}x{}", win.get_size(), frames, img.w, img.h);
    Ok(())
}

fn wait_for_enter() {
    print!("
按 [Enter] 關閉視窗...");
    std::io::stdout().flush().ok();
    let mut l = String::new();
    let _ = std::io::stdin().read_line(&mut l);
}

fn run() -> Result<()> {
    let base = parse_args()?;
    if !base.interactive {
        return run_mode(&base);
    }
    // menu mode: after every action (including the editor server stopping with Enter) come back to the menu
    use std::sync::atomic::Ordering;
    loop {
        clear_screen();
        let mut args = base.clone();
        if !menu(&mut args)? {
            std::process::exit(0); // the user chose to quit: no extra "press Enter" pause
        }
        serve::STOPPED_BY_ENTER.store(false, Ordering::SeqCst);
        let failed = match run_mode(&args) {
            Ok(()) => false,
            Err(e) => {
                eprintln!("\n[錯誤] {e:#}");
                true
            }
        };
        // The editor server already waited for Enter. Anything else (errors, cancel, re-aim summary,
        // warnings) would be wiped by the clear, so let the user read it first.
        if failed || !serve::STOPPED_BY_ENTER.load(Ordering::SeqCst) {
            print!("\n按 [Enter] 回到主選單...");
            std::io::stdout().flush().ok();
            let mut l = String::new();
            if std::io::stdin().read_line(&mut l)? == 0 {
                return Ok(());
            }
        }
    }
}

/// Wipe the console so a long session does not pile up output (no-op when output is redirected).
fn clear_screen() {
    use std::io::IsTerminal;
    if std::io::stdout().is_terminal() {
        let _ = std::process::Command::new("cmd").args(["/C", "cls"]).status();
    }
}

fn run_mode(args: &Args) -> Result<()> {
    if args.selftest {
        return self_test_window();
    }
    if args.settings {
        run_settings(args)
    } else if args.reroi {
        run_reroi(&args)
    } else if args.add {
        run_add(&args)
    } else if args.snip {
        run_snip(&args)
    } else {
        run_image(&args)
    }
}

fn main() {
    capture::enable_dpi_awareness();
    let own_console = capture::owns_console();
    // panic = "abort": the hook still runs first, so a crash is shown instead of the window vanishing.
    std::panic::set_hook(Box::new(move |info| {
        eprintln!("
[程式錯誤] {info}");
        if own_console {
            wait_for_enter();
        }
    }));
    let result = run();
    let code = match &result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("
[錯誤] {e:?}");
            1
        }
    };
    if own_console {
        wait_for_enter();
    }
    std::process::exit(code);
}

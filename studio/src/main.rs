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

use anyhow::{bail, Context, Result};
use calibrate::{estimate_adaptive_grid, refine_board_roi, slice_cells, Roi};
use export::{ExportCell, ExportTemplate};
use features::{classify_cells, CellType};
use imgproc::{resize, Image};
use std::io::Write;
use std::path::{Path, PathBuf};

struct Args {
    image: Option<PathBuf>,
    snip: bool,
    add: bool,
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
        image: None,
        snip: false,
        add: false,
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
    };
    let mut i = 1;
    while i < a.len() {
        let next = |i: usize| a.get(i + 1).cloned().context("missing value");
        match a[i].as_str() {
            "--image" | "-i" => { args.image = Some(PathBuf::from(next(i)?)); i += 1; }
            "--snip" | "-s" => args.snip = true,
            "--add" => args.add = true,
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
            "--roi" => { args.roi = Some(parse_roi(&a, i)?); i += 4; }
            "--rough-roi" => { args.rough = Some(parse_roi(&a, i)?); i += 4; }
            "-h" | "--help" => {
                println!("usage:\n  studio-rs --add (--snip | --image FILE [--select])   (add new objects to an existing profile)\n  studio-rs --snip [--monitor N]\n  studio-rs --image FILE --select\n  studio-rs --image FILE [--rough-roi X Y W H | --roi X Y W H]\n\ncommon: [--templates DIR] [--profile FILE] [--preview FILE] [--screenshot FILE] [--dump FILE] [--open | --no-browser]");
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
        i += 1;
    }
    if !args.snip && !args.selftest && args.image.is_none() {
        menu(&mut args)?;
    }
    Ok(args)
}

/// Interactive mode picker (used when the exe is started without arguments).
fn menu(args: &mut Args) -> Result<()> {
    println!("{}", "=".repeat(60));
    println!("       LianLianKan Studio 盤面分析與特徵建模工具");
    println!("{}", "=".repeat(60));
    println!("請選擇分析模式：");
    println!("  [1] 螢幕框選截圖 (滑鼠手動框選棋盤，框選範圍超出棋盤勿大於 1 個磚塊)");
    println!("  [2] 讀取現有截圖檔案 (支援拖曳圖片，並手動框選棋盤)");
    println!("  [3] 追加截圖 (新關卡出現新物件時，把新物件補進現有模板庫)");
    println!("{}", "=".repeat(60));
    print!("請選擇模式 [1/2/3] (預設 1): ");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let choice = line.trim().to_string();
    if choice == "3" {
        args.add = true;
        println!("新截圖來源：  [1] 螢幕框選   [2] 讀取截圖檔案");
        print!("請選擇 [1/2] (預設 1): ");
        std::io::stdout().flush().ok();
        let mut l2 = String::new();
        std::io::stdin().read_line(&mut l2)?;
        if l2.trim() != "2" {
            args.snip = true;
            return Ok(());
        }
    }
    if choice == "2" || choice == "3" {
        loop {
            print!("請輸入或拖曳圖片路徑: ");
            std::io::stdout().flush().ok();
            let mut p = String::new();
            if std::io::stdin().read_line(&mut p)? == 0 {
                bail!("input closed");
            }
            let p = p.trim().trim_matches('"').trim_matches('\'').to_string();
            if Path::new(&p).is_file() {
                args.image = Some(PathBuf::from(p));
                args.select = true;
                return Ok(());
            }
            println!("[錯誤] 找不到檔案: {p}，請重新輸入！");
        }
    }
    args.snip = true;
    Ok(())
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
    println!(" 2. 框選完畢按【Enter】或【Space】確認；【Esc】取消");
    println!(" 3. 框選範圍超出棋盤外緣請勿大於 1 個磚塊");
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
        println!("[Preview] 開啟編輯頁面: {:?}", args.preview);
        open_in_browser(&args.preview);
    }
}

fn choose_monitor(args: &Args, mons: &[capture::Monitor]) -> Result<capture::Monitor> {
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
        print!("請選擇遊戲所在的螢幕 [1-{}] (預設 1): ", mons.len());
        std::io::stdout().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        let n: usize = line.trim().parse().unwrap_or(1);
        return Ok(mons.iter().find(|m| m.index == n).unwrap_or(&mons[0]).clone());
    }
    Ok(mons[0].clone())
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
    let mon = choose_monitor(args, &mons)?;
    println!("[擷取] 顯示器 {}: {}x{} 實體像素, 縮放 {}%", mon.device, mon.w, mon.h, mon.scale_percent);
    let screen = capture::capture_rect(mon.x, mon.y, mon.w, mon.h)?;

    print_selection_help();
    let place = select::Placement { at: Some((mon.x as isize, mon.y as isize)), size: (mon.w as usize, mon.h as usize), borderless: true };
    let rough = match select::select_roi(&screen, "Screen Snip - drag to select the board, Enter = OK, Esc = cancel", &place)? {
        Some(r) if r.w > 20 && r.h > 20 => r,
        _ => {
            println!("[取消] 未手動框選棋盤，已結束。");
            return Ok(None);
        }
    };
    println!("[手動框選] 取得大略範圍: X={}, Y={}, 寬={}, 高={}", rough.x, rough.y, rough.w, rough.h);
    let roi = refine_board_roi(&screen, rough);
    println!("[網格對齊] 已自動校準至物理網格晶格: X={}, Y={}, 寬={}, 高={}", roi.x, roi.y, roi.w, roi.h);

    let board = screen.crop(roi.x as usize, roi.y as usize, roi.w as usize, roi.h as usize);
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

    let roi = if let Some(r) = args.roi {
        r
    } else {
        let rough = if args.select {
            print_selection_help();
            // fit the image on the primary monitor (the window shows it scaled; coordinates map back)
            let mons = capture::monitors();
            let (mw, mh) = mons.first().map(|m| (m.w as f64, m.h as f64)).unwrap_or((1280.0, 720.0));
            let sc = (mw * 0.9 / img.w as f64).min(mh * 0.9 / img.h as f64).min(1.0);
            let size = (((img.w as f64 * sc) as usize).max(100), ((img.h as f64 * sc) as usize).max(100));
            let place = select::Placement { at: None, size, borderless: false };
            match select::select_roi(&img, "Select Board ROI - drag, Enter = OK, Esc = cancel", &place)? {
                Some(r) if r.w > 20 && r.h > 20 => {
                    println!("[ROI 手動框選] 取得大略範圍: X={}, Y={}, W={}, H={}", r.x, r.y, r.w, r.h);
                    r
                }
                _ => {
                    println!("[取消] 未手動框選棋盤。");
                    return Ok(None);
                }
            }
        } else {
            args.rough.unwrap_or(Roi { x: 0, y: 0, w: img.w as i64, h: img.h as i64 })
        };
        refine_board_roi(&img, rough)
    };
    println!("[2/6] Board ROI: X={}, Y={}, W={}, H={}", roi.x, roi.y, roi.w, roi.h);
    Ok(Some((img.crop(roi.x as usize, roi.y as usize, roi.w as usize, roi.h as usize), roi)))
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
    let args = parse_args()?;
    if args.selftest {
        return self_test_window();
    }
    if args.add {
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

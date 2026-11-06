use anyhow::{Context, Result};
use engine::{
    create_capturer, Controller, GameProfile, Matcher, MatcherConfig, Solver, SolverConfig, EMPTY,
    Point,
};
use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

#[cfg(windows)]
use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;

const VK_ESCAPE: i32 = 0x1B;
const VK_F8: i32 = 0x77;

fn check_key_pressed(vk: i32) -> bool {
    #[cfg(windows)]
    unsafe {
        // High-order bit is 1 if key is down
        (GetAsyncKeyState(vk) as u16 & 0x8000) != 0
    }
    #[cfg(not(windows))]
    false
}

struct EngineArgs {
    profile_path: PathBuf,
    templates_dir: Option<PathBuf>,
    dry_run: bool,
    no_prompt: bool,
    max_moves: Option<usize>,
    move_delay_ms: u64,
}

fn parse_args() -> EngineArgs {
    let args: Vec<String> = env::args().collect();
    let mut profile_path = PathBuf::from("game_profile.json");
    let mut templates_dir = None;
    let mut dry_run = false;
    let mut no_prompt = false;
    let mut max_moves = None;
    let mut move_delay_ms = 50;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" | "-p" => {
                if i + 1 < args.len() {
                    profile_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--templates" | "-t" => {
                if i + 1 < args.len() {
                    templates_dir = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--dry-run" => {
                dry_run = true;
            }
            "--no-prompt" => {
                no_prompt = true;
            }
            "--max-moves" => {
                if i + 1 < args.len() {
                    max_moves = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--delay" => {
                if i + 1 < args.len() {
                    move_delay_ms = args[i + 1].parse().unwrap_or(50);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    EngineArgs {
        profile_path,
        templates_dir,
        dry_run,
        no_prompt,
        max_moves,
        move_delay_ms,
    }
}

fn create_sample_profile_if_missing(path: &Path) -> Result<()> {
    if !path.exists() {
        println!("[Init] Creating default template profile: {:?}", path);
        let sample = GameProfile {
            name: "LianLianKan Standard".to_string(),
            board: engine::profile::BoardConfig {
                rows: 10,
                cols: 14,
                roi: engine::profile::RoiConfig {
                    x: 200,
                    y: 150,
                    width: 700,
                    height: 500,
                },
            },
            matcher: engine::profile::MatcherProfile {
                threshold: 0.75,
                empty_variance_threshold: 15.0,
                template_dir: "templates".to_string(),
            },
            timing: engine::profile::TimingConfig {
                click_interval_ms: 30,
                animation_delay_ms: 100,
                verify_timeout_ms: 500,
            },
            rules: Default::default(),
        };
        sample.save_to_file(path)?;
    }
    Ok(())
}

/// Block until the user presses Enter (keeps the console window open at the end).
fn wait_for_enter(msg: &str) {
    println!("
{}", msg);
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);
}

/// Wait (global hotkey, works while the game window has focus) for F8 to start.
/// Returns false if the user pressed ESC to cancel.
fn wait_for_start() -> bool {
    println!();
    println!("────────────────────────────────────────────────────────");
    println!(" 準備完成，尚未開始。");
    println!(" 請先切換到遊戲視窗，確認棋盤完整顯示、沒有被本視窗擋住，");
    println!(" 然後按 [F8] 開始（按 [ESC] 取消）。");
    println!("────────────────────────────────────────────────────────");
    // Ignore keys that are still held down from before.
    while check_key_pressed(VK_F8) || check_key_pressed(VK_ESCAPE) {
        sleep(Duration::from_millis(20));
    }
    loop {
        if check_key_pressed(VK_ESCAPE) {
            return false;
        }
        if check_key_pressed(VK_F8) {
            // Release first, otherwise the main loop would read the same press as "pause".
            while check_key_pressed(VK_F8) {
                sleep(Duration::from_millis(20));
            }
            println!("[Engine] 開始執行！（F8 暫停/繼續，ESC 結束）");
            sleep(Duration::from_millis(300));
            return true;
        }
        sleep(Duration::from_millis(30));
    }
}

fn main() {
    let args = parse_args();
    let result = run(&args);
    let code = match &result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("
[Engine] 發生錯誤: {:?}", e);
            1
        }
    };
    if !args.no_prompt {
        wait_for_enter("引擎已結束。請確認上方訊息後，按 [Enter] 關閉視窗...");
    }
    std::process::exit(code);
}

fn run(args: &EngineArgs) -> Result<()> {
    println!("========================================================");
    println!(" LianLianKan (連連看) Ultra-Fast Visual Solver Engine  ");
    println!(" Hotkeys: [F8] Start/Pause/Resume | [ESC] Exit        ");
    println!("========================================================");

    create_sample_profile_if_missing(&args.profile_path)?;
    let profile = GameProfile::load_from_file(&args.profile_path)
        .with_context(|| format!("Failed to load profile {:?}", args.profile_path))?;

    let tmpl_dir = args
        .templates_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(&profile.matcher.template_dir));

    let mut matcher = Matcher::new(MatcherConfig {
        threshold: profile.matcher.threshold,
        empty_variance_threshold: profile.matcher.empty_variance_threshold,
        ..Default::default()
    });

    if tmpl_dir.exists() {
        let count = matcher.load_templates_from_dir(&tmpl_dir)?;
        println!(
            "[Matcher] Loaded {} template(s) from {:?}",
            count, tmpl_dir
        );
    } else {
        println!(
            "[Matcher] Warning: Template dir {:?} not found. Please provide templates.",
            tmpl_dir
        );
    }

    let solver = Solver::with_config(SolverConfig {
        allow_outer_path: profile.rules.allow_outer_path,
        ..Default::default()
    });
    println!(
        "[Rules] Outer-ring paths: {}",
        if profile.rules.allow_outer_path { "ALLOWED" } else { "BLOCKED (inside board only)" }
    );
    let controller = Controller::new(profile.timing.click_interval_ms);
    let mut capturer = create_capturer();

    println!(
        "[Engine] Board size: {}x{}, ROI: ({}, {}, {}x{})",
        profile.board.rows,
        profile.board.cols,
        profile.board.roi.x,
        profile.board.roi.y,
        profile.board.roi.width,
        profile.board.roi.height
    );
    if args.dry_run {
        println!("[Engine] Running in DRY-RUN mode (no mouse clicks injected)");
    }

    if !args.no_prompt && !wait_for_start() {
        println!("[Engine] 已取消，未開始執行。");
        return Ok(());
    }

    let mut paused = false;
    let mut total_moves = 0;
    let start_time = Instant::now();
    // Pairs that were proposed but could not be executed. Cleared whenever the board changes,
    // so the engine never keeps repeating the same impossible step.
    let mut failed_pairs: HashSet<(Point, Point)> = HashSet::new();
    let mut last_signature: Vec<i32> = Vec::new();
    let mut last_unknown_report: Vec<Point> = Vec::new();

    'game_loop: loop {
        // Check hotkeys
        if check_key_pressed(VK_ESCAPE) {
            println!("\n[Engine] ESC detected. Terminating engine.");
            break 'game_loop;
        }

        if check_key_pressed(VK_F8) {
            paused = !paused;
            println!(
                "\n[Engine] F8 pressed: Engine is now {}",
                if paused { "PAUSED (press F8 again to resume)" } else { "RESUMED" }
            );
            sleep(Duration::from_millis(300)); // Debounce
        }

        if paused {
            sleep(Duration::from_millis(50));
            continue;
        }

        if let Some(limit) = args.max_moves {
            if total_moves >= limit {
                println!("[Engine] Reached max moves limit ({})", limit);
                break 'game_loop;
            }
        }

        // Capture screen ROI
        let roi_img = match capturer.capture_roi(
            profile.board.roi.x,
            profile.board.roi.y,
            profile.board.roi.width,
            profile.board.roi.height,
        ) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("[Capture] Error capturing ROI: {:?}", e);
                sleep(Duration::from_millis(100));
                continue;
            }
        };

        // Convert cropped ROI into board matrix
        let mut board = match matcher.match_board_image(
            &roi_img,
            profile.board.rows,
            profile.board.cols,
        ) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[Matcher] Error matrixizing board: {:?}", e);
                sleep(Duration::from_millis(100));
                continue;
            }
        };

        // Board changed since last scan -> old failures no longer apply.
        if board.cells != last_signature {
            failed_pairs.clear();
            last_signature = board.cells.clone();
        }

        // Cells that are neither blank nor a known tile: treated as blockers, never as empty.
        let unknown = board.unknown_points();
        if unknown != last_unknown_report {
            if !unknown.is_empty() {
                println!(
                    "[Matcher] {} unrecognised cell(s) treated as blocked: {:?} (lower matcher.threshold or re-run studio if this persists)",
                    unknown.len(),
                    unknown.iter().map(|p| (p.row, p.col)).collect::<Vec<_>>()
                );
            }
            last_unknown_report = unknown.clone();
        }

        let remaining = board.remaining_tiles_count();
        if remaining == 0 && !unknown.is_empty() {
            // Something is still on the board that we cannot read: re-scan instead of finishing.
            sleep(Duration::from_millis(200));
            continue;
        }
        if remaining == 0 {
            println!("\n🎉 [VICTORY] All tiles cleared in {:?}", start_time.elapsed());
            println!("Total moves executed: {}", total_moves);
            break 'game_loop;
        }

        // Find next best match that has not already failed on this exact board state
        let best_match = solver
            .find_all_matches(&board)
            .into_iter()
            .find(|c| !failed_pairs.contains(&(c.p1, c.p2)));

        // No solution on the real board: optionally try again pretending obstacle blocks are
        // not there (the game may still refuse; a failed pair is remembered and skipped).
        let mut best_match = best_match;
        if best_match.is_none() && profile.rules.ignore_obstacles_when_stuck {
            let relaxed = board.without_obstacles();
            if relaxed.cells != board.cells {
                best_match = solver
                    .find_all_matches(&relaxed)
                    .into_iter()
                    .find(|c| !failed_pairs.contains(&(c.p1, c.p2)));
                if let Some(m) = &best_match {
                    println!(
                        "[Fallback] No normal move. Trying to ignore obstacles: {} <-> {} (Tile #{})",
                        m.p1, m.p2, m.tile_id
                    );
                }
            }
        }

        // Re-check both target cells on a FRESH capture right before clicking
        // (board may have changed / animation / highlight since the scan).
        if let (Some(m), false) = (&best_match, args.dry_run) {
            let mut still_valid = true;
            for p in [m.p1, m.p2] {
                let (x, y, w, h) = Controller::calculate_cell_rect(&profile, p);
                match capturer.capture_roi(x, y, w, h) {
                    Ok(img) => {
                        let (id, _) = matcher.match_cell(&image::DynamicImage::ImageRgba8(img));
                        if id != m.tile_id {
                            still_valid = false;
                        }
                    }
                    Err(_) => still_valid = false,
                }
            }
            if !still_valid {
                println!(
                    "[Recheck] {} <-> {} no longer shows Tile #{}. Skipping and re-scanning.",
                    m.p1, m.p2, m.tile_id
                );
                failed_pairs.insert((m.p1, m.p2));
                sleep(Duration::from_millis(60));
                continue;
            }
        }

        match best_match {
            Some(m) => {
                println!(
                    "[Move #{:03}] Matched Tile #{:2} at {} <-> {} (Type: {:?}, Score: {}, Remaining: {})",
                    total_moves + 1,
                    m.tile_id,
                    m.p1,
                    m.p2,
                    m.match_type,
                    m.score,
                    remaining
                );

                if !args.dry_run {
                    match controller.execute_pair_and_verify(
                        &mut *capturer,
                        &matcher,
                        &profile,
                        m.p1,
                        m.p2,
                    ) {
                        Ok(true) => {
                            total_moves += 1;
                        }
                        Ok(false) => {
                            eprintln!(
                                "[Verification] Elimination check timeout at {} and {}. Skipping this pair and re-scanning...",
                                m.p1, m.p2
                            );
                            failed_pairs.insert((m.p1, m.p2));
                        }
                        Err(e) => {
                            eprintln!("[Controller] Click execution error: {:?}", e);
                            failed_pairs.insert((m.p1, m.p2));
                        }
                    }
                } else {
                    // In dry-run mode, simulate board modification locally
                    board.set_point(m.p1, EMPTY);
                    board.set_point(m.p2, EMPTY);
                    total_moves += 1;
                }

                sleep(Duration::from_millis(args.move_delay_ms));
            }
            None => {
                println!(
                    "[Engine] No valid 3-segment move found. Remaining tiles: {}. Waiting / Deadlock detected.",
                    remaining
                );
                // Allow previously failed pairs to be retried after a pause.
                failed_pairs.clear();
                sleep(Duration::from_millis(200));
            }
        }
    }

    println!("[Engine] Finished. Elapsed: {:?}", start_time.elapsed());
    Ok(())
}

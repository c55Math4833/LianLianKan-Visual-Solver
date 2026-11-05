use anyhow::{Context, Result};
use engine::{
    create_capturer, Controller, GameProfile, Matcher, MatcherConfig, Solver, EMPTY,
};
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
    max_moves: Option<usize>,
    move_delay_ms: u64,
}

fn parse_args() -> EngineArgs {
    let args: Vec<String> = env::args().collect();
    let mut profile_path = PathBuf::from("game_profile.json");
    let mut templates_dir = None;
    let mut dry_run = false;
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
                threshold: 0.82,
                empty_variance_threshold: 15.0,
                template_dir: "templates".to_string(),
            },
            timing: engine::profile::TimingConfig {
                click_interval_ms: 30,
                animation_delay_ms: 100,
                verify_timeout_ms: 500,
            },
        };
        sample.save_to_file(path)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    println!("========================================================");
    println!(" LianLianKan (連連看) Ultra-Fast Visual Solver Engine  ");
    println!(" Hotkeys: [F8] Pause/Resume  |  [ESC] Exit Engine       ");
    println!("========================================================");

    let args = parse_args();

    create_sample_profile_if_missing(&args.profile_path)?;
    let profile = GameProfile::load_from_file(&args.profile_path)
        .with_context(|| format!("Failed to load profile {:?}", args.profile_path))?;

    let tmpl_dir = args
        .templates_dir
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

    let solver = Solver::new();
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

    let mut paused = false;
    let mut total_moves = 0;
    let start_time = Instant::now();
    let mut consecutive_failures = 0;

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
                if paused { "PAUSED" } else { "RESUMED" }
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

        let remaining = board.remaining_tiles_count();
        if remaining == 0 {
            println!("\n🎉 [VICTORY] All tiles cleared in {:?}", start_time.elapsed());
            println!("Total moves executed: {}", total_moves);
            break 'game_loop;
        }

        // Find next best match
        let best_match = solver.find_best_match(&board);

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
                            consecutive_failures = 0;
                        }
                        Ok(false) => {
                            eprintln!(
                                "[Verification] Elimination check timeout at {} and {}. Retrying...",
                                m.p1, m.p2
                            );
                            consecutive_failures += 1;
                        }
                        Err(e) => {
                            eprintln!("[Controller] Click execution error: {:?}", e);
                            consecutive_failures += 1;
                        }
                    }
                } else {
                    // In dry-run mode, simulate board modification locally
                    board.set_point(m.p1, EMPTY);
                    board.set_point(m.p2, EMPTY);
                    total_moves += 1;
                    consecutive_failures = 0;
                }

                if consecutive_failures >= 5 {
                    println!("[Engine] 5 consecutive execution failures. Pausing engine for safety.");
                    paused = true;
                    consecutive_failures = 0;
                }

                sleep(Duration::from_millis(args.move_delay_ms));
            }
            None => {
                println!(
                    "[Engine] No valid 3-segment move found. Remaining tiles: {}. Waiting / Deadlock detected.",
                    remaining
                );
                sleep(Duration::from_millis(200));
            }
        }
    }

    println!("[Engine] Finished. Elapsed: {:?}", start_time.elapsed());
    Ok(())
}

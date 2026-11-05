use engine::board::{Board, Point, OBSTACLE};
use engine::profile::GameProfile;
use engine::solver::{MatchType, Solver};
use std::time::Instant;

#[test]
fn test_complex_u_turn_outside_board() {
    // 4x4 board
    // Top-left (0, 0) and Top-right (0, 3) have Tile 1.
    // The cells in between (0, 1) and (0, 2) are filled with other tiles (blocked).
    let mut board = Board::new(4, 4);
    board.set(0, 0, 1);
    board.set(0, 1, 9);
    board.set(0, 2, 9);
    board.set(0, 3, 1);
    // Block interior row 1 so the solver is forced to route through the outer perimeter above row 0 (row -1)
    board.set(1, 0, OBSTACLE);
    board.set(1, 3, OBSTACLE);

    let solver = Solver::new();
    let m = solver.can_connect(&board, Point::new(0, 0), Point::new(0, 3));
    assert!(m.is_some(), "Should connect via outer margin above row 0");
    let m = m.unwrap();

    // Verify it connects via outer margin row -1
    assert_eq!(
        m.match_type,
        MatchType::TwoTurn(Point::new(-1, 0), Point::new(-1, 3))
    );
    assert!(m.path.contains(&Point::new(-1, 0)));
    assert!(m.path.contains(&Point::new(-1, 3)));
}

#[test]
fn test_vertical_u_turn_around_left_margin() {
    let mut board = Board::new(5, 5);
    // (0, 0) and (4, 0) have Tile 7
    // Intermediate cells (1, 0), (2, 0), (3, 0) blocked by obstacles
    board.set(0, 0, 7);
    board.set(1, 0, OBSTACLE);
    board.set(2, 0, OBSTACLE);
    board.set(3, 0, OBSTACLE);
    board.set(4, 0, 7);
    // Block interior column 1 so the solver is forced to route through the outer perimeter to the left (col -1)
    board.set(0, 1, OBSTACLE);
    board.set(4, 1, OBSTACLE);

    let solver = Solver::new();
    let m = solver.can_connect(&board, Point::new(0, 0), Point::new(4, 0));
    assert!(m.is_some());
    let m = m.unwrap();

    // Path goes around via col -1
    assert_eq!(
        m.match_type,
        MatchType::TwoTurn(Point::new(0, -1), Point::new(4, -1))
    );
}

#[test]
fn test_obstacle_blocks_2turn_interior_scan() {
    let mut board = Board::new(4, 4);
    board.set(1, 1, 2);
    board.set(2, 2, 2);

    // Block corners (1, 2) and (2, 1) -> eliminates 1-turn
    board.set(1, 2, 8);
    board.set(2, 1, 8);

    // Block column 0 and column 3 at row 1 and 2
    board.set(1, 0, OBSTACLE);
    board.set(2, 0, OBSTACLE);
    board.set(1, 3, OBSTACLE);
    board.set(2, 3, OBSTACLE);

    // Block row 0 and row 3 at col 1 and 2
    board.set(0, 1, OBSTACLE);
    board.set(0, 2, OBSTACLE);
    board.set(3, 1, OBSTACLE);
    board.set(3, 2, OBSTACLE);

    let solver = Solver::new();
    let m = solver.can_connect(&board, Point::new(1, 1), Point::new(2, 2));
    assert!(m.is_none(), "Enclosed tile pair should not find a valid <= 2-turn path");
}

#[test]
fn test_large_board_solver_performance() {
    // 20x30 board = 600 tiles
    let rows = 20;
    let cols = 30;
    let mut board = Board::new(rows, cols);

    // Populate with 15 pairs of 20 tile types
    let mut tile_id = 1;
    for r in 0..rows {
        for c in 0..cols {
            board.set(r as i32, c as i32, (tile_id % 20) + 1);
            tile_id += 1;
        }
    }

    let solver = Solver::new();
    let t0 = Instant::now();
    let matches = solver.find_all_matches(&board);
    let elapsed = t0.elapsed();

    println!(
        "Found {} valid moves on 20x30 board in {:?}",
        matches.len(),
        elapsed
    );
    // Even in unoptimized debug build, 600 tiles search should complete comfortably fast
    assert!(elapsed.as_millis() < 500, "Solver took too long: {:?}", elapsed);
}

#[test]
fn test_game_profile_serialization() {
    let profile_str = r#"{
        "name": "Test LLK",
        "board": {
            "rows": 8,
            "cols": 12,
            "roi": { "x": 100, "y": 200, "width": 640, "height": 480 }
        },
        "matcher": {
            "threshold": 0.85,
            "empty_variance_threshold": 12.0,
            "template_dir": "test_templates"
        },
        "timing": {
            "click_interval_ms": 25,
            "animation_delay_ms": 90,
            "verify_timeout_ms": 450
        }
    }"#;

    let profile: GameProfile = serde_json::from_str(profile_str).expect("Valid JSON");
    assert_eq!(profile.board.rows, 8);
    assert_eq!(profile.board.cols, 12);
    assert_eq!(profile.board.roi.width, 640);
    assert_eq!(profile.timing.animation_delay_ms, 90);
}

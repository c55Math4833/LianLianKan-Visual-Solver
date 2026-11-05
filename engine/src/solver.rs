use crate::board::{Board, PaddedBoard, Point};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchType {
    ZeroTurn,
    OneTurn(Point),
    TwoTurn(Point, Point),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchCandidate {
    pub p1: Point,
    pub p2: Point,
    pub tile_id: i32,
    pub match_type: MatchType,
    pub path: Vec<Point>,
    pub score: i64,
}

#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub distance_weight: i64,
    pub single_perimeter_bonus: i64,
    pub both_perimeter_bonus: i64,
    pub zero_turn_bonus: i64,
    pub one_turn_bonus: i64,
    pub two_turn_bonus: i64,
    pub outer_margin_bonus: i64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            distance_weight: 50,
            single_perimeter_bonus: 1000,
            both_perimeter_bonus: 2500,
            zero_turn_bonus: 300,
            one_turn_bonus: 150,
            two_turn_bonus: 50,
            outer_margin_bonus: 400,
        }
    }
}

pub struct Solver {
    pub config: SolverConfig,
}

impl Solver {
    pub fn new() -> Self {
        Self {
            config: SolverConfig::default(),
        }
    }

    pub fn with_config(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Evaluates if two points p1 and p2 can be connected on the given board.
    /// Original board coordinates are expected.
    pub fn can_connect(&self, board: &Board, p1: Point, p2: Point) -> Option<MatchCandidate> {
        if p1 == p2 {
            return None;
        }

        let val1 = board.get_point(p1)?;
        let val2 = board.get_point(p2)?;

        if val1 <= 0 || val1 != val2 {
            return None;
        }

        let padded = board.to_padded();
        let pp1 = PaddedBoard::to_padded_point(p1);
        let pp2 = PaddedBoard::to_padded_point(p2);

        // Try 0-turn, then 1-turn, then 2-turn
        let match_info = self
            .check_zero_turn(&padded, pp1, pp2)
            .or_else(|| self.check_one_turn(&padded, pp1, pp2))
            .or_else(|| self.check_two_turn(&padded, pp1, pp2))?;

        let (match_type, padded_path) = match_info;
        let orig_path: Vec<Point> = padded_path
            .into_iter()
            .map(PaddedBoard::to_orig_point)
            .collect();

        let score = self.calculate_score(board, p1, p2, &match_type, &orig_path);

        Some(MatchCandidate {
            p1,
            p2,
            tile_id: val1,
            match_type,
            path: orig_path,
            score,
        })
    }

    /// Finds all valid matches on the board, sorted by score descending (heuristic prioritized).
    pub fn find_all_matches(&self, board: &Board) -> Vec<MatchCandidate> {
        let grouped = board.group_by_tile_id();
        let mut candidates = Vec::new();

        for (&_tile_id, points) in grouped.iter() {
            let n = points.len();
            if n < 2 {
                continue;
            }
            for i in 0..n {
                for j in (i + 1)..n {
                    if let Some(candidate) = self.can_connect(board, points[i], points[j]) {
                        candidates.push(candidate);
                    }
                }
            }
        }

        // Sort descending by score (highest heuristic priority first)
        candidates.sort_by(|a, b| b.score.cmp(&a.score));
        candidates
    }

    /// Returns the single best match according to the heuristic.
    pub fn find_best_match(&self, board: &Board) -> Option<MatchCandidate> {
        let matches = self.find_all_matches(board);
        matches.into_iter().next()
    }

    // ------------------------------------------------------------------------
    // Internal Pathfinding Helpers on PaddedBoard
    // ------------------------------------------------------------------------

    /// Checks if horizontal line between (r, c1) and (r, c2) has all intermediate cells empty.
    #[inline]
    fn is_horizontal_clear(padded: &PaddedBoard, r: i32, c1: i32, c2: i32) -> bool {
        let min_c = c1.min(c2);
        let max_c = c1.max(c2);
        for c in (min_c + 1)..max_c {
            if !padded.is_empty(r, c) {
                return false;
            }
        }
        true
    }

    /// Checks if vertical line between (r1, c) and (r2, c) has all intermediate cells empty.
    #[inline]
    fn is_vertical_clear(padded: &PaddedBoard, c: i32, r1: i32, r2: i32) -> bool {
        let min_r = r1.min(r2);
        let max_r = r1.max(r2);
        for r in (min_r + 1)..max_r {
            if !padded.is_empty(r, c) {
                return false;
            }
        }
        true
    }

    /// Check 0-turn connection (straight line)
    fn check_zero_turn(
        &self,
        padded: &PaddedBoard,
        p1: Point,
        p2: Point,
    ) -> Option<(MatchType, Vec<Point>)> {
        if p1.row == p2.row {
            if Self::is_horizontal_clear(padded, p1.row, p1.col, p2.col) {
                return Some((MatchType::ZeroTurn, vec![p1, p2]));
            }
        } else if p1.col == p2.col {
            if Self::is_vertical_clear(padded, p1.col, p1.row, p2.row) {
                return Some((MatchType::ZeroTurn, vec![p1, p2]));
            }
        }
        None
    }

    /// Check 1-turn connection (L-shape)
    fn check_one_turn(
        &self,
        padded: &PaddedBoard,
        p1: Point,
        p2: Point,
    ) -> Option<(MatchType, Vec<Point>)> {
        if p1.row == p2.row || p1.col == p2.col {
            return None; // Straight lines are handled in 0-turn
        }

        // Corner 1: (p1.row, p2.col)
        let c1 = Point::new(p1.row, p2.col);
        if padded.is_empty(c1.row, c1.col)
            && Self::is_horizontal_clear(padded, p1.row, p1.col, c1.col)
            && Self::is_vertical_clear(padded, c1.col, c1.row, p2.row)
        {
            let orig_corner = PaddedBoard::to_orig_point(c1);
            return Some((MatchType::OneTurn(orig_corner), vec![p1, c1, p2]));
        }

        // Corner 2: (p2.row, p1.col)
        let c2 = Point::new(p2.row, p1.col);
        if padded.is_empty(c2.row, c2.col)
            && Self::is_vertical_clear(padded, p1.col, p1.row, c2.row)
            && Self::is_horizontal_clear(padded, c2.row, c2.col, p2.col)
        {
            let orig_corner = PaddedBoard::to_orig_point(c2);
            return Some((MatchType::OneTurn(orig_corner), vec![p1, c2, p2]));
        }

        None
    }

    /// Check 2-turn connection (Z-shape or U-shape) via scan lines
    fn check_two_turn(
        &self,
        padded: &PaddedBoard,
        p1: Point,
        p2: Point,
    ) -> Option<(MatchType, Vec<Point>)> {
        let max_r = padded.p_rows as i32;
        let max_c = padded.p_cols as i32;

        // Way 1: Scan horizontally from p1 to k1 = (p1.row, c), then vertically to k2 = (p2.row, c), then to p2
        // Scan right from p1.col + 1
        for c in (p1.col + 1)..max_c {
            if !padded.is_empty(p1.row, c) {
                break; // blocked in this direction
            }
            let k1 = Point::new(p1.row, c);
            let k2 = Point::new(p2.row, c);
            // k2 must be empty (unless k2 == p2, which would be 1-turn)
            if c != p2.col && padded.is_empty(k2.row, k2.col) {
                if Self::is_vertical_clear(padded, c, p1.row, p2.row)
                    && Self::is_horizontal_clear(padded, p2.row, c, p2.col)
                {
                    let orig_k1 = PaddedBoard::to_orig_point(k1);
                    let orig_k2 = PaddedBoard::to_orig_point(k2);
                    return Some((MatchType::TwoTurn(orig_k1, orig_k2), vec![p1, k1, k2, p2]));
                }
            }
        }

        // Scan left from p1.col - 1 down to 0
        for c in (0..p1.col).rev() {
            if !padded.is_empty(p1.row, c) {
                break; // blocked
            }
            let k1 = Point::new(p1.row, c);
            let k2 = Point::new(p2.row, c);
            if c != p2.col && padded.is_empty(k2.row, k2.col) {
                if Self::is_vertical_clear(padded, c, p1.row, p2.row)
                    && Self::is_horizontal_clear(padded, p2.row, c, p2.col)
                {
                    let orig_k1 = PaddedBoard::to_orig_point(k1);
                    let orig_k2 = PaddedBoard::to_orig_point(k2);
                    return Some((MatchType::TwoTurn(orig_k1, orig_k2), vec![p1, k1, k2, p2]));
                }
            }
        }

        // Way 2: Scan vertically from p1 to k1 = (r, p1.col), then horizontally to k2 = (r, p2.col), then to p2
        // Scan down from p1.row + 1
        for r in (p1.row + 1)..max_r {
            if !padded.is_empty(r, p1.col) {
                break; // blocked
            }
            let k1 = Point::new(r, p1.col);
            let k2 = Point::new(r, p2.col);
            if r != p2.row && padded.is_empty(k2.row, k2.col) {
                if Self::is_horizontal_clear(padded, r, p1.col, p2.col)
                    && Self::is_vertical_clear(padded, p2.col, r, p2.row)
                {
                    let orig_k1 = PaddedBoard::to_orig_point(k1);
                    let orig_k2 = PaddedBoard::to_orig_point(k2);
                    return Some((MatchType::TwoTurn(orig_k1, orig_k2), vec![p1, k1, k2, p2]));
                }
            }
        }

        // Scan up from p1.row - 1 down to 0
        for r in (0..p1.row).rev() {
            if !padded.is_empty(r, p1.col) {
                break; // blocked
            }
            let k1 = Point::new(r, p1.col);
            let k2 = Point::new(r, p2.col);
            if r != p2.row && padded.is_empty(k2.row, k2.col) {
                if Self::is_horizontal_clear(padded, r, p1.col, p2.col)
                    && Self::is_vertical_clear(padded, p2.col, r, p2.row)
                {
                    let orig_k1 = PaddedBoard::to_orig_point(k1);
                    let orig_k2 = PaddedBoard::to_orig_point(k2);
                    return Some((MatchType::TwoTurn(orig_k1, orig_k2), vec![p1, k1, k2, p2]));
                }
            }
        }

        None
    }

    /// Heuristic score calculation
    fn calculate_score(
        &self,
        board: &Board,
        p1: Point,
        p2: Point,
        match_type: &MatchType,
        orig_path: &[Point],
    ) -> i64 {
        let dist = p1.manhattan_distance(&p2) as i64;
        let mut score = dist * self.config.distance_weight;

        let p1_perim = p1.is_on_perimeter(board.rows, board.cols);
        let p2_perim = p2.is_on_perimeter(board.rows, board.cols);

        if p1_perim && p2_perim {
            score += self.config.both_perimeter_bonus;
        } else if p1_perim || p2_perim {
            score += self.config.single_perimeter_bonus;
        }

        match match_type {
            MatchType::ZeroTurn => score += self.config.zero_turn_bonus,
            MatchType::OneTurn(_) => score += self.config.one_turn_bonus,
            MatchType::TwoTurn(_, _) => score += self.config.two_turn_bonus,
        }

        // Bonus if path goes through outer margins (-1 or rows/cols in original coordinates)
        let uses_outer_margin = orig_path.iter().any(|pt| {
            pt.row < 0 || pt.row >= board.rows as i32 || pt.col < 0 || pt.col >= board.cols as i32
        });
        if uses_outer_margin {
            score += self.config.outer_margin_bonus;
        }

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::EMPTY;

    #[test]
    fn test_zero_turn_adjacent_horizontal() {
        let mut board = Board::new(4, 4);
        board.set(1, 1, 1);
        board.set(1, 2, 1);

        let solver = Solver::new();
        let m = solver.can_connect(&board, Point::new(1, 1), Point::new(1, 2));
        assert!(m.is_some());
        let m = m.unwrap();
        assert_eq!(m.match_type, MatchType::ZeroTurn);
        assert_eq!(m.path, vec![Point::new(1, 1), Point::new(1, 2)]);
    }

    #[test]
    fn test_zero_turn_separated_vertical() {
        let mut board = Board::new(5, 5);
        board.set(0, 2, 3);
        board.set(4, 2, 3);

        let solver = Solver::new();
        let m = solver.can_connect(&board, Point::new(0, 2), Point::new(4, 2));
        assert!(m.is_some());
        let m = m.unwrap();
        assert_eq!(m.match_type, MatchType::ZeroTurn);
    }

    #[test]
    fn test_zero_turn_blocked_by_other_tile() {
        let mut board = Board::new(4, 4);
        board.set(1, 0, 1);
        board.set(1, 1, 2); // blocking tile
        board.set(1, 2, 1);

        let solver = Solver::new();
        // Since board (0,0) to (0,2) is empty, they might connect via 2-turn around,
        // but let's test specifically that zero-turn is blocked.
        let padded = board.to_padded();
        let pp1 = PaddedBoard::to_padded_point(Point::new(1, 0));
        let pp2 = PaddedBoard::to_padded_point(Point::new(1, 2));
        assert!(solver.check_zero_turn(&padded, pp1, pp2).is_none());
    }

    #[test]
    fn test_one_turn_l_shape() {
        let mut board = Board::new(4, 4);
        board.set(0, 1, 5);
        board.set(2, 3, 5);
        // Corner (0, 3) is empty, path (0,1)->(0,3)->(2,3) is clear!

        let solver = Solver::new();
        let m = solver.can_connect(&board, Point::new(0, 1), Point::new(2, 3));
        assert!(m.is_some());
        let m = m.unwrap();
        assert_eq!(m.match_type, MatchType::OneTurn(Point::new(0, 3)));
    }

    #[test]
    fn test_one_turn_blocked_by_obstacle() {
        let mut board = Board::new(3, 3);
        board.set(0, 0, 7);
        board.set(2, 2, 7);
        // Both corners blocked
        board.set(0, 2, -1); // Obstacle at corner 1
        board.set(2, 0, -1); // Obstacle at corner 2

        let solver = Solver::new();
        let padded = board.to_padded();
        let pp1 = PaddedBoard::to_padded_point(Point::new(0, 0));
        let pp2 = PaddedBoard::to_padded_point(Point::new(2, 2));
        assert!(solver.check_one_turn(&padded, pp1, pp2).is_none());
    }

    #[test]
    fn test_two_turn_z_shape() {
        let mut board = Board::new(4, 4);
        // Place tiles at (1, 1) and (2, 3)
        // Block (1, 3) and (2, 1) to eliminate 1-turn
        board.set(1, 1, 8);
        board.set(2, 3, 8);
        board.set(1, 3, 9); // blocks Corner (1, 3)
        board.set(2, 1, 9); // blocks Corner (2, 1)

        let solver = Solver::new();
        let m = solver.can_connect(&board, Point::new(1, 1), Point::new(2, 3));
        assert!(m.is_some());
        let m = m.unwrap();
        match m.match_type {
            MatchType::TwoTurn(k1, k2) => {
                // k1 and k2 could be via col 2: (1, 2) and (2, 2)
                assert_eq!(k1.col, k2.col);
            }
            _ => panic!("Expected 2-turn match"),
        }
    }

    #[test]
    fn test_two_turn_perimeter_outer_boundary() {
        // Two tiles on the top row separated by an obstacle
        let mut board = Board::new(3, 5);
        board.set(0, 1, 4);
        board.set(0, 2, -1); // Obstacle in between
        board.set(0, 3, 4);

        let solver = Solver::new();
        let m = solver.can_connect(&board, Point::new(0, 1), Point::new(0, 3));
        assert!(m.is_some());
        let m = m.unwrap();
        // The path should use the outer row -1: (0, 1) -> (-1, 1) -> (-1, 3) -> (0, 3)
        assert_eq!(
            m.match_type,
            MatchType::TwoTurn(Point::new(-1, 1), Point::new(-1, 3))
        );
        assert_eq!(
            m.path,
            vec![
                Point::new(0, 1),
                Point::new(-1, 1),
                Point::new(-1, 3),
                Point::new(0, 3)
            ]
        );
    }

    #[test]
    fn test_obstacle_completely_enclosing_deadlock() {
        let mut board = Board::new(5, 5);
        board.set(2, 2, 1);
        board.set(0, 0, 1);
        // Completely surround (2, 2) with obstacles
        board.set(1, 2, -1);
        board.set(3, 2, -1);
        board.set(2, 1, -1);
        board.set(2, 3, -1);

        let solver = Solver::new();
        let m = solver.can_connect(&board, Point::new(2, 2), Point::new(0, 0));
        assert!(m.is_none());
        assert_eq!(solver.find_all_matches(&board).len(), 0);
    }

    #[test]
    fn test_different_tile_ids_cannot_match() {
        let mut board = Board::new(2, 2);
        board.set(0, 0, 1);
        board.set(0, 1, 2);

        let solver = Solver::new();
        assert!(solver
            .can_connect(&board, Point::new(0, 0), Point::new(0, 1))
            .is_none());
    }

    #[test]
    fn test_heuristic_prioritizes_perimeter_and_distance() {
        let mut board = Board::new(6, 6);
        // Pair A: interior, short distance (2, 2) and (2, 3)
        board.set(2, 2, 1);
        board.set(2, 3, 1);

        // Pair B: perimeter, long distance (0, 0) and (0, 5)
        board.set(0, 0, 2);
        board.set(0, 5, 2);

        let solver = Solver::new();
        let matches = solver.find_all_matches(&board);
        assert_eq!(matches.len(), 2);
        // Best match should be Pair B (Tile ID 2) due to perimeter + distance bonus
        assert_eq!(matches[0].tile_id, 2);
        assert_eq!(matches[1].tile_id, 1);
    }

    #[test]
    fn test_full_board_clearing_simulation() {
        // A solvable 2x4 board
        let mut board = Board::from_grid(&[
            vec![1, 2, 2, 1],
            vec![3, 4, 4, 3],
        ])
        .unwrap();

        let solver = Solver::new();
        let mut steps = 0;
        while let Some(best) = solver.find_best_match(&board) {
            assert!(board.set_point(best.p1, EMPTY));
            assert!(board.set_point(best.p2, EMPTY));
            steps += 1;
        }

        assert!(board.is_cleared());
        assert_eq!(steps, 4);
    }
}

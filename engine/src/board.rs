use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

pub const EMPTY: i32 = 0;
pub const OBSTACLE: i32 = -1;
/// A cell that is NOT flat background but could not be matched to any template with
/// enough confidence (hover highlight, animation frame, unseen tile ...).
/// It blocks paths exactly like an obstacle and is never offered as a tile.
pub const UNKNOWN: i32 = -2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Point {
    pub row: i32,
    pub col: i32,
}

impl Point {
    #[inline]
    pub const fn new(row: i32, col: i32) -> Self {
        Self { row, col }
    }

    #[inline]
    pub fn manhattan_distance(&self, other: &Point) -> i32 {
        (self.row - other.row).abs() + (self.col - other.col).abs()
    }

    #[inline]
    pub fn is_on_perimeter(&self, rows: usize, cols: usize) -> bool {
        self.row == 0
            || self.row == (rows as i32 - 1)
            || self.col == 0
            || self.col == (cols as i32 - 1)
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.row, self.col)
    }
}

/// Representation of the game board.
/// Cell values:
///   0   = Empty
///  -1   = Obstacle (unmatchable, blocks paths)
/// 1..N  = Matchable Tile ID
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Board {
    pub rows: usize,
    pub cols: usize,
    pub cells: Vec<i32>,
}

impl Board {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            cells: vec![EMPTY; rows * cols],
        }
    }

    pub fn from_grid(grid: &[Vec<i32>]) -> Result<Self> {
        let rows = grid.len();
        if rows == 0 {
            bail!("Grid rows cannot be 0");
        }
        let cols = grid[0].len();
        if cols == 0 {
            bail!("Grid cols cannot be 0");
        }
        let mut cells = Vec::with_capacity(rows * cols);
        for (r, row) in grid.iter().enumerate() {
            if row.len() != cols {
                bail!(
                    "Inconsistent column count at row {}: expected {}, got {}",
                    r,
                    cols,
                    row.len()
                );
            }
            cells.extend_from_slice(row);
        }
        Ok(Self { rows, cols, cells })
    }

    pub fn from_flat(rows: usize, cols: usize, cells: Vec<i32>) -> Result<Self> {
        if cells.len() != rows * cols {
            bail!(
                "Cell count mismatch: expected {}, got {}",
                rows * cols,
                cells.len()
            );
        }
        Ok(Self { rows, cols, cells })
    }

    #[inline]
    pub fn in_bounds(&self, row: i32, col: i32) -> bool {
        row >= 0 && row < self.rows as i32 && col >= 0 && col < self.cols as i32
    }

    #[inline]
    pub fn get(&self, row: i32, col: i32) -> Option<i32> {
        if self.in_bounds(row, col) {
            Some(self.cells[row as usize * self.cols + col as usize])
        } else {
            None
        }
    }

    #[inline]
    pub fn get_point(&self, p: Point) -> Option<i32> {
        self.get(p.row, p.col)
    }

    #[inline]
    pub fn set(&mut self, row: i32, col: i32, val: i32) -> bool {
        if self.in_bounds(row, col) {
            self.cells[row as usize * self.cols + col as usize] = val;
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn set_point(&mut self, p: Point, val: i32) -> bool {
        self.set(p.row, p.col, val)
    }

    #[inline]
    pub fn is_empty(&self, row: i32, col: i32) -> bool {
        self.get(row, col) == Some(EMPTY)
    }

    #[inline]
    pub fn is_obstacle(&self, row: i32, col: i32) -> bool {
        self.get(row, col) == Some(OBSTACLE)
    }

    #[inline]
    pub fn is_tile(&self, row: i32, col: i32) -> bool {
        matches!(self.get(row, col), Some(val) if val > 0)
    }

    /// Count tiles that are not empty and not obstacles
    pub fn remaining_tiles_count(&self) -> usize {
        self.cells.iter().filter(|&&v| v > 0).count()
    }

    /// Copy of the board in which every OBSTACLE cell is treated as EMPTY.
    pub fn without_obstacles(&self) -> Board {
        let mut b = self.clone();
        for v in b.cells.iter_mut() {
            if *v == OBSTACLE {
                *v = EMPTY;
            }
        }
        b
    }

    /// Cells that could not be recognised (see `UNKNOWN`).
    pub fn unknown_points(&self) -> Vec<Point> {
        let mut v = Vec::new();
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.cells[r * self.cols + c] == UNKNOWN {
                    v.push(Point::new(r as i32, c as i32));
                }
            }
        }
        v
    }

    pub fn is_cleared(&self) -> bool {
        self.remaining_tiles_count() == 0
    }

    /// Returns all tiles with val > 0 with their coordinates
    pub fn find_all_tiles(&self) -> Vec<(Point, i32)> {
        let mut tiles = Vec::new();
        for r in 0..self.rows {
            for c in 0..self.cols {
                let val = self.cells[r * self.cols + c];
                if val > 0 {
                    tiles.push((Point::new(r as i32, c as i32), val));
                }
            }
        }
        tiles
    }

    /// Group tile coordinates by their Tile ID
    pub fn group_by_tile_id(&self) -> HashMap<i32, Vec<Point>> {
        let mut map: HashMap<i32, Vec<Point>> = HashMap::new();
        for (p, id) in self.find_all_tiles() {
            map.entry(id).or_default().push(p);
        }
        map
    }

    /// Creates a PaddedBoard with 1 ring of EMPTY cells around the entire board.
    /// Original (r, c) becomes Padded (r + 1, c + 1).
    pub fn to_padded(&self) -> PaddedBoard {
        self.to_padded_with_border(EMPTY)
    }

    /// Padded copy whose outer ring is filled with `border` (EMPTY = free to route
    /// around the board, OBSTACLE = paths must stay inside the board).
    pub fn to_padded_with_border(&self, border: i32) -> PaddedBoard {
        let p_rows = self.rows + 2;
        let p_cols = self.cols + 2;
        let mut p_cells = vec![border; p_rows * p_cols];

        for r in 0..self.rows {
            for c in 0..self.cols {
                let orig_val = self.cells[r * self.cols + c];
                let p_idx = (r + 1) * p_cols + (c + 1);
                p_cells[p_idx] = orig_val;
            }
        }

        PaddedBoard {
            orig_rows: self.rows,
            orig_cols: self.cols,
            p_rows,
            p_cols,
            cells: p_cells,
        }
    }

    /// Render board to human-readable string
    pub fn display(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Board ({}x{}):\n", self.rows, self.cols));
        for r in 0..self.rows {
            for c in 0..self.cols {
                let val = self.cells[r * self.cols + c];
                match val {
                    EMPTY => out.push_str("  . "),
                    OBSTACLE => out.push_str("  # "),
                    id => out.push_str(&format!("{:3} ", id)),
                }
            }
            out.push('\n');
        }
        out
    }
}

/// Internal padded representation for pathfinding
#[derive(Debug, Clone)]
pub struct PaddedBoard {
    pub orig_rows: usize,
    pub orig_cols: usize,
    pub p_rows: usize,
    pub p_cols: usize,
    pub cells: Vec<i32>,
}

impl PaddedBoard {
    #[inline]
    pub fn to_padded_point(p: Point) -> Point {
        Point::new(p.row + 1, p.col + 1)
    }

    #[inline]
    pub fn to_orig_point(p: Point) -> Point {
        Point::new(p.row - 1, p.col - 1)
    }

    #[inline]
    pub fn in_bounds(&self, r: i32, c: i32) -> bool {
        r >= 0 && r < self.p_rows as i32 && c >= 0 && c < self.p_cols as i32
    }

    #[inline]
    pub fn get(&self, r: i32, c: i32) -> i32 {
        if self.in_bounds(r, c) {
            self.cells[r as usize * self.p_cols + c as usize]
        } else {
            OBSTACLE
        }
    }

    #[inline]
    pub fn is_empty(&self, r: i32, c: i32) -> bool {
        self.get(r, c) == EMPTY
    }
}

//! Mouse selection window built on `minifb` (no hand-written Win32 window code).
//!
//! Shows an image, lets the user drag a rectangle, Enter/Space confirms, Esc cancels.
//! Returns the rectangle in the ORIGINAL image's pixel coordinates.

use crate::calibrate::Roi;
use crate::imgproc::{resize, Image};
use anyhow::{Context, Result};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, ScaleMode, Window, WindowOptions};

pub struct Placement {
    /// Top-left of the window on the virtual desktop, if it should be placed explicitly
    /// (borderless full-monitor overlay for live screen capture).
    pub at: Option<(isize, isize)>,
    /// Window (and buffer) size in pixels.
    pub size: (usize, usize),
    pub borderless: bool,
}

fn to_u32(img: &Image) -> Vec<u32> {
    img.data
        .chunks_exact(3)
        .map(|p| ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32)
        .collect()
}

type R = (i32, i32, i32, i32); // x0, y0, x1, y1 (normalised: x0<=x1, y0<=y1)

fn norm(a: (i32, i32), b: (i32, i32)) -> R {
    (a.0.min(b.0), a.1.min(b.1), a.0.max(b.0), a.1.max(b.1))
}

/// What a mouse drag is doing. `Edge` flags: left, top, right, bottom.
#[derive(Clone, Copy)]
enum Drag {
    New((i32, i32)),
    Move((i32, i32), R),
    Resize(bool, bool, bool, bool, R),
}

const HIT: i32 = 9;

fn hit_test(r: R, p: (i32, i32)) -> Option<Drag> {
    let (x0, y0, x1, y1) = r;
    let near = |a: i32, b: i32| (a - b).abs() <= HIT;
    let inx = p.0 >= x0 - HIT && p.0 <= x1 + HIT;
    let iny = p.1 >= y0 - HIT && p.1 <= y1 + HIT;
    let (l, rt) = (inx && iny && near(p.0, x0), inx && iny && near(p.0, x1));
    let (t, b) = (inx && iny && near(p.1, y0), inx && iny && near(p.1, y1));
    if l || rt || t || b {
        return Some(Drag::Resize(l, t, rt && !l, b && !t, r));
    }
    if p.0 > x0 && p.0 < x1 && p.1 > y0 && p.1 < y1 {
        return Some(Drag::Move(p, r));
    }
    None
}

fn apply_drag(d: Drag, p: (i32, i32), w: i32, h: i32) -> R {
    let cl = |v: i32, m: i32| v.clamp(0, m - 1);
    match d {
        Drag::New(s) => norm(s, (cl(p.0, w), cl(p.1, h))),
        Drag::Move(s, r) => {
            let (rw, rh) = (r.2 - r.0, r.3 - r.1);
            let x0 = (r.0 + p.0 - s.0).clamp(0, (w - 1 - rw).max(0));
            let y0 = (r.1 + p.1 - s.1).clamp(0, (h - 1 - rh).max(0));
            (x0, y0, x0 + rw, y0 + rh)
        }
        Drag::Resize(l, t, rt, b, r) => {
            let (mut x0, mut y0, mut x1, mut y1) = r;
            if l { x0 = cl(p.0, w); }
            if rt { x1 = cl(p.0, w); }
            if t { y0 = cl(p.1, h); }
            if b { y1 = cl(p.1, h); }
            (x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1))
        }
    }
}

fn draw_rect(buf: &mut [u32], w: usize, h: usize, r: R) {
    let (x0, x1) = (r.0.clamp(0, w as i32 - 1), r.2.clamp(0, w as i32 - 1));
    let (y0, y1) = (r.1.clamp(0, h as i32 - 1), r.3.clamp(0, h as i32 - 1));
    let mut put = |x: i32, y: i32, c: u32| {
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            buf[y as usize * w + x as usize] = c;
        }
    };
    for t in 0..2 {
        let col = if t == 0 { 0xFF2020 } else { 0xFFF000 };
        for x in x0..=x1 {
            put(x, y0 + t, col);
            put(x, y1 - t, col);
        }
        for y in y0..=y1 {
            put(x0 + t, y, col);
            put(x1 - t, y, col);
        }
    }
    // 8 resize handles
    let (mx, my) = ((x0 + x1) / 2, (y0 + y1) / 2);
    for &(hx, hy) in &[(x0, y0), (mx, y0), (x1, y0), (x0, my), (x1, my), (x0, y1), (mx, y1), (x1, y1)] {
        for dy in -3i32..=3 {
            for dx in -3i32..=3 {
                let edge = dx.abs() == 3 || dy.abs() == 3;
                put(hx + dx, hy + dy, if edge { 0x000000 } else { 0xFFFFFF });
            }
        }
    }
}

/// Let the user select a rectangle on `img`. `place.size` is the window size; the image is
/// scaled to it (1:1 for a full-monitor capture, shrunk for a big image file).
pub fn select_roi(img: &Image, title: &str, place: &Placement, initial: Option<Roi>) -> Result<Option<Roi>> {
    let (ww, wh) = place.size;
    let shown = if (img.w, img.h) == (ww, wh) { img.clone() } else { resize(img, ww, wh) };
    let base = to_u32(&shown);

    let mut window = Window::new(
        title,
        ww,
        wh,
        WindowOptions {
            borderless: place.borderless,
            title: !place.borderless,
            resize: false,
            topmost: true,
            scale: Scale::X1,
            scale_mode: ScaleMode::UpperLeft,
            ..WindowOptions::default()
        },
    )
    .context("cannot open the selection window")?;
    if let Some((x, y)) = place.at {
        window.set_position(x, y);
    }
    window.set_target_fps(60);

    let mut buf = base.clone();

    let (wi, hi) = (ww as i32, wh as i32);
    // optional pre-drawn rectangle (image px -> window px) that the user can move / resize
    let mut sel: Option<R> = initial.map(|r| {
        let (kx, ky) = (ww as f64 / img.w as f64, wh as f64 / img.h as f64);
        let x0 = ((r.x as f64 * kx).round() as i32).clamp(0, wi - 1);
        let y0 = ((r.y as f64 * ky).round() as i32).clamp(0, hi - 1);
        let x1 = (((r.x + r.w) as f64 * kx).round() as i32).clamp(x0 + 1, wi - 1);
        let y1 = (((r.y + r.h) as f64 * ky).round() as i32).clamp(y0 + 1, hi - 1);
        (x0, y0, x1, y1)
    });
    let mut drag: Option<Drag> = None;
    let mut was_down = false;

    if let Some(r) = sel {
        draw_rect(&mut buf, ww, wh, r);
    }
    window.update_with_buffer(&buf, ww, wh)?;

    while window.is_open() {
        if window.is_key_pressed(Key::Escape, KeyRepeat::No) {
            return Ok(None);
        }
        let mut dirty = false;
        if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Clamp) {
            let down = window.get_mouse_down(MouseButton::Left);
            let p = (mx as i32, my as i32);
            if down && !was_down {
                drag = Some(sel.and_then(|r| hit_test(r, p)).unwrap_or(Drag::New(p)));
                if matches!(drag, Some(Drag::New(_))) {
                    sel = None;
                }
                dirty = true;
            }
            if down {
                if let Some(d) = drag {
                    let r = apply_drag(d, p, wi, hi);
                    if sel != Some(r) {
                        sel = Some(r);
                        dirty = true;
                    }
                }
            } else {
                drag = None;
            }
            was_down = down;
        }
        // keyboard fine-tuning: arrows move (Shift = 10 px); Ctrl+arrows resize right/bottom edge
        if let Some(r) = sel {
            let step = if window.is_key_down(Key::LeftShift) || window.is_key_down(Key::RightShift) { 10 } else { 1 };
            let ctrl = window.is_key_down(Key::LeftCtrl) || window.is_key_down(Key::RightCtrl);
            let (mut dx, mut dy) = (0, 0);
            if window.is_key_pressed(Key::Left, KeyRepeat::Yes) { dx = -step; }
            if window.is_key_pressed(Key::Right, KeyRepeat::Yes) { dx = step; }
            if window.is_key_pressed(Key::Up, KeyRepeat::Yes) { dy = -step; }
            if window.is_key_pressed(Key::Down, KeyRepeat::Yes) { dy = step; }
            if dx != 0 || dy != 0 {
                let n = if ctrl {
                    let x1 = (r.2 + dx).clamp(r.0 + 5, wi - 1);
                    let y1 = (r.3 + dy).clamp(r.1 + 5, hi - 1);
                    (r.0, r.1, x1, y1)
                } else {
                    apply_drag(Drag::Move((0, 0), r), (dx, dy), wi, hi)
                };
                sel = Some(n);
                dirty = true;
            }
        }
        if (window.is_key_pressed(Key::Enter, KeyRepeat::No) || window.is_key_pressed(Key::Space, KeyRepeat::No))
            && sel.map(|r| r.2 - r.0 > 4 && r.3 - r.1 > 4).unwrap_or(false)
        {
            let r = sel.unwrap();
            let (sx, sy) = (img.w as f64 / ww as f64, img.h as f64 / wh as f64);
            let x0 = (r.0 as f64 * sx).round() as i64;
            let y0 = (r.1 as f64 * sy).round() as i64;
            let x1 = (r.2 as f64 * sx).round() as i64;
            let y1 = (r.3 as f64 * sy).round() as i64;
            return Ok(Some(Roi { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }));
        }
        if dirty {
            buf.copy_from_slice(&base);
            if let Some(r) = sel {
                draw_rect(&mut buf, ww, wh, r);
            }
            window.update_with_buffer(&buf, ww, wh)?;
        } else {
            window.update();
        }
    }
    Ok(None)
}

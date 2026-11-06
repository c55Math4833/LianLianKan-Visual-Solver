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

fn draw_rect(buf: &mut [u32], w: usize, h: usize, a: (i32, i32), b: (i32, i32)) {
    let (x0, x1) = (a.0.min(b.0).clamp(0, w as i32 - 1), a.0.max(b.0).clamp(0, w as i32 - 1));
    let (y0, y1) = (a.1.min(b.1).clamp(0, h as i32 - 1), a.1.max(b.1).clamp(0, h as i32 - 1));
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
}

/// Let the user select a rectangle on `img`. `place.size` is the window size; the image is
/// scaled to it (1:1 for a full-monitor capture, shrunk for a big image file).
pub fn select_roi(img: &Image, title: &str, place: &Placement) -> Result<Option<Roi>> {
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
    window.update_with_buffer(&buf, ww, wh)?;

    let mut start: Option<(i32, i32)> = None;
    let mut cur = (0, 0);
    let mut sel: Option<((i32, i32), (i32, i32))> = None;
    let mut was_down = false;

    while window.is_open() {
        if window.is_key_pressed(Key::Escape, KeyRepeat::No) {
            return Ok(None);
        }
        let mut dirty = false;
        if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Clamp) {
            let down = window.get_mouse_down(MouseButton::Left);
            let p = (mx as i32, my as i32);
            if down && !was_down {
                start = Some(p);
                sel = None;
            }
            if down {
                if cur != p || !was_down {
                    dirty = true;
                }
                cur = p;
                if let Some(s) = start {
                    sel = Some((s, cur));
                }
            }
            if !down && was_down {
                dirty = true;
            }
            was_down = down;
        }
        if (window.is_key_pressed(Key::Enter, KeyRepeat::No) || window.is_key_pressed(Key::Space, KeyRepeat::No))
            && sel.map(|(a, b)| (a.0 - b.0).abs() > 4 && (a.1 - b.1).abs() > 4).unwrap_or(false)
        {
            let (a, b) = sel.unwrap();
            let (sx, sy) = (img.w as f64 / ww as f64, img.h as f64 / wh as f64);
            let x0 = (a.0.min(b.0) as f64 * sx).round() as i64;
            let y0 = (a.1.min(b.1) as f64 * sy).round() as i64;
            let x1 = (a.0.max(b.0) as f64 * sx).round() as i64;
            let y1 = (a.1.max(b.1) as f64 * sy).round() as i64;
            return Ok(Some(Roi { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }));
        }
        if dirty {
            buf.copy_from_slice(&base);
            if let Some((a, b)) = sel {
                draw_rect(&mut buf, ww, wh, a, b);
            }
            window.update_with_buffer(&buf, ww, wh)?;
        } else {
            window.update();
        }
    }
    Ok(None)
}

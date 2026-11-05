use crate::board::{Point, EMPTY};
use crate::capture::ScreenCapturer;
use crate::matcher::Matcher;
use crate::profile::GameProfile;
use anyhow::{bail, Result};
use image::DynamicImage;
use std::thread::sleep;
use std::time::{Duration, Instant};

#[cfg(windows)]
use windows::Win32::UI::{
    Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_MOUSE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
        MOUSEINPUT,
    },
    WindowsAndMessaging::SetCursorPos,
};

pub struct Controller {
    pub click_delay: Duration,
}

impl Default for Controller {
    fn default() -> Self {
        Self {
            click_delay: Duration::from_millis(30),
        }
    }
}

impl Controller {
    pub fn new(click_delay_ms: u64) -> Self {
        Self {
            click_delay: Duration::from_millis(click_delay_ms),
        }
    }

    /// Calculate center screen coordinates (X, Y) for a cell on the board
    pub fn calculate_cell_center(profile: &GameProfile, p: Point) -> (i32, i32) {
        let cell_w = profile.board.roi.width as f64 / profile.board.cols as f64;
        let cell_h = profile.board.roi.height as f64 / profile.board.rows as f64;

        let cx = profile.board.roi.x as f64 + (p.col as f64 + 0.5) * cell_w;
        let cy = profile.board.roi.y as f64 + (p.row as f64 + 0.5) * cell_h;

        (cx.round() as i32, cy.round() as i32)
    }

    /// Calculate bounding box (x, y, width, height) of a cell
    pub fn calculate_cell_rect(profile: &GameProfile, p: Point) -> (i32, i32, u32, u32) {
        let cell_w = profile.board.roi.width as f64 / profile.board.cols as f64;
        let cell_h = profile.board.roi.height as f64 / profile.board.rows as f64;

        let x = profile.board.roi.x as f64 + (p.col as f64) * cell_w;
        let y = profile.board.roi.y as f64 + (p.row as f64) * cell_h;

        (
            x.round() as i32,
            y.round() as i32,
            cell_w.round() as u32,
            cell_h.round() as u32,
        )
    }

    /// Inject a left mouse click at screen coordinates (x, y) using Win32 SendInput
    pub fn click_at(&self, x: i32, y: i32) -> Result<()> {
        #[cfg(windows)]
        unsafe {
            let _ = SetCursorPos(x, y);
            sleep(Duration::from_millis(10));

            let inputs = [
                INPUT {
                    r#type: INPUT_MOUSE,
                    Anonymous: INPUT_0 {
                        mi: MOUSEINPUT {
                            dx: 0,
                            dy: 0,
                            mouseData: 0,
                            dwFlags: MOUSEEVENTF_LEFTDOWN,
                            time: 0,
                            dwExtraInfo: 0,
                        },
                    },
                },
                INPUT {
                    r#type: INPUT_MOUSE,
                    Anonymous: INPUT_0 {
                        mi: MOUSEINPUT {
                            dx: 0,
                            dy: 0,
                            mouseData: 0,
                            dwFlags: MOUSEEVENTF_LEFTUP,
                            time: 0,
                            dwExtraInfo: 0,
                        },
                    },
                },
            ];

            let sent = SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
            if sent != 2 {
                bail!("SendInput failed to inject mouse clicks (sent {})", sent);
            }
        }
        #[cfg(not(windows))]
        {
            eprintln!("[Mock Controller] Click at ({}, {})", x, y);
        }
        Ok(())
    }

    /// Click tile at point p
    pub fn click_cell(&self, profile: &GameProfile, p: Point) -> Result<()> {
        let (cx, cy) = Self::calculate_cell_center(profile, p);
        self.click_at(cx, cy)
    }

    /// Execute a match pair (Click p1 -> wait interval -> Click p2)
    /// then verify that both cells became EMPTY within the animation timeout window.
    pub fn execute_pair_and_verify(
        &self,
        capturer: &mut dyn ScreenCapturer,
        matcher: &Matcher,
        profile: &GameProfile,
        p1: Point,
        p2: Point,
    ) -> Result<bool> {
        // Step 1: Click first cell
        self.click_cell(profile, p1)?;
        sleep(self.click_delay);

        // Step 2: Click second cell
        self.click_cell(profile, p2)?;

        // Step 3: Wait for game elimination animation
        sleep(Duration::from_millis(profile.timing.animation_delay_ms));

        // Step 4: Verification loop
        let start = Instant::now();
        let timeout = Duration::from_millis(profile.timing.verify_timeout_ms);

        let (x1, y1, w1, h1) = Self::calculate_cell_rect(profile, p1);
        let (x2, y2, w2, h2) = Self::calculate_cell_rect(profile, p2);

        while start.elapsed() < timeout {
            let img1 = capturer.capture_roi(x1, y1, w1, h1)?;
            let img2 = capturer.capture_roi(x2, y2, w2, h2)?;

            let (id1, _) = matcher.match_cell(&DynamicImage::ImageRgba8(img1));
            let (id2, _) = matcher.match_cell(&DynamicImage::ImageRgba8(img2));

            if id1 == EMPTY && id2 == EMPTY {
                return Ok(true);
            }

            sleep(Duration::from_millis(20));
        }

        Ok(false)
    }
}

//! Windows monitor enumeration (DPI aware, physical pixels) and GDI screen capture.

use crate::imgproc::Image;
use anyhow::{bail, Result};
use windows::Win32::Foundation::{BOOL, LPARAM, RECT};
use windows::Win32::Graphics::Gdi::{
    BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, EnumDisplayMonitors,
    GetDC, GetDIBits, GetMonitorInfoW, ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB,
    DIB_RGB_COLORS, HDC, HGDIOBJ, HMONITOR, MONITORINFO, MONITORINFOEXW, SRCCOPY,
};
use windows::Win32::UI::HiDpi::{
    GetDpiForMonitor, SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
    MDT_EFFECTIVE_DPI,
};
use windows::Win32::System::Console::GetConsoleProcessList;
use windows::Win32::UI::WindowsAndMessaging::GetDesktopWindow;

/// True when this process owns its console window (double-clicked / drag-dropped), i.e. the
/// window disappears the moment we exit. False when started from an existing terminal.
pub fn owns_console() -> bool {
    let mut list = [0u32; 4];
    let n = unsafe { GetConsoleProcessList(&mut list) };
    n == 1
}

#[derive(Clone, Debug)]
pub struct Monitor {
    pub index: usize, // 1-based
    pub device: String,
    pub primary: bool,
    /// Virtual-desktop rectangle in PHYSICAL pixels.
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
    pub scale_percent: u32,
}

/// Must run before any window/monitor call so every coordinate is a real pixel
/// (otherwise Windows hands out DPI-virtualised numbers on scaled displays).
pub fn enable_dpi_awareness() {
    unsafe {
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }
}

unsafe extern "system" fn enum_proc(hm: HMONITOR, _: HDC, _: *mut RECT, data: LPARAM) -> BOOL {
    let list = &mut *(data.0 as *mut Vec<Monitor>);
    let mut mi = MONITORINFOEXW::default();
    mi.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
    if GetMonitorInfoW(hm, &mut mi as *mut MONITORINFOEXW as *mut MONITORINFO).as_bool() {
        let r = mi.monitorInfo.rcMonitor;
        let name_len = mi.szDevice.iter().position(|&c| c == 0).unwrap_or(mi.szDevice.len());
        let (mut dx, mut dy) = (96u32, 96u32);
        let _ = GetDpiForMonitor(hm, MDT_EFFECTIVE_DPI, &mut dx, &mut dy);
        list.push(Monitor {
            index: 0,
            device: String::from_utf16_lossy(&mi.szDevice[..name_len]),
            primary: mi.monitorInfo.dwFlags & 1 != 0,
            x: r.left,
            y: r.top,
            w: (r.right - r.left) as u32,
            h: (r.bottom - r.top) as u32,
            scale_percent: dx * 100 / 96,
        });
    }
    BOOL(1)
}

pub fn monitors() -> Vec<Monitor> {
    let mut list: Vec<Monitor> = Vec::new();
    unsafe {
        let _ = EnumDisplayMonitors(None, None, Some(enum_proc), LPARAM(&mut list as *mut Vec<Monitor> as isize));
    }
    // primary first, then left-to-right (stable numbering for the prompt)
    list.sort_by_key(|m| (!m.primary, m.x, m.y));
    for (i, m) in list.iter_mut().enumerate() {
        m.index = i + 1;
    }
    list
}

/// Grab a rectangle of the virtual desktop (physical pixels) as RGB.
pub fn capture_rect(x: i32, y: i32, w: u32, h: u32) -> Result<Image> {
    unsafe {
        let hwnd = GetDesktopWindow();
        let hdc_screen = GetDC(hwnd);
        if hdc_screen.is_invalid() {
            bail!("cannot get the desktop device context");
        }
        let hdc_mem = CreateCompatibleDC(hdc_screen);
        let hbm = CreateCompatibleBitmap(hdc_screen, w as i32, h as i32);
        let old: HGDIOBJ = SelectObject(hdc_mem, hbm);

        let cleanup = |hbm, hdc_mem, old: HGDIOBJ| {
            let _ = SelectObject(hdc_mem, old);
            let _ = DeleteObject(hbm);
            let _ = DeleteDC(hdc_mem);
            ReleaseDC(hwnd, hdc_screen);
        };
        if BitBlt(hdc_mem, 0, 0, w as i32, h as i32, hdc_screen, x, y, SRCCOPY).is_err() {
            cleanup(hbm, hdc_mem, old);
            bail!("screen capture (BitBlt) failed");
        }
        let mut bmi = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: w as i32,
                biHeight: -(h as i32), // top-down
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut raw = vec![0u8; (w * h * 4) as usize];
        let lines = GetDIBits(hdc_mem, hbm, 0, h, Some(raw.as_mut_ptr() as *mut _), &mut bmi, DIB_RGB_COLORS);
        cleanup(hbm, hdc_mem, old);
        if lines == 0 {
            bail!("GetDIBits failed");
        }
        let mut img = Image::new(w as usize, h as usize);
        for i in 0..(w * h) as usize {
            img.data[i * 3] = raw[i * 4 + 2];
            img.data[i * 3 + 1] = raw[i * 4 + 1];
            img.data[i * 3 + 2] = raw[i * 4];
        }
        Ok(img)
    }
}

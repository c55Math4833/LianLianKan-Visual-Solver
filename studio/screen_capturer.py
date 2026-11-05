"""
LianLianKan Screen Capturer
Ultra-fast, DPI-aware full-screen & multi-monitor capture for Windows using ctypes & Win32 GDI.
Zero external library dependencies (uses Python standard library).
"""

import sys
import numpy as np

def get_system_monitors():
    """
    Enumerates all active display monitors with their physical rect and DPI scaling.
    Returns a list of dicts:
    [{
        'index': 1,
        'device': '\\\\.\\DISPLAY1',
        'is_primary': True,
        'rect': (x, y, w, h),
        'scale': 150, # percentage
        'dpi': 144
    }, ...]
    """
    if sys.platform != "win32":
        return []

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    try:
        shcore = ctypes.windll.shcore
    except Exception:
        shcore = None

    # Enable Per-Monitor DPI Awareness V2 so all coordinates are true physical pixels
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

    monitors = []

    def monitor_enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData):
        r = lprcMonitor.contents
        scale = 100
        dpi_val = 96
        if shcore:
            dpi_x = ctypes.c_uint()
            dpi_y = ctypes.c_uint()
            try:
                shcore.GetDpiForMonitor(hMonitor, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y))
                dpi_val = dpi_x.value
                scale = round(dpi_val / 96.0 * 100)
            except Exception:
                pass

        class MONITORINFOEX(ctypes.Structure):
            _fields_ = [
                ('cbSize', wintypes.DWORD),
                ('rcMonitor', wintypes.RECT),
                ('rcWork', wintypes.RECT),
                ('dwFlags', wintypes.DWORD),
                ('szDevice', wintypes.WCHAR * 32)
            ]
        mi = MONITORINFOEX()
        mi.cbSize = ctypes.sizeof(MONITORINFOEX)
        user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi))
        is_primary = bool(mi.dwFlags & 1)

        monitors.append({
            'index': len(monitors) + 1,
            'hMonitor': hMonitor,
            'rect': (r.left, r.top, r.right - r.left, r.bottom - r.top),
            'device': mi.szDevice,
            'is_primary': is_primary,
            'scale': scale,
            'dpi': dpi_val
        })
        return True

    MonitorEnumProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM
    )
    callback = MonitorEnumProc(monitor_enum_proc)
    user32.EnumDisplayMonitors(None, None, callback, 0)

    # Sort so primary monitor is first, then left-to-right
    monitors.sort(key=lambda m: (not m['is_primary'], m['rect'][0]))
    for idx, m in enumerate(monitors):
        m['index'] = idx + 1
    return monitors

def capture_rect(x, y, width, height):
    """
    Captures an arbitrary physical screen rectangle via Win32 GDI BitBlt.
    """
    if sys.platform != "win32":
        raise OSError("Screen capture is currently only supported on Windows")

    import ctypes
    from ctypes import wintypes
    import cv2

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        pass

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid screen metrics: {width}x{height}")

    hdc_screen = user32.GetDC(0)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    hbm = gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
    gdi32.SelectObject(hdc_mem, hbm)

    SRCCOPY = 0x00CC0020
    gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc_screen, x, y, SRCCOPY)

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    bmi.bmiHeader.biHeight = -height  # Top-down bitmap
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0

    buffer_size = width * height * 4
    raw_buffer = (ctypes.c_ubyte * buffer_size)()
    gdi32.GetDIBits(hdc_mem, hbm, 0, height, raw_buffer, ctypes.byref(bmi), 0)

    gdi32.DeleteObject(hbm)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(0, hdc_screen)

    bgra = np.frombuffer(raw_buffer, dtype=np.uint8).reshape((height, width, 4))
    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    return bgr

def capture_screen(monitor=None):
    """
    Captures either a specific monitor or the primary monitor.
    Returns (bgr_image, monitor_info_dict).
    """
    monitors = get_system_monitors()
    target_monitor = None

    if monitor is not None:
        if isinstance(monitor, dict):
            target_monitor = monitor
        elif isinstance(monitor, int) and 1 <= monitor <= len(monitors):
            target_monitor = monitors[monitor - 1]

    if target_monitor is None:
        # Default to primary monitor
        for m in monitors:
            if m['is_primary']:
                target_monitor = m
                break
        if target_monitor is None and monitors:
            target_monitor = monitors[0]

    if target_monitor is None:
        # Fallback to standard primary metrics
        import ctypes
        user32 = ctypes.windll.user32
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        target_monitor = {
            'index': 1,
            'device': 'Primary',
            'is_primary': True,
            'rect': (0, 0, w, h),
            'scale': 100,
            'dpi': 96
        }

    rx, ry, rw, rh = target_monitor['rect']
    img = capture_rect(rx, ry, rw, rh)
    return img, target_monitor

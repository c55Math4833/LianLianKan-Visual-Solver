"""
LianLianKan Screen Capturer
Ultra-fast, DPI-aware full-screen capture for Windows using ctypes & Win32 GDI.
Zero external library dependencies (uses Python standard library).
"""

import sys
import numpy as np

def capture_screen():
    """
    Captures the primary monitor screen and returns a BGR numpy image (H, W, 3).
    DPI-aware so coordinates match physical screen pixels exactly.
    """
    if sys.platform != "win32":
        raise OSError("Screen capture is currently only supported on Windows")

    import ctypes
    from ctypes import wintypes
    import cv2

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    # Enable per-monitor DPI awareness so pixel coordinates are 1:1
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

    width = user32.GetSystemMetrics(0)   # SM_CXSCREEN
    height = user32.GetSystemMetrics(1)  # SM_CYSCREEN

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid screen metrics: {width}x{height}")

    hdc_screen = user32.GetDC(0)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    hbm = gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
    gdi32.SelectObject(hdc_mem, hbm)

    SRCCOPY = 0x00CC0020
    gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc_screen, 0, 0, SRCCOPY)

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
    bmi.bmiHeader.biCompression = 0  # BI_RGB

    buffer_size = width * height * 4
    raw_buffer = (ctypes.c_ubyte * buffer_size)()
    gdi32.GetDIBits(hdc_mem, hbm, 0, height, raw_buffer, ctypes.byref(bmi), 0)

    # Cleanup GDI objects
    gdi32.DeleteObject(hbm)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(0, hdc_screen)

    # Convert BGRA buffer to BGR numpy array
    bgra = np.frombuffer(raw_buffer, dtype=np.uint8).reshape((height, width, 4))
    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    return bgr

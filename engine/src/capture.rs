use anyhow::{anyhow, bail, Context, Result};
use image::{Rgba, RgbaImage};
use std::ptr::null_mut;

#[cfg(windows)]
use windows::{
    core::Interface,
    Win32::{
        Foundation::{HWND, RECT},
        Graphics::{
            Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP, D3D_FEATURE_LEVEL_11_0},
            Direct3D11::{
                D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
                D3D11_BIND_FLAG, D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_FLAG,
                D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ, D3D11_SDK_VERSION,
                D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING,
            },
            Dxgi::{
                Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
                CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1, IDXGIOutput, IDXGIOutput1,
                IDXGIOutputDuplication, IDXGIResource, DXGI_ERROR_WAIT_TIMEOUT,
                DXGI_OUTDUPL_FRAME_INFO,
            },
            Gdi::{
                BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC,
                GetDIBits, ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB,
                DIB_RGB_COLORS, HDC, HGDIOBJ, SRCCOPY,
            },
        },
        UI::WindowsAndMessaging::{GetDesktopWindow, GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN},
    },
};

pub trait ScreenCapturer {
    fn capture_full(&mut self) -> Result<RgbaImage>;
    fn capture_roi(&mut self, x: i32, y: i32, width: u32, height: u32) -> Result<RgbaImage>;
}

/// DXGI Desktop Duplication Screen Capturer (<3ms latency)
#[cfg(windows)]
pub struct DxgiCapturer {
    _device: ID3D11Device,
    context: ID3D11DeviceContext,
    duplication: IDXGIOutputDuplication,
    staging_texture: ID3D11Texture2D,
    width: u32,
    height: u32,
    last_frame: Option<RgbaImage>,
}

#[cfg(windows)]
impl DxgiCapturer {
    pub fn new() -> Result<Self> {
        unsafe {
            let factory: IDXGIFactory1 =
                CreateDXGIFactory1().context("Failed to create DXGI factory")?;

            let mut adapter_opt: Option<IDXGIAdapter1> = None;
            let mut i = 0;
            while let Ok(adapter) = factory.EnumAdapters1(i) {
                adapter_opt = Some(adapter);
                break;
            }

            let adapter = adapter_opt.ok_or_else(|| anyhow!("No DXGI adapter found"))?;

            let mut device: Option<ID3D11Device> = None;
            let mut context: Option<ID3D11DeviceContext> = None;

            // Try hardware driver first, fallback to WARP if needed
            let driver_types = [D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP];
            let mut created = false;

            for driver_type in driver_types {
                let res = D3D11CreateDevice(
                    &adapter,
                    driver_type,
                    None,
                    D3D11_CREATE_DEVICE_FLAG(0),
                    Some(&[D3D_FEATURE_LEVEL_11_0]),
                    D3D11_SDK_VERSION,
                    Some(&mut device),
                    None,
                    Some(&mut context),
                );
                if res.is_ok() {
                    created = true;
                    break;
                }
            }

            if !created {
                bail!("Failed to create D3D11 device");
            }

            let device = device.unwrap();
            let context = context.unwrap();

            let output: IDXGIOutput = adapter
                .EnumOutputs(0)
                .context("Failed to enum primary DXGI output")?;
            let output1: IDXGIOutput1 = output.cast().context("Failed to cast to IDXGIOutput1")?;

            let duplication = output1
                .DuplicateOutput(&device)
                .context("Failed to create DXGI Desktop Duplication")?;

            let desc = duplication.GetDesc();
            let width = desc.ModeDesc.Width;
            let height = desc.ModeDesc.Height;

            let staging_desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                MiscFlags: 0,
            };

            let mut staging_texture: Option<ID3D11Texture2D> = None;
            device
                .CreateTexture2D(&staging_desc, None, Some(&mut staging_texture))
                .context("Failed to create staging texture")?;

            let staging_texture = staging_texture.unwrap();

            Ok(Self {
                _device: device,
                context,
                duplication,
                staging_texture,
                width,
                height,
                last_frame: None,
            })
        }
    }
}

#[cfg(windows)]
impl ScreenCapturer for DxgiCapturer {
    fn capture_full(&mut self) -> Result<RgbaImage> {
        unsafe {
            let mut frame_info = DXGI_OUTDUPL_FRAME_INFO::default();
            let mut desktop_resource: Option<IDXGIResource> = None;

            // Wait up to 30ms for frame
            let hr = self
                .duplication
                .AcquireNextFrame(30, &mut frame_info, &mut desktop_resource);

            if let Err(e) = hr {
                if e.code() == DXGI_ERROR_WAIT_TIMEOUT {
                    // Frame hasn't changed; reuse previous frame if available
                    if let Some(ref img) = self.last_frame {
                        return Ok(img.clone());
                    }
                }
                bail!("DXGI AcquireNextFrame error: {:?}", e);
            }

            let desktop_resource = desktop_resource.unwrap();
            let desktop_texture: ID3D11Texture2D = desktop_resource.cast()?;

            // Copy to staging texture on GPU
            self.context
                .CopyResource(&self.staging_texture, &desktop_texture);

            // Release frame immediately so GPU can continue rendering
            let _ = self.duplication.ReleaseFrame();

            // Map staging texture for CPU read
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            self.context.Map(
                &self.staging_texture,
                0,
                D3D11_MAP_READ,
                0,
                Some(&mut mapped),
            )?;

            let row_pitch = mapped.RowPitch as usize;
            let src_slice = std::slice::from_raw_parts(
                mapped.pData as *const u8,
                row_pitch * self.height as usize,
            );

            let mut img = RgbaImage::new(self.width, self.height);
            for y in 0..self.height {
                let row_offset = (y as usize) * row_pitch;
                for x in 0..self.width {
                    let px_offset = row_offset + (x as usize) * 4;
                    let b = src_slice[px_offset];
                    let g = src_slice[px_offset + 1];
                    let r = src_slice[px_offset + 2];
                    let a = src_slice[px_offset + 3];
                    img.put_pixel(x, y, Rgba([r, g, b, a]));
                }
            }

            self.context.Unmap(&self.staging_texture, 0);
            self.last_frame = Some(img.clone());
            Ok(img)
        }
    }

    fn capture_roi(&mut self, x: i32, y: i32, width: u32, height: u32) -> Result<RgbaImage> {
        let full = self.capture_full()?;
        let cropped = image::imageops::crop_imm(
            &full,
            x.max(0) as u32,
            y.max(0) as u32,
            width.min(full.width()),
            height.min(full.height()),
        )
        .to_image();
        Ok(cropped)
    }
}

/// Fallback GDI Capturer
#[cfg(windows)]
pub struct GdiCapturer {
    width: u32,
    height: u32,
}

#[cfg(windows)]
impl GdiCapturer {
    pub fn new() -> Result<Self> {
        unsafe {
            let width = GetSystemMetrics(SM_CXSCREEN) as u32;
            let height = GetSystemMetrics(SM_CYSCREEN) as u32;
            if width == 0 || height == 0 {
                bail!("Invalid screen metrics via GDI");
            }
            Ok(Self { width, height })
        }
    }
}

#[cfg(windows)]
impl ScreenCapturer for GdiCapturer {
    fn capture_full(&mut self) -> Result<RgbaImage> {
        self.capture_roi(0, 0, self.width, self.height)
    }

    fn capture_roi(&mut self, x: i32, y: i32, width: u32, height: u32) -> Result<RgbaImage> {
        unsafe {
            let hwnd_desktop = GetDesktopWindow();
            let hdc_screen = GetDC(hwnd_desktop);
            if hdc_screen.is_invalid() {
                bail!("Failed to get desktop DC");
            }

            let hdc_mem: HDC = CreateCompatibleDC(hdc_screen);
            let hbm: windows::Win32::Graphics::Gdi::HBITMAP =
                CreateCompatibleBitmap(hdc_screen, width as i32, height as i32);

            let old_obj: HGDIOBJ = SelectObject(hdc_mem, hbm);

            let blt_res = BitBlt(
                hdc_mem,
                0,
                0,
                width as i32,
                height as i32,
                hdc_screen,
                x,
                y,
                SRCCOPY,
            );

            if blt_res.is_err() {
                let _ = SelectObject(hdc_mem, old_obj);
                let _ = DeleteObject(hbm);
                let _ = DeleteDC(hdc_mem);
                ReleaseDC(hwnd_desktop, hdc_screen);
                bail!("GDI BitBlt failed");
            }

            let mut bmi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: width as i32,
                    biHeight: -(height as i32), // Top-down
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    biSizeImage: 0,
                    biXPelsPerMeter: 0,
                    biYPelsPerMeter: 0,
                    biClrUsed: 0,
                    biClrImportant: 0,
                },
                bmiColors: [windows::Win32::Graphics::Gdi::RGBQUAD::default()],
            };

            let mut raw_bytes = vec![0u8; (width * height * 4) as usize];
            let lines = GetDIBits(
                hdc_mem,
                hbm,
                0,
                height,
                Some(raw_bytes.as_mut_ptr() as *mut _),
                &mut bmi,
                DIB_RGB_COLORS,
            );

            // Cleanup GDI objects
            let _ = SelectObject(hdc_mem, old_obj);
            let _ = DeleteObject(hbm);
            let _ = DeleteDC(hdc_mem);
            ReleaseDC(hwnd_desktop, hdc_screen);

            if lines == 0 {
                bail!("GetDIBits failed to copy pixels");
            }

            // Convert BGRA to RGBA image
            let mut img = RgbaImage::new(width, height);
            for y in 0..height {
                for x in 0..width {
                    let idx = ((y * width + x) * 4) as usize;
                    let b = raw_bytes[idx];
                    let g = raw_bytes[idx + 1];
                    let r = raw_bytes[idx + 2];
                    img.put_pixel(x, y, Rgba([r, g, b, 255]));
                }
            }

            Ok(img)
        }
    }
}

/// Helper function to create the best available capturer (DXGI preferred, GDI fallback)
pub fn create_capturer() -> Box<dyn ScreenCapturer> {
    #[cfg(windows)]
    {
        match DxgiCapturer::new() {
            Ok(capturer) => {
                eprintln!("[Capture] Initialized DXGI Desktop Duplication (<3ms latency)");
                Box::new(capturer)
            }
            Err(err) => {
                eprintln!(
                    "[Capture] DXGI unavailable ({:?}), falling back to GDI Capturer",
                    err
                );
                match GdiCapturer::new() {
                    Ok(gdi) => Box::new(gdi),
                    Err(gdi_err) => {
                        panic!("Failed to initialize both DXGI and GDI: {:?}", gdi_err);
                    }
                }
            }
        }
    }
    #[cfg(not(windows))]
    {
        panic!("Screen capture only supported on Windows");
    }
}

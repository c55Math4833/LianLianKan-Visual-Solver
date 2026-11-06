//! Minimal image primitives that mirror the OpenCV calls the Python studio relied on.
//! Everything is RGB8 interleaved; formulas follow OpenCV so results line up with the
//! Python version.

#[derive(Clone, Debug)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>, // RGB, row-major
}

impl Image {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0; w * h * 3] }
    }

    pub fn from_rgb(img: &image::RgbImage) -> Self {
        Self { w: img.width() as usize, h: img.height() as usize, data: img.as_raw().clone() }
    }

    pub fn to_rgb(&self) -> image::RgbImage {
        image::RgbImage::from_raw(self.w as u32, self.h as u32, self.data.clone()).unwrap()
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [u8; 3] {
        let i = (y * self.w + x) * 3;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    /// numpy-style slice `img[y:y+h, x:x+w]` (clamped to the image).
    pub fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> Image {
        let x0 = x.min(self.w);
        let y0 = y.min(self.h);
        let x1 = (x + w).min(self.w);
        let y1 = (y + h).min(self.h);
        let (cw, ch) = (x1 - x0, y1 - y0);
        let mut out = Image::new(cw, ch);
        for yy in 0..ch {
            let src = ((y0 + yy) * self.w + x0) * 3;
            out.data[yy * cw * 3..(yy + 1) * cw * 3].copy_from_slice(&self.data[src..src + cw * 3]);
        }
        out
    }
}

/// Python's round(): half to even.
pub fn pyround(x: f64) -> i64 {
    let f = x.floor();
    let d = x - f;
    let fi = f as i64;
    if d < 0.5 {
        fi
    } else if d > 0.5 {
        fi + 1
    } else if fi % 2 == 0 {
        fi
    } else {
        fi + 1
    }
}

pub fn median(v: &[f64]) -> f64 {
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// cv2.cvtColor(..., COLOR_BGR2GRAY) fixed-point formula.
pub fn gray(img: &Image) -> Vec<f64> {
    let mut out = Vec::with_capacity(img.w * img.h);
    for p in img.data.chunks_exact(3) {
        let (r, g, b) = (p[0] as u32, p[1] as u32, p[2] as u32);
        out.push(((b * 1868 + g * 9617 + r * 4899 + 8192) >> 14) as f64);
    }
    out
}

#[inline]
fn reflect101(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let n = n as isize;
    let mut i = i;
    if i < 0 {
        i = -i;
    }
    if i >= n {
        i = 2 * (n - 1) - i;
    }
    i.clamp(0, n - 1) as usize
}

/// cv2.Sobel(gray, CV_64F, dx, dy, ksize=3) (signed, BORDER_REFLECT_101).
/// `horizontal == true` -> derivative along x (dx=1, dy=0).
pub fn sobel(g: &[f64], w: usize, h: usize, horizontal: bool) -> Vec<f64> {
    let mut out = vec![0.0; w * h];
    let at = |x: isize, y: isize| g[reflect101(y, h) * w + reflect101(x, w)];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let v = if horizontal {
                (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1))
                    - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1))
            } else {
                (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1))
                    - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1))
            };
            out[y as usize * w + x as usize] = v;
        }
    }
    out
}

/// cv2.resize(..., INTER_LINEAR).
pub fn resize(img: &Image, nw: usize, nh: usize) -> Image {
    if img.w == nw && img.h == nh {
        return img.clone();
    }
    let mut out = Image::new(nw, nh);
    let sx = img.w as f64 / nw as f64;
    let sy = img.h as f64 / nh as f64;
    for y in 0..nh {
        let fy = ((y as f64 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(img.h - 1);
        let y1 = (y0 + 1).min(img.h - 1);
        let wy = fy - y0 as f64;
        for x in 0..nw {
            let fx = ((x as f64 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(img.w - 1);
            let x1 = (x0 + 1).min(img.w - 1);
            let wx = fx - x0 as f64;
            for c in 0..3 {
                let p = |xx: usize, yy: usize| img.data[(yy * img.w + xx) * 3 + c] as f64;
                let top = p(x0, y0) * (1.0 - wx) + p(x1, y0) * wx;
                let bot = p(x0, y1) * (1.0 - wx) + p(x1, y1) * wx;
                let v = top * (1.0 - wy) + bot * wy;
                out.data[(y * nw + x) * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// cv2 BGR2HSV for 8-bit: H in 0..180, S,V in 0..255. Input given as RGB.
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let (rf, gf, bf) = (r as i32, g as i32, b as i32);
    let v = rf.max(gf).max(bf);
    let mn = rf.min(gf).min(bf);
    let diff = v - mn;
    let s = if v == 0 { 0.0 } else { 255.0 * diff as f64 / v as f64 };
    let mut h = if diff == 0 {
        0.0
    } else if v == rf {
        (gf - bf) as f64 * 30.0 / diff as f64
    } else if v == gf {
        ((bf - rf) as f64 + 2.0 * diff as f64) * 30.0 / diff as f64
    } else {
        ((rf - gf) as f64 + 4.0 * diff as f64) * 30.0 / diff as f64
    };
    if h < 0.0 {
        h += 180.0;
    }
    ((h.round() as i32 % 180) as u8, s.round() as u8, v as u8)
}

/// cv2 BGR2LAB for 8-bit (L scaled to 0..255, a/b offset by 128). Input RGB.
pub fn rgb_to_lab(r: u8, g: u8, b: u8) -> [f64; 3] {
    let lin = |c: u8| {
        let c = c as f64 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let (r, g, b) = (lin(r), lin(g), lin(b));
    let x = (0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.950456;
    let y = 0.212671 * r + 0.715160 * g + 0.072169 * b;
    let z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754;
    let f = |t: f64| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let l = if y > 0.008856 { 116.0 * y.cbrt() - 16.0 } else { 903.3 * y };
    let a = 500.0 * (f(x) - f(y));
    let bb = 200.0 * (f(y) - f(z));
    [(l * 255.0 / 100.0).round().clamp(0.0, 255.0), (a + 128.0).round().clamp(0.0, 255.0), (bb + 128.0).round().clamp(0.0, 255.0)]
}

/// cv2.resize(..., INTER_AREA): every destination pixel is the overlap-weighted average of the
/// source pixels it covers (works for shrinking and enlarging).
pub fn resize_area(img: &Image, nw: usize, nh: usize) -> Image {
    fn weights(src: usize, dst: usize) -> Vec<Vec<(usize, f64)>> {
        let s = src as f64 / dst as f64;
        (0..dst)
            .map(|d| {
                let (a, b) = (d as f64 * s, (d as f64 + 1.0) * s);
                let (j0, j1) = (a.floor() as usize, ((b.ceil() as usize).min(src)).max(a.floor() as usize + 1));
                let mut v: Vec<(usize, f64)> = (j0..j1.min(src))
                    .map(|j| (j, (b.min(j as f64 + 1.0) - a.max(j as f64)).max(0.0)))
                    .filter(|(_, w)| *w > 0.0)
                    .collect();
                let tot: f64 = v.iter().map(|(_, w)| w).sum();
                v.iter_mut().for_each(|(_, w)| *w /= tot);
                v
            })
            .collect()
    }
    if img.w == nw && img.h == nh {
        return img.clone();
    }
    let (wx, wy) = (weights(img.w, nw), weights(img.h, nh));
    // horizontal pass
    let mut tmp = vec![0.0f64; nw * img.h * 3];
    for y in 0..img.h {
        for x in 0..nw {
            for c in 0..3 {
                tmp[(y * nw + x) * 3 + c] =
                    wx[x].iter().map(|&(j, w)| img.data[(y * img.w + j) * 3 + c] as f64 * w).sum();
            }
        }
    }
    let mut out = Image::new(nw, nh);
    for y in 0..nh {
        for x in 0..nw {
            for c in 0..3 {
                let v: f64 = wy[y].iter().map(|&(j, w)| tmp[(j * nw + x) * 3 + c] * w).sum();
                out.data[(y * nw + x) * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

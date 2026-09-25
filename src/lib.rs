use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn to_grayscale(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let expected_len = (width as usize) * (height as usize) * 4;
    assert_eq!(
        data.len(),
        expected_len,
        "buffer de entrada com tamanho inesperado"
    );

    let mut out = vec![0u8; data.len()];
    for px in 0..(width as usize * height as usize) {
        let i = px * 4;
        let r = data[i] as f32;
        let g = data[i + 1] as f32;
        let b = data[i + 2] as f32;
        let y = (0.299 * r + 0.587 * g + 0.114 * b)
            .round()
            .clamp(0.0, 255.0) as u8;
        out[i] = y;
        out[i + 1] = y;
        out[i + 2] = y;
        out[i + 3] = data[i + 3];
    }
    out
}

fn gaussian_kernel_1d(sigma: f32, radius: i32) -> Vec<f32> {
    let size = (radius * 2 + 1) as usize;
    let mut kernel = vec![0f32; size];
    let mut sum = 0f32;
    for i in -radius..=radius {
        let v = (-((i * i) as f32) / (2.0 * sigma * sigma)).exp();
        kernel[(i + radius) as usize] = v;
        sum += v;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }
    kernel
}

#[wasm_bindgen]
pub fn gaussian_blur(data: &[u8], width: u32, height: u32, sigma: f32) -> Vec<u8> {
    let w = width as i32;
    let h = height as i32;
    let expected_len = (width as usize) * (height as usize) * 4;
    assert_eq!(
        data.len(),
        expected_len,
        "buffer de entrada com tamanho inesperado"
    );

    let radius = ((sigma * 3.0).round() as i32).clamp(1, 24);
    let kernel = gaussian_kernel_1d(sigma, radius);

    let mut tmp = vec![0f32; data.len()];
    for y in 0..h {
        let row_base = y * w;
        for x in 0..w {
            let (mut r, mut g, mut b, mut a) = (0f32, 0f32, 0f32, 0f32);
            for k in -radius..=radius {
                let mut xx = x + k;
                if xx < 0 {
                    xx = 0
                } else if xx >= w {
                    xx = w - 1
                }
                let idx = ((row_base + xx) * 4) as usize;
                let wgt = kernel[(k + radius) as usize];
                r += data[idx] as f32 * wgt;
                g += data[idx + 1] as f32 * wgt;
                b += data[idx + 2] as f32 * wgt;
                a += data[idx + 3] as f32 * wgt;
            }
            let oidx = ((row_base + x) * 4) as usize;
            tmp[oidx] = r;
            tmp[oidx + 1] = g;
            tmp[oidx + 2] = b;
            tmp[oidx + 3] = a;
        }
    }

    let mut out = vec![0u8; data.len()];
    for y in 0..h {
        for x in 0..w {
            let (mut r, mut g, mut b, mut a) = (0f32, 0f32, 0f32, 0f32);
            for k in -radius..=radius {
                let mut yy = y + k;
                if yy < 0 {
                    yy = 0
                } else if yy >= h {
                    yy = h - 1
                }
                let idx = ((yy * w + x) * 4) as usize;
                let wgt = kernel[(k + radius) as usize];
                r += tmp[idx] * wgt;
                g += tmp[idx + 1] * wgt;
                b += tmp[idx + 2] * wgt;
                a += tmp[idx + 3] * wgt;
            }
            let oidx = ((y * w + x) * 4) as usize;
            out[oidx] = r.round().clamp(0.0, 255.0) as u8;
            out[oidx + 1] = g.round().clamp(0.0, 255.0) as u8;
            out[oidx + 2] = b.round().clamp(0.0, 255.0) as u8;
            out[oidx + 3] = a.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

const SOBEL_GX: [i32; 9] = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
const SOBEL_GY: [i32; 9] = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

#[wasm_bindgen]
pub fn sobel_edges(data: &[u8], width: u32, height: u32, threshold: u8) -> Vec<u8> {
    let w = width as i32;
    let h = height as i32;
    let expected_len = (width as usize) * (height as usize) * 4;
    assert_eq!(
        data.len(),
        expected_len,
        "buffer de entrada com tamanho inesperado"
    );

    let mut gray = vec![0f32; (w * h) as usize];
    for (p, chunk) in data.chunks_exact(4).enumerate() {
        gray[p] = 0.299 * chunk[0] as f32 + 0.587 * chunk[1] as f32 + 0.114 * chunk[2] as f32;
    }

    let mut out = vec![0u8; data.len()];
    for y in 0..h {
        for x in 0..w {
            let mut sx = 0f32;
            let mut sy = 0f32;
            let mut k = 0usize;
            for j in -1..=1 {
                let mut yy = y + j;
                if yy < 0 {
                    yy = 0
                } else if yy >= h {
                    yy = h - 1
                }
                for i in -1..=1 {
                    let mut xx = x + i;
                    if xx < 0 {
                        xx = 0
                    } else if xx >= w {
                        xx = w - 1
                    }
                    let v = gray[(yy * w + xx) as usize];
                    sx += v * SOBEL_GX[k] as f32;
                    sy += v * SOBEL_GY[k] as f32;
                    k += 1;
                }
            }
            let mut mag = (sx * sx + sy * sy).sqrt();
            if threshold > 0 {
                mag = if mag >= threshold as f32 { 255.0 } else { 0.0 };
            } else {
                mag = mag.clamp(0.0, 255.0);
            }
            let idx = ((y * w + x) * 4) as usize;
            let m = mag as u8;
            out[idx] = m;
            out[idx + 1] = m;
            out[idx + 2] = m;
            out[idx + 3] = 255;
        }
    }
    out
}

#[wasm_bindgen]
pub fn module_info() -> String {
    "image-wasm 0.1.0 (Rust + wasm-bindgen)".to_string()
}


#[wasm_bindgen]
pub fn alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf); // a posse passa para o chamador (JS); libere com `dealloc`
    ptr
}

#[wasm_bindgen]
pub unsafe fn dealloc(ptr: *mut u8, len: usize) {
    let _ = Vec::from_raw_parts(ptr, len, len); // dropado aqui, memória liberada
}

#[wasm_bindgen]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

#[wasm_bindgen]
pub unsafe fn to_grayscale_ptr(src_ptr: *const u8, dst_ptr: *mut u8, width: u32, height: u32) {
    let len = (width as usize) * (height as usize) * 4;
    let src = std::slice::from_raw_parts(src_ptr, len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, len);
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let y = (0.299 * s[0] as f32 + 0.587 * s[1] as f32 + 0.114 * s[2] as f32)
            .round()
            .clamp(0.0, 255.0) as u8;
        d[0] = y;
        d[1] = y;
        d[2] = y;
        d[3] = s[3];
    }
}

#[wasm_bindgen]
pub unsafe fn gaussian_blur_ptr(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    width: u32,
    height: u32,
    sigma: f32,
) {
    let w = width as i32;
    let h = height as i32;
    let len = (width as usize) * (height as usize) * 4;
    let src = std::slice::from_raw_parts(src_ptr, len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, len);

    let radius = ((sigma * 3.0).round() as i32).clamp(1, 24);
    let kernel = gaussian_kernel_1d(sigma, radius);

    let mut tmp = vec![0f32; len];
    for y in 0..h {
        let row_base = y * w;
        for x in 0..w {
            let (mut r, mut g, mut b, mut a) = (0f32, 0f32, 0f32, 0f32);
            for k in -radius..=radius {
                let mut xx = x + k;
                if xx < 0 {
                    xx = 0
                } else if xx >= w {
                    xx = w - 1
                }
                let idx = ((row_base + xx) * 4) as usize;
                let wgt = kernel[(k + radius) as usize];
                r += src[idx] as f32 * wgt;
                g += src[idx + 1] as f32 * wgt;
                b += src[idx + 2] as f32 * wgt;
                a += src[idx + 3] as f32 * wgt;
            }
            let oidx = ((row_base + x) * 4) as usize;
            tmp[oidx] = r;
            tmp[oidx + 1] = g;
            tmp[oidx + 2] = b;
            tmp[oidx + 3] = a;
        }
    }

    for y in 0..h {
        for x in 0..w {
            let (mut r, mut g, mut b, mut a) = (0f32, 0f32, 0f32, 0f32);
            for k in -radius..=radius {
                let mut yy = y + k;
                if yy < 0 {
                    yy = 0
                } else if yy >= h {
                    yy = h - 1
                }
                let idx = ((yy * w + x) * 4) as usize;
                let wgt = kernel[(k + radius) as usize];
                r += tmp[idx] * wgt;
                g += tmp[idx + 1] * wgt;
                b += tmp[idx + 2] * wgt;
                a += tmp[idx + 3] * wgt;
            }
            let oidx = ((y * w + x) * 4) as usize;
            dst[oidx] = r.round().clamp(0.0, 255.0) as u8;
            dst[oidx + 1] = g.round().clamp(0.0, 255.0) as u8;
            dst[oidx + 2] = b.round().clamp(0.0, 255.0) as u8;
            dst[oidx + 3] = a.round().clamp(0.0, 255.0) as u8;
        }
    }
}

#[wasm_bindgen]
pub unsafe fn sobel_edges_ptr(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    width: u32,
    height: u32,
    threshold: u8,
) {
    let w = width as i32;
    let h = height as i32;
    let len = (width as usize) * (height as usize) * 4;
    let src = std::slice::from_raw_parts(src_ptr, len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, len);

    let mut gray = vec![0f32; (w * h) as usize];
    for (p, chunk) in src.chunks_exact(4).enumerate() {
        gray[p] = 0.299 * chunk[0] as f32 + 0.587 * chunk[1] as f32 + 0.114 * chunk[2] as f32;
    }

    for y in 0..h {
        for x in 0..w {
            let mut sx = 0f32;
            let mut sy = 0f32;
            let mut k = 0usize;
            for j in -1..=1 {
                let mut yy = y + j;
                if yy < 0 {
                    yy = 0
                } else if yy >= h {
                    yy = h - 1
                }
                for i in -1..=1 {
                    let mut xx = x + i;
                    if xx < 0 {
                        xx = 0
                    } else if xx >= w {
                        xx = w - 1
                    }
                    let v = gray[(yy * w + xx) as usize];
                    sx += v * SOBEL_GX[k] as f32;
                    sy += v * SOBEL_GY[k] as f32;
                    k += 1;
                }
            }
            let mut mag = (sx * sx + sy * sy).sqrt();
            if threshold > 0 {
                mag = if mag >= threshold as f32 { 255.0 } else { 0.0 };
            } else {
                mag = mag.clamp(0.0, 255.0);
            }
            let idx = ((y * w + x) * 4) as usize;
            let m = mag as u8;
            dst[idx] = m;
            dst[idx + 1] = m;
            dst[idx + 2] = m;
            dst[idx + 3] = 255;
        }
    }
}

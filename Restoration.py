import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import os

# ===========================================================
# UTILITAS DASAR
# ===========================================================

def pad_image(image, pad_h, pad_w=None):
    if pad_w is None:
        pad_w = pad_h
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

def convolve2d(image, kernel):
    """Konvolusi 2D manual via sliding_window_view (tanpa loop)."""
    kH, kW  = kernel.shape
    padded  = pad_image(image.astype(np.float64), kH // 2, kW // 2)
    windows = sliding_window_view(padded, (kH, kW))
    return np.clip(np.sum(windows * kernel, axis=(2, 3)), 0, 255).astype(np.uint8)

# ===========================================================
# LANGKAH 1 — DENOISING
# ===========================================================

def median_filter(channel, k=7):
    """Median filter — hapus salt-and-pepper noise."""
    pad     = k // 2
    padded  = np.pad(channel, pad, mode='edge')
    windows = sliding_window_view(padded, (k, k))
    return np.median(windows, axis=(2, 3)).astype(np.uint8)

def gaussian_kernel(size=5, sigma=1.0):
    ax     = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    g      = np.exp(-ax**2 / (2 * sigma**2))
    kernel = np.outer(g, g)
    return kernel / kernel.sum()

def gaussian_filter(channel, size=5, sigma=1.0):
    return convolve2d(channel, gaussian_kernel(size, sigma))

def denoise(channel):
    step1 = median_filter(channel, k=7)
    step2 = gaussian_filter(step1, size=5, sigma=1.2)
    return step2

# ===========================================================
# LANGKAH 2 — CLAHE MANUAL
# ===========================================================

def clahe_single(channel, clip_limit=3.0, tile_grid=(8, 8)):

    H, W   = channel.shape
    tH, tW = tile_grid
    th     = int(np.ceil(H / tH))
    tw     = int(np.ceil(W / tW))

    pad_h  = tH * th - H
    pad_w  = tW * tw - W
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge')
    pH, pW = padded.shape

    # LUT per tile: (tH, tW, 256)
    luts = np.zeros((tH, tW, 256), dtype=np.float32)

    for i in range(tH):
        for j in range(tW):
            tile   = padded[i*th:(i+1)*th, j*tw:(j+1)*tw]
            hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 256))

            limit  = max(1, clip_limit * tile.size / 256)
            excess = np.maximum(0, hist - limit).sum()
            hist   = np.minimum(hist, limit)
            hist  += excess / 256

            cdf     = hist.cumsum()
            cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
            denom   = tile.size - cdf_min
            lut     = np.where(denom > 0, (cdf - cdf_min) / denom * 255, 0)
            luts[i, j] = np.clip(lut, 0, 255)

    # Koordinat pusat tile
    cy = (np.arange(tH) + 0.5) * th
    cx = (np.arange(tW) + 0.5) * tw

    ys = np.arange(pH, dtype=np.float32)
    xs = np.arange(pW, dtype=np.float32)

    def idx_weight(coords, centers):
        idx = np.clip(np.searchsorted(centers, coords) - 1, 0, len(centers) - 2)
        w   = np.clip((coords - centers[idx]) / (centers[idx+1] - centers[idx] + 1e-6), 0, 1)
        return idx, np.minimum(idx + 1, len(centers) - 1), w

    iy0, iy1, wy = idx_weight(ys, cy)
    ix0, ix1, wx = idx_weight(xs, cx)

    pix     = padded.astype(np.int32)
    pix_f   = pix.ravel()
    iy0_2d  = np.repeat(iy0[:, None], pW, axis=1).ravel()
    iy1_2d  = np.repeat(iy1[:, None], pW, axis=1).ravel()
    ix0_2d  = np.tile(ix0[None, :],   (pH, 1)).ravel()
    ix1_2d  = np.tile(ix1[None, :],   (pH, 1)).ravel()
    wy_2d   = np.repeat(wy[:, None],  pW, axis=1).ravel()
    wx_2d   = np.tile(wx[None, :],    (pH, 1)).ravel()

    v00 = luts[iy0_2d, ix0_2d, pix_f]
    v01 = luts[iy0_2d, ix1_2d, pix_f]
    v10 = luts[iy1_2d, ix0_2d, pix_f]
    v11 = luts[iy1_2d, ix1_2d, pix_f]

    out = ((1-wy_2d)*(1-wx_2d)*v00 + (1-wy_2d)*wx_2d*v01 +
               wy_2d*(1-wx_2d)*v10 +      wy_2d*wx_2d*v11)

    return np.clip(out.reshape(pH, pW), 0, 255).astype(np.uint8)[:H, :W]


def apply_clahe_color(bgr_img, clip_limit=3.0):
    """
    CLAHE pada kanal Y saja (ruang YCrCb).
    Cr dan Cb tidak diubah → warna asli terjaga.
    """
    f   = bgr_img.astype(np.float64)
    B, G, R = f[:,:,0], f[:,:,1], f[:,:,2]

    Y  =  0.299*R + 0.587*G + 0.114*B      # luminance
    Cr = (R - Y) * 0.713 + 128.0           # chrominance merah
    Cb = (B - Y) * 0.564 + 128.0           # chrominance biru

    Y_eq = clahe_single(np.clip(Y, 0, 255).astype(np.uint8),
                        clip_limit=clip_limit).astype(np.float64)

    R_o = Y_eq + 1.402    * (Cr - 128.0)
    G_o = Y_eq - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0)
    B_o = Y_eq + 1.772    * (Cb - 128.0)

    return np.clip(np.stack([B_o, G_o, R_o], axis=2), 0, 255).astype(np.uint8)

# ===========================================================
# LANGKAH 3 — SHARPENING (Unsharp Masking)
# ===========================================================

def unsharp_mask(channel, strength=1.2, blur_size=5, blur_sigma=1.0):
    """
    sharpened = original + strength × (original − blurred)
    (original − blurred) = komponen frekuensi tinggi (tepi & detail).
    """
    blurred   = gaussian_filter(channel, size=blur_size, sigma=blur_sigma)
    high_freq = channel.astype(np.float64) - blurred.astype(np.float64)
    return np.clip(channel.astype(np.float64) + strength * high_freq, 0, 255).astype(np.uint8)

# ===========================================================
# MAIN PIPELINE
# ===========================================================

def restore(bgr_img):
    b, g, r = bgr_img[:,:,0], bgr_img[:,:,1], bgr_img[:,:,2]

    print("  [1/3] Denoising (Median 7x7 + Gaussian 5x5)...")
    denoised  = np.stack([denoise(b), denoise(g), denoise(r)], axis=2)

    print("  [2/3] CLAHE — kontras via luminance Y (warna terjaga)...")
    equalized = apply_clahe_color(denoised, clip_limit=3.0)

    print("  [3/3] Sharpening (Unsharp Masking)...")
    eb, eg, er = equalized[:,:,0], equalized[:,:,1], equalized[:,:,2]
    restored  = np.stack([unsharp_mask(eb), unsharp_mask(eg), unsharp_mask(er)], axis=2)

    return denoised, equalized, restored


def plot_results(original, denoised, equalized, restored, out_dir):
    stages = [
        (original,  "Asli (Rusak)"),
        (denoised,  "Setelah Denoising"),
        (equalized, "Setelah CLAHE"),
        (restored,  "Hasil Akhir"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Pipeline Restorasi Citra", fontsize=15, fontweight='bold')

    for i, (img, title) in enumerate(stages):
        axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(title, fontsize=11)
        axes[0, i].axis('off')

        lum  = img.mean(axis=2).astype(np.uint8)
        hist, _ = np.histogram(lum.flatten(), 256, (0, 256))
        axes[1, i].fill_between(range(256), hist, alpha=0.65, color='steelblue')
        axes[1, i].set_xlim(0, 255)
        axes[1, i].set_xlabel("Intensitas")
        axes[1, i].set_ylabel("Frekuensi")
        axes[1, i].set_title(f"Histogram — {title}", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "pipeline_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    os.makedirs('input',  exist_ok=True)
    os.makedirs('output', exist_ok=True)

    input_path  = 'input/lena_noisy.png'
    output_path = 'output/lena_restored.png'

    if not os.path.exists(input_path):
        print(f"[ERROR] Letakkan gambar di: {input_path}")
        return

    print("=" * 50)
    print("  PIPELINE: Denoising → CLAHE → Sharpening")
    print("=" * 50)

    img = cv2.imread(input_path)
    if img is None:
        print("[ERROR] Gagal membaca gambar.")
        return

    denoised, equalized, restored = restore(img)

    cv2.imwrite('output/1_denoised.png', denoised)
    cv2.imwrite('output/2_equalized.png', equalized)
    cv2.imwrite('output/lena_restored.png', restored)
    print(f"\n[✓] Tersimpan: {output_path}")

    plot_results(img, denoised, equalized, restored, 'output')

if __name__ == '__main__':
    main()
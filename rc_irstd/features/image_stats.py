from __future__ import annotations

import numpy as np
from scipy import ndimage


IMAGE_STAT_NAMES = (
    "gray_mean",
    "gray_std",
    "gray_mad",
    "gray_q01",
    "gray_q10",
    "gray_q50",
    "gray_q90",
    "gray_q99",
    "gradient_mean",
    "gradient_std",
    "gradient_q95",
    "laplacian_std",
    "local_contrast_mean",
    "local_contrast_q95",
    "entropy_256",
    "high_frequency_energy",
)


def _to_gray(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image)
    array = source.astype(np.float32, copy=False)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[-1] >= 3:
        gray = 0.2989 * array[..., 0] + 0.5870 * array[..., 1] + 0.1140 * array[..., 2]
    else:
        raise ValueError(f"Unsupported image shape {array.shape}")
    if np.issubdtype(source.dtype, np.integer):
        gray = gray / max(float(np.iinfo(source.dtype).max), 1.0)
    elif gray.max(initial=0.0) > 1.5:
        gray = gray / max(float(np.nanmax(gray)), 1.0)
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def compute_image_statistics(image: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    gray = _to_gray(image)
    median = float(np.median(gray))
    mad = float(np.median(np.abs(gray - median)))
    q01, q10, q50, q90, q99 = np.quantile(gray, [0.01, 0.10, 0.50, 0.90, 0.99])

    grad_y = ndimage.sobel(gray, axis=0, mode="reflect")
    grad_x = ndimage.sobel(gray, axis=1, mode="reflect")
    gradient = np.hypot(grad_x, grad_y)
    laplacian = ndimage.laplace(gray, mode="reflect")
    smooth = ndimage.gaussian_filter(gray, sigma=1.5, mode="reflect")
    local_contrast = np.abs(gray - smooth)

    histogram, _ = np.histogram(gray, bins=256, range=(0.0, 1.0), density=False)
    probabilities = histogram.astype(np.float64)
    probabilities /= max(probabilities.sum(), 1.0)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())

    spectrum = np.fft.rfft2(gray - gray.mean())
    power = np.abs(spectrum) ** 2
    height, width = power.shape
    y = np.fft.fftfreq(gray.shape[0])[:, None]
    x = np.fft.rfftfreq(gray.shape[1])[None, :]
    radial = np.sqrt(y * y + x * x)
    high_frequency = power[radial >= 0.25].sum() / max(power.sum(), 1e-12)

    values = np.asarray([
        gray.mean(),
        gray.std(),
        mad,
        q01,
        q10,
        q50,
        q90,
        q99,
        gradient.mean(),
        gradient.std(),
        np.quantile(gradient, 0.95),
        laplacian.std(),
        local_contrast.mean(),
        np.quantile(local_contrast, 0.95),
        entropy,
        high_frequency,
    ], dtype=np.float32)
    return values, IMAGE_STAT_NAMES

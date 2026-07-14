from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rc_irstd.candidates.peaks import build_fixed_peak_set, fixed_peak_curves


@dataclass(frozen=True)
class ImageCurveCounts:
    pixel_false: np.ndarray
    peak_false: np.ndarray
    matched_gt: np.ndarray
    predicted_pixels: np.ndarray
    predicted_peaks: np.ndarray
    total_pixels: int
    total_gt: int

    @property
    def num_thresholds(self) -> int:
        return len(self.pixel_false)


def _counts_at_thresholds(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=np.float32).reshape(-1))
    positions = np.searchsorted(ordered, thresholds, side="left")
    return (len(ordered) - positions).astype(np.int64)


def compute_image_curves(
    score_map: np.ndarray,
    gt_mask: np.ndarray,
    thresholds: np.ndarray,
    peak_min_distance: int = 2,
    peak_min_score: float = 1e-6,
    peak_border: int = 0,
    peak_tolerance: float = 2.0,
    max_candidates: int | None = None,
) -> ImageCurveCounts:
    score = np.asarray(score_map, dtype=np.float32).squeeze()
    mask = np.asarray(gt_mask).squeeze() > 0
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if score.shape != mask.shape:
        raise ValueError(f"score and mask shapes differ: {score.shape} vs {mask.shape}")
    if np.any(np.diff(thresholds) < 0):
        raise ValueError("thresholds must be ascending")

    background_values = score[~mask]
    all_values = score.reshape(-1)
    pixel_false = _counts_at_thresholds(background_values, thresholds)
    predicted_pixels = _counts_at_thresholds(all_values, thresholds)

    peak_set = build_fixed_peak_set(
        score,
        mask,
        min_distance=peak_min_distance,
        min_score=peak_min_score,
        border=peak_border,
        tolerance=peak_tolerance,
        max_candidates=max_candidates,
    )
    predicted_peaks, peak_false, matched_gt = fixed_peak_curves(peak_set, thresholds)
    return ImageCurveCounts(
        pixel_false=pixel_false,
        peak_false=peak_false,
        matched_gt=matched_gt,
        predicted_pixels=predicted_pixels,
        predicted_peaks=predicted_peaks,
        total_pixels=int(score.size),
        total_gt=int(peak_set.num_gt),
    )


def aggregate_curve_counts(records: list[ImageCurveCounts]) -> ImageCurveCounts:
    if not records:
        raise ValueError("Cannot aggregate an empty record list")
    num_thresholds = records[0].num_thresholds
    if any(record.num_thresholds != num_thresholds for record in records):
        raise ValueError("All records must use the same threshold grid")
    return ImageCurveCounts(
        pixel_false=np.sum([item.pixel_false for item in records], axis=0),
        peak_false=np.sum([item.peak_false for item in records], axis=0),
        matched_gt=np.sum([item.matched_gt for item in records], axis=0),
        predicted_pixels=np.sum([item.predicted_pixels for item in records], axis=0),
        predicted_peaks=np.sum([item.predicted_peaks for item in records], axis=0),
        total_pixels=sum(item.total_pixels for item in records),
        total_gt=sum(item.total_gt for item in records),
    )


def rates_from_counts(
    counts: ImageCurveCounts,
    pixel_epsilon: float = 1e-12,
    peak_epsilon: float = 1e-6,
) -> dict[str, np.ndarray]:
    pixel_rate = counts.pixel_false / max(counts.total_pixels, 1)
    peak_per_mp = counts.peak_false / max(counts.total_pixels / 1_000_000.0, 1e-12)
    pd = counts.matched_gt / max(counts.total_gt, 1)
    return {
        "pixel_false_rate": pixel_rate.astype(np.float64),
        "peak_false_per_mp": peak_per_mp.astype(np.float64),
        "pd": pd.astype(np.float64),
        "pixel_log_risk": np.log10(pixel_rate + pixel_epsilon),
        "peak_log_risk": np.log10(peak_per_mp + peak_epsilon),
    }


def monotone_nonincreasing_envelope(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.maximum.accumulate(values[::-1], axis=-1)[::-1]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from rc_irstd.candidates.peaks import extract_fixed_peaks
from rc_irstd.data.score_records import ScoreRecord


DEFAULT_SURVIVAL_THRESHOLDS = np.asarray([
    0.01,
    0.03,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.97,
    0.99,
    0.995,
    0.999,
    0.9995,
    0.9999,
], dtype=np.float32)
DEFAULT_QUANTILES = np.asarray([
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.995,
    0.999,
    0.9995,
], dtype=np.float32)


@dataclass(frozen=True)
class WindowFeatureConfig:
    survival_thresholds: np.ndarray = field(
        default_factory=lambda: DEFAULT_SURVIVAL_THRESHOLDS.copy()
    )
    quantiles: np.ndarray = field(default_factory=lambda: DEFAULT_QUANTILES.copy())
    peak_min_distance: int = 2
    peak_min_score: float = 0.0
    peak_border: int = 0
    max_candidates_per_image: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "survival_thresholds": np.asarray(
                self.survival_thresholds, dtype=np.float32
            ).tolist(),
            "quantiles": np.asarray(self.quantiles, dtype=np.float32).tolist(),
            "peak_min_distance": int(self.peak_min_distance),
            "peak_min_score": float(self.peak_min_score),
            "peak_border": int(self.peak_border),
            "max_candidates_per_image": (
                None
                if self.max_candidates_per_image is None
                else int(self.max_candidates_per_image)
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "WindowFeatureConfig":
        payload = payload or {}
        return cls(
            survival_thresholds=np.asarray(
                payload.get("survival_thresholds", DEFAULT_SURVIVAL_THRESHOLDS),
                dtype=np.float32,
            ),
            quantiles=np.asarray(
                payload.get("quantiles", DEFAULT_QUANTILES), dtype=np.float32
            ),
            peak_min_distance=int(payload.get("peak_min_distance", 2)),
            peak_min_score=float(payload.get("peak_min_score", 0.0)),
            peak_border=int(payload.get("peak_border", 0)),
            max_candidates_per_image=(
                None
                if payload.get("max_candidates_per_image") is None
                else int(payload["max_candidates_per_image"])
            ),
        )


class WindowFeatureExtractor:
    """Label-free score, peak and acquisition statistics for a deployment window."""

    def __init__(self, config: WindowFeatureConfig | None = None) -> None:
        self.config = config or WindowFeatureConfig()

    @staticmethod
    def _mean_std(values: np.ndarray) -> np.ndarray:
        return np.concatenate([values.mean(axis=0), values.std(axis=0)]).astype(np.float32)

    def extract(self, records: Sequence[ScoreRecord]) -> tuple[np.ndarray, tuple[str, ...]]:
        if not records:
            raise ValueError("A window must contain at least one score record")
        config = self.config
        threshold_count = len(config.survival_thresholds)
        quantile_count = len(config.quantiles)

        pixel_survival: list[np.ndarray] = []
        pixel_quantiles: list[np.ndarray] = []
        peak_survival_per_mp: list[np.ndarray] = []
        peak_quantiles: list[np.ndarray] = []
        image_statistics: list[np.ndarray] = []
        total_pixels: list[float] = []
        peak_counts: list[float] = []

        expected_stat_names = records[0].image_stat_names
        for record in records:
            if record.image_stat_names != expected_stat_names:
                raise ValueError("All records in a window must use the same image statistics")
            scores = record.probability.reshape(-1)
            pixel_survival.append(
                np.asarray([(scores >= threshold).mean() for threshold in config.survival_thresholds])
            )
            pixel_quantiles.append(np.quantile(scores, config.quantiles))

            peak_scores, _, _ = extract_fixed_peaks(
                record.probability,
                min_distance=config.peak_min_distance,
                min_score=config.peak_min_score,
                border=config.peak_border,
                max_candidates=config.max_candidates_per_image,
            )
            denominator_mp = max(record.total_pixels / 1_000_000.0, 1e-12)
            peak_survival_per_mp.append(
                np.asarray([
                    (peak_scores >= threshold).sum() / denominator_mp
                    for threshold in config.survival_thresholds
                ])
            )
            if len(peak_scores):
                peak_quantiles.append(np.quantile(peak_scores, config.quantiles))
            else:
                peak_quantiles.append(np.zeros(quantile_count, dtype=np.float32))
            image_statistics.append(record.image_stats)
            total_pixels.append(float(record.total_pixels))
            peak_counts.append(float(len(peak_scores) / denominator_mp))

        pixel_survival_array = np.log10(np.asarray(pixel_survival) + 1e-12)
        pixel_quantile_array = np.asarray(pixel_quantiles)
        peak_survival_array = np.log10(np.asarray(peak_survival_per_mp) + 1e-6)
        peak_quantile_array = np.asarray(peak_quantiles)
        image_stat_array = np.asarray(image_statistics)

        features = np.concatenate([
            self._mean_std(pixel_survival_array),
            self._mean_std(pixel_quantile_array),
            self._mean_std(peak_survival_array),
            self._mean_std(peak_quantile_array),
            self._mean_std(image_stat_array),
            np.asarray([
                np.log1p(len(records)),
                np.log1p(np.sum(total_pixels)),
                np.mean(np.log1p(total_pixels)),
                np.std(np.log1p(total_pixels)),
                np.mean(np.log1p(peak_counts)),
                np.std(np.log1p(peak_counts)),
            ], dtype=np.float32),
        ]).astype(np.float32)

        names: list[str] = []
        for prefix, base_names in (
            ("pixel_survival_log10", [f"t{value:g}" for value in config.survival_thresholds]),
            ("pixel_quantile", [f"q{value:g}" for value in config.quantiles]),
            ("peak_survival_per_mp_log10", [f"t{value:g}" for value in config.survival_thresholds]),
            ("peak_quantile", [f"q{value:g}" for value in config.quantiles]),
            ("image", list(expected_stat_names)),
        ):
            names.extend([f"{prefix}_{name}_mean" for name in base_names])
            names.extend([f"{prefix}_{name}_std" for name in base_names])
        names.extend([
            "window_log1p_num_images",
            "window_log1p_total_pixels",
            "image_log1p_pixels_mean",
            "image_log1p_pixels_std",
            "peak_log1p_per_mp_mean",
            "peak_log1p_per_mp_std",
        ])
        if len(names) != len(features):
            raise RuntimeError(f"Feature name mismatch: {len(names)} != {len(features)}")
        if not np.isfinite(features).all():
            raise ValueError("Extracted features contain NaN or infinity")
        return features, tuple(names)

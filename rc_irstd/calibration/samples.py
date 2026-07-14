from __future__ import annotations

"""Explicit calibration units for block-level and true image-shot CRC."""

from dataclasses import dataclass
import json

import numpy as np

from rc_irstd.episodes.dataset import EpisodeArrays
from rc_irstd.episodes.splits import split_iid_images


@dataclass(frozen=True)
class CalibrationSamples:
    pixel_risk: np.ndarray
    peak_risk: np.ndarray
    pd: np.ndarray
    domains: np.ndarray
    sequences: np.ndarray
    sample_ids: np.ndarray
    base_indices: np.ndarray
    base_rejected: np.ndarray
    parent_episode: np.ndarray
    protocols: np.ndarray
    label_count_per_sample: np.ndarray
    unit: str

    @property
    def num_samples(self) -> int:
        return len(self.sample_ids)


def _parse_ids(values: np.ndarray) -> list[list[str]]:
    result: list[list[str]] = []
    for value in values:
        parsed = json.loads(str(value))
        if not isinstance(parsed, list):
            raise ValueError("future_ids must encode JSON lists")
        result.append([str(item) for item in parsed])
    return result


def episode_calibration_samples(
    arrays: EpisodeArrays,
    base_indices: np.ndarray,
    base_rejected: np.ndarray,
) -> CalibrationSamples:
    protocols = (
        arrays.protocols
        if arrays.protocols is not None
        else np.asarray(["temporal"] * len(arrays.features))
    )
    return CalibrationSamples(
        pixel_risk=arrays.pixel_risk,
        peak_risk=arrays.peak_risk,
        pd=arrays.pd,
        domains=arrays.domains,
        sequences=arrays.sequences,
        sample_ids=np.asarray([f"episode_{index:08d}" for index in range(len(arrays.features))]),
        base_indices=np.asarray(base_indices, dtype=np.int64),
        base_rejected=np.asarray(base_rejected, dtype=bool),
        parent_episode=np.arange(len(arrays.features), dtype=np.int64),
        protocols=np.asarray(protocols).astype(str),
        label_count_per_sample=np.asarray(
            [len(ids) for ids in _parse_ids(arrays.future_ids)], dtype=np.int64
        ),
        unit="episode_block",
    )


def image_calibration_samples(
    arrays: EpisodeArrays,
    base_indices: np.ndarray,
    base_rejected: np.ndarray,
) -> CalibrationSamples:
    required = (
        arrays.future_pixel_risk,
        arrays.future_peak_risk,
        arrays.future_pd,
        arrays.future_gt_count,
    )
    if any(value is None for value in required):
        raise ValueError(
            "Episode file does not contain per-future-image curves. Rebuild it "
            "with the current rc-irstd-build-episodes command."
        )
    pixel = np.asarray(arrays.future_pixel_risk)
    peak = np.asarray(arrays.future_peak_risk)
    pd = np.asarray(arrays.future_pd).copy()
    gt_count = np.asarray(arrays.future_gt_count)
    if not (pixel.shape == peak.shape == pd.shape):
        raise ValueError("Per-image risk arrays have inconsistent shapes")
    if gt_count.shape != pixel.shape[:2]:
        raise ValueError("future_gt_count must have shape [episodes, horizon]")
    episodes, horizon, thresholds = pixel.shape
    ids_nested = _parse_ids(arrays.future_ids)
    if any(len(ids) != horizon for ids in ids_nested):
        raise ValueError("future_ids length does not match stored horizon")
    image_ids = np.asarray([item for row in ids_nested for item in row], dtype=np.str_)
    if len(np.unique(image_ids)) != len(image_ids):
        raise ValueError(
            "Image-shot CRC requires each labelled image once. Rebuild evaluation "
            "episodes with non-overlapping windows (stride >= context+horizon)."
        )
    parent = np.repeat(np.arange(episodes), horizon)
    domains = np.repeat(arrays.domains, horizon)
    sequences = np.repeat(arrays.sequences, horizon)
    protocols_source = (
        arrays.protocols
        if arrays.protocols is not None
        else np.asarray(["temporal"] * episodes)
    )
    protocols = np.repeat(protocols_source, horizon)
    flattened_pd = pd.reshape(-1, thresholds)
    flattened_gt = gt_count.reshape(-1)
    # Empty-target images provide false-alarm evidence but no target-detection
    # denominator. Mark Pd as NaN so they do not depress target-bearing Pd.
    flattened_pd[flattened_gt == 0] = np.nan
    return CalibrationSamples(
        pixel_risk=pixel.reshape(-1, thresholds),
        peak_risk=peak.reshape(-1, thresholds),
        pd=flattened_pd,
        domains=domains,
        sequences=sequences,
        sample_ids=image_ids,
        base_indices=np.repeat(np.asarray(base_indices, dtype=np.int64), horizon),
        base_rejected=np.repeat(np.asarray(base_rejected, dtype=bool), horizon),
        parent_episode=parent,
        protocols=np.asarray(protocols).astype(str),
        label_count_per_sample=np.ones(episodes * horizon, dtype=np.int64),
        unit="image",
    )


def split_calibration_samples(
    samples: CalibrationSamples,
    calibration_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Split exact labelled units while retaining an independent test partition."""
    if calibration_size <= 0:
        raise ValueError("calibration_size must be positive")
    if samples.unit == "image" and np.all(samples.protocols == "iid"):
        calibration, test = split_iid_images(
            samples.sample_ids, calibration_size, seed
        )
        return calibration, test, {
            "strategy": "iid_unique_images",
            "independent_groups": samples.num_samples,
        }

    # Temporal images and block episodes are sequence-blocked. calibration_size
    # still counts the selected labelled samples, while unused samples from a
    # selected sequence are discarded rather than leaked into test.
    groups = np.asarray(
        [f"{d}::{s}" for d, s in zip(samples.domains, samples.sequences, strict=True)]
    )
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Sequence-blocked calibration requires at least two groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    selected_groups: list[str] = []
    available = 0
    for group in unique[:-1]:
        selected_groups.append(str(group))
        available += int(np.sum(groups == group))
        if available >= calibration_size:
            break
    if available < calibration_size:
        raise ValueError("Not enough samples while retaining a disjoint test group")
    pool = np.flatnonzero(np.isin(groups, selected_groups))
    rng.shuffle(pool)
    calibration = np.sort(pool[:calibration_size])
    test = np.flatnonzero(~np.isin(groups, selected_groups))
    return calibration, test, {
        "strategy": "sequence_blocked_exact_samples",
        "independent_groups": len(unique),
        "selected_calibration_groups": selected_groups,
    }

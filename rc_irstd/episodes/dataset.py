from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EpisodeArrays:
    features: np.ndarray
    pixel_log_risk: np.ndarray
    peak_log_risk: np.ndarray
    pixel_risk: np.ndarray
    peak_risk: np.ndarray
    pd: np.ndarray
    context_pixel_upper: np.ndarray
    context_peak_upper: np.ndarray
    thresholds: np.ndarray
    domains: np.ndarray
    sequences: np.ndarray
    context_ids: np.ndarray
    future_ids: np.ndarray
    feature_names: tuple[str, ...]
    feature_config: dict[str, object] = field(default_factory=dict)
    protocols: np.ndarray | None = None
    future_pixel_risk: np.ndarray | None = None
    future_peak_risk: np.ndarray | None = None
    future_pd: np.ndarray | None = None
    future_gt_count: np.ndarray | None = None

    def subset(self, indices: np.ndarray) -> "EpisodeArrays":
        indices = np.asarray(indices, dtype=np.int64)
        optional = lambda value: None if value is None else value[indices]
        return EpisodeArrays(
            features=self.features[indices],
            pixel_log_risk=self.pixel_log_risk[indices],
            peak_log_risk=self.peak_log_risk[indices],
            pixel_risk=self.pixel_risk[indices],
            peak_risk=self.peak_risk[indices],
            pd=self.pd[indices],
            context_pixel_upper=self.context_pixel_upper[indices],
            context_peak_upper=self.context_peak_upper[indices],
            thresholds=self.thresholds,
            domains=self.domains[indices],
            sequences=self.sequences[indices],
            context_ids=self.context_ids[indices],
            future_ids=self.future_ids[indices],
            feature_names=self.feature_names,
            feature_config=dict(self.feature_config),
            protocols=optional(self.protocols),
            future_pixel_risk=optional(self.future_pixel_risk),
            future_peak_risk=optional(self.future_peak_risk),
            future_pd=optional(self.future_pd),
            future_gt_count=optional(self.future_gt_count),
        )


def _optional(payload, name: str, dtype):
    return np.asarray(payload[name], dtype=dtype) if name in payload else None


def load_episode_file(path: str | Path) -> EpisodeArrays:
    with np.load(path, allow_pickle=False) as payload:
        pixel_risk = np.asarray(payload["pixel_risk"], dtype=np.float32)
        peak_risk = np.asarray(payload["peak_risk"], dtype=np.float32)
        context_pixel_upper = np.asarray(
            payload["context_pixel_upper"]
            if "context_pixel_upper" in payload
            else np.full_like(pixel_risk, np.nan),
            dtype=np.float32,
        )
        context_peak_upper = np.asarray(
            payload["context_peak_upper"]
            if "context_peak_upper" in payload
            else np.full_like(peak_risk, np.nan),
            dtype=np.float32,
        )
        return EpisodeArrays(
            features=np.asarray(payload["features"], dtype=np.float32),
            pixel_log_risk=np.asarray(payload["pixel_log_risk"], dtype=np.float32),
            peak_log_risk=np.asarray(payload["peak_log_risk"], dtype=np.float32),
            pixel_risk=pixel_risk,
            peak_risk=peak_risk,
            pd=np.asarray(payload["pd"], dtype=np.float32),
            context_pixel_upper=context_pixel_upper,
            context_peak_upper=context_peak_upper,
            thresholds=np.asarray(payload["thresholds"], dtype=np.float32),
            domains=np.asarray(payload["domains"]).astype(str),
            sequences=np.asarray(payload["sequences"]).astype(str),
            context_ids=np.asarray(payload["context_ids"]).astype(str),
            future_ids=np.asarray(payload["future_ids"]).astype(str),
            feature_names=tuple(np.asarray(payload["feature_names"]).astype(str).tolist()),
            feature_config=(
                json.loads(str(np.asarray(payload["feature_config_json"]).item()))
                if "feature_config_json" in payload
                else {}
            ),
            protocols=(
                np.asarray(payload["protocols"]).astype(str)
                if "protocols" in payload
                else None
            ),
            future_pixel_risk=_optional(payload, "future_pixel_risk", np.float32),
            future_peak_risk=_optional(payload, "future_peak_risk", np.float32),
            future_pd=_optional(payload, "future_pd", np.float32),
            future_gt_count=_optional(payload, "future_gt_count", np.int32),
        )


def _concat_optional(arrays: list[EpisodeArrays], name: str):
    values = [getattr(item, name) for item in arrays]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Episode files disagree on optional field {name}")
    return np.concatenate(values, axis=0)


def concatenate_episode_files(paths: Sequence[str | Path]) -> EpisodeArrays:
    arrays = [load_episode_file(path) for path in paths]
    if not arrays:
        raise ValueError("At least one episode file is required")
    reference = arrays[0]
    for current in arrays[1:]:
        if not np.array_equal(current.thresholds, reference.thresholds):
            raise ValueError("Episode files use different threshold grids")
        if current.feature_names != reference.feature_names:
            raise ValueError("Episode files use different feature schemas")
        if current.feature_config != reference.feature_config:
            raise ValueError("Episode files use different feature configurations")
    return EpisodeArrays(
        features=np.concatenate([item.features for item in arrays], axis=0),
        pixel_log_risk=np.concatenate([item.pixel_log_risk for item in arrays], axis=0),
        peak_log_risk=np.concatenate([item.peak_log_risk for item in arrays], axis=0),
        pixel_risk=np.concatenate([item.pixel_risk for item in arrays], axis=0),
        peak_risk=np.concatenate([item.peak_risk for item in arrays], axis=0),
        pd=np.concatenate([item.pd for item in arrays], axis=0),
        context_pixel_upper=np.concatenate([item.context_pixel_upper for item in arrays], axis=0),
        context_peak_upper=np.concatenate([item.context_peak_upper for item in arrays], axis=0),
        thresholds=reference.thresholds,
        domains=np.concatenate([item.domains for item in arrays]),
        sequences=np.concatenate([item.sequences for item in arrays]),
        context_ids=np.concatenate([item.context_ids for item in arrays]),
        future_ids=np.concatenate([item.future_ids for item in arrays]),
        feature_names=reference.feature_names,
        feature_config=dict(reference.feature_config),
        protocols=_concat_optional(arrays, "protocols"),
        future_pixel_risk=_concat_optional(arrays, "future_pixel_risk"),
        future_peak_risk=_concat_optional(arrays, "future_peak_risk"),
        future_pd=_concat_optional(arrays, "future_pd"),
        future_gt_count=_concat_optional(arrays, "future_gt_count"),
    )


class RiskCurveDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        arrays: EpisodeArrays,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> None:
        self.arrays = arrays
        self.features = ((arrays.features - feature_mean) / feature_std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features[index]),
            "pixel_log_risk": torch.from_numpy(self.arrays.pixel_log_risk[index]),
            "peak_log_risk": torch.from_numpy(self.arrays.peak_log_risk[index]),
        }

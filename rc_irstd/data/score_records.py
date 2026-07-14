from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScoreRecord:
    probability: np.ndarray
    mask: np.ndarray | None
    image_stats: np.ndarray
    image_stat_names: tuple[str, ...]
    image_id: str
    dataset_name: str
    sequence_id: str
    frame_index: int
    original_hw: tuple[int, int]
    source_checkpoint: str = ""
    dataset_type: str = "iid_images"
    inference_mode: str = "resize"

    @property
    def total_pixels(self) -> int:
        return int(self.probability.size)


def save_score_record(record: ScoreRecord, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "probability": np.asarray(record.probability, dtype=np.float32),
        "image_stats": np.asarray(record.image_stats, dtype=np.float32),
        "image_stat_names": np.asarray(record.image_stat_names, dtype=np.str_),
        "image_id": np.asarray(record.image_id),
        "dataset_name": np.asarray(record.dataset_name),
        "sequence_id": np.asarray(record.sequence_id),
        "frame_index": np.asarray(record.frame_index, dtype=np.int64),
        "original_hw": np.asarray(record.original_hw, dtype=np.int32),
        "source_checkpoint": np.asarray(record.source_checkpoint),
        "dataset_type": np.asarray(record.dataset_type),
        "inference_mode": np.asarray(record.inference_mode),
        "has_mask": np.asarray(record.mask is not None),
    }
    if record.mask is not None:
        payload["mask"] = np.asarray(record.mask, dtype=np.uint8)
    np.savez_compressed(path, **payload)


def _scalar_string(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def load_score_record(
    path: str | Path,
    require_mask: bool = False,
    *,
    load_mask: bool = True,
) -> ScoreRecord:
    with np.load(path, allow_pickle=False) as payload:
        has_mask = bool(np.asarray(payload.get("has_mask", "mask" in payload)).item())
        if require_mask and not load_mask:
            raise ValueError("require_mask=True is incompatible with load_mask=False")
        mask = (
            np.asarray(payload["mask"], dtype=np.uint8)
            if load_mask and has_mask and "mask" in payload
            else None
        )
        if require_mask and mask is None:
            raise ValueError(f"Score record {path} does not contain a mask")
        probability = np.asarray(payload["probability"], dtype=np.float32).squeeze()
        if probability.ndim != 2:
            raise ValueError(f"Probability in {path} must be 2-D")
        if not np.isfinite(probability).all():
            raise ValueError(f"Probability in {path} contains invalid values")
        return ScoreRecord(
            probability=probability,
            mask=mask.squeeze() if mask is not None else None,
            image_stats=np.asarray(payload["image_stats"], dtype=np.float32),
            image_stat_names=tuple(np.asarray(payload["image_stat_names"]).astype(str).tolist()),
            image_id=_scalar_string(payload["image_id"]),
            dataset_name=_scalar_string(payload["dataset_name"]),
            sequence_id=_scalar_string(payload["sequence_id"]),
            frame_index=int(np.asarray(payload["frame_index"]).item()),
            original_hw=tuple(int(x) for x in np.asarray(payload["original_hw"]).tolist()),
            source_checkpoint=_scalar_string(payload.get("source_checkpoint", np.asarray(""))),
            dataset_type=_scalar_string(
                payload.get("dataset_type", np.asarray("iid_images"))
            ),
            inference_mode=_scalar_string(
                payload.get("inference_mode", np.asarray("resize"))
            ),
        )

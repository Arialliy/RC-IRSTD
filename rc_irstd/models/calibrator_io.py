from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rc_irstd.models.monotone_pixel_calibrator import MonotonePixelCalibrator
from rc_irstd.models.risk_curve import FeatureNormaliser


@dataclass(frozen=True)
class LoadedMonotoneCalibrator:
    model: MonotonePixelCalibrator
    normaliser: FeatureNormaliser
    feature_names: tuple[str, ...]
    feature_config: dict[str, Any]
    budgets: np.ndarray
    checkpoint: dict[str, Any]


def load_monotone_calibrator(
    path: str | Path,
    device: torch.device,
) -> LoadedMonotoneCalibrator:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("reject_head") is not False:
        raise ValueError("The main no-reject path requires reject_head=False")
    config = dict(payload["model_config"])
    model = MonotonePixelCalibrator(
        input_dim=int(config["input_dim"]),
        budgets=config["budgets"],
        hidden_dim=int(config.get("hidden_dim", 192)),
        source_hidden_dim=int(config.get("source_hidden_dim", 32)),
        source_output_dim=int(config.get("source_output_dim", 64)),
        dropout=float(config.get("dropout", 0.10)),
        min_logit_step=float(config.get("min_logit_step", 0.0)),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return LoadedMonotoneCalibrator(
        model=model,
        normaliser=FeatureNormaliser.from_dict(payload["feature_normaliser"]),
        feature_names=tuple(str(value) for value in payload["feature_names"]),
        feature_config=dict(payload["feature_config"]),
        budgets=np.asarray(payload["budgets"], dtype=np.float32),
        checkpoint=payload,
    )


def predict_threshold_curve(
    loaded: LoadedMonotoneCalibrator,
    features: np.ndarray,
    device: torch.device,
    *,
    source_distances: np.ndarray | None = None,
) -> np.ndarray:
    normalised = loaded.normaliser.transform(np.asarray(features, dtype=np.float32))
    feature_tensor = torch.from_numpy(normalised).to(device)
    source_tensor = None
    source_mask = None
    if source_distances is not None:
        source_tensor = torch.from_numpy(np.asarray(source_distances, dtype=np.float32)).to(device)
        source_mask = torch.ones_like(source_tensor, dtype=torch.bool)
    with torch.inference_mode():
        output = loaded.model(feature_tensor, source_tensor, source_mask)
    return output["threshold_logit"].cpu().numpy()

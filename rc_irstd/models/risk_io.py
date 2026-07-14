from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rc_irstd.models.risk_curve import FeatureNormaliser, RiskCurvePredictor


@dataclass(frozen=True)
class LoadedRiskModel:
    model: RiskCurvePredictor
    normaliser: FeatureNormaliser
    thresholds: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]


def load_risk_model(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedRiskModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"model", "normaliser", "thresholds", "feature_names", "model_config"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"Risk checkpoint is missing fields: {sorted(missing)}")
    config = payload["model_config"]
    model = RiskCurvePredictor(
        input_dim=int(config["input_dim"]),
        num_thresholds=int(config["num_thresholds"]),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return LoadedRiskModel(
        model=model,
        normaliser=FeatureNormaliser.from_dict(payload["normaliser"]),
        thresholds=np.asarray(payload["thresholds"], dtype=np.float32),
        feature_names=tuple(str(value) for value in payload["feature_names"]),
        metadata={
            key: value
            for key, value in payload.items()
            if key not in {"model", "normaliser", "thresholds", "feature_names"}
        },
    )


def predict_risk_curves(
    loaded: LoadedRiskModel,
    features: np.ndarray,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 1:
        array = array[None]
    normalised = loaded.normaliser.transform(array)
    outputs: dict[str, list[np.ndarray]] = {
        "pixel_log_risk": [],
        "peak_log_risk": [],
    }
    with torch.inference_mode():
        for start in range(0, len(normalised), batch_size):
            tensor = torch.from_numpy(normalised[start : start + batch_size]).to(device)
            prediction = loaded.model(tensor)
            for key in outputs:
                outputs[key].append(prediction[key].detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in outputs.items()}

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MonotoneLogRiskHead(nn.Module):
    """Produce a non-increasing log-risk curve over ascending thresholds.

    A naive cumulative-softplus head accumulates roughly ``0.69`` at every
    threshold at initialization, which explodes when the grid contains hundreds
    of points. Here a learned positive *total drop* is distributed across the
    threshold intervals by a softmax. This preserves exact monotonicity while
    keeping the initial curve numerically well scaled.
    """

    def __init__(self, hidden_dim: int, num_thresholds: int) -> None:
        super().__init__()
        if num_thresholds < 2:
            raise ValueError("num_thresholds must be at least 2")
        self.num_thresholds = num_thresholds
        # start value + total positive drop + interval allocation logits
        self.projection = nn.Linear(hidden_dim, num_thresholds + 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.projection(hidden)
        start = raw[:, :1]
        total_drop = F.softplus(raw[:, 1:2])
        allocation = torch.softmax(raw[:, 2:], dim=1)
        decrements = total_drop * allocation
        tail = start - torch.cumsum(decrements, dim=1)
        return torch.cat([start, tail], dim=1)


class RiskCurvePredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_thresholds: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_thresholds = int(num_thresholds)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pixel_head = MonotoneLogRiskHead(hidden_dim, num_thresholds)
        self.peak_head = MonotoneLogRiskHead(hidden_dim, num_thresholds)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(features)
        return {
            "pixel_log_risk": self.pixel_head(hidden),
            "peak_log_risk": self.peak_head(hidden),
        }


@dataclass(frozen=True)
class FeatureNormaliser:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, min_std: float = 1e-6) -> "FeatureNormaliser":
        mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = features.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(std, min_std)
        return cls(mean, std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureNormaliser":
        return cls(
            np.asarray(payload["mean"], dtype=np.float32),
            np.asarray(payload["std"], dtype=np.float32),
        )

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ThresholdUpdate:
    sequence_id: str
    update_index: int
    warmup_ids: tuple[str, ...]
    base_threshold_index: int
    offset_index: int
    final_threshold_index: int
    threshold: float
    predicted_pixel_risk: float
    predicted_peak_risk_per_mp: float
    rejected: bool
    feature_ood_score: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warmup_ids"] = list(self.warmup_ids)
        return payload


@dataclass
class DeploymentState:
    detector_checkpoint: str
    curve_checkpoint: str
    score_directory: str
    pixel_budget: float
    peak_budget_per_mp: float
    warmup_size: int
    offset_index: int = 0
    updates: list[ThresholdUpdate] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add(self, update: ThresholdUpdate) -> None:
        self.updates.append(update)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_checkpoint": self.detector_checkpoint,
            "curve_checkpoint": self.curve_checkpoint,
            "score_directory": self.score_directory,
            "pixel_budget": self.pixel_budget,
            "peak_budget_per_mp": self.peak_budget_per_mp,
            "warmup_size": self.warmup_size,
            "offset_index": self.offset_index,
            "created_at": self.created_at,
            "updates": [item.to_dict() for item in self.updates],
        }

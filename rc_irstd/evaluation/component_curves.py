from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from rc_irstd.evaluation.irstd_metrics import evaluate_irstd_at_threshold


@dataclass(frozen=True)
class ComponentCurveRow:
    threshold: float
    pd: float
    false_components_per_mp: float
    false_pixel_rate: float
    iou: float
    niou: float
    hiou: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_component_curve(
    probabilities: list[np.ndarray],
    masks: list[np.ndarray],
    thresholds: np.ndarray,
    object_tolerance: float = 2.0,
) -> list[ComponentCurveRow]:
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or np.any(np.diff(thresholds) < 0):
        raise ValueError("thresholds must be an ascending 1-D array")
    rows: list[ComponentCurveRow] = []
    for threshold in thresholds:
        metrics = evaluate_irstd_at_threshold(
            probabilities, masks, float(threshold), object_tolerance
        )
        rows.append(
            ComponentCurveRow(
                threshold=float(threshold),
                pd=metrics.pd,
                false_components_per_mp=metrics.false_components_per_mp,
                false_pixel_rate=metrics.false_pixel_rate,
                iou=metrics.iou,
                niou=metrics.niou,
                hiou=metrics.hiou,
                precision=metrics.precision,
                recall=metrics.recall,
                f1=metrics.f1,
            )
        )
    return rows

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    index: int
    threshold: float
    rejected: bool
    predicted_pixel_risk: float
    predicted_peak_risk: float


def select_dual_budget_threshold(
    thresholds: np.ndarray,
    pixel_log_risk: np.ndarray,
    peak_log_risk: np.ndarray,
    pixel_budget: float,
    peak_budget_per_mp: float,
) -> OperatingPoint:
    thresholds = np.asarray(thresholds, dtype=np.float64)
    pixel = np.asarray(pixel_log_risk, dtype=np.float64)
    peak = np.asarray(peak_log_risk, dtype=np.float64)
    if not (len(thresholds) == len(pixel) == len(peak)):
        raise ValueError("Threshold and risk curves must have equal lengths")
    if pixel_budget <= 0 or peak_budget_per_mp <= 0:
        raise ValueError("Budgets must be positive")
    feasible = np.flatnonzero(
        (pixel <= np.log10(pixel_budget))
        & (peak <= np.log10(peak_budget_per_mp))
    )
    if len(feasible) == 0:
        index = len(thresholds) - 1
        return OperatingPoint(
            index=index,
            threshold=float(thresholds[index]),
            rejected=True,
            predicted_pixel_risk=float(10 ** pixel[index]),
            predicted_peak_risk=float(10 ** peak[index]),
        )
    index = int(feasible[0])
    return OperatingPoint(
        index=index,
        threshold=float(thresholds[index]),
        # A threshold above one is the explicit empty-prediction/abstention
        # action included in the formal threshold grid.
        rejected=bool(thresholds[index] > 1.0),
        predicted_pixel_risk=float(10 ** pixel[index]),
        predicted_peak_risk=float(10 ** peak[index]),
    )

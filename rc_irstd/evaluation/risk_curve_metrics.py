from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rc_irstd.evaluation.budget import BudgetSummary, summarise_selected_points
from rc_irstd.evaluation.operating_point import select_dual_budget_threshold


@dataclass(frozen=True)
class RiskCurveMetrics:
    pixel_log_mae: float
    peak_log_mae: float
    pixel_pointwise_coverage: float
    peak_pointwise_coverage: float
    joint_pointwise_coverage: float
    pixel_underestimation_mae: float
    peak_underestimation_mae: float
    monotonicity_violations: int
    selected: BudgetSummary

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected"] = self.selected.to_dict()
        return payload


def select_indices_from_predictions(
    thresholds: np.ndarray,
    predicted_pixel_log: np.ndarray,
    predicted_peak_log: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> tuple[np.ndarray, np.ndarray]:
    pixel = np.asarray(predicted_pixel_log)
    peak = np.asarray(predicted_peak_log)
    if pixel.shape != peak.shape or pixel.ndim != 2:
        raise ValueError("Predicted risk curves must share shape [samples, thresholds]")
    indices: list[int] = []
    rejected: list[bool] = []
    for pixel_curve, peak_curve in zip(pixel, peak, strict=True):
        point = select_dual_budget_threshold(
            thresholds,
            pixel_curve,
            peak_curve,
            pixel_budget,
            peak_budget,
        )
        indices.append(point.index)
        rejected.append(point.rejected)
    return np.asarray(indices, dtype=np.int64), np.asarray(rejected, dtype=bool)


def evaluate_risk_curve_predictions(
    thresholds: np.ndarray,
    predicted_pixel_log: np.ndarray,
    predicted_peak_log: np.ndarray,
    true_pixel_log: np.ndarray,
    true_peak_log: np.ndarray,
    true_pixel_risk: np.ndarray,
    true_peak_risk: np.ndarray,
    true_pd: np.ndarray,
    domains: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> tuple[RiskCurveMetrics, np.ndarray, np.ndarray]:
    predicted_pixel_log = np.asarray(predicted_pixel_log, dtype=np.float64)
    predicted_peak_log = np.asarray(predicted_peak_log, dtype=np.float64)
    true_pixel_log = np.asarray(true_pixel_log, dtype=np.float64)
    true_peak_log = np.asarray(true_peak_log, dtype=np.float64)
    if not (
        predicted_pixel_log.shape
        == predicted_peak_log.shape
        == true_pixel_log.shape
        == true_peak_log.shape
    ):
        raise ValueError("All log-risk arrays must have equal shapes")

    indices, rejected = select_indices_from_predictions(
        thresholds,
        predicted_pixel_log,
        predicted_peak_log,
        pixel_budget,
        peak_budget,
    )
    rows = np.arange(len(indices))
    selected_summary = summarise_selected_points(
        np.asarray(true_pixel_risk)[rows, indices],
        np.asarray(true_peak_risk)[rows, indices],
        np.asarray(true_pd)[rows, indices],
        rejected,
        domains,
        pixel_budget,
        peak_budget,
    )
    pixel_error = predicted_pixel_log - true_pixel_log
    peak_error = predicted_peak_log - true_peak_log
    monotonicity = int(
        np.sum(np.diff(predicted_pixel_log, axis=1) > 1e-8)
        + np.sum(np.diff(predicted_peak_log, axis=1) > 1e-8)
    )
    metrics = RiskCurveMetrics(
        pixel_log_mae=float(np.abs(pixel_error).mean()),
        peak_log_mae=float(np.abs(peak_error).mean()),
        pixel_pointwise_coverage=float((pixel_error >= 0).mean()),
        peak_pointwise_coverage=float((peak_error >= 0).mean()),
        joint_pointwise_coverage=float(((pixel_error >= 0) & (peak_error >= 0)).mean()),
        pixel_underestimation_mae=float(np.maximum(-pixel_error, 0.0).mean()),
        peak_underestimation_mae=float(np.maximum(-peak_error, 0.0).mean()),
        monotonicity_violations=monotonicity,
        selected=selected_summary,
    )
    return metrics, indices, rejected

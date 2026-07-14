from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class CRCResult:
    selected_parameter: int
    selected_position: int
    empirical_risk: float
    corrected_risk: float
    alpha: float
    calibration_size: int
    feasible: bool
    minimum_possible_corrected_risk: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def minimum_calibration_size(alpha: float) -> int:
    """Smallest ``m`` for which ``1/(m+1) <= alpha`` can hold."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return int(np.ceil(1.0 / alpha - 1.0))


def corrected_empirical_risk(empirical_risk: np.ndarray, calibration_size: int) -> np.ndarray:
    if calibration_size <= 0:
        raise ValueError("calibration_size must be positive")
    empirical = np.asarray(empirical_risk, dtype=np.float64)
    return (
        calibration_size / (calibration_size + 1.0) * empirical
        + 1.0 / (calibration_size + 1.0)
    )


def _validate_loss_matrix(losses: np.ndarray) -> np.ndarray:
    matrix = np.asarray(losses, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("losses must have shape [calibration_samples, parameters]")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("losses must be non-empty")
    if not np.isfinite(matrix).all():
        raise ValueError("losses contain NaN or infinity")
    if np.any(matrix < -1e-12) or np.any(matrix > 1.0 + 1e-12):
        raise ValueError("CRC losses must lie in [0, 1]")
    return np.clip(matrix, 0.0, 1.0)


def select_crc_parameter(
    losses: np.ndarray,
    parameters: Iterable[int],
    alpha: float,
    require_monotone: bool = True,
) -> CRCResult:
    """Select the least conservative parameter satisfying standard CRC.

    The columns of ``losses`` must follow increasing conservatism. For the
    adaptive-threshold implementation, columns correspond to non-negative
    threshold-index offsets. For the raw global baseline, columns correspond to
    ascending threshold indices.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    matrix = _validate_loss_matrix(losses)
    parameter_values = np.asarray(list(parameters), dtype=np.int64)
    if len(parameter_values) != matrix.shape[1]:
        raise ValueError("Number of parameters must equal the loss-matrix width")
    if np.any(np.diff(parameter_values) < 0):
        raise ValueError("parameters must be ascending")
    if require_monotone and np.any(np.diff(matrix, axis=1) > 1e-10):
        bad = int(np.sum(np.diff(matrix, axis=1) > 1e-10))
        raise ValueError(
            f"Loss family is not nested/monotone; found {bad} increasing entries"
        )

    calibration_size = matrix.shape[0]
    empirical = matrix.mean(axis=0)
    corrected = corrected_empirical_risk(empirical, calibration_size)
    feasible_positions = np.flatnonzero(corrected <= alpha + 1e-12)
    minimum_possible = 1.0 / (calibration_size + 1.0)

    if len(feasible_positions):
        position = int(feasible_positions[0])
        return CRCResult(
            selected_parameter=int(parameter_values[position]),
            selected_position=position,
            empirical_risk=float(empirical[position]),
            corrected_risk=float(corrected[position]),
            alpha=float(alpha),
            calibration_size=int(calibration_size),
            feasible=True,
            minimum_possible_corrected_risk=float(minimum_possible),
            message="A CRC-feasible parameter was found.",
        )

    position = len(parameter_values) - 1
    if minimum_possible > alpha + 1e-12:
        message = (
            f"No formal solution is possible with m={calibration_size} and "
            f"alpha={alpha:g}: even zero empirical loss gives "
            f"1/(m+1)={minimum_possible:.6g}."
        )
    else:
        message = (
            "No parameter in the supplied nested family satisfies the corrected "
            "risk bound; the most conservative parameter is returned as a "
            "fallback and must not be labelled certified."
        )
    return CRCResult(
        selected_parameter=int(parameter_values[position]),
        selected_position=position,
        empirical_risk=float(empirical[position]),
        corrected_risk=float(corrected[position]),
        alpha=float(alpha),
        calibration_size=int(calibration_size),
        feasible=False,
        minimum_possible_corrected_risk=float(minimum_possible),
        message=message,
    )


def selected_indices_from_offsets(
    base_indices: np.ndarray,
    offsets: Iterable[int],
    num_thresholds: int,
) -> np.ndarray:
    base = np.asarray(base_indices, dtype=np.int64)
    offset_values = np.asarray(list(offsets), dtype=np.int64)
    if num_thresholds <= 0:
        raise ValueError("num_thresholds must be positive")
    if np.any(base < 0) or np.any(base >= num_thresholds):
        raise ValueError("base_indices are outside the threshold grid")
    if np.any(offset_values < 0):
        raise ValueError("offsets must be non-negative")
    return np.minimum(base[:, None] + offset_values[None, :], num_thresholds - 1)


def joint_budget_violation_losses(
    pixel_risk_curves: np.ndarray,
    peak_risk_curves: np.ndarray,
    selected_indices: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> np.ndarray:
    pixel = np.asarray(pixel_risk_curves, dtype=np.float64)
    peak = np.asarray(peak_risk_curves, dtype=np.float64)
    indices = np.asarray(selected_indices, dtype=np.int64)
    if pixel.shape != peak.shape:
        raise ValueError("Pixel and peak risk curves must have equal shapes")
    if pixel.ndim != 2:
        raise ValueError("Risk curves must have shape [samples, thresholds]")
    if indices.ndim == 1:
        indices = indices[:, None]
    if indices.shape[0] != pixel.shape[0]:
        raise ValueError("selected_indices must have one row per sample")
    rows = np.arange(pixel.shape[0])[:, None]
    selected_pixel = pixel[rows, indices]
    selected_peak = peak[rows, indices]
    return ((selected_pixel > pixel_budget) | (selected_peak > peak_budget)).astype(np.float64)


def adaptive_offset_loss_matrix(
    pixel_risk_curves: np.ndarray,
    peak_risk_curves: np.ndarray,
    base_indices: np.ndarray,
    offsets: Iterable[int],
    pixel_budget: float,
    peak_budget: float,
) -> tuple[np.ndarray, np.ndarray]:
    pixel = np.asarray(pixel_risk_curves)
    indices = selected_indices_from_offsets(base_indices, offsets, pixel.shape[1])
    losses = joint_budget_violation_losses(
        pixel_risk_curves,
        peak_risk_curves,
        indices,
        pixel_budget,
        peak_budget,
    )
    return losses, indices


def raw_global_threshold_loss_matrix(
    pixel_risk_curves: np.ndarray,
    peak_risk_curves: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> np.ndarray:
    pixel = np.asarray(pixel_risk_curves, dtype=np.float64)
    peak = np.asarray(peak_risk_curves, dtype=np.float64)
    if pixel.shape != peak.shape or pixel.ndim != 2:
        raise ValueError("Risk curves must share shape [samples, thresholds]")
    return ((pixel > pixel_budget) | (peak > peak_budget)).astype(np.float64)

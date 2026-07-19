"""Exact-rational RC5 oracle selection on a verified query event curve.

This module is a pure selection core.  It does not open labels, scores or
collections and it never converts a binary floating-point budget into a
rational count.  Each frozen budget ``a / b`` uses the integer feasibility
rule ``fp_pixels <= (a * total_native_pixels) // b``.

Among feasible event rows the diagnostic/training oracle maximizes matched
objects, then minimizes false-positive pixels, then chooses the largest
threshold.  Because tighter-budget feasible sets are nested suffixes and this
same total ordering is used for every budget, selected indices must be
nondecreasing from loose to strict even when component-level TP is itself
nonmonotone along the threshold sweep.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from model.endpoint_aware_threshold import encode_probability_numpy
from rc.stage2_context_tail_anchor import BUDGET_RATIONALS


ORACLE_SELECTION_SCHEMA = "rc-irstd.stage2-exact-rational-oracle.v2"
ORACLE_SELECTION_RULE = (
    "per-budget-feasible-max-tp-min-fp-max-threshold-v2"
)
BUDGET_COUNT_RULE = "(numerator * total_native_pixels) // denominator"


class Stage2ExactOracleError(ValueError):
    """The exact query curve or oracle selection contract was violated."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Stage2ExactOracleError(f"{name} must be int >= {minimum}")
    return value


def _budgets(
    value: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != len(BUDGET_RATIONALS):
        raise Stage2ExactOracleError(
            "oracle selection requires the frozen exact RC5 budget rationals"
        )
    normalized: list[tuple[int, int]] = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise Stage2ExactOracleError(
                "oracle selection requires canonical integer budget pairs"
            )
        numerator, denominator = pair
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or numerator <= 0
            or denominator <= 0
        ):
            raise Stage2ExactOracleError(
                "oracle selection requires canonical integer budget pairs"
            )
        normalized.append((numerator, denominator))
    if tuple(normalized) != BUDGET_RATIONALS:
        raise Stage2ExactOracleError(
            "oracle selection requires the frozen exact RC5 budget rationals"
        )
    return BUDGET_RATIONALS


def _float64_vector(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an explicit numpy array")
    if value.dtype != np.float64 or value.ndim != 1 or value.size < 2:
        raise Stage2ExactOracleError(
            f"{name} must be one float64 vector with at least two rows"
        )
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise Stage2ExactOracleError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _int64_vector(value: Any, name: str, *, size: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an explicit numpy array")
    if value.dtype != np.int64 or value.ndim != 1 or value.size != size:
        raise Stage2ExactOracleError(
            f"{name} must be one int64 vector aligned with thresholds"
        )
    result = np.array(value, dtype=np.int64, order="C", copy=True)
    if np.any(result < 0):
        raise Stage2ExactOracleError(f"{name} must contain non-negative counts")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ExactOracleSelectionV2:
    """Immutable selected rows and derived exact-curve supervision values."""

    selected_indices: tuple[int, int, int]
    allowed_false_positive_counts: tuple[int, int, int]
    thresholds: np.ndarray
    coordinates: np.ndarray
    matched_objects: np.ndarray
    false_positive_pixels: np.ndarray
    detection_probability: np.ndarray
    pixel_false_alarm_rate: np.ndarray
    total_native_pixels: int
    ground_truth_objects: int


def select_exact_oracle_v2(
    *,
    thresholds: np.ndarray,
    false_positive_pixels: np.ndarray,
    matched_objects: np.ndarray,
    total_native_pixels: int,
    ground_truth_objects: int,
    budget_rationals: Sequence[Sequence[int]] = BUDGET_RATIONALS,
) -> ExactOracleSelectionV2:
    """Select the three frozen RC5 oracle rows without float count logic."""

    budgets = _budgets(budget_rationals)
    threshold = _float64_vector(thresholds, "thresholds")
    if (
        threshold[0] != 0.0
        or threshold[-1] != 1.0
        or np.any((threshold < 0.0) | (threshold > 1.0))
        or np.any(np.diff(threshold) <= 0.0)
    ):
        raise Stage2ExactOracleError(
            "thresholds must be strictly ascending in [0,1] with exact endpoints"
        )
    fp = _int64_vector(
        false_positive_pixels,
        "false_positive_pixels",
        size=threshold.size,
    )
    tp = _int64_vector(
        matched_objects,
        "matched_objects",
        size=threshold.size,
    )
    total = _strict_int(
        total_native_pixels, "total_native_pixels", minimum=1
    )
    objects = _strict_int(
        ground_truth_objects, "ground_truth_objects", minimum=0
    )
    if np.any(fp > total):
        raise Stage2ExactOracleError(
            "false_positive_pixels cannot exceed total native pixels"
        )
    if np.any(tp > objects):
        raise Stage2ExactOracleError(
            "matched_objects cannot exceed ground_truth_objects"
        )
    if np.any(fp[1:] > fp[:-1]):
        raise Stage2ExactOracleError(
            "false-positive pixel counts must be nonincreasing with threshold"
        )
    if fp[-1] != 0 or tp[-1] != 0:
        raise Stage2ExactOracleError(
            "the strict-greater exact threshold=1 endpoint must predict nothing"
        )

    allowed = tuple((numerator * total) // denominator for numerator, denominator in budgets)
    selected: list[int] = []
    for maximum in allowed:
        feasible = np.flatnonzero(fp <= maximum)
        if feasible.size == 0:
            raise Stage2ExactOracleError(
                "exact threshold=1 must make every frozen budget feasible"
            )
        best_tp = int(tp[feasible].max())
        tied = feasible[tp[feasible] == best_tp]
        best_fp = int(fp[tied].min())
        tied = tied[fp[tied] == best_fp]
        # Thresholds and indices are strictly ascending, so the final row is
        # the unique deterministic max-threshold tie break.
        selected.append(int(tied[-1]))

    if any(right < left for left, right in zip(selected, selected[1:])):
        raise Stage2ExactOracleError(
            "nested exact-budget oracle selection unexpectedly decreased"
        )
    index = np.asarray(selected, dtype=np.intp)
    selected_thresholds = np.array(threshold[index], dtype=np.float64, copy=True)
    coordinates = np.array(
        encode_probability_numpy(selected_thresholds),
        dtype=np.float64,
        copy=True,
    )
    selected_fp = np.array(fp[index], dtype=np.int64, copy=True)
    selected_tp = np.array(tp[index], dtype=np.int64, copy=True)
    detection = (
        selected_tp.astype(np.float64) / float(objects)
        if objects > 0
        else np.zeros(3, dtype=np.float64)
    )
    risk = selected_fp.astype(np.float64) / float(total)
    for array in (
        selected_thresholds,
        coordinates,
        selected_fp,
        selected_tp,
        detection,
        risk,
    ):
        array.setflags(write=False)
    return ExactOracleSelectionV2(
        selected_indices=(selected[0], selected[1], selected[2]),
        allowed_false_positive_counts=(allowed[0], allowed[1], allowed[2]),
        thresholds=selected_thresholds,
        coordinates=coordinates,
        matched_objects=selected_tp,
        false_positive_pixels=selected_fp,
        detection_probability=detection,
        pixel_false_alarm_rate=risk,
        total_native_pixels=total,
        ground_truth_objects=objects,
    )


__all__ = [
    "BUDGET_COUNT_RULE",
    "ORACLE_SELECTION_RULE",
    "ORACLE_SELECTION_SCHEMA",
    "ExactOracleSelectionV2",
    "Stage2ExactOracleError",
    "select_exact_oracle_v2",
]

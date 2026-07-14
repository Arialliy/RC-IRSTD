from __future__ import annotations

"""Budget-aligned detector checkpoint selection.

The detector is selected from labelled *source validation* data only.  For each
source domain, the evaluator finds the earliest threshold that simultaneously
satisfies the pixel and fixed-local-peak budgets, then reports object detection
probability at that working point.  A checkpoint is preferred lexicographically
by worst-domain Pd, mean-domain Pd, and IoU.
"""

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from rc_irstd.evaluation.curves import (
    ImageCurveCounts,
    aggregate_curve_counts,
    rates_from_counts,
)


@dataclass(frozen=True)
class DomainBudgetPoint:
    domain: str
    index: int
    threshold: float
    pd: float
    pixel_risk: float
    peak_risk: float
    rejected: bool


@dataclass(frozen=True)
class DetectorBudgetSelection:
    pixel_budget: float
    peak_budget: float
    mean_domain_pd: float
    worst_domain_pd: float
    mean_threshold: float
    rejection_rate: float
    domain_points: tuple[DomainBudgetPoint, ...]
    peak_constraint_active: bool = True

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["domain_points"] = [asdict(item) for item in self.domain_points]
        return payload

    def rank_key(self, iou: float) -> tuple[float, float, float, float]:
        # Lower rejection is a final tie-breaker. Empty-action predictions are
        # not allowed to make an otherwise weak detector appear strong.
        return (
            float(self.worst_domain_pd),
            float(self.mean_domain_pd),
            float(iou),
            -float(self.rejection_rate),
        )


def validation_threshold_grid(num_points: int = 96) -> np.ndarray:
    if num_points < 16:
        raise ValueError("num_points must be at least 16")
    empty = np.nextafter(np.float32(1.0), np.float32(2.0))
    # More support is allocated near one, where IRSTD low-FA crossings occur.
    coarse = max(8, num_points // 4)
    medium = max(8, num_points // 4)
    fine = max(8, num_points - coarse - medium)
    return np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.90, coarse, endpoint=False),
                np.linspace(0.90, 0.99, medium, endpoint=False),
                np.linspace(0.99, 1.0, fine),
                np.asarray([empty], dtype=np.float32),
            ]
        )
    ).astype(np.float32)


def _select_first_feasible(
    thresholds: np.ndarray,
    pixel: np.ndarray,
    peak: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
    *,
    use_peak_constraint: bool = True,
) -> int:
    feasible_mask = pixel <= pixel_budget
    if use_peak_constraint:
        feasible_mask &= peak <= peak_budget
    feasible = np.flatnonzero(feasible_mask)
    return int(feasible[0]) if len(feasible) else len(thresholds) - 1


def summarise_detector_budget(
    domain_curves: dict[str, Iterable[ImageCurveCounts]],
    thresholds: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
    *,
    use_peak_constraint: bool = True,
) -> DetectorBudgetSelection:
    if pixel_budget <= 0 or peak_budget <= 0:
        raise ValueError("budgets must be positive")
    points: list[DomainBudgetPoint] = []
    thresholds = np.asarray(thresholds, dtype=np.float32)
    for domain, values in sorted(domain_curves.items()):
        records = list(values)
        if not records:
            continue
        rates = rates_from_counts(aggregate_curve_counts(records))
        index = _select_first_feasible(
            thresholds,
            rates["pixel_false_rate"],
            rates["peak_false_per_mp"],
            pixel_budget,
            peak_budget,
            use_peak_constraint=use_peak_constraint,
        )
        points.append(
            DomainBudgetPoint(
                domain=str(domain),
                index=index,
                threshold=float(thresholds[index]),
                pd=float(rates["pd"][index]),
                pixel_risk=float(rates["pixel_false_rate"][index]),
                peak_risk=float(rates["peak_false_per_mp"][index]),
                rejected=bool(thresholds[index] > 1.0),
            )
        )
    if not points:
        raise ValueError("No validation curves were supplied")
    pds = np.asarray([item.pd for item in points], dtype=np.float64)
    thresholds_selected = np.asarray([item.threshold for item in points], dtype=np.float64)
    rejected = np.asarray([item.rejected for item in points], dtype=np.float64)
    return DetectorBudgetSelection(
        pixel_budget=float(pixel_budget),
        peak_budget=float(peak_budget),
        mean_domain_pd=float(pds.mean()),
        worst_domain_pd=float(pds.min()),
        mean_threshold=float(thresholds_selected.mean()),
        rejection_rate=float(rejected.mean()),
        domain_points=tuple(points),
        peak_constraint_active=bool(use_peak_constraint),
    )

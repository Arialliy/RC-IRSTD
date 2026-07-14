from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BudgetSummary:
    joint_bsr: float
    pixel_bsr: float
    peak_bsr: float
    pixel_excess: float
    peak_excess: float
    mean_pd_selected: float
    effective_pd_with_rejects: float
    conditional_pd_non_rejected: float
    worst_domain_pd_selected: float
    rejection_rate: float
    count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarise_selected_points(
    pixel_risk: np.ndarray,
    peak_risk: np.ndarray,
    pd: np.ndarray,
    rejected: np.ndarray,
    domains: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> BudgetSummary:
    pixel_risk = np.asarray(pixel_risk, dtype=np.float64)
    peak_risk = np.asarray(peak_risk, dtype=np.float64)
    pd = np.asarray(pd, dtype=np.float64)
    rejected = np.asarray(rejected, dtype=bool)
    domains = np.asarray(domains).astype(str)
    if not (len(pixel_risk) == len(peak_risk) == len(pd) == len(rejected) == len(domains)):
        raise ValueError("All selected-point arrays must have equal length")
    if len(pd) == 0:
        raise ValueError("Cannot summarise an empty selection")
    pixel_ok = pixel_risk <= pixel_budget
    peak_ok = peak_risk <= peak_budget
    joint = pixel_ok & peak_ok
    valid_pd = np.isfinite(pd)
    domain_means: list[float] = []
    for domain in np.unique(domains):
        values = pd[(domains == domain) & valid_pd]
        if len(values):
            domain_means.append(float(values.mean()))
    non_rejected_valid = (~rejected) & valid_pd
    conditional = float(pd[non_rejected_valid].mean()) if non_rejected_valid.any() else 0.0
    effective_values = pd[valid_pd].copy()
    effective_values[rejected[valid_pd]] = 0.0
    return BudgetSummary(
        joint_bsr=float(joint.mean()),
        pixel_bsr=float(pixel_ok.mean()),
        peak_bsr=float(peak_ok.mean()),
        pixel_excess=float(np.maximum(pixel_risk - pixel_budget, 0.0).mean()),
        peak_excess=float(np.maximum(peak_risk - peak_budget, 0.0).mean()),
        mean_pd_selected=float(pd[valid_pd].mean()) if valid_pd.any() else 0.0,
        effective_pd_with_rejects=(
            float(effective_values.mean()) if len(effective_values) else 0.0
        ),
        conditional_pd_non_rejected=conditional,
        worst_domain_pd_selected=float(min(domain_means)) if domain_means else float("nan"),
        rejection_rate=float(rejected.mean()),
        count=len(pd),
    )

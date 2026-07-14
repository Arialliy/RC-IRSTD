"""Oracle operating-point labels for RC meta episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import BudgetSpec


@dataclass(frozen=True)
class OracleResult:
    threshold: float
    pd: float
    pixel_risk: float
    component_risk: float
    reject: bool
    p_min: float
    selected_index: int
    feasible_count: int

    def __iter__(self):
        """Allow legacy ``threshold, reject = result`` unpacking."""

        yield self.threshold
        yield self.reject

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "pd": self.pd,
            "pixel_risk": self.pixel_risk,
            "component_risk": self.component_risk,
            "reject": self.reject,
            "p_min": self.p_min,
            "selected_index": self.selected_index,
            "feasible_count": self.feasible_count,
        }


def _get_column(curve: Any, names: Sequence[str]) -> np.ndarray:
    if isinstance(curve, Mapping):
        for name in names:
            if name in curve:
                return np.asarray(curve[name], dtype=np.float64).reshape(-1)
    if hasattr(curve, "columns"):
        for name in names:
            if name in curve.columns:
                return np.asarray(curve[name], dtype=np.float64).reshape(-1)
    if isinstance(curve, Sequence) and curve and isinstance(curve[0], Mapping):
        for name in names:
            if name in curve[0]:
                return np.asarray([row[name] for row in curve], dtype=np.float64)
    raise KeyError(f"curve is missing every supported column name: {tuple(names)}")


def canonical_curve_arrays(curve: Any) -> dict[str, np.ndarray]:
    """Read a curve and force threshold=1 to be the empty, feasible sentinel."""

    threshold = _get_column(curve, ("threshold", "tau"))
    pd = _get_column(curve, ("pd", "detection_probability"))
    pixel = _get_column(curve, ("fa_pixel", "pixel_risk", "pixel_fa"))
    component = _get_column(
        curve, ("fa_component_mp", "component_risk", "component_fa")
    )
    sizes = {array.size for array in (threshold, pd, pixel, component)}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        raise ValueError("curve columns must be non-empty and have identical lengths")
    finite = np.isfinite(threshold) & np.isfinite(pd) & np.isfinite(pixel) & np.isfinite(component)
    if not finite.all():
        raise ValueError("curve values must be finite")
    if ((threshold < 0.0) | (threshold > 1.0)).any():
        raise ValueError("curve thresholds must lie in [0, 1]")
    if ((pd < 0.0) | (pd > 1.0)).any():
        raise ValueError("curve Pd values must lie in [0, 1]")
    if (pixel < 0.0).any() or (component < 0.0).any():
        raise ValueError("curve risks must be non-negative")

    # By contract threshold=1 means abstain from every prediction.  Replace an
    # evaluator's >=1 row, if any, so the oracle always has a feasible choice.
    keep = ~np.isclose(threshold, 1.0, rtol=0.0, atol=1e-12)
    threshold = np.concatenate([threshold[keep], [1.0]])
    pd = np.concatenate([pd[keep], [0.0]])
    pixel = np.concatenate([pixel[keep], [0.0]])
    component = np.concatenate([component[keep], [0.0]])
    order = np.argsort(threshold, kind="stable")
    return {
        "threshold": threshold[order],
        "pd": pd[order],
        "pixel_risk": pixel[order],
        "component_risk": component[order],
    }


def select_oracle_operating_point(
    curve: Any,
    budgets: BudgetSpec,
    *,
    p_min: float = 0.0,
    pd_tolerance: float = 1e-12,
) -> OracleResult:
    """Maximise Pd under active budgets, breaking Pd ties by lower threshold."""

    if not 0.0 <= p_min <= 1.0:
        raise ValueError("p_min must lie in [0, 1]")
    arrays = canonical_curve_arrays(curve)
    feasible = np.ones(arrays["threshold"].shape, dtype=bool)
    if budgets.active[0]:
        feasible &= arrays["pixel_risk"] <= budgets.values[0]
    if budgets.active[1]:
        feasible &= arrays["component_risk"] <= budgets.values[1]
    feasible_indices = np.flatnonzero(feasible)
    if feasible_indices.size == 0:
        raise RuntimeError("threshold=1 sentinel must always make the curve feasible")
    feasible_pd = arrays["pd"][feasible_indices]
    best_pd = float(feasible_pd.max())
    tied = feasible_indices[np.isclose(feasible_pd, best_pd, rtol=0.0, atol=pd_tolerance)]
    selected = int(tied[np.argmin(arrays["threshold"][tied])])
    return OracleResult(
        threshold=float(arrays["threshold"][selected]),
        pd=float(arrays["pd"][selected]),
        pixel_risk=float(arrays["pixel_risk"][selected]),
        component_risk=float(arrays["component_risk"][selected]),
        reject=float(arrays["pd"][selected]) < float(p_min),
        p_min=float(p_min),
        selected_index=selected,
        feasible_count=int(feasible_indices.size),
    )


def oracle_safe_threshold(
    curve_df: Any,
    pixel_budget: float | None = None,
    component_budget: float | None = None,
    *,
    budget_active: Sequence[bool] | None = None,
    p_min: float = 0.0,
) -> OracleResult:
    """Convenience wrapper accepting optional scalar budgets."""

    if budget_active is None:
        budgets = BudgetSpec.from_optional(pixel_budget, component_budget)
    else:
        active = tuple(bool(value) for value in budget_active)
        if len(active) != 2 or not any(active):
            raise ValueError("budget_active must contain two values with at least one active")
        values = (
            0.0 if pixel_budget is None else float(pixel_budget),
            0.0 if component_budget is None else float(component_budget),
        )
        budgets = BudgetSpec(values=values, active=active)  # type: ignore[arg-type]
    return select_oracle_operating_point(curve_df, budgets, p_min=p_min)

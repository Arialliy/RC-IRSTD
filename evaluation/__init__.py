"""Native-resolution risk-evaluation primitives for RC-IRSTD."""

from .budget_metrics import compute_budget_metrics, relative_budget_excess
from .component_matching import (
    MatchResult,
    aggregate_match_results,
    match_components,
)
from .evaluate_adapter_output import (
    ADAPTER_EVALUATION_SCHEMA_VERSION,
    ADAPTER_SUMMARY_SCHEMA_VERSION,
    evaluate_adapter_output,
    summarise_adapter_evaluations,
)
from .operating_point import (
    satisfies_budgets,
    select_budget_grid,
    select_operating_point,
)
from .threshold_sweep import (
    CURVE_SCHEMA_VERSION,
    ScoreMapRecord,
    THRESHOLD_GRID_VERSION,
    THRESHOLD_SEMANTICS,
    default_threshold_grid,
    normalise_thresholds,
    sweep_thresholds,
    threshold_grid_metadata,
)

__all__ = [
    "MatchResult",
    "ADAPTER_EVALUATION_SCHEMA_VERSION",
    "ADAPTER_SUMMARY_SCHEMA_VERSION",
    "CURVE_SCHEMA_VERSION",
    "ScoreMapRecord",
    "THRESHOLD_GRID_VERSION",
    "THRESHOLD_SEMANTICS",
    "aggregate_match_results",
    "compute_budget_metrics",
    "default_threshold_grid",
    "evaluate_adapter_output",
    "match_components",
    "normalise_thresholds",
    "relative_budget_excess",
    "satisfies_budgets",
    "select_budget_grid",
    "select_operating_point",
    "sweep_thresholds",
    "summarise_adapter_evaluations",
    "threshold_grid_metadata",
]

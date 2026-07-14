"""Native-resolution risk-evaluation primitives for RC-IRSTD.

Exports are resolved lazily so ``python -m evaluation.<tool>`` does not import
the target module once through this package and then execute it a second time.
This also keeps lightweight component utilities usable without importing the
calibrator-replay stack.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    "MatchResult": "component_matching",
    "aggregate_match_results": "component_matching",
    "match_components": "component_matching",
    "compute_budget_metrics": "budget_metrics",
    "relative_budget_excess": "budget_metrics",
    "ADAPTER_EVALUATION_SCHEMA_VERSION": "evaluate_adapter_output",
    "ADAPTER_SUMMARY_SCHEMA_VERSION": "evaluate_adapter_output",
    "evaluate_adapter_output": "evaluate_adapter_output",
    "summarise_adapter_evaluations": "evaluate_adapter_output",
    "satisfies_budgets": "operating_point",
    "select_budget_grid": "operating_point",
    "select_operating_point": "operating_point",
    "CURVE_SCHEMA_VERSION": "threshold_sweep",
    "ScoreMapRecord": "threshold_sweep",
    "THRESHOLD_GRID_VERSION": "threshold_sweep",
    "THRESHOLD_SEMANTICS": "threshold_sweep",
    "default_threshold_grid": "threshold_sweep",
    "normalise_thresholds": "threshold_sweep",
    "sweep_thresholds": "threshold_sweep",
    "threshold_grid_metadata": "threshold_sweep",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value

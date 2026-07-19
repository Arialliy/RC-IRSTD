"""Result-free RC5 Stage-2 variable-query geometry core.

This module is deliberately independent of datasets, files, labels, scores,
checkpoints, and metrics.  It operates only on an ordered record count and
materializes zero-based, half-open index spans.

The frozen RC5 rule keeps exactly 14 context records in every window, requires
at least 28 query records, preserves ``floor(N / 42)`` windows, balances all
remaining records across those query partitions, and consumes every ordered
record exactly once.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "rc-irstd.stage2-variable-query-geometry.v1"
CONTEXT_SIZE = 14
MINIMUM_QUERY_SIZE = 28
MINIMUM_WINDOW_SIZE = CONTEXT_SIZE + MINIMUM_QUERY_SIZE
WINDOW_COUNT_RULE = "floor(N/(context_size+minimum_query_size))"
QUERY_SIZE_POLICY = "balanced_all_remaining_over_frozen_window_count_v1"
CONSTRUCTION = (
    "ordered_non_overlapping_contiguous_variable_query_blocks_"
    "context_first_query_second_all_records"
)
INDEX_SEMANTICS = "zero_based_half_open"

_GEOMETRY_FIELDS = frozenset(
    {
        "schema_version",
        "context_size",
        "minimum_query_size",
        "minimum_window_size",
        "ordered_record_count",
        "window_count",
        "window_count_rule",
        "query_size_policy",
        "construction",
        "index_semantics",
        "all_indices_consumed_once",
        "windows",
    }
)
_WINDOW_FIELDS = frozenset(
    {
        "window_index",
        "context_start",
        "context_stop",
        "query_start",
        "query_stop",
        "context_size",
        "query_size",
    }
)
_WINDOW_INTEGER_FIELDS = tuple(sorted(_WINDOW_FIELDS))


class Stage2VariableQueryGeometryContractError(ValueError):
    """The frozen RC5 variable-query geometry contract was violated."""


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise Stage2VariableQueryGeometryContractError(
            f"{name} must be an exact integer"
        )
    if minimum is not None and value < minimum:
        raise Stage2VariableQueryGeometryContractError(
            f"{name} must be >= {minimum}"
        )
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise Stage2VariableQueryGeometryContractError(
            f"{name} field closure mismatch: missing={missing}, extra={extra}"
        )


def _validate_frozen_parameters(
    context_size: Any, minimum_query_size: Any
) -> tuple[int, int]:
    context = _strict_int(context_size, "context_size", minimum=1)
    query = _strict_int(minimum_query_size, "minimum_query_size", minimum=1)
    if context != CONTEXT_SIZE or query != MINIMUM_QUERY_SIZE:
        raise Stage2VariableQueryGeometryContractError(
            "RC5 geometry requires exactly C14/Qmin28"
        )
    return context, query


def derive_stage2_query_sizes(
    ordered_record_count: int,
    *,
    context_size: int = CONTEXT_SIZE,
    minimum_query_size: int = MINIMUM_QUERY_SIZE,
) -> tuple[int, ...]:
    """Return the frozen balanced query sizes for one ordered role sequence."""

    context, query_minimum = _validate_frozen_parameters(
        context_size, minimum_query_size
    )
    minimum_window = context + query_minimum
    record_count = _strict_int(
        ordered_record_count,
        "ordered_record_count",
        minimum=minimum_window,
    )
    window_count = record_count // minimum_window
    if window_count < 1:
        raise Stage2VariableQueryGeometryContractError(
            "geometry requires at least one complete C14/Qmin28 window"
        )

    total_query_count = record_count - context * window_count
    query_base, extra = divmod(total_query_count, window_count)
    if query_base < query_minimum:
        raise Stage2VariableQueryGeometryContractError(
            "internal geometry cannot satisfy the minimum query size"
        )
    return tuple(
        query_base + (1 if index < extra else 0)
        for index in range(window_count)
    )


def build_stage2_variable_query_geometry(
    ordered_record_count: int,
    *,
    context_size: int = CONTEXT_SIZE,
    minimum_query_size: int = MINIMUM_QUERY_SIZE,
) -> dict[str, Any]:
    """Build the canonical, result-free RC5 geometry payload."""

    context, query_minimum = _validate_frozen_parameters(
        context_size, minimum_query_size
    )
    query_sizes = derive_stage2_query_sizes(
        ordered_record_count,
        context_size=context,
        minimum_query_size=query_minimum,
    )
    record_count = _strict_int(
        ordered_record_count,
        "ordered_record_count",
        minimum=MINIMUM_WINDOW_SIZE,
    )

    windows: list[dict[str, int]] = []
    cursor = 0
    for window_index, query_size in enumerate(query_sizes):
        context_start = cursor
        context_stop = context_start + context
        query_start = context_stop
        query_stop = query_start + query_size
        windows.append(
            {
                "window_index": window_index,
                "context_start": context_start,
                "context_stop": context_stop,
                "query_start": query_start,
                "query_stop": query_stop,
                "context_size": context,
                "query_size": query_size,
            }
        )
        cursor = query_stop

    if cursor != record_count:
        raise Stage2VariableQueryGeometryContractError(
            "internal geometry did not consume every ordered index"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "context_size": context,
        "minimum_query_size": query_minimum,
        "minimum_window_size": MINIMUM_WINDOW_SIZE,
        "ordered_record_count": record_count,
        "window_count": len(windows),
        "window_count_rule": WINDOW_COUNT_RULE,
        "query_size_policy": QUERY_SIZE_POLICY,
        "construction": CONSTRUCTION,
        "index_semantics": INDEX_SEMANTICS,
        "all_indices_consumed_once": True,
        "windows": windows,
    }


def validate_stage2_variable_query_geometry(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return the canonical RC5 geometry payload.

    Validation replays the complete construction rather than trusting declared
    sizes or spans.  Booleans, integer subclasses, missing fields, extra fields,
    reordered/overlapping spans, and non-frozen parameters all fail closed.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("Stage2 variable-query geometry must be a mapping")
    _exact_keys(payload, _GEOMETRY_FIELDS, "geometry")

    exact_values = {
        "schema_version": SCHEMA_VERSION,
        "window_count_rule": WINDOW_COUNT_RULE,
        "query_size_policy": QUERY_SIZE_POLICY,
        "construction": CONSTRUCTION,
        "index_semantics": INDEX_SEMANTICS,
    }
    for field, expected in exact_values.items():
        if type(payload[field]) is not str or payload[field] != expected:
            raise Stage2VariableQueryGeometryContractError(
                f"geometry.{field} mismatch"
            )
    context, query_minimum = _validate_frozen_parameters(
        payload["context_size"], payload["minimum_query_size"]
    )
    minimum_window = _strict_int(
        payload["minimum_window_size"], "geometry.minimum_window_size", minimum=1
    )
    if minimum_window != MINIMUM_WINDOW_SIZE:
        raise Stage2VariableQueryGeometryContractError(
            "geometry.minimum_window_size mismatch"
        )
    record_count = _strict_int(
        payload["ordered_record_count"],
        "geometry.ordered_record_count",
        minimum=MINIMUM_WINDOW_SIZE,
    )
    window_count = _strict_int(
        payload["window_count"], "geometry.window_count", minimum=1
    )
    expected_window_count = record_count // minimum_window
    if window_count != expected_window_count:
        raise Stage2VariableQueryGeometryContractError(
            "geometry.window_count does not follow the frozen floor rule"
        )
    if type(payload["all_indices_consumed_once"]) is not bool or not payload[
        "all_indices_consumed_once"
    ]:
        raise Stage2VariableQueryGeometryContractError(
            "geometry.all_indices_consumed_once must be exactly true"
        )

    raw_windows = payload["windows"]
    if not isinstance(raw_windows, list) or len(raw_windows) != window_count:
        raise Stage2VariableQueryGeometryContractError(
            "geometry.windows must match the frozen window count"
        )
    expected_query_sizes = derive_stage2_query_sizes(
        record_count,
        context_size=context,
        minimum_query_size=query_minimum,
    )

    canonical_windows: list[dict[str, int]] = []
    cursor = 0
    for window_index, (raw_window, query_size) in enumerate(
        zip(raw_windows, expected_query_sizes, strict=True)
    ):
        if not isinstance(raw_window, Mapping):
            raise TypeError(f"geometry.windows[{window_index}] must be a mapping")
        _exact_keys(raw_window, _WINDOW_FIELDS, f"geometry.windows[{window_index}]")
        for field in _WINDOW_INTEGER_FIELDS:
            _strict_int(
                raw_window[field],
                f"geometry.windows[{window_index}].{field}",
                minimum=0,
            )

        expected_window = {
            "window_index": window_index,
            "context_start": cursor,
            "context_stop": cursor + context,
            "query_start": cursor + context,
            "query_stop": cursor + context + query_size,
            "context_size": context,
            "query_size": query_size,
        }
        if dict(raw_window) != expected_window:
            raise Stage2VariableQueryGeometryContractError(
                f"geometry.windows[{window_index}] differs from deterministic replay"
            )
        canonical_windows.append(expected_window)
        cursor = expected_window["query_stop"]

    if cursor != record_count:
        raise Stage2VariableQueryGeometryContractError(
            "geometry windows do not consume every ordered index exactly once"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "context_size": context,
        "minimum_query_size": query_minimum,
        "minimum_window_size": minimum_window,
        "ordered_record_count": record_count,
        "window_count": window_count,
        "window_count_rule": WINDOW_COUNT_RULE,
        "query_size_policy": QUERY_SIZE_POLICY,
        "construction": CONSTRUCTION,
        "index_semantics": INDEX_SEMANTICS,
        "all_indices_consumed_once": True,
        "windows": canonical_windows,
    }


__all__ = [
    "CONSTRUCTION",
    "CONTEXT_SIZE",
    "INDEX_SEMANTICS",
    "MINIMUM_QUERY_SIZE",
    "MINIMUM_WINDOW_SIZE",
    "QUERY_SIZE_POLICY",
    "SCHEMA_VERSION",
    "Stage2VariableQueryGeometryContractError",
    "WINDOW_COUNT_RULE",
    "build_stage2_variable_query_geometry",
    "derive_stage2_query_sizes",
    "validate_stage2_variable_query_geometry",
]

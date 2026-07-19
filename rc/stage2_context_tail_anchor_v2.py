"""Generalized exact-rational context-tail anchors for the RC5+ candidate.

The v1 anchor is intentionally frozen to the three RC5 budgets.  RC5+ learns
a denser budget-conditioned curve and therefore needs the analytic T4 anchor
at the *same* exact rational budgets used by the learned function.  This v2
artifact always binds the frozen nine-knot grid and may additionally bind one
ordered in-range request set.  Every row is recomputed directly from the same
fourteen unlabelled context score maps; requested anchors are never obtained
by interpolating grid anchors.

This module is additive and result-free.  It does not authorize execution and
does not change the v1 schema or its verifier-issued capability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from model.endpoint_aware_threshold import (
    endpoint_kinds_numpy,
    encode_probability_numpy,
    representation_contract,
)
from rc.stage2_context_tail_anchor import (
    CONTEXT_PROBABILITY_CONTENT_ALGORITHM,
    CONTEXT_SIZE,
    STRICT_THRESHOLD_SEMANTICS,
    canonical_json_bytes,
    canonical_json_sha256,
)


CONTEXT_TAIL_ANCHOR_V2_SCHEMA = "rc-irstd.stage2-context-tail-anchor.v2"
CONTEXT_TAIL_ANCHOR_V2_ARTIFACT_TYPE = "rc_irstd_stage2_context_tail_anchor_v2"
CONTEXT_TAIL_ANCHOR_V2_ALGORITHM = (
    "exact-rational-same-budget-strict-exceedance-order-statistic-"
    "numpy-partition-v2"
)
BUDGET_CURVE_COORDINATE_ALGORITHM = (
    "float64-normalized-log-exact-rational-projection-v1"
)
MAX_EXACT_RATIONAL_INTEGER = 2**63 - 1

_SHA256_HEX = frozenset("0123456789abcdef")
_VERIFIED_CAPABILITY = object()
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "context_identity_sha256",
        "context_size",
        "total_context_pixels",
        "context_probability_content_algorithm",
        "context_probability_content_sha256",
        "context_map_bindings",
        "budget_order",
        "grid_budget_rationals",
        "requested_budget_rationals",
        "grid_threshold_rows",
        "requested_threshold_rows",
        "requested_anchor_source",
        "budget_curve_coordinate_algorithm",
        "threshold_representation",
        "threshold_semantics",
        "selection_algorithm",
        "guardrails",
        "anchor_identity_sha256",
    }
)
_MAP_BINDING_FIELDS = frozenset(
    {"ordinal", "height", "width", "pixel_count", "content_sha256"}
)
_BUDGET_FIELDS = frozenset({"numerator", "denominator"})
_ROW_FIELDS = frozenset(
    {
        "budget_numerator",
        "budget_denominator",
        "allowed_strict_exceedances",
        "observed_strict_exceedances",
        "order_statistic_rank_zero_based",
        "threshold_probability_hex",
        "threshold_coordinate_hex",
        "threshold_kind",
    }
)
_GUARDRAIL_FIELDS = frozenset(
    {
        "context_labels_accessed",
        "query_scores_accessed",
        "query_labels_accessed",
        "postlabel_statistics_accessed",
        "anchor_interpolation_used",
    }
)


class Stage2ContextTailAnchorV2Error(ValueError):
    """An RC5+ generalized anchor violates its fail-closed contract."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _SHA256_HEX
    ):
        raise Stage2ContextTailAnchorV2Error(f"{name} must be lowercase SHA-256")
    return value


def _strict_int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise Stage2ContextTailAnchorV2Error(f"{name} must be int >= {minimum}")
    if maximum is not None and value > maximum:
        raise Stage2ContextTailAnchorV2Error(
            f"{name} must be int <= {maximum}"
        )
    return value


def _strict_fields(value: Any, fields: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Stage2ContextTailAnchorV2Error(
            f"{name} fields must equal {sorted(fields)}"
        )
    return value


def _canonical_float_hex(value: Any, name: str) -> float:
    if not isinstance(value, str):
        raise Stage2ContextTailAnchorV2Error(
            f"{name} must be canonical float.hex text"
        )
    try:
        result = float.fromhex(value)
    except ValueError as error:
        raise Stage2ContextTailAnchorV2Error(f"{name} is invalid") from error
    if not math.isfinite(result) or result.hex() != value:
        raise Stage2ContextTailAnchorV2Error(
            f"{name} is not canonical finite binary64"
        )
    return result


def _validate_budget_rationals(
    value: Sequence[tuple[int, int]],
    *,
    name: str,
    allow_empty: bool,
) -> tuple[tuple[int, int], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Stage2ContextTailAnchorV2Error(f"{name} must be an ordered sequence")
    if not value and not allow_empty:
        raise Stage2ContextTailAnchorV2Error(f"{name} must not be empty")
    result: list[tuple[int, int]] = []
    for index, raw in enumerate(value):
        if (
            isinstance(raw, (str, bytes))
            or not isinstance(raw, Sequence)
            or len(raw) != 2
        ):
            raise Stage2ContextTailAnchorV2Error(
                f"{name}[{index}] must be one numerator/denominator pair"
            )
        numerator = _strict_int(
            raw[0],
            f"{name}[{index}].numerator",
            minimum=1,
            maximum=MAX_EXACT_RATIONAL_INTEGER,
        )
        denominator = _strict_int(
            raw[1],
            f"{name}[{index}].denominator",
            minimum=2,
            maximum=MAX_EXACT_RATIONAL_INTEGER,
        )
        if numerator >= denominator:
            raise Stage2ContextTailAnchorV2Error(
                f"{name} must lie strictly inside (0,1)"
            )
        fraction = Fraction(numerator, denominator)
        if (fraction.numerator, fraction.denominator) != (
            numerator,
            denominator,
        ):
            raise Stage2ContextTailAnchorV2Error(
                f"{name} must contain lowest-term rational pairs"
            )
        result.append((numerator, denominator))
    fractions = tuple(Fraction(*row) for row in result)
    if not all(left > right for left, right in zip(fractions, fractions[1:])):
        raise Stage2ContextTailAnchorV2Error(
            f"{name} must descend strictly from loose to strict"
        )
    return tuple(result)


def _validate_requested_budget_rationals(
    value: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    budgets = _validate_budget_rationals(
        value,
        name="requested_budget_rationals",
        allow_empty=True,
    )
    if not budgets:
        return budgets
    loose = Fraction(*BUDGET_KNOT_RATIONALS[0])
    strict = Fraction(*BUDGET_KNOT_RATIONALS[-1])
    if any(not strict <= Fraction(*row) <= loose for row in budgets):
        raise Stage2ContextTailAnchorV2Error(
            "requested budgets must stay inside the frozen RC5+ knot range"
        )
    log_positions = tuple(
        math.log(numerator) - math.log(denominator)
        for numerator, denominator in budgets
    )
    if not all(left > right for left, right in zip(log_positions, log_positions[1:])):
        raise Stage2ContextTailAnchorV2Error(
            "requested rationals are not distinguishable in the float64 "
            "budget-curve coordinate"
        )
    return budgets


def _validated_maps(
    context_probability_maps: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    if (
        isinstance(context_probability_maps, (str, bytes))
        or not isinstance(context_probability_maps, Sequence)
        or len(context_probability_maps) != CONTEXT_SIZE
    ):
        raise Stage2ContextTailAnchorV2Error(
            f"exactly {CONTEXT_SIZE} context probability maps are required"
        )
    result: list[np.ndarray] = []
    for index, raw in enumerate(context_probability_maps):
        if not isinstance(raw, np.ndarray):
            raise Stage2ContextTailAnchorV2Error(
                f"context map {index} must be an explicit numpy array"
            )
        if raw.dtype != np.float64 or raw.ndim != 2 or raw.size == 0:
            raise Stage2ContextTailAnchorV2Error(
                f"context map {index} must be a nonempty 2D float64 array"
            )
        if not np.isfinite(raw).all() or np.any((raw < 0.0) | (raw > 1.0)):
            raise Stage2ContextTailAnchorV2Error(f"context map {index} is invalid")
        canonical = np.array(raw, dtype="<f8", order="C", copy=True)
        canonical.setflags(write=False)
        result.append(canonical)
    return tuple(result)


def _map_bindings(arrays: Sequence[np.ndarray]) -> tuple[list[dict[str, Any]], str]:
    result: list[dict[str, Any]] = []
    for ordinal, array in enumerate(arrays):
        result.append(
            {
                "ordinal": ordinal,
                "height": int(array.shape[0]),
                "width": int(array.shape[1]),
                "pixel_count": int(array.size),
                "content_sha256": hashlib.sha256(
                    array.tobytes(order="C")
                ).hexdigest(),
            }
        )
    return result, canonical_json_sha256(result)


def _threshold_rows(
    values: np.ndarray,
    budgets: tuple[tuple[int, int], ...],
) -> list[dict[str, Any]]:
    if not budgets:
        return []
    total = int(values.size)
    allowed = tuple(
        (numerator * total) // denominator for numerator, denominator in budgets
    )
    ranks = tuple(total - count - 1 for count in allowed)
    partitioned = np.partition(values, np.asarray(sorted(set(ranks)), dtype=np.intp))
    thresholds = tuple(float(partitioned[rank]) for rank in ranks)
    observed = tuple(
        int(np.count_nonzero(values > threshold)) for threshold in thresholds
    )
    if any(actual > maximum for actual, maximum in zip(observed, allowed, strict=True)):
        raise RuntimeError("exact context anchor exceeded its strict budget")
    if any(right < left for left, right in zip(thresholds, thresholds[1:])):
        raise RuntimeError("tighter budgets produced a decreasing anchor")
    coordinates = encode_probability_numpy(np.asarray(thresholds, dtype=np.float64))
    kinds = endpoint_kinds_numpy(coordinates)
    return [
        {
            "budget_numerator": budget[0],
            "budget_denominator": budget[1],
            "allowed_strict_exceedances": maximum,
            "observed_strict_exceedances": actual,
            "order_statistic_rank_zero_based": rank,
            "threshold_probability_hex": threshold.hex(),
            "threshold_coordinate_hex": float(coordinate).hex(),
            "threshold_kind": kind,
        }
        for budget, maximum, actual, rank, threshold, coordinate, kind in zip(
            budgets,
            allowed,
            observed,
            ranks,
            thresholds,
            coordinates.tolist(),
            kinds,
            strict=True,
        )
    ]


def _budget_payload(budgets: Sequence[tuple[int, int]]) -> list[dict[str, int]]:
    return [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in budgets
    ]


def build_context_tail_anchor_v2(
    *,
    context_probability_maps: Sequence[np.ndarray],
    context_identity_sha256: str,
    requested_budget_rationals: Sequence[tuple[int, int]] = (),
    grid_budget_rationals: Sequence[tuple[int, int]] = BUDGET_KNOT_RATIONALS,
) -> dict[str, Any]:
    """Build a nine-knot plus same-request-budget RC5+ anchor artifact."""

    identity = _sha256(context_identity_sha256, "context_identity_sha256")
    grid = _validate_budget_rationals(
        grid_budget_rationals,
        name="grid_budget_rationals",
        allow_empty=False,
    )
    if grid != BUDGET_KNOT_RATIONALS:
        raise Stage2ContextTailAnchorV2Error(
            "grid budgets must equal the frozen RC5+ knot lattice"
        )
    requested = _validate_requested_budget_rationals(requested_budget_rationals)
    arrays = _validated_maps(context_probability_maps)
    values = np.concatenate([array.reshape(-1) for array in arrays])
    grid_rows = _threshold_rows(values, grid)
    requested_rows = _threshold_rows(values, requested)
    bindings, content_sha256 = _map_bindings(arrays)
    payload: dict[str, Any] = {
        "schema_version": CONTEXT_TAIL_ANCHOR_V2_SCHEMA,
        "artifact_type": CONTEXT_TAIL_ANCHOR_V2_ARTIFACT_TYPE,
        "artifact_status": "complete",
        "context_identity_sha256": identity,
        "context_size": CONTEXT_SIZE,
        "total_context_pixels": int(values.size),
        "context_probability_content_algorithm": CONTEXT_PROBABILITY_CONTENT_ALGORITHM,
        "context_probability_content_sha256": content_sha256,
        "context_map_bindings": bindings,
        "budget_order": "strictly_descending_loose_to_strict",
        "grid_budget_rationals": _budget_payload(grid),
        "requested_budget_rationals": _budget_payload(requested),
        "grid_threshold_rows": grid_rows,
        "requested_threshold_rows": requested_rows,
        "requested_anchor_source": (
            "direct_same_budget_context_order_statistic_not_grid_interpolation"
        ),
        "budget_curve_coordinate_algorithm": BUDGET_CURVE_COORDINATE_ALGORITHM,
        "threshold_representation": representation_contract(),
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "selection_algorithm": CONTEXT_TAIL_ANCHOR_V2_ALGORITHM,
        "guardrails": {
            "context_labels_accessed": False,
            "query_scores_accessed": False,
            "query_labels_accessed": False,
            "postlabel_statistics_accessed": False,
            "anchor_interpolation_used": False,
        },
    }
    payload["anchor_identity_sha256"] = canonical_json_sha256(payload)
    return payload


@dataclass(frozen=True, init=False)
class VerifiedContextTailAnchorV2:
    payload: Mapping[str, Any]
    grid_budget_rationals: tuple[tuple[int, int], ...]
    requested_budget_rationals: tuple[tuple[int, int], ...]
    grid_thresholds: tuple[float, ...]
    grid_coordinates: tuple[float, ...]
    requested_thresholds: tuple[float, ...]
    requested_coordinates: tuple[float, ...]
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("VerifiedContextTailAnchorV2 is verifier-issued only")


def _verified_context_tail_anchor_v2(
    *,
    payload: Mapping[str, Any],
    grid: tuple[tuple[int, int], ...],
    requested: tuple[tuple[int, int], ...],
) -> VerifiedContextTailAnchorV2:
    value = object.__new__(VerifiedContextTailAnchorV2)
    object.__setattr__(value, "payload", _freeze(payload))
    object.__setattr__(value, "grid_budget_rationals", grid)
    object.__setattr__(value, "requested_budget_rationals", requested)
    for prefix, rows in (
        ("grid", payload["grid_threshold_rows"]),
        ("requested", payload["requested_threshold_rows"]),
    ):
        object.__setattr__(
            value,
            f"{prefix}_thresholds",
            tuple(float.fromhex(row["threshold_probability_hex"]) for row in rows),
        )
        object.__setattr__(
            value,
            f"{prefix}_coordinates",
            tuple(float.fromhex(row["threshold_coordinate_hex"]) for row in rows),
        )
    object.__setattr__(value, "_capability", _VERIFIED_CAPABILITY)
    return value


def _parse_budget_payload(value: Any, name: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise Stage2ContextTailAnchorV2Error(f"{name} must be a list")
    rows: list[tuple[int, int]] = []
    for index, raw in enumerate(value):
        row = _strict_fields(raw, _BUDGET_FIELDS, f"{name}[{index}]")
        rows.append((row["numerator"], row["denominator"]))
    return tuple(rows)


def _validate_threshold_row_payload(value: Any, *, name: str, count: int) -> None:
    if not isinstance(value, list) or len(value) != count:
        raise Stage2ContextTailAnchorV2Error(f"{name} cardinality mismatch")
    for index, raw in enumerate(value):
        row = _strict_fields(raw, _ROW_FIELDS, f"{name}[{index}]")
        for field in (
            "budget_numerator",
            "budget_denominator",
            "allowed_strict_exceedances",
            "observed_strict_exceedances",
            "order_statistic_rank_zero_based",
        ):
            _strict_int(row[field], f"{name}[{index}].{field}")
        _canonical_float_hex(
            row["threshold_probability_hex"],
            f"{name}[{index}].threshold_probability_hex",
        )
        _canonical_float_hex(
            row["threshold_coordinate_hex"],
            f"{name}[{index}].threshold_coordinate_hex",
        )
        if not isinstance(row["threshold_kind"], str):
            raise Stage2ContextTailAnchorV2Error(
                f"{name}[{index}].threshold_kind must be text"
            )


def verify_context_tail_anchor_v2(
    payload: Mapping[str, Any],
    *,
    context_probability_maps: Sequence[np.ndarray],
    expected_context_identity_sha256: str,
    expected_requested_budget_rationals: Sequence[tuple[int, int]] = (),
) -> VerifiedContextTailAnchorV2:
    """Replay all maps, ranks, exact budgets, coordinates and artifact identity."""

    top = _strict_fields(payload, _TOP_LEVEL_FIELDS, "anchor_v2")
    if top["schema_version"] != CONTEXT_TAIL_ANCHOR_V2_SCHEMA:
        raise Stage2ContextTailAnchorV2Error("anchor_v2 schema_version is invalid")
    if top["artifact_type"] != CONTEXT_TAIL_ANCHOR_V2_ARTIFACT_TYPE:
        raise Stage2ContextTailAnchorV2Error("anchor_v2 artifact_type is invalid")
    if top["artifact_status"] != "complete":
        raise Stage2ContextTailAnchorV2Error("anchor_v2 is not complete")
    identity = _sha256(
        expected_context_identity_sha256,
        "expected_context_identity_sha256",
    )
    if top["context_identity_sha256"] != identity:
        raise Stage2ContextTailAnchorV2Error("anchor_v2 context identity mismatch")
    if _strict_int(top["context_size"], "context_size", minimum=1) != CONTEXT_SIZE:
        raise Stage2ContextTailAnchorV2Error("anchor_v2 context size mismatch")
    _strict_int(top["total_context_pixels"], "total_context_pixels", minimum=1)
    if top["context_probability_content_algorithm"] != CONTEXT_PROBABILITY_CONTENT_ALGORITHM:
        raise Stage2ContextTailAnchorV2Error("context probability hash algorithm drifted")
    _sha256(
        top["context_probability_content_sha256"],
        "context_probability_content_sha256",
    )
    _sha256(top["anchor_identity_sha256"], "anchor_identity_sha256")
    if (
        not isinstance(top["context_map_bindings"], list)
        or len(top["context_map_bindings"]) != CONTEXT_SIZE
    ):
        raise Stage2ContextTailAnchorV2Error("context map binding cardinality mismatch")
    for index, raw in enumerate(top["context_map_bindings"]):
        row = _strict_fields(raw, _MAP_BINDING_FIELDS, f"context_map_bindings[{index}]")
        if _strict_int(row["ordinal"], f"context_map_bindings[{index}].ordinal") != index:
            raise Stage2ContextTailAnchorV2Error("context map ordinals are invalid")
        for field in ("height", "width", "pixel_count"):
            _strict_int(row[field], f"context_map_bindings[{index}].{field}", minimum=1)
        _sha256(row["content_sha256"], f"context_map_bindings[{index}].content_sha256")
    if top["budget_order"] != "strictly_descending_loose_to_strict":
        raise Stage2ContextTailAnchorV2Error("budget_order is invalid")
    grid = _validate_budget_rationals(
        _parse_budget_payload(top["grid_budget_rationals"], "grid_budget_rationals"),
        name="grid_budget_rationals",
        allow_empty=False,
    )
    if grid != BUDGET_KNOT_RATIONALS:
        raise Stage2ContextTailAnchorV2Error("anchor_v2 grid differs from RC5+")
    requested = _validate_requested_budget_rationals(
        _parse_budget_payload(
            top["requested_budget_rationals"],
            "requested_budget_rationals",
        )
    )
    expected_requested = _validate_requested_budget_rationals(
        expected_requested_budget_rationals
    )
    if requested != expected_requested:
        raise Stage2ContextTailAnchorV2Error("requested budget identity mismatch")
    _validate_threshold_row_payload(
        top["grid_threshold_rows"],
        name="grid_threshold_rows",
        count=len(grid),
    )
    _validate_threshold_row_payload(
        top["requested_threshold_rows"],
        name="requested_threshold_rows",
        count=len(requested),
    )
    if top["requested_anchor_source"] != (
        "direct_same_budget_context_order_statistic_not_grid_interpolation"
    ):
        raise Stage2ContextTailAnchorV2Error("requested anchor source drifted")
    if top["budget_curve_coordinate_algorithm"] != BUDGET_CURVE_COORDINATE_ALGORITHM:
        raise Stage2ContextTailAnchorV2Error("budget curve coordinate algorithm drifted")
    if top["threshold_representation"] != representation_contract():
        raise Stage2ContextTailAnchorV2Error("threshold representation drifted")
    if top["threshold_semantics"] != STRICT_THRESHOLD_SEMANTICS:
        raise Stage2ContextTailAnchorV2Error("threshold semantics drifted")
    if top["selection_algorithm"] != CONTEXT_TAIL_ANCHOR_V2_ALGORITHM:
        raise Stage2ContextTailAnchorV2Error("selection algorithm drifted")
    guardrails = _strict_fields(top["guardrails"], _GUARDRAIL_FIELDS, "guardrails")
    if any(type(value) is not bool or value for value in guardrails.values()):
        raise Stage2ContextTailAnchorV2Error("anchor_v2 records forbidden access")

    expected = build_context_tail_anchor_v2(
        context_probability_maps=context_probability_maps,
        context_identity_sha256=identity,
        requested_budget_rationals=requested,
        grid_budget_rationals=grid,
    )
    if canonical_json_bytes(top) != canonical_json_bytes(expected):
        raise Stage2ContextTailAnchorV2Error(
            "anchor_v2 replay differs from the supplied artifact"
        )
    return _verified_context_tail_anchor_v2(
        payload=expected,
        grid=grid,
        requested=requested,
    )


def assert_verified_context_tail_anchor_v2(
    value: VerifiedContextTailAnchorV2,
) -> VerifiedContextTailAnchorV2:
    if (
        not isinstance(value, VerifiedContextTailAnchorV2)
        or getattr(value, "_capability", None) is not _VERIFIED_CAPABILITY
    ):
        raise TypeError("a verifier-issued VerifiedContextTailAnchorV2 is required")
    return value


__all__ = [
    "BUDGET_CURVE_COORDINATE_ALGORITHM",
    "CONTEXT_TAIL_ANCHOR_V2_ALGORITHM",
    "CONTEXT_TAIL_ANCHOR_V2_ARTIFACT_TYPE",
    "CONTEXT_TAIL_ANCHOR_V2_SCHEMA",
    "MAX_EXACT_RATIONAL_INTEGER",
    "Stage2ContextTailAnchorV2Error",
    "VerifiedContextTailAnchorV2",
    "assert_verified_context_tail_anchor_v2",
    "build_context_tail_anchor_v2",
    "verify_context_tail_anchor_v2",
]

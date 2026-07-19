"""Exact, replayable T4 tail anchors for the RC5 Stage-2 calibrators.

The anchor consumes exactly fourteen *unlabelled context* probability maps.
For each frozen false-alarm budget ``a / b``, it sets
``k = floor(a*N/b)`` and selects ascending zero-based rank ``N-k-1``.  Under
strict-``>`` semantics this permits at most ``k`` exceedances; ties may yield
fewer.  Integer arithmetic is used throughout the budget calculation, so
decimal binary64 approximations such as ``1e-5 * N`` never decide a rank.

The emitted artifact carries only hexadecimal binary64 values and hashes of
the context score maps.  Verification recomputes the complete artifact from
the maps, making the context -> T4 anchor edge replayable before any query
labels are made available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from types import MappingProxyType
from typing import Any

import numpy as np

from model.endpoint_aware_threshold import (
    endpoint_kinds_numpy,
    encode_probability_numpy,
    representation_contract,
)


CONTEXT_TAIL_ANCHOR_SCHEMA = "rc-irstd.stage2-context-tail-anchor.v1"
CONTEXT_TAIL_ANCHOR_ARTIFACT_TYPE = "rc_irstd_stage2_context_tail_anchor"
CONTEXT_TAIL_ANCHOR_ALGORITHM = (
    "exact-rational-strict-exceedance-order-statistic-numpy-partition-v1"
)
CONTEXT_PROBABILITY_CONTENT_ALGORITHM = (
    "sha256-canonical-map-bindings-little-endian-float64-c-order-v1"
)
STRICT_THRESHOLD_SEMANTICS = "prediction = probability > threshold"
CONTEXT_SIZE = 14
BUDGET_RATIONALS: tuple[tuple[int, int], ...] = (
    (1, 10_000),
    (1, 100_000),
    (1, 1_000_000),
)

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
        "budget_rationals",
        "threshold_rows",
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
    }
)


class Stage2ContextTailAnchorError(ValueError):
    """A context-tail anchor violates the frozen RC5 contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _SHA256_HEX
    ):
        raise Stage2ContextTailAnchorError(f"{name} must be lowercase SHA-256")
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Stage2ContextTailAnchorError(f"{name} must be int >= {minimum}")
    return value


def _strict_fields(value: Any, fields: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Stage2ContextTailAnchorError(
            f"{name} fields must equal {sorted(fields)}"
        )
    return value


def _canonical_float_hex(value: Any, name: str) -> float:
    if not isinstance(value, str):
        raise Stage2ContextTailAnchorError(f"{name} must be canonical float.hex text")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise Stage2ContextTailAnchorError(f"{name} is invalid") from error
    if not np.isfinite(parsed) or parsed.hex() != value:
        raise Stage2ContextTailAnchorError(f"{name} is not canonical finite binary64")
    return parsed


def _validate_budget_rationals(
    value: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise Stage2ContextTailAnchorError("exactly three budget rationals are required")
    result: list[tuple[int, int]] = []
    for index, raw in enumerate(value):
        if (
            isinstance(raw, (str, bytes))
            or not isinstance(raw, Sequence)
            or len(raw) != 2
        ):
            raise Stage2ContextTailAnchorError(
                f"budget_rationals[{index}] must be a numerator/denominator pair"
            )
        numerator = _strict_int(raw[0], f"budget_rationals[{index}].numerator", minimum=1)
        denominator = _strict_int(
            raw[1], f"budget_rationals[{index}].denominator", minimum=2
        )
        if numerator >= denominator:
            raise Stage2ContextTailAnchorError("budget rationals must lie strictly in (0,1)")
        reduced = Fraction(numerator, denominator)
        if (reduced.numerator, reduced.denominator) != (numerator, denominator):
            raise Stage2ContextTailAnchorError(
                "budget rationals must be canonical lowest-term pairs"
            )
        result.append((numerator, denominator))
    fractions = tuple(Fraction(num, den) for num, den in result)
    if not all(left > right for left, right in zip(fractions, fractions[1:])):
        raise Stage2ContextTailAnchorError(
            "budget rationals must be strictly descending from loose to strict"
        )
    return tuple(result)


def _validated_maps(
    context_probability_maps: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    if (
        isinstance(context_probability_maps, (str, bytes))
        or not isinstance(context_probability_maps, Sequence)
        or len(context_probability_maps) != CONTEXT_SIZE
    ):
        raise Stage2ContextTailAnchorError(
            f"exactly {CONTEXT_SIZE} context probability maps are required"
        )
    result: list[np.ndarray] = []
    for index, raw in enumerate(context_probability_maps):
        if not isinstance(raw, np.ndarray):
            raise Stage2ContextTailAnchorError(
                f"context map {index} must be an explicit numpy array"
            )
        array = raw
        if array.dtype != np.float64 or array.ndim != 2 or array.size == 0:
            raise Stage2ContextTailAnchorError(
                f"context map {index} must be a nonempty 2D float64 array"
            )
        if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
            raise Stage2ContextTailAnchorError(f"context map {index} is invalid")
        canonical = np.array(array, dtype="<f8", order="C", copy=True)
        canonical.setflags(write=False)
        result.append(canonical)
    return tuple(result)


def _map_bindings(arrays: Sequence[np.ndarray]) -> tuple[list[dict[str, Any]], str]:
    bindings: list[dict[str, Any]] = []
    for ordinal, array in enumerate(arrays):
        bindings.append(
            {
                "ordinal": ordinal,
                "height": int(array.shape[0]),
                "width": int(array.shape[1]),
                "pixel_count": int(array.size),
                "content_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        )
    return bindings, canonical_json_sha256(bindings)


def exact_context_order_statistic_thresholds(
    context_probability_maps: Sequence[np.ndarray],
    *,
    budget_rationals: Sequence[tuple[int, int]] = BUDGET_RATIONALS,
) -> tuple[tuple[float, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return thresholds, allowed exceedances, observed exceedances and ranks."""

    arrays = _validated_maps(context_probability_maps)
    budgets = _validate_budget_rationals(budget_rationals)
    values = np.concatenate([array.reshape(-1) for array in arrays])
    total = int(values.size)
    allowed = tuple((numerator * total) // denominator for numerator, denominator in budgets)
    ranks = tuple(max(0, total - count - 1) for count in allowed)
    partitioned = np.partition(values, np.asarray(sorted(set(ranks)), dtype=np.intp))
    thresholds = tuple(float(partitioned[rank]) for rank in ranks)
    observed = tuple(int(np.count_nonzero(values > threshold)) for threshold in thresholds)
    if any(actual > maximum for actual, maximum in zip(observed, allowed, strict=True)):
        raise RuntimeError("strict order-statistic construction exceeded its exact budget")
    if any(right < left for left, right in zip(thresholds, thresholds[1:])):
        raise RuntimeError("tighter budgets produced decreasing T4 thresholds")
    return thresholds, allowed, observed, ranks


def build_context_tail_anchor(
    *,
    context_probability_maps: Sequence[np.ndarray],
    context_identity_sha256: str,
    budget_rationals: Sequence[tuple[int, int]] = BUDGET_RATIONALS,
) -> dict[str, Any]:
    """Build a result-free, label-blind and exactly replayable anchor artifact."""

    identity = _sha256(context_identity_sha256, "context_identity_sha256")
    arrays = _validated_maps(context_probability_maps)
    budgets = _validate_budget_rationals(budget_rationals)
    if budgets != BUDGET_RATIONALS:
        raise Stage2ContextTailAnchorError(
            "anchor artifacts require the frozen RC5 budget rationals"
        )
    thresholds, allowed, observed, ranks = exact_context_order_statistic_thresholds(
        arrays, budget_rationals=budgets
    )
    coordinates = encode_probability_numpy(np.asarray(thresholds, dtype=np.float64))
    kinds = endpoint_kinds_numpy(coordinates)
    bindings, content_sha256 = _map_bindings(arrays)
    rows = []
    for budget, maximum, actual, rank, threshold, coordinate, kind in zip(
        budgets,
        allowed,
        observed,
        ranks,
        thresholds,
        coordinates.tolist(),
        kinds,
        strict=True,
    ):
        rows.append(
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
        )
    payload: dict[str, Any] = {
        "schema_version": CONTEXT_TAIL_ANCHOR_SCHEMA,
        "artifact_type": CONTEXT_TAIL_ANCHOR_ARTIFACT_TYPE,
        "artifact_status": "complete",
        "context_identity_sha256": identity,
        "context_size": CONTEXT_SIZE,
        "total_context_pixels": sum(int(array.size) for array in arrays),
        "context_probability_content_algorithm": CONTEXT_PROBABILITY_CONTENT_ALGORITHM,
        "context_probability_content_sha256": content_sha256,
        "context_map_bindings": bindings,
        "budget_order": "strictly_descending_loose_to_strict",
        "budget_rationals": [
            {"numerator": numerator, "denominator": denominator}
            for numerator, denominator in budgets
        ],
        "threshold_rows": rows,
        "threshold_representation": representation_contract(),
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "selection_algorithm": CONTEXT_TAIL_ANCHOR_ALGORITHM,
        "guardrails": {
            "context_labels_accessed": False,
            "query_scores_accessed": False,
            "query_labels_accessed": False,
            "postlabel_statistics_accessed": False,
        },
    }
    # The identity preimage is exactly the completed payload *without* the
    # identity field itself.  Assign once: hashing a prior self field would
    # silently change this algorithm into an unreplayable recursive variant.
    payload["anchor_identity_sha256"] = canonical_json_sha256(payload)
    return payload


@dataclass(frozen=True, init=False)
class VerifiedContextTailAnchor:
    payload: Mapping[str, Any]
    thresholds: tuple[float, float, float]
    coordinates: tuple[float, float, float]
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("VerifiedContextTailAnchor is verifier-issued only")


def _verified_context_tail_anchor(
    *,
    payload: Mapping[str, Any],
    thresholds: tuple[float, float, float],
    coordinates: tuple[float, float, float],
) -> VerifiedContextTailAnchor:
    value = object.__new__(VerifiedContextTailAnchor)
    object.__setattr__(value, "payload", _freeze(payload))
    object.__setattr__(value, "thresholds", thresholds)
    object.__setattr__(value, "coordinates", coordinates)
    object.__setattr__(value, "_capability", _VERIFIED_CAPABILITY)
    return value


def verify_context_tail_anchor(
    payload: Mapping[str, Any],
    *,
    context_probability_maps: Sequence[np.ndarray],
    expected_context_identity_sha256: str,
) -> VerifiedContextTailAnchor:
    """Strictly verify by replaying ranks, score hashes, coordinates and ID."""

    top = _strict_fields(payload, _TOP_LEVEL_FIELDS, "anchor")
    if top["schema_version"] != CONTEXT_TAIL_ANCHOR_SCHEMA:
        raise Stage2ContextTailAnchorError("anchor schema_version is invalid")
    if top["artifact_type"] != CONTEXT_TAIL_ANCHOR_ARTIFACT_TYPE:
        raise Stage2ContextTailAnchorError("anchor artifact_type is invalid")
    if top["artifact_status"] != "complete":
        raise Stage2ContextTailAnchorError("anchor is not complete")
    identity = _sha256(
        expected_context_identity_sha256, "expected_context_identity_sha256"
    )
    if top["context_identity_sha256"] != identity:
        raise Stage2ContextTailAnchorError("anchor context identity mismatch")
    _strict_int(top["context_size"], "context_size", minimum=1)
    _strict_int(top["total_context_pixels"], "total_context_pixels", minimum=1)
    if top["context_probability_content_algorithm"] != CONTEXT_PROBABILITY_CONTENT_ALGORITHM:
        raise Stage2ContextTailAnchorError("context probability hash algorithm mismatch")
    _sha256(top["context_probability_content_sha256"], "context_probability_content_sha256")
    _sha256(top["anchor_identity_sha256"], "anchor_identity_sha256")
    if not isinstance(top["context_map_bindings"], list) or len(top["context_map_bindings"]) != CONTEXT_SIZE:
        raise Stage2ContextTailAnchorError("context_map_bindings cardinality mismatch")
    for index, raw in enumerate(top["context_map_bindings"]):
        row = _strict_fields(raw, _MAP_BINDING_FIELDS, f"context_map_bindings[{index}]")
        if _strict_int(row["ordinal"], f"context_map_bindings[{index}].ordinal") != index:
            raise Stage2ContextTailAnchorError("context map ordinals are not contiguous")
        for field in ("height", "width", "pixel_count"):
            _strict_int(row[field], f"context_map_bindings[{index}].{field}", minimum=1)
        _sha256(row["content_sha256"], f"context_map_bindings[{index}].content_sha256")
    if top["budget_order"] != "strictly_descending_loose_to_strict":
        raise Stage2ContextTailAnchorError("budget_order is invalid")
    if not isinstance(top["budget_rationals"], list):
        raise Stage2ContextTailAnchorError("budget_rationals must be a list")
    parsed_budgets = []
    for index, raw in enumerate(top["budget_rationals"]):
        row = _strict_fields(raw, _BUDGET_FIELDS, f"budget_rationals[{index}]")
        parsed_budgets.append((row["numerator"], row["denominator"]))
    budgets = _validate_budget_rationals(parsed_budgets)
    if budgets != BUDGET_RATIONALS:
        raise Stage2ContextTailAnchorError("budget rationals differ from the RC5 grid")
    if not isinstance(top["threshold_rows"], list) or len(top["threshold_rows"]) != 3:
        raise Stage2ContextTailAnchorError("threshold_rows must contain three rows")
    for index, raw in enumerate(top["threshold_rows"]):
        row = _strict_fields(raw, _ROW_FIELDS, f"threshold_rows[{index}]")
        for field in (
            "budget_numerator",
            "budget_denominator",
            "allowed_strict_exceedances",
            "observed_strict_exceedances",
            "order_statistic_rank_zero_based",
        ):
            _strict_int(row[field], f"threshold_rows[{index}].{field}")
        _canonical_float_hex(
            row["threshold_probability_hex"],
            f"threshold_rows[{index}].threshold_probability_hex",
        )
        _canonical_float_hex(
            row["threshold_coordinate_hex"],
            f"threshold_rows[{index}].threshold_coordinate_hex",
        )
        if not isinstance(row["threshold_kind"], str):
            raise Stage2ContextTailAnchorError("threshold_kind must be text")
    if top["threshold_representation"] != representation_contract():
        raise Stage2ContextTailAnchorError("threshold representation drifted")
    if top["threshold_semantics"] != STRICT_THRESHOLD_SEMANTICS:
        raise Stage2ContextTailAnchorError("threshold semantics drifted")
    if top["selection_algorithm"] != CONTEXT_TAIL_ANCHOR_ALGORITHM:
        raise Stage2ContextTailAnchorError("selection algorithm drifted")
    guardrails = _strict_fields(top["guardrails"], _GUARDRAIL_FIELDS, "guardrails")
    if any(guardrails.values()) or any(not isinstance(value, bool) for value in guardrails.values()):
        raise Stage2ContextTailAnchorError("anchor guardrails record forbidden access")

    expected = build_context_tail_anchor(
        context_probability_maps=context_probability_maps,
        context_identity_sha256=identity,
        budget_rationals=budgets,
    )
    if canonical_json_bytes(top) != canonical_json_bytes(expected):
        raise Stage2ContextTailAnchorError("anchor replay differs from the supplied artifact")
    thresholds = tuple(
        float.fromhex(row["threshold_probability_hex"])
        for row in expected["threshold_rows"]
    )
    coordinates = tuple(
        float.fromhex(row["threshold_coordinate_hex"])
        for row in expected["threshold_rows"]
    )
    return _verified_context_tail_anchor(
        payload=expected,
        thresholds=thresholds,  # type: ignore[arg-type]
        coordinates=coordinates,  # type: ignore[arg-type]
    )


def assert_verified_context_tail_anchor(
    value: VerifiedContextTailAnchor,
) -> VerifiedContextTailAnchor:
    if (
        not isinstance(value, VerifiedContextTailAnchor)
        or getattr(value, "_capability", None) is not _VERIFIED_CAPABILITY
    ):
        raise TypeError("a verifier-issued VerifiedContextTailAnchor is required")
    return value


__all__ = [
    "BUDGET_RATIONALS",
    "CONTEXT_SIZE",
    "CONTEXT_TAIL_ANCHOR_ALGORITHM",
    "CONTEXT_TAIL_ANCHOR_ARTIFACT_TYPE",
    "CONTEXT_TAIL_ANCHOR_SCHEMA",
    "Stage2ContextTailAnchorError",
    "VerifiedContextTailAnchor",
    "assert_verified_context_tail_anchor",
    "build_context_tail_anchor",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "exact_context_order_statistic_thresholds",
    "verify_context_tail_anchor",
]

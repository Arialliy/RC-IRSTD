"""Endpoint-aware coordinates for extreme probability thresholds.

The claim-bearing Stage-2 path must distinguish the largest finite float64
interior probability from the legal strict-``>`` upper endpoint.  A clipped
logit cannot do that: it merges a neighbourhood of one and cannot represent
``threshold == 1.0`` as a separate no-pixel decision.

RC5 therefore uses the piecewise identity/tail-log-survival coordinate

``s(p) = p`` for ``0 <= p <= 0.5``

``s(p) = 0.5 - log(2 * (1 - p))`` for ``0.5 < p < 1``

and one discrete, finite atom for ``p == 1``.  The atom is separated from the
largest float64 interior probability by one nat.  Raw model coordinates are
canonicalised with a hard forward map and an identity straight-through
gradient; exact validation and deployment always consume the hard value.

The contract is coordinate-centric: the lower half is an exact identity, the
upper endpoint is exact, and hard canonicalisation is bitwise idempotent.  It
does *not* claim bitwise probability roundtrips or codec re-encoding
idempotence in the logarithmic tail.  The frozen validation suite separately
audits strict encoding on a large adjacent-float sample and observes at most
one ULP of probability decode error on its declared sample.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


THRESHOLD_REPRESENTATION_SCHEMA = (
    "rc-irstd.endpoint-aware-piecewise-tail-coordinate.v2"
)
INTERIOR_KIND = "interior"
UPPER_ENDPOINT_KIND = "upper_endpoint"
PIECEWISE_SPLIT_PROBABILITY = 0.5
PIECEWISE_SPLIT_COORDINATE = 0.5

# ``1 - nextafter(1, 0) == 2**-53`` for IEEE-754 binary64, so the tail argument
# is ``2**-52`` and the maximum coordinate is ``0.5 + 52 * log(2)`` up to the
# platform's correctly rounded libm result.  Bind hexadecimal encodings in v7.
MAX_INTERIOR_PROBABILITY = float(np.nextafter(np.float64(1.0), np.float64(0.0)))
MAX_INTERIOR_COORDINATE = PIECEWISE_SPLIT_COORDINATE - math.log(
    2.0 * (1.0 - MAX_INTERIOR_PROBABILITY)
)
UPPER_ENDPOINT_COORDINATE = MAX_INTERIOR_COORDINATE + 1.0
UPPER_ENDPOINT_SPLIT = MAX_INTERIOR_COORDINATE + 0.5
RAW_COORDINATE_MIN = -1.0
RAW_COORDINATE_MAX = MAX_INTERIOR_COORDINATE + 1.5


class EndpointAwareThresholdError(ValueError):
    """An extreme-threshold value violates the RC5 representation contract."""


def representation_contract() -> dict[str, Any]:
    """Return the canonical, coordinate-centric EATC-v2 capability contract."""

    return {
        "schema_version": THRESHOLD_REPRESENTATION_SCHEMA,
        "coordinate": "piecewise_identity_tail_log_survival",
        "interior_encode": (
            "p if p <= 0.5 else 0.5 - log(2 * (1 - p))"
        ),
        "interior_decode": (
            "s if s <= 0.5 else 0.5 - 0.5 * expm1(-(s - 0.5))"
        ),
        "coordinate_compute_dtype": "float64",
        "piecewise_split_probability_hex": PIECEWISE_SPLIT_PROBABILITY.hex(),
        "piecewise_split_coordinate_hex": PIECEWISE_SPLIT_COORDINATE.hex(),
        "max_interior_probability_hex": MAX_INTERIOR_PROBABILITY.hex(),
        "max_interior_coordinate_hex": MAX_INTERIOR_COORDINATE.hex(),
        "upper_endpoint_coordinate_hex": UPPER_ENDPOINT_COORDINATE.hex(),
        "upper_endpoint_split_hex": UPPER_ENDPOINT_SPLIT.hex(),
        "raw_coordinate_min_hex": RAW_COORDINATE_MIN.hex(),
        "raw_coordinate_max_hex": RAW_COORDINATE_MAX.hex(),
        "endpoint_gap_coordinate_units_hex": float(1.0).hex(),
        "canonicalization": "hard_forward_identity_backward_v1",
        "threshold_semantics": "prediction = probability > threshold",
        "interior_kind": INTERIOR_KIND,
        "upper_endpoint_kind": UPPER_ENDPOINT_KIND,
    }


def _as_probability_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise EndpointAwareThresholdError("probabilities must be finite in [0, 1]")
    return array


def encode_probability_numpy(value: Any) -> np.ndarray:
    """Encode thresholds with an exact endpoint branch.

    Values through one half are copied bit-for-bit.  Tail-log decoding is
    numerical and is not promised to reproduce every input probability bit.
    """

    probability = _as_probability_array(value)
    result = np.empty_like(probability, dtype=np.float64)
    endpoint = probability == 1.0
    result[endpoint] = UPPER_ENDPOINT_COORDINATE
    interior = ~endpoint
    lower = interior & (probability <= PIECEWISE_SPLIT_PROBABILITY)
    result[lower] = probability[lower]
    tail = interior & ~lower
    if bool(np.any(tail)):
        result[tail] = PIECEWISE_SPLIT_COORDINATE - np.log(
            2.0 * (1.0 - probability[tail])
        )
    return result


def encode_probability_scalar(value: float) -> float:
    return float(encode_probability_numpy(np.asarray(value, dtype=np.float64)))


def _validate_canonical_numpy(value: Any) -> np.ndarray:
    coordinate = np.asarray(value, dtype=np.float64)
    if not np.isfinite(coordinate).all():
        raise EndpointAwareThresholdError("canonical coordinates must be finite")
    endpoint = coordinate == UPPER_ENDPOINT_COORDINATE
    interior = (coordinate >= 0.0) & (coordinate <= MAX_INTERIOR_COORDINATE)
    if not bool(np.all(endpoint | interior)):
        raise EndpointAwareThresholdError(
            "coordinate is neither an interior value nor the exact upper endpoint"
        )
    return coordinate


def decode_coordinate_numpy(value: Any) -> np.ndarray:
    """Decode coordinates while preserving the exact endpoint atom.

    Lower-half coordinates are copied bit-for-bit.  Tail decoding is subject
    to float64 rounding and is not a bitwise inverse for every probability.
    """

    coordinate = _validate_canonical_numpy(value)
    result = np.empty_like(coordinate, dtype=np.float64)
    endpoint = coordinate == UPPER_ENDPOINT_COORDINATE
    result[endpoint] = 1.0
    interior = ~endpoint
    lower = interior & (coordinate <= PIECEWISE_SPLIT_COORDINATE)
    result[lower] = coordinate[lower]
    tail = interior & ~lower
    if bool(np.any(tail)):
        delta = coordinate[tail] - PIECEWISE_SPLIT_COORDINATE
        decoded = PIECEWISE_SPLIT_PROBABILITY - 0.5 * np.expm1(-delta)
        result[tail] = np.minimum(decoded, MAX_INTERIOR_PROBABILITY)
    return result


def decode_coordinate_scalar(value: float) -> float:
    return float(decode_coordinate_numpy(np.asarray(value, dtype=np.float64)))


def canonicalize_raw_numpy(value: Any) -> np.ndarray:
    """Apply the exact, bitwise-idempotent hard map to raw coordinates."""

    raw = np.asarray(value, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise EndpointAwareThresholdError("raw coordinates must be finite")
    if np.any((raw < RAW_COORDINATE_MIN) | (raw > RAW_COORDINATE_MAX)):
        raise EndpointAwareThresholdError("raw coordinates exceed frozen bounds")
    hard = np.clip(raw, 0.0, MAX_INTERIOR_COORDINATE)
    return np.where(raw >= UPPER_ENDPOINT_SPLIT, UPPER_ENDPOINT_COORDINATE, hard)


def canonicalize_raw_torch(raw: torch.Tensor) -> torch.Tensor:
    """Bitwise-idempotent hard-forward/identity-backward map in float64."""

    if not isinstance(raw, torch.Tensor) or not raw.is_floating_point():
        raise TypeError("raw coordinates must be a floating-point tensor")
    if not bool(torch.isfinite(raw).all().item()):
        raise EndpointAwareThresholdError("raw coordinates must be finite")
    values = raw.to(dtype=torch.float64)
    if bool(
        ((values < RAW_COORDINATE_MIN) | (values > RAW_COORDINATE_MAX)).any().item()
    ):
        raise EndpointAwareThresholdError("raw coordinates exceed frozen bounds")
    hard = values.clamp(min=0.0, max=MAX_INTERIOR_COORDINATE)
    hard = torch.where(
        values >= UPPER_ENDPOINT_SPLIT,
        torch.as_tensor(
            UPPER_ENDPOINT_COORDINATE, dtype=torch.float64, device=values.device
        ),
        hard,
    )
    return values + (hard - values).detach()


def decode_coordinate_torch(coordinate: torch.Tensor) -> torch.Tensor:
    """Decode a canonical tensor, with exact endpoint and numerical interior."""

    if not isinstance(coordinate, torch.Tensor) or not coordinate.is_floating_point():
        raise TypeError("canonical coordinates must be a floating-point tensor")
    values = coordinate.to(dtype=torch.float64)
    if not bool(torch.isfinite(values).all().item()):
        raise EndpointAwareThresholdError("canonical coordinates must be finite")
    endpoint = values == UPPER_ENDPOINT_COORDINATE
    interior = (values >= 0.0) & (values <= MAX_INTERIOR_COORDINATE)
    if not bool((endpoint | interior).all().item()):
        raise EndpointAwareThresholdError(
            "coordinate is neither an interior value nor the exact upper endpoint"
        )
    interior_values = values.clamp(max=MAX_INTERIOR_COORDINATE)
    tail_delta = (interior_values - PIECEWISE_SPLIT_COORDINATE).clamp(min=0.0)
    tail_decoded = PIECEWISE_SPLIT_PROBABILITY - 0.5 * torch.expm1(-tail_delta)
    decoded = torch.where(
        interior_values <= PIECEWISE_SPLIT_COORDINATE,
        interior_values,
        tail_decoded,
    )
    decoded = decoded.clamp(max=MAX_INTERIOR_PROBABILITY)
    return torch.where(endpoint, torch.ones_like(decoded), decoded)


def endpoint_kinds_numpy(value: Any) -> tuple[str, ...]:
    coordinate = _validate_canonical_numpy(value).reshape(-1)
    return tuple(
        UPPER_ENDPOINT_KIND
        if item == UPPER_ENDPOINT_COORDINATE
        else INTERIOR_KIND
        for item in coordinate
    )


def assert_monotone_coordinate_decision(
    coordinates: Any,
    thresholds: Any,
    *,
    strict_raw_order: bool = True,
) -> None:
    """Validate the T7/T8 coordinate, decoded-order and endpoint-suffix rules.

    ``thresholds`` must be the decoder outputs carried with ``coordinates``;
    pre-encoding interior probabilities may differ from them by one ULP.
    """

    coordinate = _validate_canonical_numpy(coordinates).reshape(-1)
    probability = _as_probability_array(thresholds).reshape(-1)
    if coordinate.shape != probability.shape or coordinate.size < 2:
        raise EndpointAwareThresholdError("coordinate/threshold curve shape mismatch")
    differences = np.diff(coordinate)
    if strict_raw_order:
        # Canonicalisation may collapse distinct raw interior values at zero or
        # the maximum interior plateau.  The decision contract therefore only
        # requires nondecreasing canonical coordinates; raw strictness is
        # checked directly by the model before canonicalisation.
        if np.any(differences < 0.0):
            raise EndpointAwareThresholdError("canonical coordinates decrease")
    if np.any(np.diff(probability) < 0.0):
        raise EndpointAwareThresholdError("decoded thresholds decrease")
    expected = decode_coordinate_numpy(coordinate)
    if not np.array_equal(expected, probability):
        raise EndpointAwareThresholdError("thresholds do not exactly decode from coordinates")
    kinds = endpoint_kinds_numpy(coordinate)
    seen_endpoint = False
    for kind in kinds:
        if kind == UPPER_ENDPOINT_KIND:
            seen_endpoint = True
        elif seen_endpoint:
            raise EndpointAwareThresholdError("upper endpoints must form one suffix")


def resolve_strict_threshold_event_row(
    curve_thresholds: Any, threshold: float
) -> int:
    """Resolve a strict-``>`` threshold to its exact ascending event row."""

    values = np.asarray(curve_thresholds, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or values[0] != 0.0
        or values[-1] != 1.0
        or np.any(values[1:] <= values[:-1])
    ):
        raise EndpointAwareThresholdError(
            "curve thresholds must be finite, strictly ascending, and include 0/1"
        )
    probability = float(threshold)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise EndpointAwareThresholdError("threshold must be finite in [0, 1]")
    return max(
        0,
        min(
            values.size - 1,
            int(np.searchsorted(values, probability, side="right") - 1),
        ),
    )


__all__ = [
    "EndpointAwareThresholdError",
    "INTERIOR_KIND",
    "MAX_INTERIOR_COORDINATE",
    "MAX_INTERIOR_PROBABILITY",
    "PIECEWISE_SPLIT_COORDINATE",
    "PIECEWISE_SPLIT_PROBABILITY",
    "RAW_COORDINATE_MAX",
    "RAW_COORDINATE_MIN",
    "THRESHOLD_REPRESENTATION_SCHEMA",
    "UPPER_ENDPOINT_COORDINATE",
    "UPPER_ENDPOINT_KIND",
    "UPPER_ENDPOINT_SPLIT",
    "assert_monotone_coordinate_decision",
    "canonicalize_raw_numpy",
    "canonicalize_raw_torch",
    "decode_coordinate_numpy",
    "decode_coordinate_scalar",
    "decode_coordinate_torch",
    "encode_probability_numpy",
    "encode_probability_scalar",
    "endpoint_kinds_numpy",
    "representation_contract",
    "resolve_strict_threshold_event_row",
]

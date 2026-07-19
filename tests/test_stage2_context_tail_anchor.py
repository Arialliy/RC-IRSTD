from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import math

import numpy as np
import pytest

from model.endpoint_aware_threshold import (
    UPPER_ENDPOINT_COORDINATE,
    UPPER_ENDPOINT_KIND,
    encode_probability_numpy,
    endpoint_kinds_numpy,
)
from rc.stage2_context_tail_anchor import (
    BUDGET_RATIONALS,
    CONTEXT_SIZE,
    Stage2ContextTailAnchorError,
    VerifiedContextTailAnchor,
    assert_verified_context_tail_anchor,
    build_context_tail_anchor,
    canonical_json_sha256,
    exact_context_order_statistic_thresholds,
    verify_context_tail_anchor,
)


def _maps(values: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and values.size >= CONTEXT_SIZE
    return tuple(
        np.ascontiguousarray(chunk.reshape(1, -1), dtype=np.float64)
        for chunk in np.array_split(values, CONTEXT_SIZE)
    )


def _identity(tag: str) -> str:
    return canonical_json_sha256({"synthetic_context": tag})


def _resign(payload: dict[str, object]) -> None:
    preimage = copy.deepcopy(payload)
    preimage.pop("anchor_identity_sha256")
    payload["anchor_identity_sha256"] = canonical_json_sha256(preimage)


def test_exact_rational_boundary_never_uses_float_multiplication() -> None:
    values = np.linspace(0.0, 1.0, 100, dtype=np.float64)
    budgets = ((29, 100), (1, 4), (1, 5))

    thresholds, allowed, observed, ranks = exact_context_order_statistic_thresholds(
        _maps(values), budget_rationals=budgets
    )

    # This is the binary64 trap the implementation must avoid.
    assert math.floor(float(Fraction(29, 100)) * values.size) == 28
    assert allowed == (29, 25, 20)
    assert observed == allowed
    assert ranks == (70, 74, 79)
    assert thresholds == tuple(float(values[index]) for index in ranks)


def test_strict_greater_ties_and_zero_one_endpoints_are_exact() -> None:
    values = np.asarray([0.0] + [0.5] * 12 + [1.0], dtype=np.float64)
    budgets = ((13, 14), (1, 14), (1, 28))

    thresholds, allowed, observed, ranks = exact_context_order_statistic_thresholds(
        _maps(values), budget_rationals=budgets
    )
    coordinates = encode_probability_numpy(np.asarray(thresholds, dtype=np.float64))

    assert thresholds == (0.0, 0.5, 1.0)
    assert allowed == observed == (13, 1, 0)
    assert ranks == (0, 12, 13)
    assert tuple(int(np.count_nonzero(values > value)) for value in thresholds) == observed
    assert int(np.count_nonzero(values >= 0.5)) == 13  # proves ties are not counted
    assert coordinates[0] == 0.0
    assert coordinates[1] == 0.5
    assert coordinates[2] == UPPER_ENDPOINT_COORDINATE
    assert endpoint_kinds_numpy(coordinates)[-1] == UPPER_ENDPOINT_KIND


def test_frozen_budget_order_yields_monotone_coordinates_and_bound_hashes() -> None:
    maps = _maps(np.linspace(0.0, 1.0, 10_000, dtype=np.float64))
    payload = build_context_tail_anchor(
        context_probability_maps=maps,
        context_identity_sha256=_identity("monotone"),
    )
    rows = payload["threshold_rows"]
    thresholds = tuple(
        float.fromhex(row["threshold_probability_hex"]) for row in rows
    )
    coordinates = tuple(
        float.fromhex(row["threshold_coordinate_hex"]) for row in rows
    )

    assert payload["budget_rationals"] == [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in BUDGET_RATIONALS
    ]
    assert [row["allowed_strict_exceedances"] for row in rows] == [1, 0, 0]
    assert [row["observed_strict_exceedances"] for row in rows] == [1, 0, 0]
    assert all(left <= right for left, right in zip(thresholds, thresholds[1:]))
    assert all(left <= right for left, right in zip(coordinates, coordinates[1:]))
    assert thresholds[-1] == 1.0
    assert rows[-1]["threshold_kind"] == UPPER_ENDPOINT_KIND

    bindings = payload["context_map_bindings"]
    assert payload["context_probability_content_sha256"] == canonical_json_sha256(
        bindings
    )
    for array, binding in zip(maps, bindings, strict=True):
        canonical = np.ascontiguousarray(array, dtype="<f8")
        assert binding["content_sha256"] == hashlib.sha256(
            canonical.tobytes(order="C")
        ).hexdigest()
    preimage = copy.deepcopy(payload)
    anchor_identity = preimage.pop("anchor_identity_sha256")
    assert anchor_identity == canonical_json_sha256(preimage)


def test_anchor_identity_hashes_payload_once_without_its_self_field() -> None:
    payload = build_context_tail_anchor(
        context_probability_maps=_maps(
            np.linspace(0.0, 1.0, 140, dtype=np.float64)
        ),
        context_identity_sha256=_identity("single-anchor-identity-hash"),
    )
    preimage = copy.deepcopy(payload)
    claimed_identity = preimage.pop("anchor_identity_sha256")

    assert claimed_identity == canonical_json_sha256(preimage)
    # A historical duplicate assignment hashed the first identity as a self
    # field.  Bind that this is not the declared algorithm.
    recursive_preimage = {**preimage, "anchor_identity_sha256": claimed_identity}
    assert claimed_identity != canonical_json_sha256(recursive_preimage)


def test_builder_rejects_nonfrozen_but_otherwise_valid_budget_grid() -> None:
    with pytest.raises(Stage2ContextTailAnchorError, match="frozen RC5"):
        build_context_tail_anchor(
            context_probability_maps=_maps(
                np.linspace(0.0, 1.0, 100, dtype=np.float64)
            ),
            context_identity_sha256=_identity("custom-budget"),
            budget_rationals=((29, 100), (1, 4), (1, 5)),
        )


def test_verifier_replays_artifact_and_issues_immutable_capability() -> None:
    maps = _maps(np.linspace(0.0, 1.0, 140, dtype=np.float64))
    identity = _identity("verified")
    payload = build_context_tail_anchor(
        context_probability_maps=maps,
        context_identity_sha256=identity,
    )

    verified = verify_context_tail_anchor(
        payload,
        context_probability_maps=maps,
        expected_context_identity_sha256=identity,
    )

    assert assert_verified_context_tail_anchor(verified) is verified
    assert verified.thresholds == tuple(
        float.fromhex(row["threshold_probability_hex"])
        for row in payload["threshold_rows"]
    )
    assert verified.coordinates == tuple(
        float.fromhex(row["threshold_coordinate_hex"])
        for row in payload["threshold_rows"]
    )
    with pytest.raises(TypeError):
        verified.payload["context_size"] = 99
    with pytest.raises(TypeError):
        verified.payload["context_map_bindings"][0]["ordinal"] = 99


def test_verified_capability_cannot_be_constructed_directly() -> None:
    with pytest.raises(TypeError, match="verifier-issued only"):
        VerifiedContextTailAnchor()
    with pytest.raises(TypeError, match="verifier-issued only"):
        VerifiedContextTailAnchor(
            payload={}, thresholds=(0.0, 0.0, 0.0), coordinates=(0.0, 0.0, 0.0)
        )

    forged = object.__new__(VerifiedContextTailAnchor)
    with pytest.raises(TypeError, match="verifier-issued"):
        assert_verified_context_tail_anchor(forged)


def test_map_binding_tamper_is_rejected_even_after_internal_hashes_are_updated() -> None:
    maps = _maps(np.linspace(0.0, 1.0, 140, dtype=np.float64))
    identity = _identity("map-binding")
    payload = build_context_tail_anchor(
        context_probability_maps=maps,
        context_identity_sha256=identity,
    )
    tampered = copy.deepcopy(payload)
    tampered["context_map_bindings"][0]["content_sha256"] = "0" * 64
    tampered["context_probability_content_sha256"] = canonical_json_sha256(
        tampered["context_map_bindings"]
    )
    _resign(tampered)

    with pytest.raises(Stage2ContextTailAnchorError, match="replay differs"):
        verify_context_tail_anchor(
            tampered,
            context_probability_maps=maps,
            expected_context_identity_sha256=identity,
        )


def test_changed_map_content_and_context_identity_tampering_are_rejected() -> None:
    maps = _maps(np.linspace(0.0, 1.0, 140, dtype=np.float64))
    identity = _identity("tamper")
    payload = build_context_tail_anchor(
        context_probability_maps=maps,
        context_identity_sha256=identity,
    )

    changed_maps = [array.copy() for array in maps]
    changed_maps[0][0, 0] = 0.123
    with pytest.raises(Stage2ContextTailAnchorError, match="replay differs"):
        verify_context_tail_anchor(
            payload,
            context_probability_maps=changed_maps,
            expected_context_identity_sha256=identity,
        )

    changed_identity = copy.deepcopy(payload)
    changed_identity["context_identity_sha256"] = _identity("attacker")
    _resign(changed_identity)
    with pytest.raises(Stage2ContextTailAnchorError, match="context identity mismatch"):
        verify_context_tail_anchor(
            changed_identity,
            context_probability_maps=maps,
            expected_context_identity_sha256=identity,
        )


@pytest.mark.parametrize(
    "case",
    (
        "too_few",
        "too_many",
        "generator",
        "implicit_list",
        "float32",
        "rank_one",
        "empty",
        "nan",
        "out_of_range",
    ),
)
def test_invalid_map_count_shape_dtype_and_values_are_rejected(case: str) -> None:
    maps: object = list(_maps(np.linspace(0.0, 1.0, 140, dtype=np.float64)))
    if case == "too_few":
        maps = maps[:-1]
    elif case == "too_many":
        maps = maps + [np.zeros((1, 1), dtype=np.float64)]
    elif case == "generator":
        maps = (array for array in maps)
    elif case == "implicit_list":
        maps[0] = [[0.0, 0.1]]
    elif case == "float32":
        maps[0] = maps[0].astype(np.float32)
    elif case == "rank_one":
        maps[0] = maps[0].reshape(-1)
    elif case == "empty":
        maps[0] = np.empty((1, 0), dtype=np.float64)
    elif case == "nan":
        maps[0][0, 0] = np.nan
    elif case == "out_of_range":
        maps[0][0, 0] = np.nextafter(np.float64(1.0), np.float64(2.0))

    with pytest.raises(Stage2ContextTailAnchorError):
        exact_context_order_statistic_thresholds(maps)


@pytest.mark.parametrize(
    "budgets",
    (
        ((1, 10_000), (1, 100_000)),
        ((2, 20_000), (1, 100_000), (1, 1_000_000)),
        ((1, 100_000), (1, 10_000), (1, 1_000_000)),
        ((1, 10_000), (1, 10_000), (1, 1_000_000)),
        ((True, 10_000), (1, 100_000), (1, 1_000_000)),
        ((np.int64(1), 10_000), (1, 100_000), (1, 1_000_000)),
    ),
)
def test_invalid_or_noncanonical_budget_rationals_are_rejected(budgets) -> None:
    with pytest.raises(Stage2ContextTailAnchorError):
        exact_context_order_statistic_thresholds(
            _maps(np.linspace(0.0, 1.0, 140, dtype=np.float64)),
            budget_rationals=budgets,
        )


def test_context_identity_must_be_exact_lowercase_sha256() -> None:
    with pytest.raises(Stage2ContextTailAnchorError, match="lowercase SHA-256"):
        build_context_tail_anchor(
            context_probability_maps=_maps(
                np.linspace(0.0, 1.0, 140, dtype=np.float64)
            ),
            context_identity_sha256="A" * 64,
        )

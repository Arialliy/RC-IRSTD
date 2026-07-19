from __future__ import annotations

import copy
from fractions import Fraction
import math

import numpy as np
import pytest
import torch

from model.budget_conditioned_endpoint_calibrator import (
    BUDGET_KNOT_RATIONALS,
    BudgetConditionedMonotoneEndpointAwarePixelCalibrator,
)
from model.endpoint_aware_threshold import UPPER_ENDPOINT_COORDINATE
from rc.stage2_context_tail_anchor import (
    BUDGET_RATIONALS,
    CONTEXT_SIZE,
    build_context_tail_anchor,
    canonical_json_sha256,
)
from rc.stage2_context_tail_anchor_v2 import (
    CONTEXT_TAIL_ANCHOR_V2_SCHEMA,
    MAX_EXACT_RATIONAL_INTEGER,
    Stage2ContextTailAnchorV2Error,
    VerifiedContextTailAnchorV2,
    assert_verified_context_tail_anchor_v2,
    build_context_tail_anchor_v2,
    verify_context_tail_anchor_v2,
)


REQUESTED = ((1, 20_000), (1, 100_000), (1, 250_000))


def _maps(values: np.ndarray) -> tuple[np.ndarray, ...]:
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and values.size >= CONTEXT_SIZE
    return tuple(
        np.ascontiguousarray(chunk.reshape(1, -1), dtype=np.float64)
        for chunk in np.array_split(values, CONTEXT_SIZE)
    )


def _identity(tag: str) -> str:
    return canonical_json_sha256({"synthetic_context_v2": tag})


def _resign(payload: dict[str, object]) -> None:
    preimage = copy.deepcopy(payload)
    preimage.pop("anchor_identity_sha256")
    payload["anchor_identity_sha256"] = canonical_json_sha256(preimage)


def _coordinates(rows: list[dict[str, object]]) -> tuple[float, ...]:
    return tuple(float.fromhex(str(row["threshold_coordinate_hex"])) for row in rows)


def test_v2_binds_nine_grid_rows_and_direct_same_budget_request_rows() -> None:
    maps = _maps(np.linspace(0.0, 1.0, 140_000, dtype=np.float64))
    payload = build_context_tail_anchor_v2(
        context_probability_maps=maps,
        context_identity_sha256=_identity("grid-and-request"),
        requested_budget_rationals=REQUESTED,
    )

    assert payload["schema_version"] == CONTEXT_TAIL_ANCHOR_V2_SCHEMA
    assert payload["grid_budget_rationals"] == [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in BUDGET_KNOT_RATIONALS
    ]
    assert payload["requested_budget_rationals"] == [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in REQUESTED
    ]
    assert len(payload["grid_threshold_rows"]) == 9
    assert len(payload["requested_threshold_rows"]) == 3
    assert payload["requested_anchor_source"].endswith("not_grid_interpolation")
    assert not any(payload["guardrails"].values())

    for rows in (payload["grid_threshold_rows"], payload["requested_threshold_rows"]):
        assert all(
            row["observed_strict_exceedances"]
            <= row["allowed_strict_exceedances"]
            for row in rows
        )
        coordinates = _coordinates(rows)
        assert all(left <= right for left, right in zip(coordinates, coordinates[1:]))

    # The request contains the exact primary 1/100000 knot.  Its analytic row
    # must replay exactly rather than merely agree after numeric interpolation.
    assert payload["requested_threshold_rows"][1] == payload["grid_threshold_rows"][4]


def test_requested_anchor_is_not_interpolated_from_neighboring_grid_anchors() -> None:
    rng = np.random.default_rng(3407)
    values = np.sort(rng.beta(0.7, 3.0, size=140_000).astype(np.float64))
    requested = ((1, 25_000),)
    payload = build_context_tail_anchor_v2(
        context_probability_maps=_maps(values),
        context_identity_sha256=_identity("not-interpolated"),
        requested_budget_rationals=requested,
    )
    grid = _coordinates(payload["grid_threshold_rows"])
    actual = _coordinates(payload["requested_threshold_rows"])[0]
    b = Fraction(*requested[0])
    left_b = Fraction(*BUDGET_KNOT_RATIONALS[1])
    right_b = Fraction(*BUDGET_KNOT_RATIONALS[2])
    u = (
        (math.log(float(b)) - math.log(float(left_b)))
        / (math.log(float(right_b)) - math.log(float(left_b)))
    )
    interpolated = grid[1] + u * (grid[2] - grid[1])

    assert actual != interpolated
    row = payload["requested_threshold_rows"][0]
    assert row["allowed_strict_exceedances"] == (1 * values.size) // 25_000
    assert row["order_statistic_rank_zero_based"] == (
        values.size - row["allowed_strict_exceedances"] - 1
    )


def test_v2_zero_exceedance_rows_preserve_exact_upper_endpoint() -> None:
    values = np.asarray([0.0] + [0.5] * 12 + [1.0], dtype=np.float64)
    payload = build_context_tail_anchor_v2(
        context_probability_maps=_maps(values),
        context_identity_sha256=_identity("endpoint"),
        requested_budget_rationals=((1, 20_000),),
    )

    for row in payload["grid_threshold_rows"] + payload["requested_threshold_rows"]:
        assert row["allowed_strict_exceedances"] == 0
        assert row["observed_strict_exceedances"] == 0
        assert float.fromhex(row["threshold_probability_hex"]) == 1.0
        assert (
            float.fromhex(row["threshold_coordinate_hex"])
            == UPPER_ENDPOINT_COORDINATE
        )
        assert row["threshold_kind"] == "upper_endpoint"


def test_verifier_replays_v2_and_issues_immutable_capability() -> None:
    maps = _maps(np.linspace(0.0, 1.0, 14_000, dtype=np.float64))
    identity = _identity("verified")
    payload = build_context_tail_anchor_v2(
        context_probability_maps=maps,
        context_identity_sha256=identity,
        requested_budget_rationals=REQUESTED,
    )

    verified = verify_context_tail_anchor_v2(
        payload,
        context_probability_maps=maps,
        expected_context_identity_sha256=identity,
        expected_requested_budget_rationals=REQUESTED,
    )

    assert assert_verified_context_tail_anchor_v2(verified) is verified
    assert verified.grid_budget_rationals == BUDGET_KNOT_RATIONALS
    assert verified.requested_budget_rationals == REQUESTED
    assert verified.grid_coordinates == _coordinates(payload["grid_threshold_rows"])
    assert verified.requested_coordinates == _coordinates(
        payload["requested_threshold_rows"]
    )
    with pytest.raises(TypeError):
        verified.payload["artifact_status"] = "forged"
    with pytest.raises(TypeError):
        verified.payload["grid_threshold_rows"][0]["threshold_kind"] = "forged"


def test_v2_capability_cannot_be_constructed_or_forged() -> None:
    with pytest.raises(TypeError, match="verifier-issued only"):
        VerifiedContextTailAnchorV2()
    forged = object.__new__(VerifiedContextTailAnchorV2)
    with pytest.raises(TypeError, match="verifier-issued"):
        assert_verified_context_tail_anchor_v2(forged)


def test_model_consumes_verified_v2_grid_and_requested_coordinates() -> None:
    maps = _maps(np.linspace(0.0, 1.0, 14_000, dtype=np.float64))
    identity = _identity("model-integration")
    payload = build_context_tail_anchor_v2(
        context_probability_maps=maps,
        context_identity_sha256=identity,
        requested_budget_rationals=REQUESTED,
    )
    verified = verify_context_tail_anchor_v2(
        payload,
        context_probability_maps=maps,
        expected_context_identity_sha256=identity,
        expected_requested_budget_rationals=REQUESTED,
    )
    model = BudgetConditionedMonotoneEndpointAwarePixelCalibrator(
        93, dropout=0.0
    )
    numerators = torch.tensor([row[0] for row in REQUESTED], dtype=torch.int64)
    denominators = torch.tensor([row[1] for row in REQUESTED], dtype=torch.int64)

    output = model(
        torch.randn(1, 93),
        anchor_coordinates=torch.tensor(
            [verified.grid_coordinates], dtype=torch.float64
        ),
        budget_numerators=numerators,
        budget_denominators=denominators,
        requested_anchor_coordinates=torch.tensor(
            [verified.requested_coordinates], dtype=torch.float64
        ),
    )

    assert output.grid_coordinates.shape == (1, 9)
    assert output.requested_coordinates.shape == (1, 3)
    assert torch.all(
        output.requested_raw_coordinates[:, 1:]
        > output.requested_raw_coordinates[:, :-1]
    )


def test_changed_maps_and_resigned_internal_tampering_are_rejected() -> None:
    maps = _maps(np.linspace(0.0, 1.0, 14_000, dtype=np.float64))
    identity = _identity("tamper")
    payload = build_context_tail_anchor_v2(
        context_probability_maps=maps,
        context_identity_sha256=identity,
        requested_budget_rationals=REQUESTED,
    )
    changed_maps = [item.copy() for item in maps]
    changed_maps[0][0, 0] = 0.123
    with pytest.raises(Stage2ContextTailAnchorV2Error, match="replay differs"):
        verify_context_tail_anchor_v2(
            payload,
            context_probability_maps=changed_maps,
            expected_context_identity_sha256=identity,
            expected_requested_budget_rationals=REQUESTED,
        )

    tampered = copy.deepcopy(payload)
    tampered["requested_threshold_rows"][0]["threshold_probability_hex"] = 0.5.hex()
    _resign(tampered)
    with pytest.raises(Stage2ContextTailAnchorV2Error, match="replay differs"):
        verify_context_tail_anchor_v2(
            tampered,
            context_probability_maps=maps,
            expected_context_identity_sha256=identity,
            expected_requested_budget_rationals=REQUESTED,
        )


def test_requested_budget_identity_mismatch_is_rejected_before_capability_issue() -> None:
    maps = _maps(np.linspace(0.0, 1.0, 1_400, dtype=np.float64))
    identity = _identity("budget-mismatch")
    payload = build_context_tail_anchor_v2(
        context_probability_maps=maps,
        context_identity_sha256=identity,
        requested_budget_rationals=REQUESTED,
    )
    with pytest.raises(Stage2ContextTailAnchorV2Error, match="identity mismatch"):
        verify_context_tail_anchor_v2(
            payload,
            context_probability_maps=maps,
            expected_context_identity_sha256=identity,
            expected_requested_budget_rationals=((1, 30_000),),
        )


def test_grid_lattice_is_frozen_without_mutating_v1() -> None:
    maps = _maps(np.linspace(0.0, 1.0, 1_400, dtype=np.float64))
    changed = list(BUDGET_KNOT_RATIONALS)
    changed[1] = (1, 20_000)
    with pytest.raises(Stage2ContextTailAnchorV2Error, match=r"frozen RC5\+"):
        build_context_tail_anchor_v2(
            context_probability_maps=maps,
            context_identity_sha256=_identity("changed-grid"),
            grid_budget_rationals=changed,
        )

    v1 = build_context_tail_anchor(
        context_probability_maps=maps,
        context_identity_sha256=_identity("v1-still-frozen"),
    )
    assert v1["schema_version"] == "rc-irstd.stage2-context-tail-anchor.v1"
    assert v1["budget_rationals"] == [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in BUDGET_RATIONALS
    ]


@pytest.mark.parametrize(
    ("requested", "message"),
    [
        (((2, 20_000),), "lowest-term"),
        (((1, 20_000), (1, 20_000)), "strictly"),
        (((1, 5_000),), "knot range"),
        (((1, 2_000_000),), "knot range"),
        (((True, 20_000),), "must be int"),
        (((1, MAX_EXACT_RATIONAL_INTEGER + 1),), "must be int <="),
    ],
)
def test_invalid_requested_rationals_are_rejected(requested, message: str) -> None:
    with pytest.raises(Stage2ContextTailAnchorV2Error, match=message):
        build_context_tail_anchor_v2(
            context_probability_maps=_maps(
                np.linspace(0.0, 1.0, 1_400, dtype=np.float64)
            ),
            context_identity_sha256=_identity("invalid-request"),
            requested_budget_rationals=requested,
        )


def test_float64_colliding_requested_rationals_are_rejected() -> None:
    numerator = 90_000_000_000_000
    requested = (
        (numerator, 100_000 * numerator - 1),
        (numerator, 100_000 * numerator + 1),
    )
    assert Fraction(*requested[0]) > Fraction(*requested[1])
    with pytest.raises(Stage2ContextTailAnchorV2Error, match="distinguishable"):
        build_context_tail_anchor_v2(
            context_probability_maps=_maps(
                np.linspace(0.0, 1.0, 1_400, dtype=np.float64)
            ),
            context_identity_sha256=_identity("float64-collision"),
            requested_budget_rationals=requested,
        )


@pytest.mark.parametrize(
    "case",
    ("few", "float32", "rank_one", "nan", "out_of_range"),
)
def test_invalid_context_maps_are_rejected(case: str) -> None:
    maps: object = list(_maps(np.linspace(0.0, 1.0, 1_400, dtype=np.float64)))
    if case == "few":
        maps = maps[:-1]
    elif case == "float32":
        maps[0] = maps[0].astype(np.float32)
    elif case == "rank_one":
        maps[0] = maps[0].reshape(-1)
    elif case == "nan":
        maps[0][0, 0] = np.nan
    elif case == "out_of_range":
        maps[0][0, 0] = np.nextafter(np.float64(1.0), np.float64(2.0))

    with pytest.raises(Stage2ContextTailAnchorV2Error):
        build_context_tail_anchor_v2(
            context_probability_maps=maps,
            context_identity_sha256=_identity("invalid-maps"),
        )

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import torch

from model.budget_conditioned_endpoint_calibrator import (
    ANCHOR_MIX_RULE,
    BUDGET_CONDITIONED_SCHEMA,
    BUDGET_KNOT_RATIONALS,
    PRIMARY_BUDGET_KNOT_INDICES,
    PRIMARY_BUDGET_RATIONALS,
    BudgetConditionedCalibratorError,
    BudgetConditionedDirectEndpointAwarePixelCalibrator,
    BudgetConditionedMonotoneEndpointAwarePixelCalibrator,
)
from model.endpoint_aware_pixel_calibrator import ANCHOR_MIX_INITIAL_WEIGHT
from model.endpoint_aware_threshold import (
    MAX_INTERIOR_COORDINATE,
    UPPER_ENDPOINT_COORDINATE,
    encode_probability_numpy,
)


MODEL_TYPES = (
    BudgetConditionedDirectEndpointAwarePixelCalibrator,
    BudgetConditionedMonotoneEndpointAwarePixelCalibrator,
)


def _model(model_type: type[torch.nn.Module]) -> torch.nn.Module:
    return model_type(context_feature_dim=93, hidden_dims=(32,), dropout=0.0)


def _anchors(batch_size: int, width: int = 9) -> torch.Tensor:
    probabilities = np.linspace(0.05, 0.85, width, dtype=np.float64)
    tiled = np.tile(probabilities, (batch_size, 1))
    return torch.from_numpy(encode_probability_numpy(tiled))


def _requests(
    rationals: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([row[0] for row in rationals], dtype=torch.int64),
        torch.tensor([row[1] for row in rationals], dtype=torch.int64),
    )


def test_budget_lattice_is_exact_ordered_and_contains_primary_budgets() -> None:
    fractions = tuple(Fraction(*row) for row in BUDGET_KNOT_RATIONALS)

    assert len(fractions) == 9
    assert all(left > right for left, right in zip(fractions, fractions[1:]))
    assert all(
        (item.numerator, item.denominator) == row
        for item, row in zip(fractions, BUDGET_KNOT_RATIONALS)
    )
    assert PRIMARY_BUDGET_KNOT_INDICES == (0, 4, 8)
    assert PRIMARY_BUDGET_RATIONALS == (
        (1, 10_000),
        (1, 100_000),
        (1, 1_000_000),
    )


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_candidate_shapes_parameter_counts_and_anchor_mix(model_type) -> None:
    model = _model(model_type)
    output = model(torch.randn(2, 93), anchor_coordinates=_anchors(2))

    assert output.budget_knot_numerators.shape == (9,)
    assert output.budget_knot_denominators.shape == (9,)
    assert output.budget_knot_positions.shape == (9,)
    assert output.anchor_coordinates.shape == (2, 9)
    assert output.grid_learned_raw_coordinates.shape == (2, 9)
    assert output.grid_coordinates.shape == (2, 9)
    assert output.grid_thresholds.shape == (2, 9)
    assert output.requested_coordinates is None
    assert output.anchor_mix_weight.item() == pytest.approx(
        ANCHOR_MIX_INITIAL_WEIGHT
    )
    expected_count = (
        3306
        if model_type is BudgetConditionedDirectEndpointAwarePixelCalibrator
        else 3339
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_count


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_exact_knot_request_replays_grid_bit_for_bit(model_type) -> None:
    model = _model(model_type)
    anchor = _anchors(3)
    numerators, denominators = _requests(BUDGET_KNOT_RATIONALS)

    output = model(
        torch.randn(3, 93),
        anchor_coordinates=anchor,
        budget_numerators=numerators,
        budget_denominators=denominators,
        requested_anchor_coordinates=anchor.clone(),
    )

    assert torch.equal(
        output.requested_learned_raw_coordinates,
        output.grid_learned_raw_coordinates,
    )
    assert torch.equal(output.requested_raw_coordinates, output.grid_raw_coordinates)
    assert torch.equal(output.requested_coordinates, output.grid_coordinates)
    assert torch.equal(output.requested_thresholds, output.grid_thresholds)


def test_monotone_curve_orders_grid_and_arbitrary_valid_requests() -> None:
    model = _model(BudgetConditionedMonotoneEndpointAwarePixelCalibrator)
    requested = (
        (1, 12_000),
        (1, 25_000),
        (1, 75_000),
        (1, 250_000),
        (1, 750_000),
    )
    numerators, denominators = _requests(requested)
    requested_anchor = _anchors(4, width=len(requested))

    output = model(
        torch.randn(4, 93),
        anchor_coordinates=_anchors(4),
        budget_numerators=numerators,
        budget_denominators=denominators,
        requested_anchor_coordinates=requested_anchor,
    )

    for raw in (
        output.grid_learned_raw_coordinates,
        output.grid_raw_coordinates,
        output.requested_learned_raw_coordinates,
        output.requested_raw_coordinates,
    ):
        assert torch.all(raw[:, 1:] > raw[:, :-1])
    for value in (
        output.grid_coordinates,
        output.grid_thresholds,
        output.requested_coordinates,
        output.requested_thresholds,
    ):
        assert torch.all(value[:, 1:] >= value[:, :-1])
    endpoint = output.requested_coordinates == UPPER_ENDPOINT_COORDINATE
    assert not bool((endpoint[:, :-1] & ~endpoint[:, 1:]).any().item())


def test_arbitrary_request_mixes_its_exact_anchor_not_an_anchor_interpolation() -> None:
    model = _model(BudgetConditionedMonotoneEndpointAwarePixelCalibrator)
    numerators, denominators = _requests(((1, 20_000), (1, 200_000)))
    requested_anchor = torch.from_numpy(
        encode_probability_numpy(
            np.asarray([[0.123456789, 0.987654321]], dtype=np.float64)
        )
    )

    output = model(
        torch.randn(1, 93),
        anchor_coordinates=_anchors(1),
        budget_numerators=numerators,
        budget_denominators=denominators,
        requested_anchor_coordinates=requested_anchor,
    )

    expected = (
        (1.0 - output.anchor_mix_weight) * requested_anchor
        + output.anchor_mix_weight * output.requested_learned_raw_coordinates
    )
    assert torch.equal(output.requested_anchor_coordinates, requested_anchor)
    torch.testing.assert_close(
        output.requested_raw_coordinates, expected, rtol=0.0, atol=0.0
    )


def test_direct_control_can_express_a_nonmonotone_budget_curve() -> None:
    model = _model(BudgetConditionedDirectEndpointAwarePixelCalibrator)
    with torch.no_grad():
        model.encoder[0].weight.zero_()
        model.encoder[0].bias.zero_()
        model.coordinate_head.weight.zero_()
        model.coordinate_head.bias.copy_(
            torch.tensor([4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0])
        )
        model.anchor_mix_logit.fill_(20.0)

    output = model(
        torch.zeros(2, 93),
        anchor_coordinates=torch.zeros((2, 9), dtype=torch.float64),
    )

    assert torch.all(
        output.grid_learned_raw_coordinates[:, 0]
        > output.grid_learned_raw_coordinates[:, 1]
    )
    assert torch.all(output.grid_raw_coordinates[:, 0] > output.grid_raw_coordinates[:, 1])
    assert model.structural_monotonicity is False


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_budget_curve_and_anchor_mix_receive_finite_gradients(model_type) -> None:
    model = _model(model_type)
    numerators, denominators = _requests(((1, 20_000), (1, 200_000)))
    output = model(
        torch.randn(3, 93),
        anchor_coordinates=_anchors(3),
        budget_numerators=numerators,
        budget_denominators=denominators,
        requested_anchor_coordinates=_anchors(3, width=2),
    )

    output.requested_raw_coordinates.sum().backward()

    head = model.coordinate_head if hasattr(model, "coordinate_head") else model.spacing_head
    for parameter in (model.encoder[0].weight, head.weight, model.anchor_mix_logit):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_capability_contract_is_truthful_and_result_free(model_type) -> None:
    model = _model(model_type)
    exported = model.export_config()
    capability = model.capability_contract()

    assert exported["schema_version"] == BUDGET_CONDITIONED_SCHEMA
    assert exported["budget_knot_rationals"] == [
        list(row) for row in BUDGET_KNOT_RATIONALS
    ]
    assert exported["anchor_mix_rule"] == ANCHOR_MIX_RULE
    assert capability["supports_reject"] is False
    assert capability["supports_fallback"] is False
    assert capability["supports_exact_rational_budget_requests"] is True
    assert capability["exact_knot_replay"] == "bitwise_learned_ordinate_replay"
    assert capability["risk_guarantee"] == "empirical_not_certified"
    assert capability["requires_anchor_coordinates"] is True


def test_budget_knot_lattice_cannot_be_silently_changed() -> None:
    changed = list(BUDGET_KNOT_RATIONALS)
    changed[1] = (1, 20_000)
    with pytest.raises(BudgetConditionedCalibratorError, match="lattice is frozen"):
        BudgetConditionedMonotoneEndpointAwarePixelCalibrator(
            93, budget_knot_rationals=changed
        )


@pytest.mark.parametrize(
    ("numerators", "denominators", "message"),
    [
        (
            torch.tensor([2, 1], dtype=torch.int64),
            torch.tensor([20_000, 200_000], dtype=torch.int64),
            "lowest-term",
        ),
        (
            torch.tensor([1, 1], dtype=torch.int64),
            torch.tensor([20_000, 20_000], dtype=torch.int64),
            "strictly",
        ),
        (
            torch.tensor([1, 1], dtype=torch.int64),
            torch.tensor([5_000, 20_000], dtype=torch.int64),
            "trained knot range",
        ),
    ],
)
def test_invalid_rational_requests_are_rejected(
    numerators: torch.Tensor,
    denominators: torch.Tensor,
    message: str,
) -> None:
    model = _model(BudgetConditionedMonotoneEndpointAwarePixelCalibrator)
    with pytest.raises(BudgetConditionedCalibratorError, match=message):
        model(
            torch.zeros(1, 93),
            anchor_coordinates=_anchors(1),
            budget_numerators=numerators,
            budget_denominators=denominators,
            requested_anchor_coordinates=_anchors(1, width=2),
        )


def test_float_budget_requests_and_missing_requested_anchor_are_rejected() -> None:
    model = _model(BudgetConditionedMonotoneEndpointAwarePixelCalibrator)
    with pytest.raises(TypeError, match="int64"):
        model(
            torch.zeros(1, 93),
            anchor_coordinates=_anchors(1),
            budget_numerators=torch.tensor([1.0]),
            budget_denominators=torch.tensor([20_000.0]),
            requested_anchor_coordinates=_anchors(1, width=1),
        )
    with pytest.raises(BudgetConditionedCalibratorError, match="requested_anchor"):
        model(
            torch.zeros(1, 93),
            anchor_coordinates=_anchors(1),
            budget_numerators=torch.tensor([1], dtype=torch.int64),
            budget_denominators=torch.tensor([20_000], dtype=torch.int64),
        )


def test_noncanonical_or_decreasing_requested_anchor_is_rejected() -> None:
    model = _model(BudgetConditionedMonotoneEndpointAwarePixelCalibrator)
    numerators, denominators = _requests(((1, 20_000), (1, 200_000)))
    noncanonical = _anchors(1, width=2)
    noncanonical[0, 1] = MAX_INTERIOR_COORDINATE + 0.25
    with pytest.raises(BudgetConditionedCalibratorError, match="noncanonical"):
        model(
            torch.zeros(1, 93),
            anchor_coordinates=_anchors(1),
            budget_numerators=numerators,
            budget_denominators=denominators,
            requested_anchor_coordinates=noncanonical,
        )

    decreasing = _anchors(1, width=2).flip(1)
    with pytest.raises(BudgetConditionedCalibratorError, match="nondecreasing"):
        model(
            torch.zeros(1, 93),
            anchor_coordinates=_anchors(1),
            budget_numerators=numerators,
            budget_denominators=denominators,
            requested_anchor_coordinates=decreasing,
        )


def test_float64_colliding_rationals_are_rejected_as_invalid_curve_coordinates() -> None:
    model = _model(BudgetConditionedMonotoneEndpointAwarePixelCalibrator)
    numerator = 90_000_000_000_000
    numerators = torch.tensor([numerator, numerator], dtype=torch.int64)
    denominators = torch.tensor(
        [100_000 * numerator - 1, 100_000 * numerator + 1],
        dtype=torch.int64,
    )
    assert Fraction(
        int(numerators[0]), int(denominators[0])
    ) > Fraction(int(numerators[1]), int(denominators[1]))

    with pytest.raises(BudgetConditionedCalibratorError, match="distinguishable"):
        model(
            torch.zeros(1, 93),
            anchor_coordinates=_anchors(1),
            budget_numerators=numerators,
            budget_denominators=denominators,
            requested_anchor_coordinates=_anchors(1, width=2),
        )

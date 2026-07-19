from __future__ import annotations

import numpy as np
import pytest
import torch

from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from model.budget_conditioned_residual_transport_calibrator import (
    RESIDUAL_TRANSPORT_MONOTONE_RULE,
    RESIDUAL_TRANSPORT_SCHEMA,
    BudgetConditionedDirectResidualTransportCalibrator,
    BudgetConditionedMonotoneNoTargetAnchorCalibrator,
    BudgetConditionedMonotoneResidualTransportCalibrator,
    ResidualTransportCalibratorError,
)
from model.endpoint_aware_pixel_calibrator import ANCHOR_MIX_INITIAL_WEIGHT
from model.endpoint_aware_threshold import (
    UPPER_ENDPOINT_COORDINATE,
    encode_probability_numpy,
)


MODEL_TYPES = (
    BudgetConditionedDirectResidualTransportCalibrator,
    BudgetConditionedMonotoneResidualTransportCalibrator,
)


def _model(model_type: type[torch.nn.Module]) -> torch.nn.Module:
    return model_type(context_feature_dim=93, hidden_dims=(32,), dropout=0.0)


def _anchors(batch_size: int, width: int = 9) -> torch.Tensor:
    values = np.tile(np.linspace(0.05, 0.85, width), (batch_size, 1))
    return torch.from_numpy(encode_probability_numpy(values))


def _requests(
    rationals: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([row[0] for row in rationals], dtype=torch.int64),
        torch.tensor([row[1] for row in rationals], dtype=torch.int64),
    )


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_capacity_matched_shapes_and_initial_correction_strength(model_type) -> None:
    model = _model(model_type)
    output = model(torch.randn(2, 93), anchor_coordinates=_anchors(2))

    assert sum(parameter.numel() for parameter in model.parameters()) == 3339
    assert output.grid_residual.shape == (2, 9)
    assert output.grid_anchor_latent.shape == (2, 9)
    assert output.grid_transport_latent.shape == (2, 9)
    assert output.grid_coordinates.shape == (2, 9)
    assert output.anchor_slope.shape == (2, 1)
    assert output.correction_strength.item() == pytest.approx(
        ANCHOR_MIX_INITIAL_WEIGHT
    )
    assert model.supports_reject is False


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_exact_rational_knot_request_replays_grid_bit_for_bit(model_type) -> None:
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

    assert torch.equal(output.requested_residual, output.grid_residual)
    assert torch.equal(
        output.requested_transport_latent, output.grid_transport_latent
    )
    assert torch.equal(output.requested_raw_coordinates, output.grid_raw_coordinates)
    assert torch.equal(output.requested_coordinates, output.grid_coordinates)
    assert torch.equal(output.requested_thresholds, output.grid_thresholds)


def test_monotone_transport_orders_grid_and_arbitrary_rational_queries() -> None:
    model = _model(BudgetConditionedMonotoneResidualTransportCalibrator)
    requested = (
        (1, 12_000),
        (1, 25_000),
        (1, 75_000),
        (1, 250_000),
        (1, 750_000),
    )
    numerators, denominators = _requests(requested)
    output = model(
        torch.randn(4, 93),
        anchor_coordinates=_anchors(4),
        budget_numerators=numerators,
        budget_denominators=denominators,
        requested_anchor_coordinates=_anchors(4, len(requested)),
    )

    for value in (
        output.grid_residual,
        output.grid_transport_latent,
        output.grid_raw_coordinates,
        output.requested_residual,
        output.requested_transport_latent,
        output.requested_raw_coordinates,
    ):
        assert torch.all(value[:, 1:] >= value[:, :-1])
    for value in (
        output.grid_coordinates,
        output.grid_thresholds,
        output.requested_coordinates,
        output.requested_thresholds,
    ):
        assert torch.all(value[:, 1:] >= value[:, :-1])
    endpoint = output.requested_coordinates == UPPER_ENDPOINT_COORDINATE
    assert not bool((endpoint[:, :-1] & ~endpoint[:, 1:]).any().item())


def test_monotone_transport_remains_valid_under_float64_sigmoid_saturation() -> None:
    model = _model(BudgetConditionedMonotoneResidualTransportCalibrator)
    output = model(
        torch.full((2, 93), 1.0e6, dtype=torch.float32),
        anchor_coordinates=_anchors(2),
    )
    assert torch.all(
        output.grid_transport_latent[:, 1:]
        >= output.grid_transport_latent[:, :-1]
    )
    assert torch.all(
        output.grid_raw_coordinates[:, 1:]
        >= output.grid_raw_coordinates[:, :-1]
    )
    assert torch.all(
        output.grid_thresholds[:, 1:] >= output.grid_thresholds[:, :-1]
    )


def test_no_target_anchor_ablation_is_capacity_matched_and_has_no_anchor_api() -> None:
    anchored = _model(BudgetConditionedMonotoneResidualTransportCalibrator)
    anchor_free = _model(BudgetConditionedMonotoneNoTargetAnchorCalibrator)
    assert sum(parameter.numel() for parameter in anchored.parameters()) == 3339
    assert sum(parameter.numel() for parameter in anchor_free.parameters()) == 3339
    assert anchor_free.capability_contract()["requires_anchor_coordinates"] is False
    assert anchor_free.capability_contract()["target_anchor_accessed"] is False
    numerators, denominators = _requests(((1, 25_000), (1, 250_000)))
    output = anchor_free(
        torch.randn(3, 93),
        budget_numerators=numerators,
        budget_denominators=denominators,
    )
    assert torch.all(
        output.grid_coordinates[:, 1:] >= output.grid_coordinates[:, :-1]
    )
    assert torch.all(
        output.requested_coordinates[:, 1:]
        >= output.requested_coordinates[:, :-1]
    )
    with pytest.raises(TypeError, match="anchor_coordinates"):
        anchor_free(
            torch.randn(1, 93),
            anchor_coordinates=_anchors(1),
        )


def test_same_budget_requested_anchor_is_used_directly_not_interpolated() -> None:
    model = _model(BudgetConditionedMonotoneResidualTransportCalibrator)
    numerators, denominators = _requests(((1, 20_000), (1, 200_000)))
    first = torch.from_numpy(
        encode_probability_numpy(
            np.asarray([[0.20, 0.80]], dtype=np.float64)
        )
    )
    second = torch.from_numpy(
        encode_probability_numpy(
            np.asarray([[0.30, 0.80]], dtype=np.float64)
        )
    )
    features = torch.randn(1, 93)
    grid_anchor = _anchors(1)

    left = model(
        features,
        anchor_coordinates=grid_anchor,
        budget_numerators=numerators,
        budget_denominators=denominators,
        requested_anchor_coordinates=first,
    )
    right = model(
        features,
        anchor_coordinates=grid_anchor,
        budget_numerators=numerators,
        budget_denominators=denominators,
        requested_anchor_coordinates=second,
    )

    assert torch.equal(left.requested_anchor_coordinates, first)
    assert torch.equal(right.requested_anchor_coordinates, second)
    assert left.requested_raw_coordinates[0, 0] < right.requested_raw_coordinates[0, 0]
    assert torch.equal(
        left.requested_raw_coordinates[:, 1:],
        right.requested_raw_coordinates[:, 1:],
    )


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_transport_recovers_analytic_anchor_as_correction_strength_vanishes(
    model_type,
) -> None:
    model = _model(model_type)
    with torch.no_grad():
        model.correction_strength_logit.fill_(-30.0)
        model.transport_head.weight.zero_()
        model.transport_head.bias.zero_()
    anchor = _anchors(2)

    output = model(torch.zeros(2, 93), anchor_coordinates=anchor)

    torch.testing.assert_close(
        output.grid_raw_coordinates, anchor, rtol=0.0, atol=2e-11
    )


def test_direct_control_can_express_nonmonotone_residual_transport() -> None:
    model = _model(BudgetConditionedDirectResidualTransportCalibrator)
    with torch.no_grad():
        model.encoder[0].weight.zero_()
        model.encoder[0].bias.zero_()
        model.transport_head.weight.zero_()
        model.transport_head.bias.zero_()
        model.transport_head.bias[2] = -20.0
        model.correction_strength_logit.zero_()
    close_anchor = torch.from_numpy(
        encode_probability_numpy(
            np.tile(np.linspace(0.45, 0.53, 9), (2, 1))
        )
    )

    output = model(torch.zeros(2, 93), anchor_coordinates=close_anchor)

    assert torch.all(output.grid_residual[:, 0] > output.grid_residual[:, 1])
    assert torch.all(
        output.grid_raw_coordinates[:, 0] > output.grid_raw_coordinates[:, 1]
    )
    assert model.structural_monotonicity is False


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_transport_head_encoder_and_strength_receive_finite_gradients(model_type) -> None:
    model = _model(model_type)
    numerators, denominators = _requests(((1, 20_000), (1, 200_000)))
    output = model(
        torch.randn(3, 93),
        anchor_coordinates=_anchors(3),
        budget_numerators=numerators,
        budget_denominators=denominators,
        requested_anchor_coordinates=_anchors(3, 2),
    )

    output.requested_raw_coordinates.sum().backward()

    for parameter in (
        model.encoder[0].weight,
        model.transport_head.weight,
        model.correction_strength_logit,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


def test_monotone_capability_contract_binds_safe_novelty_delta() -> None:
    model = _model(BudgetConditionedMonotoneResidualTransportCalibrator)
    exported = model.export_config()
    capability = model.capability_contract()

    assert exported["schema_version"] == RESIDUAL_TRANSPORT_SCHEMA
    assert capability["supports_reject"] is False
    assert capability["supports_fallback"] is False
    assert capability["structural_monotonicity"] is True
    assert capability["residual_monotonicity_rule"] == RESIDUAL_TRANSPORT_MONOTONE_RULE
    assert capability["source_of_learned_correction"] == (
        "source_oof_cyclic_training_only"
    )
    assert "same_requested_rational_budget" in capability[
        "requested_anchor_semantics"
    ]
    assert capability["risk_guarantee"] == "empirical_not_certified"


def test_transport_inputs_fail_closed_on_float_budgets_and_unordered_anchor() -> None:
    model = _model(BudgetConditionedMonotoneResidualTransportCalibrator)
    anchor = _anchors(1)
    wrong = anchor.clone()
    wrong[:, 3] = wrong[:, 2] - 0.01

    with pytest.raises(ResidualTransportCalibratorError, match="nondecreasing"):
        model(torch.randn(1, 93), anchor_coordinates=wrong)
    with pytest.raises(TypeError, match="int64 numerator"):
        model(
            torch.randn(1, 93),
            anchor_coordinates=anchor,
            budget_numerators=torch.tensor([1.0]),
            budget_denominators=torch.tensor([20_000.0]),
            requested_anchor_coordinates=_anchors(1, 1),
        )
    numerators, denominators = _requests(((1, 20_000),))
    with pytest.raises(ResidualTransportCalibratorError, match="requested_anchor"):
        model(
            torch.randn(1, 93),
            anchor_coordinates=anchor,
            budget_numerators=numerators,
            budget_denominators=denominators,
        )

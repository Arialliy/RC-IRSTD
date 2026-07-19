from __future__ import annotations

import numpy as np
import pytest
import torch

from model.endpoint_aware_pixel_calibrator import (
    ANCHOR_COORDINATE_CONTRACT,
    ANCHOR_MIX_INITIAL_WEIGHT,
    ANCHOR_MIX_PARAMETERIZATION,
    ANCHOR_MIX_RULE,
    T4_ANCHOR_SOURCE,
    DirectEndpointAwarePixelCalibrator,
    EndpointAwareCalibratorOutput,
    MonotoneEndpointAwarePixelCalibrator,
)
from model.endpoint_aware_threshold import (
    MAX_INTERIOR_COORDINATE,
    RAW_COORDINATE_MAX,
    RAW_COORDINATE_MIN,
    UPPER_ENDPOINT_COORDINATE,
    encode_probability_numpy,
)


PIXEL_BUDGETS = (1e-4, 1e-5, 1e-6)
MODEL_TYPES = (
    DirectEndpointAwarePixelCalibrator,
    MonotoneEndpointAwarePixelCalibrator,
)


def _anchors(
    batch_size: int,
    probabilities: tuple[float, float, float] = (0.1, 0.5, 1.0),
) -> torch.Tensor:
    tiled = np.tile(np.asarray(probabilities, dtype=np.float64), (batch_size, 1))
    return torch.from_numpy(encode_probability_numpy(tiled))


def _model(model_type: type[torch.nn.Module]) -> torch.nn.Module:
    return model_type(
        context_feature_dim=93,
        pixel_budget_grid=PIXEL_BUDGETS,
        hidden_dims=(32,),
        dropout=0.0,
    )


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_anchor_aware_output_shapes_and_transcript_fields(model_type) -> None:
    model = _model(model_type)
    features = torch.randn(2, 93)
    anchors = _anchors(2)

    output = model(features, anchor_coordinates=anchors)

    assert output.pixel_budget_grid.shape == (3,)
    assert output.anchor_coordinates.shape == (2, 3)
    assert output.anchor_mix_weight.shape == torch.Size([])
    assert output.grid_learned_raw_coordinates.shape == (2, 3)
    assert output.grid_raw_coordinates.shape == (2, 3)
    assert output.grid_coordinates.shape == (2, 3)
    assert output.grid_thresholds.shape == (2, 3)
    assert output.requested_coordinates is None
    assert torch.equal(output.anchor_coordinates, anchors)
    assert output.anchor_mix_weight.item() == pytest.approx(
        ANCHOR_MIX_INITIAL_WEIGHT
    )
    expected_raw = (
        (1.0 - output.anchor_mix_weight) * output.anchor_coordinates
        + output.anchor_mix_weight * output.grid_learned_raw_coordinates
    )
    torch.testing.assert_close(
        output.grid_raw_coordinates, expected_raw, rtol=0.0, atol=0.0
    )
    assert torch.all(output.grid_raw_coordinates > RAW_COORDINATE_MIN)
    assert torch.all(output.grid_raw_coordinates < RAW_COORDINATE_MAX)

    assert output.grid_logits is output.grid_coordinates
    assert output.requested_logits is output.requested_coordinates
    assert EndpointAwareCalibratorOutput.grid_logits.fset is None
    assert EndpointAwareCalibratorOutput.requested_logits.fset is None


def test_anchor_coordinates_are_a_required_keyword() -> None:
    model = _model(DirectEndpointAwarePixelCalibrator)
    with pytest.raises(TypeError, match="anchor_coordinates"):
        model(torch.zeros(2, 93))


@pytest.mark.parametrize(
    ("case", "error_type", "message"),
    [
        ("not_tensor", TypeError, "torch.Tensor"),
        ("wrong_dtype", TypeError, "float64"),
        ("wrong_shape", ValueError, "shape"),
        ("nan", ValueError, "finite"),
        ("noncanonical", ValueError, "neither an interior"),
        ("decreasing", ValueError, "nondecreasing"),
    ],
)
def test_invalid_anchor_coordinates_are_rejected(
    case: str, error_type: type[Exception], message: str
) -> None:
    model = _model(DirectEndpointAwarePixelCalibrator)
    anchor: object = _anchors(2)
    if case == "not_tensor":
        anchor = [[0.1, 0.5, 1.0], [0.1, 0.5, 1.0]]
    elif case == "wrong_dtype":
        anchor = _anchors(2).to(torch.float32)
    elif case == "wrong_shape":
        anchor = _anchors(2)[:, :2]
    elif case == "nan":
        anchor = _anchors(2)
        anchor[0, 1] = float("nan")
    elif case == "noncanonical":
        anchor = _anchors(2)
        anchor[0, 1] = MAX_INTERIOR_COORDINATE + 0.25
    elif case == "decreasing":
        anchor = _anchors(2)
        anchor[0, :2] = torch.tensor([0.2, 0.1], dtype=torch.float64)

    with pytest.raises(error_type, match=message):
        model(torch.zeros(2, 93), anchor_coordinates=anchor)


def test_rc5_parameter_counts_and_anchor_contracts_are_frozen() -> None:
    direct = _model(DirectEndpointAwarePixelCalibrator)
    monotone = _model(MonotoneEndpointAwarePixelCalibrator)

    assert sum(parameter.numel() for parameter in direct.parameters()) == 3108
    assert sum(parameter.numel() for parameter in monotone.parameters()) == 3141
    assert direct.anchor_mix_logit.numel() == monotone.anchor_mix_logit.numel() == 1

    expected_anchor_contract = {
        "anchor_source": T4_ANCHOR_SOURCE,
        "anchor_coordinate_contract": ANCHOR_COORDINATE_CONTRACT,
        "anchor_mix_rule": ANCHOR_MIX_RULE,
        "anchor_mix_parameterization": ANCHOR_MIX_PARAMETERIZATION,
        "anchor_mix_initial_weight": ANCHOR_MIX_INITIAL_WEIGHT,
    }
    for model in (direct, monotone):
        exported = model.export_config()
        capability = model.capability_contract()
        assert {
            key: exported[key] for key in expected_anchor_contract
        } == expected_anchor_contract
        assert {
            key: capability[key] for key in expected_anchor_contract
        } == expected_anchor_contract
        assert capability["requires_anchor_coordinates"] is True
        assert "T4_anchor_mixed" in capability["training_objective"]

    with pytest.raises(ValueError, match="exactly three"):
        DirectEndpointAwarePixelCalibrator(93, PIXEL_BUDGETS[:2])


def test_t6_direct_head_can_remain_nonmonotone_after_anchor_mixing() -> None:
    model = _model(DirectEndpointAwarePixelCalibrator)
    with torch.no_grad():
        model.encoder[0].weight.zero_()
        model.encoder[0].bias.zero_()
        model.coordinate_head.weight.zero_()
        model.coordinate_head.bias.copy_(torch.tensor([4.0, -4.0, 0.0]))
        model.anchor_mix_logit.fill_(20.0)

    output = model(
        torch.zeros(2, 93),
        anchor_coordinates=torch.zeros((2, 3), dtype=torch.float64),
    )

    assert torch.all(
        output.grid_learned_raw_coordinates[:, 0]
        > output.grid_learned_raw_coordinates[:, 1]
    )
    assert torch.all(
        output.grid_raw_coordinates[:, 0] > output.grid_raw_coordinates[:, 1]
    )
    assert torch.all(output.grid_thresholds[:, 0] > output.grid_thresholds[:, 1])
    assert model.structural_monotonicity is False


@pytest.mark.parametrize("_method", ("T7", "T8"), ids=("T7", "T8"))
def test_t7_t8_preserve_strict_raw_order_and_endpoint_suffix(_method: str) -> None:
    model = _model(MonotoneEndpointAwarePixelCalibrator)
    with torch.no_grad():
        model.anchor_mix_logit.fill_(-20.0)

    output = model(torch.randn(4, 93), anchor_coordinates=_anchors(4))

    assert torch.all(
        output.grid_learned_raw_coordinates[:, 1:]
        > output.grid_learned_raw_coordinates[:, :-1]
    )
    assert torch.all(
        output.grid_raw_coordinates[:, 1:] > output.grid_raw_coordinates[:, :-1]
    )
    assert torch.all(
        output.grid_coordinates[:, 1:] >= output.grid_coordinates[:, :-1]
    )
    assert torch.all(output.grid_thresholds[:, 1:] >= output.grid_thresholds[:, :-1])
    endpoint = output.grid_coordinates == UPPER_ENDPOINT_COORDINATE
    assert torch.all(endpoint[:, -1])
    assert not bool((endpoint[:, :-1] & ~endpoint[:, 1:]).any().item())


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_anchor_mix_logit_receives_a_finite_gradient(model_type) -> None:
    model = _model(model_type)
    output = model(
        torch.randn(3, 93),
        anchor_coordinates=_anchors(3, probabilities=(0.1, 0.2, 0.3)),
    )

    output.grid_raw_coordinates.sum().backward()

    assert model.anchor_mix_logit.grad is not None
    assert torch.isfinite(model.anchor_mix_logit.grad)
    assert model.anchor_mix_logit.grad.abs().item() > 0.0


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_alpha_near_zero_keeps_final_raw_close_to_t4_anchor(model_type) -> None:
    model = _model(model_type)
    with torch.no_grad():
        model.anchor_mix_logit.fill_(-30.0)
    anchors = _anchors(2, probabilities=(0.1, 0.25, 0.5))

    output = model(torch.randn(2, 93), anchor_coordinates=anchors)

    assert output.anchor_mix_weight.item() < 1e-12
    torch.testing.assert_close(
        output.grid_raw_coordinates, anchors, rtol=0.0, atol=5e-12
    )

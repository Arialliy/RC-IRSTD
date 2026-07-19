from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from model.endpoint_aware_pixel_calibrator import (
    MonotoneEndpointAwarePixelCalibrator,
)
from model.endpoint_aware_pixel_calibrator_ablation import (
    T8_NO_ANCHOR_ABLATION_ROLE,
    T8_NO_ANCHOR_EXPECTED_TRAINABLE_PARAMETERS,
    T8_NO_ANCHOR_METHOD_ID,
    T8_NO_ANCHOR_MODEL_ID,
    LearnedOnlyEndpointAwareCalibratorOutput,
    LearnedOnlyMonotoneEndpointAwarePixelCalibrator,
)
from model.endpoint_aware_threshold import (
    UPPER_ENDPOINT_COORDINATE,
    canonicalize_raw_torch,
    decode_coordinate_torch,
    representation_contract,
)
from rc.stage2_calibrator_checkpoint_v7 import (
    METHODS,
    Stage2CalibratorCheckpointV7Error,
    make_calibrator_checkpoint_v7,
)


def _model() -> LearnedOnlyMonotoneEndpointAwarePixelCalibrator:
    return LearnedOnlyMonotoneEndpointAwarePixelCalibrator(
        context_feature_dim=93,
        pixel_budget_grid=[1e-4, 1e-5, 1e-6],
        hidden_dims=[32],
        dropout=0.1,
        minimum_raw_coordinate_gap=0.001,
    )


def test_t8_no_anchor_has_exact_learned_branch_and_no_anchor_state() -> None:
    model = _model()
    assert model.encoder[0].in_features == 93
    assert model.encoder[0].out_features == 32
    assert model.spacing_head.in_features == 32
    assert model.spacing_head.out_features == 4
    assert T8_NO_ANCHOR_EXPECTED_TRAINABLE_PARAMETERS == 3140
    assert sum(parameter.numel() for parameter in model.parameters()) == 3140
    assert model._parameters["anchor_mix_logit"] is None
    assert "anchor_mix_logit" not in dict(model.named_parameters())
    assert all("anchor" not in name for name in model.state_dict())

    main = MonotoneEndpointAwarePixelCalibrator(
        context_feature_dim=93,
        pixel_budget_grid=[1e-4, 1e-5, 1e-6],
        hidden_dims=[32],
        dropout=0.1,
        minimum_raw_coordinate_gap=0.001,
    )
    assert sum(parameter.numel() for parameter in main.parameters()) == 3141
    assert "anchor_mix_logit" in main.state_dict()
    with pytest.raises(ValueError, match="93->32->4"):
        LearnedOnlyMonotoneEndpointAwarePixelCalibrator(
            context_feature_dim=92,
            pixel_budget_grid=[1e-4, 1e-5, 1e-6],
        )


def test_forward_surface_cannot_accept_anchor_or_runtime_method_flags() -> None:
    model = _model().eval()
    signature = inspect.signature(model.forward)
    assert tuple(signature.parameters) == ("context_features", "pixel_budgets")
    features = torch.zeros(2, 93)
    anchor = torch.zeros(2, 3, dtype=torch.float64)
    with pytest.raises(TypeError, match="anchor_coordinates"):
        model(features, anchor_coordinates=anchor)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="use_anchor"):
        model(features, use_anchor=False)  # type: ignore[call-arg]


def test_hard_eatc_forward_is_exact_ordered_and_anchor_free() -> None:
    torch.manual_seed(12)
    model = _model().eval()
    output = model(torch.randn(5, 93))
    assert type(output) is LearnedOnlyEndpointAwareCalibratorOutput
    assert output.method_id == T8_NO_ANCHOR_METHOD_ID
    assert output.ablation_role == T8_NO_ANCHOR_ABLATION_ROLE
    assert output.claim_bearing is False
    assert not hasattr(output, "anchor_coordinates")
    assert not hasattr(output, "anchor_mix_weight")
    assert not hasattr(output, "grid_learned_raw_coordinates")
    assert output.grid_raw_coordinates.shape == (5, 3)
    assert output.grid_raw_coordinates.dtype == torch.float64
    assert bool(
        (output.grid_raw_coordinates[:, 1:] > output.grid_raw_coordinates[:, :-1])
        .all()
        .item()
    )
    assert torch.equal(
        output.grid_coordinates,
        canonicalize_raw_torch(output.grid_raw_coordinates),
    )
    assert torch.equal(
        output.grid_thresholds,
        decode_coordinate_torch(output.grid_coordinates),
    )
    assert bool(
        (output.grid_coordinates[:, 1:] >= output.grid_coordinates[:, :-1])
        .all()
        .item()
    )
    assert bool(
        (output.grid_thresholds[:, 1:] >= output.grid_thresholds[:, :-1])
        .all()
        .item()
    )
    endpoints = output.grid_coordinates == UPPER_ENDPOINT_COORDINATE
    assert not bool((endpoints[:, :-1] & ~endpoints[:, 1:]).any().item())
    assert output.grid_logits is output.grid_coordinates


def test_requested_budgets_use_same_raw_log_interpolation_and_hard_decode() -> None:
    model = _model().eval()
    features = torch.zeros(2, 93)
    requests = torch.tensor([1e-4, 3e-5, 1e-5, 1e-6], dtype=torch.float64)
    output = model(features, pixel_budgets=requests)
    assert output.requested_pixel_budgets is not None
    assert output.requested_raw_coordinates is not None
    assert output.requested_coordinates is not None
    assert output.requested_thresholds is not None
    assert output.requested_pixel_budgets.shape == (2, 4)
    assert torch.equal(
        output.requested_coordinates,
        canonicalize_raw_torch(output.requested_raw_coordinates),
    )
    assert torch.equal(
        output.requested_thresholds,
        decode_coordinate_torch(output.requested_coordinates),
    )
    assert output.requested_logits is output.requested_coordinates
    torch.testing.assert_close(
        output.requested_raw_coordinates[:, [0, 2, 3]],
        output.grid_raw_coordinates,
        rtol=0.0,
        atol=1e-14,
    )
    with pytest.raises(ValueError, match="inside the trained grid"):
        model(features, pixel_budgets=torch.tensor([1e-7]))


def test_gradients_cover_all_3140_learned_only_parameters() -> None:
    torch.manual_seed(3)
    model = _model().train()
    output = model(torch.randn(4, 93))
    output.grid_raw_coordinates.sum().backward()
    named = dict(model.named_parameters())
    assert sum(parameter.numel() for parameter in named.values()) == 3140
    assert named
    assert all(parameter.grad is not None for parameter in named.values())
    assert all(bool(torch.isfinite(parameter.grad).all().item()) for parameter in named.values())


def test_export_and_capability_bind_non_claim_bearing_ablation_identity() -> None:
    model = _model()
    exported = model.export_config()
    assert exported == {
        "method_id": T8_NO_ANCHOR_METHOD_ID,
        "model_id": T8_NO_ANCHOR_MODEL_ID,
        "ablation_role": T8_NO_ANCHOR_ABLATION_ROLE,
        "claim_bearing": False,
        "expected_trainable_parameters": 3140,
        "context_feature_dim": 93,
        "pixel_budget_grid": [1e-4, 1e-5, 1e-6],
        "hidden_dims": [32],
        "dropout": 0.1,
        "minimum_raw_coordinate_gap": 0.001,
        "raw_coordinate_min_hex": model.raw_coordinate_min.hex(),
        "raw_coordinate_max_hex": model.raw_coordinate_max.hex(),
        "threshold_representation_schema": model.threshold_representation_schema,
    }
    assert all("anchor" not in key for key in exported)

    capability = model.capability_contract()
    assert capability["method_id"] == "T8_NO_ANCHOR"
    assert capability["model_id"] == T8_NO_ANCHOR_MODEL_ID
    assert capability["ablation_role"] == "risk_aligned_ablation_only"
    assert capability["claim_bearing"] is False
    assert capability["expected_trainable_parameters"] == 3140
    assert capability["uses_analytic_anchor"] is False
    assert capability["requires_anchor_coordinates"] is False
    assert capability["runtime_anchor_toggle_supported"] is False
    assert capability["method_identity_selected_by_class_not_runtime_flag"] is True
    assert capability["checkpoint_v7_supported"] is False
    assert capability["threshold_representation"] == representation_contract()


def test_checkpoint_v7_allowlist_remains_main_methods_only() -> None:
    assert METHODS == ("T6", "T7", "T8")
    with pytest.raises(Stage2CalibratorCheckpointV7Error, match="unsupported"):
        make_calibrator_checkpoint_v7(
            method=T8_NO_ANCHOR_METHOD_ID,
            model=_model(),
            standardizer_mean=np.zeros(93),
            standardizer_scale=np.ones(93),
            training_contract_sha256="0" * 64,
        )

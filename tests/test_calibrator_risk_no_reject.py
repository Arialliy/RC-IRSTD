from __future__ import annotations

import pytest
import torch

from losses.calibrator_risk import (
    FULL_BACKGROUND,
    WEIGHTED_STRATIFIED_BACKGROUND,
    calibrator_risk_capability_contract,
    curve_query_risk_aligned_calibrator_loss,
    log10_budget_curve_smoothness,
    query_risk_aligned_calibrator_loss,
    surrogate_query_pixel_false_alarm_rate,
)
from model.monotone_pixel_calibrator import (
    PIXEL_BUDGET_ONLY_SCOPE,
    PIXEL_RISK_NO_REJECT_SCOPE,
    MonotoneNoRejectPixelRiskCalibrator,
    MonotonePixelBudgetCalibrator,
    NoRejectPixelRiskCalibratorOutput,
)


def _new_model() -> MonotoneNoRejectPixelRiskCalibrator:
    torch.manual_seed(23)
    return MonotoneNoRejectPixelRiskCalibrator(
        context_feature_dim=4,
        pixel_budget_grid=(1e-4, 1e-5, 1e-6),
        hidden_dims=(8,),
        dropout=0.0,
    )


def test_legacy_budget_calibrator_api_and_contract_are_unchanged() -> None:
    model = MonotonePixelBudgetCalibrator(
        context_feature_dim=3,
        pixel_budget_grid=(1e-4, 1e-5),
        hidden_dims=(4,),
        dropout=0.0,
    )
    output = model(torch.zeros(2, 3))

    assert output.grid_logits.shape == (2, 2)
    assert model.capability_contract() == {
        "budget_scope": PIXEL_BUDGET_ONLY_SCOPE,
        "supports_component_budget": False,
        "supports_reject": False,
        "training_pipeline_integrated": False,
    }


def test_primary_stage2_model_emits_complete_no_reject_float64_curve() -> None:
    model = _new_model()
    output = model(torch.randn(2, 4))

    assert isinstance(output, NoRejectPixelRiskCalibratorOutput)
    assert output.grid_logits.shape == (2, 3)
    assert output.grid_logits.dtype == torch.float64
    assert output.grid_thresholds.dtype == torch.float64
    assert torch.all(output.grid_logits[:, 1:] > output.grid_logits[:, :-1])
    assert torch.all(output.grid_thresholds[:, 1:] > output.grid_thresholds[:, :-1])
    assert not hasattr(output, "grid_reject_logits")
    contract = model.capability_contract()
    assert contract["stage"] == "stage2_final_no_reject"
    assert contract["budget_scope"] == PIXEL_RISK_NO_REJECT_SCOPE
    assert contract["supports_complete_budget_curve"] is True
    assert contract["supports_query_risk_aligned_loss"] is True
    assert contract["supports_reject"] is False
    assert contract["curve_compute_dtype"] == "float64"
    assert contract["training_pipeline_integrated"] is True
    assert MonotoneNoRejectPixelRiskCalibrator(
        **model.export_config()
    ).export_config() == model.export_config()


def test_no_reject_interpolation_is_log10_inside_grid_and_rejects_extrapolation() -> None:
    model = _new_model()
    features = torch.randn(2, 4)
    output = model(
        features,
        pixel_budgets=torch.tensor([1e-4, 10 ** -4.5, 1e-5, 1e-6]),
    )

    assert output.requested_logits is not None
    assert output.requested_logits.shape == (2, 4)
    torch.testing.assert_close(output.requested_logits[:, 0], output.grid_logits[:, 0])
    torch.testing.assert_close(output.requested_logits[:, 2], output.grid_logits[:, 1])
    # 10^-4.5 is the midpoint in log10 space.
    torch.testing.assert_close(
        output.requested_logits[:, 1],
        0.5 * (output.grid_logits[:, 0] + output.grid_logits[:, 1]),
        rtol=1e-5,
        atol=1e-7,
    )
    with pytest.raises(ValueError, match="extrapolation is disabled"):
        model(
            features,
            pixel_budgets=torch.tensor([1e-7], dtype=torch.float64),
        )


def test_full_background_uses_every_valid_pixel_and_exact_chunking() -> None:
    eta = torch.tensor([[0.0, 1.0]], requires_grad=True)
    background = torch.tensor([[-2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 99.0]])
    valid = torch.tensor([[True, True, True, True, True, True, False]])
    total_pixels = torch.tensor([8])

    chunked, population = surrogate_query_pixel_false_alarm_rate(
        eta,
        background,
        total_pixels,
        background_valid=valid,
        exact_chunk_size=2,
    )
    unchunked, _ = surrogate_query_pixel_false_alarm_rate(
        eta,
        background,
        total_pixels,
        background_valid=valid,
        exact_chunk_size=100,
    )

    torch.testing.assert_close(chunked, unchunked)
    torch.testing.assert_close(population, torch.tensor([6.0], dtype=torch.float64))
    chunked.sum().backward()
    assert eta.grad is not None and torch.isfinite(eta.grad).all()
    assert torch.count_nonzero(eta.grad).item() == eta.numel()


def test_weighted_stratified_supervision_matches_covered_full_population() -> None:
    eta = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    total_pixels = torch.tensor([8])
    full, _ = surrogate_query_pixel_false_alarm_rate(
        eta,
        torch.tensor([[-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]]),
        total_pixels,
        background_representation=FULL_BACKGROUND,
    )
    stratified, population = surrogate_query_pixel_false_alarm_rate(
        eta,
        torch.tensor([[-2.0, 2.0, 0.0]]),
        total_pixels,
        background_valid=torch.tensor([[True, True, False]]),
        background_representation=WEIGHTED_STRATIFIED_BACKGROUND,
        background_weights=torch.tensor([[3.0, 3.0, 0.0]]),
    )

    torch.testing.assert_close(stratified, full)
    torch.testing.assert_close(population, torch.tensor([6.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="requires explicit population weights"):
        surrogate_query_pixel_false_alarm_rate(
            eta,
            torch.tensor([[-2.0, 2.0]]),
            total_pixels,
            background_representation=WEIGHTED_STRATIFIED_BACKGROUND,
        )
    with pytest.raises(ValueError, match="forbidden for full supervision"):
        surrogate_query_pixel_false_alarm_rate(
            eta,
            torch.tensor([[-2.0, 2.0]]),
            total_pixels,
            background_weights=torch.ones(1, 2),
        )
    contract = calibrator_risk_capability_contract()
    assert contract["default_background_supervision"] == FULL_BACKGROUND
    assert contract["implicit_uniform_subsample_limit"] is None
    assert contract["background_chunking"] == "exact_all_entries_no_sampling"


def test_complete_query_risk_objective_backpropagates_to_no_reject_model() -> None:
    model = _new_model()
    output = model(torch.randn(2, 4))
    background = torch.tensor(
        [
            [8.0, 7.0, 6.0, 5.0, 4.0],
            [7.5, 6.5, 5.5, 4.5, 3.5],
        ]
    )
    objects = torch.tensor([[9.0, 7.0], [8.5, 6.5]])
    oracle = output.grid_logits.detach() + torch.tensor(
        [[0.2, -0.1, 0.1], [-0.2, 0.1, -0.1]],
        dtype=torch.float64,
    )
    loss = query_risk_aligned_calibrator_loss(
        output.grid_logits,
        model.pixel_budget_grid,
        oracle,
        background,
        torch.tensor([8, 8]),
        objects,
        lambda_violation=1.0,
        lambda_utility=1.0,
        lambda_oracle_logit=0.1,
        lambda_curve_smoothness=0.01,
        exact_background_chunk_size=2,
    )

    assert loss.surrogate_pixel_false_alarm_rate.shape == (2, 3)
    assert loss.surrogate_detection_probability.shape == (2, 3)
    assert loss.total.dtype == torch.float64
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert model.spacing_head.weight.grad is not None
    assert torch.isfinite(model.spacing_head.weight.grad).all()
    assert torch.count_nonzero(model.spacing_head.weight.grad).item() > 0


def test_curve_smoothness_is_zero_for_log_budget_linear_logits() -> None:
    eta = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    smoothness = log10_budget_curve_smoothness(
        eta,
        torch.tensor([1e-3, 1e-4, 1e-5]),
    )
    torch.testing.assert_close(
        smoothness,
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    smoothness.backward()
    assert eta.grad is not None and torch.isfinite(eta.grad).all()


def test_verified_curve_loss_interpolates_padding_and_penalises_only_partial_coverage() -> None:
    eta = torch.tensor(
        [[-1.0, 1.0], [-3.0, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    curve_logits = torch.tensor(
        [[-2.0, 0.0, 2.0, 99.0], [-2.0, 0.0, 2.0, -99.0]],
        dtype=torch.float64,
    )
    curve_risk = torch.tensor(
        [[1.0, 0.5, 0.0, 99.0], [1.0, 0.5, 0.0, -99.0]],
        dtype=torch.float64,
    )
    curve_pd = torch.tensor(
        [[1.0, 0.8, 0.2, 99.0], [1.0, 0.8, 0.2, -99.0]],
        dtype=torch.float64,
    )
    valid = torch.tensor(
        [[True, True, True, False], [True, True, True, False]]
    )
    output = curve_query_risk_aligned_calibrator_loss(
        eta,
        torch.tensor([0.8, 0.2], dtype=torch.float64),
        torch.zeros_like(eta),
        curve_logits,
        curve_risk,
        curve_pd,
        valid,
        exact_lower_bound=torch.tensor([0.0, -2.0], dtype=torch.float64),
        global_exact=torch.tensor([False, True]),
        lambda_violation=1.0,
        lambda_utility=1.0,
        lambda_oracle_logit=0.1,
        lambda_curve_smoothness=0.0,
        lambda_coverage=1.0,
    )

    # Partial row eta=-1 is below its verified lower bound 0 and is evaluated
    # at 0 without extrapolation.  Its eta=1 point is linearly interpolated.
    torch.testing.assert_close(
        output.interpolation_logits[0],
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        output.surrogate_pixel_false_alarm_rate[0],
        torch.tensor([0.5, 0.25], dtype=torch.float64),
    )
    torch.testing.assert_close(
        output.surrogate_detection_probability[0],
        torch.tensor([0.8, 0.5], dtype=torch.float64),
    )
    # The globally exact row can clamp both curve endpoints with no coverage
    # penalty, even when predictions fall outside the stored coordinate range.
    torch.testing.assert_close(
        output.interpolation_logits[1],
        torch.tensor([-2.0, 2.0], dtype=torch.float64),
    )
    assert output.interpolation_clamped_low[1, 0]
    assert output.interpolation_clamped_high[1, 1]
    assert output.coverage_shortfall_logits[0, 0] == 1.0
    assert torch.all(output.coverage_shortfall_logits[1] == 0.0)
    assert output.coverage_penalty > 0.0
    output.total.backward()
    assert eta.grad is not None and torch.isfinite(eta.grad).all()
    # Coverage supplies a gradient below the non-global exact lower bound.
    assert eta.grad[0, 0] < 0.0


def test_verified_global_exact_curve_has_zero_coverage_penalty() -> None:
    eta = torch.tensor([[-10.0, 10.0]], requires_grad=True)
    output = curve_query_risk_aligned_calibrator_loss(
        eta,
        torch.tensor([0.5, 0.1]),
        torch.zeros_like(eta),
        torch.tensor([[-2.0, 2.0]]),
        torch.tensor([[0.8, 0.0]]),
        torch.tensor([[1.0, 0.1]]),
        torch.tensor([[True, True]]),
        exact_lower_bound=torch.tensor([-2.0]),
        global_exact=torch.tensor([True]),
    )
    assert output.coverage_penalty.item() == 0.0
    assert torch.all(output.coverage_shortfall_logits == 0.0)
    assert torch.equal(
        output.interpolation_logits,
        torch.tensor([[-2.0, 2.0]], dtype=torch.float64),
    )

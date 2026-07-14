from __future__ import annotations

import pytest
import torch

from model.monotone_pixel_calibrator import (
    PIXEL_BUDGET_ONLY_SCOPE,
    PIXEL_RISK_WITH_REJECT_SCOPE,
    MonotonePixelBudgetCalibrator,
    MonotonePixelRiskCalibrator,
    pixel_budget_from_spec,
)
from model.threshold_calibrator import ThresholdCalibrator
from rc.schema import BudgetSpec


def _model() -> MonotonePixelBudgetCalibrator:
    torch.manual_seed(7)
    return MonotonePixelBudgetCalibrator(
        context_feature_dim=5,
        pixel_budget_grid=(1e-4, 1e-5, 1e-6),
        hidden_dims=(8,),
        dropout=0.0,
    )


@pytest.mark.parametrize(
    "grid",
    [
        (1e-4,),
        (1e-5, 1e-4),
        (1e-4, 1e-4),
        (1e-4, 0.0),
        (1e-4, float("nan")),
    ],
)
def test_pixel_budget_grid_must_be_positive_and_strictly_descending(grid) -> None:
    with pytest.raises(ValueError, match="pixel_budget_grid"):
        MonotonePixelBudgetCalibrator(3, grid, hidden_dims=(4,), dropout=0.0)


def test_thresholds_strictly_increase_as_pixel_budget_tightens() -> None:
    model = _model()
    features = torch.randn(4, 5)
    output = model(features)

    assert output.grid_logits.shape == (4, 3)
    assert torch.all(output.grid_logits[:, 1:] > output.grid_logits[:, :-1])
    assert torch.all(
        output.grid_thresholds[:, 1:] > output.grid_thresholds[:, :-1]
    )
    assert torch.all(output.pixel_budget_grid[:-1] > output.pixel_budget_grid[1:])


def test_float64_curve_preserves_extreme_tail_order_instead_of_saturating() -> None:
    model = MonotonePixelBudgetCalibrator(
        context_feature_dim=2,
        pixel_budget_grid=(1e-4, 1e-5, 1e-6),
        hidden_dims=(),
        dropout=0.0,
    )
    with torch.no_grad():
        model.spacing_head.weight.zero_()
        model.spacing_head.bias.copy_(
            torch.tensor([1000.0, -1000.0, -1000.0, -1000.0])
        )

    output = model(torch.zeros(1, 2))
    assert output.grid_logits.dtype == torch.float64
    assert output.grid_thresholds.dtype == torch.float64
    assert torch.all(
        output.grid_thresholds[:, 1:] > output.grid_thresholds[:, :-1]
    )


def test_interpolation_preserves_grid_endpoints_and_budget_order() -> None:
    model = _model()
    features = torch.randn(2, 5)
    request = torch.tensor([1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
    output = model(features, pixel_budgets=request)

    assert output.requested_logits is not None
    assert output.requested_thresholds is not None
    assert output.requested_pixel_budgets is not None
    assert output.requested_logits.shape == (2, 5)
    # Requests are loose -> strict, therefore thresholds must increase.
    assert torch.all(
        output.requested_logits[:, 1:] > output.requested_logits[:, :-1]
    )
    assert torch.all(
        output.requested_thresholds[:, 1:] > output.requested_thresholds[:, :-1]
    )
    torch.testing.assert_close(output.requested_logits[:, 0], output.grid_logits[:, 0])
    torch.testing.assert_close(output.requested_logits[:, 2], output.grid_logits[:, 1])
    torch.testing.assert_close(output.requested_logits[:, -1], output.grid_logits[:, -1])


def test_rank_one_requests_are_shared_not_ambiguous_per_sample_budgets() -> None:
    model = _model()
    features = torch.randn(2, 5)
    shared = model(features, pixel_budgets=torch.tensor([5e-5, 5e-6]))
    per_sample = model(
        features,
        pixel_budgets=torch.tensor([[5e-5], [5e-6]]),
    )

    assert shared.requested_thresholds is not None
    assert per_sample.requested_thresholds is not None
    assert shared.requested_thresholds.shape == (2, 2)
    assert per_sample.requested_thresholds.shape == (2, 1)


def test_requests_outside_trained_pixel_grid_are_rejected() -> None:
    model = _model()
    with pytest.raises(ValueError, match="extrapolation is disabled"):
        model(torch.zeros(2, 5), pixel_budgets=torch.tensor([1e-7]))


def test_component_budget_and_reject_remain_explicitly_unsupported() -> None:
    model = _model()
    pixel_only = BudgetSpec.from_optional(pixel_budget=1e-5)
    dual_budget = BudgetSpec.from_optional(
        pixel_budget=1e-5, component_budget=1.0
    )
    component_only = BudgetSpec.from_optional(component_budget=1.0)

    assert pixel_budget_from_spec(pixel_only) == pytest.approx(1e-5)
    with pytest.raises(ValueError, match="component budgets are not supported"):
        pixel_budget_from_spec(dual_budget)
    with pytest.raises(ValueError, match="component budgets are not supported"):
        pixel_budget_from_spec(component_only)
    assert model.budget_scope == PIXEL_BUDGET_ONLY_SCOPE
    assert model.supports_component_budget is False
    assert model.supports_reject is False
    assert model.training_pipeline_integrated is False
    assert model.capability_contract() == {
        "budget_scope": PIXEL_BUDGET_ONLY_SCOPE,
        "supports_component_budget": False,
        "supports_reject": False,
        "training_pipeline_integrated": False,
    }
    reconstructed = MonotonePixelBudgetCalibrator(**model.export_config())
    assert reconstructed.export_config() == model.export_config()

    proposed_output = model(torch.zeros(1, 5))
    assert not hasattr(proposed_output, "reject_logit")
    # The existing dual-budget + reject baseline remains a separate class/API.
    baseline_threshold, baseline_reject = ThresholdCalibrator(
        input_dim=5, hidden_dim=4, dropout=0.0
    )(torch.zeros(1, 5))
    assert baseline_threshold.shape == baseline_reject.shape == (1,)


def test_integrated_pixel_risk_threshold_and_reject_are_both_monotone() -> None:
    torch.manual_seed(11)
    model = MonotonePixelRiskCalibrator(
        context_feature_dim=5,
        pixel_budget_grid=(1e-4, 1e-5, 1e-6),
        hidden_dims=(8,),
        dropout=0.0,
    )
    output = model(
        torch.randn(4, 5),
        pixel_budgets=torch.tensor([1e-4, 1e-5, 1e-6]),
    )

    assert torch.all(output.grid_logits[:, 1:] > output.grid_logits[:, :-1])
    assert torch.all(
        output.grid_reject_logits[:, 1:] > output.grid_reject_logits[:, :-1]
    )
    assert output.requested_thresholds is not None
    assert output.requested_reject_probabilities is not None
    assert torch.all(
        output.requested_thresholds[:, 1:]
        > output.requested_thresholds[:, :-1]
    )
    assert torch.all(
        output.requested_reject_probabilities[:, 1:]
        > output.requested_reject_probabilities[:, :-1]
    )
    assert model.capability_contract() == {
        "budget_scope": PIXEL_RISK_WITH_REJECT_SCOPE,
        "supports_component_budget": False,
        "supports_reject": True,
        "training_pipeline_integrated": True,
        "risk_aligned_query_loss": False,
        "training_objective": "asymmetric_oracle_threshold_plus_reject_bce",
        "curve_compute_dtype": "float64",
        "budget_interpolation": "piecewise_linear_log10_no_extrapolation",
        "risk_guarantee": "empirical_not_certified",
        "component_budget_reason": (
            "connected-component false-alarm counts are not monotone in threshold"
        ),
    }
    reconstructed = MonotonePixelRiskCalibrator(**model.export_config())
    assert reconstructed.export_config() == model.export_config()


def test_integrated_pixel_risk_model_rejects_component_budget_contract() -> None:
    model = MonotonePixelRiskCalibrator(
        context_feature_dim=3,
        pixel_budget_grid=(1e-4, 1e-5),
        hidden_dims=(4,),
        dropout=0.0,
    )
    dual = BudgetSpec.from_optional(pixel_budget=1e-5, component_budget=1.0)
    with pytest.raises(ValueError, match="component budgets are not supported"):
        pixel_budget_from_spec(dual)
    with pytest.raises(ValueError, match="extrapolation is disabled"):
        model(torch.zeros(1, 3), pixel_budgets=torch.tensor([1e-7]))

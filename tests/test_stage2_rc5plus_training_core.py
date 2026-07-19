from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from model.budget_conditioned_residual_transport_calibrator import (
    BudgetConditionedDirectResidualTransportCalibrator,
    BudgetConditionedMonotoneNoTargetAnchorCalibrator,
    BudgetConditionedMonotoneResidualTransportCalibrator,
)
from model.endpoint_aware_threshold import encode_probability_numpy
from rc.stage2_compositional_curve_provider import (
    CYCLIC_QUERY_SIZE,
    build_compositional_exact_curve_provider,
    build_per_image_exact_event_curve,
    build_per_image_exact_event_curve_bank,
)
from rc.stage2_rc5plus_training_core import (
    RC5PLUS_LOSS_METRIC_NAMES,
    RC5PLUS_MAX_EXACT_BRACKET_ROWS,
    Stage2RC5PlusTrainingCoreError,
    compact_exact_curve_coordinate_brackets_v2,
    rc5plus_batch_loss,
)


def _identity(episode: int, image: int) -> str:
    return hashlib.sha256(f"rc5plus-{episode}-{image}".encode()).hexdigest()


def _provider(episode: int = 0, *, dense: bool = False):
    if dense:
        thresholds = np.linspace(0.0, 1.0, 20, dtype=np.float64)
        fp = np.arange(19, -1, -1, dtype=np.int64)
        tp = np.asarray([min(index, 19 - index) for index in range(20)], dtype=np.int64)
        objects = 20
    else:
        thresholds = np.asarray(
            [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0],
            dtype=np.float64,
        )
        fp = np.asarray([3000, 500, 100, 20, 5, 2, 1, 0], dtype=np.int64)
        tp = np.asarray([1, 2, 4, 8, 7, 5, 2, 0], dtype=np.int64)
        objects = 10
    curves = [
        build_per_image_exact_event_curve(
            image_identity_sha256=_identity(episode, image),
            thresholds=thresholds,
            false_positive_pixels=fp,
            matched_objects=tp,
            total_native_pixels=1_000_000,
            ground_truth_objects=objects,
        )
        for image in range(CYCLIC_QUERY_SIZE)
    ]
    bank = build_per_image_exact_event_curve_bank(curves)
    return build_compositional_exact_curve_provider(
        curve_bank=bank,
        ordered_image_identities=[curve.image_identity_sha256 for curve in curves],
    )


def _coordinates(values: np.ndarray, *, batch_size: int) -> torch.Tensor:
    return torch.from_numpy(
        encode_probability_numpy(np.tile(values, (batch_size, 1)))
    )


def _budget_tensors(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    numerators = torch.tensor(
        [row[0] for row in BUDGET_KNOT_RATIONALS], dtype=torch.int64
    ).reshape(1, -1).repeat(batch_size, 1)
    denominators = torch.tensor(
        [row[1] for row in BUDGET_KNOT_RATIONALS], dtype=torch.int64
    ).reshape(1, -1).repeat(batch_size, 1)
    return numerators, denominators


def _batch(batch_size: int = 2, *, with_providers: bool) -> dict[str, object]:
    numerators, denominators = _budget_tensors(batch_size)
    result: dict[str, object] = {
        "features": torch.tensor(
            [[0.2, -0.4, 0.7, 0.1], [-0.3, 0.6, 0.4, -0.2]][:batch_size],
            dtype=torch.float32,
        ),
        "anchor_coordinates": _coordinates(
            np.linspace(0.20, 0.90, 9), batch_size=batch_size
        ),
        "budget_numerators": numerators,
        "budget_denominators": denominators,
    }
    if with_providers:
        result["compositional_curve_providers"] = tuple(
            _provider(episode) for episode in range(batch_size)
        )
    else:
        result["oracle_coordinates"] = _coordinates(
            np.linspace(0.30, 0.96, 9), batch_size=batch_size
        )
    return result


def _loss_config() -> dict[str, float]:
    return {
        "lambda_violation": 1.0,
        "lambda_utility": 0.5,
        "lambda_oracle": 1.0,
        "lambda_smoothness": 0.01,
        "lambda_coverage": 0.0,
        "risk_epsilon": 1e-12,
        "coordinate_huber_delta": 1.0,
    }


@pytest.mark.parametrize(
    ("method", "model_type"),
    [
        ("T6_PLUS", BudgetConditionedDirectResidualTransportCalibrator),
        ("T7_PLUS", BudgetConditionedMonotoneResidualTransportCalibrator),
    ],
)
def test_t6plus_t7plus_huber_routes_are_finite_and_capacity_matched(
    method, model_type
) -> None:
    model = model_type(context_feature_dim=4, hidden_dims=(32,), dropout=0.0)

    output, losses = rc5plus_batch_loss(
        method=method,
        model=model,
        batch=_batch(with_providers=False),
        loss_config=_loss_config(),
    )
    losses["total"].backward()

    assert output.grid_coordinates.shape == (2, 9)
    assert tuple(losses) == RC5PLUS_LOSS_METRIC_NAMES
    assert torch.isfinite(losses["total"])
    assert losses["oracle_coordinate"].item() > 0.0
    assert losses["violation"].item() == 0.0
    assert model.transport_head.weight.grad is not None
    assert model.transport_head.weight.grad.abs().sum().item() > 0.0
    assert sum(parameter.numel() for parameter in model.parameters()) == 491


def test_t8plus_derives_oracle_and_risk_from_verified_providers() -> None:
    model = BudgetConditionedMonotoneResidualTransportCalibrator(
        context_feature_dim=4, hidden_dims=(32,), dropout=0.0
    )

    output, losses = rc5plus_batch_loss(
        method="T8_PLUS",
        model=model,
        batch=_batch(with_providers=True),
        loss_config=_loss_config(),
    )
    losses["total"].backward()

    assert output.grid_coordinates.shape == (2, 9)
    assert tuple(losses) == RC5PLUS_LOSS_METRIC_NAMES
    assert all(torch.isfinite(value) for value in losses.values())
    assert losses["coverage_penalty"].item() == 0.0
    assert model.transport_head.weight.grad is not None
    assert model.transport_head.weight.grad.abs().sum().item() > 0.0


def test_t8plus_no_anchor_uses_exact_risk_but_forbids_anchor_input() -> None:
    model = BudgetConditionedMonotoneNoTargetAnchorCalibrator(
        context_feature_dim=4, hidden_dims=(32,), dropout=0.0
    )
    batch = _batch(with_providers=True)
    batch.pop("anchor_coordinates")
    output, losses = rc5plus_batch_loss(
        method="T8_PLUS_NO_ANCHOR",
        model=model,
        batch=batch,
        loss_config=_loss_config(),
    )
    losses["total"].backward()
    assert output.grid_coordinates.shape == (2, 9)
    assert all(torch.isfinite(value) for value in losses.values())
    assert model.transport_head.weight.grad is not None
    assert model.transport_head.weight.grad.abs().sum().item() > 0.0

    injected = _batch(with_providers=True)
    with pytest.raises(Stage2RC5PlusTrainingCoreError, match="forbids"):
        rc5plus_batch_loss(
            method="T8_PLUS_NO_ANCHOR",
            model=model,
            batch=injected,
            loss_config=_loss_config(),
        )


def test_nine_predictions_transfer_at_most_eighteen_exact_rows() -> None:
    provider = _provider(dense=True)
    grid = np.linspace(0.0, 1.0, 20, dtype=np.float64)
    midpoints = (grid[:-1] + grid[1:]) / 2.0
    probabilities = midpoints[np.arange(0, 18, 2)]
    predicted = torch.from_numpy(
        encode_probability_numpy(probabilities)
    ).reshape(1, -1)

    coordinates, risk, pd, valid, oracle, gt_objects = (
        compact_exact_curve_coordinate_brackets_v2(predicted, (provider,))
    )

    assert coordinates.shape == (1, RC5PLUS_MAX_EXACT_BRACKET_ROWS)
    assert risk.shape == coordinates.shape
    assert pd.shape == coordinates.shape
    assert valid.all()
    assert oracle.shape == (1, 9)
    assert gt_objects.tolist() == [CYCLIC_QUERY_SIZE * 20]


@pytest.mark.parametrize("method", ["T6_PLUS", "T7_PLUS"])
def test_huber_routes_cannot_access_curve_providers(method: str) -> None:
    model_type = (
        BudgetConditionedDirectResidualTransportCalibrator
        if method == "T6_PLUS"
        else BudgetConditionedMonotoneResidualTransportCalibrator
    )
    model = model_type(context_feature_dim=4, hidden_dims=(32,), dropout=0.0)
    batch = _batch(with_providers=False)
    batch["compositional_curve_providers"] = (_provider(), _provider(1))

    with pytest.raises(Stage2RC5PlusTrainingCoreError, match="cannot access"):
        rc5plus_batch_loss(
            method=method,
            model=model,
            batch=batch,
            loss_config=_loss_config(),
        )


def test_t8plus_rejects_missing_provider_and_caller_oracle_injection() -> None:
    model = BudgetConditionedMonotoneResidualTransportCalibrator(
        context_feature_dim=4, hidden_dims=(32,), dropout=0.0
    )
    missing = _batch(with_providers=False)
    missing.pop("oracle_coordinates")
    with pytest.raises(Stage2RC5PlusTrainingCoreError, match="requires"):
        rc5plus_batch_loss(
            method="T8_PLUS",
            model=model,
            batch=missing,
            loss_config=_loss_config(),
        )

    injected = _batch(with_providers=True)
    injected["oracle_coordinates"] = _coordinates(
        np.linspace(0.1, 0.9, 9), batch_size=2
    )
    with pytest.raises(Stage2RC5PlusTrainingCoreError, match="must be derived"):
        rc5plus_batch_loss(
            method="T8_PLUS",
            model=model,
            batch=injected,
            loss_config=_loss_config(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "pixel_budgets",
        "curve_logits",
        "curve_coordinates",
        "curve_pixel_risk",
        "curve_pd",
        "curve_gt_objects",
        "decision_thresholds",
    ],
)
def test_t8plus_rejects_float_budget_curve_and_threshold_injection(field: str) -> None:
    model = BudgetConditionedMonotoneResidualTransportCalibrator(
        context_feature_dim=4, hidden_dims=(32,), dropout=0.0
    )
    batch = _batch(with_providers=True)
    batch[field] = torch.zeros((2, 9), dtype=torch.float64)

    with pytest.raises(Stage2RC5PlusTrainingCoreError, match="forbidden"):
        rc5plus_batch_loss(
            method="T8_PLUS",
            model=model,
            batch=batch,
            loss_config=_loss_config(),
        )


def test_exact_budget_lattice_and_exact_model_types_fail_closed() -> None:
    direct = BudgetConditionedDirectResidualTransportCalibrator(
        context_feature_dim=4, hidden_dims=(32,), dropout=0.0
    )
    monotone = BudgetConditionedMonotoneResidualTransportCalibrator(
        context_feature_dim=4, hidden_dims=(32,), dropout=0.0
    )
    mutated = _batch(with_providers=False)
    mutated["budget_denominators"] = mutated["budget_denominators"].clone()
    mutated["budget_denominators"][0, 4] += 1
    with pytest.raises(Stage2RC5PlusTrainingCoreError, match="exactly equal"):
        rc5plus_batch_loss(
            method="T6_PLUS",
            model=direct,
            batch=mutated,
            loss_config=_loss_config(),
        )

    with pytest.raises(Stage2RC5PlusTrainingCoreError, match="exact direct"):
        rc5plus_batch_loss(
            method="T6_PLUS",
            model=monotone,
            batch=_batch(with_providers=False),
            loss_config=_loss_config(),
        )
    with pytest.raises(Stage2RC5PlusTrainingCoreError, match="exact monotone"):
        rc5plus_batch_loss(
            method="T8_PLUS",
            model=direct,
            batch=_batch(with_providers=True),
            loss_config=_loss_config(),
        )

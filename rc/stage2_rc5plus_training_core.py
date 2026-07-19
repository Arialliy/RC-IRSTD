"""Fail-closed nine-budget training core for the RC5+ candidate.

T6+ and T7+ consume only verified nine-budget oracle coordinates and optimize
coordinate Huber loss.  T8+ consumes verifier-built compositional exact-event
providers, derives its oracle and object counts from those providers, and
transfers only the at-most-18 rows adjacent to the nine live predictions to
the model device.  Caller-supplied float budgets, aggregate curves, thresholds
and legacy clipped-logit fields are forbidden.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from losses.calibrator_risk import curve_query_risk_aligned_calibrator_loss
from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from model.budget_conditioned_residual_transport_calibrator import (
    BudgetConditionedDirectResidualTransportCalibrator,
    BudgetConditionedMonotoneNoTargetAnchorCalibrator,
    BudgetConditionedMonotoneResidualTransportCalibrator,
)
from model.endpoint_aware_threshold import (
    EndpointAwareThresholdError,
    decode_coordinate_torch,
)
from rc.stage2_compositional_curve_provider import (
    RC5PLUS_MAX_LIVE_PREDICTIONS,
    assert_compositional_exact_curve_provider,
)
from rc.stage2_rc5_training_core import oracle_coordinate_huber_loss


RC5PLUS_TRAINING_CORE_SCHEMA = "rc-irstd.stage2-rc5plus-training-core.v1"
RC5PLUS_METHODS = ("T6_PLUS", "T7_PLUS", "T8_PLUS")
RC5PLUS_ABLATION_METHODS = ("T8_PLUS_NO_ANCHOR",)
RC5PLUS_TRAINING_METHODS = RC5PLUS_METHODS + RC5PLUS_ABLATION_METHODS
RC5PLUS_LOSS_METRIC_NAMES = (
    "total",
    "violation",
    "utility",
    "oracle_coordinate",
    "curve_smoothness",
    "coverage_penalty",
)
RC5PLUS_MAX_EXACT_BRACKET_ROWS = 2 * RC5PLUS_MAX_LIVE_PREDICTIONS
_FORBIDDEN_CALLER_CURVE_FIELDS = (
    "pixel_budgets",
    "curve_logits",
    "curve_coordinates",
    "curve_pixel_risk",
    "curve_pd",
    "curve_gt_objects",
    "curve_thresholds",
    "aggregate_curve",
    "aggregate_curves",
    "decision_thresholds",
)


class Stage2RC5PlusTrainingCoreError(ValueError):
    """An RC5+ method, model or verified training batch is invalid."""


def _canonical_coordinates(
    value: Any,
    *,
    name: str,
    shape: tuple[int, int],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float64 or value.shape != shape:
        raise Stage2RC5PlusTrainingCoreError(
            f"{name} must be exact float64{list(shape)} EATC coordinates"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise Stage2RC5PlusTrainingCoreError(f"{name} must be finite")
    try:
        decode_coordinate_torch(value.detach())
    except EndpointAwareThresholdError as error:
        raise Stage2RC5PlusTrainingCoreError(
            f"{name} contains a noncanonical EATC coordinate"
        ) from error
    return value


def _exact_budget_matrix(
    batch: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    numerators = batch.get("budget_numerators")
    denominators = batch.get("budget_denominators")
    expected_shape = (batch_size, len(BUDGET_KNOT_RATIONALS))
    if (
        not isinstance(numerators, torch.Tensor)
        or not isinstance(denominators, torch.Tensor)
        or numerators.dtype != torch.int64
        or denominators.dtype != torch.int64
        or numerators.shape != expected_shape
        or denominators.shape != expected_shape
    ):
        raise Stage2RC5PlusTrainingCoreError(
            "budget_numerators/budget_denominators must be aligned int64[B,9]"
        )
    expected_n = torch.tensor(
        [row[0] for row in BUDGET_KNOT_RATIONALS], dtype=torch.int64
    ).reshape(1, -1).expand(batch_size, -1)
    expected_d = torch.tensor(
        [row[1] for row in BUDGET_KNOT_RATIONALS], dtype=torch.int64
    ).reshape(1, -1).expand(batch_size, -1)
    if not torch.equal(numerators.detach().to(device="cpu"), expected_n) or not torch.equal(
        denominators.detach().to(device="cpu"), expected_d
    ):
        raise Stage2RC5PlusTrainingCoreError(
            "training budgets must exactly equal the frozen nine-knot rational lattice"
        )
    return numerators.to(device=device, dtype=torch.float64) / denominators.to(
        device=device, dtype=torch.float64
    )


def _loss_scalar(
    config: Mapping[str, Any],
    key: str,
    *,
    positive: bool,
) -> float:
    if key not in config:
        raise Stage2RC5PlusTrainingCoreError(f"loss_config.{key} is required")
    value = config[key]
    if isinstance(value, bool):
        raise Stage2RC5PlusTrainingCoreError(
            f"loss_config.{key} must be numeric"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise Stage2RC5PlusTrainingCoreError(
            f"loss_config.{key} must be numeric"
        ) from error
    invalid = numeric <= 0.0 if positive else numeric < 0.0
    if not math.isfinite(numeric) or invalid:
        relation = "positive" if positive else "non-negative"
        raise Stage2RC5PlusTrainingCoreError(
            f"loss_config.{key} must be finite and {relation}"
        )
    return numeric


def _assert_method_model(method: str, model: nn.Module) -> None:
    if method == "T6_PLUS":
        if type(model) is not BudgetConditionedDirectResidualTransportCalibrator:
            raise Stage2RC5PlusTrainingCoreError(
                "T6_PLUS requires the exact direct residual-transport model type"
            )
    elif method in {"T7_PLUS", "T8_PLUS"}:
        if type(model) is not BudgetConditionedMonotoneResidualTransportCalibrator:
            raise Stage2RC5PlusTrainingCoreError(
                f"{method} requires the exact monotone residual-transport model type"
            )
    elif method == "T8_PLUS_NO_ANCHOR":
        if type(model) is not BudgetConditionedMonotoneNoTargetAnchorCalibrator:
            raise Stage2RC5PlusTrainingCoreError(
                "T8_PLUS_NO_ANCHOR requires the exact no-target-anchor model type"
            )
    else:
        raise Stage2RC5PlusTrainingCoreError(
            "method is not a frozen RC5+ training route"
        )


def compact_exact_curve_coordinate_brackets_v2(
    predicted_coordinates: torch.Tensor,
    providers: Any,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return local exact rows, provider-derived oracle and object counts."""

    if (
        not isinstance(predicted_coordinates, torch.Tensor)
        or predicted_coordinates.dtype != torch.float64
        or predicted_coordinates.ndim != 2
        or predicted_coordinates.shape[1] != RC5PLUS_MAX_LIVE_PREDICTIONS
    ):
        raise Stage2RC5PlusTrainingCoreError(
            "predicted_coordinates must be float64[B,9]"
        )
    batch_size = int(predicted_coordinates.shape[0])
    _canonical_coordinates(
        predicted_coordinates,
        name="predicted_coordinates",
        shape=(batch_size, RC5PLUS_MAX_LIVE_PREDICTIONS),
    )
    if isinstance(providers, torch.Tensor) or not isinstance(
        providers, (tuple, list)
    ):
        raise Stage2RC5PlusTrainingCoreError(
            "compositional_curve_providers must be a CPU tuple"
        )
    if len(providers) != batch_size:
        raise Stage2RC5PlusTrainingCoreError(
            "compositional provider batch cardinality mismatch"
        )
    detached = predicted_coordinates.detach().to(
        device="cpu", dtype=torch.float64
    ).numpy()
    checked = [assert_compositional_exact_curve_provider(item) for item in providers]
    compact = [
        provider.compact_brackets_v2(detached[row])
        for row, provider in enumerate(checked)
    ]
    oracle = [provider.select_exact_oracle_rows_v2() for provider in checked]
    if any(item.budget_rationals != BUDGET_KNOT_RATIONALS for item in oracle):
        raise RuntimeError("provider oracle budget lattice differs from RC5+")
    width = max(int(item.coordinates.size) for item in compact)
    if not 2 <= width <= RC5PLUS_MAX_EXACT_BRACKET_ROWS:
        raise RuntimeError("RC5+ compact bracket width must be in [2,18]")
    device = predicted_coordinates.device
    coordinates = torch.zeros(
        (batch_size, width), dtype=torch.float64, device=device
    )
    risk = torch.zeros_like(coordinates)
    pd = torch.zeros_like(coordinates)
    valid = torch.zeros(coordinates.shape, dtype=torch.bool, device=device)
    for row, item in enumerate(compact):
        count = int(item.coordinates.size)
        if not 2 <= count <= RC5PLUS_MAX_EXACT_BRACKET_ROWS:
            raise RuntimeError("provider emitted an invalid RC5+ bracket width")
        valid[row, :count] = True
        coordinates[row, :count] = torch.from_numpy(
            np.array(item.coordinates, dtype=np.float64, copy=True)
        ).to(device=device)
        risk[row, :count] = torch.from_numpy(
            np.array(item.pixel_false_alarm_rate, dtype=np.float64, copy=True)
        ).to(device=device)
        pd[row, :count] = torch.from_numpy(
            np.array(item.detection_probability, dtype=np.float64, copy=True)
        ).to(device=device)
    oracle_coordinates = torch.stack(
        [
            torch.from_numpy(
                np.array(item.coordinates, dtype=np.float64, copy=True)
            )
            for item in oracle
        ]
    ).to(device=device)
    gt_objects = torch.tensor(
        [provider.ground_truth_objects for provider in checked],
        dtype=torch.int64,
        device=device,
    )
    _canonical_coordinates(
        oracle_coordinates,
        name="provider oracle_coordinates",
        shape=(batch_size, RC5PLUS_MAX_LIVE_PREDICTIONS),
    )
    return coordinates, risk, pd, valid, oracle_coordinates, gt_objects


def rc5plus_batch_loss(
    *,
    method: str,
    model: nn.Module,
    batch: Mapping[str, Any],
    loss_config: Mapping[str, Any],
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Route capacity-matched T6+/T7+ and exact-risk T8+ batches."""

    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping")
    if not isinstance(loss_config, Mapping):
        raise TypeError("loss_config must be a mapping")
    _assert_method_model(method, model)
    features = batch.get("features")
    if (
        not isinstance(features, torch.Tensor)
        or features.dtype != torch.float32
        or features.ndim != 2
        or features.shape[0] == 0
    ):
        raise Stage2RC5PlusTrainingCoreError(
            "batch.features must be nonempty float32[B,D]"
        )
    batch_size = int(features.shape[0])
    anchors: torch.Tensor | None = None
    if method == "T8_PLUS_NO_ANCHOR":
        if "anchor_coordinates" in batch:
            raise Stage2RC5PlusTrainingCoreError(
                "T8_PLUS_NO_ANCHOR forbids anchor_coordinates"
            )
    else:
        anchors = _canonical_coordinates(
            batch.get("anchor_coordinates"),
            name="batch.anchor_coordinates",
            shape=(batch_size, RC5PLUS_MAX_LIVE_PREDICTIONS),
        )
    budgets = _exact_budget_matrix(
        batch, batch_size=batch_size, device=features.device
    )
    for field in _FORBIDDEN_CALLER_CURVE_FIELDS:
        if field in batch:
            raise Stage2RC5PlusTrainingCoreError(
                f"caller-supplied {field} is forbidden in RC5+ training"
            )

    providers = batch.get("compositional_curve_providers")
    if method in {"T6_PLUS", "T7_PLUS"}:
        if providers is not None:
            raise Stage2RC5PlusTrainingCoreError(
                f"{method} cannot access exact-event curve providers"
            )
        oracle = _canonical_coordinates(
            batch.get("oracle_coordinates"),
            name="batch.oracle_coordinates",
            shape=(batch_size, RC5PLUS_MAX_LIVE_PREDICTIONS),
        ).to(device=features.device)
    else:
        if "oracle_coordinates" in batch:
            raise Stage2RC5PlusTrainingCoreError(
                "T8_PLUS oracle must be derived from verifier-built providers"
            )
        if providers is None:
            raise Stage2RC5PlusTrainingCoreError(
                "T8_PLUS requires compositional_curve_providers"
            )
        oracle = None

    if method == "T8_PLUS_NO_ANCHOR":
        output = model(features)
    else:
        assert anchors is not None
        output = model(
            features, anchor_coordinates=anchors.to(device=features.device)
        )
    if not hasattr(output, "grid_coordinates"):
        raise TypeError("RC5+ model output must expose grid_coordinates")
    predicted = _canonical_coordinates(
        output.grid_coordinates,
        name="model output grid_coordinates",
        shape=(batch_size, RC5PLUS_MAX_LIVE_PREDICTIONS),
    )
    if method in {"T6_PLUS", "T7_PLUS"}:
        huber = oracle_coordinate_huber_loss(
            predicted,
            oracle,
            torch.ones_like(predicted, dtype=torch.bool),
            delta=_loss_scalar(
                loss_config, "coordinate_huber_delta", positive=True
            ),
        )
        zero = huber * 0.0
        return output, {
            "total": huber,
            "violation": zero,
            "utility": zero,
            "oracle_coordinate": huber,
            "curve_smoothness": zero,
            "coverage_penalty": zero,
        }

    curve_coordinates, curve_risk, curve_pd, curve_valid, oracle, gt_objects = (
        compact_exact_curve_coordinate_brackets_v2(predicted, providers)
    )
    result = curve_query_risk_aligned_calibrator_loss(
        predicted,
        budgets,
        oracle,
        curve_coordinates,
        curve_risk,
        curve_pd,
        curve_valid,
        curve_coordinates[:, 0],
        torch.ones(batch_size, dtype=torch.bool, device=predicted.device),
        oracle_valid=torch.ones_like(predicted, dtype=torch.bool),
        utility_episode_valid=gt_objects > 0,
        lambda_violation=_loss_scalar(
            loss_config, "lambda_violation", positive=False
        ),
        lambda_utility=_loss_scalar(
            loss_config, "lambda_utility", positive=False
        ),
        lambda_oracle_logit=_loss_scalar(
            loss_config, "lambda_oracle", positive=False
        ),
        lambda_curve_smoothness=_loss_scalar(
            loss_config, "lambda_smoothness", positive=False
        ),
        lambda_coverage=_loss_scalar(
            loss_config, "lambda_coverage", positive=False
        ),
        epsilon=_loss_scalar(loss_config, "risk_epsilon", positive=True),
        oracle_huber_delta=_loss_scalar(
            loss_config, "coordinate_huber_delta", positive=True
        ),
    )
    return output, {
        "total": result.total,
        "violation": result.violation,
        "utility": result.utility,
        "oracle_coordinate": result.oracle_logit,
        "curve_smoothness": result.curve_smoothness,
        "coverage_penalty": result.coverage_penalty,
    }


__all__ = [
    "RC5PLUS_LOSS_METRIC_NAMES",
    "RC5PLUS_ABLATION_METHODS",
    "RC5PLUS_MAX_EXACT_BRACKET_ROWS",
    "RC5PLUS_METHODS",
    "RC5PLUS_TRAINING_METHODS",
    "RC5PLUS_TRAINING_CORE_SCHEMA",
    "Stage2RC5PlusTrainingCoreError",
    "compact_exact_curve_coordinate_brackets_v2",
    "rc5plus_batch_loss",
]

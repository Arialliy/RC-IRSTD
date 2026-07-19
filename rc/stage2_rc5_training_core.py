"""Result-free RC5 training primitives in endpoint-aware coordinate space.

This module is the narrow bridge between the endpoint-aware T6/T7/T8 models
and the existing exact-curve risk objective.  It deliberately does not open
collections or publish checkpoints.  Full curves stay as ragged CPU arrays;
only the at-most-six exact event rows bracketing the three live predictions
are transferred to the model device.

The legacy risk implementation is algebraically coordinate-agnostic, but its
argument names say ``logit``.  RC5 calls it only through the wrapper below and
passes EATC-v2 coordinates for every coordinate-valued argument.  Mixing EATC
and clipped-logit values is rejected by the batch contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.calibrator_risk import curve_query_risk_aligned_calibrator_loss
from model.endpoint_aware_threshold import (
    EndpointAwareThresholdError,
    decode_coordinate_numpy,
    decode_coordinate_torch,
    encode_probability_numpy,
)
from rc.stage2_compositional_curve_provider import (
    assert_compositional_exact_curve_provider,
)


RC5_TRAINING_CORE_SCHEMA = "rc-irstd.stage2-rc5-training-core.v1"
RC5_LOSS_METRIC_NAMES = (
    "total",
    "violation",
    "utility",
    "oracle_coordinate",
    "curve_smoothness",
    "coverage_penalty",
)
_INTEGER_TORCH_DTYPES = frozenset(
    {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)


class Stage2RC5TrainingCoreError(ValueError):
    """A synthetic or verified RC5 training batch violates its contract."""


@dataclass(frozen=True)
class Stage2CurveCoordinateView:
    """Lazy EATC-v2 projection over one verified ascending threshold column."""

    thresholds: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.thresholds, np.ndarray):
            raise Stage2RC5TrainingCoreError(
                "curve thresholds must be an explicit numpy array"
            )
        values = self.thresholds
        if values.dtype != np.float64 or values.ndim != 1 or values.size < 2:
            raise Stage2RC5TrainingCoreError(
                "curve thresholds must be one float64 vector with at least two rows"
            )
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise Stage2RC5TrainingCoreError("curve thresholds must be finite in [0,1]")
        if np.any(np.diff(values) <= 0.0):
            raise Stage2RC5TrainingCoreError(
                "exact event thresholds must be strictly ascending and deduplicated"
            )
        if values[0] != 0.0 or values[-1] != 1.0:
            raise Stage2RC5TrainingCoreError(
                "exact event thresholds must include exact 0/1 endpoints"
            )
        # Own an immutable probability snapshot, but never materialize the
        # potentially million-row coordinate projection.
        frozen = np.array(values, dtype=np.float64, order="C", copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "thresholds", frozen)

    def __len__(self) -> int:
        return int(self.thresholds.size)

    def __getitem__(self, index: Any) -> Any:
        result = encode_probability_numpy(self.thresholds[index])
        return float(result) if np.ndim(result) == 0 else result

    def take(self, indices: Sequence[int]) -> np.ndarray:
        raw = np.asarray(indices)
        if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.dtype == np.bool_:
            raise TypeError("curve coordinate indices must be one integer vector")
        if np.any(raw < 0) or np.any(raw >= len(self)):
            raise IndexError("curve coordinate indices are out of range")
        return encode_probability_numpy(self.thresholds[raw.astype(np.intp)])

    def _searchsorted_coordinate_right(self, coordinate: float) -> int:
        """Binary-search one EATC coordinate without projecting the full curve."""

        left = 0
        right = len(self)
        while left < right:
            middle = (left + right) // 2
            event_coordinate = float(
                encode_probability_numpy(self.thresholds[middle])
            )
            if event_coordinate <= coordinate:
                left = middle + 1
            else:
                right = middle
        return left

    def bracket_union(self, query_coordinates: np.ndarray) -> np.ndarray:
        if (
            not isinstance(query_coordinates, np.ndarray)
            or query_coordinates.dtype != np.float64
        ):
            raise TypeError(
                "each live query must be an explicit float64 EATC array"
            )
        query = query_coordinates
        if query.shape != (3,) or not np.isfinite(query).all():
            raise Stage2RC5TrainingCoreError(
                "each live query must contain three finite EATC coordinates"
            )
        try:
            # Validation only.  Searching through decoded probabilities can
            # shift a tail event by one ULP, so comparison stays in EATC space.
            decode_coordinate_numpy(query)
        except EndpointAwareThresholdError as error:
            raise Stage2RC5TrainingCoreError(
                "live query contains a noncanonical EATC coordinate"
            ) from error
        right = np.asarray(
            [self._searchsorted_coordinate_right(float(value)) for value in query],
            dtype=np.int64,
        )
        right = np.clip(right, 1, len(self) - 1).astype(np.int64, copy=False)
        selected = np.unique(np.concatenate((right - 1, right)))
        if selected.size < 2 or selected.size > 6:
            raise RuntimeError("RC5 compact bracket cardinality is invalid")
        return selected

    @property
    def nbytes(self) -> int:
        """No materialized coordinate column is retained."""

        return 0


def _require_canonical_coordinate_tensor(
    value: Any,
    *,
    name: str,
    ndim: int | None = None,
    nonempty: bool = False,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must be an exact float64 EATC tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if nonempty and value.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    try:
        decode_coordinate_torch(value.detach())
    except EndpointAwareThresholdError as error:
        raise Stage2RC5TrainingCoreError(
            f"{name} contains a noncanonical EATC coordinate"
        ) from error
    return value


def oracle_coordinate_huber_loss(
    predicted_coordinates: torch.Tensor,
    oracle_coordinates: torch.Tensor,
    oracle_valid: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    predicted = _require_canonical_coordinate_tensor(
        predicted_coordinates,
        name="predicted_coordinates",
        ndim=2,
        nonempty=True,
    )
    if not isinstance(oracle_coordinates, torch.Tensor):
        raise TypeError("oracle_coordinates must be a torch.Tensor")
    if oracle_coordinates.dtype != torch.float64:
        raise TypeError("oracle_coordinates must be an exact float64 EATC tensor")
    if oracle_coordinates.shape != predicted.shape:
        raise ValueError("oracle_coordinates must match predicted coordinates")
    if (
        not isinstance(oracle_valid, torch.Tensor)
        or oracle_valid.dtype != torch.bool
        or oracle_valid.shape != predicted_coordinates.shape
    ):
        raise ValueError("oracle_valid must be bool and match predicted coordinates")
    if isinstance(delta, bool) or not math.isfinite(float(delta)) or float(delta) <= 0.0:
        raise ValueError("Huber delta must be finite and positive")
    mask = oracle_valid.to(device=predicted.device)
    if not bool(mask.any().item()):
        raise Stage2RC5TrainingCoreError("oracle coordinate batch has no valid target")
    oracle = oracle_coordinates.to(
        device=predicted.device, dtype=torch.float64
    )
    _require_canonical_coordinate_tensor(
        oracle[mask],
        name="valid oracle_coordinates",
        nonempty=True,
    )
    return F.huber_loss(
        predicted[mask],
        oracle[mask],
        reduction="mean",
        delta=float(delta),
    )


def _ragged_float64(
    value: Any,
    *,
    field: str,
    expected_size: int,
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.ndim != 1 or value.dtype != torch.float64:
            raise Stage2RC5TrainingCoreError(
                f"{field} must be one ragged CPU float64 vector"
            )
        array = value.detach().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise Stage2RC5TrainingCoreError(
            f"{field} must be an explicit NumPy array or CPU tensor"
        )
    if array.dtype != np.float64 or array.ndim != 1 or array.size != expected_size:
        raise Stage2RC5TrainingCoreError(
            f"{field} must be float64 and align with its exact curve"
        )
    if not np.isfinite(array).all():
        raise Stage2RC5TrainingCoreError(f"{field} contains non-finite values")
    return array


def compact_exact_curve_coordinate_brackets(
    predicted_coordinates: torch.Tensor,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the exact event union touched by three live EATC coordinates."""

    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping")
    predicted = _require_canonical_coordinate_tensor(
        predicted_coordinates,
        name="predicted_coordinates",
        ndim=2,
        nonempty=True,
    )
    if predicted.shape[1] != 3:
        raise ValueError("predicted_coordinates must have shape [B,3]")
    if "curve_logits" in batch:
        raise Stage2RC5TrainingCoreError("legacy clipped-logit curve input is forbidden")
    providers = batch.get("compositional_curve_providers")
    materialized_fields = (
        "curve_coordinates",
        "curve_pixel_risk",
        "curve_pd",
    )
    materialized_present = any(field in batch for field in materialized_fields)
    if providers is not None:
        if materialized_present:
            raise Stage2RC5TrainingCoreError(
                "live compositional providers and materialized aggregate curves "
                "are mutually exclusive"
            )
        if isinstance(providers, torch.Tensor) or not isinstance(
            providers, (tuple, list)
        ):
            raise Stage2RC5TrainingCoreError(
                "compositional_curve_providers must be a CPU tuple"
            )
        if len(providers) != predicted.shape[0]:
            raise Stage2RC5TrainingCoreError(
                "compositional provider batch cardinality mismatch"
            )
        detached = predicted.detach().to(device="cpu", dtype=torch.float64).numpy()
        compact = [
            assert_compositional_exact_curve_provider(provider).compact_brackets(
                detached[row]
            )
            for row, provider in enumerate(providers)
        ]
        width = max(int(item.coordinates.size) for item in compact)
        if width > 6:
            raise RuntimeError("a three-coordinate provider emitted more than six rows")
        device = predicted.device
        coordinates = torch.zeros(
            (predicted.shape[0], width), dtype=torch.float64, device=device
        )
        pixel_risk = torch.zeros_like(coordinates)
        pd = torch.zeros_like(coordinates)
        valid = torch.zeros(coordinates.shape, dtype=torch.bool, device=device)
        for row, item in enumerate(compact):
            count = int(item.coordinates.size)
            if not 2 <= count <= 6:
                raise RuntimeError("provider bracket width must be in [2,6]")
            valid[row, :count] = True
            coordinates[row, :count] = torch.from_numpy(
                np.array(item.coordinates, dtype=np.float64, copy=True)
            ).to(device=device)
            pixel_risk[row, :count] = torch.from_numpy(
                np.array(
                    item.pixel_false_alarm_rate,
                    dtype=np.float64,
                    copy=True,
                )
            ).to(device=device)
            pd[row, :count] = torch.from_numpy(
                np.array(
                    item.detection_probability,
                    dtype=np.float64,
                    copy=True,
                )
            ).to(device=device)
        return coordinates, pixel_risk, pd, valid

    views = batch.get("curve_coordinates")
    risks = batch.get("curve_pixel_risk")
    pds = batch.get("curve_pd")
    for field, values in (
        ("curve_coordinates", views),
        ("curve_pixel_risk", risks),
        ("curve_pd", pds),
    ):
        if isinstance(values, torch.Tensor) or not isinstance(values, (tuple, list)):
            raise Stage2RC5TrainingCoreError(
                f"{field} must be a ragged CPU tuple, not a padded tensor"
            )
        if len(values) != predicted.shape[0]:
            raise Stage2RC5TrainingCoreError(f"{field} batch cardinality mismatch")

    detached = predicted.detach().to(device="cpu").numpy()
    indices: list[np.ndarray] = []
    checked_views: list[Stage2CurveCoordinateView] = []
    for row, value in enumerate(views):
        if not isinstance(value, Stage2CurveCoordinateView):
            raise Stage2RC5TrainingCoreError(
                "curve_coordinates entries must be Stage2CurveCoordinateView"
            )
        checked_views.append(value)
        indices.append(value.bracket_union(detached[row]))
    width = max(item.size for item in indices)
    device = predicted.device
    coordinates = torch.zeros(
        (predicted.shape[0], width), dtype=torch.float64, device=device
    )
    pixel_risk = torch.zeros_like(coordinates)
    pd = torch.zeros_like(coordinates)
    valid = torch.zeros(
        coordinates.shape, dtype=torch.bool, device=device
    )
    for row, (view, selected) in enumerate(zip(checked_views, indices, strict=True)):
        size = len(view)
        risk = _ragged_float64(risks[row], field="curve_pixel_risk", expected_size=size)
        detection = _ragged_float64(pds[row], field="curve_pd", expected_size=size)
        chosen_coordinates = view.take(selected)
        if np.any(np.diff(chosen_coordinates) <= 0.0):
            raise Stage2RC5TrainingCoreError("selected EATC curve coordinates are not strict")
        count = int(selected.size)
        valid[row, :count] = True
        coordinates[row, :count] = torch.from_numpy(chosen_coordinates).to(device=device)
        pixel_risk[row, :count] = torch.from_numpy(risk[selected]).to(device=device)
        pd[row, :count] = torch.from_numpy(detection[selected]).to(device=device)
    return coordinates, pixel_risk, pd, valid


def _loss_scalar(
    config: Mapping[str, Any],
    key: str,
    *,
    positive: bool,
) -> float:
    if key not in config:
        raise Stage2RC5TrainingCoreError(f"loss_config.{key} is required")
    value = config[key]
    if isinstance(value, bool):
        raise Stage2RC5TrainingCoreError(f"loss_config.{key} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise Stage2RC5TrainingCoreError(
            f"loss_config.{key} must be numeric"
        ) from error
    if not math.isfinite(numeric) or (numeric <= 0.0 if positive else numeric < 0.0):
        relation = "positive" if positive else "non-negative"
        raise Stage2RC5TrainingCoreError(
            f"loss_config.{key} must be finite and {relation}"
        )
    return numeric


def rc5_batch_loss(
    *,
    method: str,
    model: nn.Module,
    batch: Mapping[str, Any],
    loss_config: Mapping[str, Any],
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Route T6/T7 to coordinate Huber and T8 to exact-curve risk loss."""

    if method not in {"T6", "T7", "T8"}:
        raise Stage2RC5TrainingCoreError("method must be T6, T7 or T8")
    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping")
    if not isinstance(loss_config, Mapping):
        raise TypeError("loss_config must be a mapping")
    for field in ("features", "anchor_coordinates", "oracle_coordinates"):
        if not isinstance(batch.get(field), torch.Tensor):
            raise TypeError(f"batch.{field} must be a tensor")
    output = model(
        batch["features"], anchor_coordinates=batch["anchor_coordinates"]
    )
    if not hasattr(output, "grid_coordinates"):
        raise TypeError("model output must expose grid_coordinates")
    predicted = _require_canonical_coordinate_tensor(
        output.grid_coordinates,
        name="model output grid_coordinates",
        ndim=2,
        nonempty=True,
    )
    oracle = batch["oracle_coordinates"]
    _require_canonical_coordinate_tensor(
        oracle,
        name="batch.oracle_coordinates",
        ndim=2,
        nonempty=True,
    )
    if predicted.shape != oracle.shape:
        raise Stage2RC5TrainingCoreError("model/oracle coordinate shapes differ")
    valid_oracle = torch.ones_like(oracle, dtype=torch.bool)
    if method in {"T6", "T7"}:
        huber = oracle_coordinate_huber_loss(
            predicted,
            oracle,
            valid_oracle,
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

    for field in ("pixel_budgets", "curve_gt_objects"):
        if not isinstance(batch.get(field), torch.Tensor):
            raise TypeError(f"batch.{field} must be a tensor")
    pixel_budgets = batch["pixel_budgets"]
    if pixel_budgets.dtype != torch.float64:
        raise TypeError("batch.pixel_budgets must be float64")
    gt_objects = batch["curve_gt_objects"]
    if (
        gt_objects.ndim != 1
        or gt_objects.shape[0] != predicted.shape[0]
        or gt_objects.dtype not in _INTEGER_TORCH_DTYPES
    ):
        raise TypeError("batch.curve_gt_objects must be one integer count per row")
    if bool((gt_objects < 0).any().item()):
        raise Stage2RC5TrainingCoreError(
            "batch.curve_gt_objects counts must be non-negative"
        )
    curve_coordinates, curve_risk, curve_pd, curve_valid = (
        compact_exact_curve_coordinate_brackets(predicted, batch)
    )
    size = int(predicted.shape[0])
    result = curve_query_risk_aligned_calibrator_loss(
        predicted,
        pixel_budgets,
        oracle,
        curve_coordinates,
        curve_risk,
        curve_pd,
        curve_valid,
        curve_coordinates[:, 0],
        torch.ones(size, dtype=torch.bool, device=predicted.device),
        oracle_valid=valid_oracle,
        utility_episode_valid=gt_objects.to(device=predicted.device) > 0,
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
    "RC5_LOSS_METRIC_NAMES",
    "RC5_TRAINING_CORE_SCHEMA",
    "Stage2CurveCoordinateView",
    "Stage2RC5TrainingCoreError",
    "compact_exact_curve_coordinate_brackets",
    "oracle_coordinate_huber_loss",
    "rc5_batch_loss",
]

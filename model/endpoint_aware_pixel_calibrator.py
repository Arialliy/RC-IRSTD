"""RC5 endpoint-aware no-reject pixel-risk calibrators.

These classes intentionally live beside, rather than mutate, the checkpoint-v6
logit models.  Tensor shapes happen to be compatible, but the output semantics
are not: checkpoint-v7 heads predict endpoint-aware tail coordinates and old
weights must never be silently reinterpreted.  Every RC5 head is a convex
residual around the same T4 exact-order-statistic coordinate anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.endpoint_aware_threshold import (
    RAW_COORDINATE_MAX,
    RAW_COORDINATE_MIN,
    THRESHOLD_REPRESENTATION_SCHEMA,
    UPPER_ENDPOINT_COORDINATE,
    canonicalize_raw_torch,
    decode_coordinate_torch,
    representation_contract,
)


DIRECT_ENDPOINT_AWARE_MODEL_ID = "direct_endpoint_aware_pixel_calibrator"
MONOTONE_ENDPOINT_AWARE_MODEL_ID = "monotone_endpoint_aware_pixel_calibrator"
PIXEL_RISK_NO_REJECT_SCOPE = "pixel_risk_full_curve_no_component_no_reject"
T4_ANCHOR_SOURCE = "T4_exact_order_statistic"
ANCHOR_COORDINATE_CONTRACT = "required_float64_[batch,3]_canonical_nondecreasing"
ANCHOR_MIX_RULE = "raw=(1-alpha)*anchor_coordinates+alpha*learned_raw"
ANCHOR_MIX_PARAMETERIZATION = (
    "alpha=sigmoid(global_trainable_scalar_anchor_mix_logit)"
)
ANCHOR_MIX_INITIAL_WEIGHT = 0.1


@dataclass(frozen=True)
class EndpointAwareCalibratorOutput:
    """Hard-forward RC5 threshold curve and its auditable latent values."""

    pixel_budget_grid: torch.Tensor
    anchor_coordinates: torch.Tensor
    anchor_mix_weight: torch.Tensor
    grid_learned_raw_coordinates: torch.Tensor
    grid_raw_coordinates: torch.Tensor
    grid_coordinates: torch.Tensor
    grid_thresholds: torch.Tensor
    requested_pixel_budgets: torch.Tensor | None = None
    requested_raw_coordinates: torch.Tensor | None = None
    requested_coordinates: torch.Tensor | None = None
    requested_thresholds: torch.Tensor | None = None

    @property
    def grid_logits(self) -> torch.Tensor:
        """Trainer migration alias for ``grid_coordinates``."""

        return self.grid_coordinates

    @property
    def requested_logits(self) -> torch.Tensor | None:
        """Trainer migration alias for ``requested_coordinates``."""

        return self.requested_coordinates


def _validate_common(
    *,
    context_feature_dim: int,
    pixel_budget_grid: Sequence[float],
    hidden_dims: Sequence[int],
    dropout: float,
) -> tuple[tuple[float, ...], tuple[int, ...], float]:
    if (
        isinstance(context_feature_dim, bool)
        or not isinstance(context_feature_dim, int)
        or context_feature_dim <= 0
    ):
        raise ValueError("context_feature_dim must be a positive integer")
    if isinstance(pixel_budget_grid, (str, bytes)):
        raise TypeError("pixel_budget_grid must be a numeric sequence")
    try:
        budgets = tuple(float(value) for value in pixel_budget_grid)
    except (TypeError, ValueError) as error:
        raise ValueError("pixel_budget_grid must be a numeric sequence") from error
    if len(budgets) != 3:
        raise ValueError("RC5 pixel_budget_grid must contain exactly three values")
    if not all(math.isfinite(value) and value > 0.0 for value in budgets):
        raise ValueError("pixel_budget_grid must contain finite positive values")
    if not all(loose > strict for loose, strict in zip(budgets, budgets[1:])):
        raise ValueError("pixel_budget_grid must be strictly descending")

    if isinstance(hidden_dims, (str, bytes)):
        raise TypeError("hidden_dims must be an integer sequence")
    widths: list[int] = []
    for value in hidden_dims:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("hidden_dims must contain positive integers")
        widths.append(value)
    if len(widths) != 1:
        raise ValueError("RC5 calibrators require exactly one hidden layer")
    if isinstance(dropout, bool):
        raise TypeError("dropout must be a real number, not bool")
    dropout_value = float(dropout)
    if not math.isfinite(dropout_value) or not 0.0 <= dropout_value < 1.0:
        raise ValueError("dropout must be finite and lie in [0, 1)")
    return budgets, tuple(widths), dropout_value


class _EndpointAwareBase(nn.Module):
    budget_scope = PIXEL_RISK_NO_REJECT_SCOPE
    supports_component_budget = False
    supports_reject = False
    supports_complete_budget_curve = True
    training_pipeline_integrated = True
    threshold_representation_schema = THRESHOLD_REPRESENTATION_SCHEMA

    def __init__(
        self,
        *,
        context_feature_dim: int,
        pixel_budget_grid: Sequence[float],
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        budgets, widths, dropout_value = _validate_common(
            context_feature_dim=context_feature_dim,
            pixel_budget_grid=pixel_budget_grid,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.context_feature_dim = context_feature_dim
        self.hidden_dims = widths
        self.dropout = dropout_value
        self.raw_coordinate_min = RAW_COORDINATE_MIN
        self.raw_coordinate_max = RAW_COORDINATE_MAX
        initial_mix_logit = math.log(
            ANCHOR_MIX_INITIAL_WEIGHT / (1.0 - ANCHOR_MIX_INITIAL_WEIGHT)
        )
        self.anchor_mix_logit = nn.Parameter(
            torch.tensor(initial_mix_logit, dtype=torch.float64)
        )
        self.register_buffer(
            "pixel_budget_grid", torch.tensor(budgets, dtype=torch.float64)
        )
        self.register_buffer(
            "log_pixel_budget_grid", torch.log10(self.pixel_budget_grid)
        )
        width = widths[0]
        self.encoder = nn.Sequential(
            nn.Linear(context_feature_dim, width),
            nn.GELU(),
            nn.Dropout(dropout_value),
        )

    @property
    def num_pixel_budgets(self) -> int:
        return int(self.pixel_budget_grid.numel())

    def _validate_features(self, context_features: torch.Tensor) -> None:
        if not isinstance(context_features, torch.Tensor):
            raise TypeError("context_features must be a torch.Tensor")
        if (
            context_features.ndim != 2
            or context_features.shape[1] != self.context_feature_dim
        ):
            raise ValueError(
                "context_features must have shape "
                f"[B, {self.context_feature_dim}], got {tuple(context_features.shape)}"
            )
        if not context_features.is_floating_point():
            raise TypeError("context_features must be floating point")
        if not bool(torch.isfinite(context_features).all().item()):
            raise ValueError("context_features must be finite")

    def _prepare_anchor_coordinates(
        self,
        anchor_coordinates: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(anchor_coordinates, torch.Tensor):
            raise TypeError("anchor_coordinates must be a torch.Tensor")
        if anchor_coordinates.dtype != torch.float64:
            raise TypeError("anchor_coordinates must be a float64 tensor")
        if anchor_coordinates.shape != (batch_size, 3):
            raise ValueError(
                "anchor_coordinates must have shape "
                f"[{batch_size}, 3], got {tuple(anchor_coordinates.shape)}"
            )
        values = anchor_coordinates.to(device=device)
        # The decoder is also the shared exact canonical-coordinate validator.
        decode_coordinate_torch(values)
        if bool((values[:, 1:] < values[:, :-1]).any().item()):
            raise ValueError("anchor_coordinates must be nondecreasing")
        return values

    def _mix_with_anchor(
        self,
        learned_raw: torch.Tensor,
        anchor_coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if learned_raw.shape != anchor_coordinates.shape:
            raise RuntimeError("learned and anchor coordinate shapes differ")
        mix_logit = self.anchor_mix_logit.to(
            device=learned_raw.device, dtype=torch.float64
        )
        if not bool(torch.isfinite(mix_logit).item()):
            raise FloatingPointError("anchor_mix_logit is non-finite")
        alpha = torch.sigmoid(mix_logit)
        raw = (1.0 - alpha) * anchor_coordinates + alpha * learned_raw
        if not bool(torch.isfinite(raw).all().item()):
            raise FloatingPointError("anchor mixing produced non-finite coordinates")
        if bool(
            (
                (raw <= self.raw_coordinate_min)
                | (raw >= self.raw_coordinate_max)
            )
            .any()
            .item()
        ):
            raise FloatingPointError(
                "anchor-mixed raw coordinates must remain strictly inside bounds"
            )
        return raw, alpha

    def _prepare_requests(
        self, request: torch.Tensor, *, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if not isinstance(request, torch.Tensor):
            raise TypeError("pixel_budgets must be a torch.Tensor")
        values = request.to(device=device, dtype=torch.float64)
        if values.ndim == 0:
            values = values.reshape(1, 1).expand(batch_size, 1)
        elif values.ndim == 1:
            if values.numel() == 0:
                raise ValueError("pixel_budgets must not be empty")
            values = values.reshape(1, -1).expand(batch_size, -1)
        elif values.ndim == 2 and values.shape[0] in (1, batch_size):
            if values.shape[1] == 0:
                raise ValueError("pixel_budgets must not be empty")
            if values.shape[0] == 1:
                values = values.expand(batch_size, -1)
        else:
            raise ValueError("pixel_budgets must have shape [Q], [1,Q], or [B,Q]")
        if not bool(torch.isfinite(values).all().item()) or not bool(
            (values > 0.0).all().item()
        ):
            raise ValueError("requested pixel budgets must be finite and positive")
        lower = self.pixel_budget_grid[-1].to(device=device)
        upper = self.pixel_budget_grid[0].to(device=device)
        tolerance = (
            torch.finfo(request.dtype).eps * 16.0
            if request.is_floating_point()
            else 0.0
        )
        outside = (values < lower * (1.0 - tolerance)) | (
            values > upper * (1.0 + tolerance)
        )
        if bool(outside.any().item()):
            raise ValueError("requested budgets must stay inside the trained grid")
        return values.clamp(min=lower, max=upper)

    def _interpolate_raw(
        self, raw_grid: torch.Tensor, request: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = self._prepare_requests(
            request,
            batch_size=raw_grid.shape[0],
            device=raw_grid.device,
        )
        ascending_budget = torch.flip(
            self.log_pixel_budget_grid.to(device=raw_grid.device), dims=[0]
        )
        descending_raw = torch.flip(raw_grid, dims=[1])
        query = torch.log10(values)
        right = torch.searchsorted(ascending_budget, query, right=True).clamp(
            1, self.num_pixel_budgets - 1
        )
        left = right - 1
        batch_index = torch.arange(raw_grid.shape[0], device=raw_grid.device)[:, None]
        x0 = ascending_budget[left]
        x1 = ascending_budget[right]
        y0 = descending_raw[batch_index, left]
        y1 = descending_raw[batch_index, right]
        weight = (query - x0) / (x1 - x0)
        return y0 + weight * (y1 - y0), values

    def _make_output(
        self,
        raw: torch.Tensor,
        *,
        learned_raw: torch.Tensor,
        anchor_coordinates: torch.Tensor,
        anchor_mix_weight: torch.Tensor,
        pixel_budgets: torch.Tensor | None,
        allow_interpolation: bool,
    ) -> EndpointAwareCalibratorOutput:
        coordinates = canonicalize_raw_torch(raw)
        thresholds = decode_coordinate_torch(coordinates)
        if pixel_budgets is None:
            return EndpointAwareCalibratorOutput(
                pixel_budget_grid=self.pixel_budget_grid,
                anchor_coordinates=anchor_coordinates,
                anchor_mix_weight=anchor_mix_weight,
                grid_learned_raw_coordinates=learned_raw,
                grid_raw_coordinates=raw,
                grid_coordinates=coordinates,
                grid_thresholds=thresholds,
            )
        if allow_interpolation:
            requested_raw, normalized = self._interpolate_raw(raw, pixel_budgets)
        else:
            normalized = self._prepare_requests(
                pixel_budgets,
                batch_size=raw.shape[0],
                device=raw.device,
            )
            expected = self.pixel_budget_grid.to(device=raw.device).reshape(1, -1)
            if normalized.shape != (raw.shape[0], self.num_pixel_budgets) or not torch.equal(
                normalized, expected.expand(raw.shape[0], -1)
            ):
                raise ValueError("direct calibrator accepts only its complete trained grid")
            requested_raw = raw
        requested_coordinates = canonicalize_raw_torch(requested_raw)
        requested_thresholds = decode_coordinate_torch(requested_coordinates)
        return EndpointAwareCalibratorOutput(
            pixel_budget_grid=self.pixel_budget_grid,
            anchor_coordinates=anchor_coordinates,
            anchor_mix_weight=anchor_mix_weight,
            grid_learned_raw_coordinates=learned_raw,
            grid_raw_coordinates=raw,
            grid_coordinates=coordinates,
            grid_thresholds=thresholds,
            requested_pixel_budgets=normalized,
            requested_raw_coordinates=requested_raw,
            requested_coordinates=requested_coordinates,
            requested_thresholds=requested_thresholds,
        )

    def _common_export_config(self) -> dict[str, object]:
        return {
            "context_feature_dim": self.context_feature_dim,
            "pixel_budget_grid": self.pixel_budget_grid.detach().cpu().tolist(),
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "raw_coordinate_min_hex": self.raw_coordinate_min.hex(),
            "raw_coordinate_max_hex": self.raw_coordinate_max.hex(),
            "threshold_representation_schema": self.threshold_representation_schema,
            "anchor_source": T4_ANCHOR_SOURCE,
            "anchor_coordinate_contract": ANCHOR_COORDINATE_CONTRACT,
            "anchor_mix_rule": ANCHOR_MIX_RULE,
            "anchor_mix_parameterization": ANCHOR_MIX_PARAMETERIZATION,
            "anchor_mix_initial_weight": ANCHOR_MIX_INITIAL_WEIGHT,
        }

    def _common_capability(self) -> dict[str, object]:
        return {
            "stage": "stage2_final_no_reject",
            "budget_scope": self.budget_scope,
            "budget_axis": "pixel_false_alarm_rate",
            "supports_component_budget": False,
            "supports_reject": False,
            "supports_complete_budget_curve": True,
            "curve_output_shape": "[batch,J]",
            "training_pipeline_integrated": True,
            "requires_anchor_coordinates": True,
            "anchor_source": T4_ANCHOR_SOURCE,
            "anchor_coordinate_contract": ANCHOR_COORDINATE_CONTRACT,
            "anchor_mix_rule": ANCHOR_MIX_RULE,
            "anchor_mix_parameterization": ANCHOR_MIX_PARAMETERIZATION,
            "anchor_mix_initial_weight": ANCHOR_MIX_INITIAL_WEIGHT,
            "threshold_representation": representation_contract(),
            "threshold_semantics": "prediction = probability > threshold",
            "risk_guarantee": "empirical_not_certified",
        }


class DirectEndpointAwarePixelCalibrator(_EndpointAwareBase):
    """T6 control: independent endpoint-aware coordinate per budget."""

    model_id = DIRECT_ENDPOINT_AWARE_MODEL_ID
    supports_query_risk_aligned_loss = False
    structural_monotonicity = False

    def __init__(
        self,
        context_feature_dim: int,
        pixel_budget_grid: Sequence[float],
        hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            context_feature_dim=context_feature_dim,
            pixel_budget_grid=pixel_budget_grid,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.coordinate_head = nn.Linear(self.hidden_dims[0], self.num_pixel_budgets)

    def forward(
        self,
        context_features: torch.Tensor,
        *,
        anchor_coordinates: torch.Tensor,
        pixel_budgets: torch.Tensor | None = None,
    ) -> EndpointAwareCalibratorOutput:
        self._validate_features(context_features)
        anchor = self._prepare_anchor_coordinates(
            anchor_coordinates,
            batch_size=context_features.shape[0],
            device=context_features.device,
        )
        unconstrained = self.coordinate_head(self.encoder(context_features))
        learned_raw = self.raw_coordinate_min + torch.sigmoid(
            unconstrained.to(torch.float64)
        ) * (self.raw_coordinate_max - self.raw_coordinate_min)
        if not bool(torch.isfinite(learned_raw).all().item()):
            raise FloatingPointError("direct calibrator produced non-finite coordinates")
        raw, alpha = self._mix_with_anchor(learned_raw, anchor)
        return self._make_output(
            raw,
            learned_raw=learned_raw,
            anchor_coordinates=anchor,
            anchor_mix_weight=alpha,
            pixel_budgets=pixel_budgets,
            allow_interpolation=False,
        )

    def export_config(self) -> dict[str, object]:
        return self._common_export_config()

    def capability_contract(self) -> dict[str, object]:
        return {
            **self._common_capability(),
            "model_id": self.model_id,
            "supports_query_risk_aligned_loss": False,
            "structural_monotonicity": False,
            "training_objective": (
                "T6_T4_anchor_mixed_endpoint_aware_coordinate_huber_only"
            ),
        }


class MonotoneEndpointAwarePixelCalibrator(_EndpointAwareBase):
    """T7/T8 architecture: strictly ordered raw coordinates, exact endpoint."""

    model_id = MONOTONE_ENDPOINT_AWARE_MODEL_ID
    supports_query_risk_aligned_loss = True
    structural_monotonicity = True

    def __init__(
        self,
        context_feature_dim: int,
        pixel_budget_grid: Sequence[float],
        hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.1,
        minimum_raw_coordinate_gap: float = 1e-3,
    ) -> None:
        super().__init__(
            context_feature_dim=context_feature_dim,
            pixel_budget_grid=pixel_budget_grid,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        if isinstance(minimum_raw_coordinate_gap, bool):
            raise TypeError("minimum_raw_coordinate_gap must be a real number")
        gap = float(minimum_raw_coordinate_gap)
        span = self.raw_coordinate_max - self.raw_coordinate_min
        if not math.isfinite(gap) or gap <= 0.0:
            raise ValueError("minimum_raw_coordinate_gap must be finite and positive")
        if self.num_pixel_budgets * gap >= span:
            raise ValueError("minimum_raw_coordinate_gap leaves no free span")
        self.minimum_raw_coordinate_gap = gap
        self.spacing_head = nn.Linear(
            self.hidden_dims[0], self.num_pixel_budgets + 1
        )

    def _ordered_raw(self, hidden: torch.Tensor) -> torch.Tensor:
        interval_weights = F.softmax(
            self.spacing_head(hidden).to(torch.float64), dim=1
        )
        cumulative = torch.cumsum(interval_weights[:, :-1], dim=1)
        grid_index = torch.arange(
            1,
            self.num_pixel_budgets + 1,
            dtype=torch.float64,
            device=hidden.device,
        )[None, :]
        free_span = (
            self.raw_coordinate_max
            - self.raw_coordinate_min
            - self.num_pixel_budgets * self.minimum_raw_coordinate_gap
        )
        raw = (
            self.raw_coordinate_min
            + grid_index * self.minimum_raw_coordinate_gap
            + cumulative * free_span
        )
        if not bool(torch.isfinite(raw).all().item()) or not bool(
            (raw[:, 1:] > raw[:, :-1]).all().item()
        ):
            raise FloatingPointError("numeric precision broke raw-coordinate order")
        return raw

    def forward(
        self,
        context_features: torch.Tensor,
        *,
        anchor_coordinates: torch.Tensor,
        pixel_budgets: torch.Tensor | None = None,
    ) -> EndpointAwareCalibratorOutput:
        self._validate_features(context_features)
        anchor = self._prepare_anchor_coordinates(
            anchor_coordinates,
            batch_size=context_features.shape[0],
            device=context_features.device,
        )
        learned_raw = self._ordered_raw(self.encoder(context_features))
        raw, alpha = self._mix_with_anchor(learned_raw, anchor)
        if not bool((raw[:, 1:] > raw[:, :-1]).all().item()):
            raise FloatingPointError("anchor mixing broke strict raw-coordinate order")
        output = self._make_output(
            raw,
            learned_raw=learned_raw,
            anchor_coordinates=anchor,
            anchor_mix_weight=alpha,
            pixel_budgets=pixel_budgets,
            allow_interpolation=True,
        )
        if not bool(
            (output.grid_coordinates[:, 1:] >= output.grid_coordinates[:, :-1])
            .all()
            .item()
        ):
            raise FloatingPointError("canonical coordinate curve decreased")
        endpoint = output.grid_coordinates == UPPER_ENDPOINT_COORDINATE
        if bool((endpoint[:, :-1] & ~endpoint[:, 1:]).any().item()):
            raise FloatingPointError("upper endpoints are not suffix closed")
        return output

    def export_config(self) -> dict[str, object]:
        return {
            **self._common_export_config(),
            "minimum_raw_coordinate_gap": self.minimum_raw_coordinate_gap,
        }

    def capability_contract(self) -> dict[str, object]:
        return {
            **self._common_capability(),
            "model_id": self.model_id,
            "supports_query_risk_aligned_loss": True,
            "structural_monotonicity": True,
            "raw_coordinate_order": "strictly_increasing",
            "decoded_threshold_order": "nondecreasing",
            "upper_endpoint_decisions": "suffix_closed",
            "budget_interpolation": "raw_coordinate_piecewise_linear_log10_no_extrapolation",
            "training_objective": (
                "T7_T4_anchor_mixed_endpoint_aware_coordinate_huber_or_"
                "T8_T4_anchor_mixed_exact_curve_risk_aligned"
            ),
        }


__all__ = [
    "ANCHOR_COORDINATE_CONTRACT",
    "ANCHOR_MIX_INITIAL_WEIGHT",
    "ANCHOR_MIX_PARAMETERIZATION",
    "ANCHOR_MIX_RULE",
    "DIRECT_ENDPOINT_AWARE_MODEL_ID",
    "MONOTONE_ENDPOINT_AWARE_MODEL_ID",
    "DirectEndpointAwarePixelCalibrator",
    "EndpointAwareCalibratorOutput",
    "MonotoneEndpointAwarePixelCalibrator",
    "T4_ANCHOR_SOURCE",
]

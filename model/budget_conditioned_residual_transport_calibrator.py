"""Result-free RC5+ context-anchored residual-transport calibrators.

The learned object is a continuous operating-point function over the frozen
exact-rational false-alarm budget lattice.  Unlike a free threshold regressor,
the model transports the exact same-budget target-context order-statistic
anchor in a finite EATC-v2 latent space and learns only a source-supervised
context residual.  T6+ and T7+/T8+ have identical parameter counts: T6+ uses
signed interval increments, while T7+/T8+ uses positive increments and is
monotone for every valid ordered rational query by construction.

This module is additive and result-free.  Admission into checkpoint, cyclic
training and sealed inference is governed by separate RC5+ contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.budget_conditioned_endpoint_calibrator import (
    BUDGET_AXIS_TRANSFORM,
    BUDGET_INTERPOLATION,
    BUDGET_KNOT_RATIONALS,
    PRIMARY_BUDGET_KNOT_INDICES,
)
from model.endpoint_aware_pixel_calibrator import (
    ANCHOR_MIX_INITIAL_WEIGHT,
    T4_ANCHOR_SOURCE,
)
from model.endpoint_aware_threshold import (
    RAW_COORDINATE_MAX,
    RAW_COORDINATE_MIN,
    THRESHOLD_REPRESENTATION_SCHEMA,
    UPPER_ENDPOINT_COORDINATE,
    canonicalize_raw_torch,
    decode_coordinate_torch,
    representation_contract,
)


RESIDUAL_TRANSPORT_SCHEMA = (
    "rc-irstd.budget-conditioned-anchor-residual-transport-candidate.v2"
)
RESIDUAL_TRANSPORT_DIRECT_MODEL_ID = (
    "budget_conditioned_direct_anchor_residual_transport_calibrator"
)
RESIDUAL_TRANSPORT_MONOTONE_MODEL_ID = (
    "budget_conditioned_monotone_anchor_residual_transport_calibrator"
)
RESIDUAL_TRANSPORT_NO_ANCHOR_MODEL_ID = (
    "budget_conditioned_monotone_no_target_anchor_residual_calibrator"
)
RESIDUAL_TRANSPORT_RULE = (
    "z=exp(alpha*tanh(q_beta))*logit((anchor-r_min)/(r_max-r_min))"
    "+alpha*G_context(log_budget); raw=r_min+(r_max-r_min)*sigmoid(z)"
)
RESIDUAL_TRANSPORT_MONOTONE_RULE = (
    "G_context_piecewise_linear_with_strictly_positive_interval_increments"
)


class ResidualTransportCalibratorError(ValueError):
    """A residual-transport model or rational request violates its contract."""


@dataclass(frozen=True)
class ResidualTransportCalibratorOutput:
    budget_knot_numerators: torch.Tensor
    budget_knot_denominators: torch.Tensor
    budget_knot_values: torch.Tensor
    budget_knot_positions: torch.Tensor
    anchor_coordinates: torch.Tensor
    correction_strength: torch.Tensor
    anchor_slope: torch.Tensor
    grid_anchor_latent: torch.Tensor
    grid_residual: torch.Tensor
    grid_transport_latent: torch.Tensor
    grid_raw_coordinates: torch.Tensor
    grid_coordinates: torch.Tensor
    grid_thresholds: torch.Tensor
    requested_budget_numerators: torch.Tensor | None = None
    requested_budget_denominators: torch.Tensor | None = None
    requested_budget_positions: torch.Tensor | None = None
    requested_anchor_coordinates: torch.Tensor | None = None
    requested_anchor_latent: torch.Tensor | None = None
    requested_residual: torch.Tensor | None = None
    requested_transport_latent: torch.Tensor | None = None
    requested_raw_coordinates: torch.Tensor | None = None
    requested_coordinates: torch.Tensor | None = None
    requested_thresholds: torch.Tensor | None = None


@dataclass(frozen=True)
class AnchorFreeResidualCalibratorOutput:
    budget_knot_numerators: torch.Tensor
    budget_knot_denominators: torch.Tensor
    budget_knot_values: torch.Tensor
    budget_knot_positions: torch.Tensor
    correction_strength: torch.Tensor
    context_scale: torch.Tensor
    grid_residual: torch.Tensor
    grid_transport_latent: torch.Tensor
    grid_raw_coordinates: torch.Tensor
    grid_coordinates: torch.Tensor
    grid_thresholds: torch.Tensor
    requested_budget_numerators: torch.Tensor | None = None
    requested_budget_denominators: torch.Tensor | None = None
    requested_budget_positions: torch.Tensor | None = None
    requested_residual: torch.Tensor | None = None
    requested_transport_latent: torch.Tensor | None = None
    requested_raw_coordinates: torch.Tensor | None = None
    requested_coordinates: torch.Tensor | None = None
    requested_thresholds: torch.Tensor | None = None


def _hidden_dims(value: Sequence[int]) -> tuple[int]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 1
        or type(value[0]) is not int
        or value[0] <= 0
    ):
        raise ResidualTransportCalibratorError(
            "residual transport requires exactly one positive hidden width"
        )
    return (value[0],)


class _ResidualTransportBase(nn.Module):
    supports_reject = False
    supports_complete_budget_curve = True
    supports_exact_rational_budget_requests = True
    threshold_representation_schema = THRESHOLD_REPRESENTATION_SCHEMA
    structural_monotonicity: bool
    model_id: str

    def __init__(
        self,
        *,
        context_feature_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        minimum_residual_increment: float,
    ) -> None:
        super().__init__()
        if type(context_feature_dim) is not int or context_feature_dim <= 0:
            raise ResidualTransportCalibratorError(
                "context_feature_dim must be a positive integer"
            )
        widths = _hidden_dims(hidden_dims)
        if isinstance(dropout, bool):
            raise TypeError("dropout must be a real number")
        dropout_value = float(dropout)
        if not math.isfinite(dropout_value) or not 0.0 <= dropout_value < 1.0:
            raise ResidualTransportCalibratorError(
                "dropout must be finite and lie in [0,1)"
            )
        if isinstance(minimum_residual_increment, bool):
            raise TypeError("minimum_residual_increment must be a real number")
        minimum = float(minimum_residual_increment)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ResidualTransportCalibratorError(
                "minimum_residual_increment must be finite and positive"
            )

        numerators = torch.tensor(
            [row[0] for row in BUDGET_KNOT_RATIONALS], dtype=torch.int64
        )
        denominators = torch.tensor(
            [row[1] for row in BUDGET_KNOT_RATIONALS], dtype=torch.int64
        )
        values = numerators.to(torch.float64) / denominators.to(torch.float64)
        logs = torch.log(numerators.to(torch.float64)) - torch.log(
            denominators.to(torch.float64)
        )
        positions = (logs - logs[0]) / (logs[-1] - logs[0])
        if (
            positions[0].item() != 0.0
            or positions[-1].item() != 1.0
            or not bool((positions[1:] > positions[:-1]).all().item())
        ):
            raise RuntimeError("frozen rational budget positions are not strict")

        self.context_feature_dim = context_feature_dim
        self.hidden_dims = widths
        self.dropout = dropout_value
        self.minimum_residual_increment = minimum
        self.budget_knot_rationals = BUDGET_KNOT_RATIONALS
        self.register_buffer("budget_knot_numerators", numerators)
        self.register_buffer("budget_knot_denominators", denominators)
        self.register_buffer("budget_knot_values", values)
        self.register_buffer("budget_knot_positions", positions)
        initial = math.log(
            ANCHOR_MIX_INITIAL_WEIGHT / (1.0 - ANCHOR_MIX_INITIAL_WEIGHT)
        )
        self.correction_strength_logit = nn.Parameter(
            torch.tensor(initial, dtype=torch.float64)
        )
        self.encoder = nn.Sequential(
            nn.Linear(context_feature_dim, widths[0]),
            nn.GELU(),
            nn.Dropout(dropout_value),
        )
        # One residual origin, one anchor-slope modulation and K-1 interval
        # increments.  Direct and monotone variants therefore have exactly
        # the same trainable capacity.
        self.transport_head = nn.Linear(widths[0], len(BUDGET_KNOT_RATIONALS) + 1)

    @property
    def num_budget_knots(self) -> int:
        return len(self.budget_knot_rationals)

    def _features(self, value: Any) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.ndim != 2
            or value.shape[1] != self.context_feature_dim
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ResidualTransportCalibratorError(
                f"context features must be finite float32[B,{self.context_feature_dim}]"
            )
        return value

    def _anchor(
        self,
        value: Any,
        *,
        batch_size: int,
        width: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float64
            or value.shape != (batch_size, width)
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ResidualTransportCalibratorError(
                f"{name} must be finite float64[{batch_size},{width}]"
            )
        result = value.to(device=device)
        try:
            decode_coordinate_torch(result)
        except ValueError as error:
            raise ResidualTransportCalibratorError(
                f"{name} contains a noncanonical EATC coordinate"
            ) from error
        if width > 1 and bool((result[:, 1:] < result[:, :-1]).any().item()):
            raise ResidualTransportCalibratorError(
                f"{name} must be nondecreasing from loose to strict budget"
            )
        return result

    @staticmethod
    def _anchor_latent(anchor: torch.Tensor) -> torch.Tensor:
        span = RAW_COORDINATE_MAX - RAW_COORDINATE_MIN
        normalized = (anchor - RAW_COORDINATE_MIN) / span
        if bool(((normalized <= 0.0) | (normalized >= 1.0)).any().item()):
            raise ResidualTransportCalibratorError(
                "canonical anchor must lie strictly inside transport bounds"
            )
        latent = torch.log(normalized) - torch.log1p(-normalized)
        if not bool(torch.isfinite(latent).all().item()):
            raise FloatingPointError("anchor transport latent is non-finite")
        return latent

    def _request_positions(
        self,
        numerators: Any,
        denominators: Any,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            not isinstance(numerators, torch.Tensor)
            or not isinstance(denominators, torch.Tensor)
            or numerators.dtype != torch.int64
            or denominators.dtype != torch.int64
            or numerators.shape != denominators.shape
        ):
            raise TypeError(
                "budget requests must be aligned int64 numerator/denominator tensors"
            )
        n = numerators.detach().to(device="cpu")
        d = denominators.detach().to(device="cpu")
        if n.ndim == 1 and n.numel() > 0:
            n = n.reshape(1, -1).expand(batch_size, -1).clone()
            d = d.reshape(1, -1).expand(batch_size, -1).clone()
        elif n.ndim == 2 and n.shape[0] in (1, batch_size) and n.shape[1] > 0:
            if n.shape[0] == 1:
                n = n.expand(batch_size, -1).clone()
                d = d.expand(batch_size, -1).clone()
        else:
            raise ResidualTransportCalibratorError(
                "budget requests must have shape [Q], [1,Q], or [B,Q]"
            )
        loose = Fraction(*BUDGET_KNOT_RATIONALS[0])
        strict = Fraction(*BUDGET_KNOT_RATIONALS[-1])
        for row_n, row_d in zip(n.tolist(), d.tolist(), strict=True):
            previous: Fraction | None = None
            for numerator, denominator in zip(row_n, row_d, strict=True):
                if numerator <= 0 or denominator <= numerator:
                    raise ResidualTransportCalibratorError(
                        "requested budgets must lie strictly inside (0,1)"
                    )
                fraction = Fraction(numerator, denominator)
                if (fraction.numerator, fraction.denominator) != (
                    numerator,
                    denominator,
                ):
                    raise ResidualTransportCalibratorError(
                        "requested budgets must be lowest-term rationals"
                    )
                if not strict <= fraction <= loose:
                    raise ResidualTransportCalibratorError(
                        "requested budgets must stay inside the trained knot range"
                    )
                if previous is not None and not previous > fraction:
                    raise ResidualTransportCalibratorError(
                        "requested budgets must descend strictly from loose to strict"
                    )
                previous = fraction
        n_device = n.to(device=device)
        d_device = d.to(device=device)
        log_values = torch.log(n_device.to(torch.float64)) - torch.log(
            d_device.to(torch.float64)
        )
        knot_logs = torch.log(
            self.budget_knot_numerators.to(device=device, dtype=torch.float64)
        ) - torch.log(
            self.budget_knot_denominators.to(device=device, dtype=torch.float64)
        )
        positions = (log_values - knot_logs[0]) / (knot_logs[-1] - knot_logs[0])
        if (
            not bool(torch.isfinite(positions).all().item())
            or (
                positions.shape[1] > 1
                and not bool((positions[:, 1:] > positions[:, :-1]).all().item())
            )
        ):
            raise ResidualTransportCalibratorError(
                "distinct rational requests collide in float64 log-budget space"
            )
        return n_device, d_device, positions

    def _residual_grid(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parameters = self.transport_head(hidden).to(torch.float64)
        alpha = torch.sigmoid(
            self.correction_strength_logit.to(
                device=hidden.device, dtype=torch.float64
            )
        )
        if not 0.0 < float(alpha.item()) < 1.0:
            raise FloatingPointError("correction strength must remain in (0,1)")
        beta = torch.exp(alpha * torch.tanh(parameters[:, 1:2]))
        increments = self._interval_increments(parameters[:, 2:])
        origin = parameters[:, :1]
        residual = torch.cat(
            (origin, origin + torch.cumsum(increments, dim=1)), dim=1
        )
        if residual.shape[1] != self.num_budget_knots:
            raise RuntimeError("residual grid width does not match budget lattice")
        if not bool(torch.isfinite(residual).all().item()):
            raise FloatingPointError("context residual grid is non-finite")
        return residual, beta, alpha

    def _interval_increments(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _interpolate_residual(
        self,
        residual_grid: torch.Tensor,
        positions: torch.Tensor,
        numerators: torch.Tensor,
        denominators: torch.Tensor,
    ) -> torch.Tensor:
        knots = self.budget_knot_positions.to(
            device=residual_grid.device, dtype=torch.float64
        )
        right = torch.searchsorted(knots, positions, right=True).clamp(
            1, self.num_budget_knots - 1
        )
        left = right - 1
        batch = torch.arange(residual_grid.shape[0], device=residual_grid.device)[:, None]
        x0 = knots[left]
        x1 = knots[right]
        y0 = residual_grid[batch, left]
        y1 = residual_grid[batch, right]
        result = y0 + (positions - x0) / (x1 - x0) * (y1 - y0)
        knot_n = self.budget_knot_numerators.to(device=residual_grid.device)
        knot_d = self.budget_knot_denominators.to(device=residual_grid.device)
        exact = (numerators[:, :, None] == knot_n[None, None, :]) & (
            denominators[:, :, None] == knot_d[None, None, :]
        )
        exact_index = exact.to(torch.int64).argmax(dim=2)
        result = torch.where(
            exact.any(dim=2), residual_grid.gather(1, exact_index), result
        )
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("interpolated context residual is non-finite")
        return result

    @staticmethod
    def _transport(
        anchor_latent: torch.Tensor,
        residual: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = beta * anchor_latent + alpha * residual
        span = RAW_COORDINATE_MAX - RAW_COORDINATE_MIN
        raw = RAW_COORDINATE_MIN + span * torch.sigmoid(latent)
        if (
            not bool(torch.isfinite(raw).all().item())
            or bool(((raw < RAW_COORDINATE_MIN) | (raw > RAW_COORDINATE_MAX)).any().item())
        ):
            raise FloatingPointError("residual transport left its finite bounds")
        coordinates = canonicalize_raw_torch(raw)
        thresholds = decode_coordinate_torch(coordinates)
        return latent, raw, coordinates, thresholds

    @staticmethod
    def _assert_monotone(
        residual: torch.Tensor,
        latent: torch.Tensor,
        raw: torch.Tensor,
        coordinates: torch.Tensor,
        thresholds: torch.Tensor,
        *,
        name: str,
    ) -> None:
        # Positive mathematical increments can round to equality after a
        # large cumulative sum, and sigmoid can round distinct large latents
        # to the same finite raw coordinate.  The executable contract is
        # therefore order preservation (nondecrease), not an impossible
        # machine-level strictness claim.
        if not bool((residual[:, 1:] >= residual[:, :-1]).all().item()):
            raise FloatingPointError(f"{name} residual curve decreased")
        if not bool((latent[:, 1:] >= latent[:, :-1]).all().item()):
            raise FloatingPointError(f"{name} transport latent decreased")
        if not bool((raw[:, 1:] >= raw[:, :-1]).all().item()):
            raise FloatingPointError(f"{name} raw curve decreased")
        if not bool((coordinates[:, 1:] >= coordinates[:, :-1]).all().item()):
            raise FloatingPointError(f"{name} canonical curve decreased")
        if not bool((thresholds[:, 1:] >= thresholds[:, :-1]).all().item()):
            raise FloatingPointError(f"{name} threshold curve decreased")
        endpoint = coordinates == UPPER_ENDPOINT_COORDINATE
        if bool((endpoint[:, :-1] & ~endpoint[:, 1:]).any().item()):
            raise FloatingPointError(f"{name} endpoint decisions are not suffix closed")

    def forward(
        self,
        context_features: torch.Tensor,
        *,
        anchor_coordinates: torch.Tensor,
        budget_numerators: torch.Tensor | None = None,
        budget_denominators: torch.Tensor | None = None,
        requested_anchor_coordinates: torch.Tensor | None = None,
    ) -> ResidualTransportCalibratorOutput:
        features = self._features(context_features)
        anchor = self._anchor(
            anchor_coordinates,
            batch_size=features.shape[0],
            width=self.num_budget_knots,
            device=features.device,
            name="anchor_coordinates",
        )
        residual, beta, alpha = self._residual_grid(self.encoder(features))
        anchor_latent = self._anchor_latent(anchor)
        latent, raw, coordinates, thresholds = self._transport(
            anchor_latent, residual, beta, alpha
        )
        if self.structural_monotonicity:
            self._assert_monotone(
                residual, latent, raw, coordinates, thresholds, name="grid"
            )
        common = {
            "budget_knot_numerators": self.budget_knot_numerators,
            "budget_knot_denominators": self.budget_knot_denominators,
            "budget_knot_values": self.budget_knot_values,
            "budget_knot_positions": self.budget_knot_positions,
            "anchor_coordinates": anchor,
            "correction_strength": alpha,
            "anchor_slope": beta,
            "grid_anchor_latent": anchor_latent,
            "grid_residual": residual,
            "grid_transport_latent": latent,
            "grid_raw_coordinates": raw,
            "grid_coordinates": coordinates,
            "grid_thresholds": thresholds,
        }
        if budget_numerators is None and budget_denominators is None:
            if requested_anchor_coordinates is not None:
                raise ResidualTransportCalibratorError(
                    "requested anchor requires exact-rational budget requests"
                )
            return ResidualTransportCalibratorOutput(**common)
        if budget_numerators is None or budget_denominators is None:
            raise ResidualTransportCalibratorError(
                "requested budget numerator and denominator must appear together"
            )
        n, d, positions = self._request_positions(
            budget_numerators,
            budget_denominators,
            batch_size=features.shape[0],
            device=features.device,
        )
        requested_anchor = self._anchor(
            requested_anchor_coordinates,
            batch_size=features.shape[0],
            width=n.shape[1],
            device=features.device,
            name="requested_anchor_coordinates",
        )
        requested_anchor_latent = self._anchor_latent(requested_anchor)
        requested_residual = self._interpolate_residual(residual, positions, n, d)
        requested_latent, requested_raw, requested_coordinates, requested_thresholds = (
            self._transport(
                requested_anchor_latent,
                requested_residual,
                beta,
                alpha,
            )
        )
        if self.structural_monotonicity:
            self._assert_monotone(
                requested_residual,
                requested_latent,
                requested_raw,
                requested_coordinates,
                requested_thresholds,
                name="requested",
            )
        return ResidualTransportCalibratorOutput(
            **common,
            requested_budget_numerators=n,
            requested_budget_denominators=d,
            requested_budget_positions=positions,
            requested_anchor_coordinates=requested_anchor,
            requested_anchor_latent=requested_anchor_latent,
            requested_residual=requested_residual,
            requested_transport_latent=requested_latent,
            requested_raw_coordinates=requested_raw,
            requested_coordinates=requested_coordinates,
            requested_thresholds=requested_thresholds,
        )

    def export_config(self) -> dict[str, object]:
        return {
            "schema_version": RESIDUAL_TRANSPORT_SCHEMA,
            "model_id": self.model_id,
            "context_feature_dim": self.context_feature_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "budget_knot_rationals": [list(row) for row in BUDGET_KNOT_RATIONALS],
            "primary_budget_knot_indices": list(PRIMARY_BUDGET_KNOT_INDICES),
            "minimum_residual_increment": self.minimum_residual_increment,
            "budget_axis_transform": BUDGET_AXIS_TRANSFORM,
            "budget_interpolation": BUDGET_INTERPOLATION,
            "residual_transport_rule": RESIDUAL_TRANSPORT_RULE,
            "threshold_representation_schema": THRESHOLD_REPRESENTATION_SCHEMA,
            "anchor_source": T4_ANCHOR_SOURCE,
            "correction_strength_parameterization": "sigmoid_global_scalar",
            "correction_strength_initial_value": ANCHOR_MIX_INITIAL_WEIGHT,
            "threshold_semantics": "prediction = probability > threshold",
        }

    def capability_contract(self) -> dict[str, object]:
        return {
            **self.export_config(),
            "supports_reject": False,
            "supports_fallback": False,
            "supports_exact_rational_budget_requests": True,
            "requires_anchor_coordinates": True,
            "requested_anchor_semantics": (
                "exact_context_order_statistic_at_the_same_requested_rational_budget"
            ),
            "source_of_learned_correction": "source_oof_cyclic_training_only",
            "structural_monotonicity": self.structural_monotonicity,
            "risk_guarantee": "empirical_not_certified",
            "threshold_representation": representation_contract(),
        }


class BudgetConditionedDirectResidualTransportCalibrator(
    _ResidualTransportBase
):
    """Capacity-matched T6+ control with signed residual increments."""

    model_id = RESIDUAL_TRANSPORT_DIRECT_MODEL_ID
    structural_monotonicity = False

    def __init__(
        self,
        context_feature_dim: int,
        hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.1,
        minimum_residual_increment: float = 1e-6,
    ) -> None:
        super().__init__(
            context_feature_dim=context_feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            minimum_residual_increment=minimum_residual_increment,
        )

    def _interval_increments(self, value: torch.Tensor) -> torch.Tensor:
        return value


class BudgetConditionedMonotoneResidualTransportCalibrator(
    _ResidualTransportBase
):
    """T7+/T8+ same-budget analytic-to-learned monotone transport."""

    model_id = RESIDUAL_TRANSPORT_MONOTONE_MODEL_ID
    structural_monotonicity = True

    def __init__(
        self,
        context_feature_dim: int,
        hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.1,
        minimum_residual_increment: float = 1e-6,
    ) -> None:
        super().__init__(
            context_feature_dim=context_feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            minimum_residual_increment=minimum_residual_increment,
        )

    def _interval_increments(self, value: torch.Tensor) -> torch.Tensor:
        return F.softplus(value) + self.minimum_residual_increment

    def capability_contract(self) -> dict[str, object]:
        return {
            **super().capability_contract(),
            "residual_monotonicity_rule": RESIDUAL_TRANSPORT_MONOTONE_RULE,
            "mathematical_increment_order": "strictly_positive_before_rounding",
            "raw_coordinate_order": (
                "nondecreasing_in_finite_precision_for_every_valid_ordered_"
                "same-budget-anchor_query"
            ),
            "decoded_threshold_order": "nondecreasing_for_every_valid_query",
            "upper_endpoint_decisions": "suffix_closed",
        }


class BudgetConditionedMonotoneNoTargetAnchorCalibrator(
    BudgetConditionedMonotoneResidualTransportCalibrator
):
    """Capacity-matched T8+ ablation with no target-anchor argument or read.

    The encoder and positive-increment residual branch are identical to T8+.
    With the analytic target anchor removed, the learned residual itself is
    mapped through a positive context scale and the same global sigmoid scalar:
    ``z=alpha*beta*G_context(log_budget)``.  All 3339 parameters remain live.
    """

    model_id = RESIDUAL_TRANSPORT_NO_ANCHOR_MODEL_ID

    @staticmethod
    def _anchor_free_transport(
        residual: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = alpha * beta * residual
        span = RAW_COORDINATE_MAX - RAW_COORDINATE_MIN
        raw = RAW_COORDINATE_MIN + span * torch.sigmoid(latent)
        if (
            not bool(torch.isfinite(raw).all().item())
            or bool(
                ((raw < RAW_COORDINATE_MIN) | (raw > RAW_COORDINATE_MAX))
                .any()
                .item()
            )
        ):
            raise FloatingPointError("anchor-free residual curve left finite bounds")
        coordinates = canonicalize_raw_torch(raw)
        thresholds = decode_coordinate_torch(coordinates)
        return latent, raw, coordinates, thresholds

    def forward(
        self,
        context_features: torch.Tensor,
        *,
        budget_numerators: torch.Tensor | None = None,
        budget_denominators: torch.Tensor | None = None,
    ) -> AnchorFreeResidualCalibratorOutput:
        features = self._features(context_features)
        residual, beta, alpha = self._residual_grid(self.encoder(features))
        latent, raw, coordinates, thresholds = self._anchor_free_transport(
            residual, beta, alpha
        )
        self._assert_monotone(
            residual, latent, raw, coordinates, thresholds, name="anchor-free grid"
        )
        common = {
            "budget_knot_numerators": self.budget_knot_numerators,
            "budget_knot_denominators": self.budget_knot_denominators,
            "budget_knot_values": self.budget_knot_values,
            "budget_knot_positions": self.budget_knot_positions,
            "correction_strength": alpha,
            "context_scale": beta,
            "grid_residual": residual,
            "grid_transport_latent": latent,
            "grid_raw_coordinates": raw,
            "grid_coordinates": coordinates,
            "grid_thresholds": thresholds,
        }
        if budget_numerators is None and budget_denominators is None:
            return AnchorFreeResidualCalibratorOutput(**common)
        if budget_numerators is None or budget_denominators is None:
            raise ResidualTransportCalibratorError(
                "requested budget numerator and denominator must appear together"
            )
        n, d, positions = self._request_positions(
            budget_numerators,
            budget_denominators,
            batch_size=features.shape[0],
            device=features.device,
        )
        requested_residual = self._interpolate_residual(residual, positions, n, d)
        requested_latent, requested_raw, requested_coordinates, requested_thresholds = (
            self._anchor_free_transport(requested_residual, beta, alpha)
        )
        self._assert_monotone(
            requested_residual,
            requested_latent,
            requested_raw,
            requested_coordinates,
            requested_thresholds,
            name="anchor-free requested",
        )
        return AnchorFreeResidualCalibratorOutput(
            **common,
            requested_budget_numerators=n,
            requested_budget_denominators=d,
            requested_budget_positions=positions,
            requested_residual=requested_residual,
            requested_transport_latent=requested_latent,
            requested_raw_coordinates=requested_raw,
            requested_coordinates=requested_coordinates,
            requested_thresholds=requested_thresholds,
        )

    def export_config(self) -> dict[str, object]:
        config = super().export_config()
        config.update(
            {
                "anchor_source": "none_target_anchor_ablation",
                "residual_transport_rule": (
                    "z=alpha*exp(alpha*tanh(q_beta))*G_context(log_budget);"
                    "raw=r_min+(r_max-r_min)*sigmoid(z)"
                ),
                "ablation_identity": "T8_PLUS_NO_ANCHOR",
            }
        )
        return config

    def capability_contract(self) -> dict[str, object]:
        contract = super().capability_contract()
        contract.update(
            {
                "requires_anchor_coordinates": False,
                "requested_anchor_semantics": "not_applicable_no_target_anchor",
                "target_anchor_accessed": False,
                "analytic_anchor_ablation": True,
            }
        )
        return contract


__all__ = [
    "RESIDUAL_TRANSPORT_DIRECT_MODEL_ID",
    "RESIDUAL_TRANSPORT_MONOTONE_MODEL_ID",
    "RESIDUAL_TRANSPORT_NO_ANCHOR_MODEL_ID",
    "RESIDUAL_TRANSPORT_MONOTONE_RULE",
    "RESIDUAL_TRANSPORT_RULE",
    "RESIDUAL_TRANSPORT_SCHEMA",
    "BudgetConditionedDirectResidualTransportCalibrator",
    "BudgetConditionedMonotoneResidualTransportCalibrator",
    "BudgetConditionedMonotoneNoTargetAnchorCalibrator",
    "AnchorFreeResidualCalibratorOutput",
    "ResidualTransportCalibratorError",
    "ResidualTransportCalibratorOutput",
]

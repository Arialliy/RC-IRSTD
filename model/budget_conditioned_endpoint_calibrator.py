"""Result-free RC5+ budget-conditioned endpoint-aware calibrator candidates.

The RC5 model learns only three coordinates and interpolates between them at
inference.  This candidate makes the budget axis part of the learned object:
the context encoder predicts a denser curve over an exact-rational knot
lattice and the same piecewise-linear function is queried during training and
deployment.  The monotone variant parameterizes positive interval mass, so
every in-range rational budget query is ordered by construction.

This module is additive and is not yet admitted by checkpoint-v7 or the RC5
production trainer.  It must pass the RC5+ design/audit gate before replacing
the frozen three-point candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.endpoint_aware_pixel_calibrator import (
    ANCHOR_MIX_INITIAL_WEIGHT,
    ANCHOR_MIX_PARAMETERIZATION,
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


BUDGET_CONDITIONED_SCHEMA = (
    "rc-irstd.budget-conditioned-endpoint-calibrator-candidate.v1"
)
BUDGET_CONDITIONED_DIRECT_MODEL_ID = (
    "budget_conditioned_direct_endpoint_aware_pixel_calibrator"
)
BUDGET_CONDITIONED_MONOTONE_MODEL_ID = (
    "budget_conditioned_monotone_endpoint_aware_pixel_calibrator"
)
BUDGET_AXIS_TRANSFORM = (
    "u=(log(numerator/denominator)-log(1/10000))/"
    "(log(1/1000000)-log(1/10000))"
)
BUDGET_INTERPOLATION = (
    "continuous_piecewise_linear_in_normalized_log_budget_coordinate"
)
ANCHOR_MIX_RULE = (
    "s_raw(x,b)=(1-alpha)*s_anchor(x,b)+alpha*s_learned(x,b)"
)

# Quarter-decade, lowest-term rational knots.  The three claim-bearing
# budgets are exact members at indices 0, 4 and 8.  Integer false-positive
# counts always use these pairs, never their binary64 projections.
BUDGET_KNOT_RATIONALS: tuple[tuple[int, int], ...] = (
    (1, 10_000),
    (1, 17_783),
    (1, 31_623),
    (1, 56_234),
    (1, 100_000),
    (1, 177_828),
    (1, 316_228),
    (1, 562_341),
    (1, 1_000_000),
)
PRIMARY_BUDGET_KNOT_INDICES = (0, 4, 8)
PRIMARY_BUDGET_RATIONALS = tuple(
    BUDGET_KNOT_RATIONALS[index] for index in PRIMARY_BUDGET_KNOT_INDICES
)


class BudgetConditionedCalibratorError(ValueError):
    """The candidate model or exact-rational budget request is invalid."""


@dataclass(frozen=True)
class BudgetConditionedEndpointCalibratorOutput:
    budget_knot_numerators: torch.Tensor
    budget_knot_denominators: torch.Tensor
    budget_knot_values: torch.Tensor
    budget_knot_positions: torch.Tensor
    anchor_coordinates: torch.Tensor
    anchor_mix_weight: torch.Tensor
    grid_learned_raw_coordinates: torch.Tensor
    grid_raw_coordinates: torch.Tensor
    grid_coordinates: torch.Tensor
    grid_thresholds: torch.Tensor
    requested_budget_numerators: torch.Tensor | None = None
    requested_budget_denominators: torch.Tensor | None = None
    requested_budget_positions: torch.Tensor | None = None
    requested_anchor_coordinates: torch.Tensor | None = None
    requested_learned_raw_coordinates: torch.Tensor | None = None
    requested_raw_coordinates: torch.Tensor | None = None
    requested_coordinates: torch.Tensor | None = None
    requested_thresholds: torch.Tensor | None = None


def _validated_knot_rationals(
    value: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("budget knot rationals must be an ordered sequence")
    if len(value) < 5:
        raise BudgetConditionedCalibratorError(
            "budget-conditioned curve requires at least five rational knots"
        )
    result: list[tuple[int, int]] = []
    for index, row in enumerate(value):
        if (
            isinstance(row, (str, bytes))
            or not isinstance(row, Sequence)
            or len(row) != 2
            or type(row[0]) is not int
            or type(row[1]) is not int
        ):
            raise TypeError(f"budget knot {index} must be one integer pair")
        numerator, denominator = row
        if numerator <= 0 or denominator <= numerator:
            raise BudgetConditionedCalibratorError(
                "budget knots must lie strictly inside (0,1)"
            )
        reduced = Fraction(numerator, denominator)
        if (reduced.numerator, reduced.denominator) != (numerator, denominator):
            raise BudgetConditionedCalibratorError(
                "budget knots must be canonical lowest-term rationals"
            )
        result.append((numerator, denominator))
    fractions = tuple(Fraction(numerator, denominator) for numerator, denominator in result)
    if not all(left > right for left, right in zip(fractions, fractions[1:])):
        raise BudgetConditionedCalibratorError(
            "budget knots must descend strictly from loose to strict"
        )
    return tuple(result)


def _validated_hidden_dims(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("hidden_dims must be an integer sequence")
    if (
        len(value) != 1
        or type(value[0]) is not int
        or value[0] <= 0
    ):
        raise BudgetConditionedCalibratorError(
            "candidate requires exactly one positive hidden width"
        )
    return (value[0],)


class _BudgetConditionedBase(nn.Module):
    supports_reject = False
    supports_complete_budget_curve = True
    supports_exact_rational_budget_requests = True
    threshold_representation_schema = THRESHOLD_REPRESENTATION_SCHEMA

    def __init__(
        self,
        *,
        context_feature_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        budget_knot_rationals: Sequence[tuple[int, int]],
    ) -> None:
        super().__init__()
        if (
            type(context_feature_dim) is not int
            or context_feature_dim <= 0
        ):
            raise BudgetConditionedCalibratorError(
                "context_feature_dim must be a positive integer"
            )
        widths = _validated_hidden_dims(hidden_dims)
        if isinstance(dropout, bool):
            raise TypeError("dropout must be a real number")
        dropout_value = float(dropout)
        if not math.isfinite(dropout_value) or not 0.0 <= dropout_value < 1.0:
            raise BudgetConditionedCalibratorError(
                "dropout must be finite and lie in [0,1)"
            )
        knots = _validated_knot_rationals(budget_knot_rationals)
        if knots != BUDGET_KNOT_RATIONALS:
            raise BudgetConditionedCalibratorError(
                "the RC5+ candidate budget-knot lattice is frozen"
            )
        numerators = torch.tensor(
            [row[0] for row in knots], dtype=torch.int64
        )
        denominators = torch.tensor(
            [row[1] for row in knots], dtype=torch.int64
        )
        values = numerators.to(torch.float64) / denominators.to(torch.float64)
        log_values = torch.log(numerators.to(torch.float64)) - torch.log(
            denominators.to(torch.float64)
        )
        positions = (log_values - log_values[0]) / (
            log_values[-1] - log_values[0]
        )
        if (
            positions[0].item() != 0.0
            or positions[-1].item() != 1.0
            or not bool((positions[1:] > positions[:-1]).all().item())
        ):
            raise RuntimeError("budget knot normalization is not strictly ordered")

        self.context_feature_dim = context_feature_dim
        self.hidden_dims = widths
        self.dropout = dropout_value
        self.budget_knot_rationals = knots
        self.raw_coordinate_min = RAW_COORDINATE_MIN
        self.raw_coordinate_max = RAW_COORDINATE_MAX
        self.register_buffer("budget_knot_numerators", numerators)
        self.register_buffer("budget_knot_denominators", denominators)
        self.register_buffer("budget_knot_values", values)
        self.register_buffer("budget_knot_positions", positions)
        initial_mix_logit = math.log(
            ANCHOR_MIX_INITIAL_WEIGHT / (1.0 - ANCHOR_MIX_INITIAL_WEIGHT)
        )
        self.anchor_mix_logit = nn.Parameter(
            torch.tensor(initial_mix_logit, dtype=torch.float64)
        )
        self.encoder = nn.Sequential(
            nn.Linear(context_feature_dim, widths[0]),
            nn.GELU(),
            nn.Dropout(dropout_value),
        )

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
            raise BudgetConditionedCalibratorError(
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
            raise BudgetConditionedCalibratorError(
                f"{name} must be finite float64[{batch_size},{width}]"
            )
        result = value.to(device=device)
        try:
            decode_coordinate_torch(result)
        except ValueError as error:
            raise BudgetConditionedCalibratorError(
                f"{name} contains a noncanonical EATC coordinate"
            ) from error
        if width > 1 and bool((result[:, 1:] < result[:, :-1]).any().item()):
            raise BudgetConditionedCalibratorError(
                f"{name} must be nondecreasing from loose to strict budget"
            )
        return result

    def _mix(
        self,
        learned_raw: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if learned_raw.shape != anchor.shape:
            raise RuntimeError("learned and anchor curve shapes differ")
        logit = self.anchor_mix_logit.to(
            device=learned_raw.device, dtype=torch.float64
        )
        if not bool(torch.isfinite(logit).item()):
            raise FloatingPointError("anchor mix logit is non-finite")
        alpha = torch.sigmoid(logit)
        if not 0.0 < float(alpha.item()) < 1.0:
            raise FloatingPointError("anchor mix weight must remain strictly in (0,1)")
        raw = (1.0 - alpha) * anchor + alpha * learned_raw
        if (
            not bool(torch.isfinite(raw).all().item())
            or bool((raw <= RAW_COORDINATE_MIN).any().item())
            or bool((raw >= RAW_COORDINATE_MAX).any().item())
        ):
            raise FloatingPointError("anchor-mixed raw curve left its finite bounds")
        return raw, alpha

    def _rational_requests(
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
            raise BudgetConditionedCalibratorError(
                "budget requests must have shape [Q], [1,Q], or [B,Q]"
            )
        rows_n = n.tolist()
        rows_d = d.tolist()
        loose = Fraction(*self.budget_knot_rationals[0])
        strict = Fraction(*self.budget_knot_rationals[-1])
        for row_n, row_d in zip(rows_n, rows_d):
            previous: Fraction | None = None
            for numerator, denominator in zip(row_n, row_d):
                if numerator <= 0 or denominator <= numerator:
                    raise BudgetConditionedCalibratorError(
                        "requested budgets must lie strictly inside (0,1)"
                    )
                fraction = Fraction(numerator, denominator)
                if (fraction.numerator, fraction.denominator) != (
                    numerator,
                    denominator,
                ):
                    raise BudgetConditionedCalibratorError(
                        "requested budgets must be lowest-term rationals"
                    )
                if not strict <= fraction <= loose:
                    raise BudgetConditionedCalibratorError(
                        "requested budgets must stay inside the trained knot range"
                    )
                if previous is not None and not previous > fraction:
                    raise BudgetConditionedCalibratorError(
                        "requested budgets must descend strictly from loose to strict"
                    )
                previous = fraction
        n_device = n.to(device=device)
        d_device = d.to(device=device)
        log_values = torch.log(n_device.to(torch.float64)) - torch.log(
            d_device.to(torch.float64)
        )
        knot_log = torch.log(
            self.budget_knot_numerators.to(device=device, dtype=torch.float64)
        ) - torch.log(
            self.budget_knot_denominators.to(device=device, dtype=torch.float64)
        )
        positions = (log_values - knot_log[0]) / (knot_log[-1] - knot_log[0])
        if (
            not bool(torch.isfinite(positions).all().item())
            or (
                positions.shape[1] > 1
                and not bool((positions[:, 1:] > positions[:, :-1]).all().item())
            )
        ):
            raise BudgetConditionedCalibratorError(
                "distinct rational requests are not distinguishable in the "
                "float64 budget-curve coordinate"
            )
        return n_device, d_device, positions

    def _interpolate_learned(
        self,
        learned_grid: torch.Tensor,
        positions: torch.Tensor,
        numerators: torch.Tensor,
        denominators: torch.Tensor,
    ) -> torch.Tensor:
        knots = self.budget_knot_positions.to(
            device=learned_grid.device, dtype=torch.float64
        )
        right = torch.searchsorted(knots, positions, right=True).clamp(
            1, self.num_budget_knots - 1
        )
        left = right - 1
        batch = torch.arange(
            learned_grid.shape[0], device=learned_grid.device
        )[:, None]
        x0 = knots[left]
        x1 = knots[right]
        y0 = learned_grid[batch, left]
        y1 = learned_grid[batch, right]
        weight = (positions - x0) / (x1 - x0)
        result = y0 + weight * (y1 - y0)
        # Requests that are exact members of the frozen rational lattice must
        # replay their learned ordinates bit-for-bit.  Arithmetic interpolation
        # at weight one is only numerically equal and can round differently.
        knot_n = self.budget_knot_numerators.to(device=learned_grid.device)
        knot_d = self.budget_knot_denominators.to(device=learned_grid.device)
        exact = (numerators[:, :, None] == knot_n[None, None, :]) & (
            denominators[:, :, None] == knot_d[None, None, :]
        )
        exact_any = exact.any(dim=2)
        exact_index = exact.to(torch.int64).argmax(dim=2)
        exact_value = learned_grid.gather(1, exact_index)
        result = torch.where(exact_any, exact_value, result)
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("budget-conditioned interpolation is non-finite")
        return result

    def _output(
        self,
        *,
        learned_grid: torch.Tensor,
        anchor_grid: torch.Tensor,
        request_numerators: torch.Tensor | None,
        request_denominators: torch.Tensor | None,
        requested_anchor_coordinates: torch.Tensor | None,
        structural_monotonicity: bool,
    ) -> BudgetConditionedEndpointCalibratorOutput:
        raw_grid, alpha = self._mix(learned_grid, anchor_grid)
        coordinates = canonicalize_raw_torch(raw_grid)
        thresholds = decode_coordinate_torch(coordinates)
        if structural_monotonicity:
            self._assert_monotone(raw_grid, coordinates, thresholds, "grid")
        common = {
            "budget_knot_numerators": self.budget_knot_numerators,
            "budget_knot_denominators": self.budget_knot_denominators,
            "budget_knot_values": self.budget_knot_values,
            "budget_knot_positions": self.budget_knot_positions,
            "anchor_coordinates": anchor_grid,
            "anchor_mix_weight": alpha,
            "grid_learned_raw_coordinates": learned_grid,
            "grid_raw_coordinates": raw_grid,
            "grid_coordinates": coordinates,
            "grid_thresholds": thresholds,
        }
        if request_numerators is None and request_denominators is None:
            if requested_anchor_coordinates is not None:
                raise BudgetConditionedCalibratorError(
                    "requested anchor requires exact-rational budget requests"
                )
            return BudgetConditionedEndpointCalibratorOutput(**common)
        if request_numerators is None or request_denominators is None:
            raise BudgetConditionedCalibratorError(
                "requested budget numerator and denominator must appear together"
            )
        n, d, positions = self._rational_requests(
            request_numerators,
            request_denominators,
            batch_size=learned_grid.shape[0],
            device=learned_grid.device,
        )
        anchor_requested = self._anchor(
            requested_anchor_coordinates,
            batch_size=learned_grid.shape[0],
            width=n.shape[1],
            device=learned_grid.device,
            name="requested_anchor_coordinates",
        )
        learned_requested = self._interpolate_learned(
            learned_grid,
            positions,
            n,
            d,
        )
        raw_requested, requested_alpha = self._mix(
            learned_requested, anchor_requested
        )
        if not torch.equal(alpha, requested_alpha):
            raise RuntimeError("grid/request anchor mix weights differ")
        requested_coordinates = canonicalize_raw_torch(raw_requested)
        requested_thresholds = decode_coordinate_torch(requested_coordinates)
        if structural_monotonicity:
            self._assert_monotone(
                raw_requested,
                requested_coordinates,
                requested_thresholds,
                "requested",
            )
        return BudgetConditionedEndpointCalibratorOutput(
            **common,
            requested_budget_numerators=n,
            requested_budget_denominators=d,
            requested_budget_positions=positions,
            requested_anchor_coordinates=anchor_requested,
            requested_learned_raw_coordinates=learned_requested,
            requested_raw_coordinates=raw_requested,
            requested_coordinates=requested_coordinates,
            requested_thresholds=requested_thresholds,
        )

    @staticmethod
    def _assert_monotone(
        raw: torch.Tensor,
        coordinates: torch.Tensor,
        thresholds: torch.Tensor,
        name: str,
    ) -> None:
        if not bool((raw[:, 1:] > raw[:, :-1]).all().item()):
            raise FloatingPointError(f"{name} raw curve is not strictly increasing")
        if not bool((coordinates[:, 1:] >= coordinates[:, :-1]).all().item()):
            raise FloatingPointError(f"{name} canonical curve decreased")
        if not bool((thresholds[:, 1:] >= thresholds[:, :-1]).all().item()):
            raise FloatingPointError(f"{name} threshold curve decreased")
        endpoint = coordinates == UPPER_ENDPOINT_COORDINATE
        if bool((endpoint[:, :-1] & ~endpoint[:, 1:]).any().item()):
            raise FloatingPointError(f"{name} endpoint decisions are not suffix closed")

    def _common_export_config(self) -> dict[str, object]:
        return {
            "schema_version": BUDGET_CONDITIONED_SCHEMA,
            "context_feature_dim": self.context_feature_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "budget_knot_rationals": [list(row) for row in self.budget_knot_rationals],
            "primary_budget_knot_indices": list(PRIMARY_BUDGET_KNOT_INDICES),
            "budget_axis_transform": BUDGET_AXIS_TRANSFORM,
            "budget_interpolation": BUDGET_INTERPOLATION,
            "threshold_representation_schema": THRESHOLD_REPRESENTATION_SCHEMA,
            "anchor_source": T4_ANCHOR_SOURCE,
            "anchor_mix_rule": ANCHOR_MIX_RULE,
            "anchor_mix_parameterization": ANCHOR_MIX_PARAMETERIZATION,
            "anchor_mix_initial_weight": ANCHOR_MIX_INITIAL_WEIGHT,
            "threshold_semantics": "prediction = probability > threshold",
        }

    def _common_capability(self) -> dict[str, object]:
        return {
            **self._common_export_config(),
            "supports_reject": False,
            "supports_fallback": False,
            "supports_exact_rational_budget_requests": True,
            "valid_request_coordinate": (
                "ordered_in_range_rationals_with_distinct_float64_log_positions"
            ),
            "exact_knot_replay": "bitwise_learned_ordinate_replay",
            "invalid_request_is_decision_reject": False,
            "requires_anchor_coordinates": True,
            "requested_anchor_semantics": (
                "exact_context_order_statistic_at_the_same_requested_rational_budget"
            ),
            "risk_guarantee": "empirical_not_certified",
            "threshold_representation": representation_contract(),
        }


class BudgetConditionedDirectEndpointAwarePixelCalibrator(
    _BudgetConditionedBase
):
    """T6+ control: the same budget spline with unconstrained knot ordinates."""

    model_id = BUDGET_CONDITIONED_DIRECT_MODEL_ID
    structural_monotonicity = False

    def __init__(
        self,
        context_feature_dim: int,
        hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.1,
        budget_knot_rationals: Sequence[tuple[int, int]] = BUDGET_KNOT_RATIONALS,
    ) -> None:
        super().__init__(
            context_feature_dim=context_feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            budget_knot_rationals=budget_knot_rationals,
        )
        self.coordinate_head = nn.Linear(self.hidden_dims[0], self.num_budget_knots)

    def _learned_grid(self, hidden: torch.Tensor) -> torch.Tensor:
        unconstrained = self.coordinate_head(hidden).to(torch.float64)
        return RAW_COORDINATE_MIN + torch.sigmoid(unconstrained) * (
            RAW_COORDINATE_MAX - RAW_COORDINATE_MIN
        )

    def forward(
        self,
        context_features: torch.Tensor,
        *,
        anchor_coordinates: torch.Tensor,
        budget_numerators: torch.Tensor | None = None,
        budget_denominators: torch.Tensor | None = None,
        requested_anchor_coordinates: torch.Tensor | None = None,
    ) -> BudgetConditionedEndpointCalibratorOutput:
        features = self._features(context_features)
        anchor = self._anchor(
            anchor_coordinates,
            batch_size=features.shape[0],
            width=self.num_budget_knots,
            device=features.device,
            name="anchor_coordinates",
        )
        learned = self._learned_grid(self.encoder(features))
        return self._output(
            learned_grid=learned,
            anchor_grid=anchor,
            request_numerators=budget_numerators,
            request_denominators=budget_denominators,
            requested_anchor_coordinates=requested_anchor_coordinates,
            structural_monotonicity=False,
        )

    def export_config(self) -> dict[str, object]:
        return self._common_export_config()

    def capability_contract(self) -> dict[str, object]:
        return {
            **self._common_capability(),
            "model_id": self.model_id,
            "structural_monotonicity": False,
            "learned_curve": "unconstrained_context_conditioned_rational_budget_spline",
            "training_objective": "T6_plus_oracle_coordinate_huber_only",
        }


class BudgetConditionedMonotoneEndpointAwarePixelCalibrator(
    _BudgetConditionedBase
):
    """T7+/T8+: a continuous context-conditioned monotone budget curve."""

    model_id = BUDGET_CONDITIONED_MONOTONE_MODEL_ID
    structural_monotonicity = True

    def __init__(
        self,
        context_feature_dim: int,
        hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.1,
        budget_knot_rationals: Sequence[tuple[int, int]] = BUDGET_KNOT_RATIONALS,
        minimum_raw_coordinate_gap: float = 1e-3,
    ) -> None:
        super().__init__(
            context_feature_dim=context_feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            budget_knot_rationals=budget_knot_rationals,
        )
        if isinstance(minimum_raw_coordinate_gap, bool):
            raise TypeError("minimum_raw_coordinate_gap must be a real number")
        gap = float(minimum_raw_coordinate_gap)
        span = RAW_COORDINATE_MAX - RAW_COORDINATE_MIN
        if (
            not math.isfinite(gap)
            or gap <= 0.0
            or self.num_budget_knots * gap >= span
        ):
            raise BudgetConditionedCalibratorError(
                "minimum_raw_coordinate_gap is invalid for the knot count"
            )
        self.minimum_raw_coordinate_gap = gap
        self.spacing_head = nn.Linear(
            self.hidden_dims[0], self.num_budget_knots + 1
        )

    def _learned_grid(self, hidden: torch.Tensor) -> torch.Tensor:
        mass = F.softmax(self.spacing_head(hidden).to(torch.float64), dim=1)
        cumulative = torch.cumsum(mass[:, :-1], dim=1)
        index = torch.arange(
            1,
            self.num_budget_knots + 1,
            device=hidden.device,
            dtype=torch.float64,
        )[None, :]
        free_span = (
            RAW_COORDINATE_MAX
            - RAW_COORDINATE_MIN
            - self.num_budget_knots * self.minimum_raw_coordinate_gap
        )
        result = (
            RAW_COORDINATE_MIN
            + index * self.minimum_raw_coordinate_gap
            + cumulative * free_span
        )
        if not bool((result[:, 1:] > result[:, :-1]).all().item()):
            raise FloatingPointError("positive interval mass lost strict order")
        return result

    def forward(
        self,
        context_features: torch.Tensor,
        *,
        anchor_coordinates: torch.Tensor,
        budget_numerators: torch.Tensor | None = None,
        budget_denominators: torch.Tensor | None = None,
        requested_anchor_coordinates: torch.Tensor | None = None,
    ) -> BudgetConditionedEndpointCalibratorOutput:
        features = self._features(context_features)
        anchor = self._anchor(
            anchor_coordinates,
            batch_size=features.shape[0],
            width=self.num_budget_knots,
            device=features.device,
            name="anchor_coordinates",
        )
        learned = self._learned_grid(self.encoder(features))
        return self._output(
            learned_grid=learned,
            anchor_grid=anchor,
            request_numerators=budget_numerators,
            request_denominators=budget_denominators,
            requested_anchor_coordinates=requested_anchor_coordinates,
            structural_monotonicity=True,
        )

    def export_config(self) -> dict[str, object]:
        return {
            **self._common_export_config(),
            "minimum_raw_coordinate_gap": self.minimum_raw_coordinate_gap,
        }

    def capability_contract(self) -> dict[str, object]:
        return {
            **self._common_capability(),
            "model_id": self.model_id,
            "structural_monotonicity": True,
            "raw_coordinate_order": (
                "strictly_increasing_for_all_valid_ordered_queries"
            ),
            "decoded_threshold_order": (
                "nondecreasing_for_all_valid_ordered_queries"
            ),
            "upper_endpoint_decisions": "suffix_closed",
            "learned_curve": "positive_interval_context_conditioned_rational_budget_spline",
            "training_objective": (
                "T7_plus_oracle_coordinate_huber_or_"
                "T8_plus_exact_event_risk_aligned_plus_oracle_huber"
            ),
        }


__all__ = [
    "ANCHOR_MIX_RULE",
    "BUDGET_AXIS_TRANSFORM",
    "BUDGET_CONDITIONED_DIRECT_MODEL_ID",
    "BUDGET_CONDITIONED_MONOTONE_MODEL_ID",
    "BUDGET_CONDITIONED_SCHEMA",
    "BUDGET_INTERPOLATION",
    "BUDGET_KNOT_RATIONALS",
    "BudgetConditionedCalibratorError",
    "BudgetConditionedDirectEndpointAwarePixelCalibrator",
    "BudgetConditionedEndpointCalibratorOutput",
    "BudgetConditionedMonotoneEndpointAwarePixelCalibrator",
    "PRIMARY_BUDGET_KNOT_INDICES",
    "PRIMARY_BUDGET_RATIONALS",
]

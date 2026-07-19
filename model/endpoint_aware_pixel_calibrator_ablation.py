"""Preregistered learned-only T8 anchor ablation.

This module deliberately defines a separate model identity instead of adding a
runtime switch to the claim-bearing T8 calibrator.  It reuses the frozen
monotone learned branch and endpoint-aware threshold representation, but has no
analytic-anchor input, parameter, state, or output field.  The class is an
ablation-only research artifact and is not admitted by deployment
checkpoint-v7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from model.endpoint_aware_pixel_calibrator import (
    PIXEL_RISK_NO_REJECT_SCOPE,
    MonotoneEndpointAwarePixelCalibrator,
)
from model.endpoint_aware_threshold import (
    UPPER_ENDPOINT_COORDINATE,
    canonicalize_raw_torch,
    decode_coordinate_torch,
    representation_contract,
)


T8_NO_ANCHOR_METHOD_ID = "T8_NO_ANCHOR"
T8_NO_ANCHOR_MODEL_ID = (
    "learned_only_monotone_endpoint_aware_pixel_calibrator_ablation"
)
T8_NO_ANCHOR_ABLATION_ROLE = "risk_aligned_ablation_only"
T8_NO_ANCHOR_EXPECTED_TRAINABLE_PARAMETERS = 3140


@dataclass(frozen=True)
class LearnedOnlyEndpointAwareCalibratorOutput:
    """Hard-forward output with no analytic-anchor surface."""

    pixel_budget_grid: torch.Tensor
    grid_raw_coordinates: torch.Tensor
    grid_coordinates: torch.Tensor
    grid_thresholds: torch.Tensor
    requested_pixel_budgets: torch.Tensor | None = None
    requested_raw_coordinates: torch.Tensor | None = None
    requested_coordinates: torch.Tensor | None = None
    requested_thresholds: torch.Tensor | None = None
    method_id: str = field(default=T8_NO_ANCHOR_METHOD_ID, init=False)
    ablation_role: str = field(default=T8_NO_ANCHOR_ABLATION_ROLE, init=False)
    claim_bearing: bool = field(default=False, init=False)

    @property
    def grid_logits(self) -> torch.Tensor:
        """Trainer migration alias for ``grid_coordinates``."""

        return self.grid_coordinates

    @property
    def requested_logits(self) -> torch.Tensor | None:
        """Trainer migration alias for ``requested_coordinates``."""

        return self.requested_coordinates


class LearnedOnlyMonotoneEndpointAwarePixelCalibrator(
    MonotoneEndpointAwarePixelCalibrator
):
    """T8_NO_ANCHOR: the T8 learned curve without its analytic anchor."""

    model_id = T8_NO_ANCHOR_MODEL_ID
    method_id = T8_NO_ANCHOR_METHOD_ID
    ablation_role = T8_NO_ANCHOR_ABLATION_ROLE
    claim_bearing = False
    budget_scope = PIXEL_RISK_NO_REJECT_SCOPE
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
            minimum_raw_coordinate_gap=minimum_raw_coordinate_gap,
        )
        # Replace, rather than freeze, the inherited scalar.  A None-registered
        # parameter is absent from parameters(), named_parameters(), and the
        # serialized state_dict, so this identity cannot silently carry T8's
        # anchor state.
        self.register_parameter("anchor_mix_logit", None)
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if (
            self.context_feature_dim != 93
            or self.hidden_dims != (32,)
            or parameter_count != T8_NO_ANCHOR_EXPECTED_TRAINABLE_PARAMETERS
        ):
            raise ValueError(
                "T8_NO_ANCHOR requires the frozen 93->32->4 learned branch "
                "with exactly 3140 trainable parameters"
            )

    def forward(
        self,
        context_features: torch.Tensor,
        *,
        pixel_budgets: torch.Tensor | None = None,
    ) -> LearnedOnlyEndpointAwareCalibratorOutput:
        self._validate_features(context_features)
        raw = self._ordered_raw(self.encoder(context_features))
        if not bool((raw[:, 1:] > raw[:, :-1]).all().item()):
            raise FloatingPointError("learned-only raw-coordinate curve is not strict")

        coordinates = canonicalize_raw_torch(raw)
        thresholds = decode_coordinate_torch(coordinates)
        if pixel_budgets is None:
            output = LearnedOnlyEndpointAwareCalibratorOutput(
                pixel_budget_grid=self.pixel_budget_grid,
                grid_raw_coordinates=raw,
                grid_coordinates=coordinates,
                grid_thresholds=thresholds,
            )
        else:
            requested_raw, normalized = self._interpolate_raw(raw, pixel_budgets)
            requested_coordinates = canonicalize_raw_torch(requested_raw)
            requested_thresholds = decode_coordinate_torch(requested_coordinates)
            output = LearnedOnlyEndpointAwareCalibratorOutput(
                pixel_budget_grid=self.pixel_budget_grid,
                grid_raw_coordinates=raw,
                grid_coordinates=coordinates,
                grid_thresholds=thresholds,
                requested_pixel_budgets=normalized,
                requested_raw_coordinates=requested_raw,
                requested_coordinates=requested_coordinates,
                requested_thresholds=requested_thresholds,
            )

        if not bool(
            (output.grid_coordinates[:, 1:] >= output.grid_coordinates[:, :-1])
            .all()
            .item()
        ):
            raise FloatingPointError("learned-only canonical coordinate curve decreased")
        if not bool(
            (output.grid_thresholds[:, 1:] >= output.grid_thresholds[:, :-1])
            .all()
            .item()
        ):
            raise FloatingPointError("learned-only decoded threshold curve decreased")
        endpoint = output.grid_coordinates == UPPER_ENDPOINT_COORDINATE
        if bool((endpoint[:, :-1] & ~endpoint[:, 1:]).any().item()):
            raise FloatingPointError("learned-only upper endpoints are not suffix closed")
        return output

    def export_config(self) -> dict[str, object]:
        """Return an anchor-free, identity-bound ablation configuration."""

        return {
            "method_id": self.method_id,
            "model_id": self.model_id,
            "ablation_role": self.ablation_role,
            "claim_bearing": False,
            "expected_trainable_parameters": (
                T8_NO_ANCHOR_EXPECTED_TRAINABLE_PARAMETERS
            ),
            "context_feature_dim": self.context_feature_dim,
            "pixel_budget_grid": self.pixel_budget_grid.detach().cpu().tolist(),
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "minimum_raw_coordinate_gap": self.minimum_raw_coordinate_gap,
            "raw_coordinate_min_hex": self.raw_coordinate_min.hex(),
            "raw_coordinate_max_hex": self.raw_coordinate_max.hex(),
            "threshold_representation_schema": self.threshold_representation_schema,
        }

    def capability_contract(self) -> dict[str, object]:
        """Describe the non-claim-bearing learned-only ablation capability."""

        return {
            "method_id": self.method_id,
            "model_id": self.model_id,
            "ablation_role": self.ablation_role,
            "claim_bearing": False,
            "expected_trainable_parameters": (
                T8_NO_ANCHOR_EXPECTED_TRAINABLE_PARAMETERS
            ),
            "stage": "stage2_preregistered_ablation_no_reject",
            "budget_scope": self.budget_scope,
            "budget_axis": "pixel_false_alarm_rate",
            "supports_component_budget": False,
            "supports_reject": False,
            "supports_complete_budget_curve": True,
            "curve_output_shape": "[batch,J]",
            "supports_query_risk_aligned_loss": True,
            "structural_monotonicity": True,
            "requires_anchor_coordinates": False,
            "uses_analytic_anchor": False,
            "runtime_anchor_toggle_supported": False,
            "method_identity_selected_by_class_not_runtime_flag": True,
            "checkpoint_v7_supported": False,
            "raw_coordinate_order": "strictly_increasing",
            "decoded_threshold_order": "nondecreasing",
            "upper_endpoint_decisions": "suffix_closed",
            "budget_interpolation": (
                "raw_coordinate_piecewise_linear_log10_no_extrapolation"
            ),
            "threshold_representation": representation_contract(),
            "threshold_semantics": "prediction = probability > threshold",
            "training_objective": (
                "T8_verified_global_exact_event_curve_piecewise_linear_"
                "risk_surrogate_plus_oracle_coordinate_huber"
            ),
            "risk_guarantee": "empirical_not_certified",
        }


__all__ = [
    "T8_NO_ANCHOR_ABLATION_ROLE",
    "T8_NO_ANCHOR_EXPECTED_TRAINABLE_PARAMETERS",
    "T8_NO_ANCHOR_METHOD_ID",
    "T8_NO_ANCHOR_MODEL_ID",
    "LearnedOnlyEndpointAwareCalibratorOutput",
    "LearnedOnlyMonotoneEndpointAwarePixelCalibrator",
]

from __future__ import annotations

"""No-reject monotone inverse pixel-risk calibrator.

The budget grid is stored from loose to strict, e.g. ``[1e-4, 1e-5, 1e-6]``.
The network emits a complete threshold-logit curve in one forward pass.  Its
parameterisation guarantees that stricter budgets receive no lower threshold.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def validate_budget_grid(budgets: np.ndarray | torch.Tensor | list[float]) -> np.ndarray:
    values = np.asarray(budgets, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("budgets must be a non-empty 1-D sequence")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("budgets must be finite and positive")
    if len(values) > 1 and not np.all(np.diff(values) < 0):
        raise ValueError("budgets must be strictly descending: loose -> strict")
    return values.astype(np.float32)


class PermutationInvariantSourceEncoder(nn.Module):
    """DeepSets-style encoder for an unordered set of source distances."""

    def __init__(self, hidden_dim: int = 32, output_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.element = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(2 * output_dim + 1, output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        distances: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        *,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if distances is None:
            if batch_size is None or device is None or dtype is None:
                raise ValueError("batch_size, device and dtype are required when distances=None")
            return torch.zeros((batch_size, self.output_dim), device=device, dtype=dtype)
        if distances.ndim != 2:
            raise ValueError("source distances must have shape [B, K]")
        if mask is None:
            mask = torch.ones_like(distances, dtype=torch.bool)
        if mask.shape != distances.shape:
            raise ValueError("source-distance mask must match distances")
        if distances.shape[1] == 0:
            return torch.zeros(
                (distances.shape[0], self.output_dim),
                device=distances.device,
                dtype=distances.dtype,
            )
        embedded = self.element(distances.unsqueeze(-1))
        valid = mask.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = (embedded * valid).sum(dim=1) / count
        negative_inf = torch.finfo(embedded.dtype).min
        maximum = embedded.masked_fill(~valid, negative_inf).max(dim=1).values
        no_source = ~mask.any(dim=1)
        if no_source.any():
            maximum = maximum.clone()
            maximum[no_source] = 0.0
        availability = mask.any(dim=1, keepdim=True).to(embedded.dtype)
        return self.combine(torch.cat([mean, maximum, availability], dim=1))


@dataclass(frozen=True)
class MonotoneCalibratorConfig:
    input_dim: int
    budgets: tuple[float, ...]
    hidden_dim: int = 192
    source_hidden_dim: int = 32
    source_output_dim: int = 64
    dropout: float = 0.10
    min_logit_step: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MonotonePixelCalibrator(nn.Module):
    """Predict an inverse pixel-risk curve with structural monotonicity.

    There is deliberately no reject/abstention head in this class.  Legacy
    direct-threshold + reject implementations can remain separate baselines.
    """

    def __init__(
        self,
        input_dim: int,
        budgets: list[float] | tuple[float, ...] | np.ndarray,
        *,
        hidden_dim: int = 192,
        source_hidden_dim: int = 32,
        source_output_dim: int = 64,
        dropout: float = 0.10,
        min_logit_step: float = 0.0,
    ) -> None:
        super().__init__()
        budget_array = validate_budget_grid(budgets)
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if min_logit_step < 0:
            raise ValueError("min_logit_step must be non-negative")
        self.input_dim = int(input_dim)
        self.num_budgets = int(len(budget_array))
        self.min_logit_step = float(min_logit_step)
        self.register_buffer("budget_grid", torch.from_numpy(budget_array), persistent=True)

        self.source_encoder = PermutationInvariantSourceEncoder(
            hidden_dim=source_hidden_dim,
            output_dim=source_output_dim,
        )
        self.support_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + source_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # eta_min, total positive span, and J-1 allocation logits.
        output_dim = 2 + max(self.num_budgets - 1, 0)
        self.curve_head = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            self.curve_head.bias.zero_()
            # A modest initial span avoids identical thresholds without forcing
            # an extreme low-false-alarm operating point at initialisation.
            self.curve_head.bias[1] = -1.5

    @property
    def config(self) -> MonotoneCalibratorConfig:
        return MonotoneCalibratorConfig(
            input_dim=self.input_dim,
            budgets=tuple(float(v) for v in self.budget_grid.detach().cpu().tolist()),
            hidden_dim=self.support_encoder[1].out_features,
            source_hidden_dim=self.source_encoder.element[0].out_features,
            source_output_dim=self.source_encoder.output_dim,
            dropout=float(self.fusion[2].p),
            min_logit_step=self.min_logit_step,
        )

    def _build_curve(self, raw: torch.Tensor) -> torch.Tensor:
        eta_min = raw[:, :1]
        if self.num_budgets == 1:
            return eta_min
        total_span = F.softplus(raw[:, 1:2])
        allocation = torch.softmax(raw[:, 2:], dim=1)
        cumulative = torch.cumsum(allocation, dim=1)
        steps = torch.arange(
            1,
            self.num_budgets,
            device=raw.device,
            dtype=raw.dtype,
        ).unsqueeze(0)
        tail = eta_min + self.min_logit_step * steps + total_span * cumulative
        return torch.cat([eta_min, tail], dim=1)

    def forward(
        self,
        support_features: torch.Tensor,
        source_distances: torch.Tensor | None = None,
        source_distance_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if support_features.ndim != 2 or support_features.shape[1] != self.input_dim:
            raise ValueError(
                f"support_features must have shape [B, {self.input_dim}], "
                f"got {tuple(support_features.shape)}"
            )
        support_hidden = self.support_encoder(support_features)
        source_hidden = self.source_encoder(
            source_distances,
            source_distance_mask,
            batch_size=support_features.shape[0],
            device=support_features.device,
            dtype=support_features.dtype,
        )
        hidden = self.fusion(torch.cat([support_hidden, source_hidden], dim=1))
        threshold_logit = self._build_curve(self.curve_head(hidden))
        return {
            "threshold_logit": threshold_logit,
            "threshold": torch.sigmoid(threshold_logit),
        }

    def interpolate_logit(
        self,
        threshold_logit_curve: torch.Tensor,
        requested_budgets: torch.Tensor | np.ndarray | list[float],
    ) -> torch.Tensor:
        """Interpolate in log10-budget space and reject all extrapolation."""

        if threshold_logit_curve.ndim != 2 or threshold_logit_curve.shape[1] != self.num_budgets:
            raise ValueError("threshold_logit_curve must have shape [B, J]")
        requested = torch.as_tensor(
            requested_budgets,
            device=threshold_logit_curve.device,
            dtype=threshold_logit_curve.dtype,
        ).reshape(-1)
        if requested.numel() == 0 or torch.any(~torch.isfinite(requested)) or torch.any(requested <= 0):
            raise ValueError("requested budgets must be finite and positive")
        grid = self.budget_grid.to(
            device=threshold_logit_curve.device,
            dtype=threshold_logit_curve.dtype,
        )
        lower = grid[-1]
        upper = grid[0]
        scale = torch.maximum(torch.abs(upper), torch.abs(lower)).clamp_min(1e-12)
        tolerance = torch.finfo(requested.dtype).eps * scale * 64
        if torch.any(requested < lower - tolerance) or torch.any(requested > upper + tolerance):
            raise ValueError(
                "Budget extrapolation is disabled; requested values must remain "
                f"inside [{float(lower)}, {float(upper)}]"
            )
        if self.num_budgets == 1:
            if torch.any(torch.abs(requested - grid[0]) > tolerance):
                raise ValueError("A one-point budget grid only supports its exact budget")
            return threshold_logit_curve[:, :1].expand(-1, requested.numel())

        # torch.searchsorted expects ascending coordinates.
        x = torch.log10(torch.flip(grid, dims=[0]))
        y = torch.flip(threshold_logit_curve, dims=[1])
        query = torch.log10(requested.clamp(min=lower, max=upper))
        right = torch.searchsorted(x, query, right=False).clamp(1, self.num_budgets - 1)
        left = right - 1
        x0 = x[left]
        x1 = x[right]
        weight = ((query - x0) / (x1 - x0).clamp_min(torch.finfo(x.dtype).eps)).unsqueeze(0)
        y0 = y[:, left]
        y1 = y[:, right]
        return y0 + weight * (y1 - y0)


def assert_structural_monotonicity(
    threshold_logit: torch.Tensor,
    atol: float = 1e-7,
) -> None:
    if threshold_logit.ndim != 2:
        raise ValueError("threshold_logit must be 2-D")
    if threshold_logit.shape[1] > 1 and torch.any(torch.diff(threshold_logit, dim=1) < -atol):
        raise AssertionError("Stricter budgets received lower thresholds")

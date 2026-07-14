"""Query-risk-aligned objective for the Stage-2 no-Reject calibrator.

The default background contract is exact: every valid entry is one query
background pixel and all background pixels must be supplied.  Memory is
bounded by exact chunking, not by dropping observations.  A compact
supervision tensor is accepted only through the explicit
``weighted_stratified`` contract, where each sampled entry carries its known
population mass.  There is deliberately no implicit ``65_536`` (or other)
uniform-sampling mode because it can erase the extreme tail that determines a
``1e-6`` operating point.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F


FULL_BACKGROUND = "full"
WEIGHTED_STRATIFIED_BACKGROUND = "weighted_stratified"
BackgroundRepresentation = Literal["full", "weighted_stratified"]


@dataclass(frozen=True)
class CalibratorRiskLossOutput:
    """All scalar terms and per-episode/per-budget surrogate diagnostics."""

    total: torch.Tensor
    violation: torch.Tensor
    utility: torch.Tensor
    oracle_logit: torch.Tensor
    curve_smoothness: torch.Tensor
    surrogate_pixel_false_alarm_rate: torch.Tensor
    surrogate_detection_probability: torch.Tensor
    background_population_mass: torch.Tensor
    background_representation: str


@dataclass(frozen=True)
class CurveCalibratorRiskLossOutput:
    """Risk loss and diagnostics obtained from verified exact query curves."""

    total: torch.Tensor
    violation: torch.Tensor
    utility: torch.Tensor
    oracle_logit: torch.Tensor
    curve_smoothness: torch.Tensor
    coverage_penalty: torch.Tensor
    surrogate_pixel_false_alarm_rate: torch.Tensor
    surrogate_detection_probability: torch.Tensor
    coverage_shortfall_logits: torch.Tensor
    interpolation_logits: torch.Tensor
    interpolation_clamped_low: torch.Tensor
    interpolation_clamped_high: torch.Tensor


def calibrator_risk_capability_contract() -> dict[str, object]:
    """Return the supervision semantics persisted by future trainers."""

    return {
        "scope": "query_risk_aligned_pixel_budget_no_reject",
        "terms": (
            "budget_violation",
            "object_utility",
            "oracle_threshold_logit",
            "log10_budget_curve_smoothness",
            "verified_exact_curve_coverage",
        ),
        "background_supervision": (
            FULL_BACKGROUND,
            WEIGHTED_STRATIFIED_BACKGROUND,
        ),
        "default_background_supervision": FULL_BACKGROUND,
        "implicit_uniform_subsample_limit": None,
        "background_chunking": "exact_all_entries_no_sampling",
        "verified_curve_supervision": (
            "float64_piecewise_linear_no_extrapolation_with_coverage_penalty"
        ),
        "calculation_dtype": "float64",
        "risk_guarantee": "empirical_not_certified",
    }


def _require_float_matrix(
    value: torch.Tensor,
    *,
    name: str,
    batch_size: int | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [B, N]")
    if batch_size is not None and value.shape[0] != batch_size:
        raise ValueError(f"{name} batch dimension does not match threshold_logits")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")


def _valid_mask(
    mask: torch.Tensor | None,
    *,
    like: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(like, dtype=torch.bool)
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    if mask.shape != like.shape or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a bool tensor matching {tuple(like.shape)}")
    return mask.to(device=like.device)


def _ensure_finite_where_valid(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    name: str,
) -> None:
    selected = values[valid]
    if selected.numel() and not bool(torch.isfinite(selected).all().item()):
        raise ValueError(f"valid {name} entries must be finite")


def _budget_matrix(
    pixel_budgets: torch.Tensor,
    *,
    threshold_logits: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(pixel_budgets, torch.Tensor):
        raise TypeError("pixel_budgets must be a torch.Tensor")
    batch_size, curve_size = threshold_logits.shape
    budgets = pixel_budgets.to(
        device=threshold_logits.device,
        dtype=torch.float64,
    )
    if budgets.ndim == 1 and budgets.shape == (curve_size,):
        budgets = budgets.unsqueeze(0).expand(batch_size, -1)
    elif budgets.ndim == 2 and budgets.shape == threshold_logits.shape:
        pass
    else:
        raise ValueError("pixel_budgets must have shape [J] or [B, J]")
    if not bool(torch.isfinite(budgets).all().item()) or not bool(
        (budgets > 0.0).all().item()
    ):
        raise ValueError("pixel_budgets must be finite and positive")
    if curve_size > 1 and not bool(
        (budgets[:, :-1] > budgets[:, 1:]).all().item()
    ):
        raise ValueError(
            "pixel_budgets must be strictly descending from loose to strict"
        )
    return budgets


def _query_pixel_totals(
    query_total_pixels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(query_total_pixels, torch.Tensor):
        raise TypeError("query_total_pixels must be a torch.Tensor")
    totals = query_total_pixels.to(device=device, dtype=torch.float64)
    if totals.shape != (batch_size,):
        raise ValueError("query_total_pixels must have shape [B]")
    if not bool(torch.isfinite(totals).all().item()) or not bool(
        (totals > 0.0).all().item()
    ):
        raise ValueError("query_total_pixels must be finite and positive")
    if not bool(torch.equal(totals, torch.round(totals))):
        raise ValueError("query_total_pixels must contain integer-valued counts")
    return totals


def surrogate_query_pixel_false_alarm_rate(
    threshold_logits: torch.Tensor,
    background_logits: torch.Tensor,
    query_total_pixels: torch.Tensor,
    *,
    background_valid: torch.Tensor | None = None,
    background_representation: BackgroundRepresentation = FULL_BACKGROUND,
    background_weights: torch.Tensor | None = None,
    temperature: float = 0.10,
    exact_chunk_size: int = 262_144,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute differentiable pixel false-alarm risk in float64.

    ``full`` means that every valid background logit is present exactly once;
    its population weight is therefore one.  ``weighted_stratified`` means a
    deterministic/stratified compact representation whose positive weights
    give the number of population background pixels represented by each
    entry.  The weights must cover the complete background population; this
    function never invents weights for a subset.

    ``exact_chunk_size`` limits only temporary memory.  Every valid entry is
    still evaluated, so changing it cannot change the mathematical objective
    except for floating-point reduction order.
    """

    _require_float_matrix(threshold_logits, name="threshold_logits")
    _require_float_matrix(
        background_logits,
        name="background_logits",
        batch_size=threshold_logits.shape[0],
    )
    if threshold_logits.shape[1] == 0:
        raise ValueError("threshold_logits must contain at least one budget")
    if not bool(torch.isfinite(threshold_logits).all().item()):
        raise ValueError("threshold_logits must be finite")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if (
        isinstance(exact_chunk_size, bool)
        or not isinstance(exact_chunk_size, int)
        or exact_chunk_size <= 0
    ):
        raise ValueError("exact_chunk_size must be a positive integer")
    if background_representation not in (
        FULL_BACKGROUND,
        WEIGHTED_STRATIFIED_BACKGROUND,
    ):
        raise ValueError(
            "background_representation must be 'full' or 'weighted_stratified'"
        )

    device = threshold_logits.device
    calculation_dtype = torch.float64
    background = background_logits.to(device=device, dtype=calculation_dtype)
    valid = _valid_mask(
        background_valid,
        like=background_logits,
        name="background_valid",
    ).to(device=device)
    _ensure_finite_where_valid(background, valid, name="background_logits")
    totals = _query_pixel_totals(
        query_total_pixels,
        batch_size=threshold_logits.shape[0],
        device=device,
    )

    if background_representation == FULL_BACKGROUND:
        if background_weights is not None:
            raise ValueError(
                "background_weights are forbidden for full supervision; use "
                "weighted_stratified explicitly for a compact representation"
            )
        weights = valid.to(dtype=calculation_dtype)
    else:
        if background_weights is None:
            raise ValueError(
                "weighted_stratified supervision requires explicit population weights"
            )
        if not isinstance(background_weights, torch.Tensor):
            raise TypeError("background_weights must be a torch.Tensor")
        if background_weights.shape != background_logits.shape:
            raise ValueError("background_weights must match background_logits")
        raw_weights = background_weights.to(
            device=device,
            dtype=calculation_dtype,
        )
        _ensure_finite_where_valid(raw_weights, valid, name="background_weights")
        if bool((raw_weights[valid] <= 0.0).any().item()):
            raise ValueError(
                "valid weighted_stratified entries require positive population weights"
            )
        if bool((raw_weights[~valid] != 0.0).any().item()):
            raise ValueError("padded background entries must have zero weight")
        weights = raw_weights

    population_mass = weights.sum(dim=1)
    tolerance = torch.maximum(totals, torch.ones_like(totals)) * 1e-12
    if bool((population_mass > totals + tolerance).any().item()):
        raise ValueError(
            "represented background population cannot exceed query_total_pixels"
        )

    eta = threshold_logits.to(device=device, dtype=calculation_dtype)
    soft_false_count = eta.new_zeros(eta.shape)
    for start in range(0, background.shape[1], exact_chunk_size):
        stop = min(start + exact_chunk_size, background.shape[1])
        chunk_valid = valid[:, start:stop]
        chunk_logits = torch.where(
            chunk_valid,
            background[:, start:stop],
            torch.zeros((), device=device, dtype=calculation_dtype),
        )
        chunk_weights = weights[:, start:stop]
        active = torch.sigmoid(
            (chunk_logits[:, None, :] - eta[:, :, None]) / float(temperature)
        )
        soft_false_count = soft_false_count + (
            active * chunk_weights[:, None, :]
        ).sum(dim=2)

    return soft_false_count / totals[:, None], population_mass


def surrogate_query_detection_probability(
    threshold_logits: torch.Tensor,
    object_logits: torch.Tensor,
    *,
    object_valid: torch.Tensor | None = None,
    temperature: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return soft per-object detection rate and valid-episode mask."""

    _require_float_matrix(threshold_logits, name="threshold_logits")
    _require_float_matrix(
        object_logits,
        name="object_logits",
        batch_size=threshold_logits.shape[0],
    )
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    valid = _valid_mask(
        object_valid,
        like=object_logits,
        name="object_valid",
    ).to(device=threshold_logits.device)
    objects = object_logits.to(
        device=threshold_logits.device,
        dtype=torch.float64,
    )
    _ensure_finite_where_valid(objects, valid, name="object_logits")
    safe_objects = torch.where(
        valid,
        objects,
        torch.zeros((), device=objects.device, dtype=objects.dtype),
    )
    eta = threshold_logits.to(dtype=torch.float64)
    detected = torch.sigmoid(
        (safe_objects[:, None, :] - eta[:, :, None]) / float(temperature)
    )
    valid_float = valid.to(dtype=torch.float64)
    object_count = valid_float.sum(dim=1)
    probability = (
        detected * valid_float[:, None, :]
    ).sum(dim=2) / object_count.clamp_min(1.0)[:, None]
    return probability, object_count > 0.0


def log10_budget_curve_smoothness(
    threshold_logits: torch.Tensor,
    pixel_budgets: torch.Tensor,
) -> torch.Tensor:
    """Mean squared second derivative over the log10 budget coordinate."""

    _require_float_matrix(threshold_logits, name="threshold_logits")
    budgets = _budget_matrix(
        pixel_budgets,
        threshold_logits=threshold_logits,
    )
    eta = threshold_logits.to(dtype=torch.float64)
    if eta.shape[1] < 3:
        return eta.sum() * 0.0
    coordinate = torch.log10(budgets)
    interval = coordinate[:, 1:] - coordinate[:, :-1]
    slopes = (eta[:, 1:] - eta[:, :-1]) / interval
    midpoint_interval = 0.5 * (interval[:, 1:] + interval[:, :-1])
    second_derivative = (slopes[:, 1:] - slopes[:, :-1]) / midpoint_interval
    return second_derivative.square().mean()


def query_risk_aligned_calibrator_loss(
    threshold_logits: torch.Tensor,
    pixel_budgets: torch.Tensor,
    oracle_threshold_logits: torch.Tensor,
    background_logits: torch.Tensor,
    query_total_pixels: torch.Tensor,
    object_logits: torch.Tensor,
    *,
    background_valid: torch.Tensor | None = None,
    background_representation: BackgroundRepresentation = FULL_BACKGROUND,
    background_weights: torch.Tensor | None = None,
    object_valid: torch.Tensor | None = None,
    oracle_valid: torch.Tensor | None = None,
    lambda_violation: float = 4.0,
    lambda_utility: float = 1.0,
    lambda_oracle_logit: float = 0.10,
    lambda_curve_smoothness: float = 0.01,
    pixel_temperature: float = 0.10,
    object_temperature: float = 0.20,
    epsilon: float = 1e-12,
    oracle_huber_delta: float = 1.0,
    exact_background_chunk_size: int = 262_144,
) -> CalibratorRiskLossOutput:
    """Optimise risk violation and utility on a disjoint labelled query.

    Query labels supervise this loss during meta-training only.  The Stage-2
    calibrator itself still receives only unlabeled support/context features.
    All risk calculations use the complete ``[B, J]`` threshold curve.
    """

    _require_float_matrix(threshold_logits, name="threshold_logits")
    if threshold_logits.shape[0] == 0 or threshold_logits.shape[1] == 0:
        raise ValueError("threshold_logits must have non-empty [B, J] dimensions")
    if not bool(torch.isfinite(threshold_logits).all().item()):
        raise ValueError("threshold_logits must be finite")
    weights_to_check = (
        lambda_violation,
        lambda_utility,
        lambda_oracle_logit,
        lambda_curve_smoothness,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in weights_to_check):
        raise ValueError("loss weights must be finite and non-negative")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(float(oracle_huber_delta)) or float(oracle_huber_delta) <= 0.0:
        raise ValueError("oracle_huber_delta must be finite and positive")

    budgets = _budget_matrix(
        pixel_budgets,
        threshold_logits=threshold_logits,
    )
    pixel_risk, population_mass = surrogate_query_pixel_false_alarm_rate(
        threshold_logits,
        background_logits,
        query_total_pixels,
        background_valid=background_valid,
        background_representation=background_representation,
        background_weights=background_weights,
        temperature=pixel_temperature,
        exact_chunk_size=exact_background_chunk_size,
    )
    detection_probability, has_object = surrogate_query_detection_probability(
        threshold_logits,
        object_logits,
        object_valid=object_valid,
        temperature=object_temperature,
    )

    log_excess = torch.log((pixel_risk + float(epsilon)) / (budgets + float(epsilon)))
    violation = F.relu(log_excess).square().mean()
    if bool(has_object.any().item()):
        utility = (1.0 - detection_probability[has_object]).mean()
    else:
        utility = threshold_logits.to(dtype=torch.float64).sum() * 0.0

    _require_float_matrix(
        oracle_threshold_logits,
        name="oracle_threshold_logits",
        batch_size=threshold_logits.shape[0],
    )
    if oracle_threshold_logits.shape != threshold_logits.shape:
        raise ValueError("oracle_threshold_logits must match threshold_logits")
    oracle_mask = _valid_mask(
        oracle_valid,
        like=oracle_threshold_logits,
        name="oracle_valid",
    ).to(device=threshold_logits.device)
    oracle_values = oracle_threshold_logits.to(
        device=threshold_logits.device,
        dtype=torch.float64,
    )
    _ensure_finite_where_valid(
        oracle_values,
        oracle_mask,
        name="oracle_threshold_logits",
    )
    if bool(oracle_mask.any().item()):
        oracle_logit = F.huber_loss(
            threshold_logits.to(dtype=torch.float64)[oracle_mask],
            oracle_values[oracle_mask],
            reduction="mean",
            delta=float(oracle_huber_delta),
        )
    else:
        oracle_logit = threshold_logits.to(dtype=torch.float64).sum() * 0.0

    curve_smoothness = log10_budget_curve_smoothness(
        threshold_logits,
        pixel_budgets,
    )
    total = (
        float(lambda_violation) * violation
        + float(lambda_utility) * utility
        + float(lambda_oracle_logit) * oracle_logit
        + float(lambda_curve_smoothness) * curve_smoothness
    )
    return CalibratorRiskLossOutput(
        total=total,
        violation=violation,
        utility=utility,
        oracle_logit=oracle_logit,
        curve_smoothness=curve_smoothness,
        surrogate_pixel_false_alarm_rate=pixel_risk,
        surrogate_detection_probability=detection_probability,
        background_population_mass=population_mass,
        background_representation=background_representation,
    )


def _curve_inputs(
    curve_logits: torch.Tensor,
    curve_pixel_risk: torch.Tensor,
    curve_pd: torch.Tensor,
    curve_valid: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    for value, name in (
        (curve_logits, "curve_logits"),
        (curve_pixel_risk, "curve_pixel_risk"),
        (curve_pd, "curve_pd"),
    ):
        _require_float_matrix(value, name=name, batch_size=batch_size)
    if curve_pixel_risk.shape != curve_logits.shape or curve_pd.shape != curve_logits.shape:
        raise ValueError("curve logits/risk/pd arrays must share shape [B, K]")
    if (
        not isinstance(curve_valid, torch.Tensor)
        or curve_valid.shape != curve_logits.shape
        or curve_valid.dtype != torch.bool
    ):
        raise ValueError("curve_valid must be a bool tensor with shape [B, K]")

    logits = curve_logits.to(device=device, dtype=torch.float64)
    risk = curve_pixel_risk.to(device=device, dtype=torch.float64)
    pd = curve_pd.to(device=device, dtype=torch.float64)
    valid = curve_valid.to(device=device)
    for values, name in (
        (logits, "curve_logits"),
        (risk, "curve_pixel_risk"),
        (pd, "curve_pd"),
    ):
        _ensure_finite_where_valid(values, valid, name=name)
    for values, name in ((risk, "curve_pixel_risk"), (pd, "curve_pd")):
        selected = values[valid]
        if bool(((selected < 0.0) | (selected > 1.0)).any().item()):
            raise ValueError(f"valid {name} entries must lie in [0, 1]")

    for row in range(batch_size):
        row_logits = logits[row, valid[row]]
        if row_logits.numel() < 2:
            raise ValueError("every curve row requires at least two valid points")
        if not bool((row_logits[1:] > row_logits[:-1]).all().item()):
            raise ValueError("valid curve_logits must be strictly ascending per row")
    return logits, risk, pd, valid


def _interpolate_padded_exact_curves(
    query_logits: torch.Tensor,
    curve_logits: torch.Tensor,
    curve_values: torch.Tensor,
    curve_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Piecewise-linear interpolation with endpoint clamping, never extrapolation."""

    interpolated_rows: list[torch.Tensor] = []
    clamped_query_rows: list[torch.Tensor] = []
    low_rows: list[torch.Tensor] = []
    high_rows: list[torch.Tensor] = []
    for row in range(query_logits.shape[0]):
        valid = curve_valid[row]
        coordinate = curve_logits[row, valid]
        values = curve_values[row, valid]
        query = query_logits[row]
        low = query < coordinate[0]
        high = query > coordinate[-1]
        clamped = query.clamp(min=coordinate[0], max=coordinate[-1])
        right = torch.searchsorted(coordinate, clamped, right=True).clamp(
            1, coordinate.numel() - 1
        )
        left = right - 1
        x0 = coordinate[left]
        x1 = coordinate[right]
        y0 = values[left]
        y1 = values[right]
        weight = (clamped - x0) / (x1 - x0)
        interpolated_rows.append(y0 + weight * (y1 - y0))
        clamped_query_rows.append(clamped)
        low_rows.append(low)
        high_rows.append(high)
    return (
        torch.stack(interpolated_rows),
        torch.stack(clamped_query_rows),
        torch.stack(low_rows),
        torch.stack(high_rows),
    )


def curve_query_risk_aligned_calibrator_loss(
    threshold_logits: torch.Tensor,
    pixel_budgets: torch.Tensor,
    oracle_threshold_logits: torch.Tensor,
    curve_logits: torch.Tensor,
    curve_pixel_risk: torch.Tensor,
    curve_pd: torch.Tensor,
    curve_valid: torch.Tensor,
    exact_lower_bound: torch.Tensor,
    global_exact: torch.Tensor,
    *,
    oracle_valid: torch.Tensor | None = None,
    utility_episode_valid: torch.Tensor | None = None,
    lambda_violation: float = 4.0,
    lambda_utility: float = 1.0,
    lambda_oracle_logit: float = 0.10,
    lambda_curve_smoothness: float = 0.01,
    lambda_coverage: float = 4.0,
    epsilon: float = 1e-12,
    oracle_huber_delta: float = 1.0,
) -> CurveCalibratorRiskLossOutput:
    """Train from padded verified query curves without raw pixel subsampling.

    ``curve_logits`` must be strictly ascending within each row's
    ``curve_valid`` mask.  Risk and Pd are linearly interpolated in threshold
    logit space.  ``exact_lower_bound`` is therefore also a threshold logit
    (the grouped dataset field ``curve_exact_lower_logit``), not a probability.
    Queries outside the represented curve are endpoint-clamped;
    values are never extrapolated.  For a non-global curve, predictions below
    its verified exact lower bound additionally receive a squared logit-space
    coverage penalty.  Globally exact curves receive no coverage penalty and
    may safely use either endpoint clamp.
    """

    _require_float_matrix(threshold_logits, name="threshold_logits")
    if threshold_logits.shape[0] == 0 or threshold_logits.shape[1] == 0:
        raise ValueError("threshold_logits must have non-empty [B, J] dimensions")
    if not bool(torch.isfinite(threshold_logits).all().item()):
        raise ValueError("threshold_logits must be finite")
    loss_weights = (
        lambda_violation,
        lambda_utility,
        lambda_oracle_logit,
        lambda_curve_smoothness,
        lambda_coverage,
    )
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in loss_weights
    ):
        raise ValueError("loss weights must be finite and non-negative")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if (
        not math.isfinite(float(oracle_huber_delta))
        or float(oracle_huber_delta) <= 0.0
    ):
        raise ValueError("oracle_huber_delta must be finite and positive")
    batch_size = threshold_logits.shape[0]
    device = threshold_logits.device
    budgets = _budget_matrix(pixel_budgets, threshold_logits=threshold_logits)
    logits, risk_values, pd_values, valid = _curve_inputs(
        curve_logits,
        curve_pixel_risk,
        curve_pd,
        curve_valid,
        batch_size=batch_size,
        device=device,
    )

    if not isinstance(exact_lower_bound, torch.Tensor):
        raise TypeError("exact_lower_bound must be a torch.Tensor")
    lower_bound = exact_lower_bound.to(device=device, dtype=torch.float64)
    if lower_bound.shape != (batch_size,) or not bool(
        torch.isfinite(lower_bound).all().item()
    ):
        raise ValueError("exact_lower_bound must be finite with shape [B]")
    if (
        not isinstance(global_exact, torch.Tensor)
        or global_exact.shape != (batch_size,)
        or global_exact.dtype != torch.bool
    ):
        raise ValueError("global_exact must be a bool tensor with shape [B]")
    global_mask = global_exact.to(device=device)

    curve_min = torch.stack([logits[row, valid[row]][0] for row in range(batch_size)])
    curve_max = torch.stack([logits[row, valid[row]][-1] for row in range(batch_size)])
    if bool(((~global_mask) & (lower_bound > curve_max)).any().item()):
        raise ValueError(
            "a non-global exact_lower_bound cannot exceed its largest curve logit"
        )
    # If the declared exact suffix starts below the first stored point, the
    # first stored point is the actual differentiable coverage boundary.
    effective_lower = torch.maximum(lower_bound, curve_min)
    eta = threshold_logits.to(dtype=torch.float64)
    non_global = (~global_mask)[:, None]
    coverage_shortfall = F.relu(effective_lower[:, None] - eta) * non_global
    if bool(non_global.any().item()):
        coverage_penalty = coverage_shortfall.square().sum() / (
            non_global.expand_as(eta).sum().to(dtype=torch.float64)
        )
    else:
        coverage_penalty = eta.sum() * 0.0

    # Below-bound non-global predictions are evaluated conservatively at the
    # first verified point.  The coverage term supplies the missing gradient.
    evaluation_logits = torch.where(
        non_global,
        torch.maximum(eta, effective_lower[:, None]),
        eta,
    )
    pixel_risk, interpolation_logits, clamped_low, clamped_high = (
        _interpolate_padded_exact_curves(
            evaluation_logits,
            logits,
            risk_values,
            valid,
        )
    )
    detection_probability, pd_interpolation_logits, pd_low, pd_high = (
        _interpolate_padded_exact_curves(
            evaluation_logits,
            logits,
            pd_values,
            valid,
        )
    )
    if not torch.equal(interpolation_logits, pd_interpolation_logits) or not torch.equal(
        clamped_low, pd_low
    ) or not torch.equal(clamped_high, pd_high):
        raise RuntimeError("risk and Pd curve interpolation boundaries disagree")

    log_excess = torch.log((pixel_risk + float(epsilon)) / (budgets + float(epsilon)))
    violation = F.relu(log_excess).square().mean()
    if utility_episode_valid is None:
        utility_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
    else:
        if (
            not isinstance(utility_episode_valid, torch.Tensor)
            or utility_episode_valid.shape != (batch_size,)
            or utility_episode_valid.dtype != torch.bool
        ):
            raise ValueError(
                "utility_episode_valid must be a bool tensor with shape [B]"
            )
        utility_mask = utility_episode_valid.to(device=device)
    if bool(utility_mask.any().item()):
        utility = (1.0 - detection_probability[utility_mask]).mean()
    else:
        utility = eta.sum() * 0.0

    _require_float_matrix(
        oracle_threshold_logits,
        name="oracle_threshold_logits",
        batch_size=batch_size,
    )
    if oracle_threshold_logits.shape != threshold_logits.shape:
        raise ValueError("oracle_threshold_logits must match threshold_logits")
    oracle_mask = _valid_mask(
        oracle_valid,
        like=oracle_threshold_logits,
        name="oracle_valid",
    ).to(device=device)
    oracle_values = oracle_threshold_logits.to(device=device, dtype=torch.float64)
    _ensure_finite_where_valid(
        oracle_values,
        oracle_mask,
        name="oracle_threshold_logits",
    )
    if bool(oracle_mask.any().item()):
        oracle_logit = F.huber_loss(
            eta[oracle_mask],
            oracle_values[oracle_mask],
            reduction="mean",
            delta=float(oracle_huber_delta),
        )
    else:
        oracle_logit = eta.sum() * 0.0
    curve_smoothness = log10_budget_curve_smoothness(
        threshold_logits,
        pixel_budgets,
    )

    total = (
        float(lambda_violation) * violation
        + float(lambda_utility) * utility
        + float(lambda_oracle_logit) * oracle_logit
        + float(lambda_curve_smoothness) * curve_smoothness
        + float(lambda_coverage) * coverage_penalty
    )
    return CurveCalibratorRiskLossOutput(
        total=total,
        violation=violation,
        utility=utility,
        oracle_logit=oracle_logit,
        curve_smoothness=curve_smoothness,
        coverage_penalty=coverage_penalty,
        surrogate_pixel_false_alarm_rate=pixel_risk,
        surrogate_detection_probability=detection_probability,
        coverage_shortfall_logits=coverage_shortfall,
        interpolation_logits=interpolation_logits,
        interpolation_clamped_low=clamped_low,
        interpolation_clamped_high=clamped_high,
    )


# A concise compatibility spelling for callers migrating from the engineering
# candidate.  It preserves the strict full/weighted-stratified contract above.
risk_aligned_calibrator_loss = query_risk_aligned_calibrator_loss


__all__ = [
    "CalibratorRiskLossOutput",
    "CurveCalibratorRiskLossOutput",
    "FULL_BACKGROUND",
    "WEIGHTED_STRATIFIED_BACKGROUND",
    "calibrator_risk_capability_contract",
    "curve_query_risk_aligned_calibrator_loss",
    "log10_budget_curve_smoothness",
    "query_risk_aligned_calibrator_loss",
    "risk_aligned_calibrator_loss",
    "surrogate_query_detection_probability",
    "surrogate_query_pixel_false_alarm_rate",
]

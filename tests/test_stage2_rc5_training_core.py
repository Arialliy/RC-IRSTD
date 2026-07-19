from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from losses.calibrator_risk import curve_query_risk_aligned_calibrator_loss
from model.endpoint_aware_pixel_calibrator import (
    DirectEndpointAwarePixelCalibrator,
    MonotoneEndpointAwarePixelCalibrator,
)
from model.endpoint_aware_threshold import (
    MAX_INTERIOR_COORDINATE,
    UPPER_ENDPOINT_COORDINATE,
    decode_coordinate_numpy,
    encode_probability_numpy,
)
import rc.stage2_rc5_training_core as core
from rc.stage2_compositional_curve_provider import (
    build_compositional_exact_curve_provider,
    build_per_image_exact_event_curve,
    build_per_image_exact_event_curve_bank,
)
from rc.stage2_rc5_training_core import (
    RC5_LOSS_METRIC_NAMES,
    Stage2CurveCoordinateView,
    Stage2RC5TrainingCoreError,
    compact_exact_curve_coordinate_brackets,
    oracle_coordinate_huber_loss,
    rc5_batch_loss,
)


PIXEL_BUDGETS = (1e-4, 1e-5, 1e-6)


def _coordinate_tensor(
    probabilities: list[float] | tuple[float, ...], *, batch_size: int = 1
) -> torch.Tensor:
    coordinates = encode_probability_numpy(
        np.asarray(probabilities, dtype=np.float64)
    )
    return torch.from_numpy(coordinates).reshape(1, -1).repeat(batch_size, 1)


def _exact_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds = np.asarray(
        [0.0, 0.05, 0.20, 0.40, 0.50, 0.65, 0.80, 0.90, 0.97, 0.995, 1.0],
        dtype=np.float64,
    )
    risk = np.asarray(
        [1e-2, 8e-3, 3e-3, 8e-4, 3e-4, 8e-5, 9e-6, 8e-7, 5e-8, 1e-9, 0.0],
        dtype=np.float64,
    )
    pd = np.asarray(
        [1.0, 0.995, 0.98, 0.94, 0.90, 0.82, 0.68, 0.48, 0.24, 0.08, 0.0],
        dtype=np.float64,
    )
    return thresholds, risk, pd


def _synthetic_batch(batch_size: int = 1) -> dict[str, object]:
    thresholds, risk, pd = _exact_curve()
    return {
        "features": torch.tensor(
            [[0.2, -0.4, 0.7, 0.1], [-0.3, 0.6, 0.4, -0.2]][:batch_size],
            dtype=torch.float32,
        ),
        "anchor_coordinates": _coordinate_tensor(
            [0.10, 0.55, 0.90], batch_size=batch_size
        ),
        "oracle_coordinates": _coordinate_tensor(
            [0.25, 0.70, 0.98], batch_size=batch_size
        ),
        "pixel_budgets": torch.tensor(
            [PIXEL_BUDGETS] * batch_size, dtype=torch.float64
        ),
        "curve_gt_objects": torch.ones(batch_size, dtype=torch.int64),
        "curve_coordinates": tuple(
            Stage2CurveCoordinateView(thresholds) for _ in range(batch_size)
        ),
        "curve_pixel_risk": tuple(risk.copy() for _ in range(batch_size)),
        "curve_pd": tuple(pd.copy() for _ in range(batch_size)),
    }


def _risk_loss_config() -> dict[str, float]:
    return {
        "lambda_violation": 1.0,
        "lambda_utility": 0.5,
        "lambda_oracle": 1.0,
        "lambda_smoothness": 0.01,
        "lambda_coverage": 0.0,
        "risk_epsilon": 1e-12,
        "coordinate_huber_delta": 1.0,
    }


class _FixedCoordinateModel(nn.Module):
    def __init__(self, coordinates: torch.Tensor) -> None:
        super().__init__()
        self.coordinates = coordinates
        self.calls = 0

    def forward(
        self, features: torch.Tensor, *, anchor_coordinates: torch.Tensor
    ) -> SimpleNamespace:
        del features, anchor_coordinates
        self.calls += 1
        return SimpleNamespace(grid_coordinates=self.coordinates)


def test_lazy_view_owns_thresholds_without_materializing_full_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_encode = core.encode_probability_numpy
    encoded_sizes: list[int] = []

    def record_encode(value: object) -> np.ndarray:
        encoded_sizes.append(int(np.asarray(value).size))
        return original_encode(value)

    monkeypatch.setattr(core, "encode_probability_numpy", record_encode)
    source = np.linspace(0.0, 1.0, 1_000_000, dtype=np.float64)
    view = Stage2CurveCoordinateView(source)

    assert encoded_sizes == []
    assert view.nbytes == 0
    assert not view.thresholds.flags.writeable
    assert not np.shares_memory(view.thresholds, source)
    source[0] = 0.25
    assert view.thresholds[0] == 0.0

    chosen = view.take(np.asarray([0, 500_000, 999_999], dtype=np.int64))
    assert encoded_sizes == [3]
    assert chosen[0] == 0.0
    assert chosen[-1] == UPPER_ENDPOINT_COORDINATE
    assert np.all(np.diff(chosen) > 0.0)


@pytest.mark.parametrize(
    "thresholds, message",
    [
        (np.asarray([0.0, 1.0], dtype=np.float32), "float64 vector"),
        (np.asarray([0.0], dtype=np.float64), "at least two"),
        (np.asarray([0.1, 1.0], dtype=np.float64), "exact 0/1"),
        (np.asarray([0.0, 0.9], dtype=np.float64), "exact 0/1"),
        (np.asarray([0.0, 0.5, 0.5, 1.0], dtype=np.float64), "deduplicated"),
        (np.asarray([0.0, 0.7, 0.6, 1.0], dtype=np.float64), "ascending"),
        (np.asarray([0.0, np.nan, 1.0], dtype=np.float64), "finite"),
    ],
)
def test_curve_view_rejects_wrong_types_missing_endpoints_and_ties(
    thresholds: np.ndarray, message: str
) -> None:
    with pytest.raises(Stage2RC5TrainingCoreError, match=message):
        Stage2CurveCoordinateView(thresholds)


def test_curve_view_requires_explicit_arrays_and_integer_take_indices() -> None:
    with pytest.raises(Stage2RC5TrainingCoreError, match="explicit numpy"):
        Stage2CurveCoordinateView([0.0, 1.0])  # type: ignore[arg-type]

    view = Stage2CurveCoordinateView(
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    )
    with pytest.raises(TypeError, match="integer vector"):
        view.take(np.asarray([0.0, 1.0], dtype=np.float64))
    with pytest.raises(TypeError, match="integer vector"):
        view.take(np.asarray([True, False]))
    with pytest.raises(IndexError, match="out of range"):
        view.take(np.asarray([-1, 1], dtype=np.int64))


def test_exact_tail_event_brackets_search_in_coordinate_space() -> None:
    # Encoding this value and decoding it returns the preceding binary64
    # probability.  A probability-space search therefore chooses a different
    # interval at the exact event and can change its interpolation gradient.
    event = float.fromhex("0x1.76185e7716c9bp-1")
    event_coordinate = float(
        encode_probability_numpy(np.asarray(event, dtype=np.float64))
    )
    assert float(decode_coordinate_numpy(np.asarray(event_coordinate))) != event
    thresholds = np.asarray(
        [
            0.0,
            0.40,
            np.nextafter(event, 0.0),
            event,
            np.nextafter(event, 1.0),
            0.95,
            1.0,
        ],
        dtype=np.float64,
    )
    view = Stage2CurveCoordinateView(thresholds)
    query = encode_probability_numpy(
        np.asarray([0.40, event, 0.95], dtype=np.float64)
    )
    full_coordinates = encode_probability_numpy(thresholds)
    right = np.searchsorted(full_coordinates, query, side="right")
    right = np.clip(right, 1, thresholds.size - 1)
    expected = np.unique(np.concatenate((right - 1, right)))

    np.testing.assert_array_equal(view.bracket_union(query), expected)


def test_bracket_queries_are_strict_float64_canonical_eatc_arrays() -> None:
    view = Stage2CurveCoordinateView(
        np.asarray([0.0, 0.5, 0.9, 1.0], dtype=np.float64)
    )
    with pytest.raises(TypeError, match="explicit float64"):
        view.bracket_union([0.0, 0.5, UPPER_ENDPOINT_COORDINATE])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="explicit float64"):
        view.bracket_union(
            np.asarray([0.0, 0.5, UPPER_ENDPOINT_COORDINATE], dtype=np.float32)
        )
    noncanonical = np.asarray(
        [0.0, MAX_INTERIOR_COORDINATE + 0.25, UPPER_ENDPOINT_COORDINATE],
        dtype=np.float64,
    )
    with pytest.raises(Stage2RC5TrainingCoreError, match="noncanonical"):
        view.bracket_union(noncanonical)


def test_compact_and_full_coordinate_interpolation_match_values_and_gradients() -> None:
    thresholds, risk, pd = _exact_curve()
    full_coordinates = torch.from_numpy(
        encode_probability_numpy(thresholds)
    ).reshape(1, -1)
    full_risk = torch.from_numpy(risk).reshape(1, -1)
    full_pd = torch.from_numpy(pd).reshape(1, -1)
    full_eta = _coordinate_tensor([0.12, 0.58, 0.96]).requires_grad_(True)
    compact_eta = full_eta.detach().clone().requires_grad_(True)
    budgets = torch.tensor([PIXEL_BUDGETS], dtype=torch.float64)
    oracle = _coordinate_tensor([0.15, 0.62, 0.97])
    full = curve_query_risk_aligned_calibrator_loss(
        full_eta,
        budgets,
        oracle,
        full_coordinates,
        full_risk,
        full_pd,
        torch.ones_like(full_coordinates, dtype=torch.bool),
        full_coordinates[:, 0],
        torch.ones(1, dtype=torch.bool),
    )
    compact_coordinates, compact_risk, compact_pd, compact_valid = (
        compact_exact_curve_coordinate_brackets(
            compact_eta,
            {
                "curve_coordinates": (Stage2CurveCoordinateView(thresholds),),
                "curve_pixel_risk": (risk,),
                "curve_pd": (pd,),
            },
        )
    )
    compact = curve_query_risk_aligned_calibrator_loss(
        compact_eta,
        budgets,
        oracle,
        compact_coordinates,
        compact_risk,
        compact_pd,
        compact_valid,
        compact_coordinates[:, 0],
        torch.ones(1, dtype=torch.bool),
    )

    for name in (
        "total",
        "violation",
        "utility",
        "oracle_logit",
        "curve_smoothness",
        "coverage_penalty",
        "surrogate_pixel_false_alarm_rate",
        "surrogate_detection_probability",
        "coverage_shortfall_logits",
        "interpolation_logits",
    ):
        torch.testing.assert_close(
            getattr(compact, name), getattr(full, name), rtol=0.0, atol=1e-14
        )
    for name in ("interpolation_clamped_low", "interpolation_clamped_high"):
        assert torch.equal(getattr(compact, name), getattr(full, name))
    full.total.backward()
    compact.total.backward()
    torch.testing.assert_close(
        compact_eta.grad, full_eta.grad, rtol=0.0, atol=1e-14
    )


def test_live_compositional_provider_matches_materialized_curve_values_and_gradients() -> None:
    thresholds, risk, pd = _exact_curve()
    total_pixels = 1_000_000_000
    gt_objects = 200
    false_positive_pixels = np.rint(risk * total_pixels).astype(np.int64)
    matched_objects = np.rint(pd * gt_objects).astype(np.int64)
    curves = tuple(
        build_per_image_exact_event_curve(
            image_identity_sha256=hashlib.sha256(
                f"provider-equivalence-{index}".encode()
            ).hexdigest(),
            thresholds=thresholds,
            false_positive_pixels=false_positive_pixels,
            matched_objects=matched_objects,
            total_native_pixels=total_pixels,
            ground_truth_objects=gt_objects,
        )
        for index in range(28)
    )
    bank = build_per_image_exact_event_curve_bank(curves)
    provider = build_compositional_exact_curve_provider(
        curve_bank=bank,
        ordered_image_identities=tuple(
            curve.image_identity_sha256 for curve in curves
        ),
    )
    materialized_eta = _coordinate_tensor([0.12, 0.58, 0.96]).requires_grad_(True)
    provider_eta = materialized_eta.detach().clone().requires_grad_(True)
    materialized = compact_exact_curve_coordinate_brackets(
        materialized_eta,
        {
            "curve_coordinates": (Stage2CurveCoordinateView(thresholds),),
            "curve_pixel_risk": (risk,),
            "curve_pd": (pd,),
        },
    )
    live = compact_exact_curve_coordinate_brackets(
        provider_eta,
        {"compositional_curve_providers": (provider,)},
    )
    for live_value, materialized_value in zip(live, materialized, strict=True):
        if live_value.dtype == torch.bool:
            assert torch.equal(live_value, materialized_value)
        else:
            torch.testing.assert_close(
                live_value, materialized_value, rtol=0.0, atol=0.0
            )
    assert live[0].shape[1] <= 6

    budgets = torch.tensor([PIXEL_BUDGETS], dtype=torch.float64)
    oracle = _coordinate_tensor([0.15, 0.62, 0.97])
    materialized_loss = curve_query_risk_aligned_calibrator_loss(
        materialized_eta,
        budgets,
        oracle,
        materialized[0],
        materialized[1],
        materialized[2],
        materialized[3],
        materialized[0][:, 0],
        torch.ones(1, dtype=torch.bool),
    )
    provider_loss = curve_query_risk_aligned_calibrator_loss(
        provider_eta,
        budgets,
        oracle,
        live[0],
        live[1],
        live[2],
        live[3],
        live[0][:, 0],
        torch.ones(1, dtype=torch.bool),
    )
    for name in (
        "total",
        "violation",
        "utility",
        "oracle_logit",
        "curve_smoothness",
        "coverage_penalty",
        "surrogate_pixel_false_alarm_rate",
        "surrogate_detection_probability",
        "coverage_shortfall_logits",
        "interpolation_logits",
    ):
        torch.testing.assert_close(
            getattr(provider_loss, name),
            getattr(materialized_loss, name),
            rtol=0.0,
            atol=0.0,
        )
    materialized_loss.total.backward()
    provider_loss.total.backward()
    torch.testing.assert_close(
        provider_eta.grad, materialized_eta.grad, rtol=0.0, atol=0.0
    )

    with pytest.raises(Stage2RC5TrainingCoreError, match="mutually exclusive"):
        compact_exact_curve_coordinate_brackets(
            provider_eta.detach(),
            {
                "compositional_curve_providers": (provider,),
                "curve_coordinates": (Stage2CurveCoordinateView(thresholds),),
            },
        )


def test_compact_path_rejects_legacy_logits_and_every_padded_curve_tensor() -> None:
    predicted = _coordinate_tensor([0.12, 0.58, 0.96])
    base = _synthetic_batch()
    legacy = dict(base)
    legacy["curve_logits"] = torch.zeros((1, 8), dtype=torch.float64)
    with pytest.raises(Stage2RC5TrainingCoreError, match="clipped-logit"):
        compact_exact_curve_coordinate_brackets(predicted, legacy)

    for field in ("curve_coordinates", "curve_pixel_risk", "curve_pd"):
        padded = dict(base)
        padded[field] = torch.zeros((1, 8), dtype=torch.float64)
        with pytest.raises(Stage2RC5TrainingCoreError, match="padded tensor"):
            compact_exact_curve_coordinate_brackets(predicted, padded)


def test_compact_path_rejects_implicit_or_non_float64_ragged_values() -> None:
    predicted = _coordinate_tensor([0.12, 0.58, 0.96])
    base = _synthetic_batch()
    for replacement in (
        [0.1] * len(base["curve_coordinates"][0]),
        np.asarray(base["curve_pixel_risk"][0], dtype=np.float32),
        torch.as_tensor(base["curve_pixel_risk"][0], dtype=torch.float32),
    ):
        invalid = dict(base)
        invalid["curve_pixel_risk"] = (replacement,)
        with pytest.raises(Stage2RC5TrainingCoreError, match="CPU|float64|explicit"):
            compact_exact_curve_coordinate_brackets(predicted, invalid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compact_path_rejects_padded_or_ragged_gpu_curve_storage() -> None:
    predicted = _coordinate_tensor([0.12, 0.58, 0.96]).cuda()
    base = _synthetic_batch()
    padded = dict(base)
    padded["curve_coordinates"] = torch.zeros(
        (1, 8), dtype=torch.float64, device="cuda"
    )
    with pytest.raises(Stage2RC5TrainingCoreError, match="padded tensor"):
        compact_exact_curve_coordinate_brackets(predicted, padded)

    gpu_ragged = dict(base)
    gpu_ragged["curve_pixel_risk"] = (
        torch.as_tensor(
            base["curve_pixel_risk"][0], dtype=torch.float64, device="cuda"
        ),
    )
    with pytest.raises(Stage2RC5TrainingCoreError, match="ragged CPU"):
        compact_exact_curve_coordinate_brackets(predicted, gpu_ragged)


def test_million_event_curve_selects_at_most_six_without_full_tensorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted = _coordinate_tensor([0.1234567, 0.6789123, 0.9876543])
    thresholds = np.linspace(0.0, 1.0, 1_000_000, dtype=np.float64)
    risk = np.linspace(1e-2, 0.0, thresholds.size, dtype=np.float64)
    pd = np.linspace(1.0, 0.0, thresholds.size, dtype=np.float64)
    original_encode = core.encode_probability_numpy
    original_from_numpy = core.torch.from_numpy
    encoded_sizes: list[int] = []
    tensorized_sizes: list[int] = []

    def record_encode(value: object) -> np.ndarray:
        encoded_sizes.append(int(np.asarray(value).size))
        return original_encode(value)

    def record_from_numpy(value: np.ndarray) -> torch.Tensor:
        tensorized_sizes.append(int(value.size))
        return original_from_numpy(value)

    monkeypatch.setattr(core, "encode_probability_numpy", record_encode)
    monkeypatch.setattr(core.torch, "from_numpy", record_from_numpy)
    view = Stage2CurveCoordinateView(thresholds)
    compact_coordinates, compact_risk, compact_pd, valid = (
        compact_exact_curve_coordinate_brackets(
            predicted,
            {
                "curve_coordinates": (view,),
                "curve_pixel_risk": (risk,),
                "curve_pd": (pd,),
            },
        )
    )

    assert view.nbytes == 0
    assert compact_coordinates.shape[1] <= 6
    assert int(valid.sum().item()) <= 6
    assert compact_risk.shape == compact_pd.shape == valid.shape == compact_coordinates.shape
    assert max(encoded_sizes) <= 6
    assert max(tensorized_sizes) <= 6
    assert isinstance(view, Stage2CurveCoordinateView)
    assert not isinstance(view, torch.Tensor)


def test_coordinate_huber_rejects_noncanonical_and_non_float64_inputs() -> None:
    predicted = _coordinate_tensor([0.1, 0.6, 0.95])
    oracle = _coordinate_tensor([0.2, 0.7, 0.97])
    valid = torch.ones_like(predicted, dtype=torch.bool)
    with pytest.raises(TypeError, match="float64"):
        oracle_coordinate_huber_loss(predicted.float(), oracle, valid)
    with pytest.raises(TypeError, match="float64"):
        oracle_coordinate_huber_loss(predicted, oracle.float(), valid)
    invalid = predicted.clone()
    invalid[0, 1] = MAX_INTERIOR_COORDINATE + 0.25
    with pytest.raises(Stage2RC5TrainingCoreError, match="noncanonical"):
        oracle_coordinate_huber_loss(invalid, oracle, valid)


@pytest.mark.parametrize("method", ["T6", "T7", "T8"])
def test_all_methods_route_correctly_and_have_finite_nonzero_gradients(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(4100 + int(method[-1]))
    batch = _synthetic_batch(batch_size=2)
    if method == "T6":
        model: nn.Module = DirectEndpointAwarePixelCalibrator(
            4, PIXEL_BUDGETS, hidden_dims=(8,), dropout=0.0
        )
    else:
        model = MonotoneEndpointAwarePixelCalibrator(
            4, PIXEL_BUDGETS, hidden_dims=(8,), dropout=0.0
        )
    original_risk_loss = core.curve_query_risk_aligned_calibrator_loss
    risk_calls = 0

    def record_risk_loss(*args: object, **kwargs: object) -> object:
        nonlocal risk_calls
        risk_calls += 1
        return original_risk_loss(*args, **kwargs)

    monkeypatch.setattr(
        core, "curve_query_risk_aligned_calibrator_loss", record_risk_loss
    )
    if method in {"T6", "T7"}:
        minimal_batch = {
            key: batch[key]
            for key in ("features", "anchor_coordinates", "oracle_coordinates")
        }
        loss_config: dict[str, float] = {"coordinate_huber_delta": 1.0}
    else:
        minimal_batch = batch
        loss_config = _risk_loss_config()

    _, metrics = rc5_batch_loss(
        method=method,
        model=model,
        batch=minimal_batch,
        loss_config=loss_config,
    )

    assert tuple(metrics) == RC5_LOSS_METRIC_NAMES
    assert all(bool(torch.isfinite(value).item()) for value in metrics.values())
    assert float(metrics["total"].item()) > 0.0
    if method in {"T6", "T7"}:
        assert risk_calls == 0
        torch.testing.assert_close(
            metrics["total"], metrics["oracle_coordinate"], rtol=0.0, atol=0.0
        )
        for name in (
            "violation",
            "utility",
            "curve_smoothness",
            "coverage_penalty",
        ):
            assert float(metrics[name].item()) == 0.0
    else:
        assert risk_calls == 1

    metrics["total"].backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
    assert sum(float(gradient.abs().sum().item()) for gradient in gradients) > 0.0
    assert model.anchor_mix_logit.grad is not None
    assert float(model.anchor_mix_logit.grad.abs().item()) > 0.0


def test_anchor_is_required_before_the_model_or_loss_is_invoked() -> None:
    predicted = _coordinate_tensor([0.1, 0.6, 0.95])
    model = _FixedCoordinateModel(predicted)
    batch = _synthetic_batch()
    batch.pop("anchor_coordinates")

    with pytest.raises(TypeError, match="anchor_coordinates"):
        rc5_batch_loss(
            method="T6",
            model=model,
            batch=batch,
            loss_config={"coordinate_huber_delta": 1.0},
        )
    assert model.calls == 0


def test_t8_rejects_budget_and_object_count_dtype_migrations() -> None:
    predicted = _coordinate_tensor([0.1, 0.6, 0.95])
    model = _FixedCoordinateModel(predicted)
    batch = _synthetic_batch()
    bad_budget = copy.copy(batch)
    bad_budget["pixel_budgets"] = batch["pixel_budgets"].float()
    with pytest.raises(TypeError, match="pixel_budgets.*float64"):
        rc5_batch_loss(
            method="T8",
            model=model,
            batch=bad_budget,
            loss_config=_risk_loss_config(),
        )

    bad_count = copy.copy(batch)
    bad_count["curve_gt_objects"] = torch.ones(1, dtype=torch.float64)
    with pytest.raises(TypeError, match="integer count"):
        rc5_batch_loss(
            method="T8",
            model=model,
            batch=bad_count,
            loss_config=_risk_loss_config(),
        )

    complex_count = copy.copy(batch)
    complex_count["curve_gt_objects"] = torch.ones(1, dtype=torch.complex64)
    with pytest.raises(TypeError, match="integer count"):
        rc5_batch_loss(
            method="T8",
            model=model,
            batch=complex_count,
            loss_config=_risk_loss_config(),
        )

    negative_count = copy.copy(batch)
    negative_count["curve_gt_objects"] = torch.tensor([-1], dtype=torch.int64)
    with pytest.raises(Stage2RC5TrainingCoreError, match="non-negative"):
        rc5_batch_loss(
            method="T8",
            model=model,
            batch=negative_count,
            loss_config=_risk_loss_config(),
        )

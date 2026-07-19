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
from rc.stage2_rc5_feature_mask import build_stage2_rc5_feature_mask
import rc.train_stage2_rc5plus_cyclic as trainer
from rc.train_stage2_rc5plus_cyclic import (
    Stage2RC5PlusCyclicTrainerError,
    build_rc5plus_training_model,
    collate_rc5plus_cyclic_batch,
    rc5plus_cyclic_optimization_step,
)


def _provider(episode: int):
    thresholds = np.asarray(
        [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0], dtype=np.float64
    )
    fp = np.asarray([3000, 500, 100, 20, 5, 2, 1, 0], dtype=np.int64)
    tp = np.asarray([1, 2, 4, 8, 7, 5, 2, 0], dtype=np.int64)
    curves = []
    for image in range(CYCLIC_QUERY_SIZE):
        identity = hashlib.sha256(f"trainer-{episode}-{image}".encode()).hexdigest()
        curves.append(
            build_per_image_exact_event_curve(
                image_identity_sha256=identity,
                thresholds=thresholds,
                false_positive_pixels=fp,
                matched_objects=tp,
                total_native_pixels=1_000_000,
                ground_truth_objects=10,
            )
        )
    bank = build_per_image_exact_event_curve_bank(curves)
    return build_compositional_exact_curve_provider(
        curve_bank=bank,
        ordered_image_identities=[item.image_identity_sha256 for item in curves],
    )


class _View:
    def __init__(self) -> None:
        self.providers = [_provider(0), _provider(1)]
        self.features = [
            np.linspace(0.0, 1.0, 93, dtype=np.float32),
            np.linspace(1.0, 2.0, 93, dtype=np.float32),
        ]
        self.anchors = [
            encode_probability_numpy(np.linspace(0.2, 0.9, 9)),
            encode_probability_numpy(np.linspace(0.25, 0.95, 9)),
        ]

    def feature_anchor_for_episode(self, domain: str, index: int):
        del domain
        return self.features[index], self.anchors[index]

    def feature_for_episode(self, domain: str, index: int):
        del domain
        return self.features[index]

    def provider_for_episode(self, domain: str, index: int):
        del domain
        return self.providers[index]


def _rows():
    return (
        {"source_domain": "A", "domain_episode_index": 0},
        {"source_domain": "B", "domain_episode_index": 1},
    )


def _model_config():
    return {
        "context_feature_dim": 93,
        "hidden_dims": [32],
        "dropout": 0.0,
        "minimum_residual_increment": 1e-6,
    }


def _loss_config():
    return {
        "lambda_violation": 1.0,
        "lambda_utility": 0.5,
        "lambda_oracle": 1.0,
        "lambda_smoothness": 0.01,
        "lambda_coverage": 0.0,
        "risk_epsilon": 1e-12,
        "coordinate_huber_delta": 1.0,
    }


@pytest.fixture(autouse=True)
def _view_capability(monkeypatch):
    monkeypatch.setattr(
        trainer,
        "assert_verified_stage2_rc5plus_cyclic_training_view",
        lambda value: value,
    )


@pytest.mark.parametrize(
    ("method", "model_type"),
    [
        ("T6_PLUS", BudgetConditionedDirectResidualTransportCalibrator),
        ("T7_PLUS", BudgetConditionedMonotoneResidualTransportCalibrator),
        ("T8_PLUS", BudgetConditionedMonotoneResidualTransportCalibrator),
        (
            "T8_PLUS_NO_ANCHOR",
            BudgetConditionedMonotoneNoTargetAnchorCalibrator,
        ),
    ],
)
def test_model_builder_and_cyclic_route_execute_one_finite_step(method, model_type) -> None:
    model = build_rc5plus_training_model(method, _model_config())
    assert type(model) is model_type
    batch = collate_rc5plus_cyclic_batch(
        collection=_View(),
        ordered_rows=_rows(),
        standardizer_mean=np.zeros(93, dtype=np.float64),
        standardizer_scale=np.ones(93, dtype=np.float64),
        feature_mask=build_stage2_rc5_feature_mask("C3"),
        device="cpu",
        method=method,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.transport_head.weight.detach().clone()

    output, losses = rc5plus_cyclic_optimization_step(
        method=method,
        model=model,
        optimizer=optimizer,
        batch=batch,
        loss_config=_loss_config(),
        gradient_clip_norm=5.0,
    )

    assert output.grid_coordinates.shape == (2, 9)
    assert all(torch.isfinite(value) for value in losses.values())
    assert not torch.equal(before, model.transport_head.weight.detach())
    assert batch["budget_numerators"].dtype == torch.int64
    assert batch["budget_denominators"].dtype == torch.int64
    assert batch["budget_numerators"].shape == (2, 9)
    assert batch["budget_denominators"][0].tolist() == [
        row[1] for row in BUDGET_KNOT_RATIONALS
    ]
    if method in {"T8_PLUS", "T8_PLUS_NO_ANCHOR"}:
        assert "compositional_curve_providers" in batch
        assert "oracle_coordinates" not in batch
    else:
        assert "oracle_coordinates" in batch
        assert "compositional_curve_providers" not in batch
    if method == "T8_PLUS_NO_ANCHOR":
        assert "anchor_coordinates" not in batch
    else:
        assert "anchor_coordinates" in batch


def test_collate_rejects_unbalanced_domains_and_inexact_standardizer() -> None:
    unbalanced = (
        {"source_domain": "A", "domain_episode_index": 0},
        {"source_domain": "A", "domain_episode_index": 1},
    )
    with pytest.raises(Stage2RC5PlusCyclicTrainerError, match="balanced"):
        collate_rc5plus_cyclic_batch(
            collection=_View(),
            ordered_rows=unbalanced,
            standardizer_mean=np.zeros(93, dtype=np.float64),
            standardizer_scale=np.ones(93, dtype=np.float64),
            feature_mask=build_stage2_rc5_feature_mask("C3"),
            device="cpu",
            method="T8_PLUS",
        )
    with pytest.raises(Stage2RC5PlusCyclicTrainerError, match="float64"):
        collate_rc5plus_cyclic_batch(
            collection=_View(),
            ordered_rows=_rows(),
            standardizer_mean=np.zeros(93, dtype=np.float32),
            standardizer_scale=np.ones(93, dtype=np.float64),
            feature_mask=build_stage2_rc5_feature_mask("C3"),
            device="cpu",
            method="T8_PLUS",
        )


def test_model_config_and_sampler_row_field_closures_fail_closed() -> None:
    wrong_config = _model_config()
    wrong_config["pixel_budget_grid"] = [1e-4, 1e-5, 1e-6]
    with pytest.raises(Stage2RC5PlusCyclicTrainerError, match="field closure"):
        build_rc5plus_training_model("T8_PLUS", wrong_config)
    with pytest.raises(Stage2RC5PlusCyclicTrainerError, match="lacks"):
        collate_rc5plus_cyclic_batch(
            collection=_View(),
            ordered_rows=(
                {"source_domain": "A", "unexpected": 0},
                {"source_domain": "B", "unexpected": 1},
            ),
            standardizer_mean=np.zeros(93, dtype=np.float64),
            standardizer_scale=np.ones(93, dtype=np.float64),
            feature_mask=build_stage2_rc5_feature_mask("C3"),
            device="cpu",
            method="T8_PLUS",
        )


@pytest.mark.parametrize("variant", ["C3", "C4", "C5", "C6"])
def test_collate_applies_the_verified_mask_after_standardization(variant) -> None:
    mask = build_stage2_rc5_feature_mask(variant)
    mean = np.linspace(-0.5, 0.5, 93, dtype=np.float64)
    scale = np.linspace(0.5, 1.5, 93, dtype=np.float64)
    batch = collate_rc5plus_cyclic_batch(
        collection=_View(),
        ordered_rows=_rows(),
        standardizer_mean=mean,
        standardizer_scale=scale,
        feature_mask=mask,
        device="cpu",
        method="T7_PLUS",
    )
    observed = batch["features"].numpy()
    raw = np.stack(_View().features).astype(np.float64)
    expected = ((raw - mean) / scale).astype(np.float32)
    if mask.inactive_indices:
        expected[:, list(mask.inactive_indices)] = np.float32(0.0)
        inactive = observed[:, list(mask.inactive_indices)]
        assert np.array_equal(inactive, np.zeros_like(inactive))
        assert not np.signbit(inactive).any()
    assert np.array_equal(observed, expected)


def test_collate_requires_a_verifier_issued_feature_mask() -> None:
    with pytest.raises(TypeError, match="verifier-issued"):
        collate_rc5plus_cyclic_batch(
            collection=_View(),
            ordered_rows=_rows(),
            standardizer_mean=np.zeros(93, dtype=np.float64),
            standardizer_scale=np.ones(93, dtype=np.float64),
            feature_mask={"variant": "C3"},  # type: ignore[arg-type]
            device="cpu",
            method="T8_PLUS",
        )

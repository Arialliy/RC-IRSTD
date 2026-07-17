from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rc.train_stage2_crossfit_calibrator as trainer
from model.direct_no_reject_pixel_calibrator import DirectNoRejectPixelCalibrator
from rc.stage2_crossfit_dataset import Stage2CurveLogitView, collate_stage2_crossfit_batch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "aaai27_stage2_crossfit_v2.json"
SEED_MANIFEST = (
    ROOT
    / "outputs"
    / "stage2_protocol"
    / "RC4_STAGE2_SEED_DERIVATION_MANIFEST_V1_20260716.json"
)
SHA = "a" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> trainer.VerifiedStage2CrossfitConfig:
    return trainer.verify_stage2_crossfit_config(CONFIG, _sha(CONFIG))


def _batch(rows: int = 2) -> dict[str, object]:
    items = []
    for index in range(rows):
        items.append(
            {
                "features": torch.linspace(-1.0, 1.0, 93, dtype=torch.float32),
                "pixel_budgets": torch.tensor([1e-4, 1e-5, 1e-6], dtype=torch.float64),
                "oracle_thresholds": torch.tensor([0.4, 0.6, 0.8], dtype=torch.float64),
                "oracle_logits": torch.logit(
                    torch.tensor([0.4, 0.6, 0.8], dtype=torch.float64)
                ),
                "oracle_pd": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float64),
                "oracle_pixel_risk": torch.tensor(
                    [8e-5, 8e-6, 8e-7], dtype=torch.float64
                ),
                "curve_thresholds": torch.tensor(
                    [0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64
                ),
                "curve_logits": torch.logit(
                    torch.tensor(
                        [0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64
                    ).clamp(1e-12, 1.0 - 1e-12)
                ),
                "curve_pixel_risk": torch.tensor(
                    [1e-2, 1e-3, 1e-5, 1e-7, 0.0], dtype=torch.float64
                ),
                "curve_pd": torch.tensor([1.0, 0.9, 0.8, 0.5, 0.0], dtype=torch.float64),
                "curve_fp_pixels": torch.tensor([100, 10, 1, 0, 0], dtype=torch.int64),
                "curve_tp_objects": torch.tensor([10, 9, 8, 5, 0], dtype=torch.int64),
                "curve_total_pixels": torch.tensor(10000, dtype=torch.int64),
                "curve_gt_objects": torch.tensor(10, dtype=torch.int64),
                "episode_id": f"episode-{index}",
                "window_id": f"window-{index}",
                "source_domain": "NUDT-SIRST" if index % 2 == 0 else "IRSTD-1K",
                "context_identity_sha256": hashlib.sha256(
                    f"context-{index}".encode()
                ).hexdigest(),
                "source_query_identity_sha256": hashlib.sha256(
                    f"query-{index}".encode()
                ).hexdigest(),
            }
        )
    return collate_stage2_crossfit_batch(items)


class _FakeStandardizer:
    feature_names = tuple(f"f{index}" for index in range(93))
    mean = np.zeros(93, dtype=np.float64)
    scale = np.ones(93, dtype=np.float64)
    fit_records_sha256 = SHA
    fit_manifest_sha256 = "b" * 64
    train_collection_sha256 = "c" * 64


def test_frozen_config_builds_exact_models() -> None:
    config = _config()
    assert config.payload["model"]["hidden_dims"] == [32]
    for method, expected in (("T6", 3107), ("T7", 3140), ("T8", 3140)):
        model = trainer.build_stage2_model(method, config)
        assert trainer.trainable_parameter_count(model) == expected
        assert model.export_config()["hidden_dims"] == [32]
        assert model.export_config()["min_logit"] == -10.0
        assert model.export_config()["max_logit"] == 18.0
        assert model.capability_contract()["supports_reject"] is False


def test_config_is_exact_key_and_exact_type_closed(tmp_path: Path) -> None:
    payload = _config().payload
    payload["model"]["reject_head"] = 0
    path = tmp_path / "bad.json"
    path.write_text(trainer.json.dumps(payload), encoding="utf-8")
    with pytest.raises(trainer.Stage2CalibratorContractError):
        trainer.verify_stage2_crossfit_config(path.resolve(), _sha(path))
    payload = _config().payload
    payload["unexpected"] = False
    path.write_text(trainer.json.dumps(payload), encoding="utf-8")
    with pytest.raises(trainer.Stage2CalibratorContractError):
        trainer.verify_stage2_crossfit_config(path.resolve(), _sha(path))


def test_direct_model_is_bounded_nonmonotone_and_exact_grid_only() -> None:
    model = DirectNoRejectPixelCalibrator(
        93, [1e-4, 1e-5, 1e-6], hidden_dims=[32], dropout=0.1
    )
    with torch.no_grad():
        model.encoder[0].weight.zero_()
        model.encoder[0].bias.zero_()
        model.threshold_head.weight.zero_()
        model.threshold_head.bias.copy_(torch.tensor([-5.0, 5.0, 0.0]))
    model.eval()
    output = model(torch.full((2, 93), 1e6))
    assert output.grid_logits.shape == (2, 3)
    assert torch.all(output.grid_logits >= -10.0)
    assert torch.all(output.grid_logits <= 18.0)
    assert not torch.all(output.grid_logits[:, 1:] > output.grid_logits[:, :-1])
    exact = model(
        torch.zeros(2, 93),
        pixel_budgets=torch.tensor([1e-4, 1e-5, 1e-6], dtype=torch.float64),
    )
    assert torch.equal(exact.grid_logits, exact.requested_logits)
    with pytest.raises(ValueError, match="complete trained budget grid"):
        model(
            torch.zeros(2, 93),
            pixel_budgets=torch.tensor([1e-4, 2e-5, 1e-6], dtype=torch.float64),
        )


def test_monotone_models_are_strict_on_extreme_contexts() -> None:
    config = _config()
    for method in ("T7", "T8"):
        model = trainer.build_stage2_model(method, config).eval()
        output = model(torch.full((3, 93), 1e3))
        assert torch.all(output.grid_logits[:, 1:] > output.grid_logits[:, :-1])
        assert torch.all(output.grid_thresholds[:, 1:] > output.grid_thresholds[:, :-1])


def test_oracle_huber_mask_and_empty_mask_fail_closed() -> None:
    predicted = torch.tensor([[0.0, 2.0, 9.0]], requires_grad=True)
    oracle = torch.tensor([[0.0, 0.0, float("nan")]])
    valid = torch.tensor([[True, True, False]])
    loss = trainer.oracle_logit_huber_loss(predicted, oracle, valid, delta=1.0)
    assert float(loss.detach()) == pytest.approx(0.75)
    loss.backward()
    assert predicted.grad is not None
    with pytest.raises(trainer.Stage2CalibratorContractError, match="no valid"):
        trainer.oracle_logit_huber_loss(predicted.detach(), oracle, torch.zeros_like(valid))


def test_real_lane_a_collate_routes_all_three_objectives() -> None:
    config = _config()
    batch = _batch()
    for method in trainer.METHODS:
        model = trainer.build_stage2_model(method, config)
        output, losses = trainer._batch_loss(method, model, batch, config.payload["loss"])
        assert output.grid_logits.shape == (2, 3)
        assert set(losses) == set(trainer.LOSS_METRIC_NAMES)
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
    assert float(
        trainer._batch_loss(
            "T6", trainer.build_stage2_model("T6", config), batch, config.payload["loss"]
        )[1]["violation"].detach()
    ) == 0.0
    assert float(
        trainer._batch_loss(
            "T8", trainer.build_stage2_model("T8", config), batch, config.payload["loss"]
        )[1]["violation"].detach()
    ) >= 0.0


def test_rank_is_lexicographic_and_exact_tie_keeps_earlier() -> None:
    metrics = {
        "macro_source_BSR": 0.5,
        "macro_source_LogExcess": 0.25,
        "macro_source_Pd": 0.75,
    }
    early = trainer.checkpoint_rank(metrics, 2)
    late = trainer.checkpoint_rank(metrics, 3)
    assert early > late
    assert trainer.is_better_checkpoint(early, None)
    assert not trainer.is_better_checkpoint(late, early)
    with pytest.raises(trainer.Stage2CalibratorContractError):
        trainer.checkpoint_rank({**metrics, "macro_source_BSR": float("nan")}, 0)


def test_seed_manifest_recomputes_all_rows_and_selects_frozen_roles(tmp_path: Path) -> None:
    digest = _sha(SEED_MANIFEST)
    expected = {"T6": 700658138, "T7": 1365542576, "T8": 214361673}
    for method, value in expected.items():
        selected = trainer.verify_stage2_seed_selection(
            SEED_MANIFEST,
            digest,
            base_seed=42,
            outer_fold_id="outer_leave_nuaa_sirst",
            method=method,
        )
        assert selected.derived_seed == value
        assert len(selected.row_sha256) == 64
    corrupted = SEED_MANIFEST.read_text(encoding="utf-8").replace(
        "700658138", "700658139", 1
    )
    path = tmp_path / "seed.json"
    path.write_text(corrupted, encoding="utf-8")
    with pytest.raises(trainer.Stage2CalibratorContractError, match="seed table mismatch"):
        trainer.verify_stage2_seed_selection(
            path.resolve(),
            _sha(path),
            base_seed=42,
            outer_fold_id="outer_leave_nuaa_sirst",
            method="T6",
        )


def _group(domain: str, episode_id: str, risk: float, pd: float, gt: int = 10):
    episode = SimpleNamespace(
        episode_id=episode_id,
        payload={
            "source_domain": domain,
            "window_binding": {"window_id": f"window-{episode_id}"},
        },
    )
    return SimpleNamespace(
        episode=episode,
        curve_thresholds=np.asarray([0.0, 0.5, 1.0]),
        curve_pixel_risk=np.asarray([1e-2, risk, 0.0]),
        curve_pd=np.asarray([1.0, pd, 0.0]),
        curve_tp_objects=np.asarray([gt, round(pd * gt), 0], dtype=np.int64),
        curve_fp_pixels=np.asarray([100, int(risk * 100000), 0]),
        curve_total_pixels=100000,
        curve_gt_objects=gt,
    )


def test_source_replay_uses_primary_budget_and_equal_domain_weight(monkeypatch) -> None:
    import rc.stage2_crossfit_dataset as lane_a

    monkeypatch.setattr(
        lane_a, "assert_stage2_trainer_replay_capability", lambda value, validation: value
    )
    validation = SimpleNamespace(
        manifest={"outer_fold_id": "outer_leave_nuaa_sirst"}
    )
    groups = (
        _group("NUDT-SIRST", "a", 5e-6, 0.8),
        _group("IRSTD-1K", "b", 2e-5, 0.4),
    )
    result = trainer.evaluate_source_validation_primary(
        validation, groups, np.full((2, 3), 0.5), object()
    )
    assert result["selection_pixel_budget"] == 1e-5
    assert result["selection_budget_index"] == 1
    assert result["macro_source_BSR"] == 0.5
    assert result["macro_source_LogExcess"] == pytest.approx(math.log(2.0) / 2.0)
    assert result["macro_source_Pd"] == pytest.approx(0.6)
    with pytest.raises(trainer.Stage2CalibratorContractError, match="1e-5"):
        trainer.evaluate_source_validation_primary(
            validation,
            groups,
            np.full((2, 3), 0.5),
            object(),
            selection_budget=1e-4,
        )


def test_source_replay_pools_exact_integer_tp_not_pd_round_trip(monkeypatch) -> None:
    import rc.stage2_crossfit_dataset as lane_a

    monkeypatch.setattr(
        lane_a, "assert_stage2_trainer_replay_capability", lambda value, validation: value
    )
    validation = SimpleNamespace(
        manifest={"outer_fold_id": "outer_leave_nuaa_sirst"}
    )
    groups = (
        _group("NUDT-SIRST", "a0", 5e-6, 1 / 3, gt=3),
        _group("NUDT-SIRST", "a1", 5e-6, 2 / 7, gt=7),
        _group("IRSTD-1K", "b0", 5e-6, 4 / 5, gt=5),
        _group("IRSTD-1K", "b1", 5e-6, 2 / 5, gt=5),
    )
    result = trainer.evaluate_source_validation_primary(
        validation, groups, np.full((4, 3), 0.5), object()
    )
    assert result["domain_metrics"]["NUDT-SIRST"]["Pd"] == 3 / 10
    assert result["domain_metrics"]["IRSTD-1K"]["Pd"] == 6 / 10
    assert result["macro_source_Pd"] == pytest.approx(0.45)
    assert result["complete_three_budget_window_records"][0]["tp_objects"][1] == 1
    assert result["complete_three_budget_window_records"][0]["gt_objects"] == 3


def test_source_replay_preserves_large_integer_sufficient_counts(monkeypatch) -> None:
    import rc.stage2_crossfit_dataset as lane_a

    monkeypatch.setattr(
        lane_a, "assert_stage2_trainer_replay_capability", lambda value, validation: value
    )
    validation = SimpleNamespace(
        manifest={"outer_fold_id": "outer_leave_nuaa_sirst"}
    )
    counts = {
        "NUDT-SIRST": (2**53 + 1, 2**53 + 3),
        "IRSTD-1K": (2**53 + 5, 2**53 + 7),
    }
    groups = []
    for index, (domain, (tp, gt)) in enumerate(counts.items()):
        group = _group(domain, f"large-{index}", 5e-6, tp / gt, gt=gt)
        group.curve_tp_objects = np.asarray([gt, tp, 0], dtype=np.int64)
        group.curve_pd = np.asarray([1.0, tp / gt, 0.0], dtype=np.float64)
        groups.append(group)
    result = trainer.evaluate_source_validation_primary(
        validation, tuple(groups), np.full((2, 3), 0.5), object()
    )
    for domain, (tp, gt) in counts.items():
        metrics = result["domain_metrics"][domain]
        assert type(metrics["tp_objects"]) is int
        assert type(metrics["gt_objects"]) is int
        assert metrics["tp_objects"] == tp
        assert metrics["gt_objects"] == gt


def test_million_event_curves_remain_ragged_cpu_and_compact_to_six_points() -> None:
    point_count = 1_000_001
    full_logits = torch.linspace(-20.0, 20.0, point_count, dtype=torch.float64)
    full_risk = torch.sigmoid(-full_logits) * 1e-3
    full_pd = torch.sigmoid(-full_logits)
    batch = {
        "curve_logits": (full_logits,),
        "curve_pixel_risk": (full_risk,),
        "curve_pd": (full_pd,),
    }
    eta = torch.tensor([[-5.0, 0.0, 5.0]], dtype=torch.float64, requires_grad=True)
    logits, risk, pd, valid = trainer.compact_exact_curve_brackets(eta, batch)
    assert logits.shape[1] <= 6
    assert risk.shape == logits.shape == pd.shape == valid.shape
    assert full_logits.device.type == "cpu" and full_logits.numel() == point_count
    loss = trainer.curve_query_risk_aligned_calibrator_loss(
        eta,
        torch.tensor([1e-4, 1e-5, 1e-6], dtype=torch.float64),
        torch.zeros_like(eta),
        logits,
        risk,
        pd,
        valid,
        logits[:, 0],
        torch.ones(1, dtype=torch.bool),
    )
    loss.total.backward()
    assert eta.grad is not None and torch.isfinite(eta.grad).all()
    with pytest.raises(trainer.Stage2CalibratorContractError, match="padded"):
        trainer._move_batch(
            {"curve_logits": torch.zeros((1, point_count))},
            torch.device("cpu"),
            method="T8",
        )


def test_real_lane_a_numpy_and_lazy_logit_view_stay_ragged() -> None:
    point_count = 10_001
    thresholds = np.linspace(0.0, 1.0, point_count, dtype=np.float64)
    risk = np.linspace(1e-3, 0.0, point_count, dtype=np.float64)
    pd = np.linspace(1.0, 0.0, point_count, dtype=np.float64)
    for values in (thresholds, risk, pd):
        values.setflags(write=False)
    view = Stage2CurveLogitView(thresholds)
    batch = {
        "curve_thresholds": (thresholds,),
        "curve_logits": (view,),
        "curve_pixel_risk": (risk,),
        "curve_pd": (pd,),
    }
    moved = trainer._move_batch(batch, torch.device("cpu"), method="T8")
    assert moved["curve_logits"][0] is view
    assert moved["curve_pixel_risk"][0] is risk
    eta = torch.tensor([[-5.0, 0.0, 5.0]], dtype=torch.float64, requires_grad=True)
    logits, compact_risk, compact_pd, valid = trainer.compact_exact_curve_brackets(
        eta, moved
    )
    assert view.nbytes == 0
    assert logits.shape[1] <= 6
    assert compact_risk.shape == compact_pd.shape == valid.shape == logits.shape
    assert torch.all(logits[:, 1:][valid[:, 1:]] > logits[:, :-1][valid[:, 1:]])


def test_compact_curve_loss_and_gradient_equal_full_piecewise_curve() -> None:
    curve_logits = torch.tensor(
        [[-4.0, -3.0, -1.0, 0.5, 1.5, 3.0, 4.0]], dtype=torch.float64
    )
    curve_risk = torch.tensor(
        [[8e-4, 4e-4, 9e-5, 8e-6, 9e-7, 1e-8, 0.0]], dtype=torch.float64
    )
    curve_pd = torch.tensor(
        [[1.0, 0.98, 0.91, 0.78, 0.60, 0.25, 0.0]], dtype=torch.float64
    )
    budgets = torch.tensor([[1e-4, 1e-5, 1e-6]], dtype=torch.float64)
    oracle = torch.tensor([[-1.5, 0.3, 2.0]], dtype=torch.float64)
    full_eta = torch.tensor([[-2.0, 0.2, 2.2]], dtype=torch.float64, requires_grad=True)
    compact_eta = full_eta.detach().clone().requires_grad_(True)
    full = trainer.curve_query_risk_aligned_calibrator_loss(
        full_eta,
        budgets,
        oracle,
        curve_logits,
        curve_risk,
        curve_pd,
        torch.ones_like(curve_logits, dtype=torch.bool),
        curve_logits[:, 0],
        torch.ones(1, dtype=torch.bool),
    )
    compact_logits, compact_risk, compact_pd, compact_valid = (
        trainer.compact_exact_curve_brackets(
            compact_eta,
            {
                "curve_logits": (curve_logits[0],),
                "curve_pixel_risk": (curve_risk[0],),
                "curve_pd": (curve_pd[0],),
            },
        )
    )
    compact = trainer.curve_query_risk_aligned_calibrator_loss(
        compact_eta,
        budgets,
        oracle,
        compact_logits,
        compact_risk,
        compact_pd,
        compact_valid,
        compact_logits[:, 0],
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
        "interpolation_logits",
    ):
        torch.testing.assert_close(
            getattr(compact, name), getattr(full, name), rtol=0.0, atol=1e-15
        )
    full.total.backward()
    compact.total.backward()
    torch.testing.assert_close(compact_eta.grad, full_eta.grad, rtol=0.0, atol=1e-15)


def test_lane_a_adapter_uses_payload_roles_and_all_six_hashes(monkeypatch) -> None:
    import rc.stage2_crossfit_dataset as lane_a
    import rc.stage2_crossfit_schema as lane_a_schema

    def episodes(role: str, count: int):
        domains = ("NUDT-SIRST", "IRSTD-1K")
        return tuple(
            SimpleNamespace(
                episode_id=f"{role}-{index}",
                payload={
                    "outer_fold_id": "outer_leave_nuaa_sirst",
                    "episode_role": role,
                    "base_seed": 42,
                    "source_domain": domains[index % 2],
                    "official_test_accessed": False,
                },
            )
            for index in range(count)
        )

    train = SimpleNamespace(
        episodes=episodes(lane_a.STAGE2_OOF_FIT, 26),
        manifest={"collection_role": lane_a.COLLECTION_TRAIN},
    )
    validation = SimpleNamespace(
        episodes=episodes(lane_a.SOURCE_DIAGNOSTIC_VALIDATION, 6),
        manifest={"collection_role": lane_a.COLLECTION_VALIDATION},
    )
    calls = []
    statistics_config = SimpleNamespace(
        to_dict=lambda: {"schema_version": "synthetic-statistics-config"}
    )
    verifier_calls = []

    def verify_statistics(path, sha, **kwargs):
        verifier_calls.append((path, sha, kwargs))
        return statistics_config

    monkeypatch.setattr(
        lane_a_schema, "verify_stage2_statistics_config", verify_statistics
    )

    def load(path, sha, **kwargs):
        calls.append((path, sha, kwargs))
        return train if "train" in str(path) else validation

    monkeypatch.setattr(lane_a, "load_stage2_episodes_v5", load)
    monkeypatch.setattr(lane_a, "assert_verified_episode_collection", lambda value: value)
    monkeypatch.setattr(lane_a, "assert_stage2_sample_isolation", lambda a, b: None)
    monkeypatch.setattr(
        lane_a, "assert_stage2_context_standardizer", lambda value: value
    )
    standardizer = SimpleNamespace(
        train_collection_sha256="1" * 64,
        fit_manifest_sha256="2" * 64,
    )
    monkeypatch.setattr(lane_a, "fit_stage2_context_standardizer", lambda value: standardizer)
    capability = object()
    monkeypatch.setattr(
        lane_a, "make_stage2_trainer_replay_capability", lambda a, b, c: capability
    )
    result = trainer.verify_lane_a_training_inputs(
        train_collection="train.jsonl",
        train_collection_sha256="1" * 64,
        train_manifest="train.manifest.json",
        train_manifest_sha256="2" * 64,
        train_commit="train.commit.json",
        train_commit_sha256="3" * 64,
        validation_collection="validation.jsonl",
        validation_collection_sha256="4" * 64,
        validation_manifest="validation.manifest.json",
        validation_manifest_sha256="5" * 64,
        validation_commit="validation.commit.json",
        validation_commit_sha256="6" * 64,
        statistics_config_path="statistics.json",
        statistics_config_sha256="7" * 64,
        outer_fold_id="outer_leave_nuaa_sirst",
        base_seed=42,
    )
    assert result.replay_capability is capability
    assert len(calls) == 2
    assert verifier_calls == [
        ("statistics.json", "7" * 64, {"repository_root": None})
    ]
    assert calls[0][2]["statistics_config"] is statistics_config
    assert calls[1][2]["statistics_config"] is statistics_config
    assert calls[0][2]["collection_manifest_sha256"] == "2" * 64
    assert calls[0][2]["commit_marker_sha256"] == "3" * 64
    assert result.train_binding["episode_count"] == 26
    assert result.validation_binding["episode_count"] == 6
    assert result.statistics_config is statistics_config
    assert result.statistics_config_binding["sha256"] == "7" * 64


def _checkpoint_fixture(method: str = "T6"):
    config = _config()
    model = trainer.build_stage2_model(method, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(123)
    history = [{"epoch": 0, "finite": True}]
    replay = {"epoch": 0, "macro_source_BSR": 0.5}
    contract = {"schema_version": "synthetic", "config_sha256": config.sha256}
    payload = trainer.make_calibrator_checkpoint_v6(
        method=method,
        model=model,
        optimizer=optimizer,
        completed_epoch=0,
        best_epoch=0,
        best_rank=(0.5, -0.25, 0.75, 0),
        epochs_without_improvement=0,
        training_contract=contract,
        standardizer=_FakeStandardizer(),
        data_loader_generator=generator,
        history_sha256=trainer.sha256_bytes(trainer._history_bytes(history)),
        exact_replay_sha256=trainer.sha256_bytes(trainer._replay_bytes(replay)),
        include_cuda_rng=False,
    )
    return config, model, optimizer, generator, history, replay, contract, payload


def test_v6_generation_checkpoint_and_resume_are_closed(tmp_path: Path) -> None:
    _, model, _, _, history, replay, contract, payload = _checkpoint_fixture()
    output = (tmp_path / "run").resolve()
    generation = trainer.publish_calibrator_generation(
        output,
        epoch=0,
        checkpoint_payload=payload,
        history=history,
        exact_replay=replay,
    )
    verified = trainer.verify_calibrator_checkpoint_v6(
        generation.checkpoint_path,
        generation.checkpoint_sha256,
        expected_method="T6",
        expected_training_contract=contract,
    )
    assert verified.completed_epoch == 0
    replayed = trainer._model_from_checkpoint_payload(verified.payload())
    replayed.load_state_dict(verified.payload()["model_state_dict"], strict=True)
    for left, right in zip(model.parameters(), replayed.parameters(), strict=True):
        assert torch.equal(left, right)

    resumed_model = trainer.build_stage2_model("T6", _config())
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_generator = torch.Generator().manual_seed(999)
    state = trainer.resume_calibrator_generation(
        commit_path=generation.commit_path,
        commit_sha256=generation.commit_sha256,
        model=resumed_model,
        optimizer=resumed_optimizer,
        data_loader_generator=resumed_generator,
        expected_method="T6",
        expected_training_contract=contract,
        include_cuda_rng=False,
    )
    assert state[:2] == (1, 0)
    for left, right in zip(model.parameters(), resumed_model.parameters(), strict=True):
        assert torch.equal(left, right)


def test_generation_is_no_replace_and_tamper_evident(tmp_path: Path) -> None:
    *_, history, replay, _, payload = _checkpoint_fixture()
    output = (tmp_path / "run").resolve()
    generation = trainer.publish_calibrator_generation(
        output,
        epoch=0,
        checkpoint_payload=payload,
        history=history,
        exact_replay=replay,
    )
    with pytest.raises(FileExistsError):
        trainer.publish_calibrator_generation(
            output,
            epoch=0,
            checkpoint_payload=payload,
            history=history,
            exact_replay=replay,
        )
    original = generation.checkpoint_path.read_bytes()
    generation.checkpoint_path.write_bytes(original + b"x")
    with pytest.raises(trainer.Stage2CalibratorContractError, match="SHA-256 mismatch"):
        trainer.verify_calibrator_generation(
            generation.commit_path, generation.commit_sha256
        )


def test_atomic_link_failure_rolls_back_without_commit(tmp_path: Path, monkeypatch) -> None:
    *_, history, replay, _, payload = _checkpoint_fixture()
    original_link = trainer.os.link
    calls = 0

    def fail_second_link(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic link failure")
        return original_link(*args, **kwargs)

    monkeypatch.setattr(trainer.os, "link", fail_second_link)
    output = (tmp_path / "failed-run").resolve()
    with pytest.raises(OSError, match="synthetic"):
        trainer.publish_calibrator_generation(
            output,
            epoch=0,
            checkpoint_payload=payload,
            history=history,
            exact_replay=replay,
        )
    assert not (output / "generations" / "epoch_0000").exists()
    assert not list(output.rglob("COMMIT.json"))


def _synthetic_step(model, optimizer, generator, batch):
    indices = torch.randperm(batch["features"].shape[0], generator=generator)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch["features"][indices])
    loss = trainer.oracle_logit_huber_loss(
        output.grid_logits,
        batch["oracle_logits"][indices],
        torch.ones_like(batch["oracle_logits"][indices], dtype=torch.bool),
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return indices


def test_interrupted_resume_matches_uninterrupted_next_step(tmp_path: Path) -> None:
    trainer.seed_runtime(3407)
    config = _config()
    batch = _batch(rows=4)
    model_a = trainer.build_stage2_model("T6", config)
    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3, weight_decay=1e-4)
    generator_a = torch.Generator().manual_seed(55)
    _synthetic_step(model_a, optimizer_a, generator_a, batch)
    history = [{"epoch": 0, "finite": True}]
    replay = {"epoch": 0, "macro_source_BSR": 0.5}
    contract = {"schema_version": "resume-equivalence", "config_sha256": config.sha256}
    payload = trainer.make_calibrator_checkpoint_v6(
        method="T6",
        model=model_a,
        optimizer=optimizer_a,
        completed_epoch=0,
        best_epoch=0,
        best_rank=(0.5, -0.1, 0.7, 0),
        epochs_without_improvement=0,
        training_contract=contract,
        standardizer=_FakeStandardizer(),
        data_loader_generator=generator_a,
        history_sha256=trainer.sha256_bytes(trainer._history_bytes(history)),
        exact_replay_sha256=trainer.sha256_bytes(trainer._replay_bytes(replay)),
        include_cuda_rng=False,
    )
    generation = trainer.publish_calibrator_generation(
        (tmp_path / "resume-source").resolve(),
        epoch=0,
        checkpoint_payload=payload,
        history=history,
        exact_replay=replay,
    )
    uninterrupted_indices = _synthetic_step(model_a, optimizer_a, generator_a, batch)
    uninterrupted_state = {
        key: value.detach().clone() for key, value in model_a.state_dict().items()
    }

    model_b = trainer.build_stage2_model("T6", config)
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3, weight_decay=1e-4)
    generator_b = torch.Generator().manual_seed(999)
    trainer.resume_calibrator_generation(
        commit_path=generation.commit_path,
        commit_sha256=generation.commit_sha256,
        model=model_b,
        optimizer=optimizer_b,
        data_loader_generator=generator_b,
        expected_method="T6",
        expected_training_contract=contract,
        include_cuda_rng=False,
    )
    resumed_indices = _synthetic_step(model_b, optimizer_b, generator_b, batch)
    assert torch.equal(uninterrupted_indices, resumed_indices)
    for key, value in model_b.state_dict().items():
        assert torch.equal(uninterrupted_state[key], value)


def test_v6_verifier_rejects_legacy_and_bound_mutation(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.pt"
    torch.save({"format_version": "rc-irstd.calibrator.v5"}, legacy)
    with pytest.raises(trainer.Stage2CalibratorContractError):
        trainer.verify_calibrator_checkpoint_v6(legacy.resolve(), _sha(legacy))

    *_, payload = _checkpoint_fixture()
    mutated = copy.deepcopy(payload)
    mutated["model_config"]["min_logit"] = -9.0
    path = tmp_path / "mutated.pt"
    torch.save(mutated, path)
    with pytest.raises(trainer.Stage2CalibratorContractError, match="bounds"):
        trainer.verify_calibrator_checkpoint_v6(path.resolve(), _sha(path))


def test_rng_round_trip_and_cpu_checkpoint_contains_no_cuda_state() -> None:
    trainer.seed_runtime(123)
    state = trainer.capture_rng_state(include_cuda=False)
    expected = (np.random.random(), random_value := trainer.random.random(), torch.rand(2))
    trainer.restore_rng_state(state, include_cuda=False)
    observed = (np.random.random(), trainer.random.random(), torch.rand(2))
    assert expected[0] == observed[0]
    assert random_value == observed[1]
    assert torch.equal(expected[2], observed[2])
    assert state["torch_cuda_all"] == []

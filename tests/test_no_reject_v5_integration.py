from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.evaluate_adapter_output import (
    _is_no_reject_adapter,
    evaluate_adapter_output,
)
from evaluation.calibrator_replay import ExactGroupedPixelRiskReplay
from losses.calibrator_risk import calibrator_risk_capability_contract
from model.monotone_pixel_calibrator import MonotoneNoRejectPixelRiskCalibrator
from rc.meta_dataset import FeatureStandardizer
from rc.meta_dataset import PixelRiskEpisodeGroup
from rc.online_adapter import adapt_context_to_query, load_calibrator_bundle
from rc.train_calibrator_risk_aligned import _loss_for_batch
from rc.schema import (
    BudgetSpec,
    NoRejectDeploymentProtocolContract,
    SourceContract,
    SourceReference,
    StatisticsConfig,
    canonicalize_episode_score_split_contract,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _reference() -> SourceReference:
    return SourceReference(
        domains=("S1", "S2"),
        sha256=SHA_B,
        centers=((0.0, 0.0), (1.0, 1.0)),
        scale=(1.0, 1.0),
        contract=SourceContract(
            detector_checkpoint_sha=SHA_A,
            detector_source_domains=("S1", "S2"),
            outer_fold_id="outer-fold",
            outer_target="OUT",
            held_out_domains=("OUT",),
            protocol_scope="multi_source_protocol_candidate",
        ),
    )


def _budget_contract(grid: tuple[float, ...]) -> dict[str, object]:
    canonical = (
        '{"extrapolation_allowed":false,"grid":['
        + ",".join(str(value) for value in grid)
        + '],"grid_order":"loose_to_strict",'
        '"interpolation":"piecewise_linear_log10","risk":"fa_pixel"}'
    ).encode("utf-8")
    supervision = {"all_grid_points_supervised": True}
    return {
        "schema_version": "rc-irstd.monotone-pixel-budget.v1",
        "risk": "fa_pixel",
        "component_budget_supported": False,
        "grid": list(grid),
        "grid_order": "loose_to_strict",
        "grid_policy_sha256": hashlib.sha256(canonical).hexdigest(),
        "interpolation": "piecewise_linear_log10",
        "extrapolation_allowed": False,
        "curve_compute_dtype": "float64",
        "train_supervision": supervision,
        "validation_supervision": supervision,
        "method_supports_reject": False,
        "grouped_complete_curve_supervision": True,
        "query_supervision": "verified_event_exact_or_global_exact",
        "checkpoint_selection": "exact_native_replay_BSR_LogExcess_Pd",
    }


def _split_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "role": "official_train",
        "selected_split_file": "train.txt",
        "selected_split_sha256": SHA_A,
        "selected_num_images": 2,
        "selected_ids_sha256": SHA_B,
        "official_train_split_file": "train.txt",
        "official_train_split_sha256": SHA_A,
        "official_train_num_images": 2,
        "official_train_ids_sha256": SHA_B,
        "official_train_split_image_artifact_sha256": SHA_A,
        "official_test_split_file": "test.txt",
        "official_test_split_sha256": SHA_B,
        "official_test_num_images": 1,
        "official_test_ids_sha256": SHA_A,
        "official_test_split_image_artifact_sha256": SHA_B,
        "ordered_sample_ids_algorithm": "test-order-v1",
        "split_image_artifact_algorithm": "test-artifact-v1",
        "train_test_id_overlap_count": 0,
        "train_test_id_overlap_ids": [],
        "train_test_image_content_overlap_count": 0,
        "train_test_image_content_overlap_sha256_leaves": [],
        "disjointness_verified": True,
    }
    return canonicalize_episode_score_split_contract(payload)


def _audit_row(partition: str) -> dict[str, object]:
    split = _split_contract()
    canonical = json.dumps(
        split,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "partition": partition,
        "num_episodes": 2,
        "split_contract_sha256": hashlib.sha256(canonical).hexdigest(),
        "split_contract": split,
    }


def _group_row(group_id: str, target: str) -> dict[str, object]:
    return {
        "group_id": group_id,
        "pseudo_target": target,
        "context_image_ids": [f"{target}-c"],
        "query_image_ids": [f"{target}-q"],
        "curve_file_sha256": SHA_A,
        "curve_manifest_sha256": SHA_B,
        "query_score_manifest_sha256": SHA_A,
        "label_manifest_sha256": SHA_B,
        "label_manifest_content_sha256": SHA_A,
        "matching_rule": "overlap",
        "centroid_distance": 3.0,
    }


def _checkpoint() -> tuple[dict[str, object], MonotoneNoRejectPixelRiskCalibrator]:
    grid = (1e-4, 1e-5, 1e-6)
    model = MonotoneNoRejectPixelRiskCalibrator(
        context_feature_dim=2,
        pixel_budget_grid=grid,
        hidden_dims=(4,),
        dropout=0.0,
    )
    reference = _reference()
    protocol = NoRejectDeploymentProtocolContract(context_size=1, query_size=1)
    checkpoint: dict[str, object] = {
        "format_version": "rc-irstd.calibrator.v5",
        "calibrator_model": "monotone_pixel_no_reject",
        "model_state_dict": model.state_dict(),
        "model_config": model.export_config(),
        "capability_contract": model.capability_contract(),
        "risk_loss_contract": calibrator_risk_capability_contract(),
        "monotone_budget_contract": _budget_contract(grid),
        "input_dim": 2,
        "hidden_dim": 4,
        "dropout": 0.0,
        "standardizer": FeatureStandardizer(
            ("f0", "f1"), np.zeros(2), np.ones(2)
        ).to_dict(),
        "statistics_feature_names": ["f0", "f1"],
        "input_feature_names": ["f0", "f1"],
        "statistics_config": StatisticsConfig(
            peak_kernel_size=3, peak_min_score=0.1
        ).to_dict(),
        "outer_fold_id": "outer-fold",
        "outer_target": "OUT",
        "episode_collection_provenance": {},
        "episode_collection_sha256": SHA_B,
        "training_config": {
            "evaluation_matching_rule": "overlap",
            "evaluation_centroid_distance": 3.0,
            "query_curve_mode": "verified_event_exact",
            "hard_replay": "native_resolution_every_epoch",
            "threshold_semantics": "prediction = probability > threshold",
            "pixel_budget_grid": list(grid),
        },
        "official_train_score_provenance": {
            "schema_version": "rc-irstd.calibrator-official-train-provenance.v1",
            "required_episode_schema": "rc-irstd.meta-episode.v4",
            "required_score_split_role": "official_train",
            "pseudo_target_validation_may_select_best_checkpoint": True,
            "official_test_scores_consumed": False,
            "num_episodes": 4,
            "pseudo_targets": {
                "A": _audit_row("calibrator_train"),
                "B": _audit_row("pseudo_target_validation"),
            },
        },
        "train_pseudo_targets": ["A"],
        "validation_pseudo_targets": ["B"],
        "calibration_pseudo_targets": ["A", "B"],
        "deployment_detector_source_domains": ["S1", "S2"],
        "deployment_detector_checkpoint_sha": SHA_A,
        "deployment_held_out_domains": ["OUT"],
        "deployment_protocol_scope": "multi_source_protocol_candidate",
        "deployment_source_reference": reference.to_dict(),
        "deployment_protocol_contract": protocol.to_dict(),
        "reject_head": False,
        "artifact_root_persisted": False,
        "train_group_provenance": [_group_row("train", "A")],
        "validation_group_provenance": [_group_row("validation", "B")],
        "checkpoint_selection_order": ["BSR", "LogExcess", "Pd"],
        "risk_guarantee": "empirical_meta_calibration_not_certified",
        "epoch": 0,
        "best_epoch": 0,
        "best_rank": [1.0, 0.0, 0.5],
        "validation_metrics": {
            "rank_key": [1.0, 0.0, 0.5],
            "checkpoint_selection_order": ["BSR", "LogExcess", "Pd"],
        },
    }
    return checkpoint, model


def test_v5_bundle_and_online_output_have_no_reject_fields(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _ = _checkpoint()
    checkpoint_path = tmp_path / "calibrator.pt"
    torch.save(checkpoint, checkpoint_path)
    model, standardizer, loaded = load_calibrator_bundle(
        checkpoint_path, device=torch.device("cpu")
    )
    assert isinstance(model, MonotoneNoRejectPixelRiskCalibrator)
    assert model.capability_contract()["supports_reject"] is False

    import rc.online_adapter as online_module

    monkeypatch.setattr(
        online_module,
        "load_probability_and_grayscale",
        lambda *args, **kwargs: (
            np.full((2, 2), 0.2, dtype=np.float32),
            np.full((2, 2), 0.5, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        online_module,
        "extract_unlabeled_statistics",
        lambda *args, **kwargs: SimpleNamespace(
            vector=np.asarray([0.25, 0.75], dtype=np.float32),
            feature_names=("f0", "f1"),
            metadata={"test": True},
        ),
    )
    manifest = {
        "target_dataset": "OUT",
        "outer_fold_id": "outer-fold",
        "outer_target": "OUT",
        "protocol_scope": "multi_source_protocol_candidate",
        "target_exclusion_verified": True,
        "weight_sha256": SHA_A,
        "detector_source_domains": ["S1", "S2"],
        "held_out_domains": ["OUT"],
        "items": [{"image_id": "c"}, {"image_id": "q"}],
    }
    result = adapt_context_to_query(
        model=model,
        standardizer=standardizer,
        checkpoint_metadata=loaded,
        context_records=[{"image_id": "c", "prob_path": "unused", "gray_path": "unused"}],
        query_records=[{"image_id": "q", "prob_path": "never-opened", "gray_path": "never-opened"}],
        budgets=BudgetSpec(values=(1e-5, 0.0), active=(True, False)),
        source_reference=_reference(),
        score_manifest=manifest,
        score_manifest_sha256=SHA_B,
        device=torch.device("cpu"),
        target_domain="OUT",
        claim_bearing=True,
    )
    assert result["no_reject"] is True
    assert result["decision_contract"]["reject_supported"] is False
    assert "reject_rule" not in result["decision_contract"]
    assert not {"reject", "reject_probability", "reject_cutoff", "p_min"}.intersection(
        result
    )
    assert _is_no_reject_adapter(result)


def test_v5_bundle_rejects_smuggled_reject_cutoff(tmp_path) -> None:
    checkpoint, _ = _checkpoint()
    checkpoint["reject_probability"] = 0.5
    path = tmp_path / "invalid.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="abstention metadata"):
        load_calibrator_bundle(path, device=torch.device("cpu"))

    protocol = NoRejectDeploymentProtocolContract(context_size=1, query_size=1).to_dict()
    protocol["reject_cutoff"] = 0.5
    with pytest.raises(ValueError, match="abstention fields"):
        NoRejectDeploymentProtocolContract.from_dict(protocol)


def test_v5_bundle_rejects_missing_integrated_training_evidence(tmp_path) -> None:
    checkpoint, _ = _checkpoint()
    checkpoint.pop("risk_loss_contract")
    path = tmp_path / "weak-v5.pt"
    torch.save(checkpoint, path)
    with pytest.raises(KeyError, match="integrated-method evidence"):
        load_calibrator_bundle(path, device=torch.device("cpu"))


def test_exact_replay_uses_strict_threshold_and_one_to_one_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score_path = tmp_path / "q.npz"
    np.savez(score_path, prob=np.asarray([[0.9, 0.8], [0.1, 0.2]]))
    label_path = tmp_path / "q.label.npz"
    np.savez(label_path, mask=np.asarray([[1, 0], [0, 0]], dtype=np.uint8))
    curve_manifest = tmp_path / "curve.manifest.json"
    curve_manifest.write_text(
        json.dumps(
            {
                "score_manifest_file": "scores.json",
                "label_manifest_file": "labels.json",
                "matching_rule": "overlap",
                "centroid_distance": 3.0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "scores.json").write_text("{}", encoding="utf-8")
    (tmp_path / "labels.json").write_text("{}", encoding="utf-8")
    provenance = SimpleNamespace(
        curve_manifest_sha256=hashlib.sha256(
            curve_manifest.read_bytes()
        ).hexdigest(),
        query_score_manifest_sha256=SHA_A,
        label_manifest_sha256=SHA_B,
        label_manifest_content_sha256=SHA_A,
    )
    representative = SimpleNamespace(
        metadata={"curve_manifest_file": curve_manifest.name},
        query_image_ids=("q",),
        provenance=provenance,
        pseudo_target="A",
    )
    group = PixelRiskEpisodeGroup(
        group_id="g",
        pixel_budget_grid=(0.1, 0.01),
        episodes=(representative, representative),
    )

    import evaluation.calibrator_replay as replay_module

    score_item = SimpleNamespace(image_id="q", score_path=score_path)
    label_item = SimpleNamespace(image_id="q", label_path=label_path)
    score_attachment = SimpleNamespace(
        manifest_sha256=SHA_A,
        selected_items=(score_item,),
    )
    monkeypatch.setattr(
        replay_module,
        "verify_score_manifest_artifacts",
        lambda *args, **kwargs: score_attachment,
    )
    monkeypatch.setattr(
        replay_module,
        "verify_label_attachment",
        lambda *args, **kwargs: SimpleNamespace(
            score_manifest=score_attachment,
            manifest_sha256=SHA_B,
            content_sha256=SHA_A,
            selected_items=(label_item,),
        ),
    )
    monkeypatch.setattr(
        replay_module,
        "load_label_mask",
        lambda item: np.load(item.label_path, allow_pickle=False)["mask"],
    )
    evaluator = ExactGroupedPixelRiskReplay([group], artifact_root=tmp_path)
    summary = evaluator.evaluate(
        np.asarray([[0.8, 0.9]], dtype=np.float64),
        pixel_budget_grid=(0.1, 0.01),
    )
    # At 0.8 the target 0.9 is detected but the background pixel equal to 0.8
    # is excluded; at 0.9 the target itself is excluded by strict `>`.
    assert summary.pixel_risk.tolist() == [[0.0, 0.0]]
    assert summary.pd.tolist() == [[1.0, 0.0]]
    assert summary.budget_satisfaction_rate == 1.0
    assert summary.rank_key[0] == 1.0


def test_risk_aligned_trainer_batch_backpropagates_complete_curve() -> None:
    model = MonotoneNoRejectPixelRiskCalibrator(
        context_feature_dim=2,
        pixel_budget_grid=(0.1, 0.01),
        hidden_dims=(4,),
        dropout=0.0,
        min_logit=-4.0,
        max_logit=4.0,
    )
    batch = {
        "features": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        "pixel_budgets": torch.tensor([[0.1, 0.01]], dtype=torch.float64),
        "oracle_logits": torch.tensor([[-1.0, 1.0]], dtype=torch.float64),
        "curve_logits": torch.tensor([[-4.0, 0.0, 4.0]], dtype=torch.float64),
        "curve_pixel_risk": torch.tensor([[0.2, 0.02, 0.0]], dtype=torch.float64),
        "curve_pd": torch.tensor([[1.0, 0.8, 0.0]], dtype=torch.float64),
        "curve_valid_mask": torch.tensor([[True, True, True]]),
        "curve_exact_lower_logit": torch.tensor([-4.0], dtype=torch.float64),
        "curve_global_exact": torch.tensor([True]),
        "curve_gt_objects": torch.tensor([1], dtype=torch.int64),
    }
    args = SimpleNamespace(
        lambda_violation=4.0,
        lambda_utility=1.0,
        lambda_oracle=0.1,
        lambda_smoothness=0.01,
        lambda_coverage=4.0,
        risk_epsilon=1e-12,
        oracle_huber_delta=1.0,
    )
    output, loss = _loss_for_batch(model, batch, args)
    assert output.grid_logits.shape == (1, 2)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_label_evaluation_uses_recomputed_threshold_not_tolerated_json_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evaluation.evaluate_adapter_output as evaluation_module

    score_path = tmp_path / "q.npz"
    np.savez(
        score_path,
        prob=np.asarray([[0.9, 0.0], [0.0, 0.0]], dtype=np.float64),
    )
    label_path = tmp_path / "q.label.npz"
    np.savez(
        label_path,
        mask=np.asarray([[1, 0], [0, 0]], dtype=np.uint8),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    checkpoint_path = tmp_path / "calibrator.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    verified_manifest = SimpleNamespace(payload={}, manifest_sha256=manifest_sha)
    score_item = SimpleNamespace(image_id="q", score_path=score_path)
    label_item = SimpleNamespace(image_id="q", label_path=label_path)
    attachment_score = SimpleNamespace(
        manifest_sha256=manifest_sha, selected_items=(score_item,)
    )
    monkeypatch.setattr(
        evaluation_module,
        "verify_score_manifest_artifacts",
        lambda *args, **kwargs: verified_manifest,
    )
    monkeypatch.setattr(
        evaluation_module,
        "_verify_binding",
        lambda *args, **kwargs: ("OUT", ("c",), ("q",)),
    )
    monkeypatch.setattr(
        evaluation_module,
        "_verify_recomputed_calibrator_decision",
        lambda *args, **kwargs: {"threshold": 0.9},
    )
    monkeypatch.setattr(
        evaluation_module,
        "verify_label_attachment",
        lambda *args, **kwargs: SimpleNamespace(
            score_manifest=attachment_score,
            selected_items=(label_item,),
            manifest_sha256=SHA_A,
            content_sha256=SHA_B,
        ),
    )
    monkeypatch.setattr(
        evaluation_module,
        "load_label_mask",
        lambda item: np.load(item.label_path, allow_pickle=False)["mask"],
    )
    evaluation_contract = {
        "schema_version": "rc-irstd.evaluation-matching.v1",
        "matching_rule": "overlap",
        "centroid_distance": 3.0,
        "source": "checkpoint.deployment_protocol_contract.evaluation_matching",
        "target_override_allowed": False,
        "runtime_override_requested": False,
    }
    adapter = {
        "outer_fold_id": "outer-fold",
        "calibrator_checkpoint_sha256": checkpoint_sha,
        "score_manifest_sha256": manifest_sha,
        "claim_bearing": True,
        "calibrator_model": "monotone_pixel_no_reject",
        "calibrator_capability_contract": {"supports_reject": False},
        "calibrator_budget_contract": {},
        "decision_contract": {
            "reject_supported": False,
            "evaluation_matching": evaluation_contract,
            "budget_model": {},
        },
        "evaluation_contract": evaluation_contract,
        "query_image_ids": ["q"],
        "budgets": {
            "names": ["pixel", "component"],
            "values": [0.1, 0.0],
            "active": [True, False],
        },
        # This differs by only 5e-7 and was historically tolerated. Using it
        # would detect the 0.9 target; deterministic replay at 0.9 must not.
        "threshold": 0.8999995,
        "no_reject": True,
    }
    result = evaluate_adapter_output(
        adapter,
        manifest_path,
        calibrator_checkpoint=checkpoint_path,
        label_manifest=tmp_path / "unused-label-manifest.json",
    )
    assert result["adapter_reported_threshold"] == pytest.approx(0.8999995)
    assert result["threshold"] == pytest.approx(0.9)
    assert result["pd"] == 0.0

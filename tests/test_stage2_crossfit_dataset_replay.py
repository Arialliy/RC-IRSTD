from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch

from data_ext.stage2_label_attachment import (
    canonical_json_sha256 as w05_canonical_json_sha256,
    stage2_ordered_query_identity,
)
from data_ext.stage2_threshold_decision import (
    PRELABEL_METHOD_ORDER,
    verify_stage2_threshold_decision_set,
)
from evaluation import stage2_crossfit_replay as replay_module
from evaluation.stage2_crossfit_replay import Stage2ExactGroupedPixelRiskReplay
from evaluation.stage2_paired_bootstrap import _validate_method_cell
from evaluation.stage2_threshold_family import (
    build_prelabel_decision,
    make_shared_input_bindings,
    publish_prelabel_decision_set,
)

from data_ext.stage2_label_attachment import (
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
    SOURCE_DIAGNOSTIC_VALIDATION,
)
from rc.domain_statistics import FEATURE_NAMES
from rc.stage2_crossfit_dataset import (
    Stage2CrossfitDataset,
    Stage2CurveLogitView,
    collate_stage2_crossfit_batch,
    extract_stage2_curve_brackets,
    fit_stage2_context_standardizer,
    group_stage2_pixel_risk_episodes,
)
from rc.stage2_crossfit_schema import (
    COLLECTION_OUTER,
    COLLECTION_TRAIN,
    COLLECTION_VALIDATION,
    STAGE2_OOF_FIT,
    Stage2CrossfitContractError,
    Stage2CrossfitEpisode,
    VerifiedEpisodeArtifacts,
    bootstrap_query_identity_sha256,
    canonical_json_sha256,
    full_identity_sha256,
    make_verified_collection,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _minimal_episode(
    index: int,
    role: str,
    source_domain: str,
    *,
    feature_value: float = 0.0,
) -> Stage2CrossfitEpisode:
    episode_role = {
        COLLECTION_TRAIN: STAGE2_OOF_FIT,
        COLLECTION_VALIDATION: SOURCE_DIAGNOSTIC_VALIDATION,
        COLLECTION_OUTER: OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
    }[role]
    values = np.full(93, feature_value, dtype=np.float32)
    payload = {
        "episode_id": _sha(f"episode:{role}:{index}"),
        "episode_index": index,
        "collection_role": role,
        "episode_role": episode_role,
        "outer_fold_id": "outer_leave_nuaa_sirst",
        "outer_target": "NUAA-SIRST",
        "source_domain": source_domain,
        "base_seed": 42,
        "derived_seed": 42001,
        "window_binding": {
            "path": f"synthetic/window-{index}.json",
            "window_id": f"window-{index}",
        },
        "context_statistics": {
            "values": values.tolist(),
            "vector_sha256": _sha(f"vector:{role}:{index}"),
        },
        "context_full_identity_sha256": _sha(f"context:{role}:{index}"),
        "source_ordered_query_identity_sha256": _sha(f"query:{role}:{index}"),
        "detector_identity": {
            "detector_role": "detector_oof" if role == COLLECTION_TRAIN else "detector_full_fit",
            "oof_fold_index": 0 if role == COLLECTION_TRAIN else None,
        },
        "context_records": [],
        "query_records": [],
    }
    return Stage2CrossfitEpisode(MappingProxyType(payload))


def _collection(role: str, *, curves: tuple[dict[str, object], ...] = ()):
    count = {
        COLLECTION_TRAIN: 26,
        COLLECTION_VALIDATION: 6,
        COLLECTION_OUTER: 1,
    }[role]
    domains = (
        ["NUAA-SIRST"]
        if role == COLLECTION_OUTER
        else ["NUDT-SIRST", "IRSTD-1K"]
    )
    episodes = [
        _minimal_episode(i, role, domains[i % len(domains)]) for i in range(count)
    ]
    artifacts = [
        VerifiedEpisodeArtifacts(None, None, curves if i == 0 else ())
        for i in range(count)
    ]
    manifest = {
        "collection_role": role,
        "outer_fold_id": "outer_leave_nuaa_sirst",
        "outer_target": "NUAA-SIRST",
        "base_seed": 42,
        "records": [
            {"record_sha256": _sha(f"record:{role}:{i}")} for i in range(count)
        ],
    }
    return make_verified_collection(
        path=Path(f"/{role}.jsonl"),
        manifest_path=Path(f"/{role}.collection.json"),
        commit_path=Path(f"/{role}.commit.json"),
        episodes=episodes,
        artifacts=artifacts,
        collection_sha256=_sha(f"collection:{role}"),
        manifest_sha256=_sha(f"manifest:{role}"),
        commit_sha256=_sha(f"commit:{role}"),
        manifest=manifest,
    )


def test_standardizer_exact_floor_and_training_scope() -> None:
    train = _collection(COLLECTION_TRAIN)
    standardizer = fit_stage2_context_standardizer(train)
    assert standardizer.mean.dtype == np.float64
    assert np.array_equal(standardizer.scale, np.full(93, 1e-8))
    assert standardizer.fit_manifest["scale_floor"] == 1e-8
    assert standardizer.fit_manifest["below_floor_replacement"] == 1e-8
    assert standardizer.fit_manifest["validation_or_outer_access_count"] == 0
    with pytest.raises(Stage2CrossfitContractError, match="training collection"):
        fit_stage2_context_standardizer(_collection(COLLECTION_VALIDATION))


def _curve_rows() -> tuple[dict[str, object], ...]:
    thresholds = (0.0, 0.5, 0.7, 0.9, 1.0)
    tp = (0, 1, 1, 1, 0)
    fp = (4, 2, 1, 1, 0)
    return tuple(
        {
            "threshold": threshold,
            "pd": float(true_positive / 3),
            "fa_pixel": float(false_positive / 10_000_000),
            "tp_objects": true_positive,
            "gt_objects": 3,
            "fp_pixels": false_positive,
            "total_pixels": 10_000_000,
        }
        for threshold, true_positive, false_positive in zip(thresholds, tp, fp)
    )


def test_exact_rank_integer_tp_and_dataset_zero_copy() -> None:
    outer = _collection(COLLECTION_OUTER, curves=_curve_rows())
    groups = group_stage2_pixel_risk_episodes(outer)
    group = groups[0]
    assert np.array_equal(group.oracle_thresholds, np.asarray([0.9, 0.9, 0.9]))
    assert group.curve_tp_objects.dtype == np.int64
    assert int(group.curve_tp_objects[1]) == 1
    standardizer = fit_stage2_context_standardizer(_collection(COLLECTION_TRAIN))
    dataset = Stage2CrossfitDataset(outer, standardizer)
    item = dataset[0]
    assert item["curve_tp_objects"] is dataset.groups[0].curve_tp_objects
    assert np.shares_memory(item["curve_thresholds"], dataset.groups[0].curve_thresholds)
    assert isinstance(item["curve_logits"], Stage2CurveLogitView)
    assert item["curve_logits"].nbytes == 0


def test_oracle_uses_integer_counts_at_1e5_budget_boundary() -> None:
    total_pixels = 99_999
    rows = tuple(
        {
            "threshold": threshold,
            "pd": float(true_positive / 2),
            # Deliberately round the unsafe candidate onto the budget boundary:
            # feasibility must be derived from the integer sufficient statistics.
            "fa_pixel": reported_risk,
            "tp_objects": true_positive,
            "gt_objects": 2,
            "fp_pixels": false_positive,
            "total_pixels": total_pixels,
        }
        for threshold, true_positive, false_positive, reported_risk in (
            (0.0, 0, 100, float(100 / total_pixels)),
            (0.5, 2, 1, 1e-5),
            (0.9, 1, 0, 0.0),
            (1.0, 0, 0, 0.0),
        )
    )
    assert Fraction(1, total_pixels) > Fraction.from_float(1e-5)
    group = group_stage2_pixel_risk_episodes(
        _collection(COLLECTION_OUTER, curves=rows)
    )[0]
    assert np.array_equal(group.oracle_thresholds, np.asarray([0.5, 0.9, 0.9]))
    assert np.array_equal(group.oracle_fp_pixels, np.asarray([1, 0, 0]))


def test_million_event_collate_is_ragged_zero_copy_and_brackets_are_small() -> None:
    size = 1_000_001
    thresholds = np.linspace(0.0, 1.0, size, dtype=np.float64)
    risk = np.linspace(1e-3, 0.0, size, dtype=np.float64)
    pd = np.linspace(1.0, 0.0, size, dtype=np.float64)
    fp = np.arange(size, 0, -1, dtype=np.int64)
    tp = np.arange(size, dtype=np.int64) % 7
    item = {
        "features": torch.zeros(93),
        "pixel_budgets": torch.tensor([1e-4, 1e-5, 1e-6]),
        "oracle_thresholds": torch.zeros(3),
        "oracle_logits": torch.zeros(3),
        "oracle_pd": torch.zeros(3),
        "oracle_pixel_risk": torch.zeros(3),
        "curve_total_pixels": torch.tensor(10_000_000, dtype=torch.int64),
        "curve_gt_objects": torch.tensor(7, dtype=torch.int64),
        "curve_thresholds": thresholds,
        "curve_logits": Stage2CurveLogitView(thresholds),
        "curve_pixel_risk": risk,
        "curve_pd": pd,
        "curve_fp_pixels": fp,
        "curve_tp_objects": tp,
        "episode_id": "episode",
        "window_id": "window",
        "source_domain": "NUAA-SIRST",
        "context_identity_sha256": _sha("context"),
        "source_query_identity_sha256": _sha("query"),
    }
    batch = collate_stage2_crossfit_batch([item])
    assert "curve_valid" not in batch
    assert batch["curve_thresholds"] == (thresholds,)
    assert batch["curve_tp_objects"] == (tp,)
    assert batch["curve_thresholds"][0] is thresholds
    assert batch["curve_logits"][0].nbytes == 0
    bracket = extract_stage2_curve_brackets(batch, torch.zeros((1, 3)))
    assert all(tuple(value.shape) == (1, 3) for value in bracket.values())
    assert sum(value.numel() for value in bracket.values()) <= 42


def _identity_row(partition: str, ordinal: int) -> dict[str, object]:
    token = f"{partition}-{ordinal}"
    return {
        "ordinal": ordinal,
        "partition": partition,
        "score_record_index": ordinal,
        "canonical_id": f"canonical-{token}",
        "image_id": f"image-{token}",
        "source_domain": "NUAA-SIRST",
        "original_image_path": f"images/{token}.png",
        "original_image_sha256": _sha(f"image:{token}"),
        "exclusion_group_id": f"exclusion-{token}",
        "near_duplicate_cluster_id_or_unique_sentinel": f"unique-{token}",
        "source_role_record_index": ordinal + (0 if partition == "context" else 14),
        "source_role": "outer_target_diagnostic_development",
        "outer_fold_id": "outer_leave_nuaa_sirst",
        "oof_fold_index": None,
        "score_file": f"scores/{token}.npz",
        "score_file_sha256": _sha(f"score:{token}"),
        "original_hw": [4, 5],
        "input_hw": [4, 5],
        "resized_hw": [4, 5],
        "padding_ltrb": [0, 0, 0, 0],
        "resize_mode": "native",
    }


def test_public_verified_decision_replay_recomputes_four_identity_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context_rows = [_identity_row("context", i) for i in range(14)]
    query_rows = [_identity_row("query", i) for i in range(28)]
    projection_keys = (
        "canonical_id", "image_id", "original_image_sha256",
        "exclusion_group_id", "near_duplicate_cluster_id_or_unique_sentinel",
        "source_role_record_index",
    )
    selected = {
        "window_index": 0,
        "window_id": "outer-window-0",
        "context_records": [
            {key: row[key] for key in projection_keys} for row in context_rows
        ],
        "query_records": [
            {key: row[key] for key in projection_keys} for row in query_rows
        ],
    }
    source_window_sha = w05_canonical_json_sha256(selected)
    source_query_sha = w05_canonical_json_sha256(
        stage2_ordered_query_identity(selected["query_records"])
    )
    checkpoint_sha = _sha("detector-checkpoint")
    context_sha = _sha("context-package")
    context_commit_sha = _sha("context-commit")
    score_sha = _sha("score-manifest")
    score_records_sha = _sha("score-records")
    episode_payload = {
        "episode_id": _sha("outer-episode"),
        "episode_index": 0,
        "collection_role": COLLECTION_OUTER,
        "episode_role": OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
        "outer_fold_id": "outer_leave_nuaa_sirst",
        "outer_target": "NUAA-SIRST",
        "source_domain": "NUAA-SIRST",
        "base_seed": 42,
        "derived_seed": 42001,
        "window_binding": {
            "path": "windows/outer.json",
            "window_id": selected["window_id"],
            "window_identity_sha256": source_window_sha,
        },
        "context_package_binding": {
            "path": "context/context.json",
            "sha256": context_sha,
            "commit_path": "context/context.commit.json",
            "commit_sha256": context_commit_sha,
        },
        "score_manifest_binding": {
            "sha256": score_sha,
            "records_content_sha256": score_records_sha,
        },
        "detector_identity": {"checkpoint_sha256": checkpoint_sha},
        "context_records": context_rows,
        "query_records": query_rows,
        "context_full_identity_sha256": full_identity_sha256(context_rows),
        "source_ordered_query_identity_sha256": source_query_sha,
        "context_statistics": {"values": [0.0] * 93},
    }
    episode = Stage2CrossfitEpisode(MappingProxyType(episode_payload))
    attachment = SimpleNamespace(ordered_query_identity_sha256=source_query_sha)
    context = SimpleNamespace(window=SimpleNamespace(window=selected))
    artifact = VerifiedEpisodeArtifacts(context, attachment, ())
    collection = make_verified_collection(
        path=tmp_path / "outer.jsonl",
        manifest_path=tmp_path / "outer.collection.json",
        commit_path=tmp_path / "outer.commit.json",
        episodes=[episode],
        artifacts=[artifact],
        collection_sha256=_sha("outer-collection"),
        manifest_sha256=_sha("outer-manifest"),
        commit_sha256=_sha("outer-commit"),
        manifest={
            "collection_role": COLLECTION_OUTER,
            "outer_fold_id": "outer_leave_nuaa_sirst",
            "outer_target": "NUAA-SIRST",
            "base_seed": 42,
        },
    )
    shared = make_shared_input_bindings(
        context_package_path="context/context.json",
        context_package_sha256=context_sha,
        context_package_commit_path="context/context.commit.json",
        context_package_commit_sha256=context_commit_sha,
        window_id=selected["window_id"],
        window_identity_sha256=source_window_sha,
        ordered_query_identity_sha256=source_query_sha,
        score_manifest_sha256=score_sha,
        score_records_content_sha256=score_records_sha,
        detector_checkpoint_sha256=checkpoint_sha,
    )
    decisions = [
        build_prelabel_decision(
            method_id=method,
            thresholds=(
                [0.5, 0.5, 0.5]
                if method == "T0"
                else [0.4, 0.5, 0.6]
            ),
            shared_bindings=shared,
            outer_fold_id="outer_leave_nuaa_sirst",
            outer_target_domain="NUAA-SIRST",
            base_seed=42,
            derived_seed=42001,
            method_contract={"method": method},
            method_binding=(
                {
                    "path": f"synthetic/{method}.frozen-input.json",
                    "sha256": _sha(f"method-binding-{method}"),
                }
                if method in {"T1", "T2", "T3", "T5", "T6", "T7", "T8"}
                else None
            ),
        )
        for method in PRELABEL_METHOD_ORDER
    ]
    set_path, set_sha = publish_prelabel_decision_set(
        decisions, tmp_path / "decision-bundle", repository_root=tmp_path
    )
    decision_set = verify_stage2_threshold_decision_set(
        set_path, set_sha, repository_root=tmp_path
    )
    count_sets = tuple(
        tuple(
            {
                "image_id": row["image_id"],
                "original_image_sha256": row["original_image_sha256"],
                "false_positive_pixels": budget_index,
                "total_pixels": 20,
                "background_pixels": 19,
                "matched_targets": 1,
                "ground_truth_targets": 1,
            }
            for row in query_rows
        )
        for budget_index in range(3)
    )
    monkeypatch.setattr(replay_module, "_replay_thresholds", lambda *args: count_sets)
    decision = decision_set.decision_by_method["T8"]
    result = Stage2ExactGroupedPixelRiskReplay(collection).evaluate(decision)
    bridge = result.identity_bridge
    assert bridge["source_window_identity_sha256"] == source_window_sha
    assert bridge["source_ordered_query_identity_sha256"] == source_query_sha
    assert bridge["bootstrap_ordered_query_identity_sha256"] == (
        bootstrap_query_identity_sha256(query_rows)
    )
    assert bridge["bootstrap_window_identity_sha256"] == canonical_json_sha256(
        {
            "window_id": selected["window_id"],
            "context_identity_sha256": full_identity_sha256(context_rows),
            "ordered_query_identity_sha256": bootstrap_query_identity_sha256(query_rows),
        }
    )
    validated = _validate_method_cell(
        result.primary_sufficient_counts,
        method_id="T8",
        expected_windows=1,
        name="T8",
    )
    assert len(validated["windows"][0]["query_counts"]) == 28
    decision.payload["thresholds"][0] = 0.123
    with pytest.raises(Stage2CrossfitContractError, match="mutated"):
        Stage2ExactGroupedPixelRiskReplay(collection).evaluate(decision)

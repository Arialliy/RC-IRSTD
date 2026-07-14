from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rc.meta_dataset import (
    QueryRiskCurveSupervision,
    RCGroupedPixelRiskMetaDataset,
    collate_grouped_pixel_risk_batch,
    group_pixel_risk_episodes,
    load_verified_query_risk_curve,
)
from rc.schema import (
    BudgetSpec,
    EpisodeProvenance,
    FoldContract,
    RCEpisode,
    SourceContract,
    SourceReference,
    StatisticsConfig,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_contract(role: str = "official_train") -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "role": role,
        "selected_split_file": "",
        "selected_split_sha256": "",
        "selected_num_images": 0,
        "selected_ids_sha256": "",
        "official_train_split_file": "train.txt",
        "official_train_split_sha256": SHA_A,
        "official_train_num_images": 20,
        "official_train_ids_sha256": SHA_B,
        "official_train_split_image_artifact_sha256": SHA_C,
        "official_test_split_file": "test.txt",
        "official_test_split_sha256": SHA_D,
        "official_test_num_images": 10,
        "official_test_ids_sha256": SHA_E,
        "official_test_split_image_artifact_sha256": SHA_A,
        "ordered_sample_ids_algorithm": "test-order-v1",
        "split_image_artifact_algorithm": "test-artifact-v1",
        "train_test_id_overlap_count": 0,
        "train_test_id_overlap_ids": [],
        "train_test_image_content_overlap_count": 0,
        "train_test_image_content_overlap_sha256_leaves": [],
        "disjointness_verified": True,
    }
    prefix = role
    result["selected_split_file"] = result[f"{prefix}_split_file"]
    result["selected_split_sha256"] = result[f"{prefix}_split_sha256"]
    result["selected_num_images"] = result[f"{prefix}_num_images"]
    result["selected_ids_sha256"] = result[f"{prefix}_ids_sha256"]
    return result


def _reference() -> SourceReference:
    return SourceReference(
        domains=("S1", "S2"),
        sha256=SHA_E,
        centers=((0.0, 0.0), (1.0, 1.0)),
        scale=(1.0, 1.0),
        contract=SourceContract(
            detector_checkpoint_sha=SHA_A,
            detector_source_domains=("S1", "S2"),
            outer_fold_id="outer-fold",
            outer_target="OUT",
            held_out_domains=("A", "B", "OUT"),
            protocol_scope="multi_source_protocol_candidate",
        ),
    )


def _episode(
    *,
    episode_id: str,
    budget: float,
    threshold: float,
    pixel_risk: float,
    pseudo_target: str = "A",
    curve_sha256: str = SHA_B,
    curve_manifest_sha256: str = SHA_C,
    query_score_manifest_sha256: str = SHA_D,
    label_manifest_sha256: str = SHA_E,
    label_manifest_content_sha256: str = SHA_A,
    split_role: str = "official_train",
    metadata: dict[str, object] | None = None,
) -> RCEpisode:
    reference = _reference()
    provenance = EpisodeProvenance(
        status="verified",
        curve_file_sha256=curve_sha256,
        curve_manifest_sha256=curve_manifest_sha256,
        context_score_manifest_sha256=query_score_manifest_sha256,
        query_score_manifest_sha256=query_score_manifest_sha256,
        query_score_target_dataset=pseudo_target,
        label_manifest_sha256=label_manifest_sha256,
        label_manifest_content_sha256=label_manifest_content_sha256,
        split_contract=_split_contract(split_role),
    )
    return RCEpisode.create(
        episode_id=episode_id,
        pseudo_target=pseudo_target,
        context_image_ids=[f"{pseudo_target}-c0"],
        query_image_ids=[f"{pseudo_target}-q0"],
        statistics=(0.25, 0.75),
        feature_names=("f0", "f1"),
        statistics_config=StatisticsConfig(peak_kernel_size=3, peak_min_score=0.1),
        source_reference=reference,
        fold=FoldContract(
            outer_fold_id="outer-fold",
            outer_target="OUT",
            detector_source_domains=("S1", "S2"),
            detector_checkpoint_sha=SHA_A,
            held_out_domains=("A", "B", "OUT"),
            protocol_scope="multi_source_protocol_candidate",
        ),
        provenance=provenance,
        budgets=BudgetSpec(values=(budget, 0.0), active=(True, False)),
        oracle_threshold=threshold,
        oracle_pd=0.8,
        oracle_pixel_risk=pixel_risk,
        oracle_component_risk=0.0,
        p_min=0.5,
        metadata=metadata,
    )


def _two_budget_episodes(**kwargs: object) -> list[RCEpisode]:
    return [
        _episode(
            episode_id="strict",
            budget=1e-5,
            threshold=0.9,
            pixel_risk=5e-6,
            **kwargs,
        ),
        _episode(
            episode_id="loose",
            budget=1e-4,
            threshold=0.8,
            pixel_risk=5e-5,
            **kwargs,
        ),
    ]


def test_grouped_dataset_emits_one_complete_no_reject_curve() -> None:
    dataset = RCGroupedPixelRiskMetaDataset(
        _two_budget_episodes(), pixel_budget_grid=(1e-4, 1e-5)
    )
    assert len(dataset) == 1
    item = dataset[0]
    torch.testing.assert_close(
        item["pixel_budgets"], torch.tensor([1e-4, 1e-5], dtype=torch.float64)
    )
    torch.testing.assert_close(
        item["oracle_thresholds"], torch.tensor([0.8, 0.9], dtype=torch.float64)
    )
    assert item["oracle_logits"].shape == (2,)
    assert "reject" not in item
    assert item["query_curve_available"] is False
    assert item["episode_ids"] == ("loose", "strict")
    assert item["query_score_manifest_sha256"] == SHA_D


def test_grouping_fails_closed_on_missing_duplicate_test_or_hash_drift() -> None:
    episodes = _two_budget_episodes()
    with pytest.raises(ValueError, match="complete frozen"):
        group_pixel_risk_episodes(
            episodes[:1], pixel_budget_grid=(1e-4, 1e-5)
        )
    duplicate = _episode(
        episode_id="duplicate",
        budget=1e-4,
        threshold=0.8,
        pixel_risk=5e-5,
    )
    with pytest.raises(ValueError, match="duplicate budget"):
        group_pixel_risk_episodes(
            episodes + [duplicate], pixel_budget_grid=(1e-4, 1e-5)
        )
    test_rows = _two_budget_episodes(split_role="official_test")
    with pytest.raises(ValueError, match="official_train"):
        group_pixel_risk_episodes(test_rows, pixel_budget_grid=(1e-4, 1e-5))
    drift = _episode(
        episode_id="strict-drift",
        budget=1e-5,
        threshold=0.9,
        pixel_risk=5e-6,
        curve_sha256=SHA_C,
    )
    with pytest.raises(ValueError, match="identical provenance"):
        group_pixel_risk_episodes(
            [episodes[1], drift], pixel_budget_grid=(1e-4, 1e-5)
        )


def test_verified_curve_mode_requires_explicit_artifact_root() -> None:
    with pytest.raises(ValueError, match="artifact_root"):
        RCGroupedPixelRiskMetaDataset(
            _two_budget_episodes(),
            pixel_budget_grid=(1e-4, 1e-5),
            query_curve_mode="verified_event_exact",
        )


def test_curve_contract_and_padding_keep_only_valid_strict_events() -> None:
    curve = QueryRiskCurveSupervision(
        thresholds=(0.8, 0.9, 1.0),
        logits=(1.0, 2.0, 3.0),
        pixel_risk=(0.03, 0.01, 0.0),
        pd=(1.0, 0.8, 0.0),
        fp_pixels=(3, 1, 0),
        total_pixels=100,
        gt_objects=1,
        matching_rule="overlap",
        centroid_distance=3.0,
        exact_lower_bound=0.8,
        global_exact=False,
        supervision_mode="event_exact_suffix",
        curve_file_sha256=SHA_A,
        curve_manifest_sha256=SHA_B,
        query_score_manifest_sha256=SHA_C,
        label_manifest_sha256=SHA_D,
        label_manifest_content_sha256=SHA_E,
    )
    assert curve.thresholds[0] == curve.exact_lower_bound
    with pytest.raises(ValueError, match="strictly ascending"):
        QueryRiskCurveSupervision(
            **{**curve.__dict__, "logits": (1.0, 1.0, 3.0)}
        )

    base = RCGroupedPixelRiskMetaDataset(
        _two_budget_episodes(), pixel_budget_grid=(1e-4, 1e-5)
    )[0]
    first = dict(base)
    second = dict(base)
    for item, length in ((first, 3), (second, 2)):
        item["query_curve_available"] = True
        item["curve_thresholds"] = torch.tensor(
            [0.8, 0.9, 1.0][:length], dtype=torch.float64
        )
        item["curve_logits"] = torch.tensor(
            [1.0, 2.0, 3.0][:length], dtype=torch.float64
        )
        item["curve_pixel_risk"] = torch.tensor(
            [0.03, 0.01, 0.0][:length], dtype=torch.float64
        )
        item["curve_pd"] = torch.tensor(
            [1.0, 0.8, 0.0][:length], dtype=torch.float64
        )
        item["curve_fp_pixels"] = torch.tensor(
            [3, 1, 0][:length], dtype=torch.int64
        )
        item["curve_total_pixels"] = torch.tensor(100, dtype=torch.int64)
        item["curve_gt_objects"] = torch.tensor(1, dtype=torch.int64)
        item["curve_matching_rule"] = "overlap"
        item["curve_centroid_distance"] = torch.tensor(3.0, dtype=torch.float64)
        item["curve_exact_lower_bound"] = torch.tensor(0.8, dtype=torch.float64)
        item["curve_exact_lower_logit"] = torch.tensor(1.0, dtype=torch.float64)
        item["curve_global_exact"] = False
        item["curve_supervision_mode"] = "event_exact_suffix"
    batch = collate_grouped_pixel_risk_batch([first, second])
    assert batch["curve_valid_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert batch["curve_lengths"].tolist() == [3, 2]
    assert batch["curve_gt_objects"].tolist() == [1, 1]
    assert torch.all(
        batch["curve_logits"][:, 1:][
            batch["curve_valid_mask"][:, 1:]
        ]
        > batch["curve_logits"][:, :-1][
            batch["curve_valid_mask"][:, 1:]
        ]
    )


def test_loader_reverifies_hash_bound_event_exact_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curve_path = tmp_path / "query.csv"
    curve_path.write_text(
        "threshold,pd,fa_pixel,fa_component_mp,tp_objects,gt_objects,"
        "pred_components,fp_components,fp_pixels,total_pixels,num_images\n"
        "0.8,1.0,0.03,0.0,1,1,1,0,3,100,1\n"
        "0.9,0.8,0.01,0.0,1,1,1,0,1,100,1\n"
        "1.0,0.0,0.0,0.0,0,1,0,0,0,100,1\n",
        encoding="utf-8",
    )
    score_manifest_path = tmp_path / "scores.json"
    label_manifest_path = tmp_path / "labels.json"
    score_manifest_path.write_text("{}\n", encoding="utf-8")
    label_manifest_path.write_text("{}\n", encoding="utf-8")
    query_score_sha = _sha256(score_manifest_path)
    label_sha = _sha256(label_manifest_path)
    label_content_sha = SHA_E
    manifest = {
        "curve_file": curve_path.name,
        "curve_sha256": _sha256(curve_path),
        "evaluation_scope": "score_bound_label_attachment_verified",
        "oracle_only": True,
        "selection_uses_ground_truth_labels": True,
        "deployable": False,
        "target_dataset": "A",
        "image_ids": ["A-q0"],
        "score_manifest_file": score_manifest_path.name,
        "score_manifest_sha256": query_score_sha,
        "label_manifest_file": label_manifest_path.name,
        "label_manifest_sha256": label_sha,
        "label_manifest_content_sha256": label_content_sha,
        "thresholds": [0.8, 0.9, 1.0],
        "num_images": 1,
        "gt_objects": 1,
        "total_pixels": 100,
        "matching_rule": "overlap",
        "centroid_distance": 3.0,
        "threshold_mode_requested": "adaptive",
        "event_candidate_count": 2,
        "event_threshold_count": 2,
        "event_thresholds_added": 2,
        "event_threshold_cap": None,
        "event_thresholds_capped": False,
        "event_candidate_score_lower_bound": 0.8,
        "event_coverage_score_lower_bound": 0.8,
        "event_coverage_fraction_lower_bound": 1.0,
        "global_exact": False,
    }
    manifest_path = tmp_path / "query.csv.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    metadata = {
        "curve_file": curve_path.name,
        "curve_sha256": _sha256(curve_path),
        "curve_manifest_file": manifest_path.name,
        "curve_manifest_sha256": _sha256(manifest_path),
        "curve_provenance_status": "verified",
        "causal_window_verified": True,
        "query_label_manifest_sha256": label_sha,
    }
    episodes = [
        _episode(
            episode_id="loose",
            budget=0.04,
            threshold=0.8,
            pixel_risk=0.03,
            curve_sha256=_sha256(curve_path),
            curve_manifest_sha256=_sha256(manifest_path),
            query_score_manifest_sha256=query_score_sha,
            label_manifest_sha256=label_sha,
            label_manifest_content_sha256=label_content_sha,
            metadata=metadata,
        ),
        _episode(
            episode_id="strict",
            budget=0.02,
            threshold=0.9,
            pixel_risk=0.01,
            curve_sha256=_sha256(curve_path),
            curve_manifest_sha256=_sha256(manifest_path),
            query_score_manifest_sha256=query_score_sha,
            label_manifest_sha256=label_sha,
            label_manifest_content_sha256=label_content_sha,
            metadata=metadata,
        ),
    ]
    group = group_pixel_risk_episodes(
        episodes, pixel_budget_grid=(0.04, 0.02)
    )[0]

    import data_ext.label_manifest_artifacts as label_module
    import data_ext.score_manifest_artifacts as score_module

    monkeypatch.setattr(
        score_module,
        "verify_score_manifest_artifacts",
        lambda *args, **kwargs: SimpleNamespace(
            manifest_sha256=query_score_sha,
            selected_items=(SimpleNamespace(image_id="A-q0"),),
        ),
    )
    monkeypatch.setattr(
        label_module,
        "verify_label_attachment",
        lambda *args, **kwargs: SimpleNamespace(
            manifest_sha256=label_sha,
            content_sha256=label_content_sha,
        ),
    )
    curve = load_verified_query_risk_curve(group, artifact_root=tmp_path)
    assert curve.supervision_mode == "event_exact_suffix"
    assert curve.exact_lower_bound == 0.8
    assert curve.thresholds == (0.8, 0.9, 1.0)
    assert curve.fp_pixels == (3, 1, 0)
    assert all(left < right for left, right in zip(curve.logits, curve.logits[1:]))

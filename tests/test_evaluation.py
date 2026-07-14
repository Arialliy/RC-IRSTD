from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from data_ext.split_utils import resolve_image_and_mask, resolve_split_file
from evaluation.budget_metrics import compute_budget_metrics
from evaluation.component_matching import match_components
from evaluation.evaluate_adapter_output import (
    evaluate_adapter_output,
    main as evaluate_adapter_main,
    summarise_adapter_evaluations,
)
from evaluation.operating_point import select_operating_point
from evaluation.threshold_sweep import (
    CURVE_SCHEMA_VERSION,
    ScoreMapRecord,
    THRESHOLD_GRID_VERSION,
    main as threshold_sweep_main,
    normalise_thresholds,
    read_curve_csv,
    sweep_thresholds,
    threshold_grid_metadata,
)

try:
    import torch
    from torch.utils.data import DataLoader

    from data_ext.dataset_meta import (
        build_spatial_transform,
        restore_tensor_to_original,
        sample_meta_from_batch,
    )
    from data_ext.eval_dataset import IRSTDEvalDataset
    from evaluation.export_score_maps import checkpoint_provenance, export_score_maps

    HAS_TORCH_STACK = True
except ModuleNotFoundError:
    HAS_TORCH_STACK = False


requires_torch = pytest.mark.skipif(
    not HAS_TORCH_STACK,
    reason="PyTorch/torchvision evaluation stack is not installed",
)


def _make_nuaa_style_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "img_idx").mkdir()
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[1:3, 2:6] = 200
    mask = np.zeros((4, 8), dtype=np.uint8)
    mask[2, 4] = 255
    Image.fromarray(image).save(root / "images" / "Misc_1.png")
    Image.fromarray(mask).save(root / "masks" / "Misc_1_pixels0.png")
    (root / "img_idx" / "test_NUAA-SIRST.txt").write_text(
        "Misc_1.png\n",
        encoding="utf-8",
    )
    return root


def test_split_and_nuaa_mask_resolution(tmp_path: Path) -> None:
    root = _make_nuaa_style_dataset(tmp_path / "NUAA-SIRST")
    split = resolve_split_file(root, "test")
    assert split.name == "test_NUAA-SIRST.txt"
    image_path, mask_path = resolve_image_and_mask(root, "Misc_1.png")
    assert image_path.name == "Misc_1.png"
    assert mask_path.name == "Misc_1_pixels0.png"
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        assert image.size == mask.size == (8, 4)


def test_ambiguous_split_requires_explicit_path(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "img_idx").mkdir(parents=True)
    (root / "img_idx" / "test_a.txt").write_text("a\n", encoding="utf-8")
    (root / "img_idx" / "test_b.txt").write_text("b\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Multiple split files"):
        resolve_split_file(root, "test")


@pytest.mark.parametrize("resize_mode", ["resize", "letterbox"])
@requires_torch
def test_eval_dataset_default_collate_is_stable(
    tmp_path: Path,
    resize_mode: str,
) -> None:
    root = _make_nuaa_style_dataset(tmp_path / "NUAA-SIRST")
    dataset = IRSTDEvalDataset(root, base_size=8, resize_mode=resize_mode)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    assert tuple(batch["image"].shape) == (1, 3, 8, 8)
    assert tuple(batch["mask"].shape) == (1, 1, 8, 8)
    meta = sample_meta_from_batch(batch["meta"], 0)
    assert meta.image_id == "Misc_1"
    assert meta.dataset_name == "NUAA-SIRST"
    assert meta.transform.original_hw == (4, 8)
    assert meta.transform.input_hw == (8, 8)
    assert tuple(dataset.load_original_mask(0).shape) == (1, 4, 8)
    if resize_mode == "letterbox":
        assert meta.transform.resized_hw == (4, 8)
        assert meta.transform.padding_ltrb == (0, 2, 0, 2)


@requires_torch
def test_nuaa_mismatched_mask_is_aligned_to_image_canvas(tmp_path: Path) -> None:
    root = _make_nuaa_style_dataset(tmp_path / "NUAA-SIRST")
    oversized_mask = np.zeros((8, 16), dtype=np.uint8)
    oversized_mask[4, 8] = 255
    Image.fromarray(oversized_mask).save(root / "masks" / "Misc_1_pixels0.png")
    dataset = IRSTDEvalDataset(root, base_size=8, resize_mode="letterbox")
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    meta = sample_meta_from_batch(batch["meta"], 0)
    assert meta.transform.original_hw == (4, 8)
    assert meta.mask_original_hw == (8, 16)
    assert tuple(dataset.load_original_mask(0).shape) == (1, 4, 8)


@requires_torch
def test_checkpoint_provenance_is_derived_from_checkpoint_metadata() -> None:
    provenance = checkpoint_provenance(
        {
            "state_dict": {"unused": torch.zeros(1)},
            "detector_source_domains": ["A", "B"],
            "held_out_domains": ["C"],
            "outer_fold_id": "outer-c",
            "outer_target": "C",
            "checkpoint_selection": "fixed_last_no_test_or_target_validation",
            "protocol_scope": "multi_source_protocol_candidate",
            "seed": 7,
        }
    )
    assert provenance["provenance_level"] == "checkpoint_verified"
    assert provenance["detector_source_domains"] == ["A", "B"]
    assert provenance["outer_target"] == "C"
    assert provenance["training_seed"] == 7


@requires_torch
def test_exporter_rejects_spatial_size_not_divisible_by_sixteen(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="multiples of 16"):
        export_score_maps(
            dataset_dir=tmp_path / "unused-dataset",
            weight_path=tmp_path / "unused.pt",
            output_dir=tmp_path / "unused-output",
            base_size=(32, 30),
            device="cpu",
        )


@requires_torch
def test_letterbox_prediction_is_cropped_and_restored() -> None:
    transform = build_spatial_transform((2, 4), (4, 4), "letterbox")
    input_map = torch.full((4, 4), 9.0)
    input_map[1:3, :] = 1.0
    restored = restore_tensor_to_original(input_map, transform, mode="bilinear")
    assert tuple(restored.shape) == (2, 4)
    assert torch.allclose(restored, torch.ones_like(restored))


def test_overlap_matching_is_8_connected_and_one_to_one() -> None:
    target = np.zeros((5, 6), dtype=np.uint8)
    target[1, 1] = 1
    target[1, 4] = 1
    prediction = np.zeros_like(target)
    prediction[1, 1:5] = 1  # One component overlaps two separate GT objects.

    result = match_components(prediction, target, rule="overlap")
    assert result.num_gt == 2
    assert result.num_pred_components == 1
    assert result.num_tp_objects == 1
    assert result.num_fp_components == 0
    assert result.num_fp_pixels == 2

    diagonal = np.zeros((3, 3), dtype=np.uint8)
    diagonal[0, 0] = 1
    diagonal[1, 1] = 1
    connected = match_components(diagonal, diagonal, rule="overlap")
    assert connected.num_gt == 1
    assert connected.num_pred_components == 1


def test_centroid_matching_is_one_to_one() -> None:
    target = np.zeros((7, 7), dtype=np.uint8)
    target[1, 1] = 1
    target[5, 5] = 1
    prediction = np.zeros_like(target)
    prediction[1, 2] = 1
    prediction[5, 4] = 1
    result = match_components(
        prediction,
        target,
        rule="centroid",
        centroid_distance=2.0,
    )
    assert result.num_tp_objects == 2
    assert result.num_fp_components == 0
    assert result.num_fp_pixels == 2


def test_threshold_sweep_has_explicit_endpoints_and_monotone_pixel_fa() -> None:
    record = ScoreMapRecord(
        probability=np.asarray([[0.9, 0.8], [0.2, 0.1]], dtype=np.float32),
        mask=np.asarray([[1, 0], [0, 0]], dtype=np.uint8),
        image_id="toy",
    )
    thresholds = normalise_thresholds([0.5])
    assert thresholds.tolist() == [0.0, 0.5, 1.0]
    rows = sweep_thresholds([record], [0.5])
    assert [row["threshold"] for row in rows] == [0.0, 0.5, 1.0]
    assert [row["fp_pixels"] for row in rows] == [3, 1, 0]
    assert all(
        float(rows[index]["fa_pixel"]) >= float(rows[index + 1]["fa_pixel"])
        for index in range(len(rows) - 1)
    )


def test_threshold_grid_contract_is_versioned() -> None:
    metadata = threshold_grid_metadata()
    assert metadata["schema_version"] == CURVE_SCHEMA_VERSION
    assert metadata["threshold_grid_version"] == THRESHOLD_GRID_VERSION
    assert metadata["thresholds"][0] == 0.0
    assert metadata["thresholds"][-1] == 1.0


def test_adaptive_events_resolve_high_tail_budget_counterexample() -> None:
    record = ScoreMapRecord(
        probability=np.asarray(
            [[0.999995, 0.999999], [0.0, 0.0]],
            dtype=np.float32,
        ),
        mask=np.asarray([[0, 1], [0, 0]], dtype=np.uint8),
        image_id="high-tail",
    )
    rows, audit = sweep_thresholds(
        [record],
        threshold_mode="adaptive",
        event_threshold_cap=None,
        return_metadata=True,
    )
    selected = select_operating_point(rows, pixel_budget=0.1)
    assert selected is not None
    # Strict probability > threshold excludes the equal-score FP while the
    # larger TP remains, an operating point absent from the fixed grid.
    assert selected["threshold"] == pytest.approx(0.999995, abs=1e-7)
    assert selected["pd"] == 1.0
    assert selected["fp_pixels"] == 0
    assert rows[0]["threshold"] == 0.0
    assert rows[-1]["threshold"] == 1.0
    assert audit["threshold_mode"] == "adaptive"
    assert audit["event_candidate_count"] == 2
    assert audit["event_threshold_count"] == 2
    assert audit["event_coverage_fraction_lower_bound"] == 1.0
    assert audit["global_exact"] is False


def test_capped_exact_plan_never_claims_global_exact() -> None:
    record = ScoreMapRecord(
        probability=np.asarray([[0.2, 0.3], [0.4, 0.4]], dtype=np.float32),
        mask=np.zeros((2, 2), dtype=np.uint8),
        image_id="capped",
    )
    rows, audit = sweep_thresholds(
        [record],
        thresholds=[0.5],
        threshold_mode="exact",
        event_threshold_cap=1,
        return_metadata=True,
    )
    assert [row["threshold"] for row in rows] == pytest.approx([0.0, 0.4, 0.5, 1.0])
    assert audit["threshold_mode"] == "exact_capped"
    assert audit["event_thresholds_capped"] is True
    assert audit["event_candidate_count"] == 3
    assert audit["event_threshold_count"] == 1
    assert audit["event_coverage_fraction_lower_bound"] == pytest.approx(1 / 3)
    assert audit["event_coverage_score_lower_bound"] == pytest.approx(0.4)
    assert audit["global_exact"] is False


def test_uncapped_exact_plan_is_globally_exact() -> None:
    record = ScoreMapRecord(
        probability=np.asarray([[0.2, 0.3], [0.4, 0.4]], dtype=np.float32),
        mask=np.zeros((2, 2), dtype=np.uint8),
    )
    _, audit = sweep_thresholds(
        [record],
        thresholds=[0.5],
        threshold_mode="exact",
        event_threshold_cap=None,
        return_metadata=True,
    )
    assert audit["threshold_mode"] == "exact"
    assert audit["event_threshold_count"] == 3
    assert audit["event_coverage_fraction_lower_bound"] == 1.0
    assert audit["global_exact"] is True


def test_cli_defaults_to_query_adaptive_events_and_audits_manifest(
    tmp_path: Path,
) -> None:
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    probability = np.asarray(
        [[0.999995, 0.999999], [0.0, 0.0]],
        dtype=np.float32,
    )
    mask = np.asarray([[0, 1], [0, 0]], dtype=np.uint8)
    np.savez_compressed(
        score_dir / "toy.npz",
        prob=probability,
        mask=mask,
        image_id=np.asarray("toy"),
    )
    curve_path = tmp_path / "curve.csv"
    assert threshold_sweep_main(
        [
            "--score-dir",
            str(score_dir),
            "--output",
            str(curve_path),
            "--event-threshold-cap",
            "0",
        ]
    ) == 0

    rows = read_curve_csv(curve_path)
    selected = select_operating_point(rows, pixel_budget=0.1)
    assert selected is not None
    assert selected["threshold"] == pytest.approx(0.999995, abs=1e-7)
    assert selected["pd"] == 1.0

    manifest = json.loads(
        curve_path.with_suffix(".csv.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["threshold_mode_requested"] == "adaptive"
    assert manifest["threshold_mode"] == "adaptive"
    assert manifest["event_candidate_count"] == 2
    assert manifest["event_threshold_count"] == 2
    assert manifest["event_coverage_score_lower_bound"] == pytest.approx(0.999995)
    assert manifest["event_coverage_fraction_lower_bound"] == 1.0
    assert manifest["global_exact"] is False


def test_threshold_sweep_labels_each_gt_only_once(monkeypatch) -> None:
    import evaluation.component_matching as component_matching

    calls = 0
    original_label = component_matching.label

    def counting_label(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_label(*args, **kwargs)

    monkeypatch.setattr(component_matching, "label", counting_label)
    record = ScoreMapRecord(
        probability=np.asarray([[0.9, 0.2], [0.1, 0.0]], dtype=np.float32),
        mask=np.asarray([[1, 0], [0, 0]], dtype=np.uint8),
    )
    rows = sweep_thresholds([record], thresholds=[0.5])
    # One GT label pass plus one prediction label pass for each of 3 points.
    assert len(rows) == 3
    assert calls == 1 + len(rows)


def test_dual_budget_operating_point_requires_both_constraints() -> None:
    rows = [
        {"threshold": 0.2, "pd": 0.9, "fa_pixel": 0.02, "fa_component_mp": 0.5},
        {"threshold": 0.5, "pd": 0.8, "fa_pixel": 0.005, "fa_component_mp": 2.0},
        {"threshold": 0.7, "pd": 0.7, "fa_pixel": 0.001, "fa_component_mp": 0.5},
    ]
    selected = select_operating_point(
        rows,
        pixel_budget=0.01,
        component_budget=1.0,
    )
    assert selected is not None
    assert selected["threshold"] == 0.7


def test_rejection_aware_dual_budget_metrics() -> None:
    metrics = compute_budget_metrics(
        [
            {"fa_pixel": 0.005, "fa_component_mp": 0.5},
            {"fa_pixel": 0.02, "fa_component_mp": 0.5},
            {"fa_pixel": 100.0, "fa_component_mp": 100.0, "rejected": True},
        ],
        pixel_budget=0.01,
        component_budget=1.0,
    )
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["bsr"] == pytest.approx(0.5)
    assert metrics["unconditional_bsr"] == pytest.approx(1 / 3)
    assert metrics["excess"] == pytest.approx(0.5)
    assert metrics["component_bsr"] == pytest.approx(1.0)


def _write_adapter_replay_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    np.savez_compressed(
        score_dir / "context.npz",
        prob=np.zeros((2, 2), dtype=np.float32),
        mask=np.zeros((2, 2), dtype=np.uint8),
        image_id=np.asarray("context"),
    )
    np.savez_compressed(
        score_dir / "query.npz",
        prob=np.asarray([[0.9, 0.8], [0.1, 0.2]], dtype=np.float32),
        mask=np.asarray([[1, 0], [0, 0]], dtype=np.uint8),
        image_id=np.asarray("query"),
    )
    detector_sha = "a" * 64
    manifest_path = score_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target_dataset": "TARGET",
                "weight_sha256": detector_sha,
                "outer_fold_id": "fold-target",
                "outer_target": "TARGET",
                "detector_source_domains": ["SOURCE-A", "SOURCE-B"],
                "protocol_scope": "multi_source_protocol_candidate",
                "target_exclusion_verified": True,
                "score_type": "sigmoid_probability",
                "threshold_semantics": "prediction = probability > threshold",
                "num_images": 2,
                "items": [
                    {"image_id": "context", "file": "context.npz"},
                    {"image_id": "query", "file": "query.npz"},
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    adapter: dict[str, object] = {
        "outer_fold_id": "fold-target",
        "outer_target": "TARGET",
        "target_domain": "TARGET",
        "detector_source_domains": ["SOURCE-A", "SOURCE-B"],
        "detector_checkpoint_sha": detector_sha,
        "score_manifest_sha256": manifest_sha,
        "score_manifest_target_dataset": "TARGET",
        "score_manifest_detector_checkpoint_sha": detector_sha,
        "context_image_ids": ["context"],
        "query_image_ids": ["query"],
        "budgets": {
            "names": ["pixel", "component"],
            "values": [0.30, 0.0],
            "active": [True, False],
        },
        "threshold": 0.5,
        "reject": False,
    }
    return manifest_path, adapter


def test_adapter_replay_verifies_binding_and_reports_raw_counts(tmp_path: Path) -> None:
    manifest_path, adapter = _write_adapter_replay_fixture(tmp_path)
    result = evaluate_adapter_output(adapter, manifest_path)
    assert result["rejected"] is False
    assert result["query_image_ids"] == ["query"]
    assert result["pd"] == pytest.approx(1.0)
    assert result["fa_pixel"] == pytest.approx(0.25)
    assert result["fa_component_mp"] == pytest.approx(0.0)
    assert result["tp_objects"] == result["gt_objects"] == 1
    assert result["pred_components"] == 1
    assert result["fp_components"] == 0
    assert result["fp_pixels"] == 1
    assert result["total_pixels"] == 4
    assert result["num_images"] == 1


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("score_manifest_sha256", "0" * 64, "SHA-256 mismatch"),
        ("score_manifest_target_dataset", "OTHER", "Target-domain binding mismatch"),
        ("query_image_ids", ["not-query"], "manifest prefix"),
    ],
)
def test_adapter_replay_rejects_binding_mismatches(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    manifest_path, adapter = _write_adapter_replay_fixture(tmp_path)
    adapter[field] = replacement
    with pytest.raises(ValueError, match=message):
        evaluate_adapter_output(adapter, manifest_path)


def test_rejected_adapter_has_no_fabricated_metrics_and_summary_is_coverage_aware(
    tmp_path: Path,
) -> None:
    manifest_path, adapter = _write_adapter_replay_fixture(tmp_path)
    covered = evaluate_adapter_output(adapter, manifest_path)
    adapter["reject"] = True
    rejected = evaluate_adapter_output(adapter, manifest_path)
    assert rejected["rejected"] is True
    assert all(field not in rejected for field in ("pd", "fa_pixel", "tp_objects"))

    summary = summarise_adapter_evaluations([covered, rejected])
    assert summary["coverage"] == pytest.approx(0.5)
    assert summary["bsr"] == pytest.approx(1.0)
    assert summary["excess"] == pytest.approx(0.0)
    assert summary["covered_pd"] == pytest.approx(1.0)
    assert summary["covered_tp_objects"] == 1
    assert summary["covered_gt_objects"] == 1


def test_adapter_replay_cli_writes_json(tmp_path: Path) -> None:
    manifest_path, adapter = _write_adapter_replay_fixture(tmp_path)
    adapter_path = tmp_path / "adapter.json"
    adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    output_path = tmp_path / "evaluation.json"
    assert (
        evaluate_adapter_main(
            [
                "--adapter-output",
                str(adapter_path),
                "--score-manifest",
                str(manifest_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["pd"] == pytest.approx(1.0)
    assert payload["score_manifest_sha256"] == adapter["score_manifest_sha256"]

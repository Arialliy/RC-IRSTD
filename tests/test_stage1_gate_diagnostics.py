from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from data_ext.dataset_identity import build_dataset_record, sha256_file
from evaluation.stage1_gate_diagnostics import (
    RAW_LOGIT_SCORE_RELATION,
    RawLogitItem,
    VerifiedStage1Manifest,
    fit_and_compare_raw_logits,
    load_verified_stage1_manifest,
    summarise_raw_logits,
    validate_paired_stage1_manifests,
)


def _raw_item(
    root: Path,
    image_id: str,
    values: np.ndarray,
    *,
    gray_sha: str = "a" * 64,
) -> RawLogitItem:
    path = root / f"{image_id}.npy"
    array = np.asarray(values, dtype=np.float64)
    np.save(path, array, allow_pickle=False)
    return RawLogitItem(
        image_id=image_id,
        path=path,
        sha256=sha256_file(path),
        dtype="float64",
        shape=tuple(int(value) for value in array.shape),
        gray_file_sha256=gray_sha,
    )


def _verified_manifest(
    path: Path,
    variant: str,
    items: tuple[RawLogitItem, ...],
) -> VerifiedStage1Manifest:
    payload = {
        "target_dataset": "TARGET",
        "outer_fold_id": "leave-TARGET",
        "outer_target": "TARGET",
        "detector_source_domains": ["SOURCE-A", "SOURCE-B"],
        "held_out_domains": ["TARGET"],
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        "protocol_scope": "multi_source_protocol_candidate",
        "detector_source_records": [
            {"source_name": "SOURCE-A", "identity": "1" * 64},
            {"source_name": "SOURCE-B", "identity": "2" * 64},
        ],
        "target_dataset_record": {
            "dataset_identity_sha256": "3" * 64,
            "split_sha256": "4" * 64,
            "ordered_sample_ids_sha256": "5" * 64,
            "split_image_artifact_sha256": "6" * 64,
            "split_image_artifact_items": [
                {"sample_id": item.image_id, "image_sha256": item.gray_file_sha256}
                for item in items
            ],
        },
        "split_contract": {
            "role": "detector_diagnostic",
            "selected_split_sha256": "4" * 64,
            "selected_num_images": len(items),
            "selected_ids_sha256": "5" * 64,
            "derived_split_manifest_sha256": "7" * 64,
            "partition_scope": "official_train_derived_development_diagnostic",
        },
        "raw_logit_provenance": {
            "proof_mode": "verified_held_out_domain_exclusion",
            "training_seed": 42,
            "segmentation_loss_implementation": {
                "qualified_name": "losses.sls.SLSIoULoss"
            },
            "same_dataset_partition_audit": None,
        },
        "weight_sha256": ("8" if variant == "D0" else "9") * 64,
    }
    return VerifiedStage1Manifest(
        path=path,
        manifest_sha256=("b" if variant == "D0" else "c") * 64,
        payload=payload,
        variant=variant,
        items=items,
        raw_logit_content_sha256=("d" if variant == "D0" else "e") * 64,
    )


def test_extreme_raw_logits_report_exact_float64_saturation_and_ties(
    tmp_path: Path,
) -> None:
    item = _raw_item(
        tmp_path,
        "extreme",
        np.asarray([[-1000.0, -1000.0, 0.0, 1000.0, 1000.0]]),
    )
    summary = summarise_raw_logits((item,))

    assert summary["raw_logit"]["min"] == -1000.0
    assert summary["raw_logit"]["max"] == 1000.0
    assert summary["raw_logit"]["range"] == 2000.0
    sigmoid = summary["float64_sigmoid_of_raw_logit_diagnostic"]
    assert sigmoid["pointwise_equal_to_saved_score"] is False
    assert sigmoid["exact_zero_count"] == 2
    assert sigmoid["exact_one_count"] == 2
    assert sigmoid["maximum_tie"]["count"] == 2


def test_pure_positive_shift_scale_fit_preserves_pixel_ranking(
    tmp_path: Path,
) -> None:
    d0_arrays = (
        np.asarray([[-3.0, -1.0, 0.0], [2.0, 4.0, 5.0]]),
        np.asarray([[8.0, 13.0], [21.0, 34.0]]),
    )
    d0_items = tuple(
        _raw_item(tmp_path, f"d0_{index}", values)
        for index, values in enumerate(d0_arrays)
    )
    d3_items = tuple(
        _raw_item(tmp_path, f"d3_{index}", 2.5 * values - 7.0)
        for index, values in enumerate(d0_arrays)
    )
    d3_items = tuple(
        replace(item, image_id=d0_items[index].image_id)
        for index, item in enumerate(d3_items)
    )

    report = fit_and_compare_raw_logits(d0_items, d3_items)
    fit = report["linear_fit_d3_from_d0"]
    assert fit["scale_a"] == pytest.approx(2.5, abs=1e-12)
    assert fit["shift_b"] == pytest.approx(-7.0, abs=1e-12)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    assert fit["rmse"] == pytest.approx(0.0, abs=1e-12)
    ranking = report["ranking_change"]
    assert ranking["rank_changed_pixel_count"] == 0
    assert ranking["ordering_identical_for_all_images"] is True
    assert ranking["stable_ordinal_spearman_min"] == pytest.approx(1.0)


def test_id_and_provenance_mismatches_are_rejected(tmp_path: Path) -> None:
    d0_item = _raw_item(tmp_path, "sample", np.asarray([[0.0, 1.0]]))
    d3_item = _raw_item(tmp_path, "sample_d3", np.asarray([[1.0, 2.0]]))
    d0 = _verified_manifest(tmp_path / "d0-manifest.json", "D0", (d0_item,))
    d3 = _verified_manifest(
        tmp_path / "d3-manifest.json",
        "D3",
        (replace(d3_item, image_id="different-sample"),),
    )
    with pytest.raises(ValueError, match="image IDs"):
        validate_paired_stage1_manifests(d0, d3)

    matched_d3_item = replace(d3_item, image_id="sample")
    matched_d3 = _verified_manifest(
        tmp_path / "d3-manifest.json", "D3", (matched_d3_item,)
    )
    changed_payload = dict(matched_d3.payload)
    changed_payload["outer_fold_id"] = "different-fold"
    mismatched_provenance = replace(matched_d3, payload=changed_payload)
    with pytest.raises(ValueError, match="outer_fold_id"):
        validate_paired_stage1_manifests(d0, mismatched_provenance)


def _make_development_dataset(
    repository_root: Path,
) -> tuple[Path, Path, Path, Path]:
    dataset = repository_root / "datasets" / "NUAA-SIRST"
    (dataset / "images").mkdir(parents=True)
    (dataset / "masks").mkdir()
    (dataset / "img_idx").mkdir()
    image_ids = ("fit_a", "diagnostic_b", "quarantine_c", "test_d")
    for index, image_id in enumerate(image_ids, start=1):
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[:, :, 0] = index * 19
        image[0, index] = (index * 31, index * 17, index * 13)
        mask = np.zeros((4, 8), dtype=np.uint8)
        mask[2, index] = 255
        Image.fromarray(image).save(dataset / "images" / f"{image_id}.png")
        Image.fromarray(mask).save(dataset / "masks" / f"{image_id}.png")

    official_train = dataset / "img_idx" / "train_NUAA-SIRST.txt"
    official_test = dataset / "img_idx" / "test_NUAA-SIRST.txt"
    official_train.write_text(
        "fit_a\ndiagnostic_b\nquarantine_c\n", encoding="utf-8"
    )
    official_test.write_text("test_d\n", encoding="utf-8")

    split_root = repository_root / "splits" / "aaai27_v2"
    role_root = split_root / "manifest-selected-directory"
    role_root.mkdir(parents=True)
    role_payloads = {
        "effective_development_train.txt": "fit_a\ndiagnostic_b\n",
        "detector_fit.txt": "fit_a\n",
        "detector_diagnostic.txt": "diagnostic_b\n",
        "quarantined_official_train_ids.txt": "quarantine_c\n",
    }
    for name, content in role_payloads.items():
        (role_root / name).write_text(content, encoding="utf-8")

    manifest_path = split_root / "manifest.json"
    relative_role_root = role_root.relative_to(split_root)
    manifest = {
        "schema_version": "rc-irstd.aaai27-official-train-splits.v2",
        "artifact_type": "official_train_derived_role_splits",
        "role_contract": {
            "official_test_emitted": False,
            "official_test_labels_read_for_quarantine": False,
            "outer_target_official_train_used_for_detector_fit": False,
            "outer_target_detector_diagnostic_used_for_development_evaluation": True,
            "outer_target_diagnostic_selects_checkpoint": False,
            "detector_checkpoint_selection": "fixed_last",
            "detector_diagnostic_used_for_checkpoint_selection": False,
        },
        "datasets": [
            {
                "dataset_name": "NUAA-SIRST",
                "dataset_root": "datasets/NUAA-SIRST",
                "official_train_split": (
                    "datasets/NUAA-SIRST/img_idx/train_NUAA-SIRST.txt"
                ),
                "official_train_split_sha256": sha256_file(official_train),
                "official_train_count": 3,
                "official_test_split": (
                    "datasets/NUAA-SIRST/img_idx/test_NUAA-SIRST.txt"
                ),
                "official_test_split_sha256": sha256_file(official_test),
                "official_test_count": 1,
                "official_train_test_id_overlap_count": 0,
                "detector": {
                    "fit_file": (relative_role_root / "detector_fit.txt").as_posix(),
                    "fit_sha256": sha256_file(role_root / "detector_fit.txt"),
                    "fit_count": 1,
                    "diagnostic_file": (
                        relative_role_root / "detector_diagnostic.txt"
                    ).as_posix(),
                    "diagnostic_sha256": sha256_file(
                        role_root / "detector_diagnostic.txt"
                    ),
                    "diagnostic_count": 1,
                },
                "development_quarantine": {
                    "effective_development_train_file": (
                        relative_role_root / "effective_development_train.txt"
                    ).as_posix(),
                    "effective_development_train_sha256": sha256_file(
                        role_root / "effective_development_train.txt"
                    ),
                    "effective_development_train_count": 2,
                    "quarantined_file": (
                        relative_role_root / "quarantined_official_train_ids.txt"
                    ).as_posix(),
                    "quarantined_sha256": sha256_file(
                        role_root / "quarantined_official_train_ids.txt"
                    ),
                    "quarantined_count": 1,
                    "partition_of_official_train": True,
                },
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return (
        dataset,
        manifest_path,
        role_root / "detector_fit.txt",
        role_root / "detector_diagnostic.txt",
    )


def test_save_raw_logits_keeps_score_map_identical_to_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evaluation.export_score_maps as exporter

    repository = tmp_path / "repository"
    dataset, split_manifest, detector_fit, diagnostic = _make_development_dataset(
        repository
    )
    source_record = build_dataset_record(
        dataset,
        detector_fit,
        ["fit_a"],
        source_name="NUAA-SIRST",
        training_artifacts=[
            (
                dataset / "images" / "fit_a.png",
                dataset / "masks" / "fit_a.png",
            )
        ],
    )
    checkpoint = tmp_path / "d0.pt"
    torch.save(
        {
            "state_dict": {"unused": torch.zeros(1)},
            "detector_source_domains": ["NUAA-SIRST"],
            "detector_source_records": [source_record],
            "held_out_domains": ["DEVELOPMENT-ONLY-ALL-THREE-DIAGNOSTIC"],
            "outer_fold_id": "development-all-three",
            "outer_target": "DEVELOPMENT-ONLY-ALL-THREE-DIAGNOSTIC",
            "checkpoint_selection": "fixed_last_no_test_or_target_validation",
            "protocol_scope": "single_source_inner_smoke_not_main_result",
            "seed": 42,
            "run_config_sha256": "f" * 64,
            "risk_objective_contract": {
                "name": "multiscale_sls_segmentation_only",
                "stage1_variant": "D0",
            },
            "segmentation_loss_implementation": {
                "qualified_name": "losses.sls.SLSIoULoss"
            },
        },
        checkpoint,
    )

    class PatternDetector(torch.nn.Module):
        def forward(self, image: torch.Tensor, warm_flag: bool) -> torch.Tensor:
            del warm_flag
            values = torch.linspace(
                -8.0,
                8.0,
                image.shape[2] * image.shape[3],
                dtype=image.dtype,
                device=image.device,
            )
            return values.reshape(1, 1, image.shape[2], image.shape[3])

    monkeypatch.setattr(
        exporter, "load_model", lambda *args, **kwargs: PatternDetector()
    )
    common = {
        "dataset_dir": dataset,
        "weight_path": checkpoint,
        "base_size": 16,
        "split_file": diagnostic,
        "split_role": "detector_diagnostic",
        "derived_split_manifest": split_manifest,
        "derived_split_manifest_sha256": sha256_file(split_manifest),
        "device": "cpu",
    }
    default_root = tmp_path / "default-scores"
    raw_root = tmp_path / "raw-scores"
    default_manifest = exporter.export_score_maps(
        output_dir=default_root,
        **common,
    )
    raw_manifest = exporter.export_score_maps(
        output_dir=raw_root,
        save_raw_logits=True,
        **common,
    )
    assert "raw_logits_exported" not in default_manifest
    assert all(
        "raw_logit_file" not in item for item in default_manifest["items"]
    )

    with np.load(
        default_root / default_manifest["items"][0]["file"], allow_pickle=False
    ) as default_score, np.load(
        raw_root / raw_manifest["items"][0]["file"], allow_pickle=False
    ) as raw_score:
        assert np.array_equal(default_score["prob"], raw_score["prob"])
        assert default_score["prob"].tobytes() == raw_score["prob"].tobytes()

    raw_item = raw_manifest["items"][0]
    raw_path = raw_root / raw_item["raw_logit_file"]
    assert raw_item["raw_logit_file_sha256"] == sha256_file(raw_path)
    saved_raw = np.load(raw_path, allow_pickle=False)
    assert saved_raw.dtype == np.float64
    assert list(saved_raw.shape) == raw_item["original_hw"]
    assert raw_manifest["raw_logit_score_relation"] == RAW_LOGIT_SCORE_RELATION
    assert "not_pointwise_equal" in RAW_LOGIT_SCORE_RELATION
    verified = load_verified_stage1_manifest(
        raw_root / "manifest.json", expected_variant="D0"
    )
    assert verified.items[0].image_id == "diagnostic_b"

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

import evaluation.export_stage2_development_scores as stage2_exporter
from data_ext.score_manifest_artifacts import verify_score_manifest_artifacts
from data_ext.stage2_score_manifest import (
    FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE,
    OOF_HOLDOUT_STAGE2_FIT,
    OOF_TRAIN_SOURCE_REFERENCE,
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
    SOURCE_DIAGNOSTIC_VALIDATION,
    STAGE2_DEVELOPMENT_ROLES,
    STAGE2_SCORE_ARTIFACT_TYPE,
    STAGE2_SCORE_MANIFEST_SCHEMA,
    STAGE2_SCORE_RECORDS_ALGORITHM,
    STRICT_THRESHOLD_SEMANTICS,
    stage2_score_records_sha256,
    verify_stage2_score_manifest,
)
from evaluation.export_stage2_development_scores import (
    export_stage2_development_scores,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _guardrails() -> dict[str, bool]:
    return {
        "development_only": True,
        "official_test_split_files_opened": False,
        "official_test_ids_materialized": False,
        "official_test_images_opened": False,
        "mask_or_label_files_opened": False,
        "predictions_scores_checkpoints_or_metrics_opened": False,
    }


def _selected_record(root: Path, index: int, domain: str) -> dict[str, Any]:
    image_id = f"sample_{index:02d}"
    image_path = root / "development_images" / domain.lower() / f"{image_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.full((5 + index, 7 + index), 30 + index, dtype=np.uint8), mode="L"
    ).save(image_path)
    image_sha = _sha(image_path)
    return {
        "canonical_id": f"{domain}::{image_id}",
        "image_id": image_id,
        "original_image_path": _rel(image_path, root),
        "original_image_sha256": image_sha,
        "exclusion_group_id": f"SINGLETON::{domain}::{image_id}::{image_sha}",
        "near_duplicate_cluster_id_or_unique_sentinel": (
            f"UNIQUE_NO_CONFIRMED_NEAR_DUPLICATE::{domain}::{image_id}"
        ),
        "source_role_record_index": index,
    }


def _selection_payload(
    root: Path,
    role: str,
    records: list[dict[str, Any]],
    *,
    outer_fold: str,
    outer_target: str,
    source_domain: str,
    detector_role: str,
    oof_fold_index: int | None,
) -> dict[str, Any]:
    if role in {
        OOF_TRAIN_SOURCE_REFERENCE,
        FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE,
    }:
        return {
            "artifact_type": "rc_irstd_stage2_detector_selection",
            "development_only": True,
            "execution_authorized": False,
            "official_test_accessed": False,
            "observed_results": None,
            "outer_fold_id": outer_fold,
            "outer_target_domain": outer_target,
            "source_domain": source_domain,
            "base_seed": 42,
            "derived_seed": 101,
            "detector_role": detector_role,
            "oof_fold_index": oof_fold_index,
            "record_count": len(records),
            "records": records,
        }
    if role == OOF_HOLDOUT_STAGE2_FIT:
        assignment_records = [
            {**record, "oof_fold_index": oof_fold_index, "source_role": "detector_fit"}
            for record in records
        ]
        return {
            "artifact_type": "rc_irstd_stage2_detector_fit_group_assignment",
            "artifact_status": "DEVELOPMENT_ONLY_RESULT_FREE",
            "domain": source_domain,
            "execution_authorized": False,
            "observed_results": None,
            "guardrails": _guardrails(),
            "fold_counts": {str(oof_fold_index): len(assignment_records)},
            "records": assignment_records,
        }
    return {
        "artifact_type": "rc_irstd_stage2_role_pure_episode_windows",
        "artifact_status": "DEVELOPMENT_ONLY_RESULT_FREE",
        "domain": source_domain,
        "outer_fold_id": outer_fold,
        "outer_target_domain": outer_target,
        "episode_role": role,
        "execution_authorized": False,
        "observed_results": None,
        "guardrails": _guardrails(),
        "window_record_count": len(records),
        "windows": [
            {
                "context_records": records[:1],
                "query_records": records[1:],
            }
        ],
    }


def _write_score_npz(
    path: Path,
    selected: Mapping[str, Any],
    source_domain: str,
    *,
    dtype: np.dtype[Any] = np.dtype("float64"),
    extra: bool = False,
) -> tuple[list[int], str]:
    image_path = path.parents[1] / selected["original_image_path"]
    with Image.open(image_path) as image:
        original_hw = [image.height, image.width]
    values: dict[str, np.ndarray] = {
        "prob": np.full(original_hw, 0.25, dtype=dtype),
        "raw_logit": np.full(original_hw, -1.0, dtype=dtype),
        "canonical_id": np.asarray(selected["canonical_id"]),
        "image_id": np.asarray(selected["image_id"]),
        "source_domain": np.asarray(source_domain),
        "original_hw": np.asarray(original_hw, dtype=np.int64),
        "input_hw": np.asarray([16, 16], dtype=np.int64),
        "resized_hw": np.asarray([16, 16], dtype=np.int64),
        "padding_ltrb": np.asarray([0, 0, 0, 0], dtype=np.int64),
        "resize_mode": np.asarray("resize"),
    }
    if extra:
        values["mask"] = np.zeros(original_hw, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    return original_hw, _sha(path)


def _fixture(
    tmp_path: Path,
    role: str = FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE,
    *,
    record_count: int = 2,
) -> dict[str, Any]:
    root = tmp_path
    outer_fold = "outer_leave_nuaa_sirst"
    outer_target = "NUAA-SIRST"
    source_domain = (
        outer_target if role == OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT else "NUDT-SIRST"
    )
    detector_role = (
        "detector_oof"
        if role in {OOF_TRAIN_SOURCE_REFERENCE, OOF_HOLDOUT_STAGE2_FIT}
        else "detector_full_fit"
    )
    oof_index = 0 if detector_role == "detector_oof" else None
    selected = [_selected_record(root, i, source_domain) for i in range(record_count)]
    selection = _selection_payload(
        root,
        role,
        selected,
        outer_fold=outer_fold,
        outer_target=outer_target,
        source_domain=source_domain,
        detector_role=detector_role,
        oof_fold_index=oof_index,
    )
    selection_path = root / "contracts" / "selection.json"
    _write_json(selection_path, selection)
    selection_sha = _sha(selection_path)
    other_path = root / "contracts" / "other-selection.json"
    _write_json(other_path, {"synthetic": True})

    input_paths: dict[str, Path] = {}
    for name in (
        "detector_config",
        "runtime_config",
        "seed_manifest",
        "materialization_index",
        "release_artifact",
        "environment_artifact",
        "runtime_contract",
    ):
        artifact = root / "bindings" / f"{name}.json"
        _write_json(artifact, {"artifact": name, "development_only": True})
        input_paths[name] = artifact
    checkpoint_path = root / "bindings" / "checkpoint.pt"
    checkpoint_path.write_bytes(b"synthetic-checkpoint-bytes")

    run_bindings: dict[str, Any] = {
        name: {"path": _rel(path, root), "sha256": _sha(path)}
        for name, path in input_paths.items()
        if name
        not in {"environment_artifact", "runtime_config", "runtime_contract"}
    }
    run_bindings["materialization_artifacts_sha256"] = (
        {}
        if selection["artifact_type"] == "rc_irstd_stage2_detector_selection"
        else {_rel(selection_path, root): selection_sha}
    )
    run = {
        "development_only": True,
        "official_test_accessed": False,
        "outer_fold_id": outer_fold,
        "outer_target_domain": outer_target,
        "source_domains": ["NUDT-SIRST", "IRSTD-1K"],
        "base_seed": 42,
        "derived_seed": 101,
        "detector_role": detector_role,
        "oof_fold_index": oof_index,
        "selection_contracts": [
            {
                "path": _rel(selection_path, root),
                "sha256": selection_sha,
            },
            {"path": _rel(other_path, root), "sha256": _sha(other_path)},
        ],
        "bindings": run_bindings,
    }
    run_path = root / "contracts" / "run.json"
    _write_json(run_path, run)
    _write_json(
        input_paths["runtime_config"],
        {"base_size": 16, "resize_mode": "resize"},
    )
    _write_json(input_paths["environment_artifact"], {"synthetic": True})
    _write_json(
        input_paths["runtime_contract"],
        {
            "schema_version": "rc-irstd.stage2-detector-runtime-contract.v1",
            "development_only": True,
            "official_test_accessed": False,
            "observed_results": None,
            "outer_fold_id": outer_fold,
            "outer_target_domain": outer_target,
            "detector_role": detector_role,
            "oof_fold_index": oof_index,
            "base_seed": 42,
            "derived_seed": 101,
            "input_run_contract": {
                "path": _rel(run_path, root),
                "sha256": _sha(run_path),
            },
            "run_config": {
                "path": input_paths["runtime_config"].name,
                "sha256": _sha(input_paths["runtime_config"]),
            },
            "environment_artifact": {
                "path": input_paths["environment_artifact"].name,
                "sha256": _sha(input_paths["environment_artifact"]),
            },
        },
    )
    torch.save(
        {
            "format_version": "rc-irstd.detector-inference.v1",
            "state_dict": {"synthetic.weight": torch.zeros(1)},
            "epoch": 0,
            "seed": 101,
            "source_names": ["NUDT-SIRST", "IRSTD-1K"],
            "outer_fold_id": outer_fold,
            "outer_target": outer_target,
            "held_out_domains": [outer_target],
            "detector_role": detector_role,
            "oof_fold_index": oof_index,
            "checkpoint_selection": "fixed_last_no_test_or_target_validation",
            "run_config_sha256": _sha(input_paths["runtime_config"]),
            "run_contract_sha256": _sha(run_path),
            "training_args": {"base_size": 16, "resize_mode": "resize"},
            "inference_geometry": {
                "input_hw": [16, 16],
                "resize_mode": "resize",
            },
            "official_test_accessed": False,
            "stage2_runtime_artifacts": {
                "run_config": {
                    "path": input_paths["runtime_config"].name,
                    "sha256": _sha(input_paths["runtime_config"]),
                },
                "environment_artifact": {
                    "path": input_paths["environment_artifact"].name,
                    "sha256": _sha(input_paths["environment_artifact"]),
                },
                "runtime_contract": {
                    "path": input_paths["runtime_contract"].name,
                    "sha256": _sha(input_paths["runtime_contract"]),
                },
            },
        },
        checkpoint_path,
    )

    score_records: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        score_path = root / "scores" / f"score-{index}.npz"
        original_hw, score_sha = _write_score_npz(
            score_path, record, source_domain
        )
        score_records.append(
            {
                "record_index": index,
                **record,
                "source_domain": source_domain,
                "score_file": _rel(score_path, root),
                "score_file_sha256": score_sha,
                "original_hw": original_hw,
                "input_hw": [16, 16],
                "resized_hw": [16, 16],
                "padding_ltrb": [0, 0, 0, 0],
                "resize_mode": "resize",
            }
        )
    bindings = {
        "selection_contract": {
            "path": _rel(selection_path, root),
            "sha256": selection_sha,
        },
        "run_contract": {"path": _rel(run_path, root), "sha256": _sha(run_path)},
        "checkpoint": {
            "path": _rel(checkpoint_path, root),
            "sha256": _sha(checkpoint_path),
        },
        "detector_config": run_bindings["detector_config"],
        "runtime_config": {
            "path": _rel(input_paths["runtime_config"], root),
            "sha256": _sha(input_paths["runtime_config"]),
        },
        "seed_manifest": run_bindings["seed_manifest"],
        "materialization_index": run_bindings["materialization_index"],
        "release_artifact": run_bindings["release_artifact"],
        "environment_artifact": {
            "path": _rel(input_paths["environment_artifact"], root),
            "sha256": _sha(input_paths["environment_artifact"]),
        },
        "runtime_contract": {
            "path": _rel(input_paths["runtime_contract"], root),
            "sha256": _sha(input_paths["runtime_contract"]),
        },
    }
    manifest = {
        "schema_version": STAGE2_SCORE_MANIFEST_SCHEMA,
        "artifact_type": STAGE2_SCORE_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY",
        "development_only": True,
        "execution_scope": "stage2_development",
        "official_test_accessed": False,
        "labels_embedded": False,
        "native_resolution": True,
        "restored_to_original_hw": True,
        "path_anchor": "repository_root",
        "role": role,
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "score_type": "sigmoid_probability",
        "score_dtype": "float64",
        "sigmoid_compute_dtype": "float64",
        "raw_logits_exported": True,
        "raw_logit_dtype": "float64",
        "raw_logit_space": (
            "native_original_hw_spatially_aligned_restored_model_logit"
        ),
        "probability_space": (
            "native_original_hw_float64_sigmoid_then_spatial_restore"
        ),
        "outer_fold_id": outer_fold,
        "outer_target": outer_target,
        "source_domain": source_domain,
        "base_seed": 42,
        "derived_seed": 101,
        "detector_role": detector_role,
        "oof_fold_index": oof_index,
        "input_hw": [16, 16],
        "resize_mode": "resize",
        "bindings": bindings,
        "num_images": len(score_records),
        "records_content_sha256_algorithm": STAGE2_SCORE_RECORDS_ALGORITHM,
        "records_content_sha256": stage2_score_records_sha256(score_records),
        "records": score_records,
    }
    manifest_path = root / "scores" / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "root": root,
        "path": manifest_path,
        "manifest": manifest,
        "selection": selection,
        "selection_path": selection_path,
        "run": run,
        "run_path": run_path,
        "checkpoint_path": checkpoint_path,
    }


def _rewrite_manifest(fixture: dict[str, Any], payload: Mapping[str, Any]) -> str:
    _write_json(fixture["path"], payload)
    return _sha(fixture["path"])


def _rebind_score_and_manifest(
    fixture: dict[str, Any],
    payload: dict[str, Any],
    score_path: Path,
) -> str:
    payload["records"][0]["score_file_sha256"] = _sha(score_path)
    payload["records_content_sha256"] = stage2_score_records_sha256(
        payload["records"]
    )
    return _rewrite_manifest(fixture, payload)


def _export_fixture(
    fixture: Mapping[str, Any],
    output: Path,
    *,
    model_factory: Any,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    return export_stage2_development_scores(
        fixture["selection_path"],
        fixture["run_path"],
        fixture["checkpoint_path"],
        output,
        selection_contract_sha256=_sha(fixture["selection_path"]),
        run_contract_sha256=_sha(fixture["run_path"]),
        checkpoint_sha256=(
            _sha(fixture["checkpoint_path"])
            if checkpoint_sha256 is None
            else checkpoint_sha256
        ),
        role=fixture["manifest"]["role"],
        device="cpu",
        repository_root=fixture["root"],
        model_factory=model_factory,
    )


def _assert_no_export_residue(output: Path) -> None:
    assert not os.path.lexists(output)
    assert not os.path.lexists(
        output.parent / f".{output.name}.stage2-export.lock"
    )
    assert not tuple(
        output.parent.glob(f".{output.name}.stage2-staging-*")
    )


@pytest.mark.parametrize("role", STAGE2_DEVELOPMENT_ROLES)
def test_all_five_development_roles_verify(tmp_path: Path, role: str) -> None:
    fixture = _fixture(tmp_path, role)
    verified = verify_stage2_score_manifest(
        fixture["path"], _sha(fixture["path"]), role, repository_root=tmp_path
    )
    assert verified.role == role
    assert len(verified.records) == 2
    assert verified.payload["official_test_accessed"] is False


@pytest.mark.parametrize("value", [True, 0, 1, "false", None])
def test_official_test_accessed_requires_exact_false(
    tmp_path: Path, value: object
) -> None:
    fixture = _fixture(tmp_path)
    payload = deepcopy(fixture["manifest"])
    payload["official_test_accessed"] = value
    digest = _rewrite_manifest(fixture, payload)
    with pytest.raises((TypeError, ValueError)):
        verify_stage2_score_manifest(
            fixture["path"], digest, payload["role"], repository_root=tmp_path
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("development_only", 1),
        ("labels_embedded", 0),
        ("native_resolution", "true"),
        ("restored_to_original_hw", False),
        ("raw_logits_exported", None),
    ],
)
def test_all_contract_booleans_are_exact(
    tmp_path: Path, field: str, value: object
) -> None:
    fixture = _fixture(tmp_path)
    payload = deepcopy(fixture["manifest"])
    payload[field] = value
    digest = _rewrite_manifest(fixture, payload)
    with pytest.raises((TypeError, ValueError)):
        verify_stage2_score_manifest(
            fixture["path"], digest, payload["role"], repository_root=tmp_path
        )


def test_expected_manifest_sha_and_required_role_are_mandatory(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_stage2_score_manifest(
            fixture["path"], "0" * 64, fixture["manifest"]["role"], repository_root=tmp_path
        )
    with pytest.raises(ValueError, match="role"):
        verify_stage2_score_manifest(
            fixture["path"], _sha(fixture["path"]), OOF_TRAIN_SOURCE_REFERENCE, repository_root=tmp_path
        )


@pytest.mark.parametrize(
    "binding_name",
    [
        "selection_contract",
        "run_contract",
        "checkpoint",
        "detector_config",
        "runtime_config",
        "seed_manifest",
        "materialization_index",
        "release_artifact",
        "environment_artifact",
        "runtime_contract",
    ],
)
def test_every_provenance_binding_hash_is_enforced(
    tmp_path: Path, binding_name: str
) -> None:
    fixture = _fixture(tmp_path)
    payload = deepcopy(fixture["manifest"])
    payload["bindings"][binding_name]["sha256"] = "0" * 64
    digest = _rewrite_manifest(fixture, payload)
    with pytest.raises(ValueError, match="sha256|SHA-256"):
        verify_stage2_score_manifest(
            fixture["path"], digest, payload["role"], repository_root=tmp_path
        )


@pytest.mark.parametrize(
    "bad_path",
    [
        "/tmp/score.npz",
        "../score.npz",
        "a/../b.npz",
        "official_test/score.npz",
    ],
)
def test_absolute_and_traversal_paths_are_rejected(
    tmp_path: Path, bad_path: str
) -> None:
    fixture = _fixture(tmp_path)
    payload = deepcopy(fixture["manifest"])
    payload["records"][0]["score_file"] = bad_path
    payload["records_content_sha256"] = stage2_score_records_sha256(payload["records"])
    digest = _rewrite_manifest(fixture, payload)
    with pytest.raises(ValueError, match="relative|traversal|canonical|official-test"):
        verify_stage2_score_manifest(
            fixture["path"], digest, payload["role"], repository_root=tmp_path
        )


def test_symlink_component_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    real = tmp_path / "scores" / "score-0.npz"
    link_dir = tmp_path / "linked"
    link_dir.symlink_to(real.parent, target_is_directory=True)
    payload = deepcopy(fixture["manifest"])
    payload["records"][0]["score_file"] = "linked/score-0.npz"
    payload["records_content_sha256"] = stage2_score_records_sha256(payload["records"])
    digest = _rewrite_manifest(fixture, payload)
    with pytest.raises(ValueError, match="symlink"):
        verify_stage2_score_manifest(
            fixture["path"], digest, payload["role"], repository_root=tmp_path
        )


@pytest.mark.parametrize(
    "name",
    [".export_incomplete", "manifest.json.sha256"],
)
def test_dangling_bundle_marker_or_sidecar_symlink_is_rejected(
    tmp_path: Path,
    name: str,
) -> None:
    fixture = _fixture(tmp_path)
    link = fixture["path"].parent / name
    link.symlink_to("missing-bundle-artifact")
    with pytest.raises((RuntimeError, ValueError), match="incomplete|symlink"):
        verify_stage2_score_manifest(
            fixture["path"],
            _sha(fixture["path"]),
            fixture["manifest"]["role"],
            repository_root=tmp_path,
        )


def test_order_identity_and_duplicate_output_mutations_fail(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    for mutation in ("reorder", "identity", "duplicate"):
        payload = deepcopy(fixture["manifest"])
        if mutation == "reorder":
            payload["records"] = list(reversed(payload["records"]))
            for index, record in enumerate(payload["records"]):
                record["record_index"] = index
        elif mutation == "identity":
            payload["records"][0]["canonical_id"] += "-mutated"
        else:
            payload["records"][1]["score_file"] = payload["records"][0]["score_file"]
        payload["records_content_sha256"] = stage2_score_records_sha256(payload["records"])
        digest = _rewrite_manifest(fixture, payload)
        with pytest.raises(ValueError):
            verify_stage2_score_manifest(
                fixture["path"], digest, payload["role"], repository_root=tmp_path
            )


def test_records_content_digest_is_ordered_and_enforced(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = deepcopy(fixture["manifest"])
    payload["records_content_sha256"] = "f" * 64
    digest = _rewrite_manifest(fixture, payload)
    with pytest.raises(ValueError, match="records_content"):
        verify_stage2_score_manifest(
            fixture["path"], digest, payload["role"], repository_root=tmp_path
        )


@pytest.mark.parametrize("mutation", ["float32", "nan", "range", "extra", "identity"])
def test_native_raw_logit_probability_npz_contract_is_strict(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path, record_count=1)
    payload = deepcopy(fixture["manifest"])
    record = payload["records"][0]
    score_path = tmp_path / record["score_file"]
    selected = fixture["selection"]["records"][0]
    original_hw = record["original_hw"]
    arrays: dict[str, np.ndarray] = {
        "prob": np.full(original_hw, 0.25, dtype=np.float64),
        "raw_logit": np.full(original_hw, -1.0, dtype=np.float64),
        "canonical_id": np.asarray(selected["canonical_id"]),
        "image_id": np.asarray(selected["image_id"]),
        "source_domain": np.asarray(record["source_domain"]),
        "original_hw": np.asarray(original_hw, dtype=np.int64),
        "input_hw": np.asarray([16, 16], dtype=np.int64),
        "resized_hw": np.asarray([16, 16], dtype=np.int64),
        "padding_ltrb": np.asarray([0, 0, 0, 0], dtype=np.int64),
        "resize_mode": np.asarray("resize"),
    }
    if mutation == "float32":
        arrays["raw_logit"] = arrays["raw_logit"].astype(np.float32)
    elif mutation == "nan":
        arrays["raw_logit"][0, 0] = np.nan
    elif mutation == "range":
        arrays["prob"][0, 0] = 1.1
    elif mutation == "extra":
        arrays["mask"] = np.zeros(original_hw, dtype=np.uint8)
    else:
        arrays["canonical_id"] = np.asarray("wrong::identity")
    np.savez_compressed(score_path, **arrays)
    record["score_file_sha256"] = _sha(score_path)
    payload["records_content_sha256"] = stage2_score_records_sha256(payload["records"])
    digest = _rewrite_manifest(fixture, payload)
    with pytest.raises((TypeError, ValueError)):
        verify_stage2_score_manifest(
            fixture["path"], digest, payload["role"], repository_root=tmp_path
        )


@pytest.mark.parametrize("mutation", ["wrong_order", "duplicate_zip_member"])
def test_npz_member_table_is_exact_ordered_and_unique_after_rebinding(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path, record_count=1)
    payload = deepcopy(fixture["manifest"])
    score_path = tmp_path / payload["records"][0]["score_file"]
    with np.load(score_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    if mutation == "wrong_order":
        names = list(arrays)
        names[0], names[1] = names[1], names[0]
        np.savez_compressed(
            score_path,
            **{name: arrays[name] for name in names},
        )
    else:
        with zipfile.ZipFile(score_path, mode="a") as archive:
            duplicate_payload = archive.read("prob.npy")
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("prob.npy", duplicate_payload)
    digest = _rebind_score_and_manifest(fixture, payload, score_path)
    with pytest.raises(ValueError, match="member|field order"):
        verify_stage2_score_manifest(
            fixture["path"],
            digest,
            payload["role"],
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    "field",
    ["canonical_id", "image_id", "source_domain", "resize_mode"],
)
def test_npz_strings_require_true_zero_dimensional_scalars_after_rebinding(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _fixture(tmp_path, record_count=1)
    payload = deepcopy(fixture["manifest"])
    score_path = tmp_path / payload["records"][0]["score_file"]
    with np.load(score_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    arrays[field] = np.asarray([arrays[field].item()])
    np.savez_compressed(score_path, **arrays)
    digest = _rebind_score_and_manifest(fixture, payload, score_path)
    with pytest.raises(ValueError, match="0-D"):
        verify_stage2_score_manifest(
            fixture["path"],
            digest,
            payload["role"],
            repository_root=tmp_path,
        )


def test_legacy_verifier_does_not_accept_stage2_v4(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises((ValueError, TypeError, KeyError)):
        verify_score_manifest_artifacts(fixture["path"])


class _DummyDetector(torch.nn.Module):
    def forward(self, image: torch.Tensor, _: bool) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(
            (image.shape[0], 1, image.shape[-2], image.shape[-1]),
            dtype=image.dtype,
            device=image.device,
        )
        return logits, logits


class _CountingDetector(_DummyDetector):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(
        self,
        image: torch.Tensor,
        warm_flag: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_calls += 1
        return super().forward(image, warm_flag)


class _FailOnSecondDetector(_DummyDetector):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(
        self,
        image: torch.Tensor,
        warm_flag: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_calls += 1
        if self.forward_calls == 2:
            raise RuntimeError("synthetic second-record inference failure")
        return super().forward(image, warm_flag)


@pytest.mark.parametrize(
    "mutation",
    [
        "selection_guard",
        "run_guard",
        "checkpoint_guard",
        "runtime_guard",
        "materialization_hash",
        "binding_symlink",
        "binding_hash",
        "checkpoint_hash",
    ],
)
def test_exporter_completes_all_input_qa_before_model_construction(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path, record_count=1)
    checkpoint_sha: str | None = None
    if mutation == "selection_guard":
        selection = deepcopy(fixture["selection"])
        selection["official_test_accessed"] = True
        _write_json(fixture["selection_path"], selection)
    elif mutation == "run_guard":
        run = deepcopy(fixture["run"])
        run["official_test_accessed"] = True
        _write_json(fixture["run_path"], run)
    elif mutation == "checkpoint_guard":
        checkpoint = torch.load(
            fixture["checkpoint_path"],
            map_location="cpu",
            weights_only=True,
        )
        checkpoint["official_test_accessed"] = True
        torch.save(checkpoint, fixture["checkpoint_path"])
    elif mutation == "runtime_guard":
        checkpoint = torch.load(
            fixture["checkpoint_path"],
            map_location="cpu",
            weights_only=True,
        )
        runtime_name = checkpoint["stage2_runtime_artifacts"]["runtime_contract"][
            "path"
        ]
        runtime_path = fixture["checkpoint_path"].parent / runtime_name
        runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime_payload["official_test_accessed"] = True
        _write_json(runtime_path, runtime_payload)
        checkpoint["stage2_runtime_artifacts"]["runtime_contract"]["sha256"] = _sha(
            runtime_path
        )
        torch.save(checkpoint, fixture["checkpoint_path"])
    elif mutation == "materialization_hash":
        run = deepcopy(fixture["run"])
        detector_path = run["bindings"]["detector_config"]["path"]
        run["bindings"]["materialization_artifacts_sha256"] = {
            detector_path: "0" * 64
        }
        _write_json(fixture["run_path"], run)
    elif mutation == "binding_symlink":
        run = deepcopy(fixture["run"])
        detector_path = tmp_path / run["bindings"]["detector_config"]["path"]
        link = detector_path.with_name("detector_config_link.json")
        link.symlink_to(detector_path.name)
        run["bindings"]["detector_config"] = {
            "path": _rel(link, tmp_path),
            "sha256": _sha(detector_path),
        }
        _write_json(fixture["run_path"], run)
    elif mutation == "binding_hash":
        run = deepcopy(fixture["run"])
        run["bindings"]["detector_config"]["sha256"] = "0" * 64
        _write_json(fixture["run_path"], run)
    else:
        checkpoint_sha = "0" * 64

    output = tmp_path / "qa-export"

    def forbidden_factory(*args: object, **kwargs: object) -> torch.nn.Module:
        raise AssertionError("model construction occurred before input QA completed")

    with pytest.raises(
        (FileNotFoundError, FileExistsError, OSError, RuntimeError, TypeError, ValueError)
    ):
        _export_fixture(
            fixture,
            output,
            model_factory=forbidden_factory,
            checkpoint_sha256=checkpoint_sha,
        )
    _assert_no_export_residue(output)


@pytest.mark.parametrize("occupation", ["file", "directory", "symlink"])
def test_exporter_rejects_preoccupied_bundle_before_model_construction(
    tmp_path: Path,
    occupation: str,
) -> None:
    fixture = _fixture(tmp_path, record_count=1)
    output = tmp_path / "preoccupied-export"
    if occupation == "file":
        output.write_text("preexisting\n", encoding="utf-8")
    elif occupation == "directory":
        output.mkdir()
        (output / "manifest.json.sha256").write_text(
            "preexisting\n",
            encoding="utf-8",
        )
    else:
        target = tmp_path / "preexisting-target"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)

    def forbidden_factory(*args: object, **kwargs: object) -> torch.nn.Module:
        raise AssertionError("model construction occurred for preoccupied output")

    with pytest.raises((FileExistsError, ValueError)):
        _export_fixture(
            fixture,
            output,
            model_factory=forbidden_factory,
        )
    assert os.path.lexists(output)
    assert not os.path.lexists(
        output.parent / f".{output.name}.stage2-export.lock"
    )
    assert not tuple(output.parent.glob(f".{output.name}.stage2-staging-*"))


def test_exporter_rechecks_inputs_after_model_factory_before_forward(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, record_count=1)
    output = tmp_path / "factory-recheck-export"
    detector = _CountingDetector()
    selected = fixture["selection"]["records"][0]
    image_path = tmp_path / selected["original_image_path"]

    def mutating_factory(
        checkpoint: Mapping[str, Any],
        device: torch.device,
    ) -> torch.nn.Module:
        image_path.write_bytes(image_path.read_bytes() + b"changed-after-preflight")
        return detector

    with pytest.raises(RuntimeError, match=r"bound file (bytes|identity) changed"):
        _export_fixture(
            fixture,
            output,
            model_factory=mutating_factory,
        )
    assert detector.forward_calls == 0
    _assert_no_export_residue(output)


def test_exporter_removes_staged_scores_after_mid_export_failure(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, record_count=2)
    output = tmp_path / "mid-export-failure"
    detector = _FailOnSecondDetector()
    with pytest.raises(RuntimeError, match="second-record"):
        _export_fixture(
            fixture,
            output,
            model_factory=lambda checkpoint, device: detector,
        )
    assert detector.forward_calls == 2
    _assert_no_export_residue(output)


@pytest.mark.parametrize("failure_stage", ["npz", "json", "sidecar"])
def test_exporter_bundle_write_error_leaves_no_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    fixture = _fixture(tmp_path, record_count=1)
    output = tmp_path / f"{failure_stage}-bundle-failure"
    if failure_stage == "npz":
        def fail_npz(path: Path, **arrays: np.ndarray) -> None:
            path.write_bytes(b"partial-npz")
            raise OSError("synthetic NPZ write failure")

        monkeypatch.setattr(stage2_exporter, "_write_npz_atomic", fail_npz)
    elif failure_stage == "json":
        def fail_json(path: Path, payload: Mapping[str, Any]) -> None:
            path.write_text("{", encoding="utf-8")
            raise OSError("synthetic JSON write failure")

        monkeypatch.setattr(stage2_exporter, "_write_json_atomic", fail_json)
    else:
        original_write_text = stage2_exporter._write_text_atomic

        def fail_sidecar(path: Path, text: str) -> None:
            if path.name == "manifest.json.sha256":
                path.write_text("partial-sidecar", encoding="utf-8")
                raise OSError("synthetic sidecar write failure")
            original_write_text(path, text)

        monkeypatch.setattr(stage2_exporter, "_write_text_atomic", fail_sidecar)

    with pytest.raises(OSError, match="synthetic"):
        _export_fixture(
            fixture,
            output,
            model_factory=lambda checkpoint, device: _DummyDetector(),
        )
    _assert_no_export_residue(output)


def test_atomic_publish_refuses_late_target_occupation_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, record_count=1)
    output = tmp_path / "late-occupation-export"
    original_publish = stage2_exporter._atomic_publish_directory_no_replace

    def occupy_then_publish(
        staging_root: Path,
        final_root: Path,
    ) -> tuple[int, int]:
        final_root.mkdir()
        (final_root / "preexisting.txt").write_text(
            "preserve\n",
            encoding="utf-8",
        )
        return original_publish(staging_root, final_root)

    monkeypatch.setattr(
        stage2_exporter,
        "_atomic_publish_directory_no_replace",
        occupy_then_publish,
    )
    with pytest.raises(FileExistsError, match="occupied"):
        _export_fixture(
            fixture,
            output,
            model_factory=lambda checkpoint, device: _DummyDetector(),
        )
    assert {path.name for path in output.iterdir()} == {"preexisting.txt"}
    assert not os.path.lexists(
        output.parent / f".{output.name}.stage2-export.lock"
    )
    assert not tuple(output.parent.glob(f".{output.name}.stage2-staging-*"))


def test_exporter_is_split_independent_and_sentinel_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    role = FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE
    selected = [_selected_record(root, 0, "NUDT-SIRST")]
    selection = _selection_payload(
        root,
        role,
        selected,
        outer_fold="outer_leave_nuaa_sirst",
        outer_target="NUAA-SIRST",
        source_domain="NUDT-SIRST",
        detector_role="detector_full_fit",
        oof_fold_index=None,
    )
    selection_path = root / "contracts" / "selection.json"
    other_path = root / "contracts" / "other.json"
    _write_json(selection_path, selection)
    _write_json(other_path, {"other": True})
    bound: dict[str, dict[str, str]] = {}
    for name in ("detector_config", "seed_manifest", "materialization_index", "release_artifact"):
        path = root / "bindings" / f"{name}.json"
        _write_json(path, {"name": name})
        bound[name] = {"path": _rel(path, root), "sha256": _sha(path)}
    run = {
        "development_only": True,
        "official_test_accessed": False,
        "outer_fold_id": "outer_leave_nuaa_sirst",
        "outer_target_domain": "NUAA-SIRST",
        "source_domains": ["NUDT-SIRST", "IRSTD-1K"],
        "base_seed": 42,
        "derived_seed": 101,
        "detector_role": "detector_full_fit",
        "oof_fold_index": None,
        "selection_contracts": [
            {"path": _rel(selection_path, root), "sha256": _sha(selection_path)},
            {"path": _rel(other_path, root), "sha256": _sha(other_path)},
        ],
        "bindings": {**bound, "materialization_artifacts_sha256": {}},
    }
    run_path = root / "contracts" / "run.json"
    _write_json(run_path, run)
    runtime_config = root / "run" / "config.json"
    _write_json(runtime_config, {"base_size": 16, "resize_mode": "resize"})
    environment = root / "run" / "environment.json"
    _write_json(environment, {"synthetic": True})
    runtime_contract = root / "run" / "stage2_runtime_contract.json"
    _write_json(
        runtime_contract,
        {
            "schema_version": "rc-irstd.stage2-detector-runtime-contract.v1",
            "development_only": True,
            "official_test_accessed": False,
            "observed_results": None,
            "outer_fold_id": "outer_leave_nuaa_sirst",
            "outer_target_domain": "NUAA-SIRST",
            "detector_role": "detector_full_fit",
            "oof_fold_index": None,
            "base_seed": 42,
            "derived_seed": 101,
            "input_run_contract": {
                "path": _rel(run_path, root),
                "sha256": _sha(run_path),
            },
            "run_config": {
                "path": runtime_config.name,
                "sha256": _sha(runtime_config),
            },
            "environment_artifact": {
                "path": environment.name,
                "sha256": _sha(environment),
            },
        },
    )
    checkpoint = {
        "format_version": "rc-irstd.detector-inference.v1",
        "state_dict": {"synthetic.weight": torch.zeros(1)},
        "epoch": 0,
        "run_contract_sha256": _sha(run_path),
        "run_config_sha256": _sha(runtime_config),
        "source_names": ["NUDT-SIRST", "IRSTD-1K"],
        "outer_fold_id": "outer_leave_nuaa_sirst",
        "outer_target": "NUAA-SIRST",
        "held_out_domains": ["NUAA-SIRST"],
        "seed": 101,
        "detector_role": "detector_full_fit",
        "oof_fold_index": None,
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        "training_args": {"base_size": 16, "resize_mode": "resize"},
        "inference_geometry": {"input_hw": [16, 16], "resize_mode": "resize"},
        "official_test_accessed": False,
        "stage2_runtime_artifacts": {
            "run_config": {
                "path": runtime_config.name,
                "sha256": _sha(runtime_config),
            },
            "environment_artifact": {
                "path": environment.name,
                "sha256": _sha(environment),
            },
            "runtime_contract": {
                "path": runtime_contract.name,
                "sha256": _sha(runtime_contract),
            },
        },
    }
    checkpoint_path = root / "run" / "stage2_inference_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    import evaluation.export_score_maps as legacy_exporter

    def forbidden_call(*args: object, **kwargs: object) -> object:
        raise AssertionError("legacy split discovery must not be called")

    monkeypatch.setattr(legacy_exporter, "build_official_split_contract", forbidden_call)
    original_path_open = Path.open

    def sentinel_open(self: Path, *args: object, **kwargs: object):
        if "official_test_poison" in self.as_posix():
            raise AssertionError("official-test sentinel was opened")
        return original_path_open(self, *args, **kwargs)

    (root / "official_test_poison.txt").write_text("must remain sealed\n")
    monkeypatch.setattr(Path, "open", sentinel_open)
    output = root / "scores"
    manifest = export_stage2_development_scores(
        selection_path,
        run_path,
        checkpoint_path,
        output,
        selection_contract_sha256=_sha(selection_path),
        run_contract_sha256=_sha(run_path),
        checkpoint_sha256=_sha(checkpoint_path),
        role=role,
        device="cpu",
        repository_root=root,
        model_factory=lambda checkpoint_payload, device: _DummyDetector(),
    )
    assert manifest["official_test_accessed"] is False
    assert manifest["num_images"] == 1
    assert not (output / ".export_incomplete").exists()
    assert {path.name for path in output.iterdir()} == {
        Path(manifest["records"][0]["score_file"]).name,
        "manifest.json",
        "manifest.json.sha256",
    }
    assert (output / "manifest.json.sha256").read_text(encoding="utf-8") == (
        f"{_sha(output / 'manifest.json')}  manifest.json\n"
    )
    assert not os.path.lexists(
        output.parent / f".{output.name}.stage2-export.lock"
    )
    assert not tuple(output.parent.glob(f".{output.name}.stage2-staging-*"))
    verify_stage2_score_manifest(
        output / "manifest.json",
        _sha(output / "manifest.json"),
        role,
        repository_root=root,
    )


def test_exporter_has_no_legacy_split_discovery_reference() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "export_stage2_development_scores.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = "build_" + "official_split_contract"
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert forbidden not in names | attributes
    assert "evaluation.export_score_maps" not in source

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Mapping
import tempfile

import numpy as np
from PIL import Image
import pytest
import torch

from data_ext import stage2_score_manifest as score_v4
from data_ext import stage2_rc5_atomic_decision_set as atomic
from data_ext import stage2_rc5plus_atomic_full_decision_set as atomic_full
from data_ext.stage2_rc5plus_atomic_learned_decision_set import (
    guarded_invoke_stage2_rc5plus_label_resolver,
    publish_stage2_rc5plus_atomic_learned_decision_set,
)
from data_ext.stage2_rc5_score_bundle_v2 import (
    VerifiedStage2RC5ScoreBundleV2,
    publish_stage2_rc5_score_attestation_v2,
)
from data_ext.stage2_score_manifest_metadata_v5 import (
    verify_stage2_score_manifest_metadata_v5,
)
from data_ext.stage2_variable_query_window import (
    BOUND_INPUT_NAMES,
    VerifiedStage2VariableQueryWindow,
    build_stage2_variable_query_window_payload,
    verify_stage2_variable_query_window,
)
from model.endpoint_aware_pixel_calibrator import (
    DirectEndpointAwarePixelCalibrator,
    MonotoneEndpointAwarePixelCalibrator,
)
from model.budget_conditioned_residual_transport_calibrator import (
    BudgetConditionedDirectResidualTransportCalibrator,
    BudgetConditionedMonotoneNoTargetAnchorCalibrator,
    BudgetConditionedMonotoneResidualTransportCalibrator,
)
import rc.build_stage2_rc5_context as context_producer
import rc.build_stage2_source_reference as source_reference_builder
import rc.stage2_rc5_cyclic_context as cyclic_context
import rc.stage2_rc5_source_reference_v3 as source_v3
from rc.build_stage2_rc5_context import (
    ANCHOR_FILENAME,
    COMMIT_FILENAME,
    CONTEXT_FILENAME,
    PRODUCER_MANIFEST_FILENAME,
    Stage2RC5ContextProducerError,
    VerifiedStage2RC5ContextBundle,
    build_and_publish_stage2_rc5_context_bundle,
    replay_verified_stage2_rc5_context_bundle,
    verify_stage2_rc5_context_bundle,
)
from rc.build_stage2_source_reference import (
    VerifiedStage2SourceReference,
    verify_stage2_source_reference,
)
from rc.domain_statistics import BASE_FEATURE_DIM
from rc.schema import StatisticsConfig
from rc.stage2_calibrator_checkpoint_v7 import (
    make_calibrator_checkpoint_v7,
    serialize_calibrator_checkpoint_v7,
    verify_calibrator_checkpoint_v7_bytes,
)
from rc.stage2_calibrator_checkpoint_v8 import (
    make_calibrator_checkpoint_v8,
    serialize_calibrator_checkpoint_v8,
    verify_calibrator_checkpoint_v8_bytes,
)
from rc.stage2_rc5_feature_mask import build_stage2_rc5_feature_mask
from rc.stage2_rc5plus_context_anchor_v2 import (
    build_context_tail_anchor_v2_from_producer_bundle,
)
from rc.stage2_rc5plus_infer_and_seal import (
    infer_and_seal_stage2_rc5plus,
    verify_stage2_rc5plus_inference_seal,
)
from rc.stage2_rc5plus_no_anchor_infer_and_seal import (
    infer_and_seal_stage2_rc5plus_no_anchor,
    verify_stage2_rc5plus_no_anchor_inference_seal,
)
from rc.stage2_rc5_infer_and_seal import (
    TRANSCRIPT_SCHEMA,
    infer_and_seal_stage2_rc5,
    verify_stage2_rc5_inference_seal,
)
from rc.stage2_rc5_source_reference_v3 import (
    ATTESTATION_FILENAME as SOURCE_V3_ATTESTATION_FILENAME,
    COMMIT_FILENAME as SOURCE_V3_COMMIT_FILENAME,
    Stage2RC5SourceReferenceV3Error,
    VerifiedStage2RC5SourceReferenceV3,
    assert_verified_stage2_rc5_source_reference_v3,
    publish_stage2_rc5_source_reference_v3,
    replay_verified_stage2_rc5_source_reference_v3,
)
from rc.stage2_source_reference_variable_query_v2 import (
    VerifiedStage2SourceReferenceVariableQueryV2,
    verify_stage2_source_reference_variable_query_v2,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_CONTRACT = ROOT / (
    "outputs/stage2_detector_contracts/rc4_k2_c14q28_w01_20260716/runs/"
    "outer_leave_nuaa_sirst__s42__oof_fold_0.json"
)
TRAIN_SELECTIONS = {
    "NUDT-SIRST": ROOT
    / (
        "outputs/stage2_detector_contracts/rc4_k2_c14q28_w01_20260716/"
        "selections/outer_leave_nuaa_sirst__s42__oof_fold_0/"
        "nudt_sirst.selection.json"
    ),
    "IRSTD-1K": ROOT
    / (
        "outputs/stage2_detector_contracts/rc4_k2_c14q28_w01_20260716/"
        "selections/outer_leave_nuaa_sirst__s42__oof_fold_0/"
        "irstd_1k.selection.json"
    ),
}
ASSIGNMENTS = {
    "NUDT-SIRST": ROOT
    / (
        "outputs/stage2_manifests/rc4_k2_c14q28_20260716/assignments/"
        "nudt-sirst.detector_fit.k2.assignment.json"
    ),
    "IRSTD-1K": ROOT
    / (
        "outputs/stage2_manifests/rc4_k2_c14q28_20260716/assignments/"
        "irstd-1k.detector_fit.k2.assignment.json"
    ),
}

_RUN_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_rc5_context_e2e_run_complete_helper",
    ROOT / "tests/test_stage2_detector_run_complete_v2.py",
)
assert _RUN_HELPER_SPEC is not None and _RUN_HELPER_SPEC.loader is not None
_RUN_HELPERS = importlib.util.module_from_spec(_RUN_HELPER_SPEC)
_RUN_HELPER_SPEC.loader.exec_module(_RUN_HELPERS)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return _sha(path)


def _identity_records(
    base: Path,
    *,
    domain: str,
    tag: str,
    count: int = 43,
    materialize_context: bool = False,
    deferred_members: dict[Path, bytes] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(count):
        image_id = f"{tag}_{index:03d}"
        image_path = base / "images" / domain.lower() / f"{image_id}.png"
        if materialize_context:
            if deferred_members is None:
                raise RuntimeError("deferred_members is required for query records")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            pixels = (
                np.arange(72, dtype=np.uint16).reshape(8, 9)
                + 7 * index
            ) % 256
            Image.fromarray(pixels.astype(np.uint8), mode="L").save(image_path)
            image_sha = _sha(image_path)
            if index >= 14:
                deferred_members[image_path] = image_path.read_bytes()
                image_path.unlink()
        else:
            image_sha = _digest(f"intentionally-absent-image:{domain}:{image_id}")
        records.append(
            {
                "canonical_id": f"{domain}::{image_id}",
                "image_id": image_id,
                "original_image_path": _relative(image_path),
                "original_image_sha256": image_sha,
                "exclusion_group_id": (
                    f"SINGLETON::{domain}::{image_id}::{image_sha}"
                ),
                "near_duplicate_cluster_id_or_unique_sentinel": (
                    f"UNIQUE_SYNTHETIC::{domain}::{image_id}"
                ),
                "source_role_record_index": index,
            }
        )
    return records


def _write_context_score_npz(
    path: Path,
    record: Mapping[str, Any],
    *,
    source_domain: str,
    index: int,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.linspace(0.01, 0.97, 72, dtype=np.float64).reshape(8, 9)
    probability = np.clip(base + index * 1e-4, 1e-8, 1.0 - 1e-8)
    raw_logit = np.log(probability / (1.0 - probability))
    values = {
        "prob": probability,
        "raw_logit": raw_logit,
        "canonical_id": np.asarray(record["canonical_id"]),
        "image_id": np.asarray(record["image_id"]),
        "source_domain": np.asarray(source_domain),
        "original_hw": np.asarray([8, 9], dtype=np.int64),
        "input_hw": np.asarray([256, 256], dtype=np.int64),
        "resized_hw": np.asarray([256, 256], dtype=np.int64),
        "padding_ltrb": np.asarray([0, 0, 0, 0], dtype=np.int64),
        "resize_mode": np.asarray("resize"),
    }
    assert tuple(values) == score_v4.NPZ_FIELD_ORDER
    np.savez_compressed(path, **values)
    return _sha(path)


def _score_bindings(
    run_fixture: Mapping[str, Any], selection_path: Path
) -> dict[str, dict[str, str]]:
    run = run_fixture["run"]
    runtime = run_fixture["runtime"]
    run_dir = run_fixture["run_dir"]
    return {
        "selection_contract": {
            "path": _relative(selection_path),
            "sha256": _sha(selection_path),
        },
        "run_contract": {
            "path": _relative(RUN_CONTRACT),
            "sha256": run_fixture["run_sha"],
        },
        "checkpoint": {
            "path": _relative(run_dir / "stage2_inference_checkpoint.pt"),
            "sha256": run_fixture["hashes"][
                "restricted_inference_checkpoint_sha256"
            ],
        },
        "detector_config": dict(run["bindings"]["detector_config"]),
        "runtime_config": {
            "path": _relative(run_dir / runtime["run_config"]["path"]),
            "sha256": runtime["run_config"]["sha256"],
        },
        "seed_manifest": dict(run["bindings"]["seed_manifest"]),
        "materialization_index": dict(run["bindings"]["materialization_index"]),
        "release_artifact": {
            "path": runtime["release_artifact"]["path"],
            "sha256": runtime["release_artifact"]["sha256"],
        },
        "environment_artifact": {
            "path": _relative(
                run_dir / runtime["environment_artifact"]["path"]
            ),
            "sha256": runtime["environment_artifact"]["sha256"],
        },
        "runtime_contract": {
            "path": _relative(run_dir / runtime["runtime_contract"]["path"]),
            "sha256": runtime["runtime_contract"]["sha256"],
        },
    }


def _publish_score_bundle(
    base: Path,
    *,
    name: str,
    role: str,
    source_domain: str,
    selection_path: Path,
    selected_records: list[dict[str, Any]],
    run_fixture: Mapping[str, Any],
    run_complete: Any,
    materialize_context: bool,
    deferred_members: dict[Path, bytes] | None = None,
) -> VerifiedStage2RC5ScoreBundleV2:
    export_directory = base / "score_exports" / name / "nested"
    export_directory.mkdir(parents=True)
    score_records: list[dict[str, Any]] = []
    for index, selected in enumerate(selected_records):
        score_path = export_directory / "members" / f"{index:06d}.npz"
        if materialize_context:
            if deferred_members is None:
                raise RuntimeError("deferred_members is required for query scores")
            score_sha = _write_context_score_npz(
                score_path,
                selected,
                source_domain=source_domain,
                index=index,
            )
            if index >= 14:
                deferred_members[score_path] = score_path.read_bytes()
                score_path.unlink()
        else:
            score_sha = _digest(
                f"intentionally-absent-score:{name}:{source_domain}:{index}"
            )
        score_records.append(
            {
                "record_index": index,
                **selected,
                "source_domain": source_domain,
                "score_file": _relative(score_path),
                "score_file_sha256": score_sha,
                "original_hw": [8, 9],
                "input_hw": [256, 256],
                "resized_hw": [256, 256],
                "padding_ltrb": [0, 0, 0, 0],
                "resize_mode": "resize",
            }
        )
    run = run_fixture["run"]
    manifest = {
        "schema_version": score_v4.STAGE2_SCORE_MANIFEST_SCHEMA,
        "artifact_type": score_v4.STAGE2_SCORE_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY",
        "development_only": True,
        "execution_scope": "stage2_development",
        "official_test_accessed": False,
        "labels_embedded": False,
        "native_resolution": True,
        "restored_to_original_hw": True,
        "path_anchor": "repository_root",
        "role": role,
        "threshold_semantics": score_v4.STRICT_THRESHOLD_SEMANTICS,
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
        "outer_fold_id": run["outer_fold_id"],
        "outer_target": run["outer_target_domain"],
        "source_domain": source_domain,
        "base_seed": run["base_seed"],
        "derived_seed": run["derived_seed"],
        "detector_role": run["detector_role"],
        "oof_fold_index": run["oof_fold_index"],
        "input_hw": [256, 256],
        "resize_mode": "resize",
        "bindings": _score_bindings(run_fixture, selection_path),
        "num_images": len(score_records),
        "records_content_sha256_algorithm": (
            score_v4.STAGE2_SCORE_RECORDS_ALGORITHM
        ),
        "records_content_sha256": score_v4.stage2_score_records_sha256(
            score_records
        ),
        "records": score_records,
    }
    manifest_path = export_directory / "manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    metadata = verify_stage2_score_manifest_metadata_v5(
        manifest_path,
        manifest_sha,
        role,
        repository_root=ROOT,
    )
    return publish_stage2_rc5_score_attestation_v2(metadata, run_complete)


def _window_records(
    records: list[dict[str, Any]], *, run: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "source_role": "detector_fit",
            "outer_fold_id": run["outer_fold_id"],
            "episode_role": "stage2_oof_fit",
            "oof_fold_index": run["oof_fold_index"],
        }
        for record in records
    ]


def _build_window(
    base: Path,
    *,
    domain: str,
    records: list[dict[str, Any]],
    run: Mapping[str, Any],
) -> VerifiedStage2VariableQueryWindow:
    assignment_path = ASSIGNMENTS[domain]
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    bound_inputs = {
        name: dict(assignment["bound_inputs"][name])
        for name in BOUND_INPUT_NAMES
    }
    payload = build_stage2_variable_query_window_payload(
        ordered_role_records=_window_records(records, run=run),
        outer_fold_id=run["outer_fold_id"],
        outer_target_domain=run["outer_target_domain"],
        domain=domain,
        source_role="detector_fit",
        episode_role="stage2_oof_fit",
        oof_fold_index=run["oof_fold_index"],
        role_binding={
            "path": _relative(assignment_path),
            "sha256": _sha(assignment_path),
        },
        bound_inputs=bound_inputs,
    )
    path = base / "variable_query_windows" / domain.lower() / "window.json"
    digest = _write_json(path, payload)
    return verify_stage2_variable_query_window(
        path, digest, repository_root=ROOT
    )


def _build_base_source_reference(
    base: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    source_bundles: tuple[
        VerifiedStage2RC5ScoreBundleV2, VerifiedStage2RC5ScoreBundleV2
    ],
    windows: tuple[
        VerifiedStage2VariableQueryWindow, VerifiedStage2VariableQueryWindow
    ],
    run_fixture: Mapping[str, Any],
    statistics_config: StatisticsConfig,
    statistics_path: Path,
    statistics_sha: str,
) -> VerifiedStage2SourceReference:
    manifests = tuple(bundle.score_manifest_metadata for bundle in source_bundles)
    run = run_fixture["run"]
    checkpoint = run_fixture["run_dir"] / "stage2_inference_checkpoint.pt"
    checkpoint_sha = run_fixture["hashes"][
        "restricted_inference_checkpoint_sha256"
    ]

    consumers = tuple(
        {
            "path": _relative(window.path),
            "sha256": window.manifest_sha256,
            "domain": window.payload["domain"],
            "episode_role": window.payload["episode_role"],
            "complete_window_count": window.payload["complete_window_count"],
            "record_count": window.payload["window_record_count"],
            "records": window.ordered_records,
        }
        for window in windows
    )

    def fake_reference_manifests(*args: Any, **kwargs: Any):
        del args, kwargs
        rechecks = [
            (manifest.path, manifest.manifest_sha256) for manifest in manifests
        ]
        rechecks.append((RUN_CONTRACT, run_fixture["run_sha"]))
        return manifests, run, score_v4.OOF_TRAIN_SOURCE_REFERENCE, rechecks

    def fake_consumers(*args: Any, **kwargs: Any):
        del args, kwargs
        return consumers

    centers = np.vstack(
        [
            np.linspace(0.0, 1.0, BASE_FEATURE_DIM, dtype=np.float32),
            np.linspace(1.0, 2.0, BASE_FEATURE_DIM, dtype=np.float32),
        ]
    )
    scale = np.linspace(0.25, 1.25, BASE_FEATURE_DIM, dtype=np.float32)

    def fake_centers(*args: Any, **kwargs: Any):
        del args, kwargs
        summaries = {
            domain: {
                "record_count": len(manifests[index].records),
                "num_images": len(manifests[index].records),
                "num_pixels": 72 * len(manifests[index].records),
                "num_peaks": len(manifests[index].records),
                "has_grayscale": True,
            }
            for index, domain in enumerate(run["source_domains"])
        }
        return centers, scale, summaries

    monkeypatch.setattr(
        source_reference_builder,
        "_verify_reference_manifests",
        fake_reference_manifests,
    )
    monkeypatch.setattr(
        source_reference_builder, "_verify_all_consumers", fake_consumers
    )
    monkeypatch.setattr(
        source_reference_builder, "_compute_centers", fake_centers
    )

    output = (
        base
        / "source_reference"
        / "at"
        / "a"
        / "different"
        / "root_depth"
        / "source-reference.npz"
    )
    output.parent.mkdir(parents=True)
    source_reference_builder.build_stage2_source_reference(
        [manifest.path for manifest in manifests],
        [manifest.manifest_sha256 for manifest in manifests],
        checkpoint,
        checkpoint_sha,
        statistics_path,
        statistics_sha,
        [window.path for window in windows],
        [window.manifest_sha256 for window in windows],
        output,
        repository_root=ROOT,
    )
    audit_path = output.with_suffix(".audit.json")
    return verify_stage2_source_reference(
        output,
        _sha(output),
        _sha(audit_path),
        statistics_config=statistics_config,
        repository_root=ROOT,
    )


def _checkpoint(method: str) -> Any:
    if method not in {"T6", "T7", "T8"}:
        raise ValueError(f"unsupported synthetic calibrator method: {method}")
    torch.manual_seed(1707)
    common = {
        "context_feature_dim": 93,
        "pixel_budget_grid": [1e-4, 1e-5, 1e-6],
        "hidden_dims": [32],
        "dropout": 0.1,
    }
    if method == "T6":
        model = DirectEndpointAwarePixelCalibrator(**common)
    else:
        model = MonotoneEndpointAwarePixelCalibrator(
            **common,
            minimum_raw_coordinate_gap=0.001,
        )
    training_sha = _digest(
        f"synthetic-context-producer-e2e-training-contract:{method}"
    )
    payload = make_calibrator_checkpoint_v7(
        method=method,
        model=model,
        standardizer_mean=np.linspace(-0.5, 0.5, 93, dtype=np.float64),
        standardizer_scale=np.linspace(0.5, 2.0, 93, dtype=np.float64),
        training_contract_sha256=training_sha,
    )
    data = serialize_calibrator_checkpoint_v7(payload)
    return verify_calibrator_checkpoint_v7_bytes(
        data,
        hashlib.sha256(data).hexdigest(),
        expected_method=method,
        expected_training_contract_sha256=training_sha,
    )


def _checkpoint_t8() -> Any:
    return _checkpoint("T8")


def _checkpoint_v8(method: str) -> Any:
    if method == "T6_PLUS":
        model = BudgetConditionedDirectResidualTransportCalibrator(93)
    elif method == "T8_PLUS_NO_ANCHOR":
        model = BudgetConditionedMonotoneNoTargetAnchorCalibrator(93)
    elif method in {"T7_PLUS", "T8_PLUS"}:
        model = BudgetConditionedMonotoneResidualTransportCalibrator(93)
    else:
        raise ValueError(method)
    training_sha = _digest("rc5plus-production-capability-e2e-training")
    training_view_sha = _digest("rc5plus-production-capability-e2e-view")
    payload = make_calibrator_checkpoint_v8(
        method=method,
        model=model,
        standardizer_mean=np.linspace(-0.5, 0.5, 93, dtype=np.float64),
        standardizer_scale=np.linspace(0.5, 2.0, 93, dtype=np.float64),
        training_contract_sha256=training_sha,
        training_view_identity_sha256=training_view_sha,
        feature_mask=build_stage2_rc5_feature_mask("C3"),
    )
    data = serialize_calibrator_checkpoint_v8(payload)
    return verify_calibrator_checkpoint_v8_bytes(
        data,
        hashlib.sha256(data).hexdigest(),
        expected_method=method,
        expected_training_contract_sha256=training_sha,
        expected_training_view_identity_sha256=training_view_sha,
    )


class _E2EInputs(SimpleNamespace):
    base: Path
    run_fixture: Mapping[str, Any]
    run_complete: Any
    source_score_bundles: tuple[
        VerifiedStage2RC5ScoreBundleV2, VerifiedStage2RC5ScoreBundleV2
    ]
    query_score_bundle: VerifiedStage2RC5ScoreBundleV2
    windows: tuple[
        VerifiedStage2VariableQueryWindow, VerifiedStage2VariableQueryWindow
    ]
    source_reference_v2: VerifiedStage2SourceReferenceVariableQueryV2
    source_reference_v3: VerifiedStage2RC5SourceReferenceV3
    statistics_config: StatisticsConfig
    statistics_path: Path
    statistics_sha: str
    bundle: VerifiedStage2RC5ContextBundle
    member_open_counts: Counter[str]
    context_score_paths: tuple[Path, ...]
    context_image_paths: tuple[Path, ...]
    query_score_paths: tuple[Path, ...]
    query_image_paths: tuple[Path, ...]
    query_members_absent_at_context_build: bool


@pytest.fixture(scope="module")
def e2e_inputs() -> Any:
    temporary_root = ROOT / ".tmp"
    temporary_root.mkdir(exist_ok=True)
    base = Path(
        tempfile.mkdtemp(prefix="pytest-rc5-context-e2e-", dir=temporary_root)
    )
    monkeypatch = pytest.MonkeyPatch()
    try:
        run_fixture = _RUN_HELPERS._synthetic_run(base / "completion_inputs")
        run_complete = _RUN_HELPERS._publish(run_fixture)
        run = run_fixture["run"]

        deferred_query_members: dict[Path, bytes] = {}
        selected: dict[tuple[str, str], list[dict[str, Any]]] = {
            (score_v4.OOF_TRAIN_SOURCE_REFERENCE, "NUDT-SIRST"): (
                _identity_records(
                    base,
                    domain="NUDT-SIRST",
                    tag="source_train_nudt",
                )
            ),
            (score_v4.OOF_TRAIN_SOURCE_REFERENCE, "IRSTD-1K"): (
                _identity_records(
                    base,
                    domain="IRSTD-1K",
                    tag="source_train_irstd",
                )
            ),
            (score_v4.OOF_HOLDOUT_STAGE2_FIT, "NUDT-SIRST"): (
                _identity_records(
                    base,
                    domain="NUDT-SIRST",
                    tag="query_holdout_nudt",
                    materialize_context=True,
                    deferred_members=deferred_query_members,
                )
            ),
        }
        second_consumer_records = _identity_records(
            base,
            domain="IRSTD-1K",
            tag="query_holdout_irstd",
        )

        original_selection_records = score_v4._selection_records

        def synthetic_selection_records(
            selection: Mapping[str, Any], *, role: str, oof_fold_index: Any
        ) -> list[Mapping[str, Any]]:
            del oof_fold_index
            artifact_type = selection.get("artifact_type")
            if artifact_type == "rc_irstd_stage2_detector_selection":
                domain = str(selection.get("source_domain"))
            elif artifact_type == (
                "rc_irstd_stage2_detector_fit_group_assignment"
            ):
                domain = str(selection.get("domain"))
            else:
                return original_selection_records(
                    selection, role=role, oof_fold_index=run["oof_fold_index"]
                )
            key = (role, domain)
            if key not in selected:
                return original_selection_records(
                    selection, role=role, oof_fold_index=run["oof_fold_index"]
                )
            records = selected[key]
            if role == score_v4.OOF_HOLDOUT_STAGE2_FIT:
                return [
                    {
                        **record,
                        "source_role": "detector_fit",
                        "oof_fold_index": run["oof_fold_index"],
                    }
                    for record in records
                ]
            return [dict(record) for record in records]

        monkeypatch.setattr(
            score_v4, "_selection_records", synthetic_selection_records
        )

        source_bundles = (
            _publish_score_bundle(
                base,
                name="source_nudt",
                role=score_v4.OOF_TRAIN_SOURCE_REFERENCE,
                source_domain="NUDT-SIRST",
                selection_path=TRAIN_SELECTIONS["NUDT-SIRST"],
                selected_records=selected[
                    (score_v4.OOF_TRAIN_SOURCE_REFERENCE, "NUDT-SIRST")
                ],
                run_fixture=run_fixture,
                run_complete=run_complete,
                materialize_context=False,
            ),
            _publish_score_bundle(
                base,
                name="source_irstd",
                role=score_v4.OOF_TRAIN_SOURCE_REFERENCE,
                source_domain="IRSTD-1K",
                selection_path=TRAIN_SELECTIONS["IRSTD-1K"],
                selected_records=selected[
                    (score_v4.OOF_TRAIN_SOURCE_REFERENCE, "IRSTD-1K")
                ],
                run_fixture=run_fixture,
                run_complete=run_complete,
                materialize_context=False,
            ),
        )
        query_score_bundle = _publish_score_bundle(
            base,
            name="query_nudt",
            role=score_v4.OOF_HOLDOUT_STAGE2_FIT,
            source_domain="NUDT-SIRST",
            selection_path=ASSIGNMENTS["NUDT-SIRST"],
            selected_records=selected[
                (score_v4.OOF_HOLDOUT_STAGE2_FIT, "NUDT-SIRST")
            ],
            run_fixture=run_fixture,
            run_complete=run_complete,
            materialize_context=True,
            deferred_members=deferred_query_members,
        )

        windows = (
            _build_window(
                base,
                domain="NUDT-SIRST",
                records=selected[
                    (score_v4.OOF_HOLDOUT_STAGE2_FIT, "NUDT-SIRST")
                ],
                run=run,
            ),
            _build_window(
                base,
                domain="IRSTD-1K",
                records=second_consumer_records,
                run=run,
            ),
        )
        statistics_config = StatisticsConfig(
            peak_kernel_size=3,
            peak_min_score=0.05,
            quantile_sample_limit=128,
        )
        statistics_path = base / "configuration" / "statistics.json"
        statistics_sha = _write_json(
            statistics_path, statistics_config.to_dict()
        )
        base_source_reference = _build_base_source_reference(
            base,
            monkeypatch=monkeypatch,
            source_bundles=source_bundles,
            windows=windows,
            run_fixture=run_fixture,
            statistics_config=statistics_config,
            statistics_path=statistics_path,
            statistics_sha=statistics_sha,
        )
        # Keep this promotion isolated: the final test is migrated to the
        # persistent v3 source authority as soon as that additive API lands.
        source_reference_v2 = verify_stage2_source_reference_variable_query_v2(
            base_source_reference
        )
        source_v3_output = base / "source_reference" / "rc5_authority_v3"
        source_v3_output.mkdir(parents=True)
        source_reference_v3 = publish_stage2_rc5_source_reference_v3(
            source_reference=source_reference_v2,
            score_bundles=source_bundles,
            output_directory=source_v3_output,
            repository_root=ROOT,
        )

        metadata = query_score_bundle.score_manifest_metadata
        first_window = windows[0].windows[0]
        context_ids = {
            str(record["canonical_id"])
            for record in first_window["context_records"]
        }
        query_ids = {
            str(record["canonical_id"])
            for record in first_window["query_records"]
        }
        context_items = tuple(
            item for item in metadata.items if item.canonical_id in context_ids
        )
        query_items = tuple(
            item for item in metadata.items if item.canonical_id in query_ids
        )
        context_score_paths = tuple(item.score_path for item in context_items)
        context_image_paths = tuple(item.image_path for item in context_items)
        query_score_paths = tuple(item.score_path for item in query_items)
        query_image_paths = tuple(item.image_path for item in query_items)
        assert len(context_score_paths) == len(context_image_paths) == 14
        assert len(query_score_paths) == len(query_image_paths) == 29
        assert all(path.is_file() for path in context_score_paths)
        assert all(path.is_file() for path in context_image_paths)
        assert all(not path.exists() for path in query_score_paths)
        assert all(not path.exists() for path in query_image_paths)
        query_members_absent_at_context_build = True

        member_sets = {
            "context_score": set(context_score_paths),
            "context_image": set(context_image_paths),
            "query_score": set(query_score_paths),
            "query_image": set(query_image_paths),
        }
        counts: Counter[str] = Counter()
        original_stable_read = context_producer._stable_read_member
        deployment_context_guard_active = True

        def tracked_stable_read(
            path: Path, expected_sha256: str, root: Path, name: str
        ) -> bytes:
            for category, members in member_sets.items():
                if path in members:
                    counts[category] += 1
                    if (
                        deployment_context_guard_active
                        and category.startswith("query_")
                    ):
                        raise AssertionError(f"query member was opened: {path}")
            return original_stable_read(path, expected_sha256, root, name)

        monkeypatch.setattr(
            context_producer, "_stable_read_member", tracked_stable_read
        )
        output = base / "published_context" / "deep" / "bundle"
        output.mkdir(parents=True)
        bundle = build_and_publish_stage2_rc5_context_bundle(
            variable_query_window=windows[0],
            score_bundle=query_score_bundle,
            source_reference=source_reference_v3,
            statistics_config=statistics_config,
            statistics_config_path=statistics_path,
            statistics_config_sha256=statistics_sha,
            window_index=0,
            output_directory=output,
            repository_root=ROOT,
        )
        assert counts == Counter(
            {"context_score": 14, "context_image": 14}
        )
        deployment_context_guard_active = False
        # The primary producer build above proves physical query absence.  Only
        # after that sealed build do we materialize the exact pre-hashed bytes,
        # allowing downstream cyclic-authority tests to reuse this fixture.
        for path, data in deferred_query_members.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        assert all(path.is_file() for path in query_score_paths)
        assert all(path.is_file() for path in query_image_paths)

        yield _E2EInputs(
            base=base,
            run_fixture=run_fixture,
            run_complete=run_complete,
            source_score_bundles=source_bundles,
            query_score_bundle=query_score_bundle,
            windows=windows,
            source_reference_v2=source_reference_v2,
            source_reference_v3=source_reference_v3,
            statistics_config=statistics_config,
            statistics_path=statistics_path,
            statistics_sha=statistics_sha,
            bundle=bundle,
            member_open_counts=counts,
            context_score_paths=context_score_paths,
            context_image_paths=context_image_paths,
            query_score_paths=query_score_paths,
            query_image_paths=query_image_paths,
            query_members_absent_at_context_build=(
                query_members_absent_at_context_build
            ),
        )
    finally:
        monkeypatch.undo()
        shutil.rmtree(base, ignore_errors=True)


def _verify_bundle(inputs: _E2EInputs) -> VerifiedStage2RC5ContextBundle:
    return verify_stage2_rc5_context_bundle(
        inputs.bundle.commit_path,
        inputs.bundle.commit_sha256,
        variable_query_window=inputs.windows[0],
        score_bundle=inputs.query_score_bundle,
        source_reference=inputs.source_reference_v3,
        statistics_config=inputs.statistics_config,
        statistics_config_path=inputs.statistics_path,
        statistics_config_sha256=inputs.statistics_sha,
        repository_root=ROOT,
    )


def _build_kwargs(inputs: _E2EInputs, output: Path) -> dict[str, Any]:
    return {
        "variable_query_window": inputs.windows[0],
        "score_bundle": inputs.query_score_bundle,
        "source_reference": inputs.source_reference_v3,
        "statistics_config": inputs.statistics_config,
        "statistics_config_path": inputs.statistics_path,
        "statistics_config_sha256": inputs.statistics_sha,
        "window_index": 0,
        "output_directory": output,
        "repository_root": ROOT,
    }


def _clone_score_bundle_with_run_complete(
    value: VerifiedStage2RC5ScoreBundleV2, run_complete: Any
) -> VerifiedStage2RC5ScoreBundleV2:
    forged = object.__new__(VerifiedStage2RC5ScoreBundleV2)
    for name in VerifiedStage2RC5ScoreBundleV2.__slots__:
        object.__setattr__(forged, name, getattr(value, name))
    object.__setattr__(forged, "run_complete", run_complete)
    return forged


def _clone_v2_with_forged_centers_and_scale(
    value: VerifiedStage2SourceReferenceVariableQueryV2,
) -> VerifiedStage2SourceReferenceVariableQueryV2:
    base = value.source_reference_bundle
    centers = tuple(
        tuple(float(item) + 0.125 for item in row) for row in base.centers
    )
    scale = tuple(float(item) * 1.5 for item in base.scale)
    source_reference = replace(
        base.source_reference,
        centers=centers,
        scale=scale,
    )
    forged_base = object.__new__(VerifiedStage2SourceReference)
    for name in VerifiedStage2SourceReference.__dataclass_fields__:
        object.__setattr__(forged_base, name, getattr(base, name))
    object.__setattr__(forged_base, "centers", centers)
    object.__setattr__(forged_base, "scale", scale)
    object.__setattr__(forged_base, "source_reference", source_reference)

    forged = object.__new__(VerifiedStage2SourceReferenceVariableQueryV2)
    for name in VerifiedStage2SourceReferenceVariableQueryV2.__dataclass_fields__:
        object.__setattr__(forged, name, getattr(value, name))
    object.__setattr__(forged, "source_reference_bundle", forged_base)
    return forged


def _clone_context_bundle_with_forged_manifest(
    value: VerifiedStage2RC5ContextBundle,
) -> VerifiedStage2RC5ContextBundle:
    forged = object.__new__(VerifiedStage2RC5ContextBundle)
    for name in VerifiedStage2RC5ContextBundle.__dataclass_fields__:
        object.__setattr__(forged, name, getattr(value, name))
    manifest = json.loads(context_producer.canonical_json_bytes(
        value.producer_manifest
    ))
    manifest["producer_authority"] = "forged-retained-producer-authority"
    object.__setattr__(forged, "producer_manifest", manifest)
    return forged


def test_source_v3_is_public_authority_and_v2_or_bad_domain_coverage_is_rejected(
    e2e_inputs: _E2EInputs,
) -> None:
    inputs = e2e_inputs
    assert (
        assert_verified_stage2_rc5_source_reference_v3(
            inputs.source_reference_v3
        )
        is inputs.source_reference_v3
    )
    with pytest.raises(TypeError, match="SourceReferenceV3"):
        assert_verified_stage2_rc5_source_reference_v3(
            inputs.source_reference_v2
        )
    for index, score_bundles in enumerate(
        (
            (inputs.source_score_bundles[0],),
            (
                inputs.source_score_bundles[0],
                inputs.source_score_bundles[0],
            ),
        )
    ):
        output = inputs.base / f"bad_source_coverage_{index}"
        output.mkdir()
        with pytest.raises(Stage2RC5SourceReferenceV3Error):
            publish_stage2_rc5_source_reference_v3(
                source_reference=inputs.source_reference_v2,
                score_bundles=score_bundles,
                output_directory=output,
                repository_root=ROOT,
            )
        assert not (output / SOURCE_V3_COMMIT_FILENAME).exists()


def test_source_v3_rejects_forged_centers_and_scale(
    e2e_inputs: _E2EInputs,
) -> None:
    forged = _clone_v2_with_forged_centers_and_scale(
        e2e_inputs.source_reference_v2
    )
    output = e2e_inputs.base / "forged_source_statistics"
    output.mkdir()
    with pytest.raises(
        Stage2RC5SourceReferenceV3Error,
        match="capability.*differs|centers|scale|source-reference",
    ):
        publish_stage2_rc5_source_reference_v3(
            source_reference=forged,
            score_bundles=e2e_inputs.source_score_bundles,
            output_directory=output,
            repository_root=ROOT,
        )
    assert not (output / SOURCE_V3_COMMIT_FILENAME).exists()


def test_source_v3_rejects_source_manifest_toctou_and_different_run_complete(
    e2e_inputs: _E2EInputs,
) -> None:
    inputs = e2e_inputs
    source_manifest = inputs.source_score_bundles[0].score_manifest_metadata.path
    original = source_manifest.read_bytes()
    source_manifest.write_bytes(original + b"source-manifest-toctou")
    try:
        with pytest.raises((OSError, RuntimeError, TypeError, ValueError)):
            replay_verified_stage2_rc5_source_reference_v3(
                inputs.source_reference_v3
            )
    finally:
        source_manifest.write_bytes(original)
    assert replay_verified_stage2_rc5_source_reference_v3(
        inputs.source_reference_v3
    ).attestation_sha256 == inputs.source_reference_v3.attestation_sha256

    alternate_fixture = _RUN_HELPERS._synthetic_run(
        inputs.base / "alternate_completion_inputs"
    )
    alternate_complete = _RUN_HELPERS._publish(alternate_fixture)
    mismatched = _clone_score_bundle_with_run_complete(
        inputs.source_score_bundles[0], alternate_complete
    )
    output = inputs.base / "different_run_complete"
    output.mkdir()
    with pytest.raises((OSError, RuntimeError, TypeError, ValueError)):
        publish_stage2_rc5_source_reference_v3(
            source_reference=inputs.source_reference_v2,
            score_bundles=(mismatched, inputs.source_score_bundles[1]),
            output_directory=output,
            repository_root=ROOT,
        )
    assert not (output / SOURCE_V3_COMMIT_FILENAME).exists()


def test_source_v3_commit_last_fault_leaves_no_authoritative_marker(
    e2e_inputs: _E2EInputs,
) -> None:
    output = e2e_inputs.base / "source_v3_commit_fault"
    output.mkdir()
    original_write = source_v3._atomic_write_new
    calls = 0

    def fail_commit(path: Path, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic source-v3 commit fault")
        original_write(path, data)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(source_v3, "_atomic_write_new", fail_commit)
        with pytest.raises(OSError, match="source-v3 commit fault"):
            publish_stage2_rc5_source_reference_v3(
                source_reference=e2e_inputs.source_reference_v2,
                score_bundles=e2e_inputs.source_score_bundles,
                output_directory=output,
                repository_root=ROOT,
            )
    assert calls == 2
    assert (output / SOURCE_V3_ATTESTATION_FILENAME).is_file()
    assert not (output / SOURCE_V3_COMMIT_FILENAME).exists()
    with pytest.raises(FileNotFoundError):
        source_v3.verify_stage2_rc5_source_reference_v3(
            output / SOURCE_V3_ATTESTATION_FILENAME,
            _sha(output / SOURCE_V3_ATTESTATION_FILENAME),
            source_reference=e2e_inputs.source_reference_v2,
            score_bundles=e2e_inputs.source_score_bundles,
            repository_root=ROOT,
        )


def test_build_and_verify_open_exactly_context_members_only(
    e2e_inputs: _E2EInputs,
) -> None:
    inputs = e2e_inputs
    assert inputs.member_open_counts == Counter(
        {"context_score": 14, "context_image": 14}
    )
    assert inputs.query_members_absent_at_context_build is True
    assert all(path.is_file() for path in inputs.query_score_paths)
    assert all(path.is_file() for path in inputs.query_image_paths)
    audit = dict(inputs.bundle.producer_manifest["access_audit"])
    assert audit == {
        "context_score_member_open_count": 14,
        "context_image_member_open_count": 14,
        "query_score_member_open_count": 0,
        "query_image_member_open_count": 0,
        "context_labels_accessed": False,
        "query_labels_accessed": False,
        "observed_results_accessed": False,
        "only_context_member_bytes_promoted": True,
    }
    inputs_binding = inputs.bundle.producer_manifest["inputs"]
    source_binding = inputs_binding["source_reference"]
    assert source_binding["attestation"]["sha256"] == (
        inputs.source_reference_v3.attestation_sha256
    )
    assert len(source_binding["source_score_attestations"]) == 2
    assert {
        row["sha256"] for row in source_binding["source_score_attestations"]
    } == {
        bundle.attestation_sha256 for bundle in inputs.source_score_bundles
    }
    query_score_binding = inputs_binding["score_bundle"]
    assert source_binding["shared_run_complete"] == {
        "path": query_score_binding["run_complete_path"],
        "sha256": query_score_binding["run_complete_sha256"],
        "identity_sha256": query_score_binding[
            "run_complete_identity_sha256"
        ],
    }
    assert source_binding["restricted_checkpoint"]["sha256"] == (
        query_score_binding["restricted_checkpoint_sha256"]
    )
    before = inputs.member_open_counts.copy()
    replay = _verify_bundle(inputs)
    assert replay.bundle_identity_sha256 == inputs.bundle.bundle_identity_sha256
    assert inputs.member_open_counts - before == Counter(
        {"context_score": 14, "context_image": 14}
    )
    # Score export, source reference and context output deliberately occupy
    # different path depths under one public repository root.
    assert len(inputs.bundle.context_path.relative_to(ROOT).parts) != len(
        inputs.source_reference_v3.path.relative_to(ROOT).parts
    )
    assert len(inputs.source_reference_v3.path.relative_to(ROOT).parts) != len(
        inputs.query_score_bundle.attestation_path.relative_to(ROOT).parts
    )


def test_context_bundle_public_replay_rejects_retained_in_memory_forgery(
    e2e_inputs: _E2EInputs,
) -> None:
    replayed = replay_verified_stage2_rc5_context_bundle(e2e_inputs.bundle)
    assert replayed.commit_sha256 == e2e_inputs.bundle.commit_sha256
    assert replayed.bundle_identity_sha256 == (
        e2e_inputs.bundle.bundle_identity_sha256
    )
    forged = _clone_context_bundle_with_forged_manifest(e2e_inputs.bundle)
    with pytest.raises(
        Stage2RC5ContextProducerError,
        match="material differs|identity differs|upstream capability differs",
    ):
        replay_verified_stage2_rc5_context_bundle(forged)


def test_public_entry_rejects_bare_metadata_and_bare_context(
    e2e_inputs: _E2EInputs,
) -> None:
    inputs = e2e_inputs
    output = inputs.base / "bare_metadata_rejected"
    output.mkdir()
    kwargs = _build_kwargs(inputs, output)
    kwargs["score_bundle"] = inputs.query_score_bundle.score_manifest_metadata
    with pytest.raises(TypeError, match="VerifiedStage2RC5ScoreBundleV2"):
        build_and_publish_stage2_rc5_context_bundle(**kwargs)
    legacy_output = inputs.base / "legacy_source_v2_rejected"
    legacy_output.mkdir()
    legacy_kwargs = _build_kwargs(inputs, legacy_output)
    legacy_kwargs["source_reference"] = inputs.source_reference_v2
    with pytest.raises(TypeError, match="SourceReferenceV3"):
        build_and_publish_stage2_rc5_context_bundle(**legacy_kwargs)
    with pytest.raises(TypeError, match="SourceReferenceV3"):
        verify_stage2_rc5_context_bundle(
            inputs.bundle.commit_path,
            inputs.bundle.commit_sha256,
            variable_query_window=inputs.windows[0],
            score_bundle=inputs.query_score_bundle,
            source_reference=inputs.source_reference_v2,
            statistics_config=inputs.statistics_config,
            statistics_config_path=inputs.statistics_path,
            statistics_config_sha256=inputs.statistics_sha,
            repository_root=ROOT,
        )
    checkpoint = _checkpoint_t8()
    with pytest.raises(TypeError, match="context producer bundle"):
        infer_and_seal_stage2_rc5(
            checkpoint=checkpoint,
            producer_bundle=inputs.bundle.context,
        )


def test_public_infer_v4_binds_real_producer_bundle_in_seven_fields(
    e2e_inputs: _E2EInputs,
) -> None:
    checkpoint = _checkpoint_t8()
    transcript_bytes = infer_and_seal_stage2_rc5(
        checkpoint=checkpoint,
        producer_bundle=e2e_inputs.bundle,
    )
    transcript = json.loads(transcript_bytes.decode("utf-8"))
    assert transcript["schema_version"] == TRANSCRIPT_SCHEMA
    binding = transcript["producer_bundle_binding"]
    assert set(binding) == {
        "capability_schema",
        "producer_manifest_schema",
        "commit_schema",
        "producer_identity_sha256",
        "bundle_identity_sha256",
        "producer_manifest_sha256",
        "commit_sha256",
    }
    assert len(binding) == 7
    assert binding["capability_schema"] == e2e_inputs.bundle.capability_schema
    assert binding["producer_identity_sha256"] == (
        e2e_inputs.bundle.producer_manifest["producer_identity_sha256"]
    )
    assert binding["bundle_identity_sha256"] == (
        e2e_inputs.bundle.bundle_identity_sha256
    )
    assert binding["producer_manifest_sha256"] == (
        e2e_inputs.bundle.producer_manifest_sha256
    )
    assert binding["commit_sha256"] == e2e_inputs.bundle.commit_sha256
    verified = verify_stage2_rc5_inference_seal(
        transcript_bytes,
        checkpoint=checkpoint,
        producer_bundle=e2e_inputs.bundle,
    )
    assert verified.transcript_bytes == transcript_bytes
    assert verified.producer_identity_sha256 == binding[
        "producer_identity_sha256"
    ]
    assert verified.producer_bundle_identity_sha256 == binding[
        "bundle_identity_sha256"
    ]


def test_rc5plus_production_capabilities_replay_to_atomic_prelabel_set(
    e2e_inputs: _E2EInputs,
) -> None:
    requested = ((1, 25_000), (1, 250_000))
    anchor_v2 = build_context_tail_anchor_v2_from_producer_bundle(
        producer_bundle=e2e_inputs.bundle,
        requested_budget_rationals=requested,
    )
    checkpoints = {
        method: _checkpoint_v8(method)
        for method in ("T6_PLUS", "T7_PLUS", "T8_PLUS")
    }
    seals = {}
    for method, checkpoint in checkpoints.items():
        transcript = infer_and_seal_stage2_rc5plus(
            checkpoint=checkpoint,
            producer_bundle=e2e_inputs.bundle,
            anchor_v2=anchor_v2,
        )
        seal = verify_stage2_rc5plus_inference_seal(
            transcript,
            checkpoint=checkpoint,
            producer_bundle=e2e_inputs.bundle,
            anchor_v2=anchor_v2,
        )
        assert seal.method == method
        assert len(seal.decision["grid_rows"]) == 9
        assert len(seal.decision["requested_rows"]) == len(requested)
        assert seal.decision["labels_accessed"] is False
        assert seal.decision["query_accessed"] is False
        seals[method] = seal

    output = e2e_inputs.base / "rc5plus_production_atomic"
    output.mkdir()
    verified = publish_stage2_rc5plus_atomic_learned_decision_set(
        output,
        producer_bundle=e2e_inputs.bundle,
        checkpoints=checkpoints,
        inference_seals=seals,
        anchor_v2=anchor_v2,
        repository_root=ROOT,
    )
    calls: list[str] = []

    def resolver(value: Any) -> str:
        calls.append(value.decision_set_identity_sha256)
        return value.shared_prelabel_identity_sha256

    result = guarded_invoke_stage2_rc5plus_label_resolver(
        decision_set_path=verified.decision_set_path,
        commit_path=verified.commit_path,
        expected_commit_sha256=verified.commit_sha256,
        producer_bundle=e2e_inputs.bundle,
        checkpoints=checkpoints,
        inference_seals=seals,
        anchor_v2=anchor_v2,
        label_resolver=resolver,
        repository_root=ROOT,
    )
    assert result == verified.shared_prelabel_identity_sha256
    assert calls == [verified.decision_set_identity_sha256]

    no_anchor_checkpoint = _checkpoint_v8("T8_PLUS_NO_ANCHOR")
    no_anchor_transcript = infer_and_seal_stage2_rc5plus_no_anchor(
        checkpoint=no_anchor_checkpoint,
        producer_bundle=e2e_inputs.bundle,
    )
    no_anchor_verified = verify_stage2_rc5plus_no_anchor_inference_seal(
        no_anchor_transcript,
        checkpoint=no_anchor_checkpoint,
        producer_bundle=e2e_inputs.bundle,
    )
    assert len(no_anchor_verified.decision["grid_rows"]) == 9
    assert no_anchor_verified.decision["target_anchor_accessed"] is False
    assert no_anchor_verified.transcript["anchor_binding"][
        "anchor_schema"
    ] == "not_applicable"


def test_public_infer_fresh_replay_rejects_stale_source_v3(
    e2e_inputs: _E2EInputs,
) -> None:
    checkpoint = _checkpoint_t8()
    attestation = e2e_inputs.source_reference_v3.attestation_path
    original = attestation.read_bytes()
    attestation.write_bytes(original + b"stale-source-v3")
    try:
        with pytest.raises((OSError, RuntimeError, TypeError, ValueError)):
            infer_and_seal_stage2_rc5(
                checkpoint=checkpoint,
                producer_bundle=e2e_inputs.bundle,
            )
    finally:
        attestation.write_bytes(original)
    transcript = infer_and_seal_stage2_rc5(
        checkpoint=checkpoint,
        producer_bundle=e2e_inputs.bundle,
    )
    assert json.loads(transcript)["schema_version"] == TRANSCRIPT_SCHEMA


@pytest.mark.parametrize(
    "target_name",
    (
        "score_attestation",
        "run_complete",
        "score_manifest",
        "context_commit",
        "context_score_member",
    ),
)
def test_capability_to_consumer_toctou_tamper_is_rejected(
    e2e_inputs: _E2EInputs, target_name: str
) -> None:
    inputs = e2e_inputs
    targets = {
        "score_attestation": inputs.query_score_bundle.attestation_path,
        "run_complete": Path(inputs.run_complete.artifact_path),
        "score_manifest": inputs.query_score_bundle.score_manifest_metadata.path,
        "context_commit": inputs.bundle.commit_path,
        "context_score_member": inputs.context_score_paths[0],
    }
    target = targets[target_name]
    original = target.read_bytes()
    target.write_bytes(original + b"synthetic-toctou-tamper")
    try:
        with pytest.raises((OSError, RuntimeError, TypeError, ValueError)):
            _verify_bundle(inputs)
    finally:
        target.write_bytes(original)
    assert _verify_bundle(inputs).bundle_identity_sha256 == (
        inputs.bundle.bundle_identity_sha256
    )


def test_output_parent_symlink_is_rejected(
    e2e_inputs: _E2EInputs,
) -> None:
    real = e2e_inputs.base / "real_output_parent"
    real.mkdir()
    linked = e2e_inputs.base / "linked_output_parent"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(Stage2RC5ContextProducerError, match="symlink"):
        build_and_publish_stage2_rc5_context_bundle(
            **_build_kwargs(e2e_inputs, linked)
        )
    assert not any(real.iterdir())


def test_commit_last_link_failure_leaves_no_authoritative_member(
    e2e_inputs: _E2EInputs,
) -> None:
    output = e2e_inputs.base / "commit_last_fault"
    output.mkdir()
    original_link = context_producer.os.link
    calls = 0

    def fail_on_commit_link(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("synthetic commit-last link fault")
        original_link(*args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(context_producer.os, "link", fail_on_commit_link)
        with pytest.raises(OSError, match="commit-last"):
            build_and_publish_stage2_rc5_context_bundle(
                **_build_kwargs(e2e_inputs, output)
            )
    assert calls == 4
    assert not (output / CONTEXT_FILENAME).exists()
    assert not (output / ANCHOR_FILENAME).exists()
    assert not (output / PRODUCER_MANIFEST_FILENAME).exists()
    assert not (output / COMMIT_FILENAME).exists()
    assert list(output.iterdir()) == []


class _AtomicE2E(SimpleNamespace):
    source_curves: Mapping[str, Any]
    source_threshold_reference: atomic.VerifiedExactSourceThresholdReferenceV3
    checkpoints: Mapping[str, Any]
    inference_seals: Mapping[str, Any]
    evt_seal: atomic.VerifiedStage2RC5EVTSeal
    decision_set: atomic.VerifiedStage2RC5AtomicDecisionSet


@pytest.fixture(scope="module")
def atomic_e2e(e2e_inputs: _E2EInputs) -> Any:
    detector_sha256 = atomic.canonical_json_sha256(
        dict(e2e_inputs.source_reference_v3.detector_identity)
    )
    thresholds = np.asarray(
        [0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float64
    )
    domains = tuple(
        e2e_inputs.source_reference_v3.source_reference_v2
        .source_reference_bundle.domains
    )
    assert len(domains) == 2
    false_positive_pixels = (
        np.asarray([5_000, 1_000, 100, 10, 0], dtype=np.int64),
        np.asarray([6_000, 1_500, 150, 15, 0], dtype=np.int64),
    )
    source_curves = {
        domain: atomic.build_exact_source_domain_curve_v2(
            source_domain=domain,
            detector_identity_sha256=detector_sha256,
            thresholds=thresholds,
            false_positive_pixels=false_positive_pixels[index],
            matched_objects=np.asarray([10, 9, 8, 7, 0], dtype=np.int64),
            total_native_pixels=10_000_000,
            ground_truth_objects=10,
        )
        for index, domain in enumerate(domains)
    }
    source_threshold_reference = (
        atomic.build_exact_source_threshold_reference_v3(
            domain_curves=source_curves,
            source_reference=e2e_inputs.source_reference_v3,
        )
    )

    checkpoints: dict[str, Any] = {}
    inference_seals: dict[str, Any] = {}
    for method in atomic.LEARNED_METHOD_IDS:
        checkpoint = _checkpoint(method)
        transcript = infer_and_seal_stage2_rc5(
            checkpoint=checkpoint,
            producer_bundle=e2e_inputs.bundle,
        )
        checkpoints[method] = checkpoint
        inference_seals[method] = verify_stage2_rc5_inference_seal(
            transcript,
            checkpoint=checkpoint,
            producer_bundle=e2e_inputs.bundle,
        )

    evt_seal = atomic.build_stage2_rc5_evt_seal_complete(
        producer_identity_sha256=e2e_inputs.bundle.producer_manifest[
            "producer_identity_sha256"
        ],
        context_full_identity_sha256=e2e_inputs.bundle.context.payload[
            "context_full_identity_sha256"
        ],
        anchor_identity_sha256=e2e_inputs.bundle.anchor.payload[
            "anchor_identity_sha256"
        ],
        thresholds=np.asarray([0.2, 0.4, 0.8], dtype=np.float64),
        fit_identity_sha256=_digest("synthetic-actual-capability-evt-fit"),
    )
    output = e2e_inputs.base / "atomic_actual_capability"
    output.mkdir()
    decision_set = atomic.publish_stage2_rc5_atomic_decision_set(
        output,
        producer_bundle=e2e_inputs.bundle,
        source_threshold_reference=source_threshold_reference,
        inference_seals=inference_seals,
        evt_seal=evt_seal,
        repository_root=ROOT,
    )
    return _AtomicE2E(
        source_curves=source_curves,
        source_threshold_reference=source_threshold_reference,
        checkpoints=checkpoints,
        inference_seals=inference_seals,
        evt_seal=evt_seal,
        decision_set=decision_set,
    )


def _atomic_verify_kwargs(
    inputs: _E2EInputs, materials: _AtomicE2E
) -> dict[str, Any]:
    return {
        "decision_set_path": materials.decision_set.decision_set_path,
        "commit_path": materials.decision_set.commit_path,
        "expected_commit_sha256": materials.decision_set.commit_sha256,
        "producer_bundle": inputs.bundle,
        "source_threshold_reference": materials.source_threshold_reference,
        "inference_seals": materials.inference_seals,
        "evt_seal": materials.evt_seal,
        "repository_root": ROOT,
    }


def _clone_exact_reference_with_numeric_drift(
    value: atomic.VerifiedExactSourceThresholdReferenceV3,
    field: str,
) -> atomic.VerifiedExactSourceThresholdReferenceV3:
    forged = object.__new__(atomic.VerifiedExactSourceThresholdReferenceV3)
    for name in atomic.VerifiedExactSourceThresholdReferenceV3.__dataclass_fields__:
        object.__setattr__(forged, name, getattr(value, name))
    numeric = np.array(getattr(value, field), copy=True)
    numeric.flat[0] += np.asarray(0.125, dtype=numeric.dtype)
    numeric.setflags(write=False)
    object.__setattr__(forged, field, numeric)
    return forged


def test_atomic_actual_capabilities_publish_verify_and_guard_t0_t8(
    e2e_inputs: _E2EInputs,
    atomic_e2e: _AtomicE2E,
) -> None:
    issued = atomic_e2e.decision_set
    assert tuple(issued.decision_by_method) == atomic.METHOD_IDS
    assert len(issued.decisions) == 9
    assert all(
        decision["artifact_status"] == "prelabel_complete"
        and decision["outcome"] == "complete"
        and decision["labels_accessed"] is False
        and decision["query_members_opened"] is False
        for decision in issued.decisions
    )
    assert issued.thresholds("T0") == (0.5, 0.5, 0.5)
    assert all(issued.thresholds(method) is not None for method in atomic.METHOD_IDS)

    replayed = atomic.verify_stage2_rc5_atomic_decision_set(
        **_atomic_verify_kwargs(e2e_inputs, atomic_e2e)
    )
    assert replayed.decision_set_identity_sha256 == (
        issued.decision_set_identity_sha256
    )
    assert replayed.shared_prelabel_identity_sha256 == (
        issued.shared_prelabel_identity_sha256
    )

    resolver_calls: list[tuple[Any, str, bool]] = []

    def resolver(
        verified: atomic.VerifiedStage2RC5AtomicDecisionSet,
        marker: str,
        *,
        labels_enabled: bool,
    ) -> str:
        atomic.assert_verified_stage2_rc5_atomic_decision_set(verified)
        resolver_calls.append((verified, marker, labels_enabled))
        return verified.decision_set_identity_sha256

    result = atomic.guarded_invoke_stage2_rc5_label_resolver(
        **_atomic_verify_kwargs(e2e_inputs, atomic_e2e),
        label_resolver=resolver,
        resolver_args=("after-full-verification",),
        resolver_kwargs={"labels_enabled": True},
    )
    assert result == issued.decision_set_identity_sha256
    assert len(resolver_calls) == 1
    assert resolver_calls[0][0].decision_set_identity_sha256 == (
        replayed.decision_set_identity_sha256
    )
    assert resolver_calls[0][1:] == ("after-full-verification", True)


class _RC5PlusFullAtomicE2E(SimpleNamespace):
    anchor_v2: Any
    checkpoints: Mapping[str, Any]
    inference_seals: Mapping[str, Any]
    decision_set: atomic_full.VerifiedStage2RC5PlusAtomicFullDecisionSet


@pytest.fixture(scope="module")
def rc5plus_full_atomic_e2e(
    e2e_inputs: _E2EInputs,
    atomic_e2e: _AtomicE2E,
) -> Any:
    requested = ((1, 25_000), (1, 250_000))
    anchor_v2 = build_context_tail_anchor_v2_from_producer_bundle(
        producer_bundle=e2e_inputs.bundle,
        requested_budget_rationals=requested,
    )
    checkpoints = {
        method: _checkpoint_v8(method)
        for method in ("T6_PLUS", "T7_PLUS", "T8_PLUS")
    }
    seals = {}
    for method, checkpoint in checkpoints.items():
        transcript = infer_and_seal_stage2_rc5plus(
            checkpoint=checkpoint,
            producer_bundle=e2e_inputs.bundle,
            anchor_v2=anchor_v2,
        )
        seals[method] = verify_stage2_rc5plus_inference_seal(
            transcript,
            checkpoint=checkpoint,
            producer_bundle=e2e_inputs.bundle,
            anchor_v2=anchor_v2,
        )

    output = e2e_inputs.base / "rc5plus_full_atomic_actual_capabilities"
    output.mkdir()
    decision_set = atomic_full.publish_stage2_rc5plus_atomic_full_decision_set(
        output,
        producer_bundle=e2e_inputs.bundle,
        source_threshold_reference=atomic_e2e.source_threshold_reference,
        checkpoints=checkpoints,
        inference_seals=seals,
        anchor_v2=anchor_v2,
        evt_seal=atomic_e2e.evt_seal,
        repository_root=ROOT,
    )
    return _RC5PlusFullAtomicE2E(
        anchor_v2=anchor_v2,
        checkpoints=checkpoints,
        inference_seals=seals,
        decision_set=decision_set,
    )


def _rc5plus_full_verify_kwargs(
    e2e_inputs: _E2EInputs,
    atomic_e2e: _AtomicE2E,
    materials: _RC5PlusFullAtomicE2E,
) -> dict[str, Any]:
    return {
        "decision_set_path": materials.decision_set.decision_set_path,
        "commit_path": materials.decision_set.commit_path,
        "expected_commit_sha256": materials.decision_set.commit_sha256,
        "producer_bundle": e2e_inputs.bundle,
        "source_threshold_reference": atomic_e2e.source_threshold_reference,
        "checkpoints": materials.checkpoints,
        "inference_seals": materials.inference_seals,
        "anchor_v2": materials.anchor_v2,
        "evt_seal": atomic_e2e.evt_seal,
        "repository_root": ROOT,
    }


def test_rc5plus_full_atomic_actual_capabilities_bind_t0_t8_before_labels(
    e2e_inputs: _E2EInputs,
    atomic_e2e: _AtomicE2E,
    rc5plus_full_atomic_e2e: _RC5PlusFullAtomicE2E,
) -> None:
    issued = rc5plus_full_atomic_e2e.decision_set
    checkpoints = rc5plus_full_atomic_e2e.checkpoints
    seals = rc5plus_full_atomic_e2e.inference_seals
    anchor_v2 = rc5plus_full_atomic_e2e.anchor_v2
    assert (
        atomic_full.assert_verified_stage2_rc5plus_atomic_full_decision_set(
            issued
        )
        is issued
    )
    with pytest.raises(TypeError, match="verifier-issued"):
        atomic_full.VerifiedStage2RC5PlusAtomicFullDecisionSet()
    assert tuple(issued.decision_by_method) == atomic.METHOD_IDS
    assert issued.payload["t9_included"] is False
    assert (
        issued.payload["t9_policy"]
        == "separate_postlabel_oracle_diagnostic_only"
    )
    assert len(issued.decision_by_method) == 9
    for learned_method, projected_method in atomic_full.METHOD_MAP.items():
        assert issued.decision_by_method[projected_method]["method_name"] == (
            atomic_full.METHOD_NAMES[projected_method]
        )
        projected_rows = issued.decision_by_method[projected_method]["rows"]
        sealed_rows = seals[learned_method].decision["grid_rows"]
        for projected_row, grid_index in zip(
            projected_rows, (0, 4, 8), strict=True
        ):
            sealed_row = sealed_rows[grid_index]
            assert projected_row["threshold_probability_hex"] == (
                sealed_row["decoded_threshold_hex"]
            )
            assert projected_row["threshold_coordinate_hex"] == (
                sealed_row["canonical_coordinate_hex"]
            )

    resolver_calls: list[str] = []

    def resolver(value: Any) -> str:
        resolver_calls.append(value.decision_set_identity_sha256)
        return value.decision_by_method["T8"]["decision_identity_sha256"]

    result = atomic_full.guarded_invoke_stage2_rc5plus_full_label_resolver(
        **_rc5plus_full_verify_kwargs(
            e2e_inputs, atomic_e2e, rc5plus_full_atomic_e2e
        ),
        label_resolver=resolver,
    )
    assert result == issued.decision_by_method["T8"][
        "decision_identity_sha256"
    ]
    assert resolver_calls == [issued.decision_set_identity_sha256]


def test_rc5plus_full_atomic_upstream_tamper_never_calls_label_resolver(
    e2e_inputs: _E2EInputs,
    atomic_e2e: _AtomicE2E,
    rc5plus_full_atomic_e2e: _RC5PlusFullAtomicE2E,
) -> None:
    target = e2e_inputs.source_reference_v3.attestation_path
    original = target.read_bytes()
    original_stat = target.stat(follow_symlinks=False)
    resolver_calls = 0

    def resolver(_verified: Any) -> None:
        nonlocal resolver_calls
        resolver_calls += 1

    target.write_bytes(original + b"rc5plus-full-upstream-tamper")
    try:
        with pytest.raises((OSError, RuntimeError, TypeError, ValueError)):
            atomic_full.guarded_invoke_stage2_rc5plus_full_label_resolver(
                **_rc5plus_full_verify_kwargs(
                    e2e_inputs, atomic_e2e, rc5plus_full_atomic_e2e
                ),
                label_resolver=resolver,
            )
    finally:
        target.write_bytes(original)
        os.utime(
            target,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            follow_symlinks=False,
        )
    assert resolver_calls == 0


def test_rc5plus_full_atomic_commit_last_fault_leaves_no_authority(
    e2e_inputs: _E2EInputs,
    atomic_e2e: _AtomicE2E,
    rc5plus_full_atomic_e2e: _RC5PlusFullAtomicE2E,
) -> None:
    output = e2e_inputs.base / "rc5plus_full_atomic_commit_last_fault"
    output.mkdir()
    original_link = atomic_full.os.link
    link_calls = 0

    def fail_on_commit_link(*args: Any, **kwargs: Any) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise OSError("synthetic RC5+ full commit-last link fault")
        original_link(*args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(atomic_full.os, "link", fail_on_commit_link)
        with pytest.raises(OSError, match="commit-last"):
            atomic_full.publish_stage2_rc5plus_atomic_full_decision_set(
                output,
                producer_bundle=e2e_inputs.bundle,
                source_threshold_reference=(
                    atomic_e2e.source_threshold_reference
                ),
                checkpoints=rc5plus_full_atomic_e2e.checkpoints,
                inference_seals=rc5plus_full_atomic_e2e.inference_seals,
                anchor_v2=rc5plus_full_atomic_e2e.anchor_v2,
                evt_seal=atomic_e2e.evt_seal,
                repository_root=ROOT,
            )
    assert link_calls == 2
    assert not (output / atomic_full.DECISION_SET_FILENAME).exists()
    assert not (output / atomic_full.COMMIT_FILENAME).exists()
    assert list(output.iterdir()) == []


def test_atomic_exact_reference_rejects_legacy_and_free_numeric_inputs(
    e2e_inputs: _E2EInputs,
    atomic_e2e: _AtomicE2E,
) -> None:
    with pytest.raises(TypeError, match="SourceReferenceV3|verifier-issued"):
        atomic.build_exact_source_threshold_reference_v3(
            domain_curves=atomic_e2e.source_curves,
            source_reference=e2e_inputs.source_reference_v2,
        )

    forbidden = {
        "detector_identity": dict(
            e2e_inputs.source_reference_v3.detector_identity
        ),
        "centers": np.zeros((2, 87), dtype=np.float32),
        "scale": np.ones(87, dtype=np.float32),
        "source_centers": np.zeros((2, 87), dtype=np.float32),
        "source_scale": np.ones(87, dtype=np.float32),
    }
    for keyword, value in forbidden.items():
        with pytest.raises(TypeError, match="unexpected keyword"):
            atomic.build_exact_source_threshold_reference_v3(
                domain_curves=atomic_e2e.source_curves,
                source_reference=e2e_inputs.source_reference_v3,
                **{keyword: value},
            )


@pytest.mark.parametrize("field", ("source_centers", "source_scale"))
def test_atomic_retained_token_numeric_clone_is_rejected_before_resolver(
    e2e_inputs: _E2EInputs,
    atomic_e2e: _AtomicE2E,
    field: str,
) -> None:
    forged = _clone_exact_reference_with_numeric_drift(
        atomic_e2e.source_threshold_reference, field
    )
    with pytest.raises(TypeError, match="numeric state drifted"):
        atomic.assert_verified_exact_source_threshold_reference_v3(forged)

    resolver_calls = 0

    def resolver(_verified: Any) -> None:
        nonlocal resolver_calls
        resolver_calls += 1

    kwargs = _atomic_verify_kwargs(e2e_inputs, atomic_e2e)
    kwargs["source_threshold_reference"] = forged
    with pytest.raises(TypeError, match="numeric state drifted"):
        atomic.guarded_invoke_stage2_rc5_label_resolver(
            **kwargs,
            label_resolver=resolver,
        )
    assert resolver_calls == 0


@pytest.mark.parametrize(
    "authority_name",
    ("source_v3_attestation", "context_attestation", "run_complete"),
)
def test_atomic_upstream_mismatch_never_calls_label_resolver(
    e2e_inputs: _E2EInputs,
    atomic_e2e: _AtomicE2E,
    authority_name: str,
) -> None:
    targets = {
        "source_v3_attestation": e2e_inputs.source_reference_v3.attestation_path,
        "context_attestation": e2e_inputs.bundle.producer_manifest_path,
        "run_complete": Path(e2e_inputs.run_complete.artifact_path),
    }
    target = targets[authority_name]
    original = target.read_bytes()
    original_stat = target.stat(follow_symlinks=False)
    resolver_calls = 0

    def resolver(_verified: Any) -> None:
        nonlocal resolver_calls
        resolver_calls += 1

    target.write_bytes(original + b"atomic-resolver-authority-mismatch")
    try:
        with pytest.raises((OSError, RuntimeError, TypeError, ValueError)):
            atomic.guarded_invoke_stage2_rc5_label_resolver(
                **_atomic_verify_kwargs(e2e_inputs, atomic_e2e),
                label_resolver=resolver,
            )
    finally:
        target.write_bytes(original)
        os.utime(
            target,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            follow_symlinks=False,
        )
    assert resolver_calls == 0


def test_atomic_commit_last_link_fault_leaves_no_decision_authority(
    e2e_inputs: _E2EInputs,
    atomic_e2e: _AtomicE2E,
) -> None:
    output = e2e_inputs.base / "atomic_commit_last_fault"
    output.mkdir()
    original_link = atomic.os.link
    link_calls = 0

    def fail_on_commit_link(*args: Any, **kwargs: Any) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise OSError("synthetic atomic commit-last link fault")
        original_link(*args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(atomic.os, "link", fail_on_commit_link)
        with pytest.raises(OSError, match="atomic commit-last"):
            atomic.publish_stage2_rc5_atomic_decision_set(
                output,
                producer_bundle=e2e_inputs.bundle,
                source_threshold_reference=(
                    atomic_e2e.source_threshold_reference
                ),
                inference_seals=atomic_e2e.inference_seals,
                evt_seal=atomic_e2e.evt_seal,
                repository_root=ROOT,
            )
    assert link_calls == 2
    assert not (output / atomic.DECISION_SET_FILENAME).exists()
    assert not (output / atomic.DECISION_SET_COMMIT_FILENAME).exists()
    assert list(output.iterdir()) == []


def test_cyclic_context_actual_capability_build_replay_and_geometry_isolation(
    e2e_inputs: _E2EInputs,
) -> None:
    output = e2e_inputs.base / "cyclic-context"
    collection = (
        cyclic_context.build_and_publish_stage2_rc5_cyclic_context_collection(
            score_bundle=e2e_inputs.query_score_bundle,
            source_reference=e2e_inputs.source_reference_v3,
            statistics_config=e2e_inputs.statistics_config,
            statistics_config_path=e2e_inputs.statistics_path,
            statistics_config_sha256=e2e_inputs.statistics_sha,
            output_directory=output,
            repository_root=ROOT,
        )
    )
    assert collection.context_features.shape == (43, 93)
    assert collection.anchor_coordinates.shape == (43, 3)
    assert len(collection.episodes) == 43
    assert collection.manifest["variable_query_window_accepted"] is False
    assert not isinstance(collection.context_features, np.memmap)
    assert not isinstance(collection.anchor_coordinates, np.memmap)
    assert collection.context_features.flags.owndata is True
    assert collection.anchor_coordinates.flags.owndata is True
    assert collection.context_features.flags.writeable is False
    assert collection.anchor_coordinates.flags.writeable is False

    replayed = (
        cyclic_context.replay_verified_stage2_rc5_cyclic_context_collection(
            collection
        )
    )
    assert replayed.commit_sha256 == collection.commit_sha256
    assert np.array_equal(replayed.context_features, collection.context_features)
    assert np.array_equal(
        replayed.anchor_coordinates, collection.anchor_coordinates
    )

    valid = {
        "score_bundle": e2e_inputs.query_score_bundle,
        "source_reference": e2e_inputs.source_reference_v3,
        "statistics_config": e2e_inputs.statistics_config,
        "statistics_config_path": e2e_inputs.statistics_path,
        "statistics_config_sha256": e2e_inputs.statistics_sha,
        "repository_root": ROOT,
    }
    invalid_cases = (
        (
            "bare_metadata",
            {
                "score_bundle": (
                    e2e_inputs.query_score_bundle.score_manifest_metadata
                )
            },
        ),
        ("context_as_score_bundle", {"score_bundle": e2e_inputs.bundle}),
        (
            "context_as_source_reference",
            {"source_reference": e2e_inputs.bundle},
        ),
    )
    for name, replacement in invalid_cases:
        rejected_output = e2e_inputs.base / f"cyclic-reject-{name}"
        invalid = {**valid, **replacement}
        with pytest.raises(TypeError):
            cyclic_context.build_and_publish_stage2_rc5_cyclic_context_collection(
                **invalid,
                output_directory=rejected_output,
            )
        assert not rejected_output.exists()

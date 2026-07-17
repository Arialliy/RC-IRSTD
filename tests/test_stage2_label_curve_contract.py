from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from PIL import Image
import pytest

import evaluation.export_stage2_labels as exporter
import evaluation.component_matching as component_matching
from evaluation.component_matching import aggregate_match_results, match_components, prepare_target
from data_ext.stage2_label_attachment import (
    canonical_json_sha256,
    stage2_ordered_query_identity,
    verify_stage2_label_attachment,
)
from data_ext.stage2_score_manifest import (
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
    OOF_HOLDOUT_STAGE2_FIT,
    SOURCE_DIAGNOSTIC_VALIDATION,
)
from data_ext.stage2_threshold_decision import PRELABEL_METHOD_ORDER
from evaluation.export_stage2_labels import export_stage2_labels
from evaluation.stage2_threshold_family import (
    build_prelabel_decision,
    make_shared_input_bindings,
    publish_prelabel_decision_set,
)
from evaluation.stage2_threshold_sweep import (
    ArrayBackedCurveRows,
    CURVE_FIELDS,
    _build_incremental_exact_sweep,
    _read_curve_csv,
    stage2_curve_rows_sha256,
    verify_stage2_query_curve_artifacts,
)
from rc import build_stage2_crossfit_episodes as crossfit_builder
from rc import stage2_crossfit_schema as crossfit_schema
from rc.domain_statistics import BASE_FEATURE_DIM
from rc.schema import SourceContract, SourceReference, StatisticsConfig
import tests.test_stage2_score_manifest_v4 as score_fixture


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _selected_record(root: Path, index: int, domain: str) -> dict[str, Any]:
    image_id = "Misc_111" if index == 14 else f"sample_{index:02d}"
    image_path = root / "datasets" / domain / "images" / f"{image_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    # Small synthetic native geometries keep exact-event tests fast.
    height, width = (6, 8) if image_id == "Misc_111" else (5 + index % 2, 7 + index % 3)
    Image.fromarray(
        np.full((height, width), 30 + index, dtype=np.uint8), mode="L"
    ).save(image_path)
    image_sha = _sha(image_path)
    return {
        "canonical_id": f"{domain}::{image_id}",
        "image_id": image_id,
        "original_image_path": image_path.relative_to(root).as_posix(),
        "original_image_sha256": image_sha,
        "exclusion_group_id": f"SINGLETON::{domain}::{image_id}::{image_sha}",
        "near_duplicate_cluster_id_or_unique_sentinel": (
            f"UNIQUE_NO_CONFIRMED_NEAR_DUPLICATE::{domain}::{image_id}"
        ),
        "source_role_record_index": index,
    }


def _guardrails() -> dict[str, bool]:
    return {
        "development_only": True,
        "result_free": True,
        "execution_authorized": False,
        "official_test_split_files_opened": False,
        "official_test_ids_materialized": False,
        "official_test_images_opened": False,
        "mask_or_label_files_opened": False,
        "predictions_scores_checkpoints_or_metrics_opened": False,
        "original_training_images_opened_only_for_sha256": True,
    }


def _synthetic_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    role: str = SOURCE_DIAGNOSTIC_VALIDATION,
) -> dict[str, Any]:
    monkeypatch.setattr(score_fixture, "_selected_record", _selected_record)
    fixture = score_fixture._fixture(
        tmp_path,
        role,
        record_count=42,
    )
    score_records = fixture["manifest"]["records"]
    source_domain = fixture["manifest"]["source_domain"]
    is_oof = role == OOF_HOLDOUT_STAGE2_FIT
    source_role = "detector_fit" if is_oof else "detector_diagnostic"
    window_episode_role = "stage2_oof_fit" if is_oof else role
    oof_fold_index = 0 if is_oof else None
    projected: list[dict[str, Any]] = []
    identity_fields = (
        "canonical_id",
        "image_id",
        "original_image_path",
        "original_image_sha256",
        "exclusion_group_id",
        "near_duplicate_cluster_id_or_unique_sentinel",
        "source_role_record_index",
    )
    for index, score_record in enumerate(score_records):
        record = {field: score_record[field] for field in identity_fields}
        record.update(
            {
                "source_role": source_role,
                "outer_fold_id": "outer_leave_nuaa_sirst",
                "episode_role": "context" if index < 14 else "query",
                **({"oof_fold_index": oof_fold_index} if is_oof else {}),
            }
        )
        projected.append(record)

    contracts = tmp_path / "w05-contracts"
    unused = contracts / "unused.json"
    _write_json(unused, {"synthetic_unused_suffix": [], "official_test_accessed": False})
    bound_paths = {
        "image_only_near_duplicate_audit": tmp_path / "bindings" / "detector_config.json",
        "k2_geometry_prefreeze_audit": tmp_path / "bindings" / "seed_manifest.json",
        "official_train_derived_split_manifest": tmp_path / "bindings" / "materialization_index.json",
    }
    window_id = f"outer_leave_nuaa_sirst::{role}::window_000"
    window_payload = {
        "schema_version": "rc-irstd.stage2-role-pure-c14q28-windows.v1",
        "artifact_type": "rc_irstd_stage2_role_pure_episode_windows",
        "artifact_status": "DEVELOPMENT_ONLY_RESULT_FREE",
        "execution_authorized": False,
        "observed_results": None,
        "outer_fold_id": "outer_leave_nuaa_sirst",
        "outer_target_domain": "NUAA-SIRST",
        "domain": source_domain,
        "source_role": source_role,
        "episode_role": window_episode_role,
        "oof_fold_index": oof_fold_index,
        "geometry": {
            "context_size": 14,
            "query_size": 28,
            "block_size": 42,
            "construction": "ordered_non_overlapping_contiguous_blocks_context_first_query_second",
        },
        "role_purity": {
            "allowed_source_role": source_role,
            "mixed_roles_allowed": False,
            "single_source_domain_per_window": True,
            "single_oof_checkpoint_identity_per_fit_window_required_at_future_score_binding": is_oof,
        },
        "ordered_role_record_count": 42,
        "complete_window_count": 1,
        "window_record_count": 42,
        "unused_suffix": {
            "path": unused.relative_to(tmp_path).as_posix(),
            "sha256": _sha(unused),
        },
        "role_binding": {
            "path": fixture["selection_path"].relative_to(tmp_path).as_posix(),
            "sha256": _sha(fixture["selection_path"]),
        },
        "bound_inputs": {
            name: {
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": _sha(path),
            }
            for name, path in bound_paths.items()
        },
        "guardrails": _guardrails(),
        "windows": [
            {
                "window_index": 0,
                "window_id": window_id,
                "context_records": projected[:14],
                "query_records": projected[14:],
            }
        ],
    }
    window_path = contracts / "windows.json"
    _write_json(window_path, window_payload)

    dataset = tmp_path / "datasets" / source_domain
    masks = dataset / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    for record in projected[14:]:
        image = tmp_path / record["original_image_path"]
        with Image.open(image) as handle:
            height, width = handle.height, handle.width
        if record["image_id"] == "Misc_111":
            mask_hw = (height * 2, width * 2)
        else:
            mask_hw = (height, width)
        mask = np.zeros(mask_hw, dtype=np.uint8)
        mask[0, 0] = 255
        Image.fromarray(mask, mode="L").save(masks / f"{record['image_id']}.png")
    output_parent = tmp_path / "w05-outputs"
    output_parent.mkdir()
    return {
        "root": tmp_path,
        "fixture": fixture,
        "score_path": fixture["path"],
        "score_sha": _sha(fixture["path"]),
        "window_path": window_path,
        "window_sha": _sha(window_path),
        "role": role,
        "window_id": window_id,
        "window_payload": window_payload,
        "dataset": dataset,
        "query_ids": [record["image_id"] for record in projected[14:]],
        "context_ids": [record["image_id"] for record in projected[:14]],
        "output": output_parent / "window-000",
    }


def _run(workspace: Mapping[str, Any]) -> dict[str, Any]:
    return export_stage2_labels(
        score_manifest=workspace["score_path"],
        score_manifest_sha256=workspace["score_sha"],
        window_manifest=workspace["window_path"],
        window_manifest_sha256=workspace["window_sha"],
        window_id=workspace["window_id"],
        expected_role=workspace["role"],
        dataset_dir=workspace["dataset"],
        output_dir=workspace["output"],
        sealed_decision_set=workspace.get("decision_set_path"),
        sealed_decision_set_sha256=workspace.get("decision_set_sha256"),
        statistics_config=workspace.get("statistics_config_path"),
        statistics_config_sha256=workspace.get("statistics_config_sha256"),
        repository_root=workspace["root"],
    )


def _seal_outer_decision_set(
    workspace: dict[str, Any],
    *,
    context_binding: Mapping[str, Any] | None = None,
    output_name: str = "sealed-t0-t8",
) -> dict[str, str]:
    score_payload = workspace["fixture"]["manifest"]
    selected_window = workspace["window_payload"]["windows"][0]
    if "statistics_config_path" not in workspace:
        statistics_config_path = (
            workspace["root"] / "synthetic" / "statistics-config.json"
        )
        _write_json(
            statistics_config_path,
            StatisticsConfig(
                peak_kernel_size=3,
                peak_min_score=0.05,
                quantile_sample_limit=128,
            ).to_dict(),
        )
        workspace["statistics_config_path"] = statistics_config_path
        workspace["statistics_config_sha256"] = _sha(statistics_config_path)
    context_path = (
        "synthetic/context-package.json"
        if context_binding is None
        else Path(context_binding["path"]).relative_to(workspace["root"]).as_posix()
    )
    context_commit_path = (
        "synthetic/context-package.commit.json"
        if context_binding is None
        else Path(context_binding["commit_path"])
        .relative_to(workspace["root"])
        .as_posix()
    )
    context_sha = (
        canonical_json_sha256({"synthetic": "context-package"})
        if context_binding is None
        else str(context_binding["sha256"])
    )
    context_commit_sha = (
        canonical_json_sha256({"synthetic": "context-package-commit"})
        if context_binding is None
        else str(context_binding["commit_sha256"])
    )
    shared = make_shared_input_bindings(
        context_package_path=context_path,
        context_package_sha256=context_sha,
        context_package_commit_path=context_commit_path,
        context_package_commit_sha256=context_commit_sha,
        window_id=workspace["window_id"],
        window_identity_sha256=canonical_json_sha256(selected_window),
        ordered_query_identity_sha256=canonical_json_sha256(
            stage2_ordered_query_identity(selected_window["query_records"])
        ),
        score_manifest_sha256=workspace["score_sha"],
        score_records_content_sha256=score_payload["records_content_sha256"],
        detector_checkpoint_sha256=score_payload["bindings"]["checkpoint"]["sha256"],
    )
    decisions = []
    for method_id in PRELABEL_METHOD_ORDER:
        thresholds = (
            (0.5, 0.5, 0.5)
            if method_id == "T0"
            else None
            if method_id == "T5"
            else (0.6, 0.7, 0.8)
        )
        decisions.append(
            build_prelabel_decision(
                method_id=method_id,
                thresholds=thresholds,
                shared_bindings=shared,
                outer_fold_id=score_payload["outer_fold_id"],
                outer_target_domain=score_payload["outer_target"],
                base_seed=score_payload["base_seed"],
                derived_seed=score_payload["derived_seed"],
                method_contract={"method_id": method_id, "synthetic": True},
                method_binding=(
                    {
                        "path": f"synthetic/{method_id}.frozen-input.json",
                        "sha256": canonical_json_sha256(
                            {"synthetic_method_binding": method_id}
                        ),
                    }
                    if method_id in {"T1", "T2", "T3", "T5", "T6", "T7", "T8"}
                    else None
                ),
            )
        )
    path, digest = publish_prelabel_decision_set(
        decisions,
        workspace["root"] / output_name,
        repository_root=workspace["root"],
    )
    workspace["decision_set_path"] = path
    workspace["decision_set_sha256"] = digest
    return {
        "path": path.relative_to(workspace["root"]).as_posix(),
        "sha256": digest,
        "decision_set_content_sha256": json.loads(
            path.read_text(encoding="utf-8")
        )["decision_set_content_sha256"],
        "context_package_sha256": context_sha,
        "context_package_commit_sha256": context_commit_sha,
        "statistics_config_path": Path(workspace["statistics_config_path"])
        .relative_to(workspace["root"])
        .as_posix(),
        "statistics_config_sha256": str(workspace["statistics_config_sha256"]),
    }


def _publish_real_bound_context_package(
    workspace: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Run the real Lane-A build/verifier while stubbing only W04 verification."""

    score_payload = workspace["fixture"]["manifest"]
    implementation_root = Path(crossfit_schema.__file__).resolve().parents[1]
    synthetic_rc = workspace["root"] / "rc"
    synthetic_rc.mkdir(exist_ok=True)
    (synthetic_rc / "domain_statistics.py").write_text(
        "# synthetic immutable extractor binding for contract replay\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        crossfit_schema,
        "__file__",
        str(synthetic_rc / "stage2_crossfit_schema.py"),
    )
    for binding in crossfit_schema.GOVERNANCE_BINDINGS.values():
        source = implementation_root / binding["path"]
        destination = workspace["root"] / binding["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    config_path = Path(workspace["statistics_config_path"])
    config = StatisticsConfig.from_dict(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    reference_path = workspace["root"] / "synthetic" / "source-reference.npz"
    reference_audit_path = (
        workspace["root"] / "synthetic" / "source-reference.audit.json"
    )
    reference_path.write_bytes(b"synthetic-source-reference\n")
    _write_json(reference_audit_path, {"synthetic": "source-reference-audit"})
    reference_sha = _sha(reference_path)
    reference_audit_sha = _sha(reference_audit_path)
    source_domains = ("NUDT-SIRST", "IRSTD-1K")
    source_contract = SourceContract(
        detector_checkpoint_sha=score_payload["bindings"]["checkpoint"]["sha256"],
        detector_source_domains=source_domains,
        outer_fold_id=score_payload["outer_fold_id"],
        outer_target=score_payload["outer_target"],
        held_out_domains=(score_payload["outer_target"],),
        protocol_scope="multi_source_protocol_candidate",
    )
    source_reference = SourceReference(
        domains=source_domains,
        sha256=reference_sha,
        centers=(
            tuple(np.zeros(BASE_FEATURE_DIM, dtype=np.float64)),
            tuple(np.ones(BASE_FEATURE_DIM, dtype=np.float64)),
        ),
        scale=tuple(np.ones(BASE_FEATURE_DIM, dtype=np.float64)),
        contract=source_contract,
    )
    detector_identity = {
        "outer_fold_id": score_payload["outer_fold_id"],
        "outer_target": score_payload["outer_target"],
        "base_seed": score_payload["base_seed"],
        "derived_seed": score_payload["derived_seed"],
        "detector_role": score_payload["detector_role"],
        "oof_fold_index": score_payload["oof_fold_index"],
        "checkpoint_sha256": score_payload["bindings"]["checkpoint"]["sha256"],
    }
    fake_reference = SimpleNamespace(
        path=reference_path,
        npz_sha256=reference_sha,
        audit_path=reference_audit_path,
        audit_sha256=reference_audit_sha,
        source_reference=source_reference,
        statistics_config=config,
        stage2_contract={
            "reference_role": "synthetic_outer_fullfit_source_reference",
            "detector_identity": detector_identity,
            "bindings": {
                "consumer_window_manifests": [
                    {
                        "path": workspace["window_path"]
                        .relative_to(workspace["root"])
                        .as_posix(),
                        "sha256": workspace["window_sha"],
                        "domain": score_payload["source_domain"],
                        "episode_role": workspace["role"],
                    }
                ],
                "statistics_config": {
                    "path": config_path.relative_to(workspace["root"]).as_posix(),
                    "sha256": workspace["statistics_config_sha256"],
                },
            },
        },
    )

    def verify_fake_w04(*args: Any, **kwargs: Any) -> Any:
        assert Path(args[0]).absolute() == reference_path
        assert args[1] == reference_sha
        assert args[2] == reference_audit_sha
        assert kwargs["statistics_config"] == config
        assert Path(kwargs["expected_consumer_window_path"]) == workspace["window_path"]
        assert kwargs["expected_consumer_window_sha256"] == workspace["window_sha"]
        assert kwargs["expected_consumer_window_id"] == workspace["window_id"]
        return fake_reference

    monkeypatch.setattr(
        crossfit_schema,
        "verify_stage2_source_reference",
        verify_fake_w04,
    )
    output = workspace["root"] / "synthetic" / "context-package.json"
    published = crossfit_builder.build_stage2_context_package(
        window_manifest=workspace["window_path"],
        window_manifest_sha256=workspace["window_sha"],
        window_id=workspace["window_id"],
        expected_role=workspace["role"],
        score_manifest=workspace["score_path"],
        score_manifest_sha256=workspace["score_sha"],
        source_reference=reference_path,
        source_reference_sha256=reference_sha,
        source_reference_audit_sha256=reference_audit_sha,
        statistics_config=config,
        output=output,
        repository_root_value=workspace["root"],
    )
    verified = crossfit_schema.verify_stage2_context_package(
        output,
        published["context_sha256"],
        published["commit_sha256"],
        statistics_config=config,
        repository_root=workspace["root"],
    )
    return {
        "path": verified.path,
        "sha256": verified.context_sha256,
        "commit_path": verified.commit_path,
        "commit_sha256": verified.commit_sha256,
        "verified": verified,
    }


def _assert_no_residue(output: Path) -> None:
    assert not os.path.lexists(output)
    assert not os.path.lexists(output.parent / f".{output.name}.lock")
    assert not tuple(output.parent.glob(f".{output.name}.staging-*"))


def _published_attachment(workspace: Mapping[str, Any]) -> Any:
    label_path = workspace["output"] / "label-manifest.json"
    return verify_stage2_label_attachment(
        workspace["score_path"],
        label_path,
        workspace["role"],
        score_manifest_sha256=workspace["score_sha"],
        label_manifest_sha256=_sha(label_path),
        window_manifest=workspace["window_path"],
        window_manifest_sha256=workspace["window_sha"],
        window_id=workspace["window_id"],
        repository_root=workspace["root"],
    )


def test_incremental_dsu_rows_are_exactly_legacy_equivalent() -> None:
    rng = np.random.default_rng(20260717)
    event_values = np.asarray([0.0, 0.1, 0.2, 0.5, 0.9, 1.0], dtype=np.float64)
    probabilities: list[np.ndarray] = []
    targets = []
    for image_index in range(28):
        probability = rng.choice(event_values, size=(5, 7)).astype(np.float64)
        mask = (rng.random((5, 7)) > 0.86).astype(np.uint8)
        if image_index == 0:
            # Exercise diagonal 8-connectivity and one prediction bridging GTs.
            mask.fill(0)
            mask[0, 0] = 1
            mask[2, 2] = 1
            probability[0, 0] = 0.9
            probability[1, 1] = 0.9
            probability[2, 2] = 0.9
        probabilities.append(probability)
        targets.append(prepare_target(mask))

    sweep = _build_incremental_exact_sweep(probabilities, targets)
    legacy_rows: list[dict[str, float | int]] = []
    for threshold in sweep.thresholds:
        aggregate = aggregate_match_results(
            [
                match_components(probability > threshold, target, rule="overlap")
                for probability, target in zip(probabilities, targets, strict=True)
            ]
        )
        legacy_rows.append(
            {
                "threshold": float(threshold),
                "pd": float(aggregate["pd"]),
                "fa_pixel": float(aggregate["fa_pixel"]),
                "tp_objects": int(aggregate["tp_objects"]),
                "gt_objects": int(aggregate["gt_objects"]),
                "pred_components": int(aggregate["pred_components"]),
                "fp_components": int(aggregate["fp_components"]),
                "fp_pixels": int(aggregate["fp_pixels"]),
                "total_pixels": int(aggregate["total_pixels"]),
                "num_images": int(aggregate["num_images"]),
            }
        )
    assert list(sweep.rows) == legacy_rows
    canonical = json.dumps(
        legacy_rows,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    equivalence_digest = hashlib.sha256(canonical).hexdigest()
    assert sweep.rows_sha256 == equivalence_digest
    print(
        "DSU_EQUIVALENCE_EVIDENCE="
        + json.dumps(
            {
                "images": 28,
                "operating_points": len(sweep.rows),
                "equivalence_digest": equivalence_digest,
                "row_storage_bytes": sweep.rows.storage_nbytes,
                "legacy_and_dsu_rows_exactly_equal": True,
            },
            sort_keys=True,
        )
    )


def test_incremental_dsu_high_unique_q28_64x64_is_uncapped_and_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total_pixels = 28 * 64 * 64
    values = np.arange(1, total_pixels + 1, dtype=np.float64) / (total_pixels + 1)
    probabilities = [
        values[index * 4096 : (index + 1) * 4096].reshape(64, 64)
        for index in range(28)
    ]
    targets = []
    for _ in range(28):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[3, 3] = 1
        mask[50, 50] = 1
        targets.append(prepare_target(mask))

    legacy_calls = 0

    def forbidden_legacy_match(*args: Any, **kwargs: Any) -> Any:
        nonlocal legacy_calls
        legacy_calls += 1
        raise AssertionError("legacy match_components may not run per event")

    monkeypatch.setattr(component_matching, "match_components", forbidden_legacy_match)
    started = time.perf_counter()
    sweep = _build_incremental_exact_sweep(probabilities, targets)
    elapsed = time.perf_counter() - started
    assert isinstance(sweep.rows, ArrayBackedCurveRows)
    assert sweep.unique_event_count == total_pixels
    assert len(sweep.rows) == total_pixels + 2
    assert sweep.rows[0]["threshold"] == 0.0
    assert sweep.rows[-1]["threshold"] == 1.0
    expected_storage_bytes = (total_pixels + 2) * len(CURVE_FIELDS) * 8
    assert sweep.rows.storage_nbytes == expected_storage_bytes
    assert legacy_calls == 0
    assert elapsed <= 30.0
    print(
        "DSU_HIGH_UNIQUE_EVIDENCE="
        + json.dumps(
            {
                "environment": {
                    "python_implementation": platform.python_implementation(),
                    "python_version": platform.python_version(),
                    "numpy_version": np.__version__,
                    "machine": platform.machine(),
                    "logical_cpu_count": os.cpu_count(),
                },
                "images": 28,
                "height": 64,
                "width": 64,
                "unique_event_count": total_pixels,
                "operating_points": len(sweep.rows),
                "legacy_match_components_calls": legacy_calls,
                "rows_storage_bytes": sweep.rows.storage_nbytes,
                "rows_sha256": sweep.rows_sha256,
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        )
    )


def test_atomic_q28_bundle_exact_curve_and_misc111_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    original_resolver = exporter._resolve_query_mask_direct
    calls: list[str] = []

    def monitored_resolver(*args: Any, **kwargs: Any) -> Path:
        image_id = str(args[2])
        assert image_id not in workspace["context_ids"]
        assert image_id in workspace["query_ids"]
        calls.append(image_id)
        return original_resolver(*args, **kwargs)

    monkeypatch.setattr(exporter, "_resolve_query_mask_direct", monitored_resolver)
    result = _run(workspace)
    output = workspace["output"]
    assert result["query_labels"] == 28
    assert result["curve_operating_points"] == 3
    assert result["official_test_accessed"] is False
    assert calls == workspace["query_ids"]
    assert not os.path.lexists(output.parent / f".{output.name}.lock")
    assert not tuple(output.parent.glob(f".{output.name}.staging-*"))

    base = [path for path in output.iterdir() if not path.name.endswith(".sha256")]
    sidecars = [path for path in output.iterdir() if path.name.endswith(".sha256")]
    assert len(base) == 32  # Q28 NPZ + label manifest + curve + curve manifest + audit
    assert len(sidecars) == len(base)
    for artifact in base:
        sidecar = output / f"{artifact.name}.sha256"
        assert sidecar.read_text(encoding="utf-8") == f"{_sha(artifact)}  {artifact.name}\n"

    label_manifest_path = output / "label-manifest.json"
    attachment = verify_stage2_label_attachment(
        workspace["score_path"],
        label_manifest_path,
        SOURCE_DIAGNOSTIC_VALIDATION,
        score_manifest_sha256=workspace["score_sha"],
        label_manifest_sha256=_sha(label_manifest_path),
        window_manifest=workspace["window_path"],
        window_manifest_sha256=workspace["window_sha"],
        window_id=workspace["window_id"],
        repository_root=workspace["root"],
    )
    misc = next(item for item in attachment.items if item.image_id == "Misc_111")
    assert attachment.payload["decision_seal_binding"] is None
    provenance = misc.record["alignment_provenance"]
    assert provenance["operation"] == "resize_mask_to_image_geometry"
    assert provenance["interpolation"] == "nearest_neighbor"
    assert provenance["nuaa_misc_111_policy_applied"] is False  # NUDT synthetic identity
    assert provenance["silent_crop_used"] is False
    assert provenance["bilinear_resize_used"] is False

    curve_manifest, rows = verify_stage2_query_curve_artifacts(
        output / "query-curve.csv",
        output / "curve-manifest.json",
        curve_sha256=_sha(output / "query-curve.csv"),
        curve_manifest_sha256=_sha(output / "curve-manifest.json"),
        attachment=attachment,
        repository_root=workspace["root"],
    )
    assert [row["threshold"] for row in rows] == [0.0, 0.25, 1.0]
    assert rows[1]["pred_components"] == 0  # strict probability > threshold
    assert rows[1]["fp_pixels"] == 0
    assert all(row["total_pixels"] == curve_manifest["total_native_pixels"] for row in rows)
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["decision_seal_binding"] is None
    assert audit["context_mask_paths_resolved_statted_or_opened"] == 0
    assert audit["query_mask_paths_resolved"] == 28
    assert audit["event_threshold_cap"] is None
    assert audit["exact_sweep_legacy_match_components_calls"] == 0
    assert audit["curve_rows_storage_bytes"] == 3 * len(CURVE_FIELDS) * 8
    assert audit["curve_rows_sha256"] == curve_manifest["curve_rows_sha256"]


def test_window_and_order_are_verified_before_any_mask_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)

    def forbidden(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("mask resolver must not run before window acceptance")

    monkeypatch.setattr(exporter, "_resolve_query_mask_direct", forbidden)
    with pytest.raises(KeyError, match="window_id"):
        export_stage2_labels(
            score_manifest=workspace["score_path"],
            score_manifest_sha256=workspace["score_sha"],
            window_manifest=workspace["window_path"],
            window_manifest_sha256=workspace["window_sha"],
            window_id="forged-window",
            expected_role=SOURCE_DIAGNOSTIC_VALIDATION,
            dataset_dir=workspace["dataset"],
            output_dir=workspace["output"],
            repository_root=workspace["root"],
        )
    _assert_no_residue(workspace["output"])


def test_mask_aspect_failure_rolls_back_all_bundle_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    bad_id = workspace["query_ids"][1]
    Image.fromarray(np.zeros((2, 19), dtype=np.uint8), mode="L").save(
        workspace["dataset"] / "masks" / f"{bad_id}.png"
    )
    with pytest.raises(ValueError, match="aspect-ratio mismatch"):
        _run(workspace)
    _assert_no_residue(workspace["output"])




def test_outer_target_role_opens_only_after_complete_bound_t0_t8_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(
        tmp_path,
        monkeypatch,
        role=OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
    )
    original_resolver = exporter._resolve_query_mask_direct
    resolver_calls: list[str] = []

    def monitored(*args: Any, **kwargs: Any) -> Path:
        resolver_calls.append(str(args[2]))
        return original_resolver(*args, **kwargs)

    monkeypatch.setattr(exporter, "_resolve_query_mask_direct", monitored)
    with pytest.raises(RuntimeError, match="hard-HOLD"):
        _run(workspace)
    assert resolver_calls == []
    _assert_no_residue(workspace["output"])

    _seal_outer_decision_set(workspace)
    with pytest.raises(ValueError, match="external SHA-256 mismatch"):
        export_stage2_labels(
            score_manifest=workspace["score_path"],
            score_manifest_sha256=workspace["score_sha"],
            window_manifest=workspace["window_path"],
            window_manifest_sha256=workspace["window_sha"],
            window_id=workspace["window_id"],
            expected_role=workspace["role"],
            dataset_dir=workspace["dataset"],
            output_dir=workspace["output"],
            sealed_decision_set=workspace["decision_set_path"],
            sealed_decision_set_sha256="0" * 64,
            statistics_config=workspace["statistics_config_path"],
            statistics_config_sha256=workspace["statistics_config_sha256"],
            repository_root=workspace["root"],
        )
    assert resolver_calls == []
    _assert_no_residue(workspace["output"])

    with pytest.raises(ValueError, match="context package.*does not exist"):
        _run(workspace)
    assert resolver_calls == []
    _assert_no_residue(workspace["output"])

    context_binding = _publish_real_bound_context_package(workspace, monkeypatch)
    expected_binding = _seal_outer_decision_set(
        workspace,
        context_binding=context_binding,
        output_name="sealed-valid-t0-t8",
    )
    result = _run(workspace)
    assert result["query_labels"] == 28
    assert resolver_calls == workspace["query_ids"]
    attachment = _published_attachment(workspace)
    assert attachment.payload["decision_seal_binding"] == expected_binding
    misc = next(item for item in attachment.items if item.image_id == "Misc_111")
    provenance = misc.record["alignment_provenance"]
    assert provenance["operation"] == "resize_mask_to_image_geometry"
    assert provenance["interpolation"] == "nearest_neighbor"
    assert provenance["nuaa_misc_111_policy_applied"] is True
    curve_manifest = json.loads(
        (workspace["output"] / "curve-manifest.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (workspace["output"] / "audit.json").read_text(encoding="utf-8")
    )
    assert curve_manifest["decision_seal_binding"] == expected_binding
    assert audit["decision_seal_binding"] == expected_binding


def test_source_role_rejects_nonnull_decision_set_before_any_mask_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)

    def forbidden(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("source mask resolver ran despite invalid decision-set input")

    monkeypatch.setattr(exporter, "_resolve_query_mask_direct", forbidden)
    with pytest.raises(ValueError, match="require sealed_decision_set.*null"):
        export_stage2_labels(
            score_manifest=workspace["score_path"],
            score_manifest_sha256=workspace["score_sha"],
            window_manifest=workspace["window_path"],
            window_manifest_sha256=workspace["window_sha"],
            window_id=workspace["window_id"],
            expected_role=workspace["role"],
            dataset_dir=workspace["dataset"],
            output_dir=workspace["output"],
            sealed_decision_set="forged/not-allowed-for-source.json",
            sealed_decision_set_sha256="0" * 64,
            repository_root=workspace["root"],
        )
    _assert_no_residue(workspace["output"])

    config_path = workspace["root"] / "synthetic-source-statistics-config.json"
    _write_json(
        config_path,
        StatisticsConfig(
            peak_kernel_size=3,
            peak_min_score=0.05,
            quantile_sample_limit=128,
        ).to_dict(),
    )
    with pytest.raises(ValueError, match="require sealed_decision_set.*null"):
        export_stage2_labels(
            score_manifest=workspace["score_path"],
            score_manifest_sha256=workspace["score_sha"],
            window_manifest=workspace["window_path"],
            window_manifest_sha256=workspace["window_sha"],
            window_id=workspace["window_id"],
            expected_role=workspace["role"],
            dataset_dir=workspace["dataset"],
            output_dir=workspace["output"],
            statistics_config=config_path,
            statistics_config_sha256=_sha(config_path),
            repository_root=workspace["root"],
        )
    _assert_no_residue(workspace["output"])



def test_oof_holdout_role_binds_fold_and_query_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(
        tmp_path, monkeypatch, role=OOF_HOLDOUT_STAGE2_FIT
    )
    result = _run(workspace)
    assert result["role"] == OOF_HOLDOUT_STAGE2_FIT
    payload = json.loads(
        (workspace["output"] / "label-manifest.json").read_text(encoding="utf-8")
    )
    assert payload["detector_role"] == "detector_oof"
    assert payload["oof_fold_index"] == 0
    assert payload["query_size"] == 28

def test_mid_write_failure_has_zero_residual_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    original = exporter._write_npz_exclusive
    count = 0

    def fail_after_two(path: Path, **arrays: np.ndarray) -> None:
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("synthetic producer failure")
        original(path, **arrays)

    monkeypatch.setattr(exporter, "_write_npz_exclusive", fail_after_two)
    with pytest.raises(RuntimeError, match="synthetic producer failure"):
        _run(workspace)
    _assert_no_residue(workspace["output"])


@pytest.mark.parametrize("bad_value", [0, 1, "false", None])
def test_label_v2_official_test_guard_is_exact_boolean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_value: object,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    _run(workspace)
    manifest_path = workspace["output"] / "label-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["official_test_accessed"] = bad_value
    _write_json(manifest_path, payload)
    with pytest.raises((TypeError, ValueError), match="official_test_accessed"):
        verify_stage2_label_attachment(
            workspace["score_path"],
            manifest_path,
            SOURCE_DIAGNOSTIC_VALIDATION,
            score_manifest_sha256=workspace["score_sha"],
            label_manifest_sha256=_sha(manifest_path),
            window_manifest=workspace["window_path"],
            window_manifest_sha256=workspace["window_sha"],
            window_id=workspace["window_id"],
            repository_root=workspace["root"],
        )


def test_no_replace_refuses_existing_output_without_modifying_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    workspace["output"].mkdir()
    sentinel = workspace["output"] / "owner.txt"
    sentinel.write_text("user-owned\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        _run(workspace)
    assert sentinel.read_text(encoding="utf-8") == "user-owned\n"
    assert not os.path.lexists(workspace["output"].parent / f".{workspace['output'].name}.lock")


def test_tampered_query_source_mask_is_rejected_by_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    _run(workspace)
    first_id = workspace["query_ids"][0]
    mask_path = workspace["dataset"] / "masks" / f"{first_id}.png"
    Image.fromarray(np.full((6, 8), 255, dtype=np.uint8), mode="L").save(mask_path)
    label_path = workspace["output"] / "label-manifest.json"
    with pytest.raises((ValueError, RuntimeError), match="source mask"):
        verify_stage2_label_attachment(
            workspace["score_path"],
            label_path,
            SOURCE_DIAGNOSTIC_VALIDATION,
            score_manifest_sha256=workspace["score_sha"],
            label_manifest_sha256=_sha(label_path),
            window_manifest=workspace["window_path"],
            window_manifest_sha256=workspace["window_sha"],
            window_id=workspace["window_id"],
            repository_root=workspace["root"],
        )


def test_direct_mask_resolution_rejects_all_ambiguous_query_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    image_id = workspace["query_ids"][0]
    masks = workspace["dataset"] / "masks"
    shutil.copy2(masks / f"{image_id}.png", masks / f"{image_id}_pixels0.png")
    with pytest.raises(ValueError, match="Ambiguous direct query mask"):
        _run(workspace)
    _assert_no_residue(workspace["output"])


def test_direct_mask_resolution_never_recurses_into_decoy_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    image_id = workspace["query_ids"][1]
    masks = workspace["dataset"] / "masks"
    decoy = masks / "recursive-decoy"
    decoy.mkdir()
    (masks / f"{image_id}.png").rename(decoy / f"{image_id}.png")
    with pytest.raises(FileNotFoundError, match="No direct query mask"):
        _run(workspace)
    _assert_no_residue(workspace["output"])


def test_direct_mask_resolution_rejects_symlink_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    masks = workspace["dataset"] / "masks"
    victim = masks / f"{workspace['query_ids'][1]}.png"
    target = masks / f"{workspace['query_ids'][2]}.png"
    victim.unlink()
    victim.symlink_to(target.name)
    with pytest.raises(ValueError, match="symlink"):
        _run(workspace)
    _assert_no_residue(workspace["output"])


def test_lock_write_failure_has_zero_transaction_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)

    def fail_write(*args: Any, **kwargs: Any) -> None:
        raise OSError("synthetic lock write failure")

    monkeypatch.setattr(exporter, "_write_fd_all", fail_write)
    with pytest.raises(OSError, match="synthetic lock write failure"):
        _run(workspace)
    _assert_no_residue(workspace["output"])


def test_lock_fsync_failure_has_zero_transaction_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    original = exporter.os.fsync
    calls = 0

    def fail_once(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic lock fsync failure")
        original(descriptor)

    monkeypatch.setattr(exporter.os, "fsync", fail_once)
    with pytest.raises(OSError, match="synthetic lock fsync failure"):
        _run(workspace)
    _assert_no_residue(workspace["output"])


def test_marker_unlink_failure_rolls_back_every_bundle_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    original = Path.unlink
    failed = False

    def fail_marker_once(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal failed
        if path.name == ".bundle_incomplete" and not failed:
            failed = True
            raise OSError("synthetic marker unlink failure")
        original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_marker_once)
    with pytest.raises(OSError, match="synthetic marker unlink failure"):
        _run(workspace)
    _assert_no_residue(workspace["output"])


def test_rename_success_then_exception_is_detected_and_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)

    def rename_then_fail(source: Path, target: Path, identity: tuple[int, int]) -> tuple[int, int]:
        os.rename(source, target)
        raise OSError("synthetic post-rename failure")

    monkeypatch.setattr(exporter, "_rename_directory_no_replace", rename_then_fail)
    with pytest.raises(OSError, match="synthetic post-rename failure"):
        _run(workspace)
    _assert_no_residue(workspace["output"])


def test_parent_fsync_failure_after_rename_rolls_back_published_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    original = exporter._fsync_directory
    calls = 0

    def fail_post_rename_once(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("synthetic post-rename parent fsync failure")
        original(path)

    monkeypatch.setattr(exporter, "_fsync_directory", fail_post_rename_once)
    with pytest.raises(OSError, match="synthetic post-rename parent fsync failure"):
        _run(workspace)
    _assert_no_residue(workspace["output"])


def test_postpublication_verifier_failure_rolls_back_published_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    original = exporter.verify_stage2_query_curve_artifacts
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic public verifier failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(exporter, "verify_stage2_query_curve_artifacts", fail_second)
    with pytest.raises(RuntimeError, match="synthetic public verifier failure"):
        _run(workspace)
    _assert_no_residue(workspace["output"])


def test_late_unowned_target_race_is_preserved_while_owned_state_is_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    original = exporter._rename_directory_no_replace

    def install_unowned_target(
        source: Path,
        target: Path,
        identity: tuple[int, int],
    ) -> tuple[int, int]:
        target.mkdir()
        (target / "owner.txt").write_text("unowned\n", encoding="utf-8")
        return original(source, target, identity)

    monkeypatch.setattr(exporter, "_rename_directory_no_replace", install_unowned_target)
    with pytest.raises(BaseExceptionGroup):
        _run(workspace)
    assert (workspace["output"] / "owner.txt").read_text(encoding="utf-8") == "unowned\n"
    assert not os.path.lexists(workspace["output"].parent / f".{workspace['output'].name}.lock")
    assert not tuple(workspace["output"].parent.glob(f".{workspace['output'].name}.staging-*"))


def test_persistent_first_finalize_lock_errors_trigger_full_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    original = exporter._release_lock
    calls = 0

    def fail_first_finalize(path: Path, identity: tuple[int, int]) -> bool:
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise OSError("synthetic lock release failure")
        return original(path, identity)

    monkeypatch.setattr(exporter, "_release_lock", fail_first_finalize)
    with pytest.raises(BaseExceptionGroup, match="finalization"):
        _run(workspace)
    _assert_no_residue(workspace["output"])


def test_public_consumers_reject_active_bundle_lock_before_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    _run(workspace)
    attachment = _published_attachment(workspace)
    output = workspace["output"]
    lock = output.parent / f".{output.name}.lock"
    lock.write_text("foreign in-flight publication\n", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="publication lock is active"):
            _published_attachment(workspace)
        with pytest.raises(RuntimeError, match="publication lock is active"):
            verify_stage2_query_curve_artifacts(
                output / "query-curve.csv",
                output / "curve-manifest.json",
                curve_sha256=_sha(output / "query-curve.csv"),
                curve_manifest_sha256=_sha(output / "curve-manifest.json"),
                attachment=attachment,
                repository_root=workspace["root"],
            )
    finally:
        lock.unlink()


def test_public_curve_consumer_rejects_path_and_self_consistent_value_rewrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _synthetic_workspace(tmp_path, monkeypatch)
    _run(workspace)
    output = workspace["output"]
    attachment = _published_attachment(workspace)
    curve_path = output / "query-curve.csv"
    manifest_path = output / "curve-manifest.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    path_tamper = deepcopy(original_manifest)
    path_tamper["label_manifest_binding"]["path"] = "forged/label-manifest.json"
    _write_json(manifest_path, path_tamper)
    with pytest.raises(ValueError, match="path binding mismatch"):
        verify_stage2_query_curve_artifacts(
            curve_path,
            manifest_path,
            curve_sha256=_sha(curve_path),
            curve_manifest_sha256=_sha(manifest_path),
            attachment=attachment,
            repository_root=workspace["root"],
        )

    rows = [dict(row) for row in _read_curve_csv(curve_path)]
    rows[0]["pd"] = float(rows[0]["pd"]) + 0.125
    with curve_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CURVE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    value_tamper = deepcopy(original_manifest)
    value_tamper["curve_sha256"] = _sha(curve_path)
    value_tamper["curve_rows_sha256"] = stage2_curve_rows_sha256(rows)
    _write_json(manifest_path, value_tamper)
    with pytest.raises(ValueError, match="exact native-resolution replay"):
        verify_stage2_query_curve_artifacts(
            curve_path,
            manifest_path,
            curve_sha256=_sha(curve_path),
            curve_manifest_sha256=_sha(manifest_path),
            attachment=attachment,
            repository_root=workspace["root"],
        )

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

import rc.build_stage2_source_reference as module
from data_ext.stage2_score_manifest import (
    FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE,
    OOF_TRAIN_SOURCE_REFERENCE,
)
from rc.domain_statistics import BASE_FEATURE_DIM, load_source_reference
from rc.schema import StatisticsConfig


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha(path)


def _write_json(path: Path, payload: object) -> str:
    return _write(
        path,
        (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )


def _identity(domain: str, index: int, prefix: str) -> dict[str, Any]:
    image_id = f"{prefix}_{index:03d}"
    digest = hashlib.sha256(f"{domain}:{image_id}".encode()).hexdigest()
    return {
        "canonical_id": f"{domain}::{image_id}",
        "image_id": image_id,
        "original_image_path": f"development_images/{domain.lower()}/{image_id}.png",
        "original_image_sha256": digest,
        "near_duplicate_cluster_id_or_unique_sentinel": f"UNIQUE::{domain}::{image_id}",
        "exclusion_group_id": f"SINGLETON::{domain}::{image_id}::{digest}",
        "source_role_record_index": index,
    }


def _window_record(
    domain: str,
    index: int,
    partition: str,
    *,
    outer_fold: str,
    source_role: str,
    oof_fold_index: int | None,
) -> dict[str, Any]:
    record = {
        **_identity(domain, index, "consumer"),
        "episode_role": partition,
        "outer_fold_id": outer_fold,
        "source_role": source_role,
    }
    if oof_fold_index is not None:
        record["oof_fold_index"] = oof_fold_index
    return record


def _guardrails() -> dict[str, bool]:
    return {
        "development_only": True,
        "execution_authorized": False,
        "mask_or_label_files_opened": False,
        "official_test_ids_materialized": False,
        "official_test_images_opened": False,
        "official_test_split_files_opened": False,
        "original_training_images_opened_only_for_sha256": True,
        "predictions_scores_checkpoints_or_metrics_opened": False,
        "result_free": True,
    }


def _bound_file(root: Path, name: str) -> dict[str, str]:
    path = root / "contract_inputs" / f"{name}.json"
    digest = _write_json(path, {"synthetic": name})
    return {"path": path.relative_to(root).as_posix(), "sha256": digest}


def _window_fixture(
    root: Path,
    *,
    domain: str,
    episode_role: str,
    detector_role: str,
    oof_fold_index: int | None,
    outer_target: str = "NUAA-SIRST",
) -> tuple[Path, str, dict[str, Any], dict[str, Any]]:
    outer_fold = "outer_leave_nuaa_sirst"
    source_role = "detector_fit" if detector_role == "detector_oof" else "detector_diagnostic"
    role_binding = _bound_file(root, f"{domain}-{episode_role}-role")
    unused = _bound_file(root, f"{domain}-{episode_role}-unused")
    bound_inputs = {
        name: _bound_file(root, name)
        for name in sorted(module._WINDOW_BOUND_INPUT_NAMES)
    }
    records = [
        _window_record(
            domain,
            index,
            "context" if index < 14 else "query",
            outer_fold=outer_fold,
            source_role=source_role,
            oof_fold_index=oof_fold_index,
        )
        for index in range(42)
    ]
    payload = {
        "schema_version": "rc-irstd.stage2-role-pure-c14q28-windows.v1",
        "artifact_type": "rc_irstd_stage2_role_pure_episode_windows",
        "artifact_status": "DEVELOPMENT_ONLY_RESULT_FREE",
        "bound_inputs": bound_inputs,
        "complete_window_count": 1,
        "domain": domain,
        "episode_role": episode_role,
        "execution_authorized": False,
        "geometry": {
            "block_size": 42,
            "construction": "ordered_non_overlapping_contiguous_blocks_context_first_query_second",
            "context_size": 14,
            "query_size": 28,
        },
        "guardrails": _guardrails(),
        "observed_results": None,
        "oof_fold_index": oof_fold_index,
        "ordered_role_record_count": 42,
        "outer_fold_id": outer_fold,
        "outer_target_domain": outer_target,
        "role_binding": role_binding,
        "role_purity": {
            "allowed_source_role": source_role,
            "mixed_roles_allowed": False,
            "single_oof_checkpoint_identity_per_fit_window_required_at_future_score_binding": detector_role
            == "detector_oof",
            "single_source_domain_per_window": True,
        },
        "source_role": source_role,
        "unused_suffix": unused,
        "window_record_count": 42,
        "windows": [
            {
                "window_index": 0,
                "window_id": f"{outer_fold}::{episode_role}::{domain}::window_000",
                "context_records": records[:14],
                "query_records": records[14:],
            }
        ],
    }
    path = root / "windows" / f"{domain}-{episode_role}.json"
    digest = _write_json(path, payload)
    materialization = {
        path.relative_to(root).as_posix(): digest,
        role_binding["path"]: role_binding["sha256"],
        unused["path"]: unused["sha256"],
    }
    run = {
        "run_id": "synthetic-run",
        "outer_fold_id": outer_fold,
        "outer_target_domain": outer_target,
        "source_domains": ["NUDT-SIRST", "IRSTD-1K"],
        "bindings": {"materialization_artifacts_sha256": materialization},
    }
    return path, digest, payload, run


def _rebind_window(path: Path, payload: Mapping[str, Any], run: dict[str, Any], root: Path) -> str:
    digest = _write_json(path, payload)
    run["bindings"]["materialization_artifacts_sha256"][
        path.relative_to(root).as_posix()
    ] = digest
    return digest


def test_frozen_policy_bindings_are_exact_result_free_and_official_sealed() -> None:
    bindings, rechecks = module._policy_contract(ROOT)
    assert bindings == module._POLICY_BINDINGS
    assert len(rechecks) == 3
    assert all(_sha(path) == digest for path, digest in rechecks)


@pytest.mark.parametrize(
    ("domain", "episode_role", "detector_role", "oof_fold"),
    [
        ("NUDT-SIRST", "stage2_oof_fit", "detector_oof", 0),
        (
            "NUDT-SIRST",
            "source_diagnostic_validation",
            "detector_full_fit",
            None,
        ),
        (
            "NUAA-SIRST",
            "outer_target_diagnostic_development",
            "detector_full_fit",
            None,
        ),
    ],
)
def test_exact_c14q28_consumer_window_contract_accepts_only_expected_roles(
    tmp_path: Path,
    domain: str,
    episode_role: str,
    detector_role: str,
    oof_fold: int | None,
) -> None:
    path, digest, _, run = _window_fixture(
        tmp_path,
        domain=domain,
        episode_role=episode_role,
        detector_role=detector_role,
        oof_fold_index=oof_fold,
    )
    rechecks: list[tuple[Path, str]] = []
    verified = module._verify_window_manifest(
        path,
        digest,
        root=tmp_path,
        run_contract=run,
        detector_role=detector_role,
        oof_fold_index=oof_fold,
        rechecks=rechecks,
    )
    assert verified["record_count"] == 42
    assert verified["complete_window_count"] == 1
    assert len(rechecks) == 6


@pytest.mark.parametrize(
    "mutation",
    [
        "boolean_type_confusion",
        "oof_type_confusion",
        "record_oof_type_confusion",
        "geometry",
        "forbidden_role",
        "context_query_overlap",
        "extra_field",
    ],
)
def test_window_schema_boolean_geometry_role_and_four_boundary_mutations_fail(
    tmp_path: Path, mutation: str
) -> None:
    path, _, payload, run = _window_fixture(
        tmp_path,
        domain="NUDT-SIRST",
        episode_role="stage2_oof_fit",
        detector_role="detector_oof",
        oof_fold_index=0,
    )
    payload = json.loads(json.dumps(payload))
    if mutation == "boolean_type_confusion":
        payload["guardrails"]["official_test_images_opened"] = 0
    elif mutation == "oof_type_confusion":
        payload["oof_fold_index"] = False
    elif mutation == "record_oof_type_confusion":
        payload["windows"][0]["context_records"][0]["oof_fold_index"] = False
    elif mutation == "geometry":
        payload["geometry"]["context_size"] = 13
    elif mutation == "forbidden_role":
        payload["episode_role"] = "oof_holdout_source_reference"
    elif mutation == "context_query_overlap":
        payload["windows"][0]["query_records"][0]["original_image_sha256"] = (
            payload["windows"][0]["context_records"][0]["original_image_sha256"]
        )
    else:
        payload["unexpected"] = None
    digest = _rebind_window(path, payload, run, tmp_path)
    with pytest.raises((TypeError, ValueError)):
        module._verify_window_manifest(
            path,
            digest,
            root=tmp_path,
            run_contract=run,
            detector_role="detector_oof",
            oof_fold_index=0,
            rechecks=[],
        )


def test_window_path_symlink_and_external_hash_are_fail_closed(tmp_path: Path) -> None:
    path, digest, _, run = _window_fixture(
        tmp_path,
        domain="NUDT-SIRST",
        episode_role="stage2_oof_fit",
        detector_role="detector_oof",
        oof_fold_index=0,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        module._verify_window_manifest(
            path,
            "0" * 64,
            root=tmp_path,
            run_contract=run,
            detector_role="detector_oof",
            oof_fold_index=0,
            rechecks=[],
        )
    symlink = tmp_path / "windows" / "symlink.json"
    symlink.symlink_to(path.name)
    with pytest.raises(ValueError, match="symlink"):
        module._existing_file(symlink, tmp_path, "consumer window")
    assert digest == _sha(path)


def test_identity_audit_rejects_each_reference_to_consumer_boundary() -> None:
    base_reference = _identity("NUDT-SIRST", 0, "reference")
    manifest = SimpleNamespace(records=(base_reference,))
    for field in module.IDENTITY_BOUNDARY_FIELDS:
        consumer_record = _identity("NUDT-SIRST", 1, "consumer")
        consumer_record[field] = base_reference[field]
        consumer = {
            "path": "windows/consumer.json",
            "sha256": "a" * 64,
            "domain": "NUDT-SIRST",
            "episode_role": "stage2_oof_fit",
            "complete_window_count": 1,
            "record_count": 1,
            "records": (consumer_record,),
        }
        with pytest.raises(ValueError, match="identity overlap"):
            module._identity_audit((manifest,), (consumer,))


def _fake_build_contract(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    detector_role: str = "detector_full_fit",
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"
    score_paths = [inputs / "source-a.json", inputs / "source-b.json"]
    score_hashes = [_write_json(path, {"synthetic": index}) for index, path in enumerate(score_paths)]
    checkpoint = inputs / "checkpoint.pt"
    checkpoint_sha = _write(checkpoint, b"synthetic restricted checkpoint placeholder")
    config = StatisticsConfig(
        peak_kernel_size=3,
        peak_min_score=0.05,
        quantile_sample_limit=128,
    )
    config_path = inputs / "statistics.json"
    config_sha = _write_json(config_path, config.to_dict())
    source_domains = ["NUDT-SIRST", "IRSTD-1K"]
    outer_target = "NUAA-SIRST"
    role = (
        OOF_TRAIN_SOURCE_REFERENCE
        if detector_role == "detector_oof"
        else FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE
    )
    oof_index = 0 if detector_role == "detector_oof" else None
    run_path = inputs / "run.json"
    run_sha = _write_json(run_path, {"synthetic": "run"})
    run = {
        "run_id": "outer_leave_nuaa_sirst__s42__synthetic",
        "outer_fold_id": "outer_leave_nuaa_sirst",
        "outer_target_domain": outer_target,
        "source_domains": source_domains,
        "bindings": {"materialization_artifacts_sha256": {}},
    }
    manifests = []
    for index, domain in enumerate(source_domains):
        record = _identity(domain, index, "reference")
        manifests.append(
            SimpleNamespace(
                path=score_paths[index],
                manifest_sha256=score_hashes[index],
                records_content_sha256=hashlib.sha256(domain.encode()).hexdigest(),
                records=(record,),
                items=(),
                payload={
                    "source_domain": domain,
                    "outer_fold_id": "outer_leave_nuaa_sirst",
                    "outer_target": outer_target,
                    "base_seed": 42,
                    "derived_seed": 101,
                    "detector_role": detector_role,
                    "oof_fold_index": oof_index,
                },
                bindings={
                    "run_contract": {
                        "path": run_path.relative_to(root).as_posix(),
                        "sha256": run_sha,
                    },
                    "selection_contract": {
                        "path": f"inputs/selection-{index}.json",
                        "sha256": hashlib.sha256(f"selection-{index}".encode()).hexdigest(),
                    },
                },
            )
        )
    consumer_roles = (
        [(domain, "stage2_oof_fit") for domain in source_domains]
        if detector_role == "detector_oof"
        else [
            (source_domains[0], "source_diagnostic_validation"),
            (source_domains[1], "source_diagnostic_validation"),
            (outer_target, "outer_target_diagnostic_development"),
        ]
    )
    consumer_paths = []
    consumer_hashes = []
    consumers = []
    for index, (domain, episode_role) in enumerate(consumer_roles):
        path = inputs / f"consumer-{index}.json"
        digest = _write_json(
            path,
            {
                "synthetic": index,
                "windows": [{"window_id": f"consumer-window-{index}"}],
            },
        )
        consumer_paths.append(path)
        consumer_hashes.append(digest)
        consumers.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "domain": domain,
                "episode_role": episode_role,
                "complete_window_count": 1,
                "record_count": 1,
                "records": (_identity(domain, index + 10, "consumer"),),
            }
        )

    monkeypatch.setattr(
        module,
        "_policy_contract",
        lambda _root: (dict(module._POLICY_BINDINGS), []),
    )

    def fake_manifests(*args: Any, **kwargs: Any):
        del args, kwargs
        return (
            tuple(manifests),
            run,
            role,
            list(zip(score_paths, score_hashes)) + [(run_path, run_sha)],
        )

    monkeypatch.setattr(module, "_verify_reference_manifests", fake_manifests)
    monkeypatch.setattr(
        module,
        "_verify_all_consumers",
        lambda *args, **kwargs: tuple(consumers),
    )
    centers = np.vstack(
        [
            np.linspace(0.0, 1.0, BASE_FEATURE_DIM, dtype=np.float32),
            np.linspace(1.0, 2.0, BASE_FEATURE_DIM, dtype=np.float32),
        ]
    )
    scale = np.full(BASE_FEATURE_DIM, 0.5, dtype=np.float32)
    monkeypatch.setattr(
        module,
        "_compute_centers",
        lambda *args, **kwargs: (
            centers,
            scale,
            {
                domain: {
                    "record_count": 1,
                    "num_images": 1,
                    "num_pixels": 16,
                    "num_peaks": 1,
                    "has_grayscale": True,
                }
                for domain in source_domains
            },
        ),
    )
    return {
        "score_paths": score_paths,
        "score_hashes": score_hashes,
        "checkpoint": checkpoint,
        "checkpoint_sha": checkpoint_sha,
        "config": config,
        "config_path": config_path,
        "config_sha": config_sha,
        "consumer_paths": consumer_paths,
        "consumer_hashes": consumer_hashes,
        "centers": centers,
        "scale": scale,
    }


def _invoke_fake_build(root: Path, fixture: Mapping[str, Any], output: Path) -> dict[str, Any]:
    return module.build_stage2_source_reference(
        fixture["score_paths"],
        fixture["score_hashes"],
        fixture["checkpoint"],
        fixture["checkpoint_sha"],
        fixture["config_path"],
        fixture["config_sha"],
        fixture["consumer_paths"],
        fixture["consumer_hashes"],
        output,
        repository_root=root,
    )


def _verify_fake_bundle(root: Path, output: Path) -> module.VerifiedStage2SourceReference:
    audit_path = output.with_suffix(".audit.json")
    return module.verify_stage2_source_reference(
        output,
        _sha(output),
        _sha(audit_path),
        statistics_config=StatisticsConfig(
            peak_kernel_size=3,
            peak_min_score=0.05,
            quantile_sample_limit=128,
        ),
        repository_root=root,
    )


@pytest.mark.parametrize("detector_role", ["detector_oof", "detector_full_fit"])
def test_build_publishes_exact_four_file_no_replace_bundle_and_legacy_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detector_role: str,
) -> None:
    fixture = _fake_build_contract(tmp_path, monkeypatch, detector_role=detector_role)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    output = output_dir / "source-reference.npz"
    audit = _invoke_fake_build(tmp_path, fixture, output)
    audit_path = output.with_suffix(".audit.json")
    expected = {
        output,
        audit_path,
        output.with_name(output.name + ".sha256"),
        audit_path.with_name(audit_path.name + ".sha256"),
    }
    assert set(output_dir.iterdir()) == expected
    assert audit["official_test_accessed"] is False
    assert audit["labels_or_masks_opened"] is False
    assert audit["identity_boundary_audit"]["all_four_boundaries_zero_overlap"] is True
    with np.load(output, allow_pickle=False) as payload:
        assert tuple(payload.files) == module._NPZ_FIELD_ORDER
        assert str(payload["schema_version"].item()) == module.STAGE2_SOURCE_REFERENCE_SCHEMA
        np.testing.assert_array_equal(payload["centers"], fixture["centers"])
        np.testing.assert_array_equal(payload["scale"], fixture["scale"])
    loaded = load_source_reference(output, statistics_config=fixture["config"])
    assert loaded.domains == ("NUDT-SIRST", "IRSTD-1K")
    verified = _verify_fake_bundle(tmp_path, output)
    assert verified.source_reference == loaded
    assert verified.audit["output"] == audit["output"]
    assert verified.path == output
    assert verified.npz_sha256 == _sha(output)
    assert verified.npz_sha == verified.npz_sha256
    assert verified.audit_path == audit_path
    assert verified.audit_sha256 == _sha(audit_path)
    assert verified.audit_sha == verified.audit_sha256
    assert verified.domains == loaded.domains
    assert verified.centers == loaded.centers
    assert verified.scale == loaded.scale
    assert verified.reference_role == audit["reference_role"]
    assert verified.detector_identity == audit["detector_identity"]
    assert verified.detector is verified.detector_identity
    assert verified.checkpoint_binding == audit["bindings"]["checkpoint"]
    assert verified.checkpoint is verified.checkpoint_binding
    assert tuple(dict(item) for item in verified.consumer_bindings) == tuple(
        audit["bindings"]["consumer_window_manifests"]
    )
    with pytest.raises(TypeError, match="only be created"):
        module.VerifiedStage2SourceReference()
    with pytest.raises(TypeError, match="only be created"):
        replace(verified, npz_sha256="0" * 64)
    with pytest.raises(TypeError):
        verified.stage2_contract["reference_role"] = "forged"
    assert output.with_name(output.name + ".sha256").read_text() == f"{_sha(output)}  {output.name}\n"
    assert audit_path.with_name(audit_path.name + ".sha256").read_text() == (
        f"{_sha(audit_path)}  {audit_path.name}\n"
    )
    with pytest.raises(FileExistsError, match="already exists"):
        _invoke_fake_build(tmp_path, fixture, output)


def test_public_verifier_optionally_rebinds_exact_consumer_window_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fake_build_contract(tmp_path, monkeypatch)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    output = output_dir / "consumer-bound.npz"
    _invoke_fake_build(tmp_path, fixture, output)
    audit_path = output.with_suffix(".audit.json")
    verified = module.verify_stage2_source_reference(
        output,
        _sha(output),
        _sha(audit_path),
        statistics_config=fixture["config"],
        expected_consumer_window_path=fixture["consumer_paths"][0],
        expected_consumer_window_sha256=fixture["consumer_hashes"][0],
        expected_consumer_window_id="consumer-window-0",
        repository_root=tmp_path,
    )
    assert verified.consumer_bindings[0]["path"] == fixture[
        "consumer_paths"
    ][0].relative_to(tmp_path).as_posix()
    with pytest.raises(ValueError, match="exactly once"):
        module.verify_stage2_source_reference(
            output,
            _sha(output),
            _sha(audit_path),
            statistics_config=fixture["config"],
            expected_consumer_window_path=fixture["consumer_paths"][0],
            expected_consumer_window_sha256=fixture["consumer_hashes"][0],
            expected_consumer_window_id="wrong-window-id",
            repository_root=tmp_path,
        )
    with pytest.raises(ValueError, match="provided together"):
        module.verify_stage2_source_reference(
            output,
            _sha(output),
            _sha(audit_path),
            statistics_config=fixture["config"],
            expected_consumer_window_path=fixture["consumer_paths"][0],
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    "tamper",
    ["stage2_contract", "npz_sidecar", "audit_guard"],
)
def test_public_verifier_rejects_fully_rehashed_contract_audit_and_stale_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    fixture = _fake_build_contract(tmp_path, monkeypatch)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    output = output_dir / "public-tamper.npz"
    _invoke_fake_build(tmp_path, fixture, output)
    audit_path = output.with_suffix(".audit.json")
    if tamper == "stage2_contract":
        with np.load(output, allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        arrays["stage2_contract_json"] = np.asarray(
            module._canonical_json({"tampered": True})
        )
        np.savez_compressed(output, **arrays)
        output.with_name(output.name + ".sha256").write_text(
            f"{_sha(output)}  {output.name}\n",
            encoding="ascii",
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["output"]["source_reference_npz"]["sha256"] = _sha(output)
        _write_json(audit_path, audit)
        audit_path.with_name(audit_path.name + ".sha256").write_text(
            f"{_sha(audit_path)}  {audit_path.name}\n",
            encoding="ascii",
        )
    elif tamper == "npz_sidecar":
        output.with_name(output.name + ".sha256").write_text(
            f"{'0' * 64}  {output.name}\n",
            encoding="ascii",
        )
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["official_test_accessed"] = True
        _write_json(audit_path, audit)
        audit_path.with_name(audit_path.name + ".sha256").write_text(
            f"{_sha(audit_path)}  {audit_path.name}\n",
            encoding="ascii",
        )
    with pytest.raises((TypeError, ValueError)):
        module.verify_stage2_source_reference(
            output,
            _sha(output),
            _sha(audit_path),
            statistics_config=fixture["config"],
            repository_root=tmp_path,
        )


def test_public_verifier_fails_closed_on_partial_bundle_while_lock_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fake_build_contract(tmp_path, monkeypatch)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    output = output_dir / "partial-visible.npz"
    _invoke_fake_build(tmp_path, fixture, output)
    audit_path = output.with_suffix(".audit.json")
    audit_sha = _sha(audit_path)
    audit_path.unlink()
    module._publication_lock_path(output).write_bytes(b"publishing")
    with pytest.raises(RuntimeError, match="publication lock"):
        module.verify_stage2_source_reference(
            output,
            _sha(output),
            audit_sha,
            statistics_config=fixture["config"],
            repository_root=tmp_path,
        )


def test_public_verifier_rejects_symlink_and_stable_read_identity_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fake_build_contract(tmp_path, monkeypatch)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    output = output_dir / "real.npz"
    _invoke_fake_build(tmp_path, fixture, output)
    alias = output_dir / "alias.npz"
    alias.symlink_to(output.name)
    with pytest.raises(ValueError, match="symlink"):
        module.verify_stage2_source_reference(
            alias,
            _sha(output),
            _sha(output.with_suffix(".audit.json")),
            statistics_config=fixture["config"],
            repository_root=tmp_path,
        )

    stable = tmp_path / "stable.json"
    digest = _write_json(stable, {"stable": True})
    real_load = module.json.load

    def replacing_load(handle: Any) -> Any:
        payload = real_load(handle)
        replacement = stable.with_name("replacement.json")
        replacement.write_bytes(stable.read_bytes())
        os.replace(replacement, stable)
        return payload

    monkeypatch.setattr(module.json, "load", replacing_load)
    with pytest.raises(RuntimeError, match="identity changed"):
        module._read_json_stable(stable, digest, "stable input")


@pytest.mark.parametrize("occupied_index", range(4))
def test_preoccupied_bundle_member_is_preserved_without_new_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    occupied_index: int,
) -> None:
    fixture = _fake_build_contract(tmp_path, monkeypatch)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    output = output_dir / "preoccupied.npz"
    occupied = module._bundle_paths(output)[occupied_index]
    occupied.write_bytes(b"preexisting")
    with pytest.raises(FileExistsError, match="already exists"):
        _invoke_fake_build(tmp_path, fixture, output)
    assert list(output_dir.iterdir()) == [occupied]
    assert not os.path.lexists(module._publication_lock_path(output))
    assert not tuple(output_dir.glob(".*stage2-source-reference-staging-*"))


@pytest.mark.parametrize(
    "failure_stage",
    ["npz", "audit", "npz_sidecar", "audit_sidecar"],
)
def test_bundle_write_failure_leaves_no_output_lock_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    fixture = _fake_build_contract(tmp_path, monkeypatch)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    output = output_dir / f"write-{failure_stage}.npz"
    real_npz = module._write_npz_exclusive
    real_write = module._write_exclusive

    def failing_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
        if failure_stage == "npz":
            path.write_bytes(b"partial")
            raise OSError("synthetic NPZ write failure")
        real_npz(path, arrays)

    def failing_write(path: Path, data: bytes) -> None:
        target = (
            (failure_stage == "audit" and path.name.endswith(".audit.json"))
            or (
                failure_stage == "npz_sidecar"
                and path.name.endswith(".npz.sha256")
            )
            or (
                failure_stage == "audit_sidecar"
                and path.name.endswith(".audit.json.sha256")
            )
        )
        if target:
            path.write_bytes(b"partial")
            raise OSError("synthetic bundle write failure")
        real_write(path, data)

    monkeypatch.setattr(module, "_write_npz_exclusive", failing_npz)
    monkeypatch.setattr(module, "_write_exclusive", failing_write)
    with pytest.raises(OSError, match="synthetic"):
        _invoke_fake_build(tmp_path, fixture, output)
    assert list(output_dir.iterdir()) == []


@pytest.mark.parametrize("failure_stage", ["publish_fsync", "final_fsync"])
def test_publication_and_final_fsync_failure_leave_zero_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    fixture = _fake_build_contract(tmp_path, monkeypatch)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    output = output_dir / f"{failure_stage}.npz"
    real_fsync = module._fsync_directory
    parent_calls = 0
    injected = False

    def failing_fsync(path: Path) -> None:
        nonlocal parent_calls, injected
        if path == output_dir:
            parent_calls += 1
            fail_call = 2 if failure_stage == "publish_fsync" else 3
            if parent_calls == fail_call and not injected:
                injected = True
                raise OSError(f"synthetic {failure_stage}")
        real_fsync(path)

    monkeypatch.setattr(module, "_fsync_directory", failing_fsync)
    with pytest.raises(OSError, match="synthetic"):
        _invoke_fake_build(tmp_path, fixture, output)
    assert injected is True
    assert list(output_dir.iterdir()) == []


def test_bundle_publication_failure_rolls_back_every_member_and_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fake_build_contract(tmp_path, monkeypatch)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    output = output_dir / "rollback.npz"
    real_link = os.link
    calls = 0

    def fail_second(source: str | bytes | os.PathLike[str], target: str | bytes | os.PathLike[str], **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic publication failure")
        real_link(source, target, **kwargs)

    monkeypatch.setattr(module.os, "link", fail_second)
    with pytest.raises(OSError, match="synthetic publication failure"):
        _invoke_fake_build(tmp_path, fixture, output)
    assert list(output_dir.iterdir()) == []


def test_cli_requires_explicit_external_hashes_for_every_manifest() -> None:
    parser = module.build_arg_parser()
    args = parser.parse_args(
        [
            "--score-manifest",
            "a.json",
            "--score-manifest",
            "b.json",
            "--score-manifest-sha256",
            "a" * 64,
            "--score-manifest-sha256",
            "b" * 64,
            "--checkpoint",
            "checkpoint.pt",
            "--expected-checkpoint-sha256",
            "c" * 64,
            "--statistics-config",
            "statistics.json",
            "--statistics-config-sha256",
            "d" * 64,
            "--consumer-window-manifest",
            "window.json",
            "--consumer-window-manifest-sha256",
            "e" * 64,
            "--output",
            "reference.npz",
        ]
    )
    assert len(args.score_manifest) == len(args.score_manifest_sha256) == 2
    assert len(args.consumer_window_manifest) == len(
        args.consumer_window_manifest_sha256
    ) == 1


def test_legacy_module_and_schema_versions_remain_unchanged() -> None:
    from rc import schema
    from data_ext import score_manifest_artifacts

    assert schema.SCHEMA_VERSION == "rc-irstd.meta-episode.v4"
    source = Path(score_manifest_artifacts.__file__).read_text(encoding="utf-8")
    assert "rc-irstd.score-manifest.v4" not in source

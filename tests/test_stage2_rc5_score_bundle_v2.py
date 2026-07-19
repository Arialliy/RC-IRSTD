from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pytest

from data_ext import stage2_score_manifest as score_v4
import data_ext.stage2_rc5_score_bundle_v2 as score_bundle
from data_ext.stage2_rc5_score_bundle_v2 import (
    ATTESTATION_NAME,
    ATTESTATION_SIDECAR_NAME,
    Stage2RC5ScoreBundleV2Error,
    VerifiedStage2RC5ScoreBundleV2,
    assert_verified_stage2_rc5_score_bundle_v2,
    publish_stage2_rc5_score_attestation_v2,
    replay_verified_stage2_rc5_score_bundle_v2,
    verify_stage2_rc5_score_bundle_v2,
)
from data_ext.stage2_score_manifest_metadata_v5 import (
    verify_stage2_score_manifest_metadata_v5,
)
import evaluation.export_stage2_rc5_score_bundle_v2 as score_export


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / (
    "outputs/stage2_detector_contracts/rc4_k2_c14q28_w01_20260716/"
    "selections/outer_leave_nuaa_sirst__s42__oof_fold_0/"
    "nudt_sirst.selection.json"
)

_RUN_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_stage2_detector_run_complete_test_helper",
    ROOT / "tests/test_stage2_detector_run_complete_v2.py",
)
assert _RUN_HELPER_SPEC is not None and _RUN_HELPER_SPEC.loader is not None
_RUN_HELPERS = importlib.util.module_from_spec(_RUN_HELPER_SPEC)
_RUN_HELPER_SPEC.loader.exec_module(_RUN_HELPERS)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _canonical(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@pytest.fixture
def repo_tmp_path() -> Path:
    parent = ROOT / ".tmp"
    parent.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="pytest-rc5-score-bundle-", dir=parent))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _inputs(repo_tmp_path: Path) -> dict[str, Any]:
    fixture = _RUN_HELPERS._synthetic_run(repo_tmp_path)
    complete = _RUN_HELPERS._publish(fixture)
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    run = fixture["run"]
    runtime = fixture["runtime"]
    run_dir = fixture["run_dir"]

    records: list[dict[str, Any]] = []
    for index, selected in enumerate(selection["records"]):
        records.append(
            {
                "record_index": index,
                **selected,
                "source_domain": selection["source_domain"],
                "score_file": (
                    f"{_relative(run_dir)}/scores/members/{index:06d}.npz"
                ),
                "score_file_sha256": "0" * 64,
                "original_hw": [1, 1],
                "input_hw": [256, 256],
                "resized_hw": [256, 256],
                "padding_ltrb": [0, 0, 0, 0],
                "resize_mode": "resize",
            }
        )

    bindings = {
        "selection_contract": {
            "path": _relative(SELECTION),
            "sha256": _sha(SELECTION),
        },
        "run_contract": {
            "path": _relative(_RUN_HELPERS.RUN_CONTRACT),
            "sha256": fixture["run_sha"],
        },
        "checkpoint": {
            "path": _relative(run_dir / "stage2_inference_checkpoint.pt"),
            "sha256": fixture["hashes"][
                "restricted_inference_checkpoint_sha256"
            ],
        },
        "detector_config": run["bindings"]["detector_config"],
        "runtime_config": {
            "path": _relative(run_dir / runtime["run_config"]["path"]),
            "sha256": runtime["run_config"]["sha256"],
        },
        "seed_manifest": run["bindings"]["seed_manifest"],
        "materialization_index": run["bindings"]["materialization_index"],
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
        "role": score_v4.OOF_TRAIN_SOURCE_REFERENCE,
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
        "source_domain": selection["source_domain"],
        "base_seed": run["base_seed"],
        "derived_seed": run["derived_seed"],
        "detector_role": run["detector_role"],
        "oof_fold_index": run["oof_fold_index"],
        "input_hw": [256, 256],
        "resize_mode": "resize",
        "bindings": bindings,
        "num_images": len(records),
        "records_content_sha256_algorithm": (
            score_v4.STAGE2_SCORE_RECORDS_ALGORITHM
        ),
        "records_content_sha256": score_v4.stage2_score_records_sha256(
            records
        ),
        "records": records,
    }
    output_dir = run_dir / "scores"
    output_dir.mkdir()
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "fixture": fixture,
        "complete": complete,
        "selection": selection,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "output_dir": output_dir,
    }


def _metadata(inputs: dict[str, Any]):
    return verify_stage2_score_manifest_metadata_v5(
        inputs["manifest_path"],
        _sha(inputs["manifest_path"]),
        score_v4.OOF_TRAIN_SOURCE_REFERENCE,
        repository_root=ROOT,
    )


def test_authoritative_export_binds_run_complete_without_opening_members(
    repo_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(repo_tmp_path)
    fixture = inputs["fixture"]
    manifest = inputs["manifest"]
    member_paths = {
        (ROOT / record[field]).absolute()
        for record in manifest["records"]
        for field in ("score_file", "original_image_path")
    }
    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any):
        if path.absolute() in member_paths:
            raise AssertionError(f"record member was opened: {path}")
        return original_open(path, *args, **kwargs)

    calls: list[str] = []

    def completed_legacy_export(*args: Any, **kwargs: Any) -> None:
        calls.append("legacy-v4-export-completed")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(
        score_export,
        "export_stage2_development_scores",
        completed_legacy_export,
    )
    bundle = score_export.export_stage2_rc5_score_bundle_v2(
        SELECTION,
        _RUN_HELPERS.RUN_CONTRACT,
        fixture["run_dir"] / "stage2_inference_checkpoint.pt",
        inputs["output_dir"],
        selection_contract_sha256=_sha(SELECTION),
        run_contract_sha256=fixture["run_sha"],
        checkpoint_sha256=fixture["hashes"][
            "restricted_inference_checkpoint_sha256"
        ],
        role=score_v4.OOF_TRAIN_SOURCE_REFERENCE,
        run_complete_capability=inputs["complete"],
        device="cpu",
        repository_root=ROOT,
    )
    replay = replay_verified_stage2_rc5_score_bundle_v2(bundle)

    assert calls == ["legacy-v4-export-completed"]
    assert assert_verified_stage2_rc5_score_bundle_v2(replay) is replay
    assert replay.attestation["score_manifest"]["sha256"] == _sha(
        inputs["manifest_path"]
    )
    assert replay.attestation["score_manifest"][
        "records_content_sha256"
    ] == manifest["records_content_sha256"]
    assert replay.attestation["run_complete"]["sha256"] == inputs[
        "complete"
    ].sha256
    assert replay.attestation["restricted_checkpoint"]["sha256"] == fixture[
        "hashes"
    ]["restricted_inference_checkpoint_sha256"]
    assert replay.attestation["selection_contract"]["path"] == _relative(
        SELECTION
    )
    assert replay.attestation["score_identity"]["source_domain"] == (
        inputs["selection"]["source_domain"]
    )
    assert replay.capability_contract["score_manifest_member_content_verified"] is False
    assert replay.capability_contract["score_record_files_opened"] is False
    assert replay.capability_contract["score_original_images_opened"] is False
    assert all(not (ROOT / record["score_file"]).exists() for record in manifest["records"])
    assert (inputs["output_dir"] / ATTESTATION_NAME).is_file()
    assert (inputs["output_dir"] / ATTESTATION_SIDECAR_NAME).is_file()


def test_forgery_and_current_state_tamper_are_rejected(
    repo_tmp_path: Path,
) -> None:
    inputs = _inputs(repo_tmp_path)
    metadata = _metadata(inputs)
    bundle = publish_stage2_rc5_score_attestation_v2(
        metadata, inputs["complete"]
    )
    with pytest.raises(TypeError, match="verifier-issued only"):
        VerifiedStage2RC5ScoreBundleV2(
            attestation_path=bundle.attestation_path,
            attestation_sha256=bundle.attestation_sha256,
            attestation=bundle.attestation,
            score_manifest_metadata=metadata,
            run_complete=inputs["complete"],
        )
    forged = object.__new__(VerifiedStage2RC5ScoreBundleV2)
    object.__setattr__(forged, "_capability", object())
    with pytest.raises(TypeError, match="verifier-issued"):
        assert_verified_stage2_rc5_score_bundle_v2(forged)

    attestation_path = bundle.attestation_path
    sidecar_path = attestation_path.with_name(ATTESTATION_SIDECAR_NAME)
    original_attestation = attestation_path.read_bytes()
    original_sidecar = sidecar_path.read_bytes()
    tampered_payload = json.loads(original_attestation)
    tampered_payload["causal_edge"] = "forged-causal-edge"
    tampered_bytes = _canonical(tampered_payload)
    tampered_sha = hashlib.sha256(tampered_bytes).hexdigest()
    attestation_path.write_bytes(tampered_bytes)
    sidecar_path.write_text(
        f"{tampered_sha}  {ATTESTATION_NAME}\n", encoding="utf-8"
    )
    with pytest.raises(Stage2RC5ScoreBundleV2Error, match="current-state replay"):
        verify_stage2_rc5_score_bundle_v2(
            attestation_path,
            tampered_sha,
            run_complete=inputs["complete"],
            repository_root=ROOT,
        )
    attestation_path.write_bytes(original_attestation)
    sidecar_path.write_bytes(original_sidecar)

    manifest_path = inputs["manifest_path"]
    original_manifest = manifest_path.read_bytes()
    manifest_path.write_bytes(original_manifest + b" ")
    with pytest.raises(ValueError):
        replay_verified_stage2_rc5_score_bundle_v2(bundle)
    manifest_path.write_bytes(original_manifest)

    checkpoint = inputs["fixture"]["run_dir"] / "stage2_inference_checkpoint.pt"
    original_checkpoint = checkpoint.read_bytes()
    checkpoint.write_bytes(original_checkpoint + b"tamper")
    with pytest.raises(ValueError):
        replay_verified_stage2_rc5_score_bundle_v2(bundle)
    checkpoint.write_bytes(original_checkpoint)
    assert replay_verified_stage2_rc5_score_bundle_v2(bundle).attestation_sha256 == (
        bundle.attestation_sha256
    )


def test_failed_final_verification_removes_authoritative_commit(
    repo_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(repo_tmp_path)
    metadata = _metadata(inputs)

    def fail_final_verifier(*args: Any, **kwargs: Any):
        raise Stage2RC5ScoreBundleV2Error("synthetic final verifier failure")

    monkeypatch.setattr(
        score_bundle, "verify_stage2_rc5_score_bundle_v2", fail_final_verifier
    )
    with pytest.raises(Stage2RC5ScoreBundleV2Error, match="final verifier"):
        publish_stage2_rc5_score_attestation_v2(metadata, inputs["complete"])
    assert (inputs["output_dir"] / ATTESTATION_NAME).is_file()
    assert not (inputs["output_dir"] / ATTESTATION_SIDECAR_NAME).exists()


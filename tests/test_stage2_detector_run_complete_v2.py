from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from data_ext.stage2_detector_run_complete_v2 import (
    RUN_COMPLETE_NAME,
    RUN_COMPLETE_SIDECAR_NAME,
    Stage2DetectorRunCompleteV2Error,
    VerifiedStage2DetectorRunCompleteV2,
    assert_stage2_run_complete_for_score_export_v2,
    assert_verified_stage2_detector_run_complete_v2,
    publish_stage2_detector_run_complete_v2,
    verify_stage2_detector_run_complete_v2,
)
from data_ext.stage2_role_contract import verify_stage2_run_contract_sidecar
from evaluation.export_stage2_development_scores_run_complete_v2 import (
    export_stage2_development_scores_run_complete_v2,
)
from scripts import train_multisource_tail as trainer


ROOT = Path(__file__).resolve().parents[1]
RUN_CONTRACT = ROOT / (
    "outputs/stage2_detector_contracts/rc4_k2_c14q28_w01_20260716/runs/"
    "outer_leave_nuaa_sirst__s42__oof_fold_0.json"
)
FIXED_LAST = "fixed_last_no_test_or_target_validation"
SEALED_SCOPE = "stage2_development_detector_official_test_sealed"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: object) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _sha(path)


def _sidecar(path: Path, sidecar_name: str | None = None) -> str:
    digest = _sha(path)
    sidecar = (
        path.with_suffix(path.suffix + ".sha256")
        if sidecar_name is None
        else path.parent / sidecar_name
    )
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _synthetic_run(
    tmp_path: Path,
    *,
    epochs: int = 2,
    config_overrides: dict | None = None,
    include_weights_last: bool = True,
) -> dict:
    run, run_sha = verify_stage2_run_contract_sidecar(RUN_CONTRACT)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    environment = {"python": "synthetic", "source_tree": "frozen"}
    config = {
        "epochs": epochs,
        "seed": run["derived_seed"],
        "source_names": run["source_domains"],
        "outer_fold_id": run["outer_fold_id"],
        "outer_target": run["outer_target_domain"],
        "held_out_domains": [run["outer_target_domain"]],
        "stage2_detector_role": run["detector_role"],
        "stage2_oof_fold_index": run["oof_fold_index"],
        "checkpoint_selection": FIXED_LAST,
        "protocol_scope": SEALED_SCOPE,
        "risk_objective": run["training"]["risk_objective"],
        "execution_fingerprint": environment,
        "engineering_smoke": False,
        "aaai27_pilot": False,
        "stage2_input_run_contract": {
            "path": RUN_CONTRACT.relative_to(ROOT).as_posix(),
            "sha256": run_sha,
            "schema_version": run["schema_version"],
            "run_id": run["run_id"],
            "bindings": run["bindings"],
            "official_test_accessed": False,
        },
    }
    config.update(config_overrides or {})
    config_path = run_dir / "config.json"
    config_sha = _json(config_path, config)
    _sidecar(config_path)
    runtime = trainer.write_stage2_runtime_artifacts(
        run_dir,
        SimpleNamespace(stage2_run_contract=str(RUN_CONTRACT)),
        run,
        run_sha,
        config_sha,
        environment,
    )

    rows = [
        {
            "epoch": epoch,
            "checkpoint_selection": FIXED_LAST,
            "protocol_scope": SEALED_SCOPE,
            "risk_objective": run["training"]["risk_objective"],
            "stage1_variant": run["training"]["stage1_variant"],
            "loss": 1.0 - epoch / 10.0,
        }
        for epoch in range(epochs)
    ]
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metrics_sha = _sidecar(metrics_path)

    state = {
        "weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "bias": torch.tensor([1.0, 2.0]),
    }
    common = {
        "epoch": epochs - 1,
        "seed": run["derived_seed"],
        "source_names": run["source_domains"],
        "outer_fold_id": run["outer_fold_id"],
        "outer_target": run["outer_target_domain"],
        "held_out_domains": [run["outer_target_domain"]],
        "detector_role": run["detector_role"],
        "oof_fold_index": run["oof_fold_index"],
        "checkpoint_selection": FIXED_LAST,
        "run_contract_sha256": run_sha,
        "run_config_sha256": config_sha,
        "official_test_accessed": False,
        "stage2_runtime_artifacts": runtime,
        "state_dict": state,
    }
    full = {
        **common,
        "format_version": "rc-irstd.detector.v2",
        "detector_source_domains": run["source_domains"],
        "protocol_scope": SEALED_SCOPE,
        "risk_objective": run["training"]["risk_objective"],
        "execution_fingerprint": environment,
        "training_args": {"epochs": epochs},
        "epoch_metrics": rows[-1],
    }
    checkpoint_path = run_dir / "checkpoint_last.pt"
    torch.save(full, checkpoint_path)
    checkpoint_sha = _sidecar(checkpoint_path, "checkpoint_sha256.txt")

    restricted = {
        **common,
        "format_version": "rc-irstd.detector-inference.v1",
        "training_args": {"base_size": 256, "crop_size": 256},
        "inference_geometry": {"input_hw": [256, 256], "resize_mode": "resize"},
    }
    restricted_path = run_dir / "stage2_inference_checkpoint.pt"
    torch.save(restricted, restricted_path)
    restricted_sha = _sidecar(restricted_path)

    weights_sha = None
    if include_weights_last:
        weights_path = run_dir / "weights_last.pt"
        torch.save(state, weights_path)
        weights_sha = _sha(weights_path)

    hashes = {
        "run_contract_sha256": run_sha,
        "runtime_contract_sha256": runtime["runtime_contract"]["sha256"],
        "run_config_sha256": config_sha,
        "environment_sha256": runtime["environment_artifact"]["sha256"],
        "release_artifact_sha256": runtime["release_artifact"]["sha256"],
        "release_source_archive_sha256": runtime["release_artifact"][
            "source_archive"
        ]["sha256"],
        "metrics_sha256": metrics_sha,
        "checkpoint_last_sha256": checkpoint_sha,
        "restricted_inference_checkpoint_sha256": restricted_sha,
        "weights_last_sha256": weights_sha,
    }
    return {
        "run": run,
        "run_sha": run_sha,
        "run_dir": run_dir,
        "runtime": runtime,
        "hashes": hashes,
        "metrics": rows,
    }


def _publish(fixture: dict):
    return publish_stage2_detector_run_complete_v2(
        run_dir=fixture["run_dir"],
        run_contract_path=RUN_CONTRACT,
        verified_run_contract=fixture["run"],
        verified_runtime_closure=fixture["runtime"],
        external_hashes=fixture["hashes"],
    )


def _refresh(fixture: dict, artifact_name: str, hash_key: str, sidecar: str) -> None:
    path = fixture["run_dir"] / artifact_name
    fixture["hashes"][hash_key] = _sidecar(path, sidecar)


def test_publish_replay_commit_last_and_optional_weights_are_non_authoritative(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_run(tmp_path)
    capability = _publish(fixture)
    assert capability.payload["artifact_status"] == "RUN_COMPLETE_FIXED_LAST_VERIFIED"
    assert capability.payload["target_epochs"] == 2
    assert capability.payload["completed_epoch"] == 1
    assert capability.payload["bindings"]["weights_last"]["integrity_only"] is True
    assert (
        capability.payload["bindings"]["weights_last"]["downstream_authority"]
        is False
    )
    assert capability.payload["invariants"]["metrics_values_embedded"] is False
    assert (fixture["run_dir"] / RUN_COMPLETE_NAME).is_file()
    assert (fixture["run_dir"] / RUN_COMPLETE_SIDECAR_NAME).is_file()
    replay = verify_stage2_detector_run_complete_v2(
        capability.artifact_path,
        capability.sha256,
        run_dir=fixture["run_dir"],
        run_contract_path=RUN_CONTRACT,
        verified_run_contract=fixture["run"],
        verified_runtime_closure=fixture["runtime"],
        external_hashes=fixture["hashes"],
    )
    assert replay.sha256 == capability.sha256
    assert assert_verified_stage2_detector_run_complete_v2(replay) is replay
    assert_stage2_run_complete_for_score_export_v2(
        replay,
        run_contract_path=RUN_CONTRACT,
        run_contract_sha256=fixture["run_sha"],
        checkpoint_path=fixture["run_dir"] / "stage2_inference_checkpoint.pt",
        checkpoint_sha256=fixture["hashes"][
            "restricted_inference_checkpoint_sha256"
        ],
    )
    signature = inspect.signature(export_stage2_development_scores_run_complete_v2)
    assert "run_complete_capability" in signature.parameters


def test_capability_cannot_be_constructed_or_forged(tmp_path: Path) -> None:
    fixture = _synthetic_run(tmp_path)
    capability = _publish(fixture)
    with pytest.raises(TypeError, match="verifier-issued only"):
        VerifiedStage2DetectorRunCompleteV2(
            artifact_path=capability.artifact_path,
            sha256=capability.sha256,
            run_dir=capability.run_dir,
            run_contract_path=capability.run_contract_path,
            payload={},
            external_hashes={},
            verified_run_contract={},
            verified_runtime_closure={},
        )
    forged = object.__new__(VerifiedStage2DetectorRunCompleteV2)
    object.__setattr__(forged, "_capability", object())
    with pytest.raises(TypeError, match="verifier-issued"):
        assert_verified_stage2_detector_run_complete_v2(forged)


@pytest.mark.parametrize(
    "missing",
    (
        "metrics.jsonl.sha256",
        "checkpoint_sha256.txt",
        "stage2_inference_checkpoint.pt.sha256",
    ),
)
def test_missing_completion_members_fail_closed(tmp_path: Path, missing: str) -> None:
    fixture = _synthetic_run(tmp_path)
    (fixture["run_dir"] / missing).unlink()
    with pytest.raises(FileNotFoundError):
        _publish(fixture)


def test_truncated_and_hash_drift_metrics_fail_closed(tmp_path: Path) -> None:
    truncated = _synthetic_run(tmp_path / "truncated")
    metrics = truncated["run_dir"] / "metrics.jsonl"
    metrics.write_text(json.dumps(truncated["metrics"][0]) + "\n", encoding="utf-8")
    _refresh(truncated, "metrics.jsonl", "metrics_sha256", "metrics.jsonl.sha256")
    with pytest.raises(Stage2DetectorRunCompleteV2Error, match="truncated|extra"):
        _publish(truncated)

    drift = _synthetic_run(tmp_path / "drift")
    metrics = drift["run_dir"] / "metrics.jsonl"
    metrics.write_text(metrics.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(Stage2DetectorRunCompleteV2Error, match="SHA-256"):
        _publish(drift)


def test_symlink_epoch_and_state_mismatch_fail_closed(tmp_path: Path) -> None:
    symlinked = _synthetic_run(tmp_path / "symlink")
    checkpoint = symlinked["run_dir"] / "stage2_inference_checkpoint.pt"
    target = symlinked["run_dir"] / "restricted-target.pt"
    checkpoint.rename(target)
    checkpoint.symlink_to(target.name)
    with pytest.raises(Stage2DetectorRunCompleteV2Error, match="symlink"):
        _publish(symlinked)

    epoch_mismatch = _synthetic_run(tmp_path / "epoch")
    checkpoint = epoch_mismatch["run_dir"] / "stage2_inference_checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["epoch"] = 0
    torch.save(payload, checkpoint)
    _refresh(
        epoch_mismatch,
        "stage2_inference_checkpoint.pt",
        "restricted_inference_checkpoint_sha256",
        "stage2_inference_checkpoint.pt.sha256",
    )
    with pytest.raises(Stage2DetectorRunCompleteV2Error, match="epoch mismatch"):
        _publish(epoch_mismatch)

    state_mismatch = _synthetic_run(tmp_path / "state")
    checkpoint = state_mismatch["run_dir"] / "stage2_inference_checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["state_dict"]["bias"] = torch.tensor([9.0, 9.0])
    torch.save(payload, checkpoint)
    _refresh(
        state_mismatch,
        "stage2_inference_checkpoint.pt",
        "restricted_inference_checkpoint_sha256",
        "stage2_inference_checkpoint.pt.sha256",
    )
    with pytest.raises(Stage2DetectorRunCompleteV2Error, match="tensor mismatch"):
        _publish(state_mismatch)


def test_outer_target_selection_and_weights_mutation_fail_closed(tmp_path: Path) -> None:
    run, _ = verify_stage2_run_contract_sidecar(RUN_CONTRACT)
    target_selected = _synthetic_run(
        tmp_path / "target",
        config_overrides={
            "source_names": [run["outer_target_domain"], run["source_domains"][1]]
        },
    )
    with pytest.raises(Stage2DetectorRunCompleteV2Error, match="source_names"):
        _publish(target_selected)

    weights_mutated = _synthetic_run(tmp_path / "weights")
    weights = weights_mutated["run_dir"] / "weights_last.pt"
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    payload["bias"] = torch.tensor([7.0, 7.0])
    torch.save(payload, weights)
    weights_mutated["hashes"]["weights_last_sha256"] = _sha(weights)
    with pytest.raises(Stage2DetectorRunCompleteV2Error, match="tensor mismatch"):
        _publish(weights_mutated)


def test_commit_payload_mutation_and_score_export_binding_mismatch_fail(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_run(tmp_path)
    capability = _publish(fixture)
    artifact = Path(capability.artifact_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["completed_epoch"] = 0
    artifact.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    mutated_sha = _sidecar(artifact)
    with pytest.raises(Stage2DetectorRunCompleteV2Error, match="current verified run state"):
        verify_stage2_detector_run_complete_v2(
            artifact,
            mutated_sha,
            run_dir=fixture["run_dir"],
            run_contract_path=RUN_CONTRACT,
            verified_run_contract=fixture["run"],
            verified_runtime_closure=fixture["runtime"],
            external_hashes=fixture["hashes"],
        )

    other = fixture["run_dir"] / "other.pt"
    other.write_bytes(
        (fixture["run_dir"] / "stage2_inference_checkpoint.pt").read_bytes()
    )
    with pytest.raises(Stage2DetectorRunCompleteV2Error, match="external SHA-256|score export checkpoint"):
        assert_stage2_run_complete_for_score_export_v2(
            capability,
            run_contract_path=RUN_CONTRACT,
            run_contract_sha256=fixture["run_sha"],
            checkpoint_path=other,
            checkpoint_sha256=fixture["hashes"][
                "restricted_inference_checkpoint_sha256"
            ],
        )

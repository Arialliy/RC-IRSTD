from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest

import data_ext.stage2_rc5_source_exact_curve_bank_v1 as bank
from data_ext import stage2_score_manifest as score_v4


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


@pytest.fixture
def repo_tmp_path() -> Path:
    parent = ROOT / ".tmp"
    parent.mkdir(exist_ok=True)
    import tempfile

    path = Path(tempfile.mkdtemp(prefix="pytest-source-curve-bank-", dir=parent))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_score(path: Path, record: dict[str, Any], index: int) -> str:
    probability = np.asarray(
        [
            [0.01, 0.1, 0.2, 0.3, 0.4],
            [0.05, 0.15, 0.25, 0.35, 0.45],
            [0.5, 0.6, 0.7, 0.8, 0.9],
            [0.55, 0.65, 0.75, 0.85, 0.95],
        ],
        dtype=np.float64,
    )
    probability = np.clip(probability + index * 1e-4, 1e-8, 1 - 1e-8)
    raw_logit = np.log(probability / (1.0 - probability))
    values = {
        "prob": probability,
        "raw_logit": raw_logit,
        "canonical_id": np.asarray(record["canonical_id"]),
        "image_id": np.asarray(record["image_id"]),
        "source_domain": np.asarray(record["source_domain"]),
        "original_hw": np.asarray([4, 5], dtype=np.int64),
        "input_hw": np.asarray([16, 16], dtype=np.int64),
        "resized_hw": np.asarray([16, 16], dtype=np.int64),
        "padding_ltrb": np.asarray([0, 0, 0, 0], dtype=np.int64),
        "resize_mode": np.asarray("resize"),
    }
    assert tuple(values) == score_v4.NPZ_FIELD_ORDER
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    return _sha(path)


def _role_detector(role: str) -> tuple[str, int | None]:
    if role in {
        score_v4.OOF_TRAIN_SOURCE_REFERENCE,
        score_v4.OOF_HOLDOUT_STAGE2_FIT,
    }:
        return "detector_oof", 0
    return "detector_full_fit", None


def _fake_authority(
    base: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    role: str = score_v4.OOF_HOLDOUT_STAGE2_FIT,
    count: int = 3,
    source_domain: str = "NUDT-SIRST",
) -> SimpleNamespace:
    dataset = base / source_domain
    images = dataset / "images"
    masks = dataset / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    records: list[dict[str, Any]] = []
    items: list[SimpleNamespace] = []
    for index in range(count):
        image_id = f"sample_{index:03d}"
        image_path = images / f"{image_id}.png"
        image = (
            np.arange(20, dtype=np.uint8).reshape(4, 5) + 11 * index
        ).astype(np.uint8)
        Image.fromarray(image, mode="L").save(image_path)
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[index % 4, (index + 1) % 5] = 255
        if index == 2:
            mask[3, 4] = 255
        Image.fromarray(mask, mode="L").save(masks / f"{image_id}.png")
        record: dict[str, Any] = {
            "record_index": index,
            "canonical_id": f"{source_domain}::{image_id}",
            "image_id": image_id,
            "source_domain": source_domain,
            "original_image_path": _relative(image_path),
            "original_image_sha256": _sha(image_path),
            "exclusion_group_id": f"SINGLETON::{image_id}",
            "near_duplicate_cluster_id_or_unique_sentinel": f"UNIQUE::{image_id}",
            "source_role_record_index": index,
            "score_file": "",
            "score_file_sha256": "",
            "original_hw": [4, 5],
            "input_hw": [16, 16],
            "resized_hw": [16, 16],
            "padding_ltrb": [0, 0, 0, 0],
            "resize_mode": "resize",
        }
        score_path = base / "scores" / f"{index:06d}.npz"
        record["score_file"] = _relative(score_path)
        record["score_file_sha256"] = _write_score(score_path, record, index)
        records.append(record)
        items.append(
            SimpleNamespace(
                record_index=index,
                record=record,
                score_path=score_path,
                image_path=image_path,
            )
        )

    detector_role, fold = _role_detector(role)
    outer_target = "NUAA-SIRST"
    payload = {
        "role": role,
        "outer_fold_id": "outer_leave_nuaa_sirst",
        "outer_target": outer_target,
        "source_domain": source_domain,
        "base_seed": 42,
        "derived_seed": 101,
        "detector_role": detector_role,
        "oof_fold_index": fold,
    }
    manifest_path = base / "scores" / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest_sha = _sha(manifest_path)
    records_sha = _digest(f"records:{role}:{source_domain}:{count}")
    metadata = SimpleNamespace(
        path=manifest_path,
        repository_root=ROOT,
        payload=payload,
        records=tuple(records),
        items=tuple(items),
        role=role,
        manifest_sha256=manifest_sha,
        records_content_sha256=records_sha,
        bindings={},
    )
    full = SimpleNamespace(
        path=manifest_path,
        repository_root=ROOT,
        payload=payload,
        records=tuple(records),
        items=tuple(items),
        role=role,
        manifest_sha256=manifest_sha,
        records_content_sha256=records_sha,
        bindings={},
    )
    attestation_path = base / "scores" / "RC5_SCORE_ATTESTATION.json"
    attestation_path.write_text("{}\n", encoding="utf-8")
    run_path = base / "run" / "RUN_COMPLETE.json"
    run_path.parent.mkdir()
    run_path.write_text("{}\n", encoding="utf-8")
    attestation_sha = _digest(f"attestation:{role}:{source_domain}")
    bundle = SimpleNamespace(
        score_manifest_metadata=metadata,
        attestation_path=attestation_path,
        attestation_sha256=attestation_sha,
        attestation={
            "run_complete": {
                "identity": {"identity_sha256": _digest("run-identity")}
            },
            "restricted_checkpoint": {
                "path": _relative(base / "checkpoint.pt"),
                "sha256": _digest("checkpoint"),
                "authority": "restricted_inference_checkpoint_only",
                "weights_last_authority": False,
            },
        },
        run_complete=SimpleNamespace(
            artifact_path=run_path,
            sha256=_sha(run_path),
        ),
    )
    (base / "checkpoint.pt").write_bytes(b"checkpoint")
    calls = {"assert": 0, "replay": 0, "full": 0}

    def fake_assert(value: Any) -> Any:
        calls["assert"] += 1
        if value is not bundle:
            raise TypeError("unexpected fake bundle")
        return value

    def fake_replay(value: Any) -> Any:
        calls["replay"] += 1
        if value is not bundle:
            raise TypeError("unexpected fake bundle")
        return value

    def fake_full(*args: Any, **kwargs: Any) -> Any:
        calls["full"] += 1
        assert args[0] == manifest_path
        assert args[1] == manifest_sha
        assert args[2] == role
        assert Path(kwargs["repository_root"]) == ROOT
        return full

    monkeypatch.setattr(
        bank, "assert_verified_stage2_rc5_score_bundle_v2", fake_assert
    )
    monkeypatch.setattr(
        bank, "replay_verified_stage2_rc5_score_bundle_v2", fake_replay
    )
    monkeypatch.setattr(bank, "verify_stage2_score_manifest", fake_full)
    return SimpleNamespace(
        bundle=bundle,
        full=full,
        metadata=metadata,
        dataset=dataset,
        masks=masks,
        scores=base / "scores",
        records=records,
        calls=calls,
    )


@pytest.mark.parametrize(
    "role",
    [
        score_v4.OOF_TRAIN_SOURCE_REFERENCE,
        score_v4.OOF_HOLDOUT_STAGE2_FIT,
        score_v4.FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE,
        score_v4.SOURCE_DIAGNOSTIC_VALIDATION,
    ],
)
def test_all_four_source_roles_publish_and_fresh_replay(
    repo_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    fixture = _fake_authority(repo_tmp_path, monkeypatch, role=role)
    issued = bank.build_and_publish_stage2_rc5_source_exact_curve_bank_v1(
        score_bundle=fixture.bundle,
        dataset_directory=fixture.dataset,
        output_directory=repo_tmp_path / f"bank-{role}",
        repository_root=ROOT,
    )
    assert issued.role == role
    assert issued.source_domain == "NUDT-SIRST"
    assert len(issued.curves_in_record_order) == 3
    assert issued.curve_bank.image_count == 3
    assert issued.curve_offsets.flags.owndata
    assert not issued.curve_offsets.flags.writeable
    assert issued.curve_thresholds.flags.owndata
    assert not issued.curve_thresholds.flags.writeable
    assert fixture.calls["replay"] >= 3
    assert fixture.calls["full"] >= 3

    replayed = bank.replay_verified_stage2_rc5_source_exact_curve_bank_v1(issued)
    assert replayed.commit_sha256 == issued.commit_sha256
    assert replayed.curve_bank.bank_id == issued.curve_bank.bank_id
    assert fixture.calls["replay"] >= 4
    assert fixture.calls["full"] >= 4


def test_outer_target_rejected_before_full_verifier_or_mask_path(
    repo_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fake_authority(
        repo_tmp_path,
        monkeypatch,
        role=score_v4.OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
        source_domain="NUAA-SIRST",
    )
    mask_calls = 0

    def forbidden_mask(*args: Any, **kwargs: Any) -> Path:
        nonlocal mask_calls
        mask_calls += 1
        raise AssertionError("mask path must remain unopened")

    monkeypatch.setattr(bank, "_resolve_mask_direct", forbidden_mask)
    with pytest.raises(
        bank.Stage2RC5SourceExactCurveBankV1Error,
        match="outer-target",
    ):
        bank.build_and_publish_stage2_rc5_source_exact_curve_bank_v1(
            score_bundle=fixture.bundle,
            dataset_directory=repo_tmp_path / "not-even-a-dataset",
            output_directory=repo_tmp_path / "must-not-exist",
            repository_root=ROOT,
        )
    assert mask_calls == 0
    assert fixture.calls["full"] == 0
    assert not (repo_tmp_path / "must-not-exist").exists()


def test_dataset_prefix_mismatch_rejected_before_mask_resolution(
    repo_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fake_authority(repo_tmp_path, monkeypatch)
    alternate = repo_tmp_path / "alternate" / "NUDT-SIRST"
    (alternate / "masks").mkdir(parents=True)
    (alternate / "images").mkdir()
    mask_calls = 0
    original = bank._resolve_mask_direct

    def tracked(*args: Any, **kwargs: Any) -> Path:
        nonlocal mask_calls
        mask_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(bank, "_resolve_mask_direct", tracked)
    with pytest.raises(
        bank.Stage2RC5SourceExactCurveBankV1Error,
        match="outside dataset_directory/images",
    ):
        bank.build_and_publish_stage2_rc5_source_exact_curve_bank_v1(
            score_bundle=fixture.bundle,
            dataset_directory=alternate,
            output_directory=repo_tmp_path / "prefix-reject",
            repository_root=ROOT,
        )
    assert mask_calls == 0


def test_source_mask_symlink_is_rejected(
    repo_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fake_authority(repo_tmp_path, monkeypatch)
    victim = fixture.masks / "sample_000.png"
    target = fixture.masks / "real-mask.png"
    victim.replace(target)
    victim.symlink_to(target.name)
    with pytest.raises(
        bank.Stage2RC5SourceExactCurveBankV1Error,
        match="symlink",
    ):
        bank.build_and_publish_stage2_rc5_source_exact_curve_bank_v1(
            score_bundle=fixture.bundle,
            dataset_directory=fixture.dataset,
            output_directory=repo_tmp_path / "mask-link-reject",
            repository_root=ROOT,
        )


def test_stable_bytes_detects_path_replacement_during_fd_read(
    repo_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = repo_tmp_path / "stable.bin"
    replacement = repo_tmp_path / "replacement.bin"
    target.write_bytes(b"a" * 64)
    replacement.write_bytes(b"b" * 64)
    original_read = bank.os.read
    swapped = False

    def swapping_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        data = original_read(descriptor, count)
        if data and not swapped:
            swapped = True
            os.replace(replacement, target)
        return data

    monkeypatch.setattr(bank.os, "read", swapping_read)
    with pytest.raises(RuntimeError, match="changed during stable read"):
        bank._stable_bytes(target, ROOT, "TOCTOU target")
    assert swapped


def test_public_replay_recomputes_mask_label_and_curve(
    repo_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fake_authority(repo_tmp_path, monkeypatch)
    issued = bank.build_and_publish_stage2_rc5_source_exact_curve_bank_v1(
        score_bundle=fixture.bundle,
        dataset_directory=fixture.dataset,
        output_directory=repo_tmp_path / "recompute-bank",
        repository_root=ROOT,
    )
    changed = np.ones((4, 5), dtype=np.uint8) * 255
    Image.fromarray(changed, mode="L").save(fixture.masks / "sample_001.png")
    with pytest.raises(
        bank.Stage2RC5SourceExactCurveBankV1Error,
        match="causal replay|manifest differs|source-mask",
    ):
        bank.replay_verified_stage2_rc5_source_exact_curve_bank_v1(issued)


def test_label_member_tamper_is_rejected_by_bytes_replay(
    repo_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fake_authority(repo_tmp_path, monkeypatch)
    issued = bank.build_and_publish_stage2_rc5_source_exact_curve_bank_v1(
        score_bundle=fixture.bundle,
        dataset_directory=fixture.dataset,
        output_directory=repo_tmp_path / "label-tamper-bank",
        repository_root=ROOT,
    )
    label = issued.path / str(issued.manifest["records"][0]["label_file"])
    label.write_bytes(b"not-an-npz")
    with pytest.raises(bank.Stage2RC5SourceExactCurveBankV1Error):
        bank.replay_verified_stage2_rc5_source_exact_curve_bank_v1(issued)


def test_commit_fault_leaves_no_authoritative_marker(
    repo_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fake_authority(repo_tmp_path, monkeypatch)
    output = repo_tmp_path / "commit-fault-bank"
    original = bank._write_exclusive

    def fail_commit(path: Path, data: bytes) -> None:
        if path.name == bank.COMMIT_FILENAME:
            raise OSError("synthetic curve-bank commit fault")
        original(path, data)

    monkeypatch.setattr(bank, "_write_exclusive", fail_commit)
    with pytest.raises(OSError, match="commit fault"):
        bank.build_and_publish_stage2_rc5_source_exact_curve_bank_v1(
            score_bundle=fixture.bundle,
            dataset_directory=fixture.dataset,
            output_directory=output,
            repository_root=ROOT,
        )
    assert output.is_dir()
    assert not (output / bank.COMMIT_FILENAME).exists()


def test_external_commit_sha_and_retained_token_drift_are_rejected(
    repo_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fake_authority(repo_tmp_path, monkeypatch)
    issued = bank.build_and_publish_stage2_rc5_source_exact_curve_bank_v1(
        score_bundle=fixture.bundle,
        dataset_directory=fixture.dataset,
        output_directory=repo_tmp_path / "capability-bank",
        repository_root=ROOT,
    )
    with pytest.raises(
        bank.Stage2RC5SourceExactCurveBankV1Error,
        match="external commit SHA",
    ):
        bank.verify_stage2_rc5_source_exact_curve_bank_v1(
            issued.commit_path,
            "0" * 64,
            score_bundle=fixture.bundle,
            repository_root=ROOT,
        )

    forged = object.__new__(bank.VerifiedStage2RC5SourceExactCurveBankV1)
    for name in bank.VerifiedStage2RC5SourceExactCurveBankV1.__dataclass_fields__:
        object.__setattr__(forged, name, getattr(issued, name))
    changed = np.array(issued.curve_thresholds, copy=True)
    changed[0] = np.nextafter(changed[0], 1.0)
    changed.setflags(write=False)
    object.__setattr__(forged, "curve_thresholds", changed)
    with pytest.raises(TypeError, match="retained-token state drifted"):
        bank.assert_verified_stage2_rc5_source_exact_curve_bank_v1(forged)


def test_output_parent_symlink_is_rejected_before_creation(
    repo_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fake_authority(repo_tmp_path, monkeypatch)
    real = repo_tmp_path / "real-parent"
    real.mkdir()
    linked = repo_tmp_path / "linked-parent"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(
        bank.Stage2RC5SourceExactCurveBankV1Error,
        match="symlink",
    ):
        bank.build_and_publish_stage2_rc5_source_exact_curve_bank_v1(
            score_bundle=fixture.bundle,
            dataset_directory=fixture.dataset,
            output_directory=linked / "bank",
            repository_root=ROOT,
        )
    assert not (real / "bank").exists()

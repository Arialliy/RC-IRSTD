from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.freeze_official_train_splits import (
    DatasetSpec,
    build_files,
    check_files,
    write_files,
)


def _dataset(root: Path, name: str, *, train_count: int = 30) -> DatasetSpec:
    dataset = root / name
    split_dir = dataset / "img_idx"
    split_dir.mkdir(parents=True)
    train = [f"image_{index:03d}" for index in range(train_count)]
    test = [f"test_{index:03d}" for index in range(5)]
    (split_dir / f"train_{name}.txt").write_text(
        "\n".join(train) + "\n", encoding="utf-8"
    )
    (split_dir / f"test_{name}.txt").write_text(
        "\n".join(test) + "\n", encoding="utf-8"
    )
    return DatasetSpec(name=name, root=dataset.resolve())


def _build(tmp_path: Path, dataset: DatasetSpec, output: Path):
    return build_files(
        [dataset],
        repository_root=tmp_path,
        output_dir=output,
        seed=42,
        detector_diagnostic_fraction=0.20,
        meta_validation_fraction=0.25,
        context_size=3,
        query_size=2,
    )


def _quarantine(tmp_path: Path, dataset_name: str, image_id: str) -> Path:
    audit = tmp_path / "audit.json"
    preview = tmp_path / "preview.png"
    audit.write_text("{}\n", encoding="utf-8")
    preview.write_bytes(b"preview")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "schema_version": "rc-irstd.aaai27-near-duplicate-quarantine.v1",
        "status": "resolved_by_development_quarantine",
        "source_audit": {
            "path": "audit.json",
            "sha256": digest(audit),
            "confirmed_pair_count": 1,
        },
        "visual_review": {
            "preview_path": "preview.png",
            "preview_sha256": digest(preview),
        },
        "decision_policy": {
            "official_test_labels_read": False,
            "raw_data_modified": False,
            "official_split_files_modified": False,
        },
        "candidate_decisions": [
            {
                "candidate_id": "a" * 64,
                "dataset_name": dataset_name,
                "official_train_image_id": image_id,
                "final_decision": "same_scene_related",
                "action": "exclude_official_train_member_from_all_development_roles",
            }
        ],
        "datasets": [
            {
                "dataset_name": dataset_name,
                "excluded_official_train_ids": [image_id],
                "excluded_count": 1,
            }
        ],
        "total_excluded_official_train_ids": 1,
    }
    path = tmp_path / "quarantine.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frozen_splits_are_reproducible_and_official_train_only(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, "domain-a")
    output = tmp_path / "splits"
    files = _build(tmp_path, dataset, output)
    write_files(files)
    check_files(_build(tmp_path, dataset, output))

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summary = manifest["datasets"][0]
    assert summary["official_train_count"] == 30
    assert summary["official_test_count"] == 5
    assert summary["detector"]["fit_count"] == 24
    assert summary["detector"]["diagnostic_count"] == 6
    assert summary["meta"]["train_window_count"] == 4
    assert summary["meta"]["validation_window_count"] == 2
    assert summary["meta"]["unused_image_count"] == 0
    assert manifest["role_contract"]["official_test_emitted"] is False
    assert (
        manifest["role_contract"][
            "outer_target_official_train_used_for_detector_fit"
        ]
        is False
    )
    assert (
        manifest["role_contract"][
            "outer_target_detector_diagnostic_used_for_development_evaluation"
        ]
        is True
    )
    assert (
        manifest["role_contract"]["outer_target_diagnostic_selects_checkpoint"]
        is False
    )
    assert "outer_target_official_train_used" not in manifest["role_contract"]
    assert (
        "outer_target_official_train_allowed_in_same_outer_fold"
        not in manifest["role_contract"]
    )

    emitted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*")
        if path.is_file()
    )
    assert "test_000" not in emitted


def test_frozen_split_check_detects_manual_drift(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "domain-a")
    output = tmp_path / "splits"
    files = _build(tmp_path, dataset, output)
    write_files(files)
    (output / "domain-a" / "detector_fit.txt").write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not reproducible"):
        check_files(_build(tmp_path, dataset, output))


def test_frozen_splits_fail_on_official_train_test_overlap(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "domain-a")
    test_path = dataset.root / "img_idx" / "test_domain-a.txt"
    test_path.write_text("image_000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="official train/test IDs overlap"):
        _build(tmp_path, dataset, tmp_path / "splits")


def test_v2_quarantine_is_applied_before_every_derived_role(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "domain-a")
    output = tmp_path / "splits"
    quarantine = _quarantine(tmp_path, dataset.name, "image_000")
    files = build_files(
        [dataset],
        repository_root=tmp_path,
        output_dir=output,
        seed=42,
        detector_diagnostic_fraction=0.20,
        meta_validation_fraction=0.25,
        context_size=3,
        query_size=2,
        quarantine_config=quarantine,
    )
    write_files(files)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"].endswith(".v2")
    summary = manifest["datasets"][0]
    assert summary["development_quarantine"]["quarantined_count"] == 1
    assert summary["development_quarantine"]["effective_development_train_count"] == 29
    emitted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*")
        if path.is_file()
    )
    assert "image_000" in (
        output / "domain-a" / "quarantined_official_train_ids.txt"
    ).read_text(encoding="utf-8")
    for role_file in (
        output / "domain-a" / "detector_fit.txt",
        output / "domain-a" / "detector_diagnostic.txt",
        output / "domain-a" / "meta_train_windows.json",
        output / "domain-a" / "meta_validation_windows.json",
        output / "domain-a" / "meta_unused_ids.txt",
        output / "domain-a" / "effective_development_train.txt",
    ):
        assert "image_000" not in role_file.read_text(encoding="utf-8")

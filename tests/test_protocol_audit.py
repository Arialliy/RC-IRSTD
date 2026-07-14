from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.audit_aaai_protocol import (
    assess_nested_protocol,
    audit_dataset,
    build_report,
)


def _make_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "img_idx").mkdir()
    for image_id, value in (("train_item", 20), ("test_item", 40)):
        image = np.full((4, 8, 3), value, dtype=np.uint8)
        mask = np.zeros((4, 8), dtype=np.uint8)
        Image.fromarray(image).save(root / "images" / f"{image_id}.png")
        Image.fromarray(mask).save(root / "masks" / f"{image_id}.png")
    (root / "img_idx" / f"train_{root.name}.txt").write_text(
        "train_item.png\n", encoding="utf-8"
    )
    (root / "img_idx" / f"test_{root.name}.txt").write_text(
        "test_item.png\n", encoding="utf-8"
    )
    return root


def test_dataset_audit_proves_split_and_content_disjointness(tmp_path: Path) -> None:
    result = audit_dataset(_make_dataset(tmp_path / "toy"))
    assert result["num_train"] == 1
    assert result["num_test"] == 1
    assert result["train_test_id_overlap_count"] == 0
    assert result["train_test_image_content_overlap_count"] == 0
    assert result["within_split_image_content_duplicate_group_count"] == 0
    assert result["split_contract_passed"] is True


def test_three_domains_with_two_holdouts_is_non_claim_bearing_smoke() -> None:
    result = assess_nested_protocol(
        ["IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST"],
        outer_target="NUAA-SIRST",
        pseudo_target="NUDT-SIRST",
    )
    assert result["detector_sources"] == ["IRSTD-1K"]
    assert result["claim_bearing_nested_lodo_eligible"] is False
    assert result["protocol_scope"] == "single_source_inner_smoke_not_main_result"


def test_cross_dataset_exact_image_duplicates_fail_global_contract(
    tmp_path: Path,
) -> None:
    first = _make_dataset(tmp_path / "domain-a")
    second = _make_dataset(tmp_path / "domain-b")
    report = build_report([first, second])
    assert report["cross_dataset_exact_image_duplicate_group_count"] == 2
    assert report["cross_dataset_exact_duplicate_contract_passed"] is False
    assert report["all_split_contracts_passed"] is False
    assert report["near_duplicate_audit"]["status"] == "not_run"


def test_effective_near_duplicate_audit_is_hash_bound(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path / "domain-a")
    audit = tmp_path / "near.json"
    audit.write_text(
        json.dumps(
            {
                "status": "passed",
                "near_duplicate_contract_passed": True,
                "confirmed_near_duplicate_pair_count": 0,
                "image_only": True,
                "labels_scores_checkpoints_or_metrics_read": False,
                "inputs": [{"dataset_name": "domain-a"}],
            }
        ),
        encoding="utf-8",
    )
    report = build_report([dataset], near_duplicate_audit=audit)
    assert report["near_duplicate_audit"]["status"] == "passed"
    assert len(report["near_duplicate_audit"]["sha256"]) == 64
    assert report["all_split_contracts_passed"] is True

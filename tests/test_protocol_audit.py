from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.audit_aaai_protocol import assess_nested_protocol, audit_dataset


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

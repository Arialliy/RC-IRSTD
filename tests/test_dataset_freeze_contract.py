from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.freeze_dataset_contract import build_contract


def test_dataset_contract_freezes_misc111_nearest_alignment(tmp_path: Path) -> None:
    root = tmp_path / "datasets" / "NUAA-SIRST"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "img_idx").mkdir()
    Image.fromarray(np.zeros((50, 100), dtype=np.uint8)).save(
        root / "images" / "train.png"
    )
    Image.fromarray(np.zeros((50, 100), dtype=np.uint8)).save(
        root / "masks" / "train.png"
    )
    Image.fromarray(np.zeros((50, 100), dtype=np.uint8)).save(
        root / "images" / "Misc_111.png"
    )
    Image.fromarray(np.zeros((100, 200), dtype=np.uint8)).save(
        root / "masks" / "Misc_111.png"
    )
    (root / "img_idx" / "train_NUAA-SIRST.txt").write_text(
        "train\n", encoding="utf-8"
    )
    (root / "img_idx" / "test_NUAA-SIRST.txt").write_text(
        "Misc_111\n", encoding="utf-8"
    )
    split_manifest = tmp_path / "splits" / "manifest.json"
    split_manifest.parent.mkdir()
    split_manifest.write_text(
        json.dumps(
            {"schema_version": "rc-irstd.aaai27-official-train-splits.v2"}
        ),
        encoding="utf-8",
    )
    payload = build_contract(
        [("NUAA-SIRST", root)],
        repository_root=tmp_path,
        split_manifest=split_manifest,
    )
    special = payload["required_special_case"]
    assert special["image_id"] == "Misc_111"
    assert special["image_hw"] == [50, 100]
    assert special["original_mask_hw"] == [100, 200]
    assert special["alignment_applied"] is True
    assert special["mask_alignment"] == "nearest"

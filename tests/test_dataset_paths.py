from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rc_irstd.data.dataset import IRSTDDataset


def _write_sample(root: Path, relative: str, value: int) -> None:
    image_path = root / "images" / relative
    mask_path = root / "masks" / relative
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 9, 3), value, dtype=np.uint8)
    mask = np.zeros((8, 9), dtype=np.uint8)
    mask[3:5, 4:6] = 255
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)


def test_nested_split_paths_preserve_unique_image_ids(tmp_path: Path) -> None:
    root = tmp_path / "NestedDomain"
    _write_sample(root, "scene_a/frame_0001.png", 32)
    _write_sample(root, "scene_b/frame_0001.png", 64)
    split = root / "img_idx" / "test.txt"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text(
        "scene_a/frame_0001.png\nscene_b/frame_0001.png\n",
        encoding="utf-8",
    )

    dataset = IRSTDDataset(root, split="test", resize_hw=(8, 9))
    first = dataset[0]["meta"]
    second = dataset[1]["meta"]

    assert first.image_id == "scene_a/frame_0001"
    assert second.image_id == "scene_b/frame_0001"
    assert first.image_id != second.image_id
    assert first.sequence_id != second.sequence_id


def test_misc_111_geometry_matches_basicirstd_nearest_alignment(tmp_path: Path) -> None:
    root = tmp_path / "NUAA-SIRST"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    (root / "img_idx").mkdir(parents=True)
    image = np.zeros((220, 325, 3), dtype=np.uint8)
    mask = np.zeros((400, 592), dtype=np.uint8)
    mask[73:89, 301:319] = 255
    Image.fromarray(image).save(root / "images" / "Misc_111.png")
    Image.fromarray(mask).save(root / "masks" / "Misc_111.png")
    (root / "masks" / "Misc_111.xml").write_text("<annotation/>", encoding="utf-8")
    (root / "img_idx" / "test.txt").write_text("Misc_111\n", encoding="utf-8")

    dataset = IRSTDDataset(root, split="test", resize_hw=None)
    sample = dataset[0]
    nearest = getattr(Image, "Resampling", Image).NEAREST
    expected = Image.fromarray(mask).resize((325, 220), nearest)

    assert tuple(sample["mask"].shape) == (1, 220, 325)
    assert np.array_equal(sample["mask"][0].numpy() > 0, np.asarray(expected) > 0)
    assert sample["meta"].mask_original_hw == (400, 592)
    assert sample["meta"].mask_aligned_to_image is True


def test_dataset_rejects_true_image_mask_aspect_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "WrongPair"
    _write_sample(root, "sample.png", 32)
    Image.fromarray(np.zeros((9, 9), dtype=np.uint8)).save(
        root / "masks" / "sample.png"
    )
    (root / "train.txt").write_text("sample.png\n", encoding="utf-8")

    dataset = IRSTDDataset(root, split="train", resize_hw=None)
    with pytest.raises(ValueError, match="aspect-ratio mismatch"):
        dataset[0]

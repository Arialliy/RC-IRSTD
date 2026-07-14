from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from rc_irstd.data.dataset import IRSTDDataset
from rc_irstd.data.transforms import load_image_preserve_depth, target_preserving_resize_mask
from rc_irstd.data.windows import build_iid_windows


def _make_dataset(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    (root / "img_idx").mkdir(parents=True)
    image = np.linspace(0, 65535, 8 * 12, dtype=np.uint16).reshape(8, 12)
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[7, 11] = 255
    Image.fromarray(image).save(root / "images" / "0001.png")
    Image.fromarray(mask).save(root / "masks" / "0001.png")
    (root / "img_idx" / "train.txt").write_text("0001\n", encoding="utf-8")


def test_16bit_loader_and_target_preserving_resize(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    loaded = load_image_preserve_depth(tmp_path / "images" / "0001.png")
    assert loaded.bit_depth == 16
    assert loaded.array.dtype == np.uint16

    dataset = IRSTDDataset(
        tmp_path,
        split="train",
        resize_hw=(4, 18),  # mixed down/up resize
        normalization="percentile",
        include_component_labels=True,
    )
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 4, 18)
    assert sample["meta"].bit_depth == 16
    assert int(sample["mask"].sum()) >= 1
    assert int(sample["component_labels"].max()) == 1


def test_target_preserving_resize_never_drops_single_pixel() -> None:
    mask = np.zeros((17, 23), dtype=np.uint8)
    mask[16, 22] = 1
    for target in ((3, 5), (3, 40), (40, 5), (40, 50)):
        resized = target_preserving_resize_mask(mask, target)
        assert resized.shape == target
        assert resized.max() == 1


def test_iid_windows_are_deterministic_and_disjoint() -> None:
    first = build_iid_windows(30, context_size=5, horizon=3, stride=8, seed=11)
    second = build_iid_windows(30, context_size=5, horizon=3, stride=8, seed=11)
    assert first == second
    used: set[int] = set()
    for window in first:
        assert window.protocol == "iid"
        block = set(window.context_indices) | set(window.future_indices)
        assert not set(window.context_indices) & set(window.future_indices)
        assert not used & block
        used |= block

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

from rc_irstd.data.transforms import (
    image_array_to_tensor,
    load_image_preserve_depth,
    resize_image_array,
    target_preserving_resize_mask,
)
from rc_irstd.data.mask_alignment import align_mask_to_image


RASTER_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class SampleMeta:
    image_id: str
    dataset_name: str
    original_hw: tuple[int, int]
    sequence_id: str
    frame_index: int
    image_path: str
    mask_path: str | None
    input_hw: tuple[int, int]
    bit_depth: int
    dataset_type: str
    mask_original_hw: tuple[int, int] | None
    mask_aligned_to_image: bool


def _resolve_file(
    folder: Path,
    image_id: str,
    required: bool = True,
    *,
    stem_suffixes: Sequence[str] = ("",),
) -> Path | None:
    """Resolve split entries with or without extensions and nested directories."""
    relative = Path(image_id)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Split entry must be a safe relative path, got: {image_id}")
    direct = folder / relative
    if direct.is_file() and direct.suffix.lower() in RASTER_EXTENSIONS:
        return direct
    parent = folder / relative.parent
    stem = relative.stem
    matches = []
    if parent.is_dir():
        for suffix in stem_suffixes:
            for extension in RASTER_EXTENSIONS:
                candidate = parent / f"{stem}{suffix}{extension}"
                if candidate.is_file():
                    matches.append(candidate)
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous raster files for '{image_id}' under {folder}: "
            + ", ".join(str(path) for path in matches)
        )
    if required:
        raise FileNotFoundError(f"No file for '{image_id}' under {folder}")
    return None


def read_split(root: Path, split: str | Path) -> list[str]:
    split_path = Path(split)
    candidates: list[Path] = []
    if split_path.is_file():
        candidates.append(split_path)
    else:
        candidates.extend(
            [
                root / str(split),
                root / f"{split}.txt",
                root / "img_idx" / str(split),
                root / "img_idx" / f"{split}.txt",
            ]
        )
        if (root / "img_idx").is_dir():
            candidates.extend(sorted((root / "img_idx").glob(f"{split}*.txt")))
    selected = next((path for path in candidates if path.is_file()), None)
    if selected is None:
        raise FileNotFoundError(
            f"Cannot resolve split '{split}' in {root}. "
            "Pass an explicit split file when the dataset uses a custom layout."
        )
    names = [
        line.strip()
        for line in selected.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        raise ValueError(f"Split file {selected} is empty")
    return names


def infer_sequence_and_index(image_id: str, fallback_index: int) -> tuple[str, int]:
    path = Path(image_id)
    stem = path.stem
    parent = path.parent.as_posix()
    parent_sequence = parent if parent not in {"", "."} else "default"
    for separator in ("_", "-"):
        parts = stem.split(separator)
        if len(parts) > 1 and parts[-1].isdigit():
            prefix = separator.join(parts[:-1])
            if parent_sequence != "default":
                prefix = f"{parent_sequence}/{prefix}" if prefix else parent_sequence
            return prefix or "default", int(parts[-1])
    if stem.isdigit():
        return parent_sequence, int(stem)
    return parent_sequence, fallback_index


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((np.asarray(mask) > 0).astype(np.float32)[None])


class IRSTDDataset(Dataset[dict[str, Any]]):
    """BasicIRSTD-style dataset with bit-depth and target-preserving loading."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str | Path = "train",
        resize_hw: tuple[int, int] | None = None,
        augment: bool = False,
        domain_id: int = 0,
        require_mask: bool = True,
        sequence_parser: Callable[[str, int], tuple[str, int]] | None = None,
        normalization: str = "imagenet",
        dataset_type: str = "iid_images",
        include_component_labels: bool = True,
    ) -> None:
        self.root = Path(dataset_dir)
        self.dataset_name = self.root.name
        self.names = read_split(self.root, split)
        self.resize_hw = resize_hw
        self.augment = augment
        self.domain_id = int(domain_id)
        self.require_mask = bool(require_mask)
        self.sequence_parser = sequence_parser or infer_sequence_and_index
        self.normalization = normalization
        if dataset_type not in {"iid_images", "temporal"}:
            raise ValueError("dataset_type must be iid_images or temporal")
        self.dataset_type = dataset_type
        self.include_component_labels = bool(include_component_labels)
        if not (self.root / "images").is_dir():
            raise FileNotFoundError(f"Expected images/ under {self.root}")
        if self.require_mask and not (self.root / "masks").is_dir():
            raise FileNotFoundError(f"Expected masks/ under {self.root}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.names[index]
        image_path = _resolve_file(self.root / "images", entry, required=True)
        assert image_path is not None
        mask_path = _resolve_file(
            self.root / "masks",
            entry,
            required=self.require_mask,
            stem_suffixes=("", "_pixels0"),
        )
        loaded = load_image_preserve_depth(image_path)
        image_array = loaded.array
        if image_array.ndim == 3 and image_array.shape[-1] == 4:
            image_array = image_array[..., :3]
        mask_array: np.ndarray | None = None
        mask_original_hw: tuple[int, int] | None = None
        mask_aligned_to_image = False
        if mask_path is not None:
            with Image.open(mask_path) as mask_image:
                mask_pil = mask_image.convert("L")
            mask_original_hw = (int(mask_pil.height), int(mask_pil.width))
            image_size = (int(image_array.shape[1]), int(image_array.shape[0]))
            mask_aligned_to_image = mask_pil.size != image_size
            mask_pil = align_mask_to_image(mask_pil, image_size, Path(entry).stem)
            mask_array = (np.asarray(mask_pil) > 0).astype(np.uint8)
        original_hw = tuple(int(value) for value in image_array.shape[:2])

        if self.resize_hw is not None:
            image_array = resize_image_array(image_array, self.resize_hw)
            if mask_array is not None:
                mask_array = target_preserving_resize_mask(mask_array, self.resize_hw)

        if self.augment:
            if np.random.rand() < 0.5:
                image_array = np.flip(image_array, axis=1)
                if mask_array is not None:
                    mask_array = np.flip(mask_array, axis=1)
            if np.random.rand() < 0.5:
                image_array = np.flip(image_array, axis=0)
                if mask_array is not None:
                    mask_array = np.flip(mask_array, axis=0)
            rotations = int(np.random.randint(0, 4))
            image_array = np.rot90(image_array, rotations).copy()
            if mask_array is not None:
                mask_array = np.rot90(mask_array, rotations).copy()

        sequence_id, frame_index = self.sequence_parser(entry, index)
        if self.dataset_type == "iid_images":
            # Keep path-derived grouping metadata for provenance, but callers must
            # not interpret it as temporal ordering under the iid protocol.
            frame_index = index
        component_labels = None
        if mask_array is not None and self.include_component_labels:
            component_labels = ndimage.label(mask_array > 0)[0].astype(np.int64)

        input_hw = tuple(int(value) for value in image_array.shape[:2])
        return {
            "image": image_array_to_tensor(image_array, normalization=self.normalization),
            "mask": mask_to_tensor(mask_array) if mask_array is not None else None,
            "component_labels": (
                torch.from_numpy(component_labels[None])
                if component_labels is not None
                else None
            ),
            "domain_id": self.domain_id,
            "meta": SampleMeta(
                image_id=Path(entry).with_suffix("").as_posix(),
                dataset_name=self.dataset_name,
                original_hw=original_hw,
                sequence_id=sequence_id,
                frame_index=frame_index,
                image_path=str(image_path),
                mask_path=str(mask_path) if mask_path is not None else None,
                input_hw=input_hw,
                bit_depth=loaded.bit_depth,
                dataset_type=self.dataset_type,
                mask_original_hw=mask_original_hw,
                mask_aligned_to_image=mask_aligned_to_image,
            ),
        }


def collate_samples(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    shapes = {tuple(item["image"].shape[-2:]) for item in batch}
    if len(shapes) != 1:
        raise ValueError(
            "Batches require equal tensor shapes. Configure resize_hw, "
            "or use batch_size=1 for original-resolution export."
        )
    masks = [item["mask"] for item in batch]
    if any(mask is None for mask in masks) and not all(mask is None for mask in masks):
        raise ValueError("A batch cannot mix labelled and unlabelled samples")
    mask_batch = None if all(mask is None for mask in masks) else torch.stack(masks)  # type: ignore[arg-type]

    labels = [item.get("component_labels") for item in batch]
    if all(label is None for label in labels):
        label_batch = None
    elif any(label is None for label in labels):
        raise ValueError("A batch cannot mix component-labelled and unlabelled samples")
    else:
        label_batch = torch.stack(labels)  # type: ignore[arg-type]

    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": mask_batch,
        "component_labels": label_batch,
        "domain_id": torch.tensor([item["domain_id"] for item in batch], dtype=torch.long),
        "meta": [item["meta"] for item in batch],
    }

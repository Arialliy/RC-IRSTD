"""Evaluation-only IRSTD dataset with reversible spatial transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TVF

from .dataset_meta import (
    SampleMeta,
    SpatialTransform,
    build_spatial_transform,
    normalise_hw,
)
from .split_utils import (
    read_split_entries,
    resolve_image_and_mask,
    resolve_split_file,
    sample_id_from_entry,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class IRSTDEvalDataset(Dataset):
    """Read test samples while preserving their native geometry and identity.

    The returned ``meta`` value is a nested mapping supported by PyTorch's
    default DataLoader collate.  Predictions in input space can therefore be
    restored with ``meta`` before pixel/component evaluation.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        base_size: int | Sequence[int] = 256,
        *,
        resize_mode: str = "resize",
        split: str = "test",
        split_file: str | Path | None = None,
        image_folder: str = "images",
        mask_folder: str = "masks",
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.root = Path(dataset_dir).expanduser().resolve()
        self.dataset_name = self.root.name
        self.input_hw = normalise_hw(base_size)
        self.resize_mode = resize_mode.lower()
        # Validate mode and dimensions once, before opening any sample.
        build_spatial_transform((1, 1), self.input_hw, self.resize_mode)

        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must each contain three channel values")
        if any(float(value) <= 0 for value in std):
            raise ValueError("All standard deviations must be positive")
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)

        self.split_file = resolve_split_file(self.root, split, split_file)
        self.entries = read_split_entries(self.split_file)
        self.samples: list[tuple[str, Path, Path]] = []
        for entry in self.entries:
            image_path, mask_path = resolve_image_and_mask(
                self.root,
                entry,
                image_folder=image_folder,
                mask_folder=mask_folder,
            )
            self.samples.append(
                (sample_id_from_entry(entry), image_path, mask_path)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_id, image_path, mask_path = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L")

        mask_original_hw = (mask.height, mask.width)
        # One published NUAA-SIRST sample (Misc_111 in common mirrors) has a
        # mask stored at a different canvas size.  Align GT to the image canvas
        # explicitly with nearest-neighbour interpolation before any input
        # transform; metadata preserves the source size for auditability.
        if image.size != mask.size:
            mask = mask.resize(image.size, resample=_NEAREST)
        original_hw = (image.height, image.width)
        transform = build_spatial_transform(
            original_hw,
            self.input_hw,
            self.resize_mode,
        )
        input_image = _apply_spatial_transform(
            image,
            transform,
            resample=_BILINEAR,
            fill=(0, 0, 0),
        )
        input_mask = _apply_spatial_transform(
            mask,
            transform,
            resample=_NEAREST,
            fill=0,
        )

        image_tensor = TVF.to_tensor(input_image)
        image_tensor = TVF.normalize(image_tensor, self.mean, self.std)
        mask_tensor = (TVF.pil_to_tensor(input_mask) > 0).to(torch.float32)

        metadata = SampleMeta(
            image_id=image_id,
            dataset_name=self.dataset_name,
            image_path=str(image_path),
            mask_path=str(mask_path),
            mask_original_hw=mask_original_hw,
            transform=transform,
        )
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "meta": metadata.to_collatable(),
        }

    def load_original_mask(self, index: int) -> torch.Tensor:
        """Load an exact, binary native-resolution mask for score export."""

        _, image_path, mask_path = self.samples[index]
        with Image.open(image_path) as image_file:
            image_size = image_file.size
        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L")
        if mask.size != image_size:
            mask = mask.resize(image_size, resample=_NEAREST)
        return (TVF.pil_to_tensor(mask) > 0).to(torch.uint8)


def _apply_spatial_transform(
    image: Image.Image,
    transform: SpatialTransform,
    *,
    resample: int,
    fill: int | tuple[int, int, int],
) -> Image.Image:
    resized_h, resized_w = transform.resized_hw
    result = image.resize((resized_w, resized_h), resample=resample)
    if transform.mode == "letterbox":
        result = ImageOps.expand(
            result,
            border=transform.padding_ltrb,
            fill=fill,
        )
    if result.size != (transform.input_hw[1], transform.input_hw[0]):
        raise RuntimeError(
            f"Spatial transform produced {result.size}, expected "
            f"{(transform.input_hw[1], transform.input_hw[0])}"
        )
    return result


try:
    _BILINEAR = Image.Resampling.BILINEAR
    _NEAREST = Image.Resampling.NEAREST
except AttributeError:  # Pillow < 9.1
    _BILINEAR = Image.BILINEAR
    _NEAREST = Image.NEAREST

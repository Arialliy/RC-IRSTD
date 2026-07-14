"""Strictly image-only IRSTD inference dataset for score export."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TVF

from .dataset_meta import (
    ImageSampleMeta,
    SpatialTransform,
    build_spatial_transform,
    normalise_hw,
)
from .eval_dataset import IMAGENET_MEAN, IMAGENET_STD
from .split_utils import (
    read_split_entries,
    resolve_sample_file,
    resolve_split_file,
    sample_id_from_entry,
)


class IRSTDInferenceDataset(Dataset):
    """Read inference images without resolving, enumerating or opening masks."""

    def __init__(
        self,
        dataset_dir: str | Path,
        base_size: int | Sequence[int] = 256,
        *,
        resize_mode: str = "resize",
        split: str = "test",
        split_file: str | Path | None = None,
        image_folder: str = "images",
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.root = Path(dataset_dir).expanduser().resolve()
        self.dataset_name = self.root.name
        self.input_hw = normalise_hw(base_size)
        self.resize_mode = resize_mode.lower()
        build_spatial_transform((1, 1), self.input_hw, self.resize_mode)
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must each contain three channel values")
        if any(float(value) <= 0 for value in std):
            raise ValueError("All standard deviations must be positive")
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)

        self.split_file = resolve_split_file(self.root, split, split_file)
        self.entries = read_split_entries(self.split_file)
        self.samples: list[tuple[str, Path]] = []
        for entry in self.entries:
            image_path = resolve_sample_file(
                self.root,
                image_folder,
                entry,
                kind="image",
            )
            self.samples.append((sample_id_from_entry(entry), image_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_id, image_path = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        original_hw = (image.height, image.width)
        transform = build_spatial_transform(
            original_hw,
            self.input_hw,
            self.resize_mode,
        )
        input_image = _apply_spatial_transform(image, transform)
        image_tensor = TVF.to_tensor(input_image)
        image_tensor = TVF.normalize(image_tensor, self.mean, self.std)
        metadata = ImageSampleMeta(
            image_id=image_id,
            dataset_name=self.dataset_name,
            image_path=str(image_path),
            transform=transform,
        )
        return {"image": image_tensor, "meta": metadata.to_collatable()}


def _apply_spatial_transform(
    image: Image.Image,
    transform: SpatialTransform,
) -> Image.Image:
    resized_h, resized_w = transform.resized_hw
    result = image.resize((resized_w, resized_h), resample=_BILINEAR)
    if transform.mode == "letterbox":
        result = ImageOps.expand(
            result,
            border=transform.padding_ltrb,
            fill=(0, 0, 0),
        )
    if result.size != (transform.input_hw[1], transform.input_hw[0]):
        raise RuntimeError("image transform did not produce the declared input size")
    return result


try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9.1
    _BILINEAR = Image.BILINEAR

"""Collate-safe metadata and reversible evaluation image transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


VALID_RESIZE_MODES = ("resize", "letterbox")


def normalise_hw(value: int | Sequence[int]) -> tuple[int, int]:
    """Normalise an integer or ``(height, width)`` value."""

    if isinstance(value, int):
        height = width = value
    else:
        if len(value) != 2:
            raise ValueError("Image size must contain exactly (height, width)")
        height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(height, width)}")
    return height, width


@dataclass(frozen=True)
class SpatialTransform:
    """Geometry needed to map an input-space prediction to native resolution."""

    mode: str
    original_hw: tuple[int, int]
    input_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    padding_ltrb: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if self.mode not in VALID_RESIZE_MODES:
            raise ValueError(
                f"mode must be one of {VALID_RESIZE_MODES}, got {self.mode!r}"
            )
        normalise_hw(self.original_hw)
        normalise_hw(self.input_hw)
        normalise_hw(self.resized_hw)
        if len(self.padding_ltrb) != 4 or any(value < 0 for value in self.padding_ltrb):
            raise ValueError("padding_ltrb must contain four non-negative integers")
        left, top, right, bottom = self.padding_ltrb
        resized_h, resized_w = self.resized_hw
        input_h, input_w = self.input_hw
        if resized_h + top + bottom != input_h or resized_w + left + right != input_w:
            raise ValueError(
                "resized dimensions plus padding must equal input dimensions: "
                f"resized={self.resized_hw}, padding={self.padding_ltrb}, "
                f"input={self.input_hw}"
            )


@dataclass(frozen=True)
class SampleMeta:
    """Metadata for one evaluation sample.

    ``to_collatable`` converts numeric tuples to tensors and leaves strings as
    strings, so PyTorch's default DataLoader collate handles the nested mapping
    without a custom collate function.
    """

    image_id: str
    dataset_name: str
    image_path: str
    mask_path: str
    mask_original_hw: tuple[int, int]
    transform: SpatialTransform

    def to_collatable(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "dataset_name": self.dataset_name,
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "mask_original_hw": torch.tensor(
                self.mask_original_hw,
                dtype=torch.int64,
            ),
            "resize_mode": self.transform.mode,
            "original_hw": torch.tensor(self.transform.original_hw, dtype=torch.int64),
            "input_hw": torch.tensor(self.transform.input_hw, dtype=torch.int64),
            "resized_hw": torch.tensor(self.transform.resized_hw, dtype=torch.int64),
            "padding_ltrb": torch.tensor(
                self.transform.padding_ltrb,
                dtype=torch.int64,
            ),
        }


def build_spatial_transform(
    original_hw: Sequence[int],
    input_hw: int | Sequence[int],
    mode: str,
) -> SpatialTransform:
    """Build deterministic resize or centred-letterbox geometry."""

    original_h, original_w = normalise_hw(original_hw)
    input_h, input_w = normalise_hw(input_hw)
    mode = mode.lower()
    if mode not in VALID_RESIZE_MODES:
        raise ValueError(f"Unknown resize mode {mode!r}; expected {VALID_RESIZE_MODES}")

    if mode == "resize":
        return SpatialTransform(
            mode=mode,
            original_hw=(original_h, original_w),
            input_hw=(input_h, input_w),
            resized_hw=(input_h, input_w),
            padding_ltrb=(0, 0, 0, 0),
        )

    scale = min(input_h / original_h, input_w / original_w)
    resized_h = min(input_h, max(1, int(round(original_h * scale))))
    resized_w = min(input_w, max(1, int(round(original_w * scale))))
    remaining_h = input_h - resized_h
    remaining_w = input_w - resized_w
    top = remaining_h // 2
    bottom = remaining_h - top
    left = remaining_w // 2
    right = remaining_w - left
    return SpatialTransform(
        mode=mode,
        original_hw=(original_h, original_w),
        input_hw=(input_h, input_w),
        resized_hw=(resized_h, resized_w),
        padding_ltrb=(left, top, right, bottom),
    )


def sample_meta_from_batch(
    metadata_batch: Mapping[str, Any],
    index: int = 0,
) -> SampleMeta:
    """Recover one :class:`SampleMeta` from default-collated metadata."""

    required = {
        "image_id",
        "dataset_name",
        "image_path",
        "mask_path",
        "mask_original_hw",
        "resize_mode",
        "original_hw",
        "input_hw",
        "resized_hw",
        "padding_ltrb",
    }
    missing = required.difference(metadata_batch)
    if missing:
        raise KeyError(f"Metadata batch is missing fields: {sorted(missing)}")

    transform = SpatialTransform(
        mode=str(_batch_item(metadata_batch["resize_mode"], index)),
        original_hw=_batch_int_tuple(metadata_batch["original_hw"], index, 2),
        input_hw=_batch_int_tuple(metadata_batch["input_hw"], index, 2),
        resized_hw=_batch_int_tuple(metadata_batch["resized_hw"], index, 2),
        padding_ltrb=_batch_int_tuple(metadata_batch["padding_ltrb"], index, 4),
    )
    return SampleMeta(
        image_id=str(_batch_item(metadata_batch["image_id"], index)),
        dataset_name=str(_batch_item(metadata_batch["dataset_name"], index)),
        image_path=str(_batch_item(metadata_batch["image_path"], index)),
        mask_path=str(_batch_item(metadata_batch["mask_path"], index)),
        mask_original_hw=_batch_int_tuple(
            metadata_batch["mask_original_hw"],
            index,
            2,
        ),
        transform=transform,
    )


def restore_tensor_to_original(
    tensor: torch.Tensor,
    transform: SpatialTransform,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Remove letterbox padding and resize a tensor to ``original_hw``.

    Accepted shapes are ``H×W``, ``C×H×W`` and ``N×C×H×W``.  Floating-point
    probability maps normally use bilinear interpolation; masks should use
    nearest-neighbour interpolation.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.ndim not in {2, 3, 4}:
        raise ValueError(f"Expected a 2D, 3D or 4D tensor, got shape {tuple(tensor.shape)}")

    original_ndim = tensor.ndim
    if original_ndim == 2:
        working = tensor[None, None]
    elif original_ndim == 3:
        working = tensor[None]
    else:
        working = tensor

    input_h, input_w = transform.input_hw
    if tuple(working.shape[-2:]) != (input_h, input_w):
        raise ValueError(
            f"Tensor spatial shape {tuple(working.shape[-2:])} does not match "
            f"metadata input_hw {transform.input_hw}"
        )

    left, top, _, _ = transform.padding_ltrb
    resized_h, resized_w = transform.resized_hw
    working = working[..., top : top + resized_h, left : left + resized_w]

    interpolate_kwargs: dict[str, Any] = {
        "size": transform.original_hw,
        "mode": mode,
    }
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        interpolate_kwargs["align_corners"] = False
    if not working.is_floating_point():
        working = working.float()
    restored = F.interpolate(working, **interpolate_kwargs)

    if original_ndim == 2:
        return restored[0, 0]
    if original_ndim == 3:
        return restored[0]
    return restored


def safe_output_stem(image_id: str) -> str:
    """Convert a potentially nested sample ID into a safe filename stem."""

    value = Path(image_id).as_posix().strip("/").replace("/", "__")
    if not value or value in {".", ".."}:
        raise ValueError(f"Invalid image_id for output: {image_id!r}")
    return value


def _batch_item(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        selected = value[index]
        return selected.item() if selected.ndim == 0 else selected
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _batch_int_tuple(value: Any, index: int, length: int) -> tuple[int, ...]:
    selected = _batch_item(value, index)
    if isinstance(selected, torch.Tensor):
        items = selected.detach().cpu().reshape(-1).tolist()
    elif isinstance(selected, (list, tuple)):
        items = list(selected)
    else:
        raise TypeError(f"Expected a sequence metadata field, got {type(selected).__name__}")
    if len(items) != length:
        raise ValueError(f"Expected {length} metadata values, got {items}")
    return tuple(int(item) for item in items)

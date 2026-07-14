from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class LoadedImage:
    array: np.ndarray
    bit_depth: int


def load_image_preserve_depth(path: str | Path) -> LoadedImage:
    """Load an image without silently reducing 16-bit infrared data to 8 bit."""
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.dtype == np.uint16:
        bit_depth = 16
    elif array.dtype == np.uint8:
        bit_depth = 8
    elif np.issubdtype(array.dtype, np.integer):
        bit_depth = int(np.iinfo(array.dtype).bits)
    else:
        bit_depth = 32
    return LoadedImage(array=np.asarray(array), bit_depth=bit_depth)


def _scale_to_unit(array: np.ndarray, mode: str = "dtype") -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if mode == "dtype":
        original = np.asarray(array)
        if np.issubdtype(original.dtype, np.integer):
            maximum = float(np.iinfo(original.dtype).max)
        else:
            maximum = float(np.nanmax(values))
        return np.clip(values / max(maximum, 1.0), 0.0, 1.0)
    if mode == "minmax":
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
    elif mode == "percentile":
        low, high = [float(x) for x in np.nanpercentile(values, [0.5, 99.5])]
    else:
        raise ValueError(f"Unknown intensity scaling mode: {mode}")
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def image_array_to_tensor(
    array: np.ndarray,
    normalization: str = "imagenet",
) -> torch.Tensor:
    """Convert grayscale/RGB 8- or 16-bit input to a three-channel tensor."""
    if normalization == "imagenet":
        scaled = _scale_to_unit(array, mode="dtype")
    elif normalization in {"minmax", "percentile"}:
        scaled = _scale_to_unit(array, mode=normalization)
    elif normalization == "none":
        scaled = np.asarray(array, dtype=np.float32)
    else:
        raise ValueError(
            "normalization must be one of imagenet, minmax, percentile or none"
        )

    if scaled.ndim == 2:
        scaled = np.repeat(scaled[..., None], 3, axis=2)
    elif scaled.ndim == 3 and scaled.shape[-1] == 1:
        scaled = np.repeat(scaled, 3, axis=2)
    elif scaled.ndim != 3:
        raise ValueError(f"Unsupported image shape: {scaled.shape}")
    if scaled.shape[-1] > 3:
        scaled = scaled[..., :3]
    tensor = torch.from_numpy(np.ascontiguousarray(scaled.transpose(2, 0, 1))).float()
    if normalization == "imagenet":
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = (tensor - mean) / std
    return tensor


def resize_image_array(
    array: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Resize an image while retaining its numeric dynamic range."""
    target_h, target_w = [int(value) for value in target_hw]
    source = np.asarray(array)
    tensor = torch.from_numpy(source.astype(np.float32, copy=False))
    if tensor.ndim == 2:
        tensor = tensor[None, None]
        resized = F.interpolate(
            tensor,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    elif tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)[None]
        resized = F.interpolate(
            tensor,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)
    else:
        raise ValueError(f"Unsupported image shape: {source.shape}")
    output = resized.cpu().numpy()
    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        output = np.rint(output).clip(info.min, info.max).astype(source.dtype)
    else:
        output = output.astype(source.dtype)
    return output


def target_preserving_resize_mask(
    mask: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Resize a binary mask without deleting small positive targets.

    Adaptive max pooling is used when reducing resolution. It guarantees that a
    positive input pixel contributes to at least one output bin, unlike a plain
    nearest-neighbour sample that can skip a one-pixel target.
    """
    binary = (np.asarray(mask) > 0).astype(np.float32)
    source_h, source_w = binary.shape[-2:]
    target_h, target_w = [int(value) for value in target_hw]
    tensor = torch.from_numpy(binary)[None, None]
    if target_h < source_h or target_w < source_w:
        # Adaptive max pooling also supports a larger output along the other
        # axis.  Using it whenever either axis is reduced prevents a one-pixel
        # target from disappearing in mixed resize cases such as H down / W up.
        resized = F.adaptive_max_pool2d(tensor, (target_h, target_w))
    else:
        resized = F.interpolate(tensor, size=(target_h, target_w), mode="nearest")
    return (resized[0, 0].numpy() > 0.5).astype(np.uint8)


def pad_tensor_to_stride(
    tensor: torch.Tensor,
    stride: int = 32,
    value: float = 0.0,
) -> tuple[torch.Tensor, tuple[int, int]]:
    if tensor.ndim not in {3, 4}:
        raise ValueError("tensor must be CHW or BCHW")
    if stride <= 0:
        raise ValueError("stride must be positive")
    height, width = tensor.shape[-2:]
    pad_h = (-height) % stride
    pad_w = (-width) % stride
    return F.pad(tensor, (0, pad_w, 0, pad_h), value=value), (height, width)

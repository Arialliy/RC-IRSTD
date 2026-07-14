"""Conservative image/mask geometry alignment for IRSTD datasets."""

from __future__ import annotations

from typing import Sequence

from PIL import Image


DEFAULT_ASPECT_TOLERANCE = 0.01


def aspect_ratio_relative_error(
    image_size: Sequence[int],
    mask_size: Sequence[int],
) -> float:
    """Return the relative difference between two ``(width, height)`` ratios."""

    if len(image_size) != 2 or len(mask_size) != 2:
        raise ValueError("image and mask sizes must contain (width, height)")
    image_size = (int(image_size[0]), int(image_size[1]))
    mask_size = (int(mask_size[0]), int(mask_size[1]))
    if min(*image_size, *mask_size) <= 0:
        raise ValueError(
            f"image and mask sizes must be positive: image={image_size} mask={mask_size}"
        )
    image_ratio = float(image_size[0]) / float(image_size[1])
    mask_ratio = float(mask_size[0]) / float(mask_size[1])
    return abs(image_ratio - mask_ratio) / max(abs(image_ratio), 1e-12)


def align_mask_to_image(
    mask: Image.Image,
    image_or_size: Image.Image | Sequence[int],
    image_id: str,
    *,
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
) -> Image.Image:
    """Align a same-aspect-ratio mask to an image canvas, or fail closed.

    The commonly distributed NUAA-SIRST ``Misc_111`` pair has different
    image/mask resolutions but nearly identical aspect ratios.  Nearest-
    neighbour resizing is safe for that resolution-only mismatch.  A true
    aspect-ratio mismatch usually indicates a wrong pair or corrupt dataset,
    so it must never be hidden by an unconditional resize.

    PIL sizes use ``(width, height)`` ordering.
    """

    if isinstance(image_or_size, Image.Image):
        image_size = image_or_size.size
    else:
        if len(image_or_size) != 2:
            raise ValueError("image size must contain exactly (width, height)")
        image_size = (int(image_or_size[0]), int(image_or_size[1]))

    mask_size = mask.size
    if mask_size == image_size:
        return mask
    if min(*image_size, *mask_size) <= 0:
        raise ValueError(
            f"Non-positive image/mask size for {image_id}: "
            f"image={image_size} mask={mask_size}"
        )
    tolerance = float(aspect_tolerance)
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("aspect_tolerance must be in [0, 1)")

    relative_error = aspect_ratio_relative_error(image_size, mask_size)
    if relative_error > tolerance:
        raise ValueError(
            f"Image/mask aspect-ratio mismatch for {image_id}: "
            f"image={image_size} mask={mask_size}; "
            f"relative_error={relative_error:.6%} exceeds "
            f"tolerance={tolerance:.6%}"
        )

    try:
        nearest = Image.Resampling.NEAREST
    except AttributeError:  # Pillow < 9.1
        nearest = Image.NEAREST
    return mask.resize(image_size, resample=nearest)

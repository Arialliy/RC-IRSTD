from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from rc_irstd.utils.io import atomic_json_dump, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create small BasicIRSTD-style synthetic domains.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--domains", nargs="+", default=["DomainA", "DomainB", "DomainC"])
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--frames-per-sequence", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def _background(rng: np.random.Generator, height: int, width: int, domain_index: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    base = 0.20 + 0.03 * domain_index
    image = np.full((height, width), base, dtype=np.float32)
    if domain_index % 3 == 0:
        image += 0.05 * (xx / max(width - 1, 1))
        image += rng.normal(0.0, 0.025, image.shape)
    elif domain_index % 3 == 1:
        image += 0.04 * np.sin(2.0 * np.pi * yy / 13.0)
        image += 0.03 * np.cos(2.0 * np.pi * xx / 19.0)
        image += rng.normal(0.0, 0.035, image.shape)
    else:
        image += 0.05 * np.sin(2.0 * np.pi * (xx + yy) / 17.0)
        image += rng.normal(0.0, 0.045, image.shape)
        image = ndimage.gaussian_filter(image, sigma=0.7)
    # Sparse hot clutter creates false-peak pressure.
    clutter_count = 4 + 2 * domain_index
    for _ in range(clutter_count):
        y = int(rng.integers(2, height - 2))
        x = int(rng.integers(2, width - 2))
        image[y, x] += float(rng.uniform(0.10, 0.30))
    return image


def _add_targets(
    rng: np.random.Generator,
    image: np.ndarray,
    domain_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    mask = np.zeros_like(image, dtype=np.uint8)
    target_count = int(rng.integers(1, 4))
    for _ in range(target_count):
        radius = int(rng.integers(1, 3))
        y = int(rng.integers(4, height - 4))
        x = int(rng.integers(4, width - 4))
        yy, xx = np.ogrid[:height, :width]
        disk = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
        mask[disk] = 1
        contrast = float(rng.uniform(0.45, 0.70) - 0.06 * domain_index)
        sigma = 0.7 + 0.25 * domain_index
        impulse = np.zeros_like(image)
        impulse[y, x] = contrast
        target = ndimage.gaussian_filter(impulse, sigma=sigma)
        target /= max(float(target.max()), 1e-8)
        image += contrast * target
    return np.clip(image, 0.0, 1.0), mask


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.height < 16 or args.width < 16:
        raise ValueError("Synthetic images must be at least 16x16")
    root = ensure_dir(args.output_root)
    manifest: dict[str, object] = {"domains": {}}
    for domain_index, domain_name in enumerate(args.domains):
        domain_root = ensure_dir(root / domain_name)
        images_dir = ensure_dir(domain_root / "images")
        masks_dir = ensure_dir(domain_root / "masks")
        split_dir = ensure_dir(domain_root / "img_idx")
        train_names: list[str] = []
        test_names: list[str] = []
        rng = np.random.default_rng(args.seed + 1000 * domain_index)
        for sequence in range(args.sequences):
            for frame in range(args.frames_per_sequence):
                image = _background(rng, args.height, args.width, domain_index)
                image, mask = _add_targets(rng, image, domain_index)
                image_id = f"seq{sequence:02d}_{frame:04d}"
                rgb = np.repeat((image * 255.0).round().astype(np.uint8)[..., None], 3, axis=2)
                Image.fromarray(rgb).save(images_dir / f"{image_id}.png")
                Image.fromarray(mask * 255).save(masks_dir / f"{image_id}.png")
                if sequence < max(1, args.sequences // 2):
                    train_names.append(image_id)
                else:
                    test_names.append(image_id)
        (split_dir / "train.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
        (split_dir / "test.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")
        manifest["domains"][domain_name] = {
            "path": str(domain_root.resolve()),
            "train_images": len(train_names),
            "test_images": len(test_names),
        }
    atomic_json_dump(manifest, root / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

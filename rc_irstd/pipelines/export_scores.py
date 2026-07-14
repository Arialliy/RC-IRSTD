from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rc_irstd.data.dataset import IRSTDDataset, collate_samples
from rc_irstd.data.score_records import ScoreRecord, save_score_record
from rc_irstd.data.transforms import load_image_preserve_depth, pad_tensor_to_stride
from rc_irstd.engine.worker_seed import make_generator, seed_worker
from rc_irstd.features.image_stats import compute_image_statistics
from rc_irstd.models.detector_adapter import build_detector, resize_logits
from rc_irstd.utils.arguments import parse_hw
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir
from rc_irstd.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export continuous detector score maps.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--detector", default="mshnet", choices=["mshnet", "mshnet_external", "tiny"]
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--inference-mode",
        choices=["resize", "native_pad", "tiled"],
        default="resize",
    )
    parser.add_argument("--resize", nargs=2, type=int, default=[256, 256], metavar=("H", "W"))
    parser.add_argument("--stride-multiple", type=int, default=32)
    parser.add_argument("--tile-size", nargs=2, type=int, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument(
        "--restore-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For resize mode, interpolate probabilities back to original size.",
    )
    parser.add_argument(
        "--include-mask", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--normalization",
        choices=["imagenet", "minmax", "percentile", "none"],
        default="imagenet",
    )
    parser.add_argument(
        "--dataset-type", choices=["iid_images", "temporal"], default="iid_images"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser


def _native_logits(model, image: torch.Tensor, stride: int) -> torch.Tensor:
    padded, original_hw = pad_tensor_to_stride(image, stride=stride)
    output = model(padded, training_tag=True)
    logits = resize_logits(output.logits, tuple(padded.shape[-2:]))
    return logits[..., : original_hw[0], : original_hw[1]]


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0 or overlap < 0 or overlap >= tile:
        raise ValueError("tile must be positive and overlap in [0, tile)")
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, max(length - tile + 1, 1), step))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


def _tiled_logits(
    model,
    image: torch.Tensor,
    tile_hw: tuple[int, int],
    overlap: int,
    stride: int,
) -> torch.Tensor:
    if image.shape[0] != 1:
        raise ValueError("tiled inference requires batch_size=1")
    height, width = image.shape[-2:]
    tile_h, tile_w = tile_hw
    sum_logits = image.new_zeros((1, 1, height, width))
    counts = image.new_zeros((1, 1, height, width))
    for y in _tile_starts(height, tile_h, overlap):
        for x in _tile_starts(width, tile_w, overlap):
            patch = image[..., y : min(y + tile_h, height), x : min(x + tile_w, width)]
            patch_logits = _native_logits(model, patch, stride)
            patch_h, patch_w = patch_logits.shape[-2:]
            sum_logits[..., y : y + patch_h, x : x + patch_w] += patch_logits
            counts[..., y : y + patch_h, x : x + patch_w] += 1.0
    return sum_logits / counts.clamp_min(1.0)


def _infer_logits(model, images: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.inference_mode == "resize":
        output = model(images, training_tag=True)
        return resize_logits(output.logits, tuple(images.shape[-2:]))
    if args.inference_mode == "native_pad":
        return _native_logits(model, images, args.stride_multiple)
    return _tiled_logits(
        model,
        images,
        tuple(int(value) for value in args.tile_size),
        args.tile_overlap,
        args.stride_multiple,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    resize_hw = parse_hw(args.resize) if args.inference_mode == "resize" else None
    output_dir = ensure_dir(args.output_dir)

    dataset = IRSTDDataset(
        args.dataset_dir,
        split=args.split,
        resize_hw=resize_hw,
        augment=False,
        require_mask=args.include_mask,
        normalization=args.normalization,
        dataset_type=args.dataset_type,
        include_component_labels=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed),
    )
    model = build_detector(
        args.detector,
        checkpoint=args.checkpoint,
        device=device,
        strict=True,
    )
    model.eval()

    count = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="export scores"):
            images = batch["image"].to(device, non_blocking=True)
            logits = _infer_logits(model, images, args)
            probability = torch.sigmoid(logits)
            meta = batch["meta"][0]
            if (
                args.inference_mode == "resize"
                and args.restore_original
                and tuple(probability.shape[-2:]) != meta.original_hw
            ):
                probability = F.interpolate(
                    probability,
                    size=meta.original_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            probability_np = np.clip(
                probability[0, 0].detach().cpu().numpy().astype(np.float32), 0.0, 1.0
            )

            loaded_image = load_image_preserve_depth(meta.image_path)
            image_stats, image_stat_names = compute_image_statistics(loaded_image.array)
            mask = None
            if args.include_mask:
                mask_batch = batch["mask"]
                if mask_batch is None:
                    raise RuntimeError("--include-mask was set but no mask was loaded")
                mask_tensor = mask_batch
                if tuple(mask_tensor.shape[-2:]) != tuple(probability_np.shape):
                    mask_tensor = F.interpolate(
                        mask_tensor.float(), size=probability_np.shape, mode="nearest"
                    )
                mask = (mask_tensor[0, 0].numpy() > 0.5).astype(np.uint8)

            record = ScoreRecord(
                probability=probability_np,
                mask=mask,
                image_stats=image_stats,
                image_stat_names=image_stat_names,
                image_id=meta.image_id,
                dataset_name=meta.dataset_name,
                sequence_id=meta.sequence_id,
                frame_index=meta.frame_index,
                original_hw=meta.original_hw,
                source_checkpoint=str(Path(args.checkpoint).resolve()),
                dataset_type=meta.dataset_type,
                inference_mode=args.inference_mode,
            )
            save_score_record(record, output_dir / f"{count:08d}.npz")
            count += 1

    manifest = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "dataset_name": dataset.dataset_name,
        "dataset_type": args.dataset_type,
        "split": args.split,
        "detector": args.detector,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "inference_mode": args.inference_mode,
        "resize_hw": resize_hw,
        "restore_original": args.restore_original,
        "stride_multiple": args.stride_multiple,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "normalization": args.normalization,
        "include_mask": args.include_mask,
        "score_type": "sigmoid_probability",
        "num_images": count,
        "risk_candidate_definition": "threshold-independent deterministic local maxima",
    }
    atomic_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

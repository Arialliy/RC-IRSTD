from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adagrad, AdamW
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from rc_irstd.data.dataset import IRSTDDataset, collate_samples
from rc_irstd.data.sampler import DomainBalancedBatchSampler
from rc_irstd.engine.worker_seed import (
    capture_rng_state,
    make_generator,
    restore_rng_state,
    seed_worker,
)
from rc_irstd.evaluation.curves import compute_image_curves
from rc_irstd.evaluation.detector_selection import (
    DetectorBudgetSelection,
    summarise_detector_budget,
    validation_threshold_grid,
)
from rc_irstd.evaluation.segmentation import evaluate_binary_segmentation
from rc_irstd.losses.risk_aware import RiskAwareDetectorLoss, fallback_segmentation_loss
from rc_irstd.losses.target_background_margin import DomainTailSeparationDetectorLoss
from rc_irstd.losses.sls import SLSIoULoss
from rc_irstd.models.detector_adapter import build_detector, resize_logits
from rc_irstd.utils.arguments import parse_hw
from rc_irstd.utils.checkpoint import atomic_torch_save
from rc_irstd.utils.device import autocast_context, create_grad_scaler, resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir, normalise_state_dict
from rc_irstd.utils.logging import JsonlLogger
from rc_irstd.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a domain-balanced, risk-aware IRSTD detector."
    )
    parser.add_argument(
        "--source-dataset",
        action="append",
        required=True,
        help="BasicIRSTD-style source directory; repeat for multiple domains.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument(
        "--val-split",
        default=None,
        help=(
            "Source-internal validation split. There is deliberately no fallback "
            "to official test."
        ),
    )
    parser.add_argument("--source-train-split", action="append", default=None)
    parser.add_argument("--source-val-split", action="append", default=None)
    parser.add_argument(
        "--detector",
        default="mshnet",
        choices=["mshnet", "mshnet_external", "tiny"],
        help="mshnet is the self-contained implementation bundled in this package.",
    )
    parser.add_argument(
        "--base-loss",
        default="auto",
        choices=["auto", "sls", "bce_dice"],
    )
    parser.add_argument("--resize", nargs=2, type=int, default=[256, 256], metavar=("H", "W"))
    parser.add_argument(
        "--normalization",
        choices=["imagenet", "minmax", "percentile", "none"],
        default="imagenet",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["iid_images", "temporal"],
        default="iid_images",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--warm-epoch", type=int, default=5)
    parser.add_argument("--optimizer", choices=["adagrad", "adamw"], default="adagrad")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--detector-objective",
        choices=["domain_tail_separation", "baseline", "legacy_tail_miss"],
        default="domain_tail_separation",
    )
    parser.add_argument("--lambda-sep", type=float, default=0.20)
    parser.add_argument("--separation-margin", type=float, default=1.0)
    parser.add_argument("--background-tail-fraction", type=float, default=0.05)
    parser.add_argument("--object-top-fraction", type=float, default=0.25)
    parser.add_argument("--hard-object-fraction", type=float, default=0.25)
    parser.add_argument("--risk-start-epoch", type=int, default=5)
    parser.add_argument("--risk-ramp-epochs", type=int, default=10)
    parser.add_argument("--lambda-tail", type=float, default=0.10)
    parser.add_argument("--lambda-miss", type=float, default=0.10)
    parser.add_argument("--tail-quantile", type=float, default=0.95)
    parser.add_argument("--miss-quantile", type=float, default=0.80)
    parser.add_argument("--peak-kernel", type=int, default=5)
    parser.add_argument("--exclusion-radius", type=int, default=2)
    parser.add_argument("--worst-gamma", type=float, default=10.0)
    parser.add_argument("--auxiliary-weight", type=float, default=1.0)
    parser.add_argument(
        "--pixel-budget",
        type=float,
        default=1e-5,
        help="Source-validation budget used to select best_budget.pt.",
    )
    parser.add_argument(
        "--peak-budget",
        type=float,
        default=5.0,
        help="Fixed false local peaks/MP budget used for checkpoint selection.",
    )
    parser.add_argument(
        "--selection-use-peak-constraint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Main path is pixel-only; enable only for the fixed-candidate compatibility baseline.",
    )
    parser.add_argument("--selection-grid-points", type=int, default=96)
    parser.add_argument("--selection-peak-min-distance", type=int, default=2)
    parser.add_argument("--selection-peak-tolerance", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--engineering-smoke-no-validation",
        action="store_true",
        help=(
            "Run fixed-last engineering only without constructing a validation "
            "loader. Artifacts are non-claim-bearing and cannot enter model selection."
        ),
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Engineering-smoke step cap; requires --engineering-smoke-no-validation.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def _build_base_loss(name: str, detector_name: str):
    requested = name
    if requested == "auto":
        requested = "sls" if detector_name.startswith("mshnet") else "bce_dice"
    if requested == "bce_dice":
        return fallback_segmentation_loss, "bce_dice"
    if detector_name == "mshnet_external":
        try:
            from model.loss import SLSIoULoss as ExternalSLSIoULoss  # type: ignore

            return ExternalSLSIoULoss(), "sls_external"
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "mshnet_external with SLS requires model.loss.SLSIoULoss on PYTHONPATH"
            ) from exc
    return SLSIoULoss(), "sls_internal"


def _source_splits(args: argparse.Namespace, kind: str) -> list[str]:
    values = args.source_train_split if kind == "train" else args.source_val_split
    fallback = args.train_split if kind == "train" else args.val_split
    if values is None:
        if fallback is None:
            raise ValueError(
                "Source-internal validation split is required; official test is "
                "never used as an implicit validation fallback"
            )
        return [str(fallback)] * len(args.source_dataset)
    if len(values) != len(args.source_dataset):
        raise ValueError(
            f"--source-{kind}-split must occur once per --source-dataset "
            f"({len(args.source_dataset)} required, got {len(values)})"
        )
    return [str(value) for value in values]


def _make_datasets(args: argparse.Namespace, split_kind: str, augment: bool):
    resize_hw = parse_hw(args.resize)
    splits = _source_splits(args, split_kind)
    return [
        IRSTDDataset(
            path,
            split=split,
            resize_hw=resize_hw,
            augment=augment,
            domain_id=domain_id,
            require_mask=True,
            normalization=args.normalization,
            dataset_type=args.dataset_type,
            include_component_labels=True,
        )
        for domain_id, (path, split) in enumerate(
            zip(args.source_dataset, splits, strict=True)
        )
    ]


def _make_train_loader(args: argparse.Namespace) -> tuple[DataLoader, DomainBalancedBatchSampler]:
    datasets = _make_datasets(args, "train", augment=True)
    concatenated = ConcatDataset(datasets)
    domain_ids: list[int] = []
    for domain_id, dataset in enumerate(datasets):
        domain_ids.extend([domain_id] * len(dataset))
    sampler = DomainBalancedBatchSampler(
        domain_ids,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        seed=args.seed,
    )
    loader = DataLoader(
        concatenated,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed),
    )
    return loader, sampler


def _make_val_loader(args: argparse.Namespace) -> DataLoader:
    datasets = _make_datasets(args, "val", augment=False)
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed + 1),
    )


def _build_optimizer(args: argparse.Namespace, parameters):
    if args.optimizer == "adagrad":
        return Adagrad(parameters, lr=args.lr, weight_decay=args.weight_decay)
    return AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)


def _validate(
    model,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    warm_epoch: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float], DetectorBudgetSelection]:
    model.eval()
    intersection = union = false_pixels = total_pixels = 0
    detected_objects = gt_objects = false_components = 0
    thresholds = validation_threshold_grid(args.selection_grid_points)
    domain_curves: dict[str, list[Any]] = {}

    with torch.inference_mode():
        for batch in tqdm(loader, desc="validate", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"]
            if masks is None:
                raise RuntimeError("Validation requires masks")
            masks = masks.to(device, non_blocking=True)
            output = model(images, training_tag=epoch >= warm_epoch)
            logits = resize_logits(output.logits, tuple(masks.shape[-2:]))
            probabilities = torch.sigmoid(logits)
            prediction = probabilities >= 0.5
            for pred, probability, target, meta in zip(
                prediction, probabilities, masks, batch["meta"], strict=True
            ):
                target_np = target.detach().cpu().numpy()
                metrics = evaluate_binary_segmentation(
                    pred.detach().cpu().numpy(), target_np
                )
                intersection += metrics.intersection
                union += metrics.union
                false_pixels += metrics.false_positive_pixels
                total_pixels += int(target.numel())
                detected_objects += metrics.detected_objects
                gt_objects += metrics.gt_objects
                false_components += metrics.false_components
                curve = compute_image_curves(
                    probability[0].detach().cpu().numpy(),
                    target_np,
                    thresholds,
                    peak_min_distance=args.selection_peak_min_distance,
                    peak_tolerance=args.selection_peak_tolerance,
                )
                domain_curves.setdefault(meta.dataset_name, []).append(curve)

    metrics = {
        "iou": float(intersection / max(union, 1)),
        "pd_0p5": float(detected_objects / max(gt_objects, 1)),
        "fa_pixel_0p5": float(false_pixels / max(total_pixels, 1)),
        "fa_component_per_mp_0p5": float(
            false_components / max(total_pixels / 1_000_000.0, 1e-12)
        ),
    }
    selection = summarise_detector_budget(
        domain_curves,
        thresholds,
        args.pixel_budget,
        args.peak_budget,
        use_peak_constraint=args.selection_use_peak_constraint,
    )
    metrics.update(
        {
            "budget_mean_domain_pd": selection.mean_domain_pd,
            "budget_worst_domain_pd": selection.worst_domain_pd,
            "budget_empty_prediction_rate": selection.rejection_rate,
            "budget_mean_threshold": selection.mean_threshold,
        }
    )
    return metrics, selection


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_iou: float,
    best_budget_key: tuple[float, float, float, float],
    resolved_base_loss: str,
    args: argparse.Namespace,
    validation: dict[str, float] | None,
    budget_selection: DetectorBudgetSelection | None,
) -> dict[str, Any]:
    return {
        "model": model.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "rng_state": capture_rng_state(),
        "epoch": epoch,
        "best_iou": best_iou,
        "best_budget_key": list(best_budget_key),
        "detector": args.detector,
        "base_loss": resolved_base_loss,
        "validation": validation,
        "budget_selection": (
            budget_selection.to_dict() if budget_selection is not None else None
        ),
        "arguments": vars(args),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.batch_size % len(args.source_dataset) != 0:
        raise ValueError(
            "batch-size must be divisible by the number of source domains"
        )
    if args.pixel_budget <= 0 or args.peak_budget <= 0:
        raise ValueError("selection budgets must be positive")
    if args.max_train_steps is not None:
        if args.max_train_steps <= 0:
            raise ValueError("--max-train-steps must be positive")
        if not args.engineering_smoke_no_validation:
            raise ValueError(
                "--max-train-steps is restricted to non-claim-bearing engineering smoke"
            )
    _source_splits(args, "train")
    if not args.engineering_smoke_no_validation:
        _source_splits(args, "val")

    seed_everything(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    requested_output = Path(args.output_dir)
    if (
        requested_output.is_dir()
        and any(requested_output.iterdir())
        and args.resume is None
    ):
        raise FileExistsError(
            f"Refusing to mix a new run into non-empty output directory: {requested_output}"
        )
    output_dir = ensure_dir(requested_output)
    run_scope = (
        "engineering_smoke_fixed_last_no_validation"
        if args.engineering_smoke_no_validation
        else "source_internal_validation_candidate"
    )
    arguments_record = {**vars(args), "run_scope": run_scope}
    atomic_json_dump(arguments_record, output_dir / "arguments.json")
    logger = JsonlLogger(output_dir / "metrics.jsonl")

    train_loader, batch_sampler = _make_train_loader(args)
    val_loader = (
        None if args.engineering_smoke_no_validation else _make_val_loader(args)
    )
    model = build_detector(args.detector, device=device)
    base_loss, resolved_base_loss = _build_base_loss(args.base_loss, args.detector)
    if args.detector_objective in {"domain_tail_separation", "baseline"}:
        criterion = DomainTailSeparationDetectorLoss(
            base_loss=base_loss,
            lambda_sep=(args.lambda_sep if args.detector_objective == "domain_tail_separation" else 0.0),
            margin=args.separation_margin,
            background_tail_fraction=args.background_tail_fraction,
            object_top_fraction=args.object_top_fraction,
            hard_object_fraction=args.hard_object_fraction,
            peak_kernel_size=args.peak_kernel,
            exclusion_radius=args.exclusion_radius,
            worst_gamma=args.worst_gamma,
            risk_start_epoch=args.risk_start_epoch,
            risk_ramp_epochs=args.risk_ramp_epochs,
            auxiliary_weight=args.auxiliary_weight,
        )
    else:
        criterion = RiskAwareDetectorLoss(
            base_loss=base_loss,
            lambda_tail=args.lambda_tail,
            lambda_miss=args.lambda_miss,
            tail_quantile=args.tail_quantile,
            miss_quantile=args.miss_quantile,
            peak_kernel=args.peak_kernel,
            exclusion_radius=args.exclusion_radius,
            worst_gamma=args.worst_gamma,
            auxiliary_weight=args.auxiliary_weight,
        )
    optimizer = _build_optimizer(args, (p for p in model.parameters() if p.requires_grad))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    scaler = create_grad_scaler(device, args.amp)

    start_epoch = 0
    best_iou = -math.inf
    best_budget_key = (-math.inf, -math.inf, -math.inf, -math.inf)
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.model.load_state_dict(normalise_state_dict(payload), strict=True)
        if isinstance(payload, dict) and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
            if "scheduler" in payload:
                scheduler.load_state_dict(payload["scheduler"])
            if "scaler" in payload and payload["scaler"] is not None:
                scaler.load_state_dict(payload["scaler"])
            start_epoch = int(payload.get("epoch", -1)) + 1
            best_iou = float(payload.get("best_iou", best_iou))
            loaded_key = payload.get("best_budget_key")
            if loaded_key is not None:
                best_budget_key = tuple(float(value) for value in loaded_key)  # type: ignore[assignment]
            restore_rng_state(payload.get("rng_state"))

    for epoch in range(start_epoch, args.epochs):
        batch_sampler.set_epoch(epoch)
        model.train()
        totals: dict[str, float] = {}
        sample_count = 0
        optimizer_steps = 0
        gradient_norm_sum = 0.0
        gradient_norm_max = 0.0
        progress = tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress):
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"]
            if masks is None:
                raise RuntimeError("Detector training requires masks")
            masks = masks.to(device, non_blocking=True)
            component_labels = batch.get("component_labels")
            if component_labels is not None:
                component_labels = component_labels.to(device, non_blocking=True)
            domain_ids = batch["domain_id"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                output = model(images, training_tag=epoch >= args.warm_epoch)
                logits = resize_logits(output.logits, tuple(masks.shape[-2:]))
                loss_terms = criterion(
                    logits,
                    masks,
                    domain_ids,
                    auxiliary_logits=output.auxiliary_logits,
                    component_labels=component_labels,
                    warm_epoch=args.warm_epoch,
                    epoch=epoch,
                )
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(
                    f"non-finite detector logits at epoch={epoch}, step={step}"
                )
            if any(not bool(torch.isfinite(value)) for value in loss_terms.values()):
                raise FloatingPointError(
                    f"non-finite detector loss at epoch={epoch}, step={step}"
                )
            scaler.scale(loss_terms["total"]).backward()
            scaler.unscale_(optimizer)
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip if args.grad_clip > 0 else float("inf"),
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            if any(
                not bool(torch.isfinite(parameter).all())
                for parameter in model.parameters()
            ):
                raise FloatingPointError(
                    f"non-finite detector parameter after epoch={epoch}, step={step}"
                )
            gradient_value = float(gradient_norm.detach())
            gradient_norm_sum += gradient_value
            gradient_norm_max = max(gradient_norm_max, gradient_value)
            optimizer_steps += 1

            current_batch = images.shape[0]
            sample_count += current_batch
            if not totals:
                totals = {key: 0.0 for key in loss_terms}
            for key in totals:
                totals[key] += float(loss_terms[key].detach()) * current_batch
            progress.set_postfix(loss=f"{totals['total'] / sample_count:.4f}")

        scheduler.step()
        if optimizer_steps == 0:
            raise RuntimeError("training epoch produced zero optimizer steps")
        record: dict[str, Any] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "base_loss": resolved_base_loss,
            "run_scope": run_scope,
            "optimizer_steps": optimizer_steps,
            "gradient_norm_mean": gradient_norm_sum / optimizer_steps,
            "gradient_norm_max": gradient_norm_max,
            "gradients_finite": True,
            "parameters_finite_after_step": True,
            **{
                f"train_{key}": value / max(sample_count, 1)
                for key, value in totals.items()
            },
        }
        validation: dict[str, float] | None = None
        budget_selection: DetectorBudgetSelection | None = None
        if val_loader is not None and (
            (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
        ):
            validation, budget_selection = _validate(
                model, val_loader, device, epoch, args.warm_epoch, args
            )
            record.update({f"val_{key}": value for key, value in validation.items()})
            record["val_budget_domains"] = [
                point.__dict__ for point in budget_selection.domain_points
            ]

        is_best_iou = validation is not None and validation["iou"] > best_iou
        if is_best_iou:
            best_iou = validation["iou"]
        current_budget_key = (
            budget_selection.rank_key(validation["iou"])
            if budget_selection is not None and validation is not None
            else (-math.inf, -math.inf, -math.inf, -math.inf)
        )
        is_best_budget = current_budget_key > best_budget_key
        if is_best_budget:
            best_budget_key = current_budget_key

        checkpoint = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_iou,
            best_budget_key,
            resolved_base_loss,
            args,
            validation,
            budget_selection,
        )
        atomic_torch_save(checkpoint, output_dir / "last.pt")
        if (epoch + 1) % args.save_every == 0:
            atomic_torch_save(checkpoint, output_dir / f"epoch_{epoch + 1:04d}.pt")
        if is_best_iou:
            atomic_torch_save(checkpoint, output_dir / "best_iou.pt")
        if is_best_budget:
            atomic_torch_save(checkpoint, output_dir / "best_budget.pt")
            # Backward-compatible default consumed by existing export/smoke code.
            atomic_torch_save(checkpoint, output_dir / "best.pt")
            if budget_selection is not None:
                atomic_json_dump(
                    budget_selection.to_dict(), output_dir / "best_budget_metrics.json"
                )

        record["is_best_iou"] = is_best_iou
        record["is_best_budget"] = is_best_budget
        logger.log(record)
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()

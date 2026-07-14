from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from rc_irstd.engine.worker_seed import (
    capture_rng_state,
    make_generator,
    restore_rng_state,
    seed_worker,
)
from rc_irstd.episodes.dataset import (
    EpisodeArrays,
    RiskCurveDataset,
    concatenate_episode_files,
)
from rc_irstd.episodes.splits import grouped_train_val_split
from rc_irstd.evaluation.risk_curve_metrics import evaluate_risk_curve_predictions
from rc_irstd.losses.quantile import budget_focused_weight, crossing_loss, pinball_loss
from rc_irstd.models.risk_curve import FeatureNormaliser, RiskCurvePredictor
from rc_irstd.utils.checkpoint import atomic_torch_save
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir
from rc_irstd.utils.logging import JsonlLogger
from rc_irstd.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a budget-focused monotone upper-quantile risk curve."
    )
    parser.add_argument("--train-episode", action="append", required=True)
    parser.add_argument("--val-episode", action="append", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lambda-peak", type=float, default=1.0)
    parser.add_argument("--lambda-crossing", type=float, default=0.25)
    parser.add_argument("--crossing-temperature", type=float, default=0.25)
    parser.add_argument("--focus-base-weight", type=float, default=1.0)
    parser.add_argument("--focus-weight", type=float, default=4.0)
    parser.add_argument("--focus-log-scale", type=float, default=1.0)
    parser.add_argument("--empty-action-weight", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--pixel-budget", type=float, default=1e-6)
    parser.add_argument("--peak-budget", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def _split_arrays(args: argparse.Namespace) -> tuple[EpisodeArrays, EpisodeArrays]:
    train_all = concatenate_episode_files(args.train_episode)
    if args.val_episode:
        validation = concatenate_episode_files(args.val_episode)
        if not np.array_equal(train_all.thresholds, validation.thresholds):
            raise ValueError("Training and validation threshold grids differ")
        if train_all.feature_names != validation.feature_names:
            raise ValueError("Training and validation feature schemas differ")
        if train_all.feature_config != validation.feature_config:
            raise ValueError("Training and validation feature configurations differ")
        return train_all, validation
    train_indices, val_indices = grouped_train_val_split(
        train_all, val_fraction=args.val_fraction, seed=args.seed
    )
    return train_all.subset(train_indices), train_all.subset(val_indices)


def _predict(
    model,
    dataset: RiskCurveDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs: dict[str, list[np.ndarray]] = {
        "pixel_log_risk": [],
        "peak_log_risk": [],
    }
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            prediction = model(batch["features"].to(device))
            for key in outputs:
                outputs[key].append(prediction[key].cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outputs.items()}


def _loss_terms(
    prediction: dict[str, torch.Tensor],
    target_pixel: torch.Tensor,
    target_peak: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    pixel_weight = budget_focused_weight(
        target_pixel,
        args.pixel_budget,
        args.focus_base_weight,
        args.focus_weight,
        args.focus_log_scale,
        args.empty_action_weight,
    )
    peak_weight = budget_focused_weight(
        target_peak,
        args.peak_budget,
        args.focus_base_weight,
        args.focus_weight,
        args.focus_log_scale,
        args.empty_action_weight,
    )
    pixel_pinball = pinball_loss(
        prediction["pixel_log_risk"], target_pixel, args.quantile, pixel_weight
    )
    peak_pinball = pinball_loss(
        prediction["peak_log_risk"], target_peak, args.quantile, peak_weight
    )
    pixel_crossing = crossing_loss(
        prediction["pixel_log_risk"],
        target_pixel,
        args.pixel_budget,
        args.crossing_temperature,
        pixel_weight,
    )
    peak_crossing = crossing_loss(
        prediction["peak_log_risk"],
        target_peak,
        args.peak_budget,
        args.crossing_temperature,
        peak_weight,
    )
    pinball = pixel_pinball + args.lambda_peak * peak_pinball
    crossing = pixel_crossing + args.lambda_peak * peak_crossing
    total = pinball + args.lambda_crossing * crossing
    return {
        "total": total,
        "pinball": pinball,
        "crossing": crossing,
        "pixel_pinball": pixel_pinball,
        "peak_pinball": peak_pinball,
    }


def _checkpoint_payload(
    model: RiskCurvePredictor,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    best_selected_key: tuple[float, float, float, float],
    best_selected_epoch: int,
    best_pinball: float,
    normaliser: FeatureNormaliser,
    arrays: EpisodeArrays,
    args: argparse.Namespace,
    validation_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "epoch": epoch,
        "best_selected_key": list(best_selected_key),
        "best_selected_epoch": best_selected_epoch,
        "best_pinball": best_pinball,
        "normaliser": normaliser.to_dict(),
        "thresholds": arrays.thresholds.tolist(),
        "feature_names": list(arrays.feature_names),
        "feature_config": dict(arrays.feature_config),
        "model_config": {
            "input_dim": int(arrays.features.shape[1]),
            "num_thresholds": int(arrays.thresholds.shape[0]),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
        "training_config": vars(args),
        "validation_metrics": validation_metrics,
        "risk_definition": "pixel false rate and fixed false local peaks per megapixel",
        "guarantee_note": (
            "The model is an empirical conditional upper-quantile estimator. "
            "Any finite-sample marginal statement belongs to the optional CRC stage."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    if args.pixel_budget <= 0 or args.peak_budget <= 0:
        raise ValueError("budgets must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    atomic_json_dump(vars(args), output_dir / "arguments.json")
    logger = JsonlLogger(output_dir / "metrics.jsonl")

    train_arrays, val_arrays = _split_arrays(args)
    normaliser = FeatureNormaliser.fit(train_arrays.features)
    train_dataset = RiskCurveDataset(train_arrays, normaliser.mean, normaliser.std)
    val_dataset = RiskCurveDataset(val_arrays, normaliser.mean, normaliser.std)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=make_generator(args.seed),
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )

    model = RiskCurvePredictor(
        input_dim=train_arrays.features.shape[1],
        num_thresholds=train_arrays.thresholds.shape[0],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(5, args.patience // 4)
    )

    start_epoch = 0
    best_selected_key = (-math.inf, -math.inf, -math.inf, -math.inf)
    best_selected_epoch = -1
    best_pinball = math.inf
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        loaded_key = payload.get("best_selected_key")
        if loaded_key is not None:
            best_selected_key = tuple(float(value) for value in loaded_key)  # type: ignore[assignment]
        best_selected_epoch = int(payload.get("best_selected_epoch", -1))
        best_pinball = float(payload.get("best_pinball", math.inf))
        restore_rng_state(payload.get("rng_state"))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {"total": 0.0, "pinball": 0.0, "crossing": 0.0}
        count = 0
        for batch in tqdm(train_loader, desc=f"curve {epoch + 1}/{args.epochs}", leave=False):
            features = batch["features"].to(device)
            target_pixel = batch["pixel_log_risk"].to(device)
            target_peak = batch["peak_log_risk"].to(device)
            prediction = model(features)
            terms = _loss_terms(prediction, target_pixel, target_peak, args)
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            current = len(features)
            count += current
            for key in running:
                running[key] += float(terms[key].detach()) * current

        predictions = _predict(model, val_dataset, device, args.batch_size)
        val_prediction = {
            "pixel_log_risk": torch.from_numpy(predictions["pixel_log_risk"]),
            "peak_log_risk": torch.from_numpy(predictions["peak_log_risk"]),
        }
        val_terms = _loss_terms(
            val_prediction,
            torch.from_numpy(val_arrays.pixel_log_risk),
            torch.from_numpy(val_arrays.peak_log_risk),
            args,
        )
        val_objective = float(val_terms["total"])
        val_pinball = float(val_terms["pinball"])
        scheduler.step(val_objective)
        metrics, _, _ = evaluate_risk_curve_predictions(
            val_arrays.thresholds,
            predictions["pixel_log_risk"],
            predictions["peak_log_risk"],
            val_arrays.pixel_log_risk,
            val_arrays.peak_log_risk,
            val_arrays.pixel_risk,
            val_arrays.peak_risk,
            val_arrays.pd,
            val_arrays.domains,
            args.pixel_budget,
            args.peak_budget,
        )
        selected = metrics.selected
        normalised_excess = (
            selected.pixel_excess / args.pixel_budget
            + selected.peak_excess / args.peak_budget
        )
        current_key = (
            -float(normalised_excess),
            float(selected.effective_pd_with_rejects),
            float(selected.joint_bsr),
            -val_objective,
        )
        is_best_selected = current_key > best_selected_key
        if is_best_selected:
            best_selected_key = current_key
            best_selected_epoch = epoch
        is_best_pinball = val_pinball < best_pinball
        if is_best_pinball:
            best_pinball = val_pinball

        record = {
            "epoch": epoch,
            "train_total": running["total"] / max(count, 1),
            "train_pinball": running["pinball"] / max(count, 1),
            "train_crossing": running["crossing"] / max(count, 1),
            "val_objective": val_objective,
            "val_pinball": val_pinball,
            "val_crossing": float(val_terms["crossing"]),
            "normalised_selected_excess": normalised_excess,
            "lr": optimizer.param_groups[0]["lr"],
            **metrics.to_dict(),
            "is_best_selected": is_best_selected,
            "is_best_pinball": is_best_pinball,
        }
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            best_selected_key,
            best_selected_epoch,
            best_pinball,
            normaliser,
            train_arrays,
            args,
            metrics.to_dict(),
        )
        atomic_torch_save(payload, output_dir / "last.pt")
        if is_best_selected:
            atomic_torch_save(payload, output_dir / "best_selected.pt")
            atomic_torch_save(payload, output_dir / "best.pt")
        if is_best_pinball:
            atomic_torch_save(payload, output_dir / "best_pinball.pt")
        logger.log(record)
        print(json.dumps(record, ensure_ascii=False))

        if epoch - best_selected_epoch >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; selected-point best was "
                f"epoch {best_selected_epoch}."
            )
            break


if __name__ == "__main__":
    main()

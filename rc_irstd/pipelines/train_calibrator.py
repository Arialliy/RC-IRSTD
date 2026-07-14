from __future__ import annotations

"""Train the final multi-budget, no-reject RC-IRSTD calibrator."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from rc_irstd.engine.worker_seed import make_generator, seed_worker
from rc_irstd.episodes.meta_dataset import (
    MetaEpisodeArrays,
    MultiBudgetMetaDataset,
    concatenate_meta_episode_files,
)
from rc_irstd.evaluation.calibrator_replay import HardReplayEvaluator
from rc_irstd.losses.calibrator import risk_aligned_calibrator_loss
from rc_irstd.models.monotone_pixel_calibrator import (
    MonotonePixelCalibrator,
    assert_structural_monotonicity,
)
from rc_irstd.models.risk_curve import FeatureNormaliser
from rc_irstd.utils.checkpoint import atomic_torch_save
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir
from rc_irstd.utils.logging import JsonlLogger
from rc_irstd.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the no-reject monotone inverse pixel-risk calibrator."
    )
    parser.add_argument("--train-meta", action="append", required=True)
    parser.add_argument("--val-meta", action="append", required=True)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--source-hidden-dim", type=int, default=32)
    parser.add_argument("--source-output-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--min-logit-step", type=float, default=0.0)
    parser.add_argument("--lambda-violation", type=float, default=4.0)
    parser.add_argument("--lambda-utility", type=float, default=1.0)
    parser.add_argument("--lambda-oracle", type=float, default=0.10)
    parser.add_argument("--lambda-smoothness", type=float, default=0.01)
    parser.add_argument("--pixel-temperature", type=float, default=0.10)
    parser.add_argument("--object-temperature", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def _json_id_set(values: np.ndarray) -> set[str]:
    result: set[str] = set()
    for value in values:
        parsed = json.loads(str(value))
        if not isinstance(parsed, list):
            raise ValueError("Episode ID fields must encode JSON lists")
        result.update(str(item) for item in parsed)
    return result


def _validate_protocol(train: MetaEpisodeArrays, validation: MetaEpisodeArrays) -> None:
    if not np.array_equal(train.budgets, validation.budgets):
        raise ValueError("Train and validation budget grids differ")
    if train.feature_names != validation.feature_names:
        raise ValueError("Train and validation feature schemas differ")
    if train.feature_config != validation.feature_config:
        raise ValueError("Train and validation feature configurations differ")
    train_ids = _json_id_set(train.support_ids) | _json_id_set(train.query_ids)
    validation_ids = _json_id_set(validation.support_ids) | _json_id_set(validation.query_ids)
    overlap = train_ids.intersection(validation_ids)
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(
            "Meta train/validation image leakage detected; examples: " + ", ".join(examples)
        )


def _loader(
    arrays: MetaEpisodeArrays,
    normaliser: FeatureNormaliser,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = MultiBudgetMetaDataset(arrays, normaliser.mean, normaliser.std)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        persistent_workers=args.num_workers > 0,
    )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _forward_loss(
    model: MonotonePixelCalibrator,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
):
    prediction = model(
        batch["features"],
        batch.get("source_distances"),
        batch.get("source_distance_valid"),
    )
    assert_structural_monotonicity(prediction["threshold_logit"])
    loss = risk_aligned_calibrator_loss(
        prediction["threshold_logit"],
        batch["budgets"][0],
        batch["oracle_threshold_logit"],
        batch["background_logits"],
        batch["background_valid"],
        batch["background_fraction"],
        batch["object_scores"],
        batch["object_valid"],
        lambda_violation=args.lambda_violation,
        lambda_utility=args.lambda_utility,
        lambda_oracle=args.lambda_oracle,
        lambda_smoothness=args.lambda_smoothness,
        pixel_temperature=args.pixel_temperature,
        object_temperature=args.object_temperature,
    )
    return prediction, loss


def _surrogate_validation(
    model: MonotonePixelCalibrator,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray]:
    totals = {"total": 0.0, "violation": 0.0, "utility": 0.0, "oracle": 0.0, "smoothness": 0.0}
    rows = 0
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            prediction, loss = _forward_loss(model, batch, args)
            size = batch["features"].shape[0]
            rows += size
            for key in totals:
                totals[key] += float(getattr(loss, key).detach()) * size
            predictions.append(prediction["threshold_logit"].cpu().numpy())
    return (
        {key: value / max(rows, 1) for key, value in totals.items()},
        np.concatenate(predictions, axis=0),
    )


def _checkpoint_payload(
    model: MonotonePixelCalibrator,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    normaliser: FeatureNormaliser,
    train: MetaEpisodeArrays,
    args: argparse.Namespace,
    replay: dict[str, Any] | None,
    best_rank: tuple[float, float, float],
    best_epoch: int,
) -> dict[str, Any]:
    return {
        "method": "two_stage_no_reject_monotone_inverse_pixel_risk",
        "reject_head": False,
        "risk_constraint": "original_resolution_pixel_false_alarm_rate",
        "component_fa_role": "compatibility_evaluation_only",
        "guarantee_claim": "empirical_meta_calibration_not_certified",
        "model": model.state_dict(),
        "model_config": model.config.to_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "feature_normaliser": normaliser.to_dict(),
        "feature_names": list(train.feature_names),
        "feature_config": train.feature_config,
        "budgets": train.budgets.tolist(),
        "arguments": vars(args),
        "hard_replay": replay,
        "checkpoint_selection_order": ["BSR", "LogExcess", "Pd"],
        "best_rank": list(best_rank),
        "best_epoch": int(best_epoch),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    seed_everything(args.seed, deterministic=True)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    atomic_json_dump(vars(args), output_dir / "arguments.json")
    logger = JsonlLogger(output_dir / "metrics.jsonl")

    train = concatenate_meta_episode_files(args.train_meta)
    validation = concatenate_meta_episode_files(args.val_meta)
    _validate_protocol(train, validation)
    normaliser = FeatureNormaliser.fit(train.features)
    train_loader = _loader(train, normaliser, args, shuffle=True, seed=args.seed)
    val_loader = _loader(validation, normaliser, args, shuffle=False, seed=args.seed + 1)
    replay_evaluator = HardReplayEvaluator(validation)

    model = MonotonePixelCalibrator(
        input_dim=train.features.shape[1],
        budgets=train.budgets,
        hidden_dim=args.hidden_dim,
        source_hidden_dim=args.source_hidden_dim,
        source_output_dim=args.source_output_dim,
        dropout=args.dropout,
        min_logit_step=args.min_logit_step,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, args.patience // 4)
    )

    start_epoch = 0
    best_rank = (-math.inf, -math.inf, -math.inf)
    best_epoch = -1
    epochs_without_improvement = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best_rank = tuple(float(value) for value in payload.get("best_rank", best_rank))
        best_epoch = int(payload.get("best_epoch", best_epoch))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = {"total": 0.0, "violation": 0.0, "utility": 0.0, "oracle": 0.0, "smoothness": 0.0}
        rows = 0
        progress = tqdm(train_loader, desc=f"calibrator {epoch + 1}/{args.epochs}")
        for raw_batch in progress:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = _forward_loss(model, batch, args)
            loss.total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            size = batch["features"].shape[0]
            rows += size
            for key in totals:
                totals[key] += float(getattr(loss, key).detach()) * size
            progress.set_postfix(loss=f"{totals['total'] / max(rows, 1):.4f}")

        train_metrics = {key: value / max(rows, 1) for key, value in totals.items()}
        val_surrogate, val_eta = _surrogate_validation(model, val_loader, device, args)
        replay = replay_evaluator.evaluate(val_eta)
        replay_dict = replay.to_dict()
        scheduler.step(replay.log_excess)
        rank = replay.rank_key
        improved = rank > best_rank
        if improved:
            best_rank = rank
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_surrogate_{key}": value for key, value in val_surrogate.items()},
            "val_bsr": replay.budget_satisfaction_rate,
            "val_log_excess": replay.log_excess,
            "val_pd": replay.mean_pd,
            "val_worst_domain_pd": replay.worst_domain_pd,
            "checkpoint_rank": list(rank),
            "checkpoint_selection_order": ["BSR", "LogExcess", "Pd"],
            "is_best": improved,
        }
        checkpoint = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            normaliser,
            train,
            args,
            replay_dict,
            best_rank,
            best_epoch,
        )
        atomic_torch_save(checkpoint, output_dir / "last.pt")
        if improved:
            atomic_torch_save(checkpoint, output_dir / "best.pt")
            atomic_json_dump(replay_dict, output_dir / "best_hard_replay.json")
        logger.log(record)
        print(json.dumps(record, ensure_ascii=False))
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            break

    atomic_json_dump(
        {
            "status": "completed",
            "best_epoch": best_epoch,
            "best_rank": list(best_rank),
            "selection_order": ["BSR", "LogExcess", "Pd"],
            "reject_head": False,
        },
        output_dir / "training_summary.json",
    )


if __name__ == "__main__":
    main()

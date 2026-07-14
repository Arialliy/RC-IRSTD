from __future__ import annotations

"""Small CPU smoke training for the final two-stage/no-reject implementation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rc_irstd.data.score_records import ScoreRecord, save_score_record
from rc_irstd.episodes.meta_dataset import MetaEpisodeBuildConfig, build_meta_episode_file
from rc_irstd.evaluation.calibrator_replay import HardReplayEvaluator
from rc_irstd.episodes.meta_dataset import load_meta_episode_file
from rc_irstd.losses.target_background_margin import domain_tail_separation_loss
from rc_irstd.models.calibrator_io import load_monotone_calibrator, predict_threshold_curve
from rc_irstd.pipelines.train_calibrator import main as train_calibrator_main
from rc_irstd.utils.io import atomic_json_dump, ensure_dir


def _make_records(directory: Path, prefix: str, count: int, seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        height = width = 16
        probability = np.clip(rng.beta(0.7, 18.0, size=(height, width)), 1e-5, 1 - 1e-5).astype(
            np.float32
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        if index % 3 != 0:
            y = 2 + (3 * index) % 12
            x = 2 + (5 * index) % 12
            mask[y, x] = 1
            probability[y, x] = np.float32(0.82 + 0.12 * rng.random())
        # Domain-shift-like clutter peak.
        cy = (7 * index + 1) % height
        cx = (11 * index + 3) % width
        if not mask[cy, cx]:
            probability[cy, cx] = np.float32(0.25 + 0.55 * rng.random())
        save_score_record(
            ScoreRecord(
                probability=probability,
                mask=mask,
                image_stats=np.asarray(
                    [probability.mean(), probability.std(), np.median(probability)],
                    dtype=np.float32,
                ),
                image_stat_names=("mean", "std", "median"),
                image_id=f"{prefix}_{index:04d}",
                dataset_name=prefix,
                sequence_id="iid",
                frame_index=index,
                original_hw=(height, width),
                source_checkpoint="synthetic-frozen-detector",
                dataset_type="iid_images",
            ),
            directory / f"{index:04d}.npz",
        )


def _detector_loss_smoke() -> float:
    torch.manual_seed(2)
    logits = torch.randn((4, 1, 16, 16), requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 4, 4] = 1
    target[1, 0, 8, 8] = 1
    target[2, 0, 5, 10] = 1
    domains = torch.tensor([0, 0, 1, 1])
    output = domain_tail_separation_loss(logits, target, domains)
    output.loss.backward()
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        raise RuntimeError("Domain-tail separation backward smoke failed")
    return float(output.loss.detach())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the two-stage/no-reject CPU smoke training.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = ensure_dir(args.output_dir)
    train_scores = root / "train_scores"
    val_scores = root / "val_scores"
    _make_records(train_scores, "PseudoTrain", 24, seed=4)
    _make_records(val_scores, "PseudoVal", 16, seed=9)
    budgets = [0.10, 0.05, 0.02]
    config = MetaEpisodeBuildConfig(
        context_size=4,
        horizon=4,
        stride=8,
        protocol="iid",
        background_sample_limit=1024,
        seed=5,
        split_role="synthetic_smoke",
    )
    train_meta = build_meta_episode_file(
        train_scores, root / "train_meta.npz", budgets=budgets, config=config
    )
    val_meta = build_meta_episode_file(
        val_scores, root / "val_meta.npz", budgets=budgets, config=config
    )
    calibrator_dir = root / "calibrator"
    train_calibrator_main(
        [
            "--train-meta",
            str(train_meta),
            "--val-meta",
            str(val_meta),
            "--hidden-dim",
            "32",
            "--source-output-dim",
            "16",
            "--source-hidden-dim",
            "8",
            "--dropout",
            "0",
            "--batch-size",
            "3",
            "--epochs",
            str(args.epochs),
            "--patience",
            "0",
            "--device",
            "cpu",
            "--output-dir",
            str(calibrator_dir),
        ]
    )
    arrays = load_meta_episode_file(val_meta)
    loaded = load_monotone_calibrator(calibrator_dir / "best.pt", torch.device("cpu"))
    eta = predict_threshold_curve(loaded, arrays.features, torch.device("cpu"))
    replay = HardReplayEvaluator(arrays).evaluate(eta)
    detector_loss = _detector_loss_smoke()
    summary = {
        "status": "passed",
        "method": "two_stage_no_reject",
        "detector_tail_separation_backward": True,
        "detector_tail_separation_loss": detector_loss,
        "calibrator_best_checkpoint": str((calibrator_dir / "best.pt").resolve()),
        "calibrator_reject_head": False,
        "hard_replay": replay.to_dict(),
        "train_episodes": int(len(load_meta_episode_file(train_meta).features)),
        "val_episodes": int(len(arrays.features)),
        "budget_count_per_episode": len(budgets),
    }
    atomic_json_dump(summary, root / "smoke_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

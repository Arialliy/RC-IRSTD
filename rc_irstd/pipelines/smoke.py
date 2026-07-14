from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rc_irstd.pipelines import (
    build_episodes,
    calibrate_and_evaluate,
    evaluate_zero_label,
    export_scores,
    make_synthetic_data,
    train_curve,
    train_detector,
)
from rc_irstd.utils.io import atomic_json_dump, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an in-process synthetic end-to-end RC-IRSTD smoke test."
    )
    parser.add_argument("--work-dir", default="outputs/smoke")
    parser.add_argument("--clean", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    work = Path(args.work_dir).expanduser().resolve()
    if args.clean and work.exists():
        shutil.rmtree(work)
    ensure_dir(work)
    data = work / "data"
    run = work / "run"

    make_synthetic_data.main(
        [
            "--output-root",
            str(data),
            "--domains",
            "DomainA",
            "DomainB",
            "DomainC",
            "--height",
            "16",
            "--width",
            "16",
            "--sequences",
            "4",
            "--frames-per-sequence",
            "6",
            "--seed",
            "7",
        ]
    )

    detector_dir = run / "detector"
    train_detector.main(
        [
            "--source-dataset",
            str(data / "DomainA"),
            "--source-dataset",
            str(data / "DomainB"),
            "--train-split",
            "train",
            "--val-split",
            "test",
            "--detector",
            "tiny",
            "--base-loss",
            "bce_dice",
            "--resize",
            "16",
            "16",
            "--batch-size",
            "24",
            "--epochs",
            "1",
            "--warm-epoch",
            "0",
            "--optimizer",
            "adamw",
            "--lr",
            "0.001",
            "--lambda-tail",
            "0.05",
            "--lambda-miss",
            "0.05",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--no-amp",
            "--output-dir",
            str(detector_dir),
        ]
    )

    episode_paths: dict[str, Path] = {}
    for domain in ("DomainA", "DomainB", "DomainC"):
        score_dir = run / "scores" / domain
        export_scores.main(
            [
                "--dataset-dir",
                str(data / domain),
                "--split",
                "test",
                "--detector",
                "tiny",
                "--checkpoint",
                str(detector_dir / "best.pt"),
                "--resize",
                "16",
                "16",
                "--restore-original",
                "--include-mask",
                "--num-workers",
                "0",
                "--device",
                "cpu",
                "--output-dir",
                str(score_dir),
            ]
        )
        episode_path = run / "episodes" / f"{domain}.npz"
        build_episodes.main(
            [
                "--score-dir",
                str(score_dir),
                "--output",
                str(episode_path),
                "--context-size",
                "2",
                "--horizon",
                "1",
                "--stride",
                "3",
                "--peak-min-distance",
                "2",
                "--max-candidates",
                "1024",
            ]
        )
        episode_paths[domain] = episode_path

    curve_dir = run / "curve"
    train_curve.main(
        [
            "--train-episode",
            str(episode_paths["DomainA"]),
            "--train-episode",
            str(episode_paths["DomainB"]),
            "--quantile",
            "0.90",
            "--hidden-dim",
            "32",
            "--dropout",
            "0.0",
            "--batch-size",
            "4",
            "--epochs",
            "3",
            "--patience",
            "3",
            "--lr",
            "0.001",
            "--pixel-budget",
            "1.0",
            "--peak-budget",
            "1000000000",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--output-dir",
            str(curve_dir),
        ]
    )

    zero_dir = run / "zero"
    evaluate_zero_label.main(
        [
            "--episode",
            str(episode_paths["DomainC"]),
            "--curve-checkpoint",
            str(curve_dir / "best.pt"),
            "--pixel-budget",
            "1.0",
            "--peak-budget",
            "1000000000",
            "--device",
            "cpu",
            "--output-dir",
            str(zero_dir),
        ]
    )

    crc_dir = run / "crc"
    calibrate_and_evaluate.main(
        [
            "--episode",
            str(episode_paths["DomainC"]),
            "--curve-checkpoint",
            str(curve_dir / "best.pt"),
            "--pixel-budget",
            "1.0",
            "--peak-budget",
            "1000000000",
            "--alpha",
            "0.50",
            "--calibration-sizes",
            "2",
            "--seeds",
            "0",
            "--offset-step",
            "1",
            "--device",
            "cpu",
            "--output-dir",
            str(crc_dir),
        ]
    )

    required = [
        detector_dir / "best.pt",
        curve_dir / "best.pt",
        zero_dir / "summary.json",
        crc_dir / "summary.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Smoke test is missing artifacts: {missing}")
    summary = {
        "status": "passed",
        "work_dir": str(work),
        "artifacts": [str(path) for path in required],
        "notes": (
            "This validates the complete training/export/episode/risk-curve/"
            "zero-label/CRC software path with TinyUNet and synthetic data. It "
            "does not substitute for real MSHNet or benchmark experiments."
        ),
    }
    atomic_json_dump(summary, work / "smoke_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rc_irstd.episodes.meta_dataset import MetaEpisodeBuildConfig, build_meta_episode_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build grouped multi-budget support/query meta episodes."
    )
    parser.add_argument("--score-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", action="append", type=float, required=True)
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--protocol", choices=["auto", "iid", "temporal"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--background-sample-limit", type=int, default=65536)
    parser.add_argument("--object-top-fraction", type=float, default=0.25)
    parser.add_argument("--source-reference", default=None)
    parser.add_argument("--split-role", default="official_train_meta")
    parser.add_argument(
        "--allow-overlapping-episodes",
        action="store_true",
        help="Diagnostic only; claim-bearing meta train/validation should remain disjoint.",
    )
    parser.add_argument("--threshold-file", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    thresholds = None
    if args.threshold_file:
        thresholds = np.loadtxt(args.threshold_file, dtype=np.float32)
    config = MetaEpisodeBuildConfig(
        context_size=args.context_size,
        horizon=args.horizon,
        stride=args.stride,
        protocol=args.protocol,
        seed=args.seed,
        background_sample_limit=args.background_sample_limit,
        object_top_fraction=args.object_top_fraction,
        enforce_global_disjoint=not args.allow_overlapping_episodes,
        split_role=args.split_role,
    )
    path = build_meta_episode_file(
        args.score_directory,
        args.output,
        budgets=args.budget,
        thresholds=thresholds,
        config=config,
        source_reference=args.source_reference,
    )
    print(json.dumps({"output": str(Path(path).resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

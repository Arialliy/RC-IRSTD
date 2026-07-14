from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from rc_irstd.episodes.builder import EpisodeBuildConfig, build_episode_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build disjoint support-to-query risk-curve episodes."
    )
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold-grid", default=None)
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument(
        "--protocol",
        choices=["auto", "iid", "temporal"],
        default="auto",
        help="IID is for unordered static images; temporal is prefix-to-future.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--peak-min-distance", type=int, default=2)
    parser.add_argument("--peak-min-score", type=float, default=0.0)
    parser.add_argument("--peak-border", type=int, default=0)
    parser.add_argument("--peak-tolerance", type=float, default=2.0)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Maximum fixed peaks per image; 0 disables truncation.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_candidates < 0:
        raise ValueError("max-candidates must be non-negative")
    thresholds = np.load(args.threshold_grid) if args.threshold_grid else None
    config = EpisodeBuildConfig(
        context_size=args.context_size,
        horizon=args.horizon,
        stride=args.stride,
        protocol=args.protocol,
        seed=args.seed,
        peak_min_distance=args.peak_min_distance,
        peak_min_score=args.peak_min_score,
        peak_border=args.peak_border,
        peak_tolerance=args.peak_tolerance,
        max_candidates_per_image=(None if args.max_candidates <= 0 else args.max_candidates),
    )
    path = build_episode_file(
        args.score_dir, args.output, thresholds=thresholds, config=config
    )
    print(
        json.dumps(
            {"output": str(path.resolve()), "config": asdict(config)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

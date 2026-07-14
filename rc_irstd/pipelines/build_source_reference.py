from __future__ import annotations

import argparse

from rc_irstd.features.source_distance import build_source_reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a label-free source-distance reference.")
    parser.add_argument("--score-directory", action="append", required=True)
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path = build_source_reference(
        args.score_directory,
        args.output,
        context_size=args.context_size,
        stride=args.stride,
    )
    print(path)


if __name__ == "__main__":
    main()

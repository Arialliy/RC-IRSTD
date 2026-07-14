from __future__ import annotations

from pathlib import Path
from typing import Iterable


def parse_hw(values: Iterable[int] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    items = list(values)
    if len(items) != 2:
        raise ValueError("Image size must contain exactly two integers: H W")
    height, width = int(items[0]), int(items[1])
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    return height, width


def existing_paths(values: Iterable[str]) -> list[Path]:
    paths = [Path(value) for value in values]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing paths: {missing}")
    return paths

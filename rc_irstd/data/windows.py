from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CausalWindow:
    context_indices: tuple[int, ...]
    future_indices: tuple[int, ...]
    sequence_id: str
    protocol: str = "temporal"


def build_causal_windows(
    sequence_ids: Sequence[str],
    frame_indices: Sequence[int],
    context_size: int,
    horizon: int,
    stride: int = 1,
) -> list[CausalWindow]:
    """Build prefix-to-future windows within each real temporal sequence."""
    if context_size <= 0 or horizon <= 0 or stride <= 0:
        raise ValueError("context_size, horizon and stride must be positive")
    groups: dict[str, list[tuple[int, int]]] = {}
    for global_index, (sequence, frame) in enumerate(
        zip(sequence_ids, frame_indices, strict=True)
    ):
        groups.setdefault(str(sequence), []).append((int(frame), global_index))

    windows: list[CausalWindow] = []
    for sequence, pairs in sorted(groups.items()):
        ordered = [index for _, index in sorted(pairs)]
        total = context_size + horizon
        for start in range(0, len(ordered) - total + 1, stride):
            context = tuple(ordered[start : start + context_size])
            future = tuple(ordered[start + context_size : start + total])
            if set(context).intersection(future):
                raise RuntimeError("Context and future windows overlap")
            windows.append(CausalWindow(context, future, sequence, "temporal"))
    return windows


def build_iid_windows(
    num_samples: int,
    context_size: int,
    horizon: int,
    stride: int | None = None,
    seed: int = 0,
) -> list[CausalWindow]:
    """Build deterministic support/query blocks for unordered static images.

    The permutation is fixed by ``seed``.  There is no claim of temporal
    causality: each output is explicitly tagged ``protocol='iid'``.  Setting
    ``stride=context_size+horizon`` yields non-overlapping statistical blocks;
    smaller strides are permitted for meta-training and are handled by the
    overlap-aware split utilities.
    """
    if num_samples <= 0 or context_size <= 0 or horizon <= 0:
        raise ValueError("num_samples, context_size and horizon must be positive")
    total = int(context_size + horizon)
    step = total if stride is None else int(stride)
    if step <= 0:
        raise ValueError("stride must be positive")
    if num_samples < total:
        return []
    permutation = np.random.default_rng(seed).permutation(num_samples).tolist()
    windows: list[CausalWindow] = []
    for block_index, start in enumerate(range(0, num_samples - total + 1, step)):
        context = tuple(int(value) for value in permutation[start : start + context_size])
        future = tuple(
            int(value) for value in permutation[start + context_size : start + total]
        )
        if set(context).intersection(future):
            raise RuntimeError("IID context and future blocks overlap")
        windows.append(
            CausalWindow(context, future, f"iid_block_{block_index:06d}", "iid")
        )
    return windows

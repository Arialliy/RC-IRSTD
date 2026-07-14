from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class DomainBalancedBatchSampler(Sampler[list[int]]):
    """Create batches with the same number of examples from each domain."""

    def __init__(
        self,
        domain_ids: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        self.domain_ids = np.asarray(domain_ids, dtype=np.int64)
        self.domains = sorted(np.unique(self.domain_ids).tolist())
        if not self.domains:
            raise ValueError("domain_ids must not be empty")
        if batch_size % len(self.domains) != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by num_domains={len(self.domains)}"
            )
        self.per_domain = batch_size // len(self.domains)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.indices = {
            domain: np.flatnonzero(self.domain_ids == domain)
            for domain in self.domains
        }
        if any(len(values) == 0 for values in self.indices.values()):
            raise ValueError("Every domain must contain at least one sample")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        max_size = max(len(values) for values in self.indices.values())
        if self.drop_last:
            # Sampling is with replacement for smaller domains, so a positive
            # dataset should still yield one balanced batch even when
            # ``per_domain`` exceeds the largest raw domain size.
            return max(1, max_size // self.per_domain)
        return math.ceil(max_size / self.per_domain)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pools: dict[int, np.ndarray] = {}
        target_size = len(self) * self.per_domain
        for domain, values in self.indices.items():
            selected = values.copy()
            if self.shuffle:
                rng.shuffle(selected)
            if len(selected) < target_size:
                extra = rng.choice(selected, size=target_size - len(selected), replace=True)
                selected = np.concatenate([selected, extra])
            pools[domain] = selected[:target_size]

        for batch_index in range(len(self)):
            batch: list[int] = []
            start = batch_index * self.per_domain
            stop = start + self.per_domain
            for domain in self.domains:
                batch.extend(pools[domain][start:stop].tolist())
            if self.shuffle:
                rng.shuffle(batch)
            yield batch

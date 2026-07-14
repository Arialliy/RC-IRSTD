"""Deterministic, equally weighted mini-batches from multiple source domains."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_ext.multi_source_dataset import DomainDataset


def _seed_worker(_: int) -> None:
    """Seed NumPy and Python from the worker seed assigned by PyTorch."""

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BalancedDomainLoader:
    """Yield one same-sized mini-batch from every domain at every step.

    By default an epoch has as many steps as one non-replacement pass through
    the longest domain (with incomplete batches dropped).  Shorter domains are
    restarted and reshuffled whenever exhausted.  Call :meth:`set_epoch`
    before each epoch; ordering is then a deterministic function of
    ``seed``, epoch number, and domain position.
    """

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        batch_size_per_domain: int,
        *,
        epoch_steps: Optional[int] = None,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if not datasets:
            raise ValueError("at least one source domain is required")
        if batch_size_per_domain <= 0:
            raise ValueError("batch_size_per_domain must be positive")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")

        self.domain_names = list(datasets)
        self.batch_size_per_domain = int(batch_size_per_domain)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.epoch = 0

        self.datasets: Dict[str, DomainDataset] = {}
        used_ids = set()
        for position, (name, dataset) in enumerate(datasets.items()):
            wrapped = (
                dataset
                if isinstance(dataset, DomainDataset)
                else DomainDataset(dataset, position, name)
            )
            if wrapped.domain_id in used_ids:
                raise ValueError(f"duplicate domain id {wrapped.domain_id}")
            if len(wrapped) < self.batch_size_per_domain:
                raise ValueError(
                    f"domain {name!r} has {len(wrapped)} samples, fewer than "
                    f"batch_size_per_domain={self.batch_size_per_domain}"
                )
            used_ids.add(wrapped.domain_id)
            self.datasets[name] = wrapped

        complete_batches = {
            name: len(dataset) // self.batch_size_per_domain
            for name, dataset in self.datasets.items()
        }
        if epoch_steps is None:
            self.steps_per_epoch = max(complete_batches.values())
        elif epoch_steps <= 0:
            raise ValueError("epoch_steps must be positive when supplied")
        else:
            self.steps_per_epoch = int(epoch_steps)
        self.complete_batches = complete_batches
        self.last_cycle_counts = {name: 0 for name in self.domain_names}

    def __len__(self) -> int:
        return self.steps_per_epoch

    @property
    def total_batch_size(self) -> int:
        return self.batch_size_per_domain * len(self.domain_names)

    @property
    def domain_ids(self) -> Dict[str, int]:
        return {
            name: dataset.domain_id
            for name, dataset in self.datasets.items()
        }

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _make_loaders(self) -> Dict[str, DataLoader]:
        loaders = {}
        for position, (name, dataset) in enumerate(self.datasets.items()):
            generator = torch.Generator()
            generator.manual_seed(
                self.seed + self.epoch * 1_000_003 + position * 10_007
            )
            loaders[name] = DataLoader(
                dataset,
                batch_size=self.batch_size_per_domain,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                worker_init_fn=_seed_worker,
                generator=generator,
                persistent_workers=False,
            )
        return loaders

    @staticmethod
    def _merge_batches(batches: List[Dict[str, object]]) -> Dict[str, object]:
        """Interleave domains sample-first instead of concatenating domains.

        A domain-contiguous batch is unsafe with ``nn.DataParallel``: its
        contiguous scatter can send one complete domain to each GPU, and only
        device 0's BatchNorm buffers are retained.  Stacking as ``[B, D, ...]``
        before flattening makes every contiguous replica chunk contain the
        same balanced domain mixture whenever the chunk size is a multiple of
        the number of domains.
        """

        batch_sizes = [int(batch["image"].shape[0]) for batch in batches]
        if len(set(batch_sizes)) != 1:
            raise ValueError(f"domain batch sizes must match, got {batch_sizes}")

        images = torch.stack([batch["image"] for batch in batches], dim=1).flatten(0, 1)
        masks = torch.stack([batch["mask"] for batch in batches], dim=1).flatten(0, 1)
        domain_ids = torch.stack(
            [batch["domain_id"].reshape(-1) for batch in batches], dim=1
        ).flatten()
        sample_indices = torch.stack(
            [batch["sample_index"].reshape(-1) for batch in batches], dim=1
        ).flatten()

        names_by_domain: List[List[str]] = []
        for batch in batches:
            names = batch["domain_name"]
            if isinstance(names, str):
                names_by_domain.append([names] * batch_sizes[0])
            else:
                values = [str(value) for value in names]
                if len(values) != batch_sizes[0]:
                    raise ValueError("domain_name metadata does not match batch size")
                names_by_domain.append(values)
        domain_names = [
            names_by_domain[domain_position][sample_position]
            for sample_position in range(batch_sizes[0])
            for domain_position in range(len(batches))
        ]

        return {
            "image": images,
            "mask": masks,
            "domain_id": domain_ids,
            "domain_name": domain_names,
            "sample_index": sample_indices,
        }

    def __iter__(self) -> Iterator[Dict[str, object]]:
        loaders = self._make_loaders()
        iterators = {name: iter(loader) for name, loader in loaders.items()}
        cycle_counts = {name: 0 for name in self.domain_names}

        for _ in range(self.steps_per_epoch):
            batches = []
            for name in self.domain_names:
                try:
                    batch = next(iterators[name])
                except StopIteration:
                    cycle_counts[name] += 1
                    iterators[name] = iter(loaders[name])
                    batch = next(iterators[name])
                if int(batch["image"].shape[0]) != self.batch_size_per_domain:
                    raise RuntimeError(
                        f"domain {name!r} produced an incomplete batch despite drop_last"
                    )
                batches.append(batch)

            self.last_cycle_counts = dict(cycle_counts)
            yield self._merge_batches(batches)

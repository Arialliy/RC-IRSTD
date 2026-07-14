"""Dataset wrappers that attach explicit source-domain metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Dict, Optional

import torch
from torch.utils.data import ConcatDataset, Dataset


class DomainDataset(Dataset):
    """Attach a stable integer id and name to every sample of one domain."""

    def __init__(
        self,
        dataset: Dataset,
        domain_id: int,
        domain_name: str,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError(f"domain {domain_name!r} has no samples")
        if not domain_name:
            raise ValueError("domain_name must be non-empty")
        self.dataset = dataset
        self.domain_id = int(domain_id)
        self.domain_name = str(domain_name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.dataset[index]
        if isinstance(sample, Mapping):
            if "image" not in sample or "mask" not in sample:
                raise KeyError("mapping samples must contain 'image' and 'mask'")
            result = dict(sample)
        elif isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)):
            if len(sample) < 2:
                raise ValueError("sequence samples must contain at least image and mask")
            result = {"image": sample[0], "mask": sample[1]}
        else:
            raise TypeError(
                "wrapped datasets must return a mapping or an (image, mask) sequence"
            )

        result.update(
            {
                "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
                "domain_name": self.domain_name,
                "sample_index": torch.tensor(index, dtype=torch.long),
            }
        )
        return result


class MultiSourceDataset(ConcatDataset):
    """Concatenated view useful for inspection, not for balanced sampling.

    Training should normally use :class:`BalancedDomainLoader`; a plain
    concatenation would let the largest dataset dominate the optimizer.
    """

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        domain_ids: Optional[Mapping[str, int]] = None,
    ) -> None:
        if not datasets:
            raise ValueError("at least one source dataset is required")
        names = list(datasets)
        if len(set(names)) != len(names):
            raise ValueError("domain names must be unique")
        ids = domain_ids or {name: index for index, name in enumerate(names)}
        if set(ids) != set(names):
            raise ValueError("domain_ids must define exactly the supplied domains")
        if len(set(int(value) for value in ids.values())) != len(ids):
            raise ValueError("domain ids must be unique")

        wrapped = [
            dataset
            if isinstance(dataset, DomainDataset)
            else DomainDataset(dataset, ids[name], name)
            for name, dataset in datasets.items()
        ]
        self.domain_names = names
        self.domain_ids = {name: int(ids[name]) for name in names}
        super().__init__(wrapped)

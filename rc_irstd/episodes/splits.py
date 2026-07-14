from __future__ import annotations

import json

import numpy as np

from rc_irstd.episodes.dataset import EpisodeArrays


def _json_ids(value: str) -> set[str]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return set()
    return {str(item) for item in parsed} if isinstance(parsed, list) else set()


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def _iid_overlap_groups(arrays: EpisodeArrays, indices: np.ndarray) -> dict[int, str]:
    """Group IID episodes connected by any shared context/future image."""
    union = _UnionFind(len(indices))
    owners: dict[str, int] = {}
    for local, global_index in enumerate(indices):
        ids = _json_ids(arrays.context_ids[global_index]) | _json_ids(
            arrays.future_ids[global_index]
        )
        for image_id in ids:
            if image_id in owners:
                union.union(local, owners[image_id])
            else:
                owners[image_id] = local
    return {
        int(global_index): f"iid_overlap_{union.find(local):08d}"
        for local, global_index in enumerate(indices)
    }


def _group_labels(arrays: EpisodeArrays) -> np.ndarray:
    protocols = (
        np.asarray(arrays.protocols).astype(str)
        if arrays.protocols is not None
        else np.asarray(["temporal"] * len(arrays.domains), dtype=np.str_)
    )
    labels = np.empty(len(arrays.domains), dtype=object)
    iid_indices = np.flatnonzero(protocols == "iid")
    iid_groups = _iid_overlap_groups(arrays, iid_indices) if len(iid_indices) else {}
    for index, (domain, sequence, protocol) in enumerate(
        zip(arrays.domains, arrays.sequences, protocols, strict=True)
    ):
        if protocol == "iid":
            labels[index] = f"{domain}::{iid_groups[index]}"
        else:
            labels[index] = f"{domain}::{sequence}"
    return labels.astype(np.str_)


def leave_domains_out(
    arrays: EpisodeArrays,
    held_out_domains: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    held = np.isin(arrays.domains, np.asarray(held_out_domains).astype(str))
    return np.flatnonzero(~held), np.flatnonzero(held)


def grouped_train_val_split(
    arrays: EpisodeArrays,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split independent temporal sequences or IID overlap components."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    groups = _group_labels(arrays)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError(
            "The episode graph has fewer than two independent groups. For IID "
            "data, build non-overlapping windows with stride >= context+horizon; "
            "for temporal data, provide at least two sequences."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    count = min(len(unique) - 1, max(1, int(round(len(unique) * val_fraction))))
    val_groups = set(unique[:count].tolist())
    val_mask = np.asarray([group in val_groups for group in groups], dtype=bool)
    train, validation = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    if not len(train) or not len(validation):
        raise RuntimeError("Grouped train/validation split produced an empty partition")
    return train, validation


def grouped_calibration_test_split(
    arrays: EpisodeArrays,
    calibration_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select exact calibration episodes and an independent-group test set.

    ``calibration_size`` counts episodes/blocks.  Use the image-calibration path
    in :mod:`rc_irstd.calibration.samples` when a number of labelled images is
    being claimed.
    """
    if calibration_size <= 0:
        raise ValueError("calibration_size must be positive")
    groups = _group_labels(arrays)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError(
            "Calibration/test splitting needs at least two independent groups. "
            "Use non-overlapping IID windows or multiple temporal sequences."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    group_indices: dict[str, np.ndarray] = {}
    for group in unique:
        values = np.flatnonzero(groups == group)
        rng.shuffle(values)
        group_indices[str(group)] = values

    selected_groups: list[str] = []
    available = 0
    for group in unique[:-1]:
        selected_groups.append(str(group))
        available += len(group_indices[str(group)])
        if available >= calibration_size:
            break
    if available < calibration_size:
        max_possible = sum(len(group_indices[str(group)]) for group in unique[:-1])
        raise ValueError(
            f"Requested {calibration_size} calibration episodes, but only "
            f"{max_possible} are available while retaining an independent test group"
        )

    pool = np.concatenate([group_indices[group] for group in selected_groups])
    rng.shuffle(pool)
    calibration = np.sort(pool[:calibration_size].astype(np.int64))
    selected_group_set = set(selected_groups)
    test = np.flatnonzero(
        np.asarray([str(group) not in selected_group_set for group in groups], dtype=bool)
    ).astype(np.int64)
    if len(test) == 0:
        raise RuntimeError("Independent-group split produced an empty test set")
    if set(groups[calibration]).intersection(set(groups[test])):
        raise RuntimeError("Calibration and test groups overlap")
    return calibration, test


def split_iid_images(
    image_ids: np.ndarray,
    calibration_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact labelled-image split for IID calibration samples."""
    image_ids = np.asarray(image_ids).astype(str)
    if len(np.unique(image_ids)) != len(image_ids):
        raise ValueError("Image-shot calibration requires unique image IDs")
    if calibration_size <= 0 or calibration_size >= len(image_ids):
        raise ValueError("calibration_size must be in [1, num_images-1]")
    permutation = np.random.default_rng(seed).permutation(len(image_ids))
    calibration = np.sort(permutation[:calibration_size].astype(np.int64))
    test = np.sort(permutation[calibration_size:].astype(np.int64))
    return calibration, test


def split_sequence_units(
    sequences: np.ndarray,
    calibration_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select complete temporal sequences; size counts independent sequences."""
    sequences = np.asarray(sequences).astype(str)
    unique = np.unique(sequences)
    if calibration_size <= 0 or calibration_size >= len(unique):
        raise ValueError("calibration_size must leave at least one test sequence")
    selected = set(np.random.default_rng(seed).permutation(unique)[:calibration_size].tolist())
    calibration = np.flatnonzero(np.asarray([item in selected for item in sequences]))
    test = np.flatnonzero(np.asarray([item not in selected for item in sequences]))
    return calibration, test

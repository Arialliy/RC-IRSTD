"""Torch dataset and train-only feature standardisation for RC episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import RCEpisode


def load_episodes(path: str | Path) -> list[RCEpisode]:
    path = Path(path)
    episodes: list[RCEpisode] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    episodes.append(RCEpisode.from_dict(json.loads(line)))
                except Exception as error:
                    raise ValueError(f"invalid episode at {path}:{line_number}: {error}") from error
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("episodes", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            raise ValueError("episode JSON must be a list or {'episodes': [...]} object")
        episodes = [RCEpisode.from_dict(row) for row in rows]
    if not episodes:
        raise ValueError(f"no episodes found in {path}")
    validate_episode_collection(episodes)
    return episodes


def validate_episode_collection(episodes: Sequence[RCEpisode]) -> None:
    if not episodes:
        raise ValueError("episode collection must be non-empty")
    feature_names = episodes[0].feature_names
    transform = episodes[0].threshold_transform
    statistics_config = episodes[0].statistics_config
    p_min = episodes[0].p_min
    outer_fold_id = episodes[0].outer_fold_id
    outer_target = episodes[0].outer_target
    episode_ids = set()
    for episode in episodes:
        if episode.feature_names != feature_names:
            raise ValueError("all episodes must use exactly the same feature schema")
        if episode.threshold_transform != transform:
            raise ValueError("all episodes must use exactly the same threshold_transform")
        if episode.statistics_config != statistics_config:
            raise ValueError("statistics_config must be collection-wide and identical")
        if episode.p_min != p_min:
            raise ValueError("p_min must be explicit and identical across the collection")
        if episode.outer_fold_id != outer_fold_id or episode.outer_target != outer_target:
            raise ValueError("one calibrator collection must belong to one outer fold/target")
        if episode.episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id: {episode.episode_id}")
        episode_ids.add(episode.episode_id)
    pseudo_targets = {episode.pseudo_target for episode in episodes}
    if outer_target in pseudo_targets:
        raise ValueError("outer_target must not appear among calibration pseudo-targets")


def assert_verified_provenance(episodes: Sequence[RCEpisode]) -> None:
    unverified = [
        episode.episode_id for episode in episodes if episode.provenance.status != "verified"
    ]
    if unverified:
        raise ValueError(
            "main-protocol calibrator training rejects asserted_unverified provenance; "
            f"episodes={unverified}"
        )


def assert_pseudo_target_isolation(
    train_episodes: Sequence[RCEpisode],
    validation_episodes: Sequence[RCEpisode],
) -> None:
    train_targets = {episode.pseudo_target for episode in train_episodes}
    validation_targets = {episode.pseudo_target for episode in validation_episodes}
    overlap = train_targets.intersection(validation_targets)
    if overlap:
        raise ValueError(
            "train/validation pseudo-targets must be disjoint; "
            f"overlap={sorted(overlap)}"
        )


def split_by_pseudo_target(
    episodes: Sequence[RCEpisode],
    validation_targets: Iterable[str],
) -> tuple[list[RCEpisode], list[RCEpisode]]:
    targets = {str(value) for value in validation_targets}
    if not targets:
        raise ValueError("at least one validation pseudo-target is required")
    available = {episode.pseudo_target for episode in episodes}
    missing = targets.difference(available)
    if missing:
        raise ValueError(f"validation pseudo-targets not found: {sorted(missing)}")
    train = [episode for episode in episodes if episode.pseudo_target not in targets]
    validation = [episode for episode in episodes if episode.pseudo_target in targets]
    if not train or not validation:
        raise ValueError("pseudo-target split must leave non-empty train and validation sets")
    assert_pseudo_target_isolation(train, validation)
    return train, validation


def encoded_matrix(episodes: Sequence[RCEpisode]) -> np.ndarray:
    validate_episode_collection(episodes)
    matrix = np.asarray([episode.encoded_features() for episode in episodes], dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("encoded episode matrix must be finite and two-dimensional")
    return matrix


def context_matrix(episodes: Sequence[RCEpisode]) -> np.ndarray:
    """Return budget-invariant, unlabeled context statistics only."""

    validate_episode_collection(episodes)
    matrix = np.asarray([episode.statistics for episode in episodes], dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("context statistics matrix must be finite and two-dimensional")
    return matrix


@dataclass(frozen=True)
class FeatureStandardizer:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        scale = np.asarray(self.scale, dtype=np.float64).reshape(-1)
        expected = len(self.feature_names)
        if mean.size != expected or scale.size != expected:
            raise ValueError("standardizer arrays do not match feature_names")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("standardizer mean/scale must be finite and scale positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        feature_names: Sequence[str],
        min_scale: float = 1e-8,
    ) -> "FeatureStandardizer":
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("standardizer requires a non-empty 2D matrix")
        if values.shape[1] != len(feature_names):
            raise ValueError("matrix width does not match feature_names")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale < min_scale, 1.0, scale)
        return cls(tuple(feature_names), mean, scale)

    @classmethod
    def fit_train(cls, train_episodes: Sequence[RCEpisode]) -> "FeatureStandardizer":
        """Fit only from training episodes; callers never pass validation data."""

        matrix = encoded_matrix(train_episodes)
        return cls.fit(matrix, train_episodes[0].input_feature_names)

    @classmethod
    def fit_context_train(
        cls, train_episodes: Sequence[RCEpisode]
    ) -> "FeatureStandardizer":
        """Fit only budget-invariant context features from training episodes."""

        matrix = context_matrix(train_episodes)
        return cls.fit(matrix, train_episodes[0].feature_names)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape[-1] != len(self.feature_names):
            raise ValueError("input width does not match standardizer")
        return (values - self.mean) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureStandardizer":
        return cls(
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
        )


class RCMetaDataset(Dataset):
    def __init__(
        self,
        episodes: Sequence[RCEpisode] | str | Path,
        *,
        standardizer: FeatureStandardizer | None = None,
    ) -> None:
        if isinstance(episodes, (str, Path)):
            episodes = load_episodes(episodes)
        self.episodes = list(episodes)
        validate_episode_collection(self.episodes)
        self.standardizer = standardizer
        raw = encoded_matrix(self.episodes)
        if standardizer is not None:
            if standardizer.feature_names != self.episodes[0].input_feature_names:
                raise ValueError("standardizer feature schema differs from episodes")
            raw = standardizer.transform(raw)
        self.features = raw.astype(np.float32)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self.episodes[index]
        return {
            "features": torch.from_numpy(self.features[index]),
            "threshold": torch.tensor(episode.oracle_threshold, dtype=torch.float32),
            "reject": torch.tensor(float(episode.reject), dtype=torch.float32),
            "oracle_pd": torch.tensor(episode.oracle_pd, dtype=torch.float32),
            "episode_id": episode.episode_id,
            "pseudo_target": episode.pseudo_target,
        }

    @property
    def input_dim(self) -> int:
        return self.features.shape[1]

    @property
    def pseudo_targets(self) -> set[str]:
        return {episode.pseudo_target for episode in self.episodes}


class RCPixelRiskMetaDataset(Dataset):
    """Pixel-budget episodes with budgets kept outside the context encoder."""

    def __init__(
        self,
        episodes: Sequence[RCEpisode] | str | Path,
        *,
        standardizer: FeatureStandardizer | None = None,
    ) -> None:
        if isinstance(episodes, (str, Path)):
            episodes = load_episodes(episodes)
        self.episodes = list(episodes)
        validate_episode_collection(self.episodes)
        unsupported = [
            episode.episode_id
            for episode in self.episodes
            if episode.budgets.active != (True, False)
        ]
        if unsupported:
            raise ValueError(
                "monotone pixel-risk training requires pixel-only budgets; "
                f"unsupported episodes={unsupported}"
            )
        self.standardizer = standardizer
        raw = context_matrix(self.episodes)
        if standardizer is not None:
            if standardizer.feature_names != self.episodes[0].feature_names:
                raise ValueError(
                    "context standardizer feature schema differs from episodes"
                )
            raw = standardizer.transform(raw)
        self.features = raw.astype(np.float32)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self.episodes[index]
        return {
            "features": torch.from_numpy(self.features[index]),
            "pixel_budget": torch.tensor(
                episode.budgets.values[0], dtype=torch.float64
            ),
            "threshold": torch.tensor(
                episode.oracle_threshold, dtype=torch.float32
            ),
            "reject": torch.tensor(float(episode.reject), dtype=torch.float32),
            "oracle_pd": torch.tensor(episode.oracle_pd, dtype=torch.float32),
            "episode_id": episode.episode_id,
            "pseudo_target": episode.pseudo_target,
        }

    @property
    def input_dim(self) -> int:
        return self.features.shape[1]

    @property
    def pseudo_targets(self) -> set[str]:
        return {episode.pseudo_target for episode in self.episodes}

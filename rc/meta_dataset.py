"""Torch dataset and train-only feature standardisation for RC episodes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import OFFICIAL_TRAIN_SPLIT_ROLE, RCEpisode, SCHEMA_VERSION


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


GroupedQueryCurveMode = Literal["none", "verified_event_exact"]


def _validate_pixel_budget_grid(
    pixel_budget_grid: Sequence[float],
) -> tuple[float, ...]:
    try:
        grid = tuple(float(value) for value in pixel_budget_grid)
    except (TypeError, ValueError) as error:
        raise ValueError("pixel_budget_grid must be a sequence of numbers") from error
    if len(grid) < 2:
        raise ValueError("pixel_budget_grid must contain at least two values")
    if not all(math.isfinite(value) and value > 0.0 for value in grid):
        raise ValueError("pixel_budget_grid values must be finite and positive")
    if not all(loose > strict for loose, strict in zip(grid, grid[1:])):
        raise ValueError(
            "pixel_budget_grid must be strictly descending from loose to strict"
        )
    return grid


def _budget_grid_index(value: float, grid: Sequence[float]) -> int | None:
    for index, candidate in enumerate(grid):
        if math.isclose(float(value), float(candidate), rel_tol=1e-12, abs_tol=0.0):
            return index
    return None


def _curve_group_identity(
    episode: RCEpisode,
) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...]]:
    # Deliberately exclude hashes from the identity.  Two copies of the same
    # causal window with different provenance must collide and fail rather
    # than silently becoming two apparently valid groups.
    return (
        episode.outer_fold_id,
        episode.outer_target,
        episode.pseudo_target,
        episode.context_image_ids,
        episode.query_image_ids,
    )


def _curve_group_id(episode: RCEpisode) -> str:
    payload = {
        "outer_fold_id": episode.outer_fold_id,
        "outer_target": episode.outer_target,
        "pseudo_target": episode.pseudo_target,
        "context_image_ids": list(episode.context_image_ids),
        "query_image_ids": list(episode.query_image_ids),
        "curve_file_sha256": episode.provenance.curve_file_sha256,
        "curve_manifest_sha256": episode.provenance.curve_manifest_sha256,
        "query_score_manifest_sha256": (
            episode.provenance.query_score_manifest_sha256
        ),
        "label_manifest_sha256": episode.provenance.label_manifest_sha256,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"pixel-curve:{hashlib.sha256(canonical).hexdigest()}"


@dataclass(frozen=True)
class PixelRiskEpisodeGroup:
    """One causal context/query window supervised on a complete budget grid.

    ``episodes`` are ordered exactly like ``pixel_budget_grid`` (loose to
    strict).  The class is an adapter over schema-v4 scalar episodes; it does
    not introduce a weaker score/label record format or mutate the persisted
    episode schema.
    """

    group_id: str
    pixel_budget_grid: tuple[float, ...]
    episodes: tuple[RCEpisode, ...]

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must be non-empty")
        grid = _validate_pixel_budget_grid(self.pixel_budget_grid)
        if grid != self.pixel_budget_grid:
            raise ValueError("pixel_budget_grid must be stored canonically")
        if len(self.episodes) != len(grid):
            raise ValueError("episodes must contain one row per pixel budget")

    @property
    def representative(self) -> RCEpisode:
        return self.episodes[0]

    @property
    def pseudo_target(self) -> str:
        return self.representative.pseudo_target

    @property
    def context_image_ids(self) -> tuple[str, ...]:
        return self.representative.context_image_ids

    @property
    def query_image_ids(self) -> tuple[str, ...]:
        return self.representative.query_image_ids


def _assert_group_row_matches(
    representative: RCEpisode,
    episode: RCEpisode,
) -> None:
    if episode.provenance != representative.provenance:
        raise ValueError(
            "one context/query curve group must use identical provenance and hashes"
        )
    for name in (
        "statistics",
        "feature_names",
        "statistics_config",
        "source_reference",
        "fold",
        "p_min",
        "threshold_transform",
    ):
        if getattr(episode, name) != getattr(representative, name):
            raise ValueError(
                "one context/query curve group must share identical "
                f"{name}"
            )

    # These redundant metadata hashes are not the authority (provenance is),
    # but when present they must not contradict it.
    metadata_bindings = {
        "curve_sha256": episode.provenance.curve_file_sha256,
        "curve_manifest_sha256": episode.provenance.curve_manifest_sha256,
        "query_label_manifest_sha256": episode.provenance.label_manifest_sha256,
    }
    for metadata_name, expected in metadata_bindings.items():
        observed = episode.metadata.get(metadata_name)
        if observed is not None and str(observed).lower() != expected:
            raise ValueError(
                f"episode metadata {metadata_name} contradicts provenance"
            )


def group_pixel_risk_episodes(
    episodes: Sequence[RCEpisode],
    *,
    pixel_budget_grid: Sequence[float],
) -> tuple[PixelRiskEpisodeGroup, ...]:
    """Fail-closed grouping of scalar v4 episodes into complete ``[J]`` curves.

    This main-protocol adapter accepts only verified ``official_train`` pixel
    risk episodes.  A grid must be supplied explicitly, so accidentally
    omitting one budget from the whole collection cannot redefine a smaller
    inferred grid.
    """

    validate_episode_collection(episodes)
    grid = _validate_pixel_budget_grid(pixel_budget_grid)
    grouped: dict[
        tuple[str, str, str, tuple[str, ...], tuple[str, ...]],
        list[RCEpisode],
    ] = {}
    for episode in episodes:
        if episode.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "grouped main-protocol training requires meta-episode schema v4"
            )
        if episode.provenance.status != "verified":
            raise ValueError(
                "grouped main-protocol training requires verified provenance"
            )
        split_contract = episode.provenance.split_contract
        if split_contract is None or split_contract.get("role") != (
            OFFICIAL_TRAIN_SPLIT_ROLE
        ):
            raise ValueError(
                "grouped main-protocol training accepts official_train scores only"
            )
        if split_contract.get("disjointness_verified") is not True:
            raise ValueError("official train/test disjointness must be verified")
        if episode.budgets.active != (True, False):
            raise ValueError(
                "grouped pixel-risk training requires active=(True, False)"
            )
        if _budget_grid_index(episode.budgets.values[0], grid) is None:
            raise ValueError(
                "episode pixel budget is outside the frozen pixel_budget_grid: "
                f"episode={episode.episode_id}, budget={episode.budgets.values[0]}"
            )
        budget = float(episode.budgets.values[0])
        tolerance = max(1e-15, budget * 1e-9)
        if float(episode.oracle_pixel_risk) > budget + tolerance:
            raise ValueError(
                "oracle pixel risk exceeds its declared budget: "
                f"episode={episode.episode_id}"
            )
        grouped.setdefault(_curve_group_identity(episode), []).append(episode)

    result: list[PixelRiskEpisodeGroup] = []
    for key in sorted(grouped):
        rows = grouped[key]
        representative = rows[0]
        by_grid_index: dict[int, RCEpisode] = {}
        for episode in rows:
            _assert_group_row_matches(representative, episode)
            index = _budget_grid_index(episode.budgets.values[0], grid)
            if index is None:  # guarded above; retained against future edits
                raise RuntimeError("pixel-budget grid membership changed during grouping")
            if index in by_grid_index:
                raise ValueError(
                    "duplicate budget in one context/query curve group: "
                    f"group={key}, budget={grid[index]}"
                )
            by_grid_index[index] = episode
        missing = [grid[index] for index in range(len(grid)) if index not in by_grid_index]
        if missing:
            raise ValueError(
                "each context/query curve group must contain the complete frozen "
                f"pixel-budget grid; group={key}, missing={missing}"
            )
        ordered = tuple(by_grid_index[index] for index in range(len(grid)))
        thresholds = tuple(float(row.oracle_threshold) for row in ordered)
        if any(
            strict + 1e-12 < loose
            for loose, strict in zip(thresholds, thresholds[1:])
        ):
            raise ValueError(
                "oracle thresholds decrease as the pixel budget tightens: "
                f"group={key}, thresholds={thresholds}"
            )
        result.append(
            PixelRiskEpisodeGroup(
                group_id=_curve_group_id(representative),
                pixel_budget_grid=grid,
                episodes=ordered,
            )
        )
    if not result:
        raise ValueError("grouped pixel-risk collection must be non-empty")
    return tuple(result)


def _probability_logits(values: np.ndarray, *, eps: float) -> np.ndarray:
    if not math.isfinite(float(eps)) or not 0.0 < float(eps) < 0.5:
        raise ValueError("oracle_logit_eps must be finite and lie in (0, 0.5)")
    probabilities = np.asarray(values, dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("probability values must be finite and lie in [0, 1]")
    clipped = np.clip(probabilities, float(eps), 1.0 - float(eps))
    return np.log(clipped) - np.log1p(-clipped)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_episode_artifact(
    artifact_root: Path,
    raw_path: object,
    *,
    name: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"verified query curve requires metadata {name}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = artifact_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _manifest_nonnegative_integer(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"curve manifest {name} must be an integer")
    if value < 0:
        raise ValueError(f"curve manifest {name} must be non-negative")
    return value


@dataclass(frozen=True)
class QueryRiskCurveSupervision:
    """Hash-bound exact event curve used as query-side meta supervision.

    The representation retains every recorded event in the certified exact
    score suffix.  It never uniformly subsamples a fixed number of background
    pixels, which would be unsafe for budgets around ``1e-6``.
    """

    thresholds: tuple[float, ...]
    logits: tuple[float, ...]
    pixel_risk: tuple[float, ...]
    pd: tuple[float, ...]
    fp_pixels: tuple[int, ...]
    total_pixels: int
    gt_objects: int
    matching_rule: str
    centroid_distance: float
    exact_lower_bound: float
    global_exact: bool
    supervision_mode: str
    curve_file_sha256: str
    curve_manifest_sha256: str
    query_score_manifest_sha256: str
    label_manifest_sha256: str
    label_manifest_content_sha256: str

    def __post_init__(self) -> None:
        lengths = {
            len(self.thresholds),
            len(self.logits),
            len(self.pixel_risk),
            len(self.pd),
            len(self.fp_pixels),
        }
        if len(lengths) != 1 or next(iter(lengths)) < 2:
            raise ValueError("query risk curve arrays must have one common length >= 2")
        thresholds = np.asarray(self.thresholds, dtype=np.float64)
        logits = np.asarray(self.logits, dtype=np.float64)
        risks = np.asarray(self.pixel_risk, dtype=np.float64)
        pd = np.asarray(self.pd, dtype=np.float64)
        fp_pixels = np.asarray(self.fp_pixels, dtype=np.int64)
        if not all(
            np.isfinite(values).all() for values in (thresholds, logits, risks, pd)
        ):
            raise ValueError("query risk curve arrays must be finite")
        if np.any(np.diff(thresholds) <= 0.0):
            raise ValueError("query risk curve thresholds must be strictly ascending")
        if np.any(np.diff(logits) <= 0.0):
            raise ValueError(
                "query risk curve logits must be strictly ascending; "
                "reduce oracle_logit_eps to preserve distinct event thresholds"
            )
        if np.any(np.diff(risks) > 1e-15) or np.any(np.diff(fp_pixels) > 0):
            raise ValueError("query pixel false alarms must not increase with threshold")
        if np.any((thresholds < 0.0) | (thresholds > 1.0)):
            raise ValueError("query risk curve thresholds must lie in [0, 1]")
        if np.any((pd < 0.0) | (pd > 1.0)) or np.any(risks < 0.0):
            raise ValueError("query curve Pd/risk values are outside their legal range")
        if isinstance(self.total_pixels, bool) or int(self.total_pixels) <= 0:
            raise ValueError("query curve total_pixels must be a positive integer")
        if (
            isinstance(self.gt_objects, bool)
            or not isinstance(self.gt_objects, int)
            or self.gt_objects < 0
        ):
            raise ValueError("query curve gt_objects must be a non-negative integer")
        if self.matching_rule not in {"overlap", "centroid"}:
            raise ValueError("query curve matching_rule must be overlap or centroid")
        if not math.isfinite(float(self.centroid_distance)) or (
            float(self.centroid_distance) <= 0.0
        ):
            raise ValueError("query curve centroid_distance must be finite and positive")
        expected_risk = fp_pixels.astype(np.float64) / int(self.total_pixels)
        if not np.allclose(risks, expected_risk, rtol=0.0, atol=1e-15):
            raise ValueError("query curve pixel risk disagrees with exact pixel counts")
        if not math.isclose(
            thresholds[0], float(self.exact_lower_bound), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("query curve must begin at its exact lower bound")
        if not math.isclose(thresholds[-1], 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("query curve must retain the threshold=1 sentinel")
        if self.supervision_mode not in {"global_exact", "event_exact_suffix"}:
            raise ValueError("unsupported query curve supervision_mode")
        if bool(self.global_exact) != (self.supervision_mode == "global_exact"):
            raise ValueError("global_exact contradicts supervision_mode")


def load_verified_query_risk_curve(
    group: PixelRiskEpisodeGroup,
    *,
    artifact_root: str | Path,
    oracle_logit_eps: float = 1e-12,
) -> QueryRiskCurveSupervision:
    """Load and reverify one event-complete labeled-query curve.

    ``artifact_root`` is mandatory because episode metadata paths are relative
    to the build-spec root, not necessarily the JSONL directory.  This
    function intentionally never guesses an anchor from cwd.
    """

    root = Path(artifact_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"artifact_root is not a directory: {root}")
    episode = group.representative
    if episode.metadata.get("curve_provenance_status") != "verified":
        raise ValueError("query risk supervision requires verified curve metadata")
    if episode.metadata.get("causal_window_verified") is not True:
        raise ValueError("query risk supervision requires a verified causal window")

    curve_path = _resolve_episode_artifact(
        root, episode.metadata.get("curve_file"), name="curve_file"
    )
    curve_manifest_path = _resolve_episode_artifact(
        root,
        episode.metadata.get("curve_manifest_file"),
        name="curve_manifest_file",
    )
    provenance = episode.provenance
    if _sha256_file(curve_path) != provenance.curve_file_sha256:
        raise ValueError("curve CSV SHA-256 differs from episode provenance")
    if _sha256_file(curve_manifest_path) != provenance.curve_manifest_sha256:
        raise ValueError("curve manifest SHA-256 differs from episode provenance")
    manifest = json.loads(curve_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError("curve manifest must be a JSON object")
    declared_curve = _resolve_episode_artifact(
        curve_manifest_path.parent,
        manifest.get("curve_file"),
        name="curve manifest curve_file",
    )
    if declared_curve != curve_path:
        raise ValueError("episode metadata and curve manifest resolve different curves")
    if str(manifest.get("curve_sha256", "")).lower() != (
        provenance.curve_file_sha256
    ):
        raise ValueError("curve manifest curve_sha256 differs from provenance")
    if manifest.get("evaluation_scope") != "score_bound_label_attachment_verified":
        raise ValueError("query curve lacks an independently bound label attachment")
    if (
        manifest.get("oracle_only") is not True
        or manifest.get("selection_uses_ground_truth_labels") is not True
        or manifest.get("deployable") is not False
    ):
        raise ValueError("query curve must be explicitly labeled oracle-only supervision")
    if str(manifest.get("target_dataset", "")) != episode.pseudo_target:
        raise ValueError("query curve target_dataset differs from pseudo_target")
    if tuple(str(value) for value in manifest.get("image_ids", ())) != (
        episode.query_image_ids
    ):
        raise ValueError("query curve image_ids differ from the causal query window")
    manifest_hash_bindings = {
        "score_manifest_sha256": provenance.query_score_manifest_sha256,
        "label_manifest_sha256": provenance.label_manifest_sha256,
        "label_manifest_content_sha256": (
            provenance.label_manifest_content_sha256
        ),
    }
    for name, expected in manifest_hash_bindings.items():
        if str(manifest.get(name, "")).lower() != expected:
            raise ValueError(f"curve manifest {name} differs from episode provenance")

    # Reverify native score schema v3 + official_train and every independently
    # attached label artifact.  The curve CSV alone is not accepted as proof.
    from data_ext.label_manifest_artifacts import verify_label_attachment
    from data_ext.score_manifest_artifacts import verify_score_manifest_artifacts

    score_manifest_path = _resolve_episode_artifact(
        curve_manifest_path.parent,
        manifest.get("score_manifest_file"),
        name="score_manifest_file",
    )
    label_manifest_path = _resolve_episode_artifact(
        curve_manifest_path.parent,
        manifest.get("label_manifest_file"),
        name="label_manifest_file",
    )
    verified_scores = verify_score_manifest_artifacts(
        score_manifest_path,
        image_ids=episode.query_image_ids,
        require_mask=False,
        require_native_contract=True,
        required_split_role=OFFICIAL_TRAIN_SPLIT_ROLE,
    )
    if verified_scores.manifest_sha256 != provenance.query_score_manifest_sha256:
        raise ValueError("reverified query score manifest differs from provenance")
    if tuple(item.image_id for item in verified_scores.selected_items) != (
        episode.query_image_ids
    ):
        raise ValueError("reverified query score order differs from the episode")
    label_attachment = verify_label_attachment(
        score_manifest_path,
        label_manifest_path,
        image_ids=episode.query_image_ids,
    )
    if label_attachment.manifest_sha256 != provenance.label_manifest_sha256:
        raise ValueError("reverified label manifest differs from provenance")
    if label_attachment.content_sha256 != provenance.label_manifest_content_sha256:
        raise ValueError("reverified label content differs from provenance")

    from evaluation.threshold_sweep import read_curve_csv

    rows = read_curve_csv(curve_path)
    thresholds = np.asarray([row["threshold"] for row in rows], dtype=np.float64)
    declared_thresholds = np.asarray(manifest.get("thresholds", ()), dtype=np.float64)
    if thresholds.shape != declared_thresholds.shape or not np.array_equal(
        thresholds, declared_thresholds
    ):
        raise ValueError("curve CSV thresholds differ from the curve manifest")
    if np.any(np.diff(thresholds) <= 0.0):
        raise ValueError("curve CSV thresholds must be strictly ascending")
    for field in ("num_images", "gt_objects", "total_pixels"):
        values = {int(row[field]) for row in rows}
        if values != {int(manifest.get(field, -1))}:
            raise ValueError(f"curve CSV {field} differs from the curve manifest")
    if int(manifest["num_images"]) != len(episode.query_image_ids):
        raise ValueError("curve num_images differs from the causal query window")

    # Reuse the episode builder's audited suffix contract for every supervised
    # budget.  This is cheap and keeps construction/loading semantics aligned.
    from .build_meta_episodes import _verify_oracle_event_coverage

    for row in group.episodes:
        _verify_oracle_event_coverage(
            manifest, oracle_threshold=float(row.oracle_threshold)
        )
    global_exact = manifest.get("global_exact")
    if not isinstance(global_exact, bool):
        raise TypeError("curve manifest global_exact must be boolean")
    candidate_count = _manifest_nonnegative_integer(
        manifest, "event_candidate_count"
    )
    if global_exact:
        exact_lower_bound = 0.0
        supervision_mode = "global_exact"
    elif candidate_count == 0:
        exact_lower_bound = float(manifest["event_candidate_score_lower_bound"])
        supervision_mode = "event_exact_suffix"
    else:
        raw_lower_bound = manifest.get("event_coverage_score_lower_bound")
        if raw_lower_bound is None:
            raise ValueError("curve manifest has no complete event-exact suffix")
        exact_lower_bound = float(raw_lower_bound)
        supervision_mode = "event_exact_suffix"
    if not math.isfinite(exact_lower_bound) or not 0.0 <= exact_lower_bound <= 1.0:
        raise ValueError("curve exact suffix lower bound must lie in [0, 1]")
    exact_mask = thresholds >= exact_lower_bound - 1e-12
    exact_indices = np.flatnonzero(exact_mask)
    if exact_indices.size < 2:
        raise ValueError("event-exact query curve suffix needs at least two points")
    exact_thresholds = thresholds[exact_indices]
    if not math.isclose(
        float(exact_thresholds[0]), exact_lower_bound, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("event-exact lower-bound threshold is absent from curve CSV")
    pixel_risk = np.asarray([row["fa_pixel"] for row in rows], dtype=np.float64)[
        exact_indices
    ]
    pd = np.asarray([row["pd"] for row in rows], dtype=np.float64)[exact_indices]
    fp_pixels = np.asarray([row["fp_pixels"] for row in rows], dtype=np.int64)[
        exact_indices
    ]
    total_pixels = int(rows[0]["total_pixels"])
    logits = _probability_logits(exact_thresholds, eps=oracle_logit_eps)
    return QueryRiskCurveSupervision(
        thresholds=tuple(float(value) for value in exact_thresholds),
        logits=tuple(float(value) for value in logits),
        pixel_risk=tuple(float(value) for value in pixel_risk),
        pd=tuple(float(value) for value in pd),
        fp_pixels=tuple(int(value) for value in fp_pixels),
        total_pixels=total_pixels,
        gt_objects=int(rows[0]["gt_objects"]),
        matching_rule=str(manifest.get("matching_rule", "")),
        centroid_distance=float(manifest.get("centroid_distance", float("nan"))),
        exact_lower_bound=exact_lower_bound,
        global_exact=global_exact,
        supervision_mode=supervision_mode,
        curve_file_sha256=provenance.curve_file_sha256,
        curve_manifest_sha256=provenance.curve_manifest_sha256,
        query_score_manifest_sha256=provenance.query_score_manifest_sha256,
        label_manifest_sha256=provenance.label_manifest_sha256,
        label_manifest_content_sha256=provenance.label_manifest_content_sha256,
    )


class RCGroupedPixelRiskMetaDataset(Dataset):
    """One item per causal window with a complete multi-budget target curve."""

    def __init__(
        self,
        episodes: Sequence[RCEpisode] | str | Path,
        *,
        pixel_budget_grid: Sequence[float],
        standardizer: FeatureStandardizer | None = None,
        query_curve_mode: GroupedQueryCurveMode = "none",
        artifact_root: str | Path | None = None,
        oracle_logit_eps: float = 1e-12,
    ) -> None:
        if isinstance(episodes, (str, Path)):
            episodes = load_episodes(episodes)
        self.episodes = list(episodes)
        self.groups = group_pixel_risk_episodes(
            self.episodes, pixel_budget_grid=pixel_budget_grid
        )
        if query_curve_mode not in {"none", "verified_event_exact"}:
            raise ValueError(
                "query_curve_mode must be 'none' or 'verified_event_exact'"
            )
        if not math.isfinite(float(oracle_logit_eps)) or not (
            0.0 < float(oracle_logit_eps) < 0.5
        ):
            raise ValueError("oracle_logit_eps must be finite and lie in (0, 0.5)")
        if query_curve_mode == "verified_event_exact" and artifact_root is None:
            raise ValueError(
                "artifact_root is required for verified_event_exact query curves"
            )
        self.pixel_budget_grid = self.groups[0].pixel_budget_grid
        self.standardizer = standardizer
        self.query_curve_mode = query_curve_mode
        self.oracle_logit_eps = float(oracle_logit_eps)

        raw = np.asarray(
            [group.representative.statistics for group in self.groups],
            dtype=np.float64,
        )
        if standardizer is not None:
            if standardizer.feature_names != self.groups[0].representative.feature_names:
                raise ValueError(
                    "context standardizer feature schema differs from episodes"
                )
            raw = standardizer.transform(raw)
        self.features = raw.astype(np.float32)
        self.query_curves: tuple[QueryRiskCurveSupervision, ...] | None
        if query_curve_mode == "verified_event_exact":
            assert artifact_root is not None
            self.query_curves = tuple(
                load_verified_query_risk_curve(
                    group,
                    artifact_root=artifact_root,
                    oracle_logit_eps=self.oracle_logit_eps,
                )
                for group in self.groups
            )
        else:
            self.query_curves = None

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self.groups[index]
        representative = group.representative
        provenance = representative.provenance
        oracle_thresholds = np.asarray(
            [episode.oracle_threshold for episode in group.episodes],
            dtype=np.float64,
        )
        item: dict[str, Any] = {
            "features": torch.from_numpy(self.features[index]),
            "pixel_budgets": torch.tensor(
                group.pixel_budget_grid, dtype=torch.float64
            ),
            "oracle_thresholds": torch.from_numpy(oracle_thresholds),
            "oracle_logits": torch.from_numpy(
                _probability_logits(oracle_thresholds, eps=self.oracle_logit_eps)
            ),
            "oracle_pd": torch.tensor(
                [episode.oracle_pd for episode in group.episodes],
                dtype=torch.float64,
            ),
            "oracle_pixel_risk": torch.tensor(
                [episode.oracle_pixel_risk for episode in group.episodes],
                dtype=torch.float64,
            ),
            "group_id": group.group_id,
            "episode_ids": tuple(episode.episode_id for episode in group.episodes),
            "pseudo_target": group.pseudo_target,
            "context_image_ids": group.context_image_ids,
            "query_image_ids": group.query_image_ids,
            "curve_file_sha256": provenance.curve_file_sha256,
            "curve_manifest_sha256": provenance.curve_manifest_sha256,
            "query_score_manifest_sha256": (
                provenance.query_score_manifest_sha256
            ),
            "label_manifest_sha256": provenance.label_manifest_sha256,
            "label_manifest_content_sha256": (
                provenance.label_manifest_content_sha256
            ),
            "query_curve_available": self.query_curves is not None,
        }
        if self.query_curves is not None:
            curve = self.query_curves[index]
            item.update(
                {
                    "curve_thresholds": torch.tensor(
                        curve.thresholds, dtype=torch.float64
                    ),
                    "curve_logits": torch.tensor(curve.logits, dtype=torch.float64),
                    "curve_pixel_risk": torch.tensor(
                        curve.pixel_risk, dtype=torch.float64
                    ),
                    "curve_pd": torch.tensor(curve.pd, dtype=torch.float64),
                    "curve_fp_pixels": torch.tensor(
                        curve.fp_pixels, dtype=torch.int64
                    ),
                    "curve_total_pixels": torch.tensor(
                        curve.total_pixels, dtype=torch.int64
                    ),
                    "curve_gt_objects": torch.tensor(
                        curve.gt_objects, dtype=torch.int64
                    ),
                    "curve_matching_rule": curve.matching_rule,
                    "curve_centroid_distance": torch.tensor(
                        curve.centroid_distance, dtype=torch.float64
                    ),
                    "curve_exact_lower_bound": torch.tensor(
                        curve.exact_lower_bound, dtype=torch.float64
                    ),
                    "curve_exact_lower_logit": torch.tensor(
                        curve.logits[0], dtype=torch.float64
                    ),
                    "curve_global_exact": bool(curve.global_exact),
                    "curve_supervision_mode": curve.supervision_mode,
                }
            )
        return item

    @property
    def input_dim(self) -> int:
        return self.features.shape[1]

    @property
    def pseudo_targets(self) -> set[str]:
        return {group.pseudo_target for group in self.groups}


def collate_grouped_pixel_risk_batch(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collate fixed ``[J]`` targets and pad variable exact event curves."""

    if not batch:
        raise ValueError("cannot collate an empty grouped pixel-risk batch")
    tensor_fields = (
        "features",
        "pixel_budgets",
        "oracle_thresholds",
        "oracle_logits",
        "oracle_pd",
        "oracle_pixel_risk",
    )
    result: dict[str, Any] = {
        name: torch.stack([item[name] for item in batch]) for name in tensor_fields
    }
    metadata_fields = (
        "group_id",
        "episode_ids",
        "pseudo_target",
        "context_image_ids",
        "query_image_ids",
        "curve_file_sha256",
        "curve_manifest_sha256",
        "query_score_manifest_sha256",
        "label_manifest_sha256",
        "label_manifest_content_sha256",
    )
    result.update({name: [item[name] for item in batch] for name in metadata_fields})
    available = [item.get("query_curve_available") is True for item in batch]
    if any(available) and not all(available):
        raise ValueError("a batch cannot mix present and absent query risk curves")
    result["query_curve_available"] = all(available)
    if not all(available):
        return result

    lengths = torch.tensor(
        [int(item["curve_thresholds"].numel()) for item in batch],
        dtype=torch.int64,
    )
    maximum = int(lengths.max().item())
    valid = torch.zeros((len(batch), maximum), dtype=torch.bool)
    float_curve_fields = (
        "curve_thresholds",
        "curve_logits",
        "curve_pixel_risk",
        "curve_pd",
    )
    for name in float_curve_fields:
        padded = torch.zeros((len(batch), maximum), dtype=torch.float64)
        for index, item in enumerate(batch):
            length = int(lengths[index].item())
            padded[index, :length] = item[name]
        result[name] = padded
    fp_pixels = torch.zeros((len(batch), maximum), dtype=torch.int64)
    for index, item in enumerate(batch):
        length = int(lengths[index].item())
        valid[index, :length] = True
        fp_pixels[index, :length] = item["curve_fp_pixels"]
    result["curve_fp_pixels"] = fp_pixels
    result["curve_valid_mask"] = valid
    result["curve_lengths"] = lengths
    result["curve_total_pixels"] = torch.stack(
        [item["curve_total_pixels"] for item in batch]
    )
    result["curve_gt_objects"] = torch.stack(
        [item["curve_gt_objects"] for item in batch]
    )
    result["curve_matching_rule"] = [
        item["curve_matching_rule"] for item in batch
    ]
    result["curve_centroid_distance"] = torch.stack(
        [item["curve_centroid_distance"] for item in batch]
    )
    result["curve_exact_lower_bound"] = torch.stack(
        [item["curve_exact_lower_bound"] for item in batch]
    )
    result["curve_exact_lower_logit"] = torch.stack(
        [item["curve_exact_lower_logit"] for item in batch]
    )
    result["curve_global_exact"] = torch.tensor(
        [bool(item["curve_global_exact"]) for item in batch], dtype=torch.bool
    )
    result["curve_supervision_mode"] = [
        item["curve_supervision_mode"] for item in batch
    ]
    return result

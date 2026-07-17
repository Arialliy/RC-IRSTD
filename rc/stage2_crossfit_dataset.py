"""Stage-2 episode-v5 loading, sample isolation and grouped supervision."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from rc.build_stage2_crossfit_episodes import verify_stage2_episode_collection_bundle
from rc.domain_statistics import FEATURE_NAMES
from rc.schema import StatisticsConfig
from rc.stage2_crossfit_schema import (
    ALL_DOMAINS,
    COLLECTION_TRAIN,
    COLLECTION_VALIDATION,
    OUTER_TARGETS,
    STAGE2_OOF_FIT,
    SOURCE_DIAGNOSTIC_VALIDATION,
    Stage2CrossfitContractError,
    Stage2CrossfitEpisode,
    VerifiedStage2EpisodeCollection,
    assert_verified_episode_collection,
    canonical_json_bytes,
    repository_root,
)


PIXEL_BUDGET_GRID = (1e-4, 1e-5, 1e-6)
STANDARDIZER_SCHEMA = "rc-irstd.stage2-standardizer-fit-records.v1"
STANDARDIZER_ALGORITHM = "population-mean-std-ddof0-scale-floor-v1"
_STANDARDIZER_CAPABILITY = object()
_TRAINER_REPLAY_CAPABILITY = object()


def load_stage2_episodes_v5(
    path: str | Path,
    expected_sha256: str,
    *,
    collection_manifest_path: str | Path,
    collection_manifest_sha256: str,
    commit_marker_path: str | Path,
    commit_marker_sha256: str,
    statistics_config: StatisticsConfig,
    repository_root: str | Path | None = None,
) -> VerifiedStage2EpisodeCollection:
    """Load a complete v5 bundle; all three external digests are mandatory."""

    return verify_stage2_episode_collection_bundle(
        path,
        expected_sha256,
        collection_manifest_path,
        collection_manifest_sha256,
        commit_marker_path,
        commit_marker_sha256,
        statistics_config=statistics_config,
        repository_root_value=repository_root,
    )


def _all_rows(collection: VerifiedStage2EpisodeCollection) -> list[Mapping[str, Any]]:
    return [
        row
        for episode in collection.episodes
        for key in ("context_records", "query_records")
        for row in episode.payload[key]
    ]


def _boundary_sets(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    fields = (
        "canonical_id",
        "original_image_sha256",
        "near_duplicate_cluster_id_or_unique_sentinel",
        "exclusion_group_id",
    )
    result = {field: {str(row[field]) for row in rows} for field in fields}
    for field in fields:
        if len(result[field]) != len(rows):
            raise Stage2CrossfitContractError(
                f"collection contains duplicate sample boundary: {field}"
            )
    return result


def assert_stage2_sample_isolation(
    train: VerifiedStage2EpisodeCollection,
    validation: VerifiedStage2EpisodeCollection,
) -> None:
    """Enforce four-boundary train/validation isolation and role purity."""

    train = assert_verified_episode_collection(train)
    validation = assert_verified_episode_collection(validation)
    if train.manifest["collection_role"] != COLLECTION_TRAIN:
        raise Stage2CrossfitContractError("train collection role is not Stage2 OOF fit")
    if validation.manifest["collection_role"] != COLLECTION_VALIDATION:
        raise Stage2CrossfitContractError("validation collection role is not source diagnostic")
    for field in ("outer_fold_id", "outer_target", "base_seed"):
        if train.manifest[field] != validation.manifest[field]:
            raise Stage2CrossfitContractError(f"train/validation {field} mismatch")
    outer_fold = str(train.manifest["outer_fold_id"])
    target = OUTER_TARGETS[outer_fold]
    expected_sources = set(ALL_DOMAINS) - {target}
    train_domains = {str(item.payload["source_domain"]) for item in train}
    validation_domains = {str(item.payload["source_domain"]) for item in validation}
    if train_domains != expected_sources or validation_domains != expected_sources:
        raise Stage2CrossfitContractError("both source domains are required in train and validation")
    for episode in train:
        payload = episode.payload
        if payload["episode_role"] != STAGE2_OOF_FIT:
            raise Stage2CrossfitContractError("train contains non-OOF-fit episode")
        identity = payload["detector_identity"]
        if identity["detector_role"] != "detector_oof" or identity["oof_fold_index"] not in {0, 1}:
            raise Stage2CrossfitContractError("train detector identity is not OOF")
        if payload["source_domain"] == target:
            raise Stage2CrossfitContractError("outer target leaked into train")
    for episode in validation:
        payload = episode.payload
        if payload["episode_role"] != SOURCE_DIAGNOSTIC_VALIDATION:
            raise Stage2CrossfitContractError("validation contains wrong episode role")
        identity = payload["detector_identity"]
        if identity["detector_role"] != "detector_full_fit" or identity["oof_fold_index"] is not None:
            raise Stage2CrossfitContractError("validation detector identity is not full-fit")
        if payload["source_domain"] == target:
            raise Stage2CrossfitContractError("outer target leaked into validation")
    train_rows = _all_rows(train)
    validation_rows = _all_rows(validation)
    train_boundaries = _boundary_sets(train_rows)
    validation_boundaries = _boundary_sets(validation_rows)
    for field in train_boundaries:
        overlap = train_boundaries[field] & validation_boundaries[field]
        if overlap:
            raise Stage2CrossfitContractError(
                f"train/validation identity overlap at {field}"
            )


@dataclass(frozen=True)
class Stage2ContextStandardizer:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    fit_records: tuple[Mapping[str, Any], ...]
    fit_records_sha256: str
    fit_manifest: Mapping[str, Any]
    fit_manifest_sha256: str
    train_collection_sha256: str
    _capability: object

    def __post_init__(self) -> None:
        if self._capability is not _STANDARDIZER_CAPABILITY:
            raise TypeError("Stage2ContextStandardizer is fit-factory-only")
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if self.feature_names != tuple(FEATURE_NAMES) or mean.shape != (93,) or scale.shape != (93,):
            raise Stage2CrossfitContractError("standardizer feature schema mismatch")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise Stage2CrossfitContractError("standardizer mean/scale invalid")
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def transform(
        self,
        matrix: np.ndarray,
        *,
        active_indices: Sequence[int] | None = None,
    ) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape[-1] != 93 or not np.isfinite(values).all():
            raise Stage2CrossfitContractError("standardizer input must be finite [...,93]")
        result = (values - self.mean) / self.scale
        if active_indices is not None:
            active: set[int] = set()
            for raw in active_indices:
                if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw < 93:
                    raise Stage2CrossfitContractError("active feature index out of range")
                if raw in active:
                    raise Stage2CrossfitContractError("duplicate active feature index")
                active.add(raw)
            excluded = [index for index in range(93) if index not in active]
            result[..., excluded] = 0.0
            if excluded and np.signbit(result[..., excluded]).any():
                raise RuntimeError("excluded features must be exact positive zero")
        return result


def assert_stage2_context_standardizer(
    value: object,
) -> Stage2ContextStandardizer:
    if (
        not isinstance(value, Stage2ContextStandardizer)
        or value._capability is not _STANDARDIZER_CAPABILITY
    ):
        raise TypeError("a training-fit Stage2 standardizer capability is required")
    return value


def _context_vector(episode: Stage2CrossfitEpisode) -> np.ndarray:
    values = np.asarray(episode.payload["context_statistics"]["values"], dtype=np.float64)
    if values.shape != (93,) or not np.isfinite(values).all():
        raise Stage2CrossfitContractError("episode context vector is invalid")
    return values


def fit_stage2_context_standardizer(
    train: VerifiedStage2EpisodeCollection,
    *,
    output: str | Path | None = None,
    repository_root_value: str | Path | None = None,
) -> Stage2ContextStandardizer:
    """Fit float64 population statistics from training contexts and nothing else."""

    train = assert_verified_episode_collection(train)
    if train.manifest["collection_role"] != COLLECTION_TRAIN:
        raise Stage2CrossfitContractError("standardizer may fit only Stage2 training collection")
    matrix = np.stack([_context_vector(item) for item in train], axis=0)
    mean = matrix.mean(axis=0, dtype=np.float64)
    raw_scale = matrix.std(axis=0, dtype=np.float64, ddof=0)
    scale = np.maximum(raw_scale, 1e-8)
    records = tuple(
        {
            "fit_record_index": index,
            "episode_id": episode.episode_id,
            "episode_record_sha256": train.manifest["records"][index]["record_sha256"],
            "window_id": episode.payload["window_binding"]["window_id"],
            "source_domain": episode.payload["source_domain"],
            "context_full_identity_sha256": episode.payload["context_full_identity_sha256"],
            "context_vector_sha256": episode.payload["context_statistics"]["vector_sha256"],
        }
        for index, episode in enumerate(train)
    )
    records_sha = hashlib.sha256(canonical_json_bytes(list(records))).hexdigest()
    manifest = {
        "schema_version": STANDARDIZER_SCHEMA,
        "artifact_type": "rc_irstd_stage2_context_standardizer_fit_records",
        "artifact_status": "DEVELOPMENT_ONLY_VERIFIED",
        "development_only": True,
        "official_test_accessed": False,
        "fit_scope": "stage2_training_contexts_only",
        "validation_or_outer_access_count": 0,
        "feature_dim": 93,
        "feature_names": list(FEATURE_NAMES),
        "calculation_dtype": "float64",
        "algorithm": STANDARDIZER_ALGORITHM,
        "scale_floor": 1e-8,
        "below_floor_replacement": 1e-8,
        "feature_mask_application": "after_standardization_then_exact_positive_zero",
        "train_collection": {
            "path": str(train.path),
            "sha256": train.collection_sha256,
            "manifest_sha256": train.manifest_sha256,
            "commit_sha256": train.commit_sha256,
        },
        "fit_record_count": len(records),
        "fit_records_sha256": records_sha,
        "fit_records": list(records),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    manifest_data = canonical_json_bytes(manifest) + b"\n"
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    if output is not None:
        root = repository_root(repository_root_value)
        path = Path(output).expanduser().absolute()
        if root not in path.parents or path.parent.is_symlink() or not path.parent.is_dir():
            raise Stage2CrossfitContractError("standardizer output must be below repository_root")
        sidecar = path.with_name(path.name + ".sha256")
        if os.path.lexists(path) or os.path.lexists(sidecar):
            raise FileExistsError("standardizer output already exists")
        written: list[Path] = []
        try:
            _write_file(path, manifest_data)
            written.append(path)
            _write_file(sidecar, f"{manifest_sha}  {path.name}\n".encode("ascii"))
            written.append(sidecar)
            _fsync_dir(path.parent)
        except BaseException:
            for item in reversed(written):
                try:
                    item.unlink()
                except FileNotFoundError:
                    pass
            raise
    return Stage2ContextStandardizer(
        feature_names=tuple(FEATURE_NAMES),
        mean=mean,
        scale=scale,
        fit_records=records,
        fit_records_sha256=records_sha,
        fit_manifest=MappingProxyType(manifest),
        fit_manifest_sha256=manifest_sha,
        train_collection_sha256=train.collection_sha256,
        _capability=_STANDARDIZER_CAPABILITY,
    )


def _write_file(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _logit(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


@dataclass(frozen=True)
class Stage2PixelRiskEpisodeGroup:
    episode: Stage2CrossfitEpisode
    artifact_index: int
    pixel_budget_grid: tuple[float, float, float]
    oracle_thresholds: np.ndarray
    oracle_pd: np.ndarray
    oracle_pixel_risk: np.ndarray
    oracle_fp_pixels: np.ndarray
    curve_thresholds: np.ndarray
    curve_pd: np.ndarray
    curve_pixel_risk: np.ndarray
    curve_fp_pixels: np.ndarray
    curve_tp_objects: np.ndarray
    curve_total_pixels: int
    curve_gt_objects: int


@dataclass(frozen=True)
class Stage2CurveLogitView:
    """Zero-copy lazy logit projection over a verified threshold column."""

    thresholds: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.thresholds, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise Stage2CrossfitContractError("curve thresholds must be finite 1D")
        object.__setattr__(self, "thresholds", values)

    def __len__(self) -> int:
        return int(self.thresholds.size)

    def __getitem__(self, index: Any) -> Any:
        result = _logit(self.thresholds[index])
        return float(result) if np.ndim(result) == 0 else result

    def take(self, indices: Sequence[int]) -> np.ndarray:
        """Compute logits only for explicitly requested event indices."""

        raw = np.asarray(indices, dtype=np.int64)
        if raw.ndim != 1 or np.any(raw < 0) or np.any(raw >= len(self)):
            raise IndexError("curve logit indices out of range")
        return _logit(self.thresholds[raw])

    @property
    def nbytes(self) -> int:
        return 0


def _curve_column(rows: Any, field: str, dtype: Any) -> np.ndarray:
    if hasattr(rows, "column"):
        values = np.asarray(rows.column(field), dtype=dtype)
    else:
        values = np.asarray([row[field] for row in rows], dtype=dtype)
    if values.ndim != 1 or values.size == 0:
        raise Stage2CrossfitContractError(f"curve {field} column is invalid")
    values.setflags(write=False)
    return values


def _validate_grid(grid: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in grid)
    if values != PIXEL_BUDGET_GRID:
        raise Stage2CrossfitContractError(
            "Stage2 pixel budget grid must equal descending [1e-4,1e-5,1e-6]"
        )
    return values


def group_stage2_pixel_risk_episodes(
    collection: VerifiedStage2EpisodeCollection,
    grid: Sequence[float] = PIXEL_BUDGET_GRID,
) -> tuple[Stage2PixelRiskEpisodeGroup, ...]:
    collection = assert_verified_episode_collection(collection)
    budgets = _validate_grid(grid)
    groups: list[Stage2PixelRiskEpisodeGroup] = []
    for index, (episode, artifact) in enumerate(
        zip(collection.episodes, collection.artifacts, strict=True)
    ):
        rows = artifact.curve_rows
        if not rows:
            raise Stage2CrossfitContractError("verified exact curve is empty")
        thresholds = _curve_column(rows, "threshold", np.float64)
        pd = _curve_column(rows, "pd", np.float64)
        risk = _curve_column(rows, "fa_pixel", np.float64)
        fp = _curve_column(rows, "fp_pixels", np.int64)
        tp = _curve_column(rows, "tp_objects", np.int64)
        totals = _curve_column(rows, "total_pixels", np.int64)
        gt = _curve_column(rows, "gt_objects", np.int64)
        if not (
            thresholds.size == pd.size == risk.size == fp.size == tp.size
            == totals.size == gt.size
        ):
            raise Stage2CrossfitContractError("curve column cardinality mismatch")
        if np.any(totals != totals[0]) or np.any(gt != gt[0]):
            raise Stage2CrossfitContractError("curve count denominators changed")
        total_pixels = int(totals[0])
        gt_objects = int(gt[0])
        selected: list[int] = []
        for budget in budgets:
            exact_budget = Fraction.from_float(budget)
            maximum_feasible_fp = (
                exact_budget.numerator * total_pixels
            ) // exact_budget.denominator
            feasible = np.flatnonzero(fp <= maximum_feasible_fp)
            if feasible.size == 0:
                raise Stage2CrossfitContractError("threshold=1 must make each budget feasible")
            best_tp = int(tp[feasible].max())
            tied = feasible[tp[feasible] == best_tp]
            best_fp = int(fp[tied].min())
            tied = tied[fp[tied] == best_fp]
            chosen = int(tied[np.argmax(thresholds[tied])])
            selected.append(chosen)
        oracle_thresholds = thresholds[selected]
        if np.any(np.diff(oracle_thresholds) < 0):
            raise Stage2CrossfitContractError("oracle thresholds violate tighter-budget order")
        groups.append(
            Stage2PixelRiskEpisodeGroup(
                episode=episode,
                artifact_index=index,
                pixel_budget_grid=budgets,
                oracle_thresholds=oracle_thresholds,
                oracle_pd=pd[selected],
                oracle_pixel_risk=risk[selected],
                oracle_fp_pixels=fp[selected],
                curve_thresholds=thresholds,
                curve_pd=pd,
                curve_pixel_risk=risk,
                curve_fp_pixels=fp,
                curve_tp_objects=tp,
                curve_total_pixels=total_pixels,
                curve_gt_objects=gt_objects,
            )
        )
    return tuple(groups)


class Stage2CrossfitDataset(Dataset):
    def __init__(
        self,
        collection: VerifiedStage2EpisodeCollection,
        standardizer: Stage2ContextStandardizer,
        *,
        active_indices: Sequence[int] | None = None,
    ) -> None:
        self.collection = assert_verified_episode_collection(collection)
        self.standardizer = assert_stage2_context_standardizer(standardizer)
        self.groups = group_stage2_pixel_risk_episodes(self.collection)
        raw = np.stack([_context_vector(group.episode) for group in self.groups])
        self.features = standardizer.transform(raw, active_indices=active_indices).astype(np.float32)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self.groups[index]
        payload = group.episode.payload
        return {
            "features": torch.from_numpy(self.features[index]),
            "pixel_budgets": torch.tensor(group.pixel_budget_grid, dtype=torch.float64),
            "oracle_thresholds": torch.from_numpy(group.oracle_thresholds.copy()),
            "oracle_logits": torch.from_numpy(_logit(group.oracle_thresholds)),
            "oracle_pd": torch.from_numpy(group.oracle_pd.copy()),
            "oracle_pixel_risk": torch.from_numpy(group.oracle_pixel_risk.copy()),
            "curve_thresholds": group.curve_thresholds,
            "curve_logits": Stage2CurveLogitView(group.curve_thresholds),
            "curve_pixel_risk": group.curve_pixel_risk,
            "curve_pd": group.curve_pd,
            "curve_fp_pixels": group.curve_fp_pixels,
            "curve_tp_objects": group.curve_tp_objects,
            "curve_total_pixels": torch.tensor(group.curve_total_pixels, dtype=torch.int64),
            "curve_gt_objects": torch.tensor(group.curve_gt_objects, dtype=torch.int64),
            "episode_id": group.episode.episode_id,
            "window_id": payload["window_binding"]["window_id"],
            "source_domain": payload["source_domain"],
            "context_identity_sha256": payload["context_full_identity_sha256"],
            "source_query_identity_sha256": payload["source_ordered_query_identity_sha256"],
        }

    @property
    def input_dim(self) -> int:
        return 93


def collate_stage2_crossfit_batch(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate empty Stage2 batch")
    fixed = (
        "features", "pixel_budgets", "oracle_thresholds", "oracle_logits",
        "oracle_pd", "oracle_pixel_risk", "curve_total_pixels", "curve_gt_objects",
    )
    result: dict[str, Any] = {name: torch.stack([item[name] for item in batch]) for name in fixed}
    curve_fields = (
        "curve_thresholds", "curve_logits", "curve_pixel_risk", "curve_pd",
        "curve_fp_pixels", "curve_tp_objects",
    )
    for field in curve_fields:
        result[field] = tuple(item[field] for item in batch)
    for name in (
        "episode_id", "window_id", "source_domain", "context_identity_sha256",
        "source_query_identity_sha256",
    ):
        result[name] = tuple(item[name] for item in batch)
    return result


def extract_stage2_curve_brackets(
    batch: Mapping[str, Any], predicted_logits: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return two exact neighboring events per episode/budget, never a pad."""

    if predicted_logits.ndim != 2 or predicted_logits.shape[1] != 3:
        raise Stage2CrossfitContractError("predicted_logits must have shape [B,3]")
    batch_size = int(predicted_logits.shape[0])
    fields = (
        "curve_thresholds", "curve_pixel_risk", "curve_pd",
        "curve_fp_pixels", "curve_tp_objects",
    )
    for field in fields:
        if not isinstance(batch.get(field), tuple) or len(batch[field]) != batch_size:
            raise Stage2CrossfitContractError(f"ragged {field} batch mismatch")
    detached = predicted_logits.detach().to("cpu", torch.float64)
    if not torch.isfinite(detached).all():
        raise Stage2CrossfitContractError("predicted logits must be finite")
    predicted = torch.sigmoid(detached).numpy()
    shape = (batch_size, 3)
    float_names = ("thresholds", "pixel_risk", "pd")
    integer_names = ("fp_pixels", "tp_objects")
    lower_float = {name: np.empty(shape, np.float64) for name in float_names}
    upper_float = {name: np.empty(shape, np.float64) for name in float_names}
    lower_int = {name: np.empty(shape, np.int64) for name in integer_names}
    upper_int = {name: np.empty(shape, np.int64) for name in integer_names}
    for row in range(batch_size):
        threshold = np.asarray(batch["curve_thresholds"][row], dtype=np.float64)
        risk = np.asarray(batch["curve_pixel_risk"][row], dtype=np.float64)
        pd = np.asarray(batch["curve_pd"][row], dtype=np.float64)
        fp = np.asarray(batch["curve_fp_pixels"][row], dtype=np.int64)
        tp = np.asarray(batch["curve_tp_objects"][row], dtype=np.int64)
        if not (
            threshold.ndim == 1 and threshold.size > 0
            and np.all(threshold[1:] > threshold[:-1])
            and threshold.size == risk.size == pd.size == fp.size == tp.size
        ):
            raise Stage2CrossfitContractError("ragged exact curve is invalid")
        for column, target in enumerate(predicted[row]):
            upper = min(int(np.searchsorted(threshold, target, side="left")), threshold.size - 1)
            lower = max(upper - 1, 0)
            lower_float["thresholds"][row, column] = threshold[lower]
            upper_float["thresholds"][row, column] = threshold[upper]
            lower_float["pixel_risk"][row, column] = risk[lower]
            upper_float["pixel_risk"][row, column] = risk[upper]
            lower_float["pd"][row, column] = pd[lower]
            upper_float["pd"][row, column] = pd[upper]
            lower_int["fp_pixels"][row, column] = fp[lower]
            upper_int["fp_pixels"][row, column] = fp[upper]
            lower_int["tp_objects"][row, column] = tp[lower]
            upper_int["tp_objects"][row, column] = tp[upper]
    device = predicted_logits.device
    result: dict[str, torch.Tensor] = {}
    for name in float_names:
        result[f"lower_{name}"] = torch.from_numpy(lower_float[name]).to(device)
        result[f"upper_{name}"] = torch.from_numpy(upper_float[name]).to(device)
    for name in integer_names:
        result[f"lower_{name}"] = torch.from_numpy(lower_int[name]).to(device)
        result[f"upper_{name}"] = torch.from_numpy(upper_int[name]).to(device)
    result["lower_logits"] = torch.from_numpy(
        _logit(lower_float["thresholds"])
    ).to(device)
    result["upper_logits"] = torch.from_numpy(
        _logit(upper_float["thresholds"])
    ).to(device)
    return result


@dataclass(frozen=True)
class Stage2TrainerReplayCapability:
    train_collection_sha256: str
    validation_collection_sha256: str
    standardizer_fit_manifest_sha256: str
    _capability: object


def make_stage2_trainer_replay_capability(
    train: VerifiedStage2EpisodeCollection,
    validation: VerifiedStage2EpisodeCollection,
    standardizer: Stage2ContextStandardizer,
) -> Stage2TrainerReplayCapability:
    assert_stage2_sample_isolation(train, validation)
    standardizer = assert_stage2_context_standardizer(standardizer)
    if standardizer.train_collection_sha256 != train.collection_sha256:
        raise Stage2CrossfitContractError("standardizer was not fit on supplied train collection")
    return Stage2TrainerReplayCapability(
        train_collection_sha256=train.collection_sha256,
        validation_collection_sha256=validation.collection_sha256,
        standardizer_fit_manifest_sha256=standardizer.fit_manifest_sha256,
        _capability=_TRAINER_REPLAY_CAPABILITY,
    )


def assert_stage2_trainer_replay_capability(
    value: object,
    validation: VerifiedStage2EpisodeCollection,
) -> Stage2TrainerReplayCapability:
    if not isinstance(value, Stage2TrainerReplayCapability) or value._capability is not _TRAINER_REPLAY_CAPABILITY:
        raise TypeError("trainer replay capability required")
    if value.validation_collection_sha256 != validation.collection_sha256:
        raise Stage2CrossfitContractError("trainer capability/validation collection mismatch")
    return value


__all__ = [
    "PIXEL_BUDGET_GRID",
    "Stage2ContextStandardizer",
    "Stage2CrossfitDataset",
    "Stage2CurveLogitView",
    "Stage2PixelRiskEpisodeGroup",
    "Stage2TrainerReplayCapability",
    "assert_stage2_sample_isolation",
    "assert_stage2_context_standardizer",
    "assert_stage2_trainer_replay_capability",
    "collate_stage2_crossfit_batch",
    "extract_stage2_curve_brackets",
    "fit_stage2_context_standardizer",
    "group_stage2_pixel_risk_episodes",
    "load_stage2_episodes_v5",
    "make_stage2_trainer_replay_capability",
]

"""Exact native-resolution Stage2 query curves (schema v3).

Every unique float64 probability observed in the 28 ordered query maps is an
event threshold.  Endpoints 0 and 1 are always included, no event cap or
subsampling is accepted, predictions use ``probability > threshold``, and
object Pd uses deterministic 8-connected one-to-one overlap matching.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from data_ext.stage2_label_attachment import (
    STAGE2_QUERY_IDENTITY_ALGORITHM,
    VerifiedStage2LabelAttachment,
    canonical_json_sha256,
    load_stage2_label_mask,
    verify_stage2_label_attachment,
)
from data_ext.stage2_score_manifest import STRICT_THRESHOLD_SEMANTICS
from evaluation.component_matching import PreparedTarget, prepare_target


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE2_CURVE_SCHEMA = "rc-irstd.stage2-query-curve.v3"
STAGE2_CURVE_ARTIFACT_TYPE = "rc_irstd_stage2_exact_query_curve"
STAGE2_CURVE_ROWS_ALGORITHM = (
    "sha256-canonical-json-ordered-stage2-query-curve-rows-v3"
)
STAGE2_THRESHOLD_ALGORITHM = (
    "all-unique-float64-query-probability-events-plus-0-and-1-no-cap-v1"
)
STAGE2_MATCHING_CONTRACT = "native-resolution-8-connected-one-to-one-overlap-v1"

CURVE_FIELDS = (
    "threshold",
    "pd",
    "fa_pixel",
    "tp_objects",
    "gt_objects",
    "pred_components",
    "fp_components",
    "fp_pixels",
    "total_pixels",
    "num_images",
)

CURVE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "execution_scope",
        "official_test_accessed",
        "path_anchor",
        "role",
        "window_id",
        "window_identity_sha256",
        "outer_fold_id",
        "outer_target",
        "source_domain",
        "base_seed",
        "derived_seed",
        "detector_role",
        "oof_fold_index",
        "threshold_semantics",
        "threshold_algorithm",
        "score_dtype",
        "event_threshold_cap",
        "event_thresholds_capped",
        "global_exact",
        "endpoints",
        "num_unique_query_probability_events",
        "num_operating_points",
        "matching_contract",
        "connectivity",
        "object_matching",
        "false_alarm_pixel_numerator",
        "false_alarm_pixel_denominator",
        "query_size",
        "decision_seal_binding",
        "gt_objects",
        "total_native_pixels",
        "ordered_query_identity_sha256_algorithm",
        "ordered_query_identity_sha256",
        "window_binding",
        "score_manifest_binding",
        "label_manifest_binding",
        "score_bindings",
        "curve_file",
        "curve_sha256",
        "curve_rows_sha256_algorithm",
        "curve_rows_sha256",
    }
)


@dataclass(frozen=True)
class Stage2QueryCurve:
    attachment: VerifiedStage2LabelAttachment
    thresholds: np.ndarray
    rows: "ArrayBackedCurveRows"
    unique_event_count: int
    rows_sha256: str
    gt_objects: int
    total_pixels: int


class ArrayBackedCurveRows(Sequence[Mapping[str, float | int]]):
    """Read-only row sequence backed by ten contiguous NumPy columns."""

    _FLOAT_FIELDS = frozenset({"threshold", "pd", "fa_pixel"})

    def __init__(self, columns: Mapping[str, np.ndarray]) -> None:
        if tuple(columns) != CURVE_FIELDS:
            raise ValueError("curve column closure/order mismatch")
        lengths = {int(np.asarray(value).size) for value in columns.values()}
        if len(lengths) != 1:
            raise ValueError("curve columns have different lengths")
        self._length = lengths.pop()
        normalized: dict[str, np.ndarray] = {}
        for field in CURVE_FIELDS:
            dtype = np.float64 if field in self._FLOAT_FIELDS else np.int64
            value = np.ascontiguousarray(np.asarray(columns[field], dtype=dtype).reshape(-1))
            value.setflags(write=False)
            normalized[field] = value
        self._columns = normalized

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int | slice) -> Mapping[str, float | int] | tuple[Mapping[str, float | int], ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(self._length)))
        position = int(index)
        if position < 0:
            position += self._length
        if position < 0 or position >= self._length:
            raise IndexError("curve row index out of range")
        return {
            field: (
                float(self._columns[field][position])
                if field in self._FLOAT_FIELDS
                else int(self._columns[field][position])
            )
            for field in CURVE_FIELDS
        }

    def column(self, field: str) -> np.ndarray:
        if field not in self._columns:
            raise KeyError(field)
        return self._columns[field]

    @property
    def storage_nbytes(self) -> int:
        """Exact owned bytes of the ten read-only contiguous column arrays."""

        return sum(int(column.nbytes) for column in self._columns.values())


def stage2_curve_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("curve rows must be a non-empty ordered sequence")
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != set(CURVE_FIELDS):
            raise ValueError(f"curve row[{index}] field closure mismatch")
        if index:
            digest.update(b",")
        digest.update(
            json.dumps(
                dict(row),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    digest.update(b"]")
    return digest.hexdigest()


def build_stage2_query_curve(
    window_contract: str | Path,
    score_manifest: str | Path,
    label_manifest: str | Path,
    *,
    window_manifest_sha256: str,
    score_manifest_sha256: str,
    label_manifest_sha256: str,
    window_id: str,
    expected_role: str,
    repository_root: str | Path | None = None,
    bundle_root_override: str | Path | None = None,
    _owned_publication_lock: tuple[str | Path, tuple[int, int]] | None = None,
) -> Stage2QueryCurve:
    """Build an uncapped exact event replay for one bound Q28 window."""

    attachment = verify_stage2_label_attachment(
        score_manifest,
        label_manifest,
        expected_role,
        score_manifest_sha256=score_manifest_sha256,
        label_manifest_sha256=label_manifest_sha256,
        window_manifest=window_contract,
        window_manifest_sha256=window_manifest_sha256,
        window_id=window_id,
        repository_root=repository_root,
        bundle_root_override=bundle_root_override,
        _owned_publication_lock=_owned_publication_lock,
    )
    probabilities: list[np.ndarray] = []
    targets: list[PreparedTarget] = []
    for index, item in enumerate(attachment.items):
        score_path = item.score_item.score_path
        score_sha = str(item.score_item.record["score_file_sha256"])
        if _sha256_file_stable(score_path) != score_sha:
            raise RuntimeError(f"query score changed before curve replay at {index}")
        with np.load(score_path, allow_pickle=False) as payload:
            probability = np.asarray(payload["prob"])
        if probability.dtype != np.float64:
            raise ValueError("exact Stage2 curve requires float64 probability maps")
        if probability.ndim != 2 or probability.shape != item.original_hw:
            raise ValueError("query score is not in bound native image geometry")
        if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("query probability contains invalid values")
        if _sha256_file_stable(score_path) != score_sha:
            raise RuntimeError(f"query score changed during curve replay at {index}")
        mask = load_stage2_label_mask(item)
        probabilities.append(probability)
        targets.append(prepare_target(mask))

    if len(probabilities) != 28:
        raise RuntimeError("exact query curve requires Q28")
    sweep = _build_incremental_exact_sweep(probabilities, targets)
    return Stage2QueryCurve(
        attachment=attachment,
        thresholds=sweep.thresholds,
        rows=sweep.rows,
        unique_event_count=sweep.unique_event_count,
        rows_sha256=sweep.rows_sha256,
        gt_objects=sweep.gt_objects,
        total_pixels=sweep.total_pixels,
    )


@dataclass(frozen=True)
class _ExactSweepResult:
    thresholds: np.ndarray
    rows: ArrayBackedCurveRows
    unique_event_count: int
    rows_sha256: str
    gt_objects: int
    total_pixels: int


class _IncrementalImageState:
    """One image's descending 8-connected prediction filtration."""

    def __init__(self, target: PreparedTarget) -> None:
        if target.binary.ndim != 2 or target.labels.shape != target.binary.shape:
            raise ValueError("prepared target geometry is invalid")
        self.height, self.width = (int(value) for value in target.binary.shape)
        self.pixel_count = self.height * self.width
        self.parent = np.full(self.pixel_count, -1, dtype=np.int32)
        self.size = np.zeros(self.pixel_count, dtype=np.int32)
        self.gt_labels = np.asarray(target.labels, dtype=np.int32).reshape(-1)
        self.gt_foreground = np.asarray(target.binary, dtype=bool).reshape(-1)
        self.num_gt = int(target.num_gt)
        self.component_count = 0
        self.tp_count = 0
        self.gt_by_root: dict[int, set[int]] = {}

    def _find(self, index: int) -> int:
        root = index
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while index != root:
            parent = int(self.parent[index])
            self.parent[index] = root
            index = parent
        return root

    def activate(self, index: int) -> tuple[int, bool]:
        """Activate one pixel; return component delta and GT-graph dirtiness."""

        if index < 0 or index >= self.pixel_count or int(self.parent[index]) >= 0:
            raise RuntimeError("DSU pixel activation is duplicate or out of range")
        row, column = divmod(index, self.width)
        neighbour_roots: set[int] = set()
        for other_row in range(max(0, row - 1), min(self.height, row + 2)):
            base = other_row * self.width
            for other_column in range(max(0, column - 1), min(self.width, column + 2)):
                neighbour = base + other_column
                if neighbour == index or int(self.parent[neighbour]) < 0:
                    continue
                neighbour_roots.add(self._find(neighbour))

        old_gt_sets = tuple(
            frozenset(self.gt_by_root[root])
            for root in neighbour_roots
            if root in self.gt_by_root
        )
        merged_gt: set[int] = set()
        for values in old_gt_sets:
            merged_gt.update(values)
        gt_label = int(self.gt_labels[index])
        if gt_label > 0:
            merged_gt.add(gt_label - 1)

        self.parent[index] = index
        self.size[index] = 1
        candidates = [index, *neighbour_roots]
        winner = min(
            candidates,
            key=lambda root: (-int(self.size[root]), int(root)),
        )
        total_size = 0
        for root in candidates:
            total_size += int(self.size[root])
            if root != winner:
                self.parent[root] = winner
            self.gt_by_root.pop(root, None)
        self.parent[winner] = winner
        self.size[winner] = total_size
        if merged_gt:
            self.gt_by_root[winner] = merged_gt

        component_delta = 1 - len(neighbour_roots)
        self.component_count += component_delta
        if not old_gt_sets:
            graph_changed = bool(merged_gt)
        elif len(old_gt_sets) == 1:
            graph_changed = merged_gt != set(old_gt_sets[0])
        else:
            graph_changed = True
        return component_delta, graph_changed

    def recompute_tp(self) -> int:
        """Exact maximum-cardinality root-to-GT overlap matching."""

        if self.num_gt <= 0 or not self.gt_by_root:
            self.tp_count = 0
            return 0
        if self.num_gt == 1:
            self.tp_count = 1
            return 1
        candidates_by_gt: list[list[int]] = [[] for _ in range(self.num_gt)]
        for root, gt_ids in self.gt_by_root.items():
            for gt_id in gt_ids:
                if gt_id < 0 or gt_id >= self.num_gt:
                    raise RuntimeError("DSU overlap graph contains an invalid GT id")
                candidates_by_gt[gt_id].append(root)
        for candidates in candidates_by_gt:
            candidates.sort()

        root_to_gt: dict[int, int] = {}

        def augment(gt_id: int, visited_roots: set[int]) -> bool:
            for root in candidates_by_gt[gt_id]:
                if root in visited_roots:
                    continue
                visited_roots.add(root)
                previous = root_to_gt.get(root)
                if previous is None or augment(previous, visited_roots):
                    root_to_gt[root] = gt_id
                    return True
            return False

        for gt_id in sorted(
            range(self.num_gt),
            key=lambda value: (len(candidates_by_gt[value]), value),
        ):
            augment(gt_id, set())
        self.tp_count = len(root_to_gt)
        return self.tp_count


def _build_incremental_exact_sweep(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[PreparedTarget],
) -> _ExactSweepResult:
    """All-event, no-cap strict-``>`` curve via a descending DSU sweep."""

    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("probabilities and targets must be non-empty and aligned")
    states: list[_IncrementalImageState] = []
    flattened: list[np.ndarray] = []
    sizes: list[int] = []
    for index, (probability, target) in enumerate(zip(probabilities, targets, strict=True)):
        value = np.asarray(probability)
        if value.dtype != np.float64 or value.ndim != 2:
            raise TypeError(f"probability[{index}] must be a native 2D float64 map")
        if value.shape != target.binary.shape:
            raise ValueError(f"probability/target geometry mismatch at image {index}")
        if not np.isfinite(value).all() or np.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"probability[{index}] contains invalid values")
        state = _IncrementalImageState(target)
        states.append(state)
        flat = np.ascontiguousarray(value.reshape(-1))
        flattened.append(flat)
        sizes.append(int(flat.size))

    all_scores = np.concatenate(flattened).astype(np.float64, copy=False)
    unique_events = np.unique(all_scores).astype(np.float64, copy=False)
    thresholds = np.unique(
        np.concatenate((unique_events, np.asarray([0.0, 1.0], dtype=np.float64)))
    )
    thresholds.sort()
    if thresholds.dtype != np.float64 or thresholds[0] != 0.0 or thresholds[-1] != 1.0:
        raise RuntimeError("exact threshold event construction failed")
    if thresholds.size != np.unique(thresholds).size:
        raise RuntimeError("exact thresholds contain duplicates")
    if not np.isin(unique_events, thresholds, assume_unique=True).all():
        raise RuntimeError("an exact float64 query event was omitted")

    point_count = int(thresholds.size)
    total_pixels = int(sum(sizes))
    total_gt = int(sum(state.num_gt for state in states))
    num_images = len(states)
    columns: dict[str, np.ndarray] = {
        "threshold": thresholds.copy(),
        "pd": np.empty(point_count, dtype=np.float64),
        "fa_pixel": np.empty(point_count, dtype=np.float64),
        "tp_objects": np.empty(point_count, dtype=np.int64),
        "gt_objects": np.full(point_count, total_gt, dtype=np.int64),
        "pred_components": np.empty(point_count, dtype=np.int64),
        "fp_components": np.empty(point_count, dtype=np.int64),
        "fp_pixels": np.empty(point_count, dtype=np.int64),
        "total_pixels": np.full(point_count, total_pixels, dtype=np.int64),
        "num_images": np.full(point_count, num_images, dtype=np.int64),
    }

    offsets = np.zeros(num_images + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.asarray(sizes, dtype=np.int64))
    owners = np.repeat(np.arange(num_images, dtype=np.int32), np.asarray(sizes, dtype=np.int64))
    order = np.argsort(all_scores, kind="stable")
    cursor = int(order.size) - 1
    total_components = 0
    total_tp = 0
    total_fp_pixels = 0

    for threshold_index in range(point_count - 1, -1, -1):
        threshold = float(thresholds[threshold_index])
        columns["tp_objects"][threshold_index] = total_tp
        columns["pred_components"][threshold_index] = total_components
        columns["fp_components"][threshold_index] = total_components - total_tp
        columns["fp_pixels"][threshold_index] = total_fp_pixels
        columns["pd"][threshold_index] = total_tp / total_gt if total_gt else 0.0
        columns["fa_pixel"][threshold_index] = (
            total_fp_pixels / total_pixels if total_pixels else 0.0
        )

        dirty_images: set[int] = set()
        while cursor >= 0:
            global_index = int(order[cursor])
            if float(all_scores[global_index]) != threshold:
                break
            image_index = int(owners[global_index])
            local_index = global_index - int(offsets[image_index])
            state = states[image_index]
            component_delta, graph_changed = state.activate(local_index)
            total_components += component_delta
            if not bool(state.gt_foreground[local_index]):
                total_fp_pixels += 1
            if graph_changed:
                dirty_images.add(image_index)
            cursor -= 1
        for image_index in dirty_images:
            state = states[image_index]
            previous = state.tp_count
            current = state.recompute_tp()
            total_tp += current - previous

    if cursor != -1:
        raise RuntimeError("descending exact sweep omitted probability events")
    rows = ArrayBackedCurveRows(columns)
    if rows[-1]["pred_components"] != 0 or rows[-1]["fp_pixels"] != 0:
        raise RuntimeError("strict threshold 1 endpoint must predict no pixels")
    fp_column = rows.column("fp_pixels")
    if fp_column.size > 1 and np.any(fp_column[1:] > fp_column[:-1]):
        raise RuntimeError("FP pixels increased with ascending strict threshold")
    thresholds.setflags(write=False)
    return _ExactSweepResult(
        thresholds=thresholds,
        rows=rows,
        unique_event_count=int(unique_events.size),
        rows_sha256=stage2_curve_rows_sha256(rows),
        gt_objects=total_gt,
        total_pixels=total_pixels,
    )


def write_stage2_query_curve_artifacts(
    curve: Stage2QueryCurve,
    *,
    staging_root: str | Path,
    final_root: str | Path,
    repository_root: str | Path | None = None,
    curve_filename: str = "query-curve.csv",
) -> tuple[dict[str, Any], dict[str, str]]:
    """Write CSV+manifest into an owned private staging directory."""

    root = _repository_root(repository_root)
    staging = _owned_directory(staging_root, root, "staging_root")
    final = _future_directory(final_root, root, "final_root")
    if not isinstance(curve_filename, str) or not curve_filename.endswith(".csv"):
        raise ValueError("curve_filename must be one direct .csv filename")
    if PurePosixPath(curve_filename).name != curve_filename:
        raise ValueError("curve_filename must not contain directories")
    curve_path = staging / curve_filename
    manifest_path = staging / "curve-manifest.json"
    if os.path.lexists(curve_path) or os.path.lexists(manifest_path):
        raise FileExistsError("curve staging members already exist")
    _write_curve_csv(curve_path, curve.rows)
    curve_sha = _sha256_file_stable(curve_path)
    attachment = curve.attachment
    payload = attachment.payload
    score = attachment.score_manifest
    window = attachment.window
    final_curve = final / curve_filename
    final_label_manifest = final / "label-manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": STAGE2_CURVE_SCHEMA,
        "artifact_type": STAGE2_CURVE_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY",
        "development_only": True,
        "execution_scope": "stage2_development_exact_query_curve",
        "official_test_accessed": False,
        "path_anchor": "repository_root",
        "role": payload["role"],
        "window_id": payload["window_id"],
        "window_identity_sha256": payload["window_binding"]["window_identity_sha256"],
        "outer_fold_id": payload["outer_fold_id"],
        "outer_target": payload["outer_target"],
        "source_domain": payload["source_domain"],
        "base_seed": payload["base_seed"],
        "derived_seed": payload["derived_seed"],
        "detector_role": payload["detector_role"],
        "oof_fold_index": payload["oof_fold_index"],
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "threshold_algorithm": STAGE2_THRESHOLD_ALGORITHM,
        "score_dtype": "float64",
        "event_threshold_cap": None,
        "event_thresholds_capped": False,
        "global_exact": True,
        "endpoints": [0.0, 1.0],
        "num_unique_query_probability_events": curve.unique_event_count,
        "num_operating_points": len(curve.rows),
        "matching_contract": STAGE2_MATCHING_CONTRACT,
        "connectivity": 8,
        "object_matching": "maximum-cardinality-one-to-one-overlap",
        "false_alarm_pixel_numerator": "predicted_pixels_outside_GT_foreground",
        "false_alarm_pixel_denominator": "all_native_resolution_query_pixels",
        "query_size": 28,
        "decision_seal_binding": payload["decision_seal_binding"],
        "gt_objects": curve.gt_objects,
        "total_native_pixels": curve.total_pixels,
        "ordered_query_identity_sha256_algorithm": STAGE2_QUERY_IDENTITY_ALGORITHM,
        "ordered_query_identity_sha256": attachment.ordered_query_identity_sha256,
        "window_binding": dict(payload["window_binding"]),
        "score_manifest_binding": dict(payload["score_manifest_binding"]),
        "label_manifest_binding": {
            "path": _repo_relative(final_label_manifest, root),
            "sha256": attachment.manifest_sha256,
            "labels_content_sha256": attachment.labels_content_sha256,
        },
        "score_bindings": dict(score.bindings),
        "curve_file": _repo_relative(final_curve, root),
        "curve_sha256": curve_sha,
        "curve_rows_sha256_algorithm": STAGE2_CURVE_ROWS_ALGORITHM,
        "curve_rows_sha256": curve.rows_sha256,
    }
    _write_json_exclusive(manifest_path, manifest)
    manifest_sha = _sha256_file_stable(manifest_path)
    return manifest, {
        "curve_sha256": curve_sha,
        "curve_manifest_sha256": manifest_sha,
    }


def verify_stage2_query_curve_artifacts(
    curve_path: str | Path,
    curve_manifest: str | Path,
    *,
    curve_sha256: str,
    curve_manifest_sha256: str,
    attachment: VerifiedStage2LabelAttachment,
    repository_root: str | Path | None = None,
    bundle_root_override: str | Path | None = None,
    _owned_publication_lock: tuple[str | Path, tuple[int, int]] | None = None,
) -> tuple[Mapping[str, Any], ArrayBackedCurveRows]:
    root = _repository_root(repository_root)
    raw_csv_path = Path(curve_path).expanduser().absolute()
    raw_manifest_path = Path(curve_manifest).expanduser().absolute()
    if raw_csv_path.parent != raw_manifest_path.parent:
        raise ValueError("curve CSV and manifest must be direct members of one bundle")
    _verify_bundle_publication_lock(
        raw_csv_path.parent,
        _owned_publication_lock,
    )
    csv_path = _input_file(curve_path, root, "curve CSV", bundle_root_override)
    manifest_path = _input_file(curve_manifest, root, "curve manifest", bundle_root_override)
    if csv_path.parent != attachment.path.parent:
        raise ValueError("curve and verified label attachment are in different bundles")
    if _sha256_file_stable(csv_path) != _sha256(curve_sha256, "curve_sha256"):
        raise ValueError("curve CSV external SHA-256 mismatch")
    if _sha256_file_stable(manifest_path) != _sha256(curve_manifest_sha256, "curve_manifest_sha256"):
        raise ValueError("curve manifest external SHA-256 mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("curve manifest must contain an object")
    if frozenset(payload) != CURVE_MANIFEST_FIELDS:
        raise ValueError("curve manifest field closure mismatch")
    _verify_curve_manifest_identity(payload, attachment, root, csv_path, bundle_root_override)
    rows = _read_curve_csv(csv_path)
    if len(rows) != payload["num_operating_points"]:
        raise ValueError("curve row count mismatch")
    if stage2_curve_rows_sha256(rows) != payload["curve_rows_sha256"]:
        raise ValueError("curve rows content SHA-256 mismatch")
    thresholds = rows.column("threshold")
    if thresholds[0] != 0.0 or thresholds[-1] != 1.0 or not np.all(thresholds[1:] > thresholds[:-1]):
        raise ValueError("curve thresholds are not strict ordered endpoints")
    if payload["num_operating_points"] not in {
        payload["num_unique_query_probability_events"],
        payload["num_unique_query_probability_events"] + 1,
        payload["num_unique_query_probability_events"] + 2,
    }:
        raise ValueError("curve endpoint/event cardinality is impossible")
    if (
        np.any(rows.column("num_images") != 28)
        or np.any(rows.column("gt_objects") != payload["gt_objects"])
        or np.any(rows.column("total_pixels") != payload["total_native_pixels"])
    ):
        raise ValueError("curve invariant denominator mismatch")
    fp_pixels = rows.column("fp_pixels")
    if fp_pixels.size > 1 and np.any(fp_pixels[1:] > fp_pixels[:-1]):
        raise ValueError("curve FP pixels are non-monotone")
    if _sha256_file_stable(csv_path) != payload["curve_sha256"]:
        raise RuntimeError("curve CSV changed while verified")
    replayed = build_stage2_query_curve(
        attachment.window.path,
        attachment.score_manifest.path,
        attachment.path,
        window_manifest_sha256=attachment.window.manifest_sha256,
        score_manifest_sha256=attachment.score_manifest.manifest_sha256,
        label_manifest_sha256=attachment.manifest_sha256,
        window_id=attachment.window.window_id,
        expected_role=attachment.window.role,
        repository_root=root,
        bundle_root_override=bundle_root_override,
        _owned_publication_lock=_owned_publication_lock,
    )
    if replayed.unique_event_count != payload["num_unique_query_probability_events"]:
        raise ValueError("curve omitted or invented a float64 query event")
    if replayed.rows_sha256 != payload["curve_rows_sha256"] or not _curve_rows_equal(replayed.rows, rows):
        raise ValueError("curve rows differ from exact native-resolution replay")
    return payload, rows


def _curve_rows_equal(left: ArrayBackedCurveRows, right: ArrayBackedCurveRows) -> bool:
    if len(left) != len(right):
        return False
    return all(
        np.array_equal(left.column(field), right.column(field), equal_nan=False)
        for field in CURVE_FIELDS
    )


def _verify_curve_manifest_identity(
    payload: Mapping[str, Any],
    attachment: VerifiedStage2LabelAttachment,
    root: Path,
    csv_path: Path,
    bundle_root_override: str | Path | None,
) -> None:
    label = attachment.payload
    expected = {
        "schema_version": STAGE2_CURVE_SCHEMA,
        "artifact_type": STAGE2_CURVE_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY",
        "execution_scope": "stage2_development_exact_query_curve",
        "role": label["role"],
        "window_id": label["window_id"],
        "window_identity_sha256": label["window_binding"]["window_identity_sha256"],
        "outer_fold_id": label["outer_fold_id"],
        "outer_target": label["outer_target"],
        "source_domain": label["source_domain"],
        "base_seed": label["base_seed"],
        "derived_seed": label["derived_seed"],
        "detector_role": label["detector_role"],
        "oof_fold_index": label["oof_fold_index"],
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "threshold_algorithm": STAGE2_THRESHOLD_ALGORITHM,
        "score_dtype": "float64",
        "event_threshold_cap": None,
        "event_thresholds_capped": False,
        "global_exact": True,
        "endpoints": [0.0, 1.0],
        "matching_contract": STAGE2_MATCHING_CONTRACT,
        "connectivity": 8,
        "object_matching": "maximum-cardinality-one-to-one-overlap",
        "false_alarm_pixel_numerator": "predicted_pixels_outside_GT_foreground",
        "false_alarm_pixel_denominator": "all_native_resolution_query_pixels",
        "query_size": 28,
        "decision_seal_binding": label["decision_seal_binding"],
        "ordered_query_identity_sha256_algorithm": STAGE2_QUERY_IDENTITY_ALGORITHM,
        "ordered_query_identity_sha256": attachment.ordered_query_identity_sha256,
        "window_binding": dict(label["window_binding"]),
        "score_manifest_binding": dict(label["score_manifest_binding"]),
        "score_bindings": dict(attachment.score_manifest.bindings),
        "curve_sha256": _sha256_file_stable(csv_path),
        "curve_rows_sha256_algorithm": STAGE2_CURVE_ROWS_ALGORITHM,
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise ValueError(f"curve manifest {field} mismatch")
    _exact_bool(payload["development_only"], "development_only", True)
    _exact_bool(payload["official_test_accessed"], "official_test_accessed", False)
    if payload["path_anchor"] != "repository_root":
        raise ValueError("curve path_anchor mismatch")
    label_binding = payload["label_manifest_binding"]
    if not isinstance(label_binding, Mapping) or set(label_binding) != {
        "path", "sha256", "labels_content_sha256"
    }:
        raise ValueError("label_manifest_binding closure mismatch")
    if label_binding["sha256"] != attachment.manifest_sha256 or label_binding["labels_content_sha256"] != attachment.labels_content_sha256:
        raise ValueError("curve/label binding mismatch")
    declared_label_path = _relative_path(label_binding["path"], "label_manifest_binding.path")
    if bundle_root_override is None:
        expected_label_path = _repo_relative(attachment.path, root)
    else:
        first_label = attachment.payload["labels"][0]["label_file"]
        expected_label_path = (
            PurePosixPath(_relative_path(first_label, "labels[0].label_file")).parent
            / "label-manifest.json"
        ).as_posix()
    if declared_label_path != expected_label_path:
        raise ValueError("curve label-manifest path binding mismatch")
    curve_relative = _relative_path(payload["curve_file"], "curve_file")
    expected_curve_path = (
        PurePosixPath(expected_label_path).parent / csv_path.name
    ).as_posix()
    if curve_relative != expected_curve_path:
        raise ValueError("curve_file full path mismatch")


def _verify_bundle_publication_lock(
    bundle_root: Path,
    owned_lock: tuple[str | Path, tuple[int, int]] | None,
) -> None:
    lock = bundle_root.parent / f".{bundle_root.name}.lock"
    if not os.path.lexists(lock):
        return
    if owned_lock is None:
        raise RuntimeError(f"bundle publication lock is active: {lock}")
    allowed_path = Path(owned_lock[0]).expanduser().absolute()
    if allowed_path != lock.absolute():
        raise RuntimeError("bundle publication lock authorization path mismatch")
    info = lock.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != owned_lock[1]:
        raise RuntimeError("bundle publication lock authorization inode mismatch")


def _write_curve_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            descriptor = -1
            writer = csv.DictWriter(stream, fieldnames=CURVE_FIELDS, extrasaction="raise", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                if set(row) != set(CURVE_FIELDS):
                    raise ValueError("curve row field closure mismatch")
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_curve_csv(path: Path) -> ArrayBackedCurveRows:
    integer_fields = {
        "tp_objects", "gt_objects", "pred_components", "fp_components",
        "fp_pixels", "total_pixels", "num_images",
    }
    values: dict[str, list[float | int]] = {field: [] for field in CURVE_FIELDS}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CURVE_FIELDS:
            raise ValueError("curve CSV header closure/order mismatch")
        for raw in reader:
            if set(raw) != set(CURVE_FIELDS) or any(value in {None, ""} for value in raw.values()):
                raise ValueError("curve CSV row is incomplete")
            for field in CURVE_FIELDS:
                if field in integer_fields:
                    values[field].append(int(raw[field]))
                else:
                    value = float(raw[field])
                    if not np.isfinite(value):
                        raise ValueError("curve CSV contains nonfinite values")
                    values[field].append(value)
    if not values["threshold"]:
        raise ValueError("curve CSV is empty")
    return ArrayBackedCurveRows(
        {
            field: np.asarray(
                values[field],
                dtype=(np.float64 if field in ArrayBackedCurveRows._FLOAT_FIELDS else np.int64),
            )
            for field in CURVE_FIELDS
        }
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file_stable(path: Path) -> str:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("curve input must be a regular file")
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    digest = _sha256_file(path)
    after = path.stat(follow_symlinks=False)
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
        raise RuntimeError("curve input changed while hashed")
    return digest


def _input_file(value: str | Path, root: Path, name: str, override: str | Path | None) -> Path:
    path = Path(value).expanduser().absolute()
    allowed = [root]
    if override is not None:
        allowed.append(Path(override).expanduser().absolute())
    if not any(path == parent or parent in path.parents for parent in allowed):
        raise ValueError(f"{name} is outside allowed roots")
    current = next(parent for parent in allowed if path == parent or parent in path.parents)
    for part in path.relative_to(current).parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{name} contains a symlink component")
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular file")
    return path


def _owned_directory(value: str | Path, root: Path, name: str) -> Path:
    path = Path(value).expanduser().absolute()
    if root not in path.parents or not path.is_dir() or path.is_symlink():
        raise ValueError(f"{name} must be a real repository subdirectory")
    return path


def _future_directory(value: str | Path, root: Path, name: str) -> Path:
    path = Path(value).expanduser().absolute()
    if root not in path.parents:
        raise ValueError(f"{name} must be under repository root")
    if os.path.lexists(path):
        raise FileExistsError(f"{name} already exists")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"{name} parent must be a real directory")
    return path


def _repo_relative(path: Path, root: Path) -> str:
    absolute = path.absolute()
    if root not in absolute.parents:
        raise ValueError("artifact path must be repository-relative")
    return absolute.relative_to(root).as_posix()


def _relative_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or value != pure.as_posix() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} must be canonical repository-relative POSIX")
    lowered = value.lower().replace("-", "_")
    if "official_test" in lowered or "officialtest" in lowered:
        raise ValueError(f"{name} may not reference official test")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_bool(value: object, name: str, expected: bool) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact JSON boolean")
    if value is not expected:
        raise ValueError(f"{name} must be {expected}")


def _repository_root(value: str | Path | None) -> Path:
    root = (REPOSITORY_ROOT if value is None else Path(value).expanduser()).absolute()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("repository_root must be a real directory")
    return root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-manifest", required=True)
    parser.add_argument("--window-manifest-sha256", required=True)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--score-manifest", required=True)
    parser.add_argument("--score-manifest-sha256", required=True)
    parser.add_argument("--label-manifest", required=True)
    parser.add_argument("--label-manifest-sha256", required=True)
    parser.add_argument("--expected-role", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    curve = build_stage2_query_curve(
        args.window_manifest,
        args.score_manifest,
        args.label_manifest,
        window_manifest_sha256=args.window_manifest_sha256,
        score_manifest_sha256=args.score_manifest_sha256,
        label_manifest_sha256=args.label_manifest_sha256,
        window_id=args.window_id,
        expected_role=args.expected_role,
        repository_root=args.repository_root,
    )
    # The standalone CLI requires a new bundle parent so publication can be
    # all-or-nothing.  The canonical label exporter writes the same artifacts
    # into the complete per-window label+curve bundle.
    root = _repository_root(args.repository_root)
    output = Path(args.output).expanduser().absolute()
    final_root = output.parent
    if os.path.lexists(final_root):
        raise FileExistsError("--output parent must not exist for atomic bundle publication")
    grandparent = final_root.parent
    if not grandparent.is_dir() or grandparent.is_symlink() or root not in final_root.parents:
        raise ValueError("--output must name a new repository-local bundle directory")
    staging = Path(tempfile.mkdtemp(prefix=f".{final_root.name}.staging-", dir=grandparent))
    try:
        write_stage2_query_curve_artifacts(
            curve,
            staging_root=staging,
            final_root=final_root,
            repository_root=root,
            curve_filename=output.name,
        )
        os.rename(staging, final_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Wrote {len(curve.rows)} exact operating points to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

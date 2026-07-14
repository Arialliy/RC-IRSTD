"""Monotonic probability-threshold sweeps over native-resolution score maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

from data_ext.label_manifest_artifacts import (
    load_label_mask,
    verify_label_attachment,
)
from data_ext.score_manifest_artifacts import (
    STRICT_THRESHOLD_SEMANTICS,
    verify_score_manifest_artifacts,
)

from .component_matching import (
    aggregate_match_results,
    match_components,
    prepare_target,
)


CURVE_SCHEMA_VERSION = 2
THRESHOLD_GRID_VERSION = "rc-tail-grid-v1"
THRESHOLD_SEMANTICS = STRICT_THRESHOLD_SEMANTICS
THRESHOLD_MODES = ("fixed", "adaptive", "exact")
DEFAULT_HIGH_TAIL_LOWER_BOUND = 0.99
DEFAULT_EVENT_THRESHOLD_CAP = 4096

ThresholdMode = Literal["fixed", "adaptive", "exact"]


CURVE_FIELDS = (
    "threshold",
    "pd",
    "fa_pixel",
    "fa_component_mp",
    "tp_objects",
    "gt_objects",
    "pred_components",
    "fp_components",
    "fp_pixels",
    "total_pixels",
    "num_images",
)


@dataclass(frozen=True)
class ScoreMapRecord:
    probability: np.ndarray
    mask: np.ndarray
    image_id: str = ""
    path: str = ""


@dataclass(frozen=True)
class ThresholdPlan:
    """Threshold values plus an auditable event-score coverage contract."""

    thresholds: np.ndarray
    audit: dict[str, object]


def default_threshold_grid() -> np.ndarray:
    """Return a dense grid with both legal probability-domain endpoints.

    Under strict ``probability > threshold`` semantics, 1 is the guaranteed
    empty-prediction endpoint.  Zero is only the lowest legal threshold; it is
    not a guaranteed full-prediction endpoint because zero-score pixels remain
    excluded.
    """

    return np.unique(
        np.concatenate(
            [
                np.asarray([0.0], dtype=np.float64),
                np.linspace(0.0, 0.90, 91, dtype=np.float64),
                np.linspace(0.90, 0.99, 181, dtype=np.float64),
                np.linspace(0.99, 0.999, 181, dtype=np.float64),
                np.linspace(0.999, 0.99999, 201, dtype=np.float64),
                np.asarray([1.0], dtype=np.float64),
            ]
        )
    )


def threshold_grid_metadata(
    thresholds: Sequence[float] | np.ndarray | None = None,
    *,
    threshold_audit: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Describe the reproducible threshold-grid and curve schema contract."""

    grid = normalise_thresholds(thresholds)
    grid_version = (
        THRESHOLD_GRID_VERSION
        if np.array_equal(grid, default_threshold_grid())
        else "custom"
    )
    metadata = {
        "schema_version": CURVE_SCHEMA_VERSION,
        "threshold_grid_version": grid_version,
        "threshold_semantics": THRESHOLD_SEMANTICS,
        "num_operating_points": int(grid.size),
        "thresholds": [float(value) for value in grid],
    }
    if threshold_audit is not None:
        metadata.update(dict(threshold_audit))
        # These fields describe the materialised curve and cannot be
        # overridden by caller-provided audit metadata.
        metadata["schema_version"] = CURVE_SCHEMA_VERSION
        metadata["threshold_semantics"] = THRESHOLD_SEMANTICS
        metadata["num_operating_points"] = int(grid.size)
    return metadata


def normalise_thresholds(
    thresholds: Sequence[float] | np.ndarray | None,
    *,
    include_endpoints: bool = True,
) -> np.ndarray:
    """Validate, deduplicate and sort thresholds in ascending order."""

    if thresholds is None:
        result = default_threshold_grid()
    else:
        result = np.asarray(list(thresholds), dtype=np.float64).reshape(-1)
        if result.size == 0:
            raise ValueError("At least one threshold is required")
        if not np.isfinite(result).all():
            raise ValueError("Thresholds contain NaN or infinite values")
        if np.any((result < 0.0) | (result > 1.0)):
            raise ValueError("Probability thresholds must lie within [0, 1]")
        if include_endpoints:
            result = np.concatenate((result, np.asarray([0.0, 1.0])))
        result = np.unique(result)
    result.sort()
    if include_endpoints and (result[0] != 0.0 or result[-1] != 1.0):
        raise AssertionError("Threshold normalisation failed to include 0 and 1")
    return result


def build_threshold_plan(
    records: Sequence[ScoreMapRecord],
    thresholds: Sequence[float] | np.ndarray | None = None,
    *,
    mode: ThresholdMode = "fixed",
    high_tail_lower_bound: float = DEFAULT_HIGH_TAIL_LOWER_BOUND,
    event_threshold_cap: int | None = None,
) -> ThresholdPlan:
    """Build fixed, adaptive-tail, or globally exact event thresholds.

    An event threshold is a probability value observed in the current query.
    With strict ``probability > threshold`` semantics, evaluating the value
    itself captures the operating point immediately after all equal-score
    pixels disappear.  ``exact`` uses every unique query score. ``adaptive``
    uses scores at or above ``high_tail_lower_bound``.  When capped, the
    highest scores are retained so the manifest can state a concrete suffix
    of the score range that is still covered exactly.
    """

    if mode not in THRESHOLD_MODES:
        raise ValueError(f"mode must be one of {THRESHOLD_MODES}, got {mode!r}")
    if not 0.0 <= high_tail_lower_bound <= 1.0:
        raise ValueError(
            "high_tail_lower_bound must lie within [0, 1], got "
            f"{high_tail_lower_bound}"
        )
    if event_threshold_cap is not None and event_threshold_cap <= 0:
        raise ValueError("event_threshold_cap must be positive or None")

    base = normalise_thresholds(thresholds)
    if mode == "fixed":
        return ThresholdPlan(
            thresholds=base,
            audit={
                "threshold_mode_requested": "fixed",
                "threshold_mode": "fixed",
                "event_candidate_count": 0,
                "event_threshold_count": 0,
                "event_thresholds_added": 0,
                "event_threshold_cap": None,
                "event_thresholds_capped": False,
                "event_candidate_score_lower_bound": None,
                "event_coverage_score_lower_bound": None,
                "event_coverage_fraction_lower_bound": 0.0,
                "global_exact": False,
            },
        )

    candidate_lower_bound = 0.0 if mode == "exact" else high_tail_lower_bound
    candidate_arrays = [
        record.probability[record.probability >= candidate_lower_bound].astype(
            np.float64,
            copy=False,
        )
        for record in records
    ]
    non_empty = [values.reshape(-1) for values in candidate_arrays if values.size]
    candidates = (
        np.unique(np.concatenate(non_empty))
        if non_empty
        else np.empty(0, dtype=np.float64)
    )

    already_covered = np.isin(candidates, base, assume_unique=True)
    additional_candidates = candidates[~already_covered]
    capped = (
        event_threshold_cap is not None
        and additional_candidates.size > event_threshold_cap
    )
    if capped:
        # A top-score suffix gives a stronger audit statement than evenly
        # spaced sampling: every unique event at or above the reported lower
        # bound is represented exactly.
        selected = additional_candidates[-event_threshold_cap:]
    else:
        selected = additional_candidates

    planned = np.unique(np.concatenate((base, selected)))
    planned.sort()
    covered_mask = np.isin(candidates, planned, assume_unique=True)
    covered_count = int(np.count_nonzero(covered_mask))
    candidate_count = int(candidates.size)
    coverage_fraction = (
        covered_count / candidate_count if candidate_count else 1.0
    )

    coverage_score_lower_bound: float | None = None
    if candidate_count:
        first_suffix_index = candidate_count
        for index in range(candidate_count - 1, -1, -1):
            if not covered_mask[index]:
                break
            first_suffix_index = index
        if first_suffix_index < candidate_count:
            coverage_score_lower_bound = float(candidates[first_suffix_index])

    global_exact = mode == "exact" and covered_count == candidate_count and not capped
    effective_mode = f"{mode}_capped" if capped else mode
    return ThresholdPlan(
        thresholds=planned,
        audit={
            "threshold_mode_requested": mode,
            "threshold_mode": effective_mode,
            "event_candidate_count": candidate_count,
            "event_threshold_count": covered_count,
            "event_thresholds_added": int(selected.size),
            "event_threshold_cap": event_threshold_cap,
            "event_thresholds_capped": bool(capped),
            "event_candidate_score_lower_bound": float(candidate_lower_bound),
            "event_coverage_score_lower_bound": coverage_score_lower_bound,
            "event_coverage_fraction_lower_bound": float(coverage_fraction),
            "global_exact": bool(global_exact),
        },
    )


def load_score_map(path: str | Path) -> ScoreMapRecord:
    """Load and validate one compressed score-map artifact."""

    score_path = Path(path)
    with np.load(score_path, allow_pickle=False) as payload:
        if "prob" not in payload or "mask" not in payload:
            raise KeyError(f"Score map must contain 'prob' and 'mask': {score_path}")
        probability = np.asarray(payload["prob"], dtype=np.float32).squeeze()
        mask = np.asarray(payload["mask"]).squeeze()
        if "image_id" in payload:
            image_id = str(np.asarray(payload["image_id"]).item())
        else:
            image_id = score_path.stem
    return validate_score_map(
        ScoreMapRecord(
            probability=probability,
            mask=mask,
            image_id=image_id,
            path=str(score_path),
        )
    )


def validate_score_map(record: ScoreMapRecord) -> ScoreMapRecord:
    probability = np.asarray(record.probability, dtype=np.float32).squeeze()
    mask = np.asarray(record.mask).squeeze()
    if probability.ndim != 2 or mask.ndim != 2:
        raise ValueError(
            f"Score map and mask must be 2D, got {probability.shape} and {mask.shape}"
        )
    if probability.shape != mask.shape:
        raise ValueError(
            f"Score map/mask shape mismatch for {record.image_id!r}: "
            f"{probability.shape} vs {mask.shape}"
        )
    if not np.isfinite(probability).all():
        raise ValueError(f"Probability map contains NaN/Inf: {record.image_id!r}")
    if probability.size and (probability.min() < 0.0 or probability.max() > 1.0):
        raise ValueError(f"Probability map is outside [0, 1]: {record.image_id!r}")
    return ScoreMapRecord(
        probability=probability,
        mask=mask.astype(bool, copy=False),
        image_id=record.image_id,
        path=record.path,
    )


def discover_score_maps(
    score_dir: str | Path,
    image_ids: Sequence[str] | None = None,
    *,
    allow_legacy_combined_diagnostic: bool = False,
) -> list[Path]:
    """Resolve score maps in manifest order and ignore stale unlisted files.

    ``image_ids`` enables a causal query-only curve.  When a score manifest is
    present, every requested ID must occur exactly once and the returned order
    follows the request.  An interrupted export marker is always fatal.
    """

    root = Path(score_dir)
    incomplete_marker = root / ".export_incomplete"
    if incomplete_marker.exists():
        raise RuntimeError(
            f"Score export under {root} is incomplete; rerun export with --overwrite"
        )

    requested = None if image_ids is None else [str(value) for value in image_ids]
    if requested is not None and len(set(requested)) != len(requested):
        raise ValueError("Requested image IDs contain duplicates")

    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        verified = verify_score_manifest_artifacts(
            manifest_path,
            image_ids=requested,
            require_mask=allow_legacy_combined_diagnostic,
            require_native_contract=True,
            allow_legacy_combined_diagnostic=allow_legacy_combined_diagnostic,
        )
        paths = [item.score_path for item in verified.selected_items]
    else:
        paths = sorted(root.glob("*.npz"), key=lambda path: path.name)
        if requested is not None:
            by_stem = {path.stem: path for path in paths}
            missing = [image_id for image_id in requested if image_id not in by_stem]
            if missing:
                raise KeyError(f"Requested image IDs are absent from score directory: {missing}")
            paths = [by_stem[image_id] for image_id in requested]

    if not paths:
        raise FileNotFoundError(f"No .npz score maps found under {score_dir}")
    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Score manifest references missing files: {missing_paths}")
    return paths


def load_attached_score_maps(
    score_manifest: str | Path,
    label_manifest: str | Path,
    *,
    image_ids: Sequence[str] | None = None,
) -> tuple[list[ScoreMapRecord], str]:
    """Load strict label-free scores and their independently bound labels."""

    attachment = verify_label_attachment(
        score_manifest,
        label_manifest,
        image_ids=image_ids,
    )
    score_items = attachment.score_manifest.selected_items
    label_items = attachment.selected_items
    if len(score_items) != len(label_items):
        raise RuntimeError("verified score/label selections have different lengths")
    records: list[ScoreMapRecord] = []
    for score_item, label_item in zip(score_items, label_items):
        if score_item.image_id != label_item.image_id:
            raise RuntimeError("verified score/label selection order changed")
        with np.load(score_item.score_path, allow_pickle=False) as payload:
            probability = np.asarray(payload["prob"], dtype=np.float32)
        records.append(
            validate_score_map(
                ScoreMapRecord(
                    probability=probability,
                    mask=load_label_mask(label_item),
                    image_id=score_item.image_id,
                    path=str(score_item.score_path),
                )
            )
        )
    return records, attachment.manifest_sha256


def sweep_thresholds(
    score_maps: Iterable[ScoreMapRecord | str | Path | Mapping[str, object]],
    thresholds: Sequence[float] | np.ndarray | None = None,
    *,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    threshold_mode: ThresholdMode = "fixed",
    high_tail_lower_bound: float = DEFAULT_HIGH_TAIL_LOWER_BOUND,
    event_threshold_cap: int | None = None,
    return_metadata: bool = False,
) -> (
    list[dict[str, float | int]]
    | tuple[list[dict[str, float | int]], dict[str, object]]
):
    """Evaluate thresholds from low to high and return aggregate curve rows."""

    records = [_coerce_score_map(item) for item in score_maps]
    if not records:
        raise ValueError("At least one score map is required")
    plan = build_threshold_plan(
        records,
        thresholds,
        mode=threshold_mode,
        high_tail_lower_bound=high_tail_lower_bound,
        event_threshold_cap=event_threshold_cap,
    )
    ordered_thresholds = plan.thresholds
    # GT masks never change during a sweep.  Preparing them here avoids a
    # connected-component label/regionprops pass for every threshold.
    prepared_targets = [prepare_target(record.mask) for record in records]

    rows: list[dict[str, float | int]] = []
    previous_fa_pixel = float("inf")
    for threshold in ordered_thresholds:
        image_results = [
            match_components(
                record.probability > threshold,
                prepared_target,
                rule=matching_rule,
                centroid_distance=centroid_distance,
            )
            for record, prepared_target in zip(records, prepared_targets)
        ]
        aggregate = aggregate_match_results(image_results)
        row: dict[str, float | int] = {
            "threshold": float(threshold),
            **aggregate,
        }
        # Because predictions use ``probability > threshold``, the predicted
        # pixel sets are nested and pixel false alarms must be monotone.
        fa_pixel = float(row["fa_pixel"])
        if fa_pixel > previous_fa_pixel + 1e-15:
            raise RuntimeError(
                "fa_pixel increased during an ascending threshold sweep; "
                "score maps or threshold ordering are inconsistent"
            )
        previous_fa_pixel = fa_pixel
        rows.append(row)
    if return_metadata:
        return rows, dict(plan.audit)
    return rows


def write_curve_csv(
    rows: Iterable[Mapping[str, object]],
    output_path: str | Path,
    *,
    write_manifest: bool = True,
    image_ids: Sequence[str] | None = None,
    score_manifest: str | Path | None = None,
    threshold_audit: Mapping[str, object] | None = None,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
    label_manifest: str | Path | None = None,
    diagnostic_legacy_combined: bool = False,
) -> Path:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty threshold curve")
    if matching_rule not in {"overlap", "centroid"}:
        raise ValueError("matching_rule must be 'overlap' or 'centroid'")
    if isinstance(centroid_distance, bool):
        raise TypeError("centroid_distance must be a finite positive number")
    centroid_distance = float(centroid_distance)
    if not np.isfinite(centroid_distance) or centroid_distance <= 0.0:
        raise ValueError("centroid_distance must be a finite positive number")
    if label_manifest is not None and diagnostic_legacy_combined:
        raise ValueError(
            "label_manifest and diagnostic_legacy_combined are mutually exclusive"
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    invariant_fields = ("gt_objects", "total_pixels", "num_images")
    for field in invariant_fields:
        values = {int(row[field]) for row in rows}
        if len(values) != 1:
            raise ValueError(f"curve field {field!r} changes across thresholds")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CURVE_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if write_manifest:
        manifest = threshold_grid_metadata(
            [float(row["threshold"]) for row in rows],
            threshold_audit=threshold_audit,
        )
        manifest["path_anchor"] = "manifest_directory"
        manifest["curve_file"] = path.name
        manifest["curve_sha256"] = _sha256(path)
        manifest["num_images"] = int(rows[0]["num_images"])
        manifest["gt_objects"] = int(rows[0]["gt_objects"])
        manifest["total_pixels"] = int(rows[0]["total_pixels"])
        manifest["matching_rule"] = matching_rule
        manifest["centroid_distance"] = centroid_distance
        manifest["evaluation_scope"] = (
            "score_bound_label_attachment_verified"
            if label_manifest is not None
            else "legacy_combined_npz_diagnostic"
        )
        if image_ids is not None:
            ids = [str(value) for value in image_ids]
            if len(ids) != int(rows[0].get("num_images", len(ids))):
                raise ValueError("image_ids length does not match curve num_images")
            if len(set(ids)) != len(ids):
                raise ValueError("image_ids contain duplicates")
            manifest["image_ids"] = ids
        if score_manifest is not None:
            source = Path(score_manifest).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Score manifest does not exist: {source}")
            source_payload = json.loads(source.read_text(encoding="utf-8"))
            manifest["score_manifest_file"] = Path(
                os.path.relpath(source, start=path.parent.resolve())
            ).as_posix()
            manifest["score_manifest_sha256"] = _sha256(source)
            manifest["target_dataset"] = source_payload.get("target_dataset")
            manifest["detector_weight_sha256"] = source_payload.get("weight_sha256")
            manifest["score_manifest_num_images"] = source_payload.get("num_images")
        if label_manifest is not None:
            if score_manifest is None:
                raise ValueError("label_manifest requires score_manifest")
            label_source = Path(label_manifest).expanduser().resolve()
            if not label_source.is_file():
                raise FileNotFoundError(
                    f"Label manifest does not exist: {label_source}"
                )
            label_attachment = verify_label_attachment(
                Path(score_manifest).expanduser().resolve(),
                label_source,
                image_ids=image_ids,
            )
            label_payload = label_attachment.payload
            manifest["label_manifest_file"] = Path(
                os.path.relpath(label_source, start=path.parent.resolve())
            ).as_posix()
            manifest["label_manifest_sha256"] = (
                label_attachment.manifest_sha256
            )
            manifest["label_manifest_content_sha256"] = (
                label_attachment.content_sha256
            )
            manifest["label_manifest_num_images"] = label_payload.get("num_images")
            manifest["label_manifest_target_dataset"] = label_payload.get(
                "target_dataset"
            )
        elif not diagnostic_legacy_combined:
            raise ValueError(
                "claim-bearing curve manifests require an independent label_manifest; "
                "set diagnostic_legacy_combined=True only for legacy diagnostics"
            )
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest_temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
        try:
            manifest_temporary.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.replace(manifest_temporary, manifest_path)
        finally:
            if manifest_temporary.exists():
                manifest_temporary.unlink()
    return path


def read_curve_csv(path: str | Path) -> list[dict[str, float | int]]:
    integer_fields = {
        "tp_objects",
        "gt_objects",
        "pred_components",
        "fp_components",
        "fp_pixels",
        "total_pixels",
        "num_images",
    }
    rows: list[dict[str, float | int]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for raw_row in csv.DictReader(stream):
            row: dict[str, float | int] = {}
            for field in CURVE_FIELDS:
                if field not in raw_row or raw_row[field] in {None, ""}:
                    raise ValueError(f"Curve CSV is missing value for {field!r}")
                row[field] = (
                    int(raw_row[field])
                    if field in integer_fields
                    else float(raw_row[field])
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Curve CSV is empty: {path}")
    return rows


def _coerce_score_map(
    item: ScoreMapRecord | str | Path | Mapping[str, object],
) -> ScoreMapRecord:
    if isinstance(item, ScoreMapRecord):
        return validate_score_map(item)
    if isinstance(item, (str, Path)):
        return load_score_map(item)
    if isinstance(item, Mapping):
        if "prob" not in item or "mask" not in item:
            raise KeyError("Score-map mapping must contain 'prob' and 'mask'")
        return validate_score_map(
            ScoreMapRecord(
                probability=np.asarray(item["prob"]),
                mask=np.asarray(item["mask"]),
                image_id=str(item.get("image_id", "")),
            )
        )
    raise TypeError(f"Unsupported score-map type: {type(item).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--label-manifest",
        help="Independent score-bound label manifest required for verified curves.",
    )
    parser.add_argument(
        "--diagnostic-allow-embedded-mask",
        action="store_true",
        help="Allow legacy combined score/mask NPZs only for diagnostics.",
    )
    parser.add_argument("--matching-rule", choices=("overlap", "centroid"), default="overlap")
    parser.add_argument("--centroid-distance", type=float, default=3.0)
    parser.add_argument(
        "--threshold-mode",
        choices=THRESHOLD_MODES,
        default="adaptive",
        help=(
            "fixed grid, adaptive query high-tail events (default), or every "
            "unique query score for an exact event sweep"
        ),
    )
    parser.add_argument(
        "--high-tail-lower-bound",
        type=float,
        default=DEFAULT_HIGH_TAIL_LOWER_BOUND,
        help="Lowest query score added in adaptive mode.",
    )
    parser.add_argument(
        "--event-threshold-cap",
        type=int,
        default=DEFAULT_EVENT_THRESHOLD_CAP,
        help=(
            "Maximum query event thresholds added beyond the base grid; "
            "0 disables the cap. Capped runs are explicitly marked non-exact."
        ),
    )
    parser.add_argument(
        "--image-id",
        action="append",
        default=[],
        help="Restrict the curve to this image ID; repeat to preserve query order.",
    )
    parser.add_argument(
        "--image-id-file",
        help="Text file containing one query image ID per line.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    requested_ids = list(args.image_id)
    if args.image_id_file:
        requested_ids.extend(
            line.strip()
            for line in Path(args.image_id_file).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("Requested image IDs contain duplicates")
    score_manifest = Path(args.score_dir) / "manifest.json"
    if args.label_manifest:
        if args.diagnostic_allow_embedded_mask:
            raise ValueError(
                "--label-manifest and --diagnostic-allow-embedded-mask are mutually exclusive"
            )
        if not score_manifest.is_file():
            raise FileNotFoundError(
                "--label-manifest requires <score-dir>/manifest.json"
            )
        records, _ = load_attached_score_maps(
            score_manifest,
            args.label_manifest,
            image_ids=requested_ids or None,
        )
    else:
        if not args.diagnostic_allow_embedded_mask:
            raise ValueError(
                "verified threshold sweeps require --label-manifest; use "
                "--diagnostic-allow-embedded-mask only for legacy diagnostics"
            )
        score_paths = discover_score_maps(
            args.score_dir,
            requested_ids or None,
            allow_legacy_combined_diagnostic=True,
        )
        records = [load_score_map(path) for path in score_paths]
    if args.event_threshold_cap < 0:
        raise ValueError("--event-threshold-cap must be non-negative")
    event_threshold_cap = args.event_threshold_cap or None
    rows, threshold_audit = sweep_thresholds(
        records,
        matching_rule=args.matching_rule,
        centroid_distance=args.centroid_distance,
        threshold_mode=args.threshold_mode,
        high_tail_lower_bound=args.high_tail_lower_bound,
        event_threshold_cap=event_threshold_cap,
        return_metadata=True,
    )
    output_path = write_curve_csv(
        rows,
        args.output,
        image_ids=[record.image_id for record in records],
        score_manifest=score_manifest if score_manifest.is_file() else None,
        threshold_audit=threshold_audit,
        matching_rule=args.matching_rule,
        centroid_distance=args.centroid_distance,
        label_manifest=args.label_manifest,
        diagnostic_legacy_combined=args.diagnostic_allow_embedded_mask,
    )
    print(
        f"Wrote {len(rows)} operating points to {output_path} "
        f"(mode={threshold_audit['threshold_mode']}, "
        f"events={threshold_audit['event_threshold_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

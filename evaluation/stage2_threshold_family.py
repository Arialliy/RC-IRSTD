"""Deterministic Stage-2 T0--T9 threshold family and prelabel sealing.

T0--T8 consume only an already verified unlabeled context package, source-only
references and (for T6--T8) immutable calibrator checkpoints.  Their outputs
can be published as one atomic T0--T8 bundle.  T9 is intentionally exposed
only through :func:`build_t9_postlabel_diagnostic` and can never enter that
bundle.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from data_ext.stage2_threshold_decision import (
    COMPLETE_OUTCOME,
    DECISION_ARTIFACT_TYPE,
    DECISION_SCHEMA,
    DECISION_SET_ARTIFACT_TYPE,
    DECISION_SET_COMMIT_ARTIFACT_TYPE,
    DECISION_SET_COMMIT_SCHEMA,
    DECISION_SET_SCHEMA,
    PIXEL_BUDGET_GRID,
    PRELABEL_METHOD_ORDER,
    STRICT_THRESHOLD_SEMANTICS,
    T5_MISSING_OUTCOME,
    canonical_json_bytes,
    canonical_json_sha256,
    verify_stage2_threshold_decision_set,
)


SOURCE_THRESHOLD_REFERENCE_SCHEMA = "rc-irstd.stage2-source-threshold-reference.v1"
SOURCE_THRESHOLD_REFERENCE_ARTIFACT_TYPE = (
    "rc_irstd_stage2_source_threshold_reference"
)
T9_DIAGNOSTIC_SCHEMA = "rc-irstd.stage2-postlabel-oracle-diagnostic.v1"
METHOD_NAMES = {
    "T0": "fixed_0.5",
    "T1": "pooled_two_source_safe_threshold",
    "T2": "safer_of_two_source_thresholds",
    "T3": "nearest_source_safe_threshold",
    "T4": "context_order_statistic",
    "T5": "context_evt_gpd",
    "T6": "direct_no_reject_calibrator",
    "T7": "monotone_oracle_no_reject_calibrator",
    "T8": "risk_aligned_monotone_no_reject_calibrator",
    "T9": "postlabel_target_future_oracle_diagnostic",
}
_SHA_HEX = frozenset("0123456789abcdef")


class Stage2ThresholdFamilyError(ValueError):
    """Raised when a threshold-family input violates the frozen contract."""


@dataclass(frozen=True)
class _Publication:
    final_root: Path
    staging_root: Path
    staging_identity: tuple[int, int]
    staging_descriptor: int
    lock_path: Path
    lock_identity: tuple[int, int]
    lock_descriptor: int


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _SHA_HEX
    ):
        raise Stage2ThresholdFamilyError(f"{name} must be lowercase SHA-256")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Stage2ThresholdFamilyError(f"{name} must be non-empty text")
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage2ThresholdFamilyError(f"{name} must be int >= {minimum}")
    return value


def _budget_grid(value: Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise Stage2ThresholdFamilyError("budget grid must contain exactly three values")
    result = tuple(float(item) for item in value)
    if result != PIXEL_BUDGET_GRID:
        raise Stage2ThresholdFamilyError(
            f"budget grid must equal {list(PIXEL_BUDGET_GRID)}"
        )
    return result  # type: ignore[return-value]


def make_shared_input_bindings(
    *,
    context_package_path: str,
    context_package_sha256: str,
    context_package_commit_path: str,
    context_package_commit_sha256: str,
    window_id: str,
    window_identity_sha256: str,
    ordered_query_identity_sha256: str,
    score_manifest_sha256: str,
    score_records_content_sha256: str,
    detector_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Create the exact method-agnostic identity shared by T0--T8."""

    binding: dict[str, Any] = {
        "context_package": {
            "path": _text(context_package_path, "context_package_path"),
            "sha256": _sha(context_package_sha256, "context_package_sha256"),
        },
        "context_package_commit": {
            "path": _text(context_package_commit_path, "context_package_commit_path"),
            "sha256": _sha(
                context_package_commit_sha256, "context_package_commit_sha256"
            ),
        },
        "window_id": _text(window_id, "window_id"),
        "window_identity_sha256": _sha(
            window_identity_sha256, "window_identity_sha256"
        ),
        "ordered_query_identity_sha256": _sha(
            ordered_query_identity_sha256, "ordered_query_identity_sha256"
        ),
        "score_manifest_sha256": _sha(
            score_manifest_sha256, "score_manifest_sha256"
        ),
        "score_records_content_sha256": _sha(
            score_records_content_sha256, "score_records_content_sha256"
        ),
        "detector_checkpoint_sha256": _sha(
            detector_checkpoint_sha256, "detector_checkpoint_sha256"
        ),
    }
    binding["shared_input_identity_sha256"] = canonical_json_sha256(
        {
            "context_package_sha256": binding["context_package"]["sha256"],
            "context_package_commit_sha256": binding["context_package_commit"][
                "sha256"
            ],
            "window_id": binding["window_id"],
            "window_identity_sha256": binding["window_identity_sha256"],
            "ordered_query_identity_sha256": binding[
                "ordered_query_identity_sha256"
            ],
            "score_manifest_sha256": binding["score_manifest_sha256"],
            "score_records_content_sha256": binding[
                "score_records_content_sha256"
            ],
            "detector_checkpoint_sha256": binding[
                "detector_checkpoint_sha256"
            ],
            "budget_grid": list(PIXEL_BUDGET_GRID),
        }
    )
    return binding


def _row_counts(row: Mapping[str, Any], name: str) -> tuple[Fraction, int, float]:
    required = {"threshold", "tp_objects", "gt_objects", "fp_pixels", "total_pixels"}
    if not required <= set(row):
        raise Stage2ThresholdFamilyError(f"{name} lacks sufficient-count fields")
    threshold = float(row["threshold"])
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise Stage2ThresholdFamilyError(f"{name}.threshold is invalid")
    tp = _strict_int(row["tp_objects"], f"{name}.tp_objects")
    gt = _strict_int(row["gt_objects"], f"{name}.gt_objects")
    fp = _strict_int(row["fp_pixels"], f"{name}.fp_pixels")
    pixels = _strict_int(row["total_pixels"], f"{name}.total_pixels", minimum=1)
    if tp > gt or fp > pixels:
        raise Stage2ThresholdFamilyError(f"{name} counts are impossible")
    pd = Fraction(tp, gt) if gt else Fraction(0, 1)
    return pd, fp, threshold


def select_source_safe_threshold(
    rows: Sequence[Mapping[str, Any]], pixel_budget: float
) -> dict[str, Any]:
    """Select exact safe row: Pd max, FP min, threshold max."""

    budget = float(pixel_budget)
    if budget not in PIXEL_BUDGET_GRID:
        raise Stage2ThresholdFamilyError("pixel_budget is outside the frozen grid")
    if not rows:
        raise Stage2ThresholdFamilyError("source safe curve is empty")
    eligible: list[tuple[Fraction, int, float, Mapping[str, Any]]] = []
    for index, row in enumerate(rows):
        pd, fp, threshold = _row_counts(row, f"rows[{index}]")
        pixels = int(row["total_pixels"])
        if Fraction(fp, pixels) <= Fraction.from_float(budget):
            eligible.append((pd, fp, threshold, row))
    if not eligible:
        raise Stage2ThresholdFamilyError("safe curve lacks a feasible endpoint")
    selected = max(eligible, key=lambda item: (item[0], -item[1], item[2]))
    row = selected[3]
    return {
        "threshold": float(selected[2]),
        "tp_objects": int(row["tp_objects"]),
        "gt_objects": int(row["gt_objects"]),
        "fp_pixels": int(row["fp_pixels"]),
        "total_pixels": int(row["total_pixels"]),
        "pd_numerator": int(selected[0].numerator),
        "pd_denominator": int(selected[0].denominator),
    }


def build_source_threshold_reference(
    *,
    pooled_curve: Sequence[Mapping[str, Any]],
    domain_curves: Mapping[str, Sequence[Mapping[str, Any]]],
    standardized_source_centers_0_86: Mapping[str, Sequence[float]],
    outer_fold_id: str,
    outer_target_domain: str,
    base_seed: int,
    derived_seed: int,
    detector_checkpoint_sha256: str,
    collection_path: str,
    collection_sha256: str,
    collection_commit_path: str,
    collection_commit_sha256: str,
    collection_identity_sha256: str,
    standardizer_fit_manifest_sha256: str,
    standardizer_train_collection_sha256: str,
) -> dict[str, Any]:
    """Build the source-only T1/T2/T3 reference from two complete domains."""

    domains = tuple(sorted(domain_curves))
    if len(domains) != 2 or outer_target_domain in domains:
        raise Stage2ThresholdFamilyError(
            "source threshold reference requires exactly two non-outer domains"
        )
    if tuple(sorted(standardized_source_centers_0_86)) != domains:
        raise Stage2ThresholdFamilyError("source-center domains do not match curves")
    centers: dict[str, list[float]] = {}
    for domain in domains:
        center = np.asarray(standardized_source_centers_0_86[domain], dtype=np.float64)
        if center.shape != (87,) or not np.isfinite(center).all():
            raise Stage2ThresholdFamilyError(
                f"standardized source center for {domain} must be finite 87D"
            )
        centers[domain] = center.tolist()
    pooled: list[dict[str, Any]] = []
    by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in domains}
    for budget in PIXEL_BUDGET_GRID:
        pooled.append(select_source_safe_threshold(pooled_curve, budget))
        for domain in domains:
            by_domain[domain].append(
                select_source_safe_threshold(domain_curves[domain], budget)
            )
    payload: dict[str, Any] = {
        "schema_version": SOURCE_THRESHOLD_REFERENCE_SCHEMA,
        "artifact_type": SOURCE_THRESHOLD_REFERENCE_ARTIFACT_TYPE,
        "artifact_status": "SOURCE_ONLY_COMPLETE",
        "development_only": True,
        "official_test_accessed": False,
        "outer_target_present": False,
        "outer_fold_id": _text(outer_fold_id, "outer_fold_id"),
        "outer_target_domain": _text(outer_target_domain, "outer_target_domain"),
        "base_seed": _strict_int(base_seed, "base_seed"),
        "derived_seed": _strict_int(derived_seed, "derived_seed", minimum=1),
        "detector_checkpoint_sha256": _sha(
            detector_checkpoint_sha256, "detector_checkpoint_sha256"
        ),
        "budget_grid": list(PIXEL_BUDGET_GRID),
        "source_domains": list(domains),
        "collection_binding": {
            "path": _text(collection_path, "collection_path"),
            "sha256": _sha(collection_sha256, "collection_sha256"),
            "commit_path": _text(collection_commit_path, "collection_commit_path"),
            "commit_sha256": _sha(
                collection_commit_sha256, "collection_commit_sha256"
            ),
            "collection_identity_sha256": _sha(
                collection_identity_sha256, "collection_identity_sha256"
            ),
        },
        "standardizer_binding": {
            "fit_manifest_sha256": _sha(
                standardizer_fit_manifest_sha256,
                "standardizer_fit_manifest_sha256",
            ),
            "train_collection_sha256": _sha(
                standardizer_train_collection_sha256,
                "standardizer_train_collection_sha256",
            ),
        },
        "selection_rule": [
            "Pd_max_subject_to_FP_over_total_pixels_le_budget",
            "FP_pixels_min_on_tie",
            "threshold_max_on_exact_tie",
        ],
        "pooled_safe_rows": pooled,
        "domain_safe_rows": by_domain,
        "standardized_source_centers_feature_indices": [0, 86],
        "standardized_source_centers": centers,
        "content_sha256_algorithm": (
            "sha256-canonical-json-stage2-source-threshold-reference-v1"
        ),
        "content_sha256": "",
    }
    projection = dict(payload)
    projection.pop("content_sha256")
    payload["content_sha256"] = canonical_json_sha256(projection)
    return payload


def t0_fixed_thresholds() -> tuple[float, float, float]:
    return (0.5, 0.5, 0.5)


def t1_pooled_source_thresholds(reference: Mapping[str, Any]) -> tuple[float, ...]:
    return _reference_thresholds(reference["pooled_safe_rows"], "pooled_safe_rows")


def t2_safer_source_thresholds(reference: Mapping[str, Any]) -> tuple[float, ...]:
    domain_rows = reference["domain_safe_rows"]
    if not isinstance(domain_rows, Mapping) or len(domain_rows) != 2:
        raise Stage2ThresholdFamilyError("T2 requires two source-domain curves")
    curves = [_reference_thresholds(rows, f"domain_safe_rows.{domain}") for domain, rows in sorted(domain_rows.items())]
    return tuple(max(left, right) for left, right in zip(*curves, strict=True))


def t3_nearest_source_thresholds(
    reference: Mapping[str, Any], standardized_context_features: Sequence[float]
) -> tuple[float, ...]:
    context = np.asarray(standardized_context_features, dtype=np.float64)
    if context.shape != (93,) or not np.isfinite(context).all():
        raise Stage2ThresholdFamilyError("T3 context features must be finite 93D")
    centers = reference["standardized_source_centers"]
    if not isinstance(centers, Mapping) or len(centers) != 2:
        raise Stage2ThresholdFamilyError("T3 requires exactly two source centers")
    distances = []
    for domain, raw in sorted(centers.items()):
        center = np.asarray(raw, dtype=np.float64)
        if center.shape != (87,) or not np.isfinite(center).all():
            raise Stage2ThresholdFamilyError("T3 source center must be finite 87D")
        distances.append((float(np.linalg.norm(context[:87] - center)), str(domain)))
    nearest = min(distances, key=lambda item: (item[0], item[1]))[1]
    return _reference_thresholds(
        reference["domain_safe_rows"][nearest], f"domain_safe_rows.{nearest}"
    )


def _reference_thresholds(rows: Any, name: str) -> tuple[float, ...]:
    if not isinstance(rows, list) or len(rows) != 3:
        raise Stage2ThresholdFamilyError(f"{name} must contain three rows")
    result = tuple(float(row["threshold"]) for row in rows)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in result):
        raise Stage2ThresholdFamilyError(f"{name} thresholds are invalid")
    return result


def t4_context_order_statistic(
    context_probability_maps: Sequence[np.ndarray],
    budget_grid: Sequence[float] = PIXEL_BUDGET_GRID,
) -> tuple[float, float, float]:
    budgets = _budget_grid(budget_grid)
    arrays = [np.asarray(value) for value in context_probability_maps]
    if len(arrays) != 14:
        raise Stage2ThresholdFamilyError("T4 requires exactly fourteen context maps")
    for index, array in enumerate(arrays):
        if array.dtype != np.float64 or array.ndim != 2 or array.size == 0:
            raise Stage2ThresholdFamilyError(
                f"T4 context map {index} must be a nonempty 2D float64 array"
            )
        if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
            raise Stage2ThresholdFamilyError(f"T4 context map {index} is invalid")
    values = np.sort(np.concatenate([array.reshape(-1) for array in arrays]))
    count = int(values.size)
    thresholds = []
    for budget in budgets:
        rank = max(0, count - math.floor(budget * count) - 1)
        thresholds.append(float(values[rank]))
    return tuple(thresholds)  # type: ignore[return-value]


def t5_evt_gpd_thresholds(
    context_probability_maps: Sequence[np.ndarray],
    budget_grid: Sequence[float] = PIXEL_BUDGET_GRID,
) -> tuple[float, float, float] | None:
    """Frozen scipy GPD MLE; return ``None`` on every registered failure."""

    budgets = _budget_grid(budget_grid)
    arrays = [np.asarray(value) for value in context_probability_maps]
    if len(arrays) != 14:
        raise Stage2ThresholdFamilyError("T5 requires exactly fourteen context maps")
    if any(array.dtype != np.float64 or array.ndim != 2 or array.size == 0 for array in arrays):
        raise Stage2ThresholdFamilyError("T5 context maps must be nonempty 2D float64")
    values = np.concatenate([array.reshape(-1) for array in arrays])
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise Stage2ThresholdFamilyError("T5 context probabilities are invalid")
    u = float(np.quantile(values, 0.95, method="higher"))
    excesses = values[values > u] - u
    if excesses.size < 50:
        return None
    try:
        from scipy.stats import genpareto

        xi, location, beta = genpareto.fit(excesses, floc=0.0, method="MLE")
    except Exception:
        return None
    xi = float(xi)
    location = float(location)
    beta = float(beta)
    if not all(math.isfinite(value) for value in (xi, location, beta)) or location != 0.0 or beta <= 0.0:
        return None
    p_u = float(excesses.size / values.size)
    thresholds: list[float] = []
    for budget in budgets:
        try:
            if abs(xi) <= 1e-12:
                raw = u + beta * math.log(p_u / budget)
            else:
                raw = u + beta / xi * ((p_u / budget) ** xi - 1.0)
        except (ArithmeticError, OverflowError, ValueError):
            return None
        if not math.isfinite(raw):
            return None
        if xi < 0.0:
            endpoint = u - beta / xi
            if not math.isfinite(endpoint) or raw > endpoint + 1e-12:
                return None
        thresholds.append(min(1.0, max(0.0, float(raw))))
    return tuple(thresholds)  # type: ignore[return-value]


def calibrator_logits_to_thresholds(
    logits: Sequence[float], *, method_id: str
) -> tuple[float, float, float]:
    if method_id not in {"T6", "T7", "T8"}:
        raise Stage2ThresholdFamilyError("checkpoint logits are only valid for T6--T8")
    raw = np.asarray(logits, dtype=np.float64)
    if raw.shape != (3,) or not np.isfinite(raw).all():
        raise Stage2ThresholdFamilyError("calibrator logits must be finite length three")
    if np.any((raw < -10.0) | (raw > 18.0)):
        raise Stage2ThresholdFamilyError(
            "calibrator logits violate the frozen [-10,18] bounds"
        )
    thresholds = 1.0 / (1.0 + np.exp(-raw))
    if method_id in {"T7", "T8"} and np.any(thresholds[1:] < thresholds[:-1]):
        raise Stage2ThresholdFamilyError(f"{method_id} checkpoint violates monotonicity")
    return tuple(float(value) for value in thresholds)  # type: ignore[return-value]


def build_prelabel_decision(
    *,
    method_id: str,
    thresholds: Sequence[float] | None,
    shared_bindings: Mapping[str, Any],
    outer_fold_id: str,
    outer_target_domain: str,
    base_seed: int,
    derived_seed: int,
    method_contract: Mapping[str, Any],
    method_binding: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if method_id not in PRELABEL_METHOD_ORDER:
        raise Stage2ThresholdFamilyError("prelabel decision method must be T0--T8")
    if method_id == "T5" and thresholds is None:
        outcome = T5_MISSING_OUTCOME
        threshold_values = None
    else:
        outcome = COMPLETE_OUTCOME
        if thresholds is None or len(thresholds) != 3:
            raise Stage2ThresholdFamilyError("complete decision requires three thresholds")
        threshold_values = [float(value) for value in thresholds]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in threshold_values):
            raise Stage2ThresholdFamilyError("thresholds must be finite in [0,1]")
    payload: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA,
        "artifact_type": DECISION_ARTIFACT_TYPE,
        "artifact_status": outcome,
        "development_only": True,
        "official_test_accessed": False,
        "query_labels_or_masks_opened": False,
        "training_performed": False,
        "gpu_used": False,
        "method_id": method_id,
        "method_name": METHOD_NAMES[method_id],
        "prelabel_eligible": True,
        "diagnostic_only": False,
        "outcome": outcome,
        "budget_grid": list(PIXEL_BUDGET_GRID),
        "thresholds": threshold_values,
        "prediction_semantics": STRICT_THRESHOLD_SEMANTICS,
        "reject_supported": False,
        "fallback_used": False,
        "outer_fold_id": _text(outer_fold_id, "outer_fold_id"),
        "outer_target_domain": _text(outer_target_domain, "outer_target_domain"),
        "base_seed": _strict_int(base_seed, "base_seed"),
        "derived_seed": _strict_int(derived_seed, "derived_seed", minimum=1),
        "shared_bindings": dict(shared_bindings),
        "method_binding": None if method_binding is None else dict(method_binding),
        "method_contract_sha256": canonical_json_sha256(dict(method_contract)),
        "decision_content_sha256_algorithm": (
            "sha256-canonical-json-stage2-threshold-decision-v1"
        ),
        "decision_content_sha256": "",
    }
    projection = dict(payload)
    projection.pop("decision_content_sha256")
    payload["decision_content_sha256"] = canonical_json_sha256(projection)
    return payload


def build_t9_postlabel_diagnostic(
    *,
    query_curve_rows: Sequence[Mapping[str, Any]],
    query_curve_sha256: str,
    outer_fold_id: str,
    outer_target_domain: str,
    base_seed: int,
    derived_seed: int,
) -> dict[str, Any]:
    """Build T9 only after labels exist; output cannot satisfy prelabel schema."""

    thresholds = [
        select_source_safe_threshold(query_curve_rows, budget)["threshold"]
        for budget in PIXEL_BUDGET_GRID
    ]
    return {
        "schema_version": T9_DIAGNOSTIC_SCHEMA,
        "artifact_type": "rc_irstd_stage2_postlabel_oracle_diagnostic",
        "artifact_status": "POSTLABEL_DIAGNOSTIC_ONLY",
        "method_id": "T9",
        "method_name": METHOD_NAMES["T9"],
        "prelabel_eligible": False,
        "diagnostic_only": True,
        "may_enter_selection_or_gate": False,
        "query_labels_opened": True,
        "official_test_accessed": False,
        "outer_fold_id": _text(outer_fold_id, "outer_fold_id"),
        "outer_target_domain": _text(outer_target_domain, "outer_target_domain"),
        "base_seed": _strict_int(base_seed, "base_seed"),
        "derived_seed": _strict_int(derived_seed, "derived_seed", minimum=1),
        "budget_grid": list(PIXEL_BUDGET_GRID),
        "thresholds": thresholds,
        "query_curve_sha256": _sha(query_curve_sha256, "query_curve_sha256"),
    }


def publish_prelabel_decision_set(
    decisions: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> tuple[Path, str]:
    """Atomically publish nine decisions, set, sidecars and commit marker.

    The final directory is installed with ``RENAME_NOREPLACE``.  The public
    verifier runs after that installation while the publisher still owns the
    same lock inode; consumers without that private proof continue to fail
    closed.  Every rollback removes only inodes created by this transaction.
    """

    if len(decisions) != 9 or [item.get("method_id") for item in decisions] != list(PRELABEL_METHOD_ORDER):
        raise Stage2ThresholdFamilyError("publish requires exact ordered T0--T8 decisions")
    first = decisions[0]
    shared_fields = (
        "budget_grid",
        "outer_fold_id",
        "outer_target_domain",
        "base_seed",
        "derived_seed",
        "shared_bindings",
    )
    for method, item in zip(PRELABEL_METHOD_ORDER, decisions, strict=True):
        if item.get("schema_version") != DECISION_SCHEMA or item.get("method_id") != method:
            raise Stage2ThresholdFamilyError(f"invalid {method} decision identity")
        if any(item.get(field) != first.get(field) for field in shared_fields):
            raise Stage2ThresholdFamilyError(f"{method} shared bindings differ")
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).expanduser()
    )
    if not root.is_absolute() or root.is_symlink() or root.resolve(strict=True) != root or not root.is_dir():
        raise Stage2ThresholdFamilyError("repository_root must be canonical")
    publication = _prepare_publication(output_dir, root)
    rename_started = False
    try:
        staging = publication.staging_root
        members: list[dict[str, str]] = []
        set_members: list[dict[str, str]] = []
        for method, payload in zip(PRELABEL_METHOD_ORDER, decisions, strict=True):
            filename = f"{method}.decision.json"
            path = staging / filename
            _write_json_exclusive(path, payload)
            digest = _sha256_file(path)
            _write_sidecar(path, digest)
            members.append({"method_id": method, "path": filename, "sha256": digest})
            set_members.append(
                {
                    "method_id": method,
                    "path": filename,
                    "sha256": digest,
                    "outcome": str(payload["outcome"]),
                    "decision_content_sha256": str(payload["decision_content_sha256"]),
                }
            )
        set_payload: dict[str, Any] = {
            "schema_version": DECISION_SET_SCHEMA,
            "artifact_type": DECISION_SET_ARTIFACT_TYPE,
            "artifact_status": "SEALED_COMPLETE_T0_T8",
            "development_only": True,
            "official_test_accessed": False,
            "query_labels_or_masks_opened": False,
            "training_performed": False,
            "gpu_used": False,
            "method_order": list(PRELABEL_METHOD_ORDER),
            "budget_grid": list(first["budget_grid"]),
            "outer_fold_id": first["outer_fold_id"],
            "outer_target_domain": first["outer_target_domain"],
            "base_seed": first["base_seed"],
            "derived_seed": first["derived_seed"],
            "shared_bindings": dict(first["shared_bindings"]),
            "decisions": set_members,
            "decision_set_content_sha256_algorithm": (
                "sha256-canonical-json-ordered-t0-t8-decision-members-v1"
            ),
            "decision_set_content_sha256": canonical_json_sha256(set_members),
        }
        set_path = staging / "decision-set.json"
        _write_json_exclusive(set_path, set_payload)
        set_sha = _sha256_file(set_path)
        _write_sidecar(set_path, set_sha)
        commit_projection = {
            "decision_set_file": set_path.name,
            "decision_set_sha256": set_sha,
            "decision_files": members,
        }
        commit = {
            "schema_version": DECISION_SET_COMMIT_SCHEMA,
            "artifact_type": DECISION_SET_COMMIT_ARTIFACT_TYPE,
            "artifact_status": "COMMITTED_COMPLETE_T0_T8",
            "development_only": True,
            "official_test_accessed": False,
            **commit_projection,
            "inventory_sha256_algorithm": (
                "sha256-canonical-json-stage2-decision-bundle-inventory-v1"
            ),
            "inventory_sha256": canonical_json_sha256(commit_projection),
        }
        commit_path = staging / "COMMIT.json"
        _write_json_exclusive(commit_path, commit)
        _write_sidecar(commit_path, _sha256_file(commit_path))
        _fsync_directory(staging)

        # Full staging verification precedes publication.  COMMIT.json and its
        # sidecar are the last files created in the private directory.
        preflight = verify_stage2_threshold_decision_set(
            set_path,
            set_sha,
            repository_root=root,
        )
        if preflight.manifest_sha256 != set_sha:
            raise RuntimeError("staged decision-set verification identity mismatch")
        _assert_owned_directory(
            publication.staging_root,
            publication.staging_identity,
            "staging decision bundle",
            descriptor=publication.staging_descriptor,
        )

        rename_started = True
        final_identity = _rename_directory_no_replace(
            publication.staging_root,
            publication.final_root,
            publication.staging_identity,
        )
        _assert_owned_directory(
            publication.final_root,
            final_identity,
            "published decision bundle",
            descriptor=publication.staging_descriptor,
        )
        _fsync_directory(publication.final_root.parent)
        verified = verify_stage2_threshold_decision_set(
            publication.final_root / "decision-set.json",
            set_sha,
            repository_root=root,
            _owned_publication_lock=(
                publication.lock_path,
                publication.lock_identity,
            ),
        )
        if verified.manifest_sha256 != set_sha:
            raise RuntimeError("published decision-set verification identity mismatch")
    except BaseException as primary:
        cleanup_errors = _cleanup_publication(
            publication,
            include_final=rename_started,
        )
        cleanup_errors.extend(_close_publication_descriptors(publication))
        if cleanup_errors:
            raise BaseExceptionGroup(
                "T0--T8 publication failed and cleanup also reported errors",
                [primary, *cleanup_errors],
            ) from primary
        raise

    finalize_errors = _cleanup_publication(publication, include_final=False)
    if finalize_errors:
        rollback_errors = _cleanup_publication(publication, include_final=True)
        rollback_errors.extend(_close_publication_descriptors(publication))
        primary = RuntimeError(
            "T0--T8 publication finalization failed; bundle was rolled back"
        )
        raise BaseExceptionGroup(
            "T0--T8 publication finalization/rollback failure",
            [primary, *finalize_errors, *rollback_errors],
        ) from primary
    close_errors = _close_publication_descriptors(publication)
    if close_errors:
        primary = RuntimeError("T0--T8 publication descriptor finalization failed")
        raise BaseExceptionGroup(
            "T0--T8 publication descriptor finalization failure",
            [primary, *close_errors],
        ) from primary
    return verified.path, verified.manifest_sha256


def _prepare_publication(value: str | Path, root: Path) -> _Publication:
    raw = Path(value).expanduser()
    if ".." in raw.parts:
        raise Stage2ThresholdFamilyError("output_dir may not contain '..'")
    final = (raw if raw.is_absolute() else root / raw).absolute()
    if final == root or root not in final.parents:
        raise Stage2ThresholdFamilyError(
            "output_dir must be a repository subdirectory"
        )
    if os.path.lexists(final):
        raise FileExistsError("output_dir already exists; overwrite is forbidden")
    parent = final.parent
    _assert_real_directory(parent, root, "output parent")
    lock = parent / f".{final.name}.lock"
    descriptor = -1
    lock_identity: tuple[int, int] | None = None
    staging: Path | None = None
    staging_identity: tuple[int, int] | None = None
    staging_descriptor = -1
    try:
        descriptor = os.open(
            lock,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        lock_info = os.fstat(descriptor)
        if not stat.S_ISREG(lock_info.st_mode):
            raise RuntimeError("new publication lock is not a regular file")
        lock_identity = (lock_info.st_dev, lock_info.st_ino)
        _write_fd_all(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        _assert_owned_regular_file(
            lock,
            lock_identity,
            "publication lock",
            descriptor=descriptor,
        )
        staging = Path(
            tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=parent)
        )
        staged_path_info = staging.stat(follow_symlinks=False)
        if not stat.S_ISDIR(staged_path_info.st_mode):
            raise RuntimeError("new staging path is not a directory")
        staging_identity = (staged_path_info.st_dev, staged_path_info.st_ino)
        staging_descriptor = os.open(
            staging,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        staging_info = os.fstat(staging_descriptor)
        if not stat.S_ISDIR(staging_info.st_mode):
            raise RuntimeError("new staging path is not a directory")
        if (staging_info.st_dev, staging_info.st_ino) != staging_identity:
            raise RuntimeError("new staging directory changed while opened")
        _assert_owned_directory(
            staging,
            staging_identity,
            "new staging directory",
            descriptor=staging_descriptor,
        )
        _fsync_directory(parent)
        return _Publication(
            final_root=final,
            staging_root=staging,
            staging_identity=staging_identity,
            staging_descriptor=staging_descriptor,
            lock_path=lock,
            lock_identity=lock_identity,
            lock_descriptor=descriptor,
        )
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        if staging is not None and staging_identity is not None:
            cleanup_errors.extend(
                _retry_cleanup(
                    lambda: _remove_owned_directory(
                        staging,
                        staging_identity,
                        staging_descriptor if staging_descriptor >= 0 else None,
                    ),
                    "remove failed staging directory",
                )
            )
        if lock_identity is not None:
            cleanup_errors.extend(
                _retry_cleanup(
                    lambda: _release_lock(lock, lock_identity, descriptor),
                    "release failed publication lock",
                )
            )
        if staging_descriptor >= 0:
            try:
                os.close(staging_descriptor)
            except BaseException as error:
                cleanup_errors.append(error)
        if descriptor >= 0:
            try:
                os.close(descriptor)
                descriptor = -1
            except BaseException as error:
                cleanup_errors.append(error)
        cleanup_errors.extend(
            _retry_cleanup(
                lambda: _fsync_directory(parent),
                "fsync output parent after failed setup",
            )
        )
        if cleanup_errors:
            raise BaseExceptionGroup(
                "T0--T8 publication setup and cleanup failed",
                [primary, *cleanup_errors],
            ) from primary
        raise


def _rename_directory_no_replace(
    source: Path,
    target: Path,
    expected_identity: tuple[int, int],
) -> tuple[int, int]:
    if os.path.lexists(target):
        raise FileExistsError("target bundle appeared before publication")
    _assert_owned_directory(source, expected_identity, "staging decision bundle")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is required")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError("target bundle already exists")
        raise OSError(error, os.strerror(error), str(target))
    return expected_identity


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(payload) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_sidecar(path: Path, digest: str) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        data = f"{digest}  {path.name}\n".encode("ascii")
        if os.write(descriptor, data) != len(data):
            raise OSError("short sidecar write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise Stage2ThresholdFamilyError("artifact member is not regular")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        if (
            before := (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"artifact changed while hashing: {path.name}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _write_fd_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise OSError("short write while creating publication lock")
        offset += written


def _assert_owned_directory(
    path: Path,
    identity: tuple[int, int],
    name: str,
    *,
    descriptor: int | None = None,
) -> None:
    if descriptor is not None:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != identity
        ):
            raise RuntimeError(f"{name} owned descriptor identity changed")
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
        raise RuntimeError(f"{name} inode/type changed")


def _assert_owned_regular_file(
    path: Path,
    identity: tuple[int, int],
    name: str,
    *,
    descriptor: int | None = None,
) -> None:
    if descriptor is not None:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != identity
        ):
            raise RuntimeError(f"{name} owned descriptor identity changed")
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != identity:
        raise RuntimeError(f"{name} inode/type changed")


def _retry_cleanup(
    action: Any,
    description: str,
    *,
    attempts: int = 3,
) -> list[BaseException]:
    last: BaseException | None = None
    for _ in range(attempts):
        try:
            action()
            return []
        except BaseException as error:
            last = error
    assert last is not None
    return [RuntimeError(f"{description}: {last}")]


def _cleanup_publication(
    publication: _Publication,
    *,
    include_final: bool,
) -> list[BaseException]:
    errors: list[BaseException] = []
    if include_final:
        errors.extend(
            _retry_cleanup(
                lambda: _remove_owned_directory(
                    publication.final_root,
                    publication.staging_identity,
                    publication.staging_descriptor,
                ),
                "remove owned final decision bundle",
            )
        )
    errors.extend(
        _retry_cleanup(
            lambda: _remove_owned_directory(
                publication.staging_root,
                publication.staging_identity,
                publication.staging_descriptor,
            ),
            "remove owned staging decision bundle",
        )
    )
    errors.extend(
        _retry_cleanup(
            lambda: _release_lock(
                publication.lock_path,
                publication.lock_identity,
                publication.lock_descriptor,
            ),
            "release owned decision publication lock",
        )
    )
    errors.extend(
        _retry_cleanup(
            lambda: _fsync_directory(publication.final_root.parent),
            "fsync decision publication parent after cleanup",
        )
    )
    return errors


def _remove_owned_directory(
    path: Path,
    identity: tuple[int, int],
    descriptor: int | None,
) -> bool:
    if not os.path.lexists(path):
        return False
    _assert_owned_directory(
        path,
        identity,
        f"cleanup directory {path}",
        descriptor=descriptor,
    )
    shutil.rmtree(path)
    return True


def _release_lock(
    path: Path,
    identity: tuple[int, int],
    descriptor: int,
) -> bool:
    if not os.path.lexists(path):
        return False
    _assert_owned_regular_file(
        path,
        identity,
        "decision publication lock",
        descriptor=descriptor,
    )
    path.unlink()
    return True


def _close_publication_descriptors(
    publication: _Publication,
) -> list[BaseException]:
    errors: list[BaseException] = []
    for descriptor, name in (
        (publication.staging_descriptor, "staging directory descriptor"),
        (publication.lock_descriptor, "publication lock descriptor"),
    ):
        try:
            os.close(descriptor)
        except BaseException as error:
            errors.append(RuntimeError(f"failed to close {name}: {error}"))
    return errors


def _assert_real_directory(path: Path, root: Path, name: str) -> None:
    if path != root and root not in path.parents:
        raise Stage2ThresholdFamilyError(f"{name} must remain under repository root")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise Stage2ThresholdFamilyError(f"{name} contains a symlink component")
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode):
        raise Stage2ThresholdFamilyError(f"{name} must be a real directory")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("fsync target is not a directory")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2ThresholdFamilyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_external_decision_list(
    value: str | Path,
    expected_sha256: str,
    root: Path,
) -> list[Mapping[str, Any]]:
    raw = Path(value).expanduser()
    if ".." in raw.parts:
        raise Stage2ThresholdFamilyError("decision-list path may not contain '..'")
    path = (raw if raw.is_absolute() else root / raw).absolute()
    if path == root or root not in path.parents or path.is_symlink():
        raise Stage2ThresholdFamilyError(
            "decision-list must be a canonical repository file"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise Stage2ThresholdFamilyError(
            "decision-list must be a canonical repository file"
        )
    _assert_real_directory(path.parent, root, "decision-list parent")
    expected = _sha(expected_sha256, "decision-list SHA-256")
    before = _sha256_file(path)
    if before != expected:
        raise Stage2ThresholdFamilyError("decision-list external SHA-256 mismatch")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise Stage2ThresholdFamilyError("decision-list is not a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Stage2ThresholdFamilyError(
            f"invalid decision-list JSON: {error}"
        ) from error
    finally:
        os.close(descriptor)
    if _sha256_file(path) != before:
        raise RuntimeError("decision-list changed while reading")
    if not isinstance(payload, list) or any(
        not isinstance(item, Mapping) for item in payload
    ):
        raise Stage2ThresholdFamilyError(
            "decision-list JSON must be an array of decision objects"
        )
    return list(payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomically publish one externally hashed T0--T8 decision list."
    )
    parser.add_argument("--decision-list", "--decisions", dest="decision_list", required=True)
    parser.add_argument(
        "--decision-list-sha256",
        "--decisions-sha256",
        dest="decision_list_sha256",
        required=True,
    )
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True)
    parser.add_argument("--repository-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = (
        Path(__file__).resolve().parents[1]
        if args.repository_root is None
        else Path(args.repository_root).expanduser()
    )
    if (
        not root.is_absolute()
        or root.is_symlink()
        or root.resolve(strict=True) != root
        or not root.is_dir()
    ):
        raise Stage2ThresholdFamilyError("repository_root must be canonical")
    decisions = _load_external_decision_list(
        args.decision_list,
        args.decision_list_sha256,
        root,
    )
    path, digest = publish_prelabel_decision_set(
        decisions,
        args.output_dir,
        repository_root=root,
    )
    print(
        json.dumps(
            {
                "decision_set_path": path.relative_to(root).as_posix(),
                "decision_set_sha256": digest,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "METHOD_NAMES",
    "SOURCE_THRESHOLD_REFERENCE_SCHEMA",
    "Stage2ThresholdFamilyError",
    "build_arg_parser",
    "build_prelabel_decision",
    "build_source_threshold_reference",
    "build_t9_postlabel_diagnostic",
    "calibrator_logits_to_thresholds",
    "make_shared_input_bindings",
    "main",
    "publish_prelabel_decision_set",
    "select_source_safe_threshold",
    "t0_fixed_thresholds",
    "t1_pooled_source_thresholds",
    "t2_safer_source_thresholds",
    "t3_nearest_source_thresholds",
    "t4_context_order_statistic",
    "t5_evt_gpd_thresholds",
]


if __name__ == "__main__":
    raise SystemExit(main())

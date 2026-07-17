"""Streaming source-only threshold references for Stage-2 T1/T2/T3.

The input must be the complete, public-verifier-produced
``source_diagnostic_validation`` collection.  Exact per-window curves remain
array backed.  A k-way event merge pools integer sufficient counts without
materialising a second union curve, and applies the frozen rank

``Pd max -> FP pixels min -> threshold max``.

No outer-target query labels or official-test artifact is accepted here.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import heapq
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data_ext.stage2_threshold_decision import PIXEL_BUDGET_GRID
from evaluation.stage2_threshold_family import (
    Stage2ThresholdFamilyError,
    build_source_threshold_reference,
)


def _column(rows: Any, field: str, dtype: Any) -> np.ndarray:
    if hasattr(rows, "column"):
        value = np.asarray(rows.column(field), dtype=dtype)
    else:
        value = np.asarray([row[field] for row in rows], dtype=dtype)
    if value.ndim != 1 or value.size < 2:
        raise Stage2ThresholdFamilyError(f"curve {field} must be nonempty 1D")
    return value


def _validated_curve_columns(rows: Any, name: str) -> dict[str, np.ndarray]:
    columns = {
        "threshold": _column(rows, "threshold", np.float64),
        "tp_objects": _column(rows, "tp_objects", np.int64),
        "gt_objects": _column(rows, "gt_objects", np.int64),
        "fp_pixels": _column(rows, "fp_pixels", np.int64),
        "total_pixels": _column(rows, "total_pixels", np.int64),
    }
    lengths = {int(value.size) for value in columns.values()}
    if len(lengths) != 1:
        raise Stage2ThresholdFamilyError(f"{name} curve columns do not align")
    threshold = columns["threshold"]
    if (
        not np.isfinite(threshold).all()
        or threshold[0] != 0.0
        or threshold[-1] != 1.0
        or not np.all(threshold[1:] > threshold[:-1])
    ):
        raise Stage2ThresholdFamilyError(
            f"{name} thresholds must be strict ascending endpoints 0 and 1"
        )
    tp = columns["tp_objects"]
    gt = columns["gt_objects"]
    fp = columns["fp_pixels"]
    pixels = columns["total_pixels"]
    if (
        np.any(tp < 0)
        or np.any(gt < 0)
        or np.any(fp < 0)
        or np.any(pixels <= 0)
        or np.any(tp > gt)
        or np.any(fp > pixels)
        or np.any(gt != gt[0])
        or np.any(pixels != pixels[0])
        or np.any(fp[1:] > fp[:-1])
    ):
        raise Stage2ThresholdFamilyError(f"{name} sufficient counts are invalid")
    return columns


def _safe_candidate(
    *, threshold: float, tp: int, gt: int, fp: int, pixels: int
) -> tuple[tuple[Fraction, int, float], dict[str, Any]]:
    pd = Fraction(tp, gt) if gt else Fraction(0, 1)
    rank = (pd, -fp, threshold)
    row = {
        "threshold": float(threshold),
        "tp_objects": int(tp),
        "gt_objects": int(gt),
        "fp_pixels": int(fp),
        "total_pixels": int(pixels),
        "pd_numerator": int(pd.numerator),
        "pd_denominator": int(pd.denominator),
    }
    return rank, row


def pool_source_validation_safe_rows(
    curves_by_domain: Mapping[str, Sequence[Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Pool exact curves in one streaming k-way merge.

    Returns the three pooled safe rows and three safe rows for each of exactly
    two source domains.  The union event sequence is never materialised.
    """

    domains = tuple(sorted(curves_by_domain))
    if len(domains) != 2 or any(not curves_by_domain[domain] for domain in domains):
        raise Stage2ThresholdFamilyError(
            "source pooling requires two nonempty source-domain curve sets"
        )
    curves: list[dict[str, np.ndarray]] = []
    curve_domains: list[str] = []
    for domain in domains:
        for index, rows in enumerate(curves_by_domain[domain]):
            curves.append(_validated_curve_columns(rows, f"{domain}[{index}]"))
            curve_domains.append(domain)

    domain_gt = {domain: 0 for domain in domains}
    domain_pixels = {domain: 0 for domain in domains}
    for domain, curve in zip(curve_domains, curves, strict=True):
        domain_gt[domain] += int(curve["gt_objects"][0])
        domain_pixels[domain] += int(curve["total_pixels"][0])
    if any(domain_gt[domain] <= 0 or domain_pixels[domain] <= 0 for domain in domains):
        raise Stage2ThresholdFamilyError("each source domain must have estimable counts")

    current_index = np.full(len(curves), -1, dtype=np.int64)
    domain_tp = {domain: 0 for domain in domains}
    domain_fp = {domain: 0 for domain in domains}
    heap: list[tuple[float, int, int]] = [
        (float(curve["threshold"][0]), index, 0)
        for index, curve in enumerate(curves)
    ]
    heapq.heapify(heap)
    best: dict[str, list[tuple[tuple[Fraction, int, float], dict[str, Any]] | None]] = {
        "__pooled__": [None, None, None],
        **{domain: [None, None, None] for domain in domains},
    }
    last_threshold = -math.inf
    while heap:
        threshold = heap[0][0]
        if not math.isfinite(threshold) or threshold <= last_threshold:
            raise Stage2ThresholdFamilyError("pooled union thresholds are not strict")
        last_threshold = threshold
        updates: list[tuple[int, int]] = []
        while heap and heap[0][0] == threshold:
            _, curve_index, row_index = heapq.heappop(heap)
            updates.append((curve_index, row_index))
        updated_domains = {curve_domains[curve_index] for curve_index, _ in updates}
        for curve_index, row_index in updates:
            curve = curves[curve_index]
            domain = curve_domains[curve_index]
            previous = int(current_index[curve_index])
            if previous >= 0:
                domain_tp[domain] -= int(curve["tp_objects"][previous])
                domain_fp[domain] -= int(curve["fp_pixels"][previous])
            current_index[curve_index] = row_index
            domain_tp[domain] += int(curve["tp_objects"][row_index])
            domain_fp[domain] += int(curve["fp_pixels"][row_index])
            next_index = row_index + 1
            if next_index < curve["threshold"].size:
                heapq.heappush(
                    heap,
                    (float(curve["threshold"][next_index]), curve_index, next_index),
                )
        if np.any(current_index < 0):
            # Every verified curve begins at zero, so only the first event can
            # initialise the state.  Anything else is a malformed input.
            raise Stage2ThresholdFamilyError("source curves do not share endpoint zero")

        pooled_tp = sum(domain_tp.values())
        pooled_gt = sum(domain_gt.values())
        pooled_fp = sum(domain_fp.values())
        pooled_pixels = sum(domain_pixels.values())
        scopes = {
            "__pooled__": (pooled_tp, pooled_gt, pooled_fp, pooled_pixels),
            **{
                domain: (
                    domain_tp[domain],
                    domain_gt[domain],
                    domain_fp[domain],
                    domain_pixels[domain],
                )
                for domain in domains
            },
        }
        for scope, (tp, gt, fp, pixels) in scopes.items():
            # Source-specific curves use only their own exact event union.
            if scope != "__pooled__" and scope not in updated_domains:
                continue
            rank, row = _safe_candidate(
                threshold=threshold, tp=tp, gt=gt, fp=fp, pixels=pixels
            )
            for budget_index, budget in enumerate(PIXEL_BUDGET_GRID):
                if Fraction(fp, pixels) > Fraction.from_float(float(budget)):
                    continue
                incumbent = best[scope][budget_index]
                if incumbent is None or rank > incumbent[0]:
                    best[scope][budget_index] = (rank, row)

    if last_threshold != 1.0:
        raise Stage2ThresholdFamilyError("pooled union lacks endpoint one")
    if any(item is None for rows in best.values() for item in rows):
        raise Stage2ThresholdFamilyError("source curves lack a feasible safe endpoint")
    pooled = [item[1] for item in best["__pooled__"] if item is not None]
    by_domain = {
        domain: [item[1] for item in best[domain] if item is not None]
        for domain in domains
    }
    return pooled, by_domain


def build_source_threshold_reference_from_verified_collection(
    collection: Any,
    standardizer: Any,
    replay_capability: Any,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build T1/T2/T3 input from one complete verified Lane-A collection."""

    # Delayed imports keep the deterministic pooling primitive independently
    # testable and avoid a schema import cycle.
    from rc.stage2_crossfit_dataset import (
        _context_vector,
        assert_stage2_context_standardizer,
        assert_stage2_trainer_replay_capability,
    )
    from rc.stage2_crossfit_schema import (
        COLLECTION_VALIDATION,
        assert_verified_episode_collection,
        canonical_json_bytes,
    )

    verified = assert_verified_episode_collection(collection)
    if verified.manifest["collection_role"] != COLLECTION_VALIDATION:
        raise Stage2ThresholdFamilyError(
            "T1/T2/T3 require source_diagnostic_validation collection"
        )
    standardizer = assert_stage2_context_standardizer(standardizer)
    replay = assert_stage2_trainer_replay_capability(
        replay_capability,
        verified,
    )
    if replay.standardizer_fit_manifest_sha256 != standardizer.fit_manifest_sha256:
        raise Stage2ThresholdFamilyError(
            "trainer replay capability/standardizer fit mismatch"
        )
    if replay.train_collection_sha256 != standardizer.train_collection_sha256:
        raise Stage2ThresholdFamilyError(
            "trainer replay capability/standardizer train collection mismatch"
        )
    fit_manifest = dict(standardizer.fit_manifest)
    observed_fit_sha = hashlib.sha256(
        canonical_json_bytes(fit_manifest) + b"\n"
    ).hexdigest()
    if observed_fit_sha != standardizer.fit_manifest_sha256:
        raise Stage2ThresholdFamilyError("standardizer fit manifest was mutated")
    try:
        manifest_mean = np.asarray(fit_manifest.get("mean"), dtype=np.float64)
        manifest_scale = np.asarray(fit_manifest.get("scale"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise Stage2ThresholdFamilyError(
            "standardizer fit manifest state is malformed"
        ) from error
    train_binding = fit_manifest.get("train_collection")
    if (
        manifest_mean.shape != (93,)
        or manifest_scale.shape != (93,)
        or not np.array_equal(standardizer.mean, manifest_mean)
        or not np.array_equal(standardizer.scale, manifest_scale)
        or not isinstance(train_binding, Mapping)
        or train_binding.get("sha256")
        != standardizer.train_collection_sha256
    ):
        raise Stage2ThresholdFamilyError(
            "standardizer transform state/fit manifest mismatch"
        )
    identities = [dict(episode.payload["detector_identity"]) for episode in verified]
    if not identities or any(identity != identities[0] for identity in identities[1:]):
        raise Stage2ThresholdFamilyError("source validation detector identity changed")
    identity = identities[0]
    if identity.get("detector_role") != "detector_full_fit" or identity.get(
        "oof_fold_index"
    ) is not None:
        raise Stage2ThresholdFamilyError("source validation must use full-fit detector")

    outer_fold = str(verified.manifest["outer_fold_id"])
    outer_target = str(verified.manifest["outer_target"])
    base_seed = int(verified.manifest["base_seed"])
    exact_identity = {
        "outer_fold_id": outer_fold,
        "outer_target": outer_target,
        "base_seed": base_seed,
    }
    for field, expected in exact_identity.items():
        if identity.get(field) != expected:
            raise Stage2ThresholdFamilyError(f"detector identity {field} mismatch")
    checkpoint_sha = str(identity.get("checkpoint_sha256"))
    derived_seed = int(identity.get("derived_seed"))

    curves_by_domain: dict[str, list[Any]] = {}
    raw_features: list[np.ndarray] = []
    episode_domains: list[str] = []
    for episode, artifact in zip(verified.episodes, verified.artifacts, strict=True):
        domain = str(episode.payload["source_domain"])
        if domain == outer_target:
            raise Stage2ThresholdFamilyError("outer target entered source reference")
        curves_by_domain.setdefault(domain, []).append(artifact.curve_rows)
        raw_features.append(_context_vector(episode))
        episode_domains.append(domain)
    pooled_rows, domain_rows = pool_source_validation_safe_rows(curves_by_domain)
    standardized = np.asarray(
        standardizer.transform(np.stack(raw_features, axis=0)), dtype=np.float64
    )
    if standardized.shape != (len(verified), 93) or not np.isfinite(standardized).all():
        raise Stage2ThresholdFamilyError("standardized source contexts are invalid")
    centers = {
        domain: standardized[
            np.asarray([value == domain for value in episode_domains], dtype=bool), :87
        ].mean(axis=0, dtype=np.float64)
        for domain in sorted(curves_by_domain)
    }

    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).expanduser()
    ).resolve(strict=True)
    for path in (verified.path, verified.commit_path):
        if path.is_symlink() or root not in path.resolve(strict=True).parents:
            raise Stage2ThresholdFamilyError("collection member is outside repository root")
    return build_source_threshold_reference(
        pooled_curve=pooled_rows,
        domain_curves=domain_rows,
        standardized_source_centers_0_86=centers,
        outer_fold_id=outer_fold,
        outer_target_domain=outer_target,
        base_seed=base_seed,
        derived_seed=derived_seed,
        detector_checkpoint_sha256=checkpoint_sha,
        collection_path=verified.path.relative_to(root).as_posix(),
        collection_sha256=verified.collection_sha256,
        collection_commit_path=verified.commit_path.relative_to(root).as_posix(),
        collection_commit_sha256=verified.commit_sha256,
        collection_identity_sha256=str(verified.manifest["ordered_record_sha256"]),
        standardizer_fit_manifest_sha256=standardizer.fit_manifest_sha256,
        standardizer_train_collection_sha256=standardizer.train_collection_sha256,
    )


__all__ = [
    "build_source_threshold_reference_from_verified_collection",
    "pool_source_validation_safe_rows",
]

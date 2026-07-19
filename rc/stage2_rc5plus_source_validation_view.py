"""Nine-budget source-only validation authority for RC5+.

The frozen RC5 validation view remains a three-budget artifact.  This module
binds one commit-last nine-budget anchor overlay per source validation domain
and evaluates all nine exact rational budgets.  Epoch ranking remains fixed
to the preregistered 1/100000 knot (index 4); the other knots are curve
evidence and cannot rescue the primary selection metric.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from model.budget_conditioned_endpoint_calibrator import (
    BUDGET_KNOT_RATIONALS,
    PRIMARY_BUDGET_KNOT_INDICES,
)
from model.budget_conditioned_residual_transport_calibrator import (
    BudgetConditionedDirectResidualTransportCalibrator,
    BudgetConditionedMonotoneNoTargetAnchorCalibrator,
    BudgetConditionedMonotoneResidualTransportCalibrator,
)
from model.endpoint_aware_threshold import (
    EndpointAwareThresholdError,
    decode_coordinate_numpy,
)
from rc.build_stage2_rc5_context import (
    VerifiedStage2RC5ContextBundle,
    replay_verified_stage2_rc5_context_bundle,
)
from rc.stage2_crossfit_schema_v6 import context_inference_material_v2
from rc.stage2_rc5_feature_mask import (
    VerifiedStage2RC5FeatureMask,
    apply_stage2_rc5_feature_mask_numpy,
    assert_verified_stage2_rc5_feature_mask,
)
from rc.stage2_rc5plus_cyclic_anchor_overlay import (
    VerifiedStage2RC5PlusCyclicAnchorOverlay,
    assert_verified_stage2_rc5plus_cyclic_anchor_overlay,
    replay_verified_stage2_rc5plus_cyclic_anchor_overlay,
)
from rc.stage2_rc5plus_context_anchor_v2 import (
    build_context_tail_anchor_v2_from_producer_bundle,
)
from rc.stage2_source_validation_views import (
    VerifiedSourceValidationCyclicSelectionView,
    VerifiedSourceVariableQuerySanityView,
    assert_verified_source_validation_cyclic_selection_view,
    assert_verified_source_variable_query_sanity_view,
)


RC5PLUS_SOURCE_VALIDATION_SCHEMA = (
    "rc-irstd.stage2-rc5plus-source-validation-cyclic-selection-view.v1"
)
RC5PLUS_SELECTION_GEOMETRY = (
    "source_validation_cyclic_selection_view_c14_q28_all_n_starts_nine_budget"
)
RC5PLUS_VARIABLE_QUERY_SANITY_SCHEMA = (
    "rc-irstd.stage2-rc5plus-source-variable-query-sanity-view.v1"
)
PRIMARY_SELECTION_BUDGET = (1, 100_000)
PRIMARY_SELECTION_INDEX = 4
SELECTION_RANK = (
    "macro_source_BSR_max",
    "macro_source_LogExcess_min",
    "macro_source_Pd_max",
    "earlier_epoch_on_exact_tie",
)
_CAPABILITY = object()
_SANITY_CAPABILITY = object()


class Stage2RC5PlusSourceValidationViewError(ValueError):
    """The base view, overlay or nine-budget ranking contract is invalid."""


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _anchor_matrix(value: Any, count: int, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float64
        or value.shape != (count, len(BUDGET_KNOT_RATIONALS))
        or not np.isfinite(value).all()
    ):
        raise TypeError(f"{name} must be finite float64[N,9]")
    try:
        decode_coordinate_numpy(value)
    except EndpointAwareThresholdError as error:
        raise Stage2RC5PlusSourceValidationViewError(
            f"{name} is not canonical EATC-v2"
        ) from error
    if np.any(value[:, 1:] < value[:, :-1]):
        raise Stage2RC5PlusSourceValidationViewError(f"{name} decreased")
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5PlusSourceValidationView:
    base_view: VerifiedSourceValidationCyclicSelectionView
    anchor_overlays: tuple[VerifiedStage2RC5PlusCyclicAnchorOverlay, ...]
    anchor_coordinates_by_domain: Mapping[str, np.ndarray]
    overlay_commit_by_domain: Mapping[str, str]
    identity_sha256: str
    artifact_scope: str
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("RC5+ source validation views are verifier-issued only")

    @property
    def source_domains(self) -> tuple[str, str]:
        return self.base_view.source_domains

    def provider_for_start(self, domain: str, start: int):
        assert_verified_stage2_rc5plus_source_validation_view(self)
        return self.base_view.provider_for_start(domain, start)


def assert_verified_stage2_rc5plus_source_validation_view(
    value: object,
) -> VerifiedStage2RC5PlusSourceValidationView:
    if (
        type(value) is not VerifiedStage2RC5PlusSourceValidationView
        or getattr(value, "_capability", None) is not _CAPABILITY
    ):
        raise TypeError("a verifier-issued RC5+ source validation view is required")
    base = assert_verified_source_validation_cyclic_selection_view(value.base_view)
    expected_overlay_count = 0 if base.artifact_scope == "synthetic_cpu_contract_test" else 2
    if len(value.anchor_overlays) != expected_overlay_count:
        raise TypeError("RC5+ source validation overlay cardinality mismatch")
    for item in value.anchor_overlays:
        assert_verified_stage2_rc5plus_cyclic_anchor_overlay(item)
    if set(value.anchor_coordinates_by_domain) != set(base.source_domains):
        raise TypeError("RC5+ source validation domain closure mismatch")
    for domain in base.source_domains:
        count = len(base.domain_materials[domain]["image_identities"])
        _anchor_matrix(
            value.anchor_coordinates_by_domain[domain], count, f"{domain}.anchors"
        )
    return value


def _issue(
    *,
    base: VerifiedSourceValidationCyclicSelectionView,
    overlays: tuple[VerifiedStage2RC5PlusCyclicAnchorOverlay, ...],
    anchors: Mapping[str, np.ndarray],
    overlay_commits: Mapping[str, str],
    identity: str,
) -> VerifiedStage2RC5PlusSourceValidationView:
    value = object.__new__(VerifiedStage2RC5PlusSourceValidationView)
    for name, item in {
        "base_view": base,
        "anchor_overlays": overlays,
        "anchor_coordinates_by_domain": MappingProxyType(dict(anchors)),
        "overlay_commit_by_domain": MappingProxyType(dict(overlay_commits)),
        "identity_sha256": identity,
        "artifact_scope": base.artifact_scope,
        "_capability": _CAPABILITY,
    }.items():
        object.__setattr__(value, name, item)
    return assert_verified_stage2_rc5plus_source_validation_view(value)


def build_synthetic_stage2_rc5plus_source_validation_view(
    *,
    base_view: VerifiedSourceValidationCyclicSelectionView,
    anchor_coordinates_by_domain: Mapping[str, np.ndarray],
) -> VerifiedStage2RC5PlusSourceValidationView:
    """Test-only view that still enforces exact primary-knot v1 replay."""

    base = assert_verified_source_validation_cyclic_selection_view(base_view)
    if base.artifact_scope != "synthetic_cpu_contract_test":
        raise Stage2RC5PlusSourceValidationViewError(
            "synthetic RC5+ builder requires a synthetic base view"
        )
    if not isinstance(anchor_coordinates_by_domain, Mapping) or set(
        anchor_coordinates_by_domain
    ) != set(base.source_domains):
        raise Stage2RC5PlusSourceValidationViewError(
            "synthetic RC5+ anchors need exactly two source domains"
        )
    anchors: dict[str, np.ndarray] = {}
    projection_rows = []
    for domain in base.source_domains:
        material = base.domain_materials[domain]
        count = len(material["image_identities"])
        matrix = _anchor_matrix(
            anchor_coordinates_by_domain[domain], count, f"{domain}.anchors"
        )
        if not np.array_equal(
            matrix[:, list(PRIMARY_BUDGET_KNOT_INDICES)],
            np.asarray(material["anchor_coordinates"]),
        ):
            raise Stage2RC5PlusSourceValidationViewError(
                "synthetic nine-budget anchors do not replay v1 primary knots"
            )
        anchors[domain] = matrix
        projection_rows.append(
            {
                "source_domain": domain,
                "anchor_coordinates_sha256": hashlib.sha256(
                    matrix.astype("<f8", copy=False).tobytes(order="C")
                ).hexdigest(),
            }
        )
    identity = _identity(
        {
            "schema_version": RC5PLUS_SOURCE_VALIDATION_SCHEMA,
            "base_view_identity_sha256": base.identity_sha256,
            "grid_budget_rationals": [list(row) for row in BUDGET_KNOT_RATIONALS],
            "domains": projection_rows,
            "artifact_scope": "synthetic_cpu_contract_test",
            "anchor_interpolation_used": False,
        }
    )
    return _issue(
        base=base,
        overlays=(),
        anchors=anchors,
        overlay_commits={},
        identity=identity,
    )


def build_stage2_rc5plus_source_validation_view(
    *,
    base_view: VerifiedSourceValidationCyclicSelectionView,
    anchor_overlays: Sequence[VerifiedStage2RC5PlusCyclicAnchorOverlay],
) -> VerifiedStage2RC5PlusSourceValidationView:
    """Bind two full-fit source-validation overlays to one frozen base view."""

    base = assert_verified_source_validation_cyclic_selection_view(base_view)
    if base.artifact_scope != "production":
        raise Stage2RC5PlusSourceValidationViewError(
            "production RC5+ builder requires a production base view"
        )
    if (
        isinstance(anchor_overlays, (str, bytes))
        or not isinstance(anchor_overlays, Sequence)
        or len(anchor_overlays) != 2
    ):
        raise Stage2RC5PlusSourceValidationViewError(
            "exactly two source-validation anchor overlays are required"
        )
    overlays = tuple(
        replay_verified_stage2_rc5plus_cyclic_anchor_overlay(
            assert_verified_stage2_rc5plus_cyclic_anchor_overlay(item)
        )
        for item in anchor_overlays
    )
    by_domain: dict[str, VerifiedStage2RC5PlusCyclicAnchorOverlay] = {}
    for item in overlays:
        manifest = item.manifest
        domain = str(manifest["source_domain"])
        if domain in by_domain:
            raise Stage2RC5PlusSourceValidationViewError(
                "duplicate source-validation anchor overlay"
            )
        if (
            domain not in base.source_domains
            or manifest["outer_fold_id"] != base.outer_fold_id
            or manifest["outer_target"] != base.outer_target
            or manifest["score_role"] != "source_diagnostic_validation"
        ):
            raise Stage2RC5PlusSourceValidationViewError(
                "anchor overlay is not the matching full-fit validation role"
            )
        by_domain[domain] = item
    if set(by_domain) != set(base.source_domains):
        raise Stage2RC5PlusSourceValidationViewError(
            "source-validation overlay coverage is incomplete"
        )
    anchors: dict[str, np.ndarray] = {}
    projection_rows = []
    for domain in base.source_domains:
        material = base.domain_materials[domain]
        item = by_domain[domain]
        expected_commit = material.get("cyclic_context_collection_sha256")
        if (
            item.base_collection.commit_sha256 != expected_commit
            or item.manifest["base_cyclic_context"]["sha256"] != expected_commit
        ):
            raise Stage2RC5PlusSourceValidationViewError(
                "validation overlay binds a different cyclic context"
            )
        if not np.array_equal(
            np.asarray(material["context_features"]),
            np.asarray(item.base_collection.context_features),
        ) or not np.array_equal(
            np.asarray(material["anchor_coordinates"]),
            np.asarray(item.base_collection.anchor_coordinates),
        ):
            raise Stage2RC5PlusSourceValidationViewError(
                "validation base features/anchors differ from overlay replay"
            )
        count = len(material["image_identities"])
        matrix = _anchor_matrix(item.anchor_coordinates, count, f"{domain}.anchors")
        anchors[domain] = matrix
        projection_rows.append(
            {
                "source_domain": domain,
                "overlay_commit_sha256": item.commit_sha256,
                "base_cyclic_context_commit_sha256": expected_commit,
            }
        )
    identity = _identity(
        {
            "schema_version": RC5PLUS_SOURCE_VALIDATION_SCHEMA,
            "base_view_identity_sha256": base.identity_sha256,
            "grid_budget_rationals": [list(row) for row in BUDGET_KNOT_RATIONALS],
            "domains": projection_rows,
            "artifact_scope": "production",
            "anchor_interpolation_used": False,
        }
    )
    return _issue(
        base=base,
        overlays=overlays,
        anchors=anchors,
        overlay_commits={domain: by_domain[domain].commit_sha256 for domain in base.source_domains},
        identity=identity,
    )


def _standardizer(mean: Any, scale: Any) -> tuple[np.ndarray, np.ndarray]:
    if (
        not isinstance(mean, np.ndarray)
        or not isinstance(scale, np.ndarray)
        or mean.dtype != np.float64
        or scale.dtype != np.float64
        or mean.shape != (93,)
        or scale.shape != (93,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale < 1e-8)
    ):
        raise Stage2RC5PlusSourceValidationViewError(
            "standardizer must be finite float64[93] with scale >=1e-8"
        )
    return mean, scale


def _predict(
    *,
    model: nn.Module,
    features: np.ndarray,
    anchors: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    feature_mask: VerifiedStage2RC5FeatureMask,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if type(model) not in {
        BudgetConditionedDirectResidualTransportCalibrator,
        BudgetConditionedMonotoneNoTargetAnchorCalibrator,
        BudgetConditionedMonotoneResidualTransportCalibrator,
    }:
        raise TypeError("RC5+ validation requires an exact residual-transport model")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive exact integer")
    mask = assert_verified_stage2_rc5_feature_mask(feature_mask)
    outputs: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, features.shape[0], batch_size):
                stop = min(start + batch_size, features.shape[0])
                standardized = (
                    (features[start:stop].astype(np.float64) - mean) / scale
                ).astype(np.float32)
                standardized = apply_stage2_rc5_feature_mask_numpy(
                    standardized, mask
                )
                model_input = torch.from_numpy(standardized).to(device=device)
                if type(model) is BudgetConditionedMonotoneNoTargetAnchorCalibrator:
                    output = model(model_input)
                else:
                    output = model(
                        model_input,
                        anchor_coordinates=torch.from_numpy(
                            np.array(
                                anchors[start:stop], dtype=np.float64, copy=True
                            )
                        ).to(device=device),
                    )
                outputs.append(output.grid_coordinates.detach().cpu().numpy())
    finally:
        if was_training:
            model.train()
    coordinates = np.concatenate(outputs).astype(np.float64, copy=False)
    if (
        coordinates.shape != (features.shape[0], len(BUDGET_KNOT_RATIONALS))
        or not np.isfinite(coordinates).all()
    ):
        raise Stage2RC5PlusSourceValidationViewError(
            "model validation output must be finite float64[N,9]"
        )
    try:
        decode_coordinate_numpy(coordinates)
    except EndpointAwareThresholdError as error:
        raise Stage2RC5PlusSourceValidationViewError(
            "model validation output is not EATC-v2"
        ) from error
    return coordinates


def _resolve_all(provider: Any, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    compact = provider.compact_brackets_v2(
        np.asarray(coordinates, dtype=np.float64)
    )
    indices = np.searchsorted(compact.coordinates, coordinates, side="right") - 1
    indices = np.clip(indices, 0, compact.coordinates.size - 1)
    return (
        np.asarray(compact.false_positive_pixels[indices], dtype=np.int64),
        np.asarray(compact.matched_objects[indices], dtype=np.int64),
        int(compact.total_native_pixels),
        int(compact.ground_truth_objects),
    )


def _selection_record(
    *, macro_bsr: float, macro_log_excess: float, macro_pd: float
) -> dict[str, Any]:
    values = (macro_bsr, macro_log_excess, macro_pd)
    if (
        any(not math.isfinite(float(value)) for value in values)
        or not 0.0 <= macro_bsr <= 1.0
        or macro_log_excess < 0.0
        or not 0.0 <= macro_pd <= 1.0
    ):
        raise Stage2RC5PlusSourceValidationViewError(
            "RC5+ primary selection metrics are invalid"
        )
    return {
        "schema_version": "rc-irstd.calibrator-source-selection-record.v3",
        "selection_geometry": RC5PLUS_SELECTION_GEOMETRY,
        "grid_budget_rationals": [
            {"numerator": row[0], "denominator": row[1]}
            for row in BUDGET_KNOT_RATIONALS
        ],
        "primary_selection_budget": {
            "numerator": PRIMARY_SELECTION_BUDGET[0],
            "denominator": PRIMARY_SELECTION_BUDGET[1],
            "grid_index": PRIMARY_SELECTION_INDEX,
        },
        "nonprimary_budgets_can_rescue_epoch_selection": False,
        "source_domain_weighting": "equal_one_half",
        "within_domain_bsr": "equal_exhaustive_cyclic_start_mean",
        "within_domain_log_excess": "equal_exhaustive_cyclic_start_mean",
        "within_domain_pd": (
            "pooled_tp_divided_by_pooled_gt_across_all_cyclic_starts"
        ),
        "source_variable_query_sanity_excluded_from_epoch_ranking": True,
        "cyclic_starts_claimed_independent": False,
        "cyclic_start_confidence_interval_reported": False,
        "rank": list(SELECTION_RANK),
        "macro_source_bsr_hex": float(macro_bsr).hex(),
        "macro_source_log_excess_hex": float(macro_log_excess).hex(),
        "macro_source_pd_hex": float(macro_pd).hex(),
        "outer_target_accessed": False,
        "official_test_accessed": False,
    }
def evaluate_stage2_rc5plus_source_validation_view(
    *,
    model: nn.Module,
    view: VerifiedStage2RC5PlusSourceValidationView,
    standardizer_mean: np.ndarray,
    standardizer_scale: np.ndarray,
    feature_mask: VerifiedStage2RC5FeatureMask,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
) -> dict[str, Any]:
    """Evaluate nine budgets but rank epochs only at exact 1/100000."""

    verified = assert_verified_stage2_rc5plus_source_validation_view(view)
    mean, scale = _standardizer(standardizer_mean, standardizer_scale)
    mask = assert_verified_stage2_rc5_feature_mask(feature_mask)
    resolved_device = torch.device(device)
    if resolved_device.type != "cpu" and verified.artifact_scope == "synthetic_cpu_contract_test":
        raise Stage2RC5PlusSourceValidationViewError(
            "synthetic validation is CPU-only"
        )
    domain_metrics: dict[str, Any] = {}
    for domain in verified.source_domains:
        material = verified.base_view.domain_materials[domain]
        features = material["context_features"]
        coordinates = _predict(
            model=model,
            features=features,
            anchors=verified.anchor_coordinates_by_domain[domain],
            mean=mean,
            scale=scale,
            feature_mask=mask,
            device=resolved_device,
            batch_size=batch_size,
        )
        satisfied = np.zeros((features.shape[0], 9), dtype=np.float64)
        log_excess = np.zeros_like(satisfied)
        pooled_tp = np.zeros(9, dtype=np.int64)
        pooled_gt = np.zeros(9, dtype=np.int64)
        for start in range(features.shape[0]):
            fp, tp, total, gt = _resolve_all(
                verified.provider_for_start(domain, start), coordinates[start]
            )
            for index, (numerator, denominator) in enumerate(BUDGET_KNOT_RATIONALS):
                satisfied[start, index] = float(
                    int(fp[index]) * denominator <= numerator * total
                )
                ratio = (int(fp[index]) * denominator) / float(numerator * total)
                log_excess[start, index] = math.log(max(ratio, 1.0))
            pooled_tp += tp
            pooled_gt += gt
        if np.any(pooled_gt <= 0):
            raise Stage2RC5PlusSourceValidationViewError(
                "cyclic source domain has zero pooled GT at a budget"
            )
        budget_rows = []
        for index, budget in enumerate(BUDGET_KNOT_RATIONALS):
            budget_rows.append(
                {
                    "budget_numerator": budget[0],
                    "budget_denominator": budget[1],
                    "BSR": float(np.mean(satisfied[:, index], dtype=np.float64)),
                    "LogExcess": float(
                        np.mean(log_excess[:, index], dtype=np.float64)
                    ),
                    "Pd": float(pooled_tp[index] / pooled_gt[index]),
                    "pooled_tp_objects": int(pooled_tp[index]),
                    "pooled_gt_objects": int(pooled_gt[index]),
                }
            )
        domain_metrics[domain] = {
            "budget_rows": budget_rows,
            "exhaustive_cyclic_start_count": int(features.shape[0]),
        }
    macro_rows = []
    for index, budget in enumerate(BUDGET_KNOT_RATIONALS):
        rows = [domain_metrics[domain]["budget_rows"][index] for domain in verified.source_domains]
        macro_rows.append(
            {
                "budget_numerator": budget[0],
                "budget_denominator": budget[1],
                "macro_source_BSR": sum(row["BSR"] for row in rows) / 2.0,
                "macro_source_LogExcess": sum(row["LogExcess"] for row in rows) / 2.0,
                "macro_source_Pd": sum(row["Pd"] for row in rows) / 2.0,
            }
        )
    primary = macro_rows[PRIMARY_SELECTION_INDEX]
    selection = _selection_record(
        macro_bsr=primary["macro_source_BSR"],
        macro_log_excess=primary["macro_source_LogExcess"],
        macro_pd=primary["macro_source_Pd"],
    )
    return {
        "schema_version": RC5PLUS_SOURCE_VALIDATION_SCHEMA,
        "selection_geometry": RC5PLUS_SELECTION_GEOMETRY,
        "view_identity_sha256": verified.identity_sha256,
        "grid_budget_rationals": [
            {"numerator": row[0], "denominator": row[1]}
            for row in BUDGET_KNOT_RATIONALS
        ],
        "primary_selection_budget": {
            "numerator": PRIMARY_SELECTION_BUDGET[0],
            "denominator": PRIMARY_SELECTION_BUDGET[1],
            "grid_index": PRIMARY_SELECTION_INDEX,
        },
        "nonprimary_budgets_can_rescue_epoch_selection": False,
        "source_domain_weighting": "equal_one_half",
        "within_domain_BSR_LogExcess": "equal_exhaustive_cyclic_start_mean",
        "within_domain_Pd": "pooled_tp_divided_by_pooled_gt_across_all_cyclic_starts",
        "feature_mask_variant": mask.variant,
        "feature_mask_identity_sha256": mask.identity_sha256,
        "domain_metrics": domain_metrics,
        "macro_budget_rows": macro_rows,
        "macro_source_BSR": primary["macro_source_BSR"],
        "macro_source_LogExcess": primary["macro_source_LogExcess"],
        "macro_source_Pd": primary["macro_source_Pd"],
        "selection_record": selection,
        "cyclic_starts_claimed_independent": False,
        "cyclic_start_confidence_interval_reported": False,
        "source_variable_query_sanity_excluded_from_epoch_ranking": True,
        "outer_target_accessed": False,
        "official_test_accessed": False,
    }


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5PlusVariableQuerySanityView:
    base_view: VerifiedSourceVariableQuerySanityView
    context_bundles: tuple[VerifiedStage2RC5ContextBundle, ...]
    anchor_coordinates: np.ndarray
    anchor_v2_identities: tuple[str, ...]
    identity_sha256: str
    artifact_scope: str
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("RC5+ variable-Q sanity views are verifier-issued only")

    @property
    def source_domains(self) -> tuple[str, str]:
        return self.base_view.source_domains

    @property
    def rows(self) -> tuple[Mapping[str, Any], ...]:
        return self.base_view.rows


def assert_verified_stage2_rc5plus_variable_query_sanity_view(
    value: object,
) -> VerifiedStage2RC5PlusVariableQuerySanityView:
    if (
        type(value) is not VerifiedStage2RC5PlusVariableQuerySanityView
        or getattr(value, "_capability", None) is not _SANITY_CAPABILITY
    ):
        raise TypeError("a verifier-issued RC5+ variable-Q sanity view is required")
    base = assert_verified_source_variable_query_sanity_view(value.base_view)
    expected_contexts = 0 if base.artifact_scope == "synthetic_cpu_contract_test" else len(base.rows)
    if len(value.context_bundles) != expected_contexts:
        raise TypeError("RC5+ variable-Q context-bundle cardinality mismatch")
    _anchor_matrix(value.anchor_coordinates, len(base.rows), "variable-Q anchors")
    if len(value.anchor_v2_identities) != len(base.rows):
        raise TypeError("RC5+ variable-Q anchor identity cardinality mismatch")
    return value


def _issue_sanity(
    *,
    base: VerifiedSourceVariableQuerySanityView,
    contexts: tuple[VerifiedStage2RC5ContextBundle, ...],
    anchors: np.ndarray,
    anchor_identities: tuple[str, ...],
    identity: str,
) -> VerifiedStage2RC5PlusVariableQuerySanityView:
    value = object.__new__(VerifiedStage2RC5PlusVariableQuerySanityView)
    for name, item in {
        "base_view": base,
        "context_bundles": contexts,
        "anchor_coordinates": anchors,
        "anchor_v2_identities": anchor_identities,
        "identity_sha256": identity,
        "artifact_scope": base.artifact_scope,
        "_capability": _SANITY_CAPABILITY,
    }.items():
        object.__setattr__(value, name, item)
    return assert_verified_stage2_rc5plus_variable_query_sanity_view(value)


def build_synthetic_stage2_rc5plus_variable_query_sanity_view(
    *,
    base_view: VerifiedSourceVariableQuerySanityView,
    anchor_coordinates: np.ndarray,
) -> VerifiedStage2RC5PlusVariableQuerySanityView:
    base = assert_verified_source_variable_query_sanity_view(base_view)
    if base.artifact_scope != "synthetic_cpu_contract_test":
        raise Stage2RC5PlusSourceValidationViewError(
            "synthetic RC5+ sanity builder requires a synthetic base view"
        )
    anchors = _anchor_matrix(anchor_coordinates, len(base.rows), "variable-Q anchors")
    old = np.stack([row["anchor_coordinates"] for row in base.rows]).astype(np.float64)
    if not np.array_equal(anchors[:, list(PRIMARY_BUDGET_KNOT_INDICES)], old):
        raise Stage2RC5PlusSourceValidationViewError(
            "synthetic variable-Q anchors do not replay v1 primary knots"
        )
    identities = tuple(
        hashlib.sha256(anchors[index].astype("<f8", copy=False).tobytes()).hexdigest()
        for index in range(len(base.rows))
    )
    identity = _identity(
        {
            "schema_version": RC5PLUS_VARIABLE_QUERY_SANITY_SCHEMA,
            "base_view_identity_sha256": base.identity_sha256,
            "grid_budget_rationals": [list(row) for row in BUDGET_KNOT_RATIONALS],
            "anchor_v2_identities": list(identities),
            "artifact_scope": "synthetic_cpu_contract_test",
            "excluded_from_epoch_ranking": True,
            "anchor_interpolation_used": False,
        }
    )
    return _issue_sanity(
        base=base,
        contexts=(),
        anchors=anchors,
        anchor_identities=identities,
        identity=identity,
    )


def build_stage2_rc5plus_variable_query_sanity_view(
    *,
    base_view: VerifiedSourceVariableQuerySanityView,
    context_bundles: Sequence[VerifiedStage2RC5ContextBundle],
) -> VerifiedStage2RC5PlusVariableQuerySanityView:
    """Rebind every all-once row to its same-map producer-derived anchor-v2."""

    base = assert_verified_source_variable_query_sanity_view(base_view)
    if base.artifact_scope != "production":
        raise Stage2RC5PlusSourceValidationViewError(
            "production RC5+ sanity builder requires a production base view"
        )
    if (
        isinstance(context_bundles, (str, bytes))
        or not isinstance(context_bundles, Sequence)
        or len(context_bundles) != len(base.rows)
    ):
        raise Stage2RC5PlusSourceValidationViewError(
            "one producer bundle is required for every variable-Q row"
        )
    contexts = tuple(
        replay_verified_stage2_rc5_context_bundle(item) for item in context_bundles
    )
    anchors = np.empty((len(base.rows), 9), dtype=np.float64)
    identities: list[str] = []
    seen_contexts: set[str] = set()
    for index, (row, bundle) in enumerate(zip(base.rows, contexts, strict=True)):
        payload = bundle.context.payload
        context_identity = str(payload["context_full_identity_sha256"])
        if context_identity in seen_contexts:
            raise Stage2RC5PlusSourceValidationViewError(
                "variable-Q producer context identity is duplicated"
            )
        seen_contexts.add(context_identity)
        material = context_inference_material_v2(bundle.context)
        if (
            payload["outer_fold_id"] != base.outer_fold_id
            or payload["source_domain"] != row["source_domain"]
            or len(payload["query_identity_records"]) != row["query_size"]
            or not np.array_equal(
                np.asarray(material.feature_values, dtype=np.float32),
                np.asarray(row["context_features"], dtype=np.float32),
            )
            or not np.array_equal(
                np.asarray(bundle.anchor.coordinates, dtype=np.float64),
                np.asarray(row["anchor_coordinates"], dtype=np.float64),
            )
        ):
            raise Stage2RC5PlusSourceValidationViewError(
                "variable-Q row differs from producer bundle replay"
            )
        anchor_v2 = build_context_tail_anchor_v2_from_producer_bundle(
            producer_bundle=bundle
        )
        anchors[index] = np.asarray(anchor_v2.grid_coordinates, dtype=np.float64)
        identities.append(str(anchor_v2.payload["anchor_identity_sha256"]))
    anchors = _anchor_matrix(anchors, len(base.rows), "variable-Q anchors")
    identity = _identity(
        {
            "schema_version": RC5PLUS_VARIABLE_QUERY_SANITY_SCHEMA,
            "base_view_identity_sha256": base.identity_sha256,
            "grid_budget_rationals": [list(row) for row in BUDGET_KNOT_RATIONALS],
            "anchor_v2_identities": identities,
            "context_bundle_commits": [item.commit_sha256 for item in contexts],
            "artifact_scope": "production",
            "excluded_from_epoch_ranking": True,
            "anchor_interpolation_used": False,
        }
    )
    return _issue_sanity(
        base=base,
        contexts=contexts,
        anchors=anchors,
        anchor_identities=tuple(identities),
        identity=identity,
    )


def evaluate_stage2_rc5plus_variable_query_sanity_view(
    *,
    model: nn.Module,
    view: VerifiedStage2RC5PlusVariableQuerySanityView,
    standardizer_mean: np.ndarray,
    standardizer_scale: np.ndarray,
    feature_mask: VerifiedStage2RC5FeatureMask,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
) -> dict[str, Any]:
    """Execute every variable-Q window once; never return a ranking record."""

    verified = assert_verified_stage2_rc5plus_variable_query_sanity_view(view)
    mean, scale = _standardizer(standardizer_mean, standardizer_scale)
    mask = assert_verified_stage2_rc5_feature_mask(feature_mask)
    resolved_device = torch.device(device)
    if resolved_device.type != "cpu" and verified.artifact_scope == "synthetic_cpu_contract_test":
        raise Stage2RC5PlusSourceValidationViewError("synthetic sanity is CPU-only")
    features = np.stack([row["context_features"] for row in verified.rows]).astype(np.float32)
    coordinates = _predict(
        model=model,
        features=features,
        anchors=verified.anchor_coordinates,
        mean=mean,
        scale=scale,
        feature_mask=mask,
        device=resolved_device,
        batch_size=batch_size,
    )
    thresholds = decode_coordinate_numpy(coordinates)
    aggregates: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(verified.rows):
        curve = row["aggregate_curve"]
        domain = str(row["source_domain"])
        aggregate = aggregates.setdefault(
            domain,
            {
                "satisfied": [[] for _ in BUDGET_KNOT_RATIONALS],
                "log_excess": [[] for _ in BUDGET_KNOT_RATIONALS],
                "tp": np.zeros(9, dtype=np.int64),
                "gt": np.zeros(9, dtype=np.int64),
                "windows": 0,
                "query_sizes": [],
            },
        )
        for index, (numerator, denominator) in enumerate(BUDGET_KNOT_RATIONALS):
            curve_index = curve.resolve_threshold(float(thresholds[row_index, index]))
            fp = int(curve.false_positive_pixels[curve_index])
            tp = int(curve.matched_objects[curve_index])
            total = int(curve.total_native_pixels)
            gt = int(curve.ground_truth_objects)
            aggregate["satisfied"][index].append(
                float(fp * denominator <= numerator * total)
            )
            ratio = (fp * denominator) / float(numerator * total)
            aggregate["log_excess"][index].append(math.log(max(ratio, 1.0)))
            aggregate["tp"][index] += tp
            aggregate["gt"][index] += gt
        aggregate["windows"] += 1
        aggregate["query_sizes"].append(int(row["query_size"]))
    domain_metrics: dict[str, Any] = {}
    for domain in verified.source_domains:
        aggregate = aggregates[domain]
        budget_rows = []
        for index, budget in enumerate(BUDGET_KNOT_RATIONALS):
            gt = int(aggregate["gt"][index])
            budget_rows.append(
                {
                    "budget_numerator": budget[0],
                    "budget_denominator": budget[1],
                    "BSR": float(np.mean(aggregate["satisfied"][index], dtype=np.float64)),
                    "LogExcess": float(
                        np.mean(aggregate["log_excess"][index], dtype=np.float64)
                    ),
                    "Pd": float(aggregate["tp"][index] / gt) if gt else 0.0,
                    "pooled_tp_objects": int(aggregate["tp"][index]),
                    "pooled_gt_objects": gt,
                }
            )
        domain_metrics[domain] = {
            "budget_rows": budget_rows,
            "window_count": int(aggregate["windows"]),
            "query_sizes": list(aggregate["query_sizes"]),
        }
    return {
        "schema_version": RC5PLUS_VARIABLE_QUERY_SANITY_SCHEMA,
        "geometry": "mandatory_variable_query_all_records_consumed_once_nine_budget",
        "view_identity_sha256": verified.identity_sha256,
        "grid_budget_rationals": [
            {"numerator": row[0], "denominator": row[1]}
            for row in BUDGET_KNOT_RATIONALS
        ],
        "excluded_from_epoch_ranking": True,
        "selection_record_present": False,
        "feature_mask_variant": mask.variant,
        "feature_mask_identity_sha256": mask.identity_sha256,
        "domain_metrics": domain_metrics,
        "window_count": len(verified.rows),
        "all_records_consumed_once": True,
        "outer_target_accessed": False,
        "official_test_accessed": False,
    }


__all__ = [
    "PRIMARY_SELECTION_BUDGET",
    "PRIMARY_SELECTION_INDEX",
    "RC5PLUS_SELECTION_GEOMETRY",
    "RC5PLUS_SOURCE_VALIDATION_SCHEMA",
    "RC5PLUS_VARIABLE_QUERY_SANITY_SCHEMA",
    "SELECTION_RANK",
    "Stage2RC5PlusSourceValidationViewError",
    "VerifiedStage2RC5PlusSourceValidationView",
    "VerifiedStage2RC5PlusVariableQuerySanityView",
    "assert_verified_stage2_rc5plus_source_validation_view",
    "assert_verified_stage2_rc5plus_variable_query_sanity_view",
    "build_stage2_rc5plus_source_validation_view",
    "build_stage2_rc5plus_variable_query_sanity_view",
    "build_synthetic_stage2_rc5plus_source_validation_view",
    "build_synthetic_stage2_rc5plus_variable_query_sanity_view",
    "evaluate_stage2_rc5plus_source_validation_view",
    "evaluate_stage2_rc5plus_variable_query_sanity_view",
]

"""Exact low-false-alarm estimability and RC5 primary-gate decisions.

The paper's false-alarm metric remains FP divided by *all* native-resolution
query pixels.  Background pixels have one narrower role: before interpreting
the 1e-5 primary operating point, every outer domain must contain enough
background mass for at least twenty expected false-positive pixels.  This
module keeps those two denominators explicit and uses integer arithmetic for
all estimability decisions.

The pre-label audit can only establish a necessary condition from declared
image geometry.  The definitive audit is post-label and uses background
counts, but it is not allowed to change thresholds, checkpoints, metrics or
bootstrap draws.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from data_ext.stage2_score_manifest import (
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
)
from data_ext.stage2_score_manifest_metadata_v5 import (
    VerifiedStage2ScoreManifestMetadataV5,
    assert_verified_stage2_score_manifest_metadata_v5,
)
from data_ext.stage2_variable_query_window import (
    VerifiedStage2VariableQueryWindow,
    assert_verified_stage2_variable_query_window,
)
from evaluation.stage2_crossed_bootstrap_v3 import (
    CROSSED_BOOTSTRAP_SCHEMA,
    DOMAIN_ORDER,
    PRIMARY_RESAMPLES,
    PROTOCOL_ID,
    validate_crossed_pair,
)


ESTIMABILITY_SCHEMA = "rc-irstd.stage2-rc5-low-fa-estimability.v1"
PRIMARY_GATE_SCHEMA = "rc-irstd.stage2-rc5-primary-gate.v1"
PRIMARY_BUDGET_NUMERATOR = 1
PRIMARY_BUDGET_DENOMINATOR = 100_000
MINIMUM_EXPECTED_BACKGROUND_FALSE_POSITIVES = 20
MINIMUM_REQUIRED_BACKGROUND_PIXELS = (
    MINIMUM_EXPECTED_BACKGROUND_FALSE_POSITIVES
    * PRIMARY_BUDGET_DENOMINATOR
    + PRIMARY_BUDGET_NUMERATOR
    - 1
) // PRIMARY_BUDGET_NUMERATOR


class Stage2RC5EstimabilityError(ValueError):
    """An estimability or primary-gate input failed closed."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Stage2RC5EstimabilityError(
            f"{name} must be an exact integer >= {minimum}"
        )
    return value


def _finite(value: Any, name: str) -> float:
    if type(value) not in {int, float}:
        raise Stage2RC5EstimabilityError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Stage2RC5EstimabilityError(f"{name} must be finite")
    return result


def _rational_report(pixel_count: int) -> dict[str, Any]:
    pixels = _strict_int(pixel_count, "pixel_count", minimum=1)
    scaled = PRIMARY_BUDGET_NUMERATOR * pixels
    required_scaled = (
        MINIMUM_EXPECTED_BACKGROUND_FALSE_POSITIVES
        * PRIMARY_BUDGET_DENOMINATOR
    )
    return {
        "pixel_count": pixels,
        "primary_budget": {
            "numerator": PRIMARY_BUDGET_NUMERATOR,
            "denominator": PRIMARY_BUDGET_DENOMINATOR,
        },
        "expected_false_positive_count_rational": {
            "numerator": scaled,
            "denominator": PRIMARY_BUDGET_DENOMINATOR,
        },
        "floor_allowed_false_positive_pixels": (
            scaled // PRIMARY_BUDGET_DENOMINATOR
        ),
        "minimum_expected_false_positive_count": (
            MINIMUM_EXPECTED_BACKGROUND_FALSE_POSITIVES
        ),
        "minimum_required_pixels": MINIMUM_REQUIRED_BACKGROUND_PIXELS,
        "comparison": (
            "primary_budget_numerator*pixel_count"
            ">=minimum_expected_count*primary_budget_denominator"
        ),
        "estimable": scaled >= required_scaled,
    }


def prelabel_total_pixel_necessary_audit(
    query_image_shapes: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Audit the label-blind necessary condition from exact HxW metadata."""

    if (
        isinstance(query_image_shapes, (str, bytes))
        or not isinstance(query_image_shapes, Sequence)
        or not query_image_shapes
    ):
        raise Stage2RC5EstimabilityError(
            "query_image_shapes must be one nonempty sequence"
        )
    total = 0
    canonical_shapes: list[list[int]] = []
    for index, raw in enumerate(query_image_shapes):
        if (
            isinstance(raw, (str, bytes))
            or not isinstance(raw, Sequence)
            or len(raw) != 2
        ):
            raise Stage2RC5EstimabilityError(
                f"query_image_shapes[{index}] must be [height,width]"
            )
        height = _strict_int(raw[0], f"query_image_shapes[{index}].height", minimum=1)
        width = _strict_int(raw[1], f"query_image_shapes[{index}].width", minimum=1)
        canonical_shapes.append([height, width])
        total += height * width
    report = _rational_report(total)
    return {
        "schema_version": ESTIMABILITY_SCHEMA,
        "audit_phase": "prelabel_total_pixel_necessary_only",
        "query_image_count": len(canonical_shapes),
        "query_image_shapes": canonical_shapes,
        "total_native_query_pixels": total,
        "background_pixels_known": False,
        "necessary_condition": report,
        "artifact_status": (
            "NECESSARY_TOTAL_PIXEL_PASS_PENDING_POSTLABEL_BACKGROUND"
            if report["estimable"]
            else "INESTIMABLE_PRIMARY_GATE_FALSE"
        ),
        "primary_fa_pixel_denominator": "all_native_resolution_query_pixels",
        "background_pixels_role": "estimability_only",
        "query_labels_accessed": False,
        "thresholds_or_checkpoints_changed": False,
    }


def prelabel_outer_manifest_estimability_audit(
    variable_query_window: VerifiedStage2VariableQueryWindow,
    score_manifest_metadata: VerifiedStage2ScoreManifestMetadataV5,
) -> dict[str, Any]:
    """Bind the necessary audit to one verified outer variable-Q manifest."""

    window = assert_verified_stage2_variable_query_window(variable_query_window)
    score = assert_verified_stage2_score_manifest_metadata_v5(
        score_manifest_metadata
    )
    if score.role != OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT:
        raise Stage2RC5EstimabilityError(
            "prelabel estimability requires the outer-target development role"
        )
    for window_field, score_field in (
        ("outer_fold_id", "outer_fold_id"),
        ("outer_target_domain", "outer_target"),
        ("domain", "source_domain"),
        ("oof_fold_index", "oof_fold_index"),
    ):
        if window.payload[window_field] != score.payload[score_field]:
            raise Stage2RC5EstimabilityError(
                f"window/score identity mismatch: {window_field}"
            )
    if len(window.ordered_records) != len(score.records):
        raise Stage2RC5EstimabilityError(
            "window/score record cardinality mismatch"
        )
    for index, (left, right) in enumerate(
        zip(window.ordered_records, score.records, strict=True)
    ):
        for field in (
            "canonical_id",
            "image_id",
            "original_image_path",
            "original_image_sha256",
            "exclusion_group_id",
            "near_duplicate_cluster_id_or_unique_sentinel",
            "source_role_record_index",
        ):
            if left[field] != right[field]:
                raise Stage2RC5EstimabilityError(
                    f"window/score order mismatch at records[{index}].{field}"
                )
    by_canonical = {
        str(record["canonical_id"]): record for record in score.records
    }
    if len(by_canonical) != len(score.records):
        raise Stage2RC5EstimabilityError("score canonical IDs are not unique")
    query_ids: list[str] = []
    for raw_window in window.windows:
        for record in raw_window["query_records"]:
            query_ids.append(str(record["canonical_id"]))
    if len(query_ids) != len(set(query_ids)):
        raise Stage2RC5EstimabilityError(
            "mandatory variable-Q queries are not globally unique"
        )
    try:
        shapes = [by_canonical[identity]["original_hw"] for identity in query_ids]
    except KeyError as error:
        raise Stage2RC5EstimabilityError(
            "window query identity is absent from score metadata"
        ) from error
    result = prelabel_total_pixel_necessary_audit(shapes)
    return {
        **result,
        "outer_fold_id": window.payload["outer_fold_id"],
        "outer_target_domain": window.payload["outer_target_domain"],
        "window_manifest_sha256": window.manifest_sha256,
        "score_manifest_sha256": score.manifest_sha256,
        "records_content_sha256": score.records_content_sha256,
        "query_full_identity_sha256_by_window": [
            raw["query_full_identity_sha256"] for raw in window.windows
        ],
    }


def postlabel_background_estimability_audit(
    crossed_pair: Mapping[str, Any],
) -> dict[str, Any]:
    """Definitive background-mass audit without changing any metric input."""

    canonical = validate_crossed_pair(crossed_pair)
    domains: list[dict[str, Any]] = []
    for domain in canonical["domains"]:
        # validate_crossed_pair already proves byte-identical query identity,
        # total pixels and background pixels across methods and all seed cells.
        windows = domain["cells"][0]["methods"]["T8"]["windows"]
        background = sum(
            int(row["background_pixels"])
            for window in windows
            for row in window["query_counts"]
        )
        total = sum(
            int(row["total_pixels"])
            for window in windows
            for row in window["query_counts"]
        )
        report = _rational_report(background)
        domains.append(
            {
                "outer_fold_id": domain["outer_fold_id"],
                "window_count": domain["window_count"],
                "total_native_query_pixels": total,
                "total_background_query_pixels": background,
                "background_estimability": report,
                "estimable": report["estimable"],
            }
        )
    all_estimable = all(row["estimable"] for row in domains)
    return {
        "schema_version": ESTIMABILITY_SCHEMA,
        "audit_phase": "postlabel_definitive_background",
        "crossed_pair_schema": CROSSED_BOOTSTRAP_SCHEMA,
        "domain_order": list(DOMAIN_ORDER),
        "domains": domains,
        "all_primary_domains_estimable": all_estimable,
        "artifact_status": (
            "PRIMARY_ESTIMABILITY_PASS"
            if all_estimable
            else "INESTIMABLE_PRIMARY_GATE_FALSE"
        ),
        "primary_fa_pixel_denominator": "all_native_resolution_query_pixels",
        "background_pixels_role": "estimability_only",
        "thresholds_or_checkpoints_changed": False,
        "imputation_used": False,
    }


def evaluate_rc5_primary_gate(
    bootstrap_result: Mapping[str, Any],
    estimability_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered T8-T4 gate; every missing predicate is false."""

    if not isinstance(bootstrap_result, Mapping):
        bootstrap_result = {}
    if not isinstance(estimability_audit, Mapping):
        estimability_audit = {}
    parse_error: str | None = None
    delta_bsr: float | None = None
    delta_pd: float | None = None
    bsr_lower: float | None = None
    pd_lower: float | None = None
    try:
        point = bootstrap_result["point_estimate"]
        interval = bootstrap_result["confidence_interval"]
        delta_bsr = _finite(point["delta_macro_bsr"], "delta_macro_bsr")
        delta_pd = _finite(point["delta_macro_pd"], "delta_macro_pd")
        bsr_interval = interval["delta_macro_bsr"]
        pd_interval = interval["delta_macro_pd"]
        if len(bsr_interval) != 2 or len(pd_interval) != 2:
            raise Stage2RC5EstimabilityError(
                "confidence intervals must have two endpoints"
            )
        bsr_lower = _finite(bsr_interval[0], "delta_macro_bsr CI lower")
        pd_lower = _finite(pd_interval[0], "delta_macro_pd CI lower")
    except (KeyError, TypeError, Stage2RC5EstimabilityError) as error:
        parse_error = str(error)
    protocol_valid = (
        bootstrap_result.get("schema_version") == CROSSED_BOOTSTRAP_SCHEMA
        and bootstrap_result.get("protocol_id") == PROTOCOL_ID
        and bootstrap_result.get("factor_mode") == "crossed"
        and bootstrap_result.get("resamples") == PRIMARY_RESAMPLES
        and bootstrap_result.get(
            "shared_window_query_draw_across_seed_slots_and_methods"
        )
        is True
        and bootstrap_result.get("method_id_in_any_factor_preimage") is False
        and bootstrap_result.get("selected_seed_in_window_query_preimage")
        is False
    )
    estimable = (
        estimability_audit.get("schema_version") == ESTIMABILITY_SCHEMA
        and estimability_audit.get("audit_phase")
        == "postlabel_definitive_background"
        and estimability_audit.get("all_primary_domains_estimable") is True
    )
    predicates = {
        "crossed_bootstrap_protocol_exact": protocol_valid and parse_error is None,
        "all_domains_background_estimable": estimable,
        "delta_macro_bsr_point_ge_0.05": (
            delta_bsr is not None and delta_bsr >= 0.05
        ),
        "delta_macro_bsr_ci_lower_gt_0": (
            bsr_lower is not None and bsr_lower > 0.0
        ),
        "delta_macro_pd_point_ge_minus_0.02": (
            delta_pd is not None and delta_pd >= -0.02
        ),
        "delta_macro_pd_ci_lower_ge_minus_0.02": (
            pd_lower is not None and pd_lower >= -0.02
        ),
    }
    passed = all(predicates.values())
    return {
        "schema_version": PRIMARY_GATE_SCHEMA,
        "comparison": "T8_minus_T4",
        "decision": "GO" if passed else "NO_GO",
        "all_predicates_pass": passed,
        "predicates": predicates,
        "observed": {
            "delta_macro_bsr_point": delta_bsr,
            "delta_macro_bsr_ci_lower": bsr_lower,
            "delta_macro_pd_point": delta_pd,
            "delta_macro_pd_ci_lower": pd_lower,
        },
        "input_error": parse_error,
        "missing_or_nonfinite_policy": "gate_false_no_imputation",
        "inestimable_primary_cell_policy": "primary_gate_false_no_imputation",
        "secondary_metric_rescue_forbidden": True,
    }


__all__ = [
    "ESTIMABILITY_SCHEMA",
    "MINIMUM_EXPECTED_BACKGROUND_FALSE_POSITIVES",
    "MINIMUM_REQUIRED_BACKGROUND_PIXELS",
    "PRIMARY_GATE_SCHEMA",
    "Stage2RC5EstimabilityError",
    "evaluate_rc5_primary_gate",
    "postlabel_background_estimability_audit",
    "prelabel_outer_manifest_estimability_audit",
    "prelabel_total_pixel_necessary_audit",
]

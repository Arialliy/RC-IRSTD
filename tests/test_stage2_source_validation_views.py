from __future__ import annotations

import hashlib

import numpy as np

from model.endpoint_aware_pixel_calibrator import DirectEndpointAwarePixelCalibrator
from model.endpoint_aware_threshold import encode_probability_numpy
from rc.stage2_compositional_curve_provider import build_per_image_exact_event_curve
from rc.stage2_source_validation_views import (
    CYCLIC_SELECTION_GEOMETRY,
    build_synthetic_source_validation_cyclic_selection_view,
    build_synthetic_source_variable_query_sanity_view,
    evaluate_source_validation_cyclic_selection_view,
    evaluate_source_variable_query_sanity_view,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _curve(identity: str, *, total: int = 100_000):
    return build_per_image_exact_event_curve(
        image_identity_sha256=identity,
        thresholds=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        false_positive_pixels=np.asarray([10, 0, 0], dtype=np.int64),
        matched_objects=np.asarray([1, 1, 0], dtype=np.int64),
        total_native_pixels=total,
        ground_truth_objects=1,
    )


def _selection_view():
    anchor = encode_probability_numpy(np.asarray([0.2, 0.5, 0.8], dtype=np.float64))
    materials = {}
    for domain in ("NUDT-SIRST", "IRSTD-1K"):
        identities = tuple(_sha(f"val-{domain}-{index}") for index in range(42))
        materials[domain] = {
            "image_identities": identities,
            "context_features": np.zeros((42, 93), dtype=np.float32),
            "anchor_coordinates": np.repeat(anchor[None, :], 42, axis=0),
            "per_image_curves": tuple(_curve(identity) for identity in identities),
        }
    return build_synthetic_source_validation_cyclic_selection_view(
        outer_fold_id="outer_leave_nuaa_sirst", domain_materials=materials)


def _model():
    return DirectEndpointAwarePixelCalibrator(
        context_feature_dim=93,
        pixel_budget_grid=(1e-4, 1e-5, 1e-6),
        hidden_dims=(32,),
        dropout=0.1,
    )


def test_exhaustive_cyclic_selection_has_v2_rank_semantics() -> None:
    result = evaluate_source_validation_cyclic_selection_view(
        model=_model(), view=_selection_view(),
        standardizer_mean=np.zeros(93, dtype=np.float64),
        standardizer_scale=np.ones(93, dtype=np.float64),
    )
    assert result["selection_geometry"] == CYCLIC_SELECTION_GEOMETRY
    assert result["cyclic_starts_claimed_independent"] is False
    assert result["cyclic_start_confidence_interval_reported"] is False
    assert result["source_variable_query_sanity_excluded_from_epoch_ranking"] is True
    assert {row["exhaustive_cyclic_start_count"]
            for row in result["domain_metrics"].values()} == {42}
    record = result["selection_record"]
    assert record["schema_version"].endswith(".v2")
    assert record["selection_geometry"] == CYCLIC_SELECTION_GEOMETRY


def test_variable_query_sanity_executes_but_is_never_ranked() -> None:
    anchor = encode_probability_numpy(np.asarray([0.2, 0.5, 0.8], dtype=np.float64))
    rows = []
    for index, domain in enumerate(("NUDT-SIRST", "IRSTD-1K")):
        rows.append({
            "source_domain": domain,
            "query_size": 43 + index,
            "context_features": np.zeros(93, dtype=np.float32),
            "anchor_coordinates": anchor,
            "aggregate_curve": _curve(_sha(f"sanity-{domain}"), total=4_300_000),
        })
    view = build_synthetic_source_variable_query_sanity_view(
        outer_fold_id="outer_leave_nuaa_sirst", rows=rows)
    result = evaluate_source_variable_query_sanity_view(
        model=_model(), view=view,
        standardizer_mean=np.zeros(93, dtype=np.float64),
        standardizer_scale=np.ones(93, dtype=np.float64),
    )
    assert result["geometry"] == "mandatory_variable_query_all_records_consumed_once"
    assert result["excluded_from_epoch_ranking"] is True
    assert "selection_record" not in result

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from model.budget_conditioned_residual_transport_calibrator import (
    BudgetConditionedDirectResidualTransportCalibrator,
    BudgetConditionedMonotoneNoTargetAnchorCalibrator,
    BudgetConditionedMonotoneResidualTransportCalibrator,
)
from model.endpoint_aware_pixel_calibrator import DirectEndpointAwarePixelCalibrator
from model.endpoint_aware_threshold import encode_probability_numpy
from rc.stage2_compositional_curve_provider import build_per_image_exact_event_curve
from rc.stage2_rc5_feature_mask import build_stage2_rc5_feature_mask
from rc.stage2_rc5plus_source_validation_view import (
    PRIMARY_SELECTION_BUDGET,
    PRIMARY_SELECTION_INDEX,
    RC5PLUS_SELECTION_GEOMETRY,
    Stage2RC5PlusSourceValidationViewError,
    _predict,
    build_synthetic_stage2_rc5plus_source_validation_view,
    build_synthetic_stage2_rc5plus_variable_query_sanity_view,
    evaluate_stage2_rc5plus_source_validation_view,
    evaluate_stage2_rc5plus_variable_query_sanity_view,
)
from rc.stage2_source_validation_views import (
    build_synthetic_source_validation_cyclic_selection_view,
    build_synthetic_source_variable_query_sanity_view,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _curve(identity: str):
    return build_per_image_exact_event_curve(
        image_identity_sha256=identity,
        thresholds=np.asarray([0.0, 0.35, 0.55, 0.75, 1.0], dtype=np.float64),
        false_positive_pixels=np.asarray([1000, 100, 10, 1, 0], dtype=np.int64),
        matched_objects=np.asarray([1, 1, 1, 1, 0], dtype=np.int64),
        total_native_pixels=1_000_000,
        ground_truth_objects=1,
    )


def _views():
    probabilities = np.asarray(
        [0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8],
        dtype=np.float64,
    )
    nine = encode_probability_numpy(probabilities)
    primary = nine[[0, 4, 8]]
    materials = {}
    anchors = {}
    for domain_index, domain in enumerate(("NUDT-SIRST", "IRSTD-1K")):
        identities = tuple(_sha(f"rc5plus-val-{domain}-{index}") for index in range(42))
        features = np.zeros((42, 93), dtype=np.float32)
        features[:, 92] = np.float32(1.0 + domain_index)
        materials[domain] = {
            "image_identities": identities,
            "context_features": features,
            "anchor_coordinates": np.repeat(primary[None, :], 42, axis=0),
            "per_image_curves": tuple(_curve(identity) for identity in identities),
        }
        anchors[domain] = np.repeat(nine[None, :], 42, axis=0)
    base = build_synthetic_source_validation_cyclic_selection_view(
        outer_fold_id="outer_leave_nuaa_sirst",
        domain_materials=materials,
    )
    return base, build_synthetic_stage2_rc5plus_source_validation_view(
        base_view=base,
        anchor_coordinates_by_domain=anchors,
    )


def _model(monotone: bool = True):
    kwargs = {
        "context_feature_dim": 93,
        "hidden_dims": (32,),
        "dropout": 0.1,
        "minimum_residual_increment": 1e-6,
    }
    return (
        BudgetConditionedMonotoneResidualTransportCalibrator(**kwargs)
        if monotone
        else BudgetConditionedDirectResidualTransportCalibrator(**kwargs)
    )


def _sanity_view():
    probabilities = np.asarray(
        [0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8],
        dtype=np.float64,
    )
    nine = encode_probability_numpy(probabilities)
    rows = []
    for index, domain in enumerate(("NUDT-SIRST", "IRSTD-1K")):
        feature = np.zeros(93, dtype=np.float32)
        feature[92] = np.float32(index + 1)
        rows.append(
            {
                "source_domain": domain,
                "query_size": 43 + index,
                "context_features": feature,
                "anchor_coordinates": nine[[0, 4, 8]],
                "aggregate_curve": _curve(_sha(f"rc5plus-sanity-{domain}")),
            }
        )
    base = build_synthetic_source_variable_query_sanity_view(
        outer_fold_id="outer_leave_nuaa_sirst", rows=rows
    )
    anchors = np.repeat(nine[None, :], len(rows), axis=0)
    return build_synthetic_stage2_rc5plus_variable_query_sanity_view(
        base_view=base, anchor_coordinates=anchors
    )


@pytest.mark.parametrize("monotone", [False, True])
def test_nine_budget_validation_reports_curve_but_ranks_only_primary(monotone) -> None:
    _, view = _views()
    mask = build_stage2_rc5_feature_mask("C3")
    result = evaluate_stage2_rc5plus_source_validation_view(
        model=_model(monotone),
        view=view,
        standardizer_mean=np.zeros(93, dtype=np.float64),
        standardizer_scale=np.ones(93, dtype=np.float64),
        feature_mask=mask,
        batch_size=13,
    )

    assert result["selection_geometry"] == RC5PLUS_SELECTION_GEOMETRY
    assert result["primary_selection_budget"] == {
        "numerator": PRIMARY_SELECTION_BUDGET[0],
        "denominator": PRIMARY_SELECTION_BUDGET[1],
        "grid_index": PRIMARY_SELECTION_INDEX,
    }
    assert result["nonprimary_budgets_can_rescue_epoch_selection"] is False
    assert len(result["macro_budget_rows"]) == len(BUDGET_KNOT_RATIONALS)
    primary = result["macro_budget_rows"][PRIMARY_SELECTION_INDEX]
    assert result["macro_source_BSR"] == primary["macro_source_BSR"]
    assert result["macro_source_LogExcess"] == primary["macro_source_LogExcess"]
    assert result["macro_source_Pd"] == primary["macro_source_Pd"]
    record = result["selection_record"]
    assert record["schema_version"].endswith(".v3")
    assert record["selection_geometry"] == RC5PLUS_SELECTION_GEOMETRY
    assert float.fromhex(record["macro_source_bsr_hex"]) == primary[
        "macro_source_BSR"
    ]
    assert record["primary_selection_budget"] == result[
        "primary_selection_budget"
    ]
    assert record["nonprimary_budgets_can_rescue_epoch_selection"] is False
    assert result["cyclic_starts_claimed_independent"] is False
    assert result["cyclic_start_confidence_interval_reported"] is False
    assert result["outer_target_accessed"] is False
    assert result["official_test_accessed"] is False
    for domain in view.source_domains:
        rows = result["domain_metrics"][domain]["budget_rows"]
        assert len(rows) == 9
        assert result["domain_metrics"][domain]["exhaustive_cyclic_start_count"] == 42
        assert all(0.0 <= row["BSR"] <= 1.0 for row in rows)
        assert all(row["pooled_gt_objects"] == 42 * 28 for row in rows)


def test_synthetic_view_rejects_nine_budget_anchor_with_changed_primary_knot() -> None:
    base, view = _views()
    changed = {
        domain: np.array(view.anchor_coordinates_by_domain[domain], copy=True)
        for domain in view.source_domains
    }
    changed[view.source_domains[0]][0, 4] += 0.01
    with pytest.raises(Stage2RC5PlusSourceValidationViewError, match="primary knots"):
        build_synthetic_stage2_rc5plus_source_validation_view(
            base_view=base,
            anchor_coordinates_by_domain=changed,
        )


def test_validation_applies_feature_mask_after_standardization() -> None:
    _, view = _views()
    domain = view.source_domains[0]
    features = view.base_view.domain_materials[domain]["context_features"][:2]
    anchors = view.anchor_coordinates_by_domain[domain][:2]
    model = _model(monotone=False)
    with torch.no_grad():
        model.encoder[0].weight.zero_()
        model.encoder[0].bias.zero_()
        model.encoder[0].weight[0, 92] = 1.0
        model.transport_head.weight.zero_()
        model.transport_head.bias.zero_()
        model.transport_head.weight[0, 0] = 10.0
    c3 = _predict(
        model=model,
        features=features,
        anchors=anchors,
        mean=np.zeros(93, dtype=np.float64),
        scale=np.ones(93, dtype=np.float64),
        feature_mask=build_stage2_rc5_feature_mask("C3"),
        device=torch.device("cpu"),
        batch_size=2,
    )
    c4 = _predict(
        model=model,
        features=features,
        anchors=anchors,
        mean=np.zeros(93, dtype=np.float64),
        scale=np.ones(93, dtype=np.float64),
        feature_mask=build_stage2_rc5_feature_mask("C4"),
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert not np.array_equal(c3, c4)


def test_no_anchor_validation_is_bitwise_invariant_to_anchor_values() -> None:
    _, view = _views()
    domain = view.source_domains[0]
    features = view.base_view.domain_materials[domain]["context_features"][:2]
    anchors = view.anchor_coordinates_by_domain[domain][:2]
    changed = np.array(anchors, copy=True)
    changed[:, :] = encode_probability_numpy(
        np.tile(np.linspace(0.01, 0.99, 9), (2, 1))
    )
    model = BudgetConditionedMonotoneNoTargetAnchorCalibrator(93, dropout=0.0)
    kwargs = {
        "model": model,
        "features": features,
        "mean": np.zeros(93, dtype=np.float64),
        "scale": np.ones(93, dtype=np.float64),
        "feature_mask": build_stage2_rc5_feature_mask("C3"),
        "device": torch.device("cpu"),
        "batch_size": 2,
    }
    first = _predict(anchors=anchors, **kwargs)
    second = _predict(anchors=changed, **kwargs)
    assert np.array_equal(first, second)


def test_validation_rejects_old_three_budget_model_and_bare_mask() -> None:
    _, view = _views()
    old = DirectEndpointAwarePixelCalibrator(
        context_feature_dim=93,
        pixel_budget_grid=(1e-4, 1e-5, 1e-6),
        hidden_dims=(32,),
        dropout=0.1,
    )
    with pytest.raises(TypeError, match="residual-transport"):
        evaluate_stage2_rc5plus_source_validation_view(
            model=old,
            view=view,
            standardizer_mean=np.zeros(93, dtype=np.float64),
            standardizer_scale=np.ones(93, dtype=np.float64),
            feature_mask=build_stage2_rc5_feature_mask("C3"),
        )
    with pytest.raises(TypeError, match="verifier-issued"):
        evaluate_stage2_rc5plus_source_validation_view(
            model=_model(),
            view=view,
            standardizer_mean=np.zeros(93, dtype=np.float64),
            standardizer_scale=np.ones(93, dtype=np.float64),
            feature_mask={"variant": "C3"},  # type: ignore[arg-type]
        )


def test_variable_query_sanity_runs_all_windows_and_never_ranks() -> None:
    view = _sanity_view()
    result = evaluate_stage2_rc5plus_variable_query_sanity_view(
        model=_model(),
        view=view,
        standardizer_mean=np.zeros(93, dtype=np.float64),
        standardizer_scale=np.ones(93, dtype=np.float64),
        feature_mask=build_stage2_rc5_feature_mask("C3"),
    )

    assert result["geometry"] == (
        "mandatory_variable_query_all_records_consumed_once_nine_budget"
    )
    assert result["excluded_from_epoch_ranking"] is True
    assert result["selection_record_present"] is False
    assert "selection_record" not in result
    assert result["all_records_consumed_once"] is True
    assert result["window_count"] == 2
    assert result["outer_target_accessed"] is False
    assert result["official_test_accessed"] is False
    assert result["domain_metrics"]["NUDT-SIRST"]["query_sizes"] == [43]
    assert result["domain_metrics"]["IRSTD-1K"]["query_sizes"] == [44]
    assert all(
        len(result["domain_metrics"][domain]["budget_rows"]) == 9
        for domain in view.source_domains
    )


def test_variable_query_sanity_rejects_changed_primary_anchor() -> None:
    view = _sanity_view()
    changed = np.array(view.anchor_coordinates, copy=True)
    changed[0, 4] += 0.01
    with pytest.raises(Stage2RC5PlusSourceValidationViewError, match="primary knots"):
        build_synthetic_stage2_rc5plus_variable_query_sanity_view(
            base_view=view.base_view,
            anchor_coordinates=changed,
        )

from __future__ import annotations

from copy import deepcopy

import pytest

from evaluation.stage2_crossed_bootstrap_v3 import (
    CROSSED_BOOTSTRAP_SCHEMA,
    PRIMARY_RESAMPLES,
    PROTOCOL_ID,
)
from evaluation.stage2_rc5_estimability import (
    MINIMUM_REQUIRED_BACKGROUND_PIXELS,
    Stage2RC5EstimabilityError,
    evaluate_rc5_primary_gate,
    postlabel_background_estimability_audit,
    prelabel_total_pixel_necessary_audit,
)
from tests.test_stage2_crossed_bootstrap_v3 import _pair


def _bootstrap_result(
    *,
    delta_bsr: float = 0.06,
    bsr_lower: float = 0.01,
    delta_pd: float = -0.01,
    pd_lower: float = -0.019,
) -> dict:
    return {
        "schema_version": CROSSED_BOOTSTRAP_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "factor_mode": "crossed",
        "resamples": PRIMARY_RESAMPLES,
        "shared_window_query_draw_across_seed_slots_and_methods": True,
        "method_id_in_any_factor_preimage": False,
        "selected_seed_in_window_query_preimage": False,
        "point_estimate": {
            "delta_macro_bsr": delta_bsr,
            "delta_macro_pd": delta_pd,
        },
        "confidence_interval": {
            "delta_macro_bsr": [bsr_lower, 0.2],
            "delta_macro_pd": [pd_lower, 0.1],
        },
    }


def test_prelabel_total_pixel_gate_uses_exact_integer_boundary() -> None:
    fail = prelabel_total_pixel_necessary_audit(
        [[1, MINIMUM_REQUIRED_BACKGROUND_PIXELS - 1]]
    )
    passed = prelabel_total_pixel_necessary_audit(
        [[1, MINIMUM_REQUIRED_BACKGROUND_PIXELS]]
    )
    assert fail["necessary_condition"]["estimable"] is False
    assert fail["artifact_status"] == "INESTIMABLE_PRIMARY_GATE_FALSE"
    assert passed["necessary_condition"]["estimable"] is True
    assert (
        passed["necessary_condition"]["floor_allowed_false_positive_pixels"]
        == 20
    )
    assert passed["query_labels_accessed"] is False


@pytest.mark.parametrize(
    "bad",
    [[], [[0, 2]], [[2, True]], [[2]], "2x2"],
)
def test_prelabel_shape_contract_fails_closed(bad) -> None:
    with pytest.raises((Stage2RC5EstimabilityError, TypeError)):
        prelabel_total_pixel_necessary_audit(bad)


def test_postlabel_background_audit_passes_and_uses_background_only_for_gate() -> None:
    report = postlabel_background_estimability_audit(_pair())
    assert report["all_primary_domains_estimable"] is True
    assert report["primary_fa_pixel_denominator"] == (
        "all_native_resolution_query_pixels"
    )
    assert report["background_pixels_role"] == "estimability_only"
    assert all(
        row["total_background_query_pixels"] >= MINIMUM_REQUIRED_BACKGROUND_PIXELS
        for row in report["domains"]
    )


def test_one_inestimable_domain_forces_primary_no_go_without_imputation() -> None:
    pair = deepcopy(_pair())
    for cell in pair["domains"][0]["cells"]:
        for method in ("T8", "T4"):
            for window in cell["methods"][method]["windows"]:
                for row in window["query_counts"]:
                    row["background_pixels"] = 100
                    row["false_positive_pixels"] = min(
                        row["false_positive_pixels"], 100
                    )
    estimability = postlabel_background_estimability_audit(pair)
    assert estimability["all_primary_domains_estimable"] is False
    gate = evaluate_rc5_primary_gate(_bootstrap_result(), estimability)
    assert gate["decision"] == "NO_GO"
    assert gate["predicates"]["all_domains_background_estimable"] is False
    assert gate["inestimable_primary_cell_policy"].endswith("no_imputation")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delta_bsr", 0.049999),
        ("bsr_lower", 0.0),
        ("delta_pd", -0.020001),
        ("pd_lower", -0.020001),
    ],
)
def test_every_preregistered_gate_inequality_is_independently_mandatory(
    field: str, value: float
) -> None:
    estimability = postlabel_background_estimability_audit(_pair())
    kwargs = {field: value}
    gate = evaluate_rc5_primary_gate(_bootstrap_result(**kwargs), estimability)
    assert gate["decision"] == "NO_GO"
    assert gate["all_predicates_pass"] is False


def test_exact_boundary_semantics_match_preregistration() -> None:
    estimability = postlabel_background_estimability_audit(_pair())
    # BSR point, Pd point and Pd lower are inclusive; BSR lower is strict.
    inclusive = evaluate_rc5_primary_gate(
        _bootstrap_result(
            delta_bsr=0.05,
            bsr_lower=float.fromhex("0x0.0000000000001p-1022"),
            delta_pd=-0.02,
            pd_lower=-0.02,
        ),
        estimability,
    )
    assert inclusive["decision"] == "GO"
    strict_fail = evaluate_rc5_primary_gate(
        _bootstrap_result(bsr_lower=0.0), estimability
    )
    assert strict_fail["decision"] == "NO_GO"


def test_protocol_drift_and_nonfinite_never_pass() -> None:
    estimability = postlabel_background_estimability_audit(_pair())
    drift = _bootstrap_result()
    drift["resamples"] = PRIMARY_RESAMPLES - 1
    assert evaluate_rc5_primary_gate(drift, estimability)["decision"] == "NO_GO"
    nonfinite = _bootstrap_result(delta_bsr=float("nan"))
    result = evaluate_rc5_primary_gate(nonfinite, estimability)
    assert result["decision"] == "NO_GO"
    assert result["input_error"] is not None

    missing = _bootstrap_result()
    del missing["point_estimate"]["delta_macro_bsr"]
    result = evaluate_rc5_primary_gate(missing, estimability)
    assert result["decision"] == "NO_GO"
    assert result["input_error"] is not None

from __future__ import annotations

import numpy as np
import pytest

from model.endpoint_aware_threshold import encode_probability_numpy
from rc.stage2_context_tail_anchor import BUDGET_RATIONALS
from rc.stage2_exact_oracle_v2 import (
    BUDGET_COUNT_RULE,
    ORACLE_SELECTION_RULE,
    ORACLE_SELECTION_SCHEMA,
    Stage2ExactOracleError,
    select_exact_oracle_v2,
)


def _curve() -> dict[str, object]:
    return {
        "thresholds": np.asarray(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64
        ),
        "false_positive_pixels": np.asarray(
            [500, 100, 10, 1, 0, 0], dtype=np.int64
        ),
        "matched_objects": np.asarray(
            [0, 1, 3, 2, 1, 0], dtype=np.int64
        ),
        "total_native_pixels": 1_000_000,
        "ground_truth_objects": 3,
    }


def test_exact_oracle_uses_nested_integer_feasibility_and_eatc() -> None:
    result = select_exact_oracle_v2(**_curve())

    assert ORACLE_SELECTION_SCHEMA.endswith(".v2")
    assert "max-tp-min-fp-max-threshold" in ORACLE_SELECTION_RULE
    assert BUDGET_COUNT_RULE == "(numerator * total_native_pixels) // denominator"
    assert result.allowed_false_positive_counts == (100, 10, 1)
    assert result.selected_indices == (2, 2, 3)
    assert result.thresholds.tolist() == [0.4, 0.4, 0.6]
    np.testing.assert_array_equal(
        result.coordinates, encode_probability_numpy(result.thresholds)
    )
    assert result.false_positive_pixels.tolist() == [10, 10, 1]
    assert result.matched_objects.tolist() == [3, 3, 2]
    np.testing.assert_allclose(
        result.pixel_false_alarm_rate, [1e-5, 1e-5, 1e-6], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        result.detection_probability, [1.0, 1.0, 2.0 / 3.0]
    )
    assert np.all(np.diff(result.thresholds) >= 0.0)


def test_exact_rational_count_differs_from_binary64_budget_counterexample() -> None:
    total = 1_000_000_000_000_000
    result = select_exact_oracle_v2(
        thresholds=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        false_positive_pixels=np.asarray(
            [total, 999_999_999, 0], dtype=np.int64
        ),
        matched_objects=np.asarray([0, 1, 0], dtype=np.int64),
        total_native_pixels=total,
        ground_truth_objects=1,
    )

    exact = total // 1_000_000
    # Exact integer ratio of the binary64 value commonly written as 1e-6.
    binary64_numerator = 4_722_366_482_869_645
    binary64_denominator = 4_722_366_482_869_645_213_696
    legacy = (binary64_numerator * total) // binary64_denominator
    assert exact == 1_000_000_000
    assert legacy == 999_999_999
    assert exact != legacy
    assert result.allowed_false_positive_counts[-1] == exact
    assert result.selected_indices[-1] == 1


def test_tie_break_is_min_fp_then_highest_threshold() -> None:
    payload = _curve()
    payload["matched_objects"] = np.asarray(
        [0, 1, 3, 3, 3, 0], dtype=np.int64
    )
    result = select_exact_oracle_v2(**payload)

    # Loose/middle budgets can attain TP=3 at multiple rows.  FP=0 wins,
    # then the largest threshold sustaining TP=3 wins.
    assert result.selected_indices[:2] == (4, 4)
    assert result.selected_indices[-1] == 4


def test_target_free_episode_selects_exact_upper_endpoint() -> None:
    payload = _curve()
    payload["matched_objects"] = np.zeros(6, dtype=np.int64)
    payload["ground_truth_objects"] = 0
    result = select_exact_oracle_v2(**payload)

    assert result.selected_indices == (5, 5, 5)
    assert result.thresholds.tolist() == [1.0, 1.0, 1.0]
    assert result.detection_probability.tolist() == [0.0, 0.0, 0.0]


def test_selection_owns_readonly_snapshots() -> None:
    payload = _curve()
    thresholds = payload["thresholds"]
    assert isinstance(thresholds, np.ndarray)
    result = select_exact_oracle_v2(**payload)
    thresholds[2] = 0.45

    assert result.thresholds.tolist() == [0.4, 0.4, 0.6]
    for array in (
        result.thresholds,
        result.coordinates,
        result.matched_objects,
        result.false_positive_pixels,
        result.detection_probability,
        result.pixel_false_alarm_rate,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array[0] = 0


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("threshold_dtype", Stage2ExactOracleError),
        ("threshold_endpoint", Stage2ExactOracleError),
        ("threshold_duplicate", Stage2ExactOracleError),
        ("fp_dtype", Stage2ExactOracleError),
        ("fp_increase", Stage2ExactOracleError),
        ("fp_too_large", Stage2ExactOracleError),
        ("tp_too_large", Stage2ExactOracleError),
        ("endpoint_predicts", Stage2ExactOracleError),
        ("bool_total", Stage2ExactOracleError),
        ("wrong_budgets", Stage2ExactOracleError),
    ],
)
def test_exact_oracle_fails_closed_on_curve_or_budget_drift(
    mutation: str, error: type[Exception]
) -> None:
    payload = _curve()
    if mutation == "threshold_dtype":
        payload["thresholds"] = payload["thresholds"].astype(np.float32)
    elif mutation == "threshold_endpoint":
        payload["thresholds"] = payload["thresholds"].copy()
        payload["thresholds"][-1] = 0.99
    elif mutation == "threshold_duplicate":
        payload["thresholds"] = payload["thresholds"].copy()
        payload["thresholds"][2] = payload["thresholds"][1]
    elif mutation == "fp_dtype":
        payload["false_positive_pixels"] = payload[
            "false_positive_pixels"
        ].astype(np.float64)
    elif mutation == "fp_increase":
        payload["false_positive_pixels"] = payload[
            "false_positive_pixels"
        ].copy()
        payload["false_positive_pixels"][3] = 11
    elif mutation == "fp_too_large":
        payload["false_positive_pixels"] = payload[
            "false_positive_pixels"
        ].copy()
        payload["false_positive_pixels"][0] = 1_000_001
    elif mutation == "tp_too_large":
        payload["matched_objects"] = payload["matched_objects"].copy()
        payload["matched_objects"][2] = 4
    elif mutation == "endpoint_predicts":
        payload["matched_objects"] = payload["matched_objects"].copy()
        payload["matched_objects"][-1] = 1
    elif mutation == "bool_total":
        payload["total_native_pixels"] = True
    elif mutation == "wrong_budgets":
        payload["budget_rationals"] = ((1, 10_000), (1, 100_000), (2, 2_000_000))
    with pytest.raises(error):
        select_exact_oracle_v2(**payload)


@pytest.mark.parametrize(
    "budget_rationals",
    [
        list(BUDGET_RATIONALS),
        [list(pair) for pair in BUDGET_RATIONALS],
        tuple(tuple(pair) for pair in BUDGET_RATIONALS),
    ],
)
def test_exact_budget_object_canonicalizes_json_and_tuple_containers(
    budget_rationals: object,
) -> None:
    payload = _curve()
    payload["budget_rationals"] = budget_rationals
    # The values, not the caller container type, are frozen; a JSON-decoded
    # list of exact pairs remains admissible and is canonicalized internally.
    result = select_exact_oracle_v2(**payload)
    assert result.allowed_false_positive_counts == (100, 10, 1)


@pytest.mark.parametrize(
    "budget_rationals",
    [
        [[True, 10_000], [1, 100_000], [1, 1_000_000]],
        [[1.0, 10_000], [1, 100_000], [1, 1_000_000]],
        [[1, 10_000.0], [1, 100_000], [1, 1_000_000]],
        [[2, 20_000], [1, 100_000], [1, 1_000_000]],
        [[1, 10_000, 1], [1, 100_000], [1, 1_000_000]],
        [[1, 100_000], [1, 10_000], [1, 1_000_000]],
    ],
)
def test_exact_budget_object_rejects_noncanonical_integer_contract(
    budget_rationals: object,
) -> None:
    payload = _curve()
    payload["budget_rationals"] = budget_rationals
    with pytest.raises(Stage2ExactOracleError):
        select_exact_oracle_v2(**payload)

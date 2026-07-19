from __future__ import annotations

import hashlib
from itertools import chain

import numpy as np
import pytest

from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from model.endpoint_aware_threshold import (
    MAX_INTERIOR_COORDINATE,
    encode_probability_numpy,
)
from rc.stage2_compositional_curve_provider import (
    BUDGET_COUNT_RULE,
    COMPOSITIONAL_ORACLE_SELECTION_RULE,
    CYCLIC_QUERY_SIZE,
    RC5PLUS_MAX_LIVE_PREDICTIONS,
    CompositionalExactCurveProvider,
    Stage2CompositionalCurveError,
    assert_compositional_exact_curve_provider,
    assert_per_image_exact_event_curve,
    assert_per_image_exact_event_curve_bank,
    build_compositional_exact_curve_provider,
    build_per_image_exact_event_curve,
    build_per_image_exact_event_curve_bank,
)
from rc.stage2_exact_oracle_v2 import select_exact_oracle_v2


def _identity(index: int) -> str:
    return hashlib.sha256(f"synthetic-image-{index}".encode()).hexdigest()


def _curve(
    index: int,
    thresholds: list[float] | np.ndarray,
    fp: list[int] | np.ndarray,
    tp: list[int] | np.ndarray,
    *,
    total: int = 1_000_000,
    objects: int = 10,
):
    return build_per_image_exact_event_curve(
        image_identity_sha256=_identity(index),
        thresholds=np.asarray(thresholds, dtype=np.float64),
        false_positive_pixels=np.asarray(fp, dtype=np.int64),
        matched_objects=np.asarray(tp, dtype=np.int64),
        total_native_pixels=total,
        ground_truth_objects=objects,
    )


def _provider(curves: list[object]) -> CompositionalExactCurveProvider:
    bank = build_per_image_exact_event_curve_bank(curves)
    identities = [curve.image_identity_sha256 for curve in curves]
    return build_compositional_exact_curve_provider(
        curve_bank=bank,
        ordered_image_identities=identities,
    )


def _materialize_conceptual_union(
    provider: CompositionalExactCurveProvider,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent test-only brute force; production never calls this."""

    thresholds = np.asarray(
        sorted(
            set(
                chain.from_iterable(
                    curve.thresholds.tolist() for curve in provider.curves
                )
            )
        ),
        dtype=np.float64,
    )
    fp_rows: list[int] = []
    tp_rows: list[int] = []
    for threshold in thresholds:
        false_positives = 0
        matched = 0
        for curve in provider.curves:
            # Deliberately spell out conceptual strict-threshold resolution
            # rather than calling the provider's resolver.
            eligible = np.flatnonzero(curve.thresholds <= threshold)
            row = int(eligible[-1])
            false_positives += int(curve.false_positive_pixels[row])
            matched += int(curve.matched_objects[row])
        fp_rows.append(false_positives)
        tp_rows.append(matched)
    return (
        thresholds,
        np.asarray(fp_rows, dtype=np.int64),
        np.asarray(tp_rows, dtype=np.int64),
    )


def _assert_brackets_equal_materialized(
    provider: CompositionalExactCurveProvider,
    predictions: np.ndarray,
) -> None:
    thresholds, fp, tp = _materialize_conceptual_union(provider)
    coordinates = encode_probability_numpy(thresholds)
    right = np.searchsorted(coordinates, predictions, side="right")
    right = np.clip(right, 1, thresholds.size - 1)
    selected = np.unique(np.concatenate((right - 1, right)))

    compact = provider.compact_brackets(predictions)
    np.testing.assert_array_equal(compact.thresholds, thresholds[selected])
    np.testing.assert_array_equal(compact.coordinates, coordinates[selected])
    np.testing.assert_array_equal(compact.false_positive_pixels, fp[selected])
    np.testing.assert_array_equal(compact.matched_objects, tp[selected])
    np.testing.assert_array_equal(
        compact.pixel_false_alarm_rate,
        fp[selected].astype(np.float64) / provider.total_native_pixels,
    )
    expected_pd = (
        tp[selected].astype(np.float64) / provider.ground_truth_objects
        if provider.ground_truth_objects
        else np.zeros(selected.size, dtype=np.float64)
    )
    np.testing.assert_array_equal(compact.detection_probability, expected_pd)
    assert 2 <= compact.thresholds.size <= 2 * predictions.size <= 6
    for value in (
        compact.thresholds,
        compact.coordinates,
        compact.false_positive_pixels,
        compact.matched_objects,
        compact.pixel_false_alarm_rate,
        compact.detection_probability,
    ):
        assert value.flags.writeable is False


def _assert_brackets_v2_equal_materialized(
    provider: CompositionalExactCurveProvider,
    predictions: np.ndarray,
) -> None:
    thresholds, fp, tp = _materialize_conceptual_union(provider)
    coordinates = encode_probability_numpy(thresholds)
    right = np.searchsorted(coordinates, predictions, side="right")
    right = np.clip(right, 1, thresholds.size - 1)
    selected = np.unique(np.concatenate((right - 1, right)))

    compact = provider.compact_brackets_v2(predictions)
    np.testing.assert_array_equal(compact.thresholds, thresholds[selected])
    np.testing.assert_array_equal(compact.coordinates, coordinates[selected])
    np.testing.assert_array_equal(compact.false_positive_pixels, fp[selected])
    np.testing.assert_array_equal(compact.matched_objects, tp[selected])
    np.testing.assert_array_equal(
        compact.pixel_false_alarm_rate,
        fp[selected].astype(np.float64) / provider.total_native_pixels,
    )
    expected_pd = (
        tp[selected].astype(np.float64) / provider.ground_truth_objects
        if provider.ground_truth_objects
        else np.zeros(selected.size, dtype=np.float64)
    )
    np.testing.assert_array_equal(compact.detection_probability, expected_pd)
    assert 2 <= compact.thresholds.size <= 2 * predictions.size <= 18
    for value in (
        compact.thresholds,
        compact.coordinates,
        compact.false_positive_pixels,
        compact.matched_objects,
        compact.pixel_false_alarm_rate,
        compact.detection_probability,
    ):
        assert value.flags.writeable is False


def _assert_oracle_equal_materialized(
    provider: CompositionalExactCurveProvider,
) -> None:
    thresholds, fp, tp = _materialize_conceptual_union(provider)
    brute = select_exact_oracle_v2(
        thresholds=thresholds,
        false_positive_pixels=fp,
        matched_objects=tp,
        total_native_pixels=provider.total_native_pixels,
        ground_truth_objects=provider.ground_truth_objects,
    )
    composed = provider.select_exact_oracle_rows()
    assert composed.allowed_false_positive_counts == (
        brute.allowed_false_positive_counts
    )
    np.testing.assert_array_equal(composed.thresholds, brute.thresholds)
    np.testing.assert_array_equal(composed.coordinates, brute.coordinates)
    np.testing.assert_array_equal(
        composed.false_positive_pixels, brute.false_positive_pixels
    )
    np.testing.assert_array_equal(composed.matched_objects, brute.matched_objects)
    np.testing.assert_array_equal(
        composed.pixel_false_alarm_rate, brute.pixel_false_alarm_rate
    )
    np.testing.assert_array_equal(
        composed.detection_probability, brute.detection_probability
    )
    assert composed.global_rows_examined <= thresholds.size
    if composed.stopped_at_first_loose_infeasible:
        allowed = composed.allowed_false_positive_counts[0]
        descending_fp = fp[::-1]
        first_bad = int(np.flatnonzero(descending_fp > allowed)[0])
        assert composed.global_rows_examined == first_bad + 1
    else:
        assert composed.global_rows_examined == thresholds.size


def _assert_oracle_v2_equal_materialized(
    provider: CompositionalExactCurveProvider,
) -> None:
    thresholds, fp, tp = _materialize_conceptual_union(provider)
    allowed = tuple(
        (numerator * provider.total_native_pixels) // denominator
        for numerator, denominator in BUDGET_KNOT_RATIONALS
    )
    expected_rows: list[int] = []
    for maximum in allowed:
        feasible = np.flatnonzero(fp <= maximum)
        assert feasible.size > 0
        expected_rows.append(
            max(
                feasible.tolist(),
                key=lambda row: (
                    int(tp[row]),
                    -int(fp[row]),
                    float(thresholds[row]),
                ),
            )
        )

    composed = provider.select_exact_oracle_rows_v2()
    assert composed.budget_rationals == BUDGET_KNOT_RATIONALS
    assert composed.allowed_false_positive_counts == allowed
    np.testing.assert_array_equal(
        composed.thresholds, thresholds[expected_rows]
    )
    np.testing.assert_array_equal(
        composed.coordinates,
        encode_probability_numpy(thresholds[expected_rows]),
    )
    np.testing.assert_array_equal(
        composed.false_positive_pixels, fp[expected_rows]
    )
    np.testing.assert_array_equal(composed.matched_objects, tp[expected_rows])
    np.testing.assert_array_equal(
        composed.pixel_false_alarm_rate,
        fp[expected_rows].astype(np.float64) / provider.total_native_pixels,
    )
    expected_pd = (
        tp[expected_rows].astype(np.float64) / provider.ground_truth_objects
        if provider.ground_truth_objects
        else np.zeros(len(expected_rows), dtype=np.float64)
    )
    np.testing.assert_array_equal(composed.detection_probability, expected_pd)
    assert composed.thresholds.shape == (RC5PLUS_MAX_LIVE_PREDICTIONS,)
    for value in (
        composed.thresholds,
        composed.coordinates,
        composed.false_positive_pixels,
        composed.matched_objects,
        composed.pixel_false_alarm_rate,
        composed.detection_probability,
    ):
        assert value.flags.writeable is False


def _mixed_curves() -> list[object]:
    curves: list[object] = []
    for index in range(CYCLIC_QUERY_SIZE):
        if index % 7 == 0:
            curves.append(
                _curve(index, [0.0, 1.0], [0, 0], [0, 0], objects=0)
            )
        elif index % 2 == 0:
            curves.append(
                _curve(
                    index,
                    [0.0, 0.2, 0.5, 0.9, 1.0],
                    [3000, 300, 30, 1, 0],
                    [1, 4, 8, 3, 0],
                )
            )
        else:
            curves.append(
                _curve(
                    index,
                    [0.0, 0.1, 0.5, 0.8, 0.95, 1.0],
                    [4000, 500, 40, 4, 0, 0],
                    [2, 5, 9, 6, 2, 0],
                )
            )
    return curves


def test_bank_is_identity_sorted_immutable_and_order_independent() -> None:
    curves = _mixed_curves()
    forward = build_per_image_exact_event_curve_bank(curves)
    reverse = build_per_image_exact_event_curve_bank(list(reversed(curves)))

    assert forward.bank_id == reverse.bank_id
    assert forward.image_count == CYCLIC_QUERY_SIZE
    assert forward.total_curve_rows == sum(
        curve.thresholds.size for curve in curves
    )
    assert [curve.image_identity_sha256 for curve in forward.curves] == sorted(
        curve.image_identity_sha256 for curve in curves
    )
    assert_per_image_exact_event_curve_bank(forward)
    with pytest.raises(TypeError):
        forward._curves_by_identity[_identity(99)] = curves[0]


def test_global_brackets_equal_materialized_union_with_ties_and_endpoints() -> None:
    provider = _provider(_mixed_curves())
    predictions = encode_probability_numpy(
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    )

    _assert_brackets_equal_materialized(provider, predictions)
    # Common 0.5 events across 24 images are one global row, not 24 rows.
    compact = provider.compact_brackets(predictions)
    assert compact.thresholds.tolist() == [0.0, 0.1, 0.5, 0.8, 0.95, 1.0]


@pytest.mark.parametrize("count", [1, 2, 3])
def test_one_to_three_live_predictions_and_repeated_predictions(count: int) -> None:
    provider = _provider(_mixed_curves())
    predictions = encode_probability_numpy(
        np.asarray([0.5] * count, dtype=np.float64)
    )

    _assert_brackets_equal_materialized(provider, predictions)
    assert provider.compact_brackets(predictions).thresholds.size == 2


def test_zero_event_curves_select_endpoint_and_keep_zero_object_pd_defined() -> None:
    curves = [
        _curve(
            index,
            [0.0, 1.0],
            [0, 0],
            [0, 0],
            total=100,
            objects=0,
        )
        for index in range(CYCLIC_QUERY_SIZE)
    ]
    provider = _provider(curves)
    predictions = encode_probability_numpy(
        np.asarray([0.0, 0.73, 1.0], dtype=np.float64)
    )

    _assert_brackets_equal_materialized(provider, predictions)
    _assert_oracle_equal_materialized(provider)
    compact = provider.compact_brackets(predictions)
    assert compact.thresholds.tolist() == [0.0, 1.0]
    assert compact.detection_probability.tolist() == [0.0, 0.0]
    oracle = provider.select_exact_oracle_rows()
    assert oracle.thresholds.tolist() == [1.0, 1.0, 1.0]
    assert oracle.global_rows_examined == 2
    assert oracle.stopped_at_first_loose_infeasible is False


def test_rc5plus_nine_live_predictions_equal_materialized_union() -> None:
    provider = _provider(_mixed_curves())
    predictions = encode_probability_numpy(
        np.asarray(
            [0.0, 0.1, 0.2, 0.5, 0.73, 0.8, 0.9, 0.95, 1.0],
            dtype=np.float64,
        )
    )

    _assert_brackets_v2_equal_materialized(provider, predictions)
    with pytest.raises(Stage2CompositionalCurveError, match=r"length 1\.\.3"):
        provider.compact_brackets(predictions)


def test_rc5plus_nine_budget_oracle_matches_independent_brute_force() -> None:
    provider = _provider(_mixed_curves())

    _assert_oracle_v2_equal_materialized(provider)


def test_rc5plus_zero_event_oracle_is_nine_immutable_endpoints() -> None:
    provider = _provider(
        [
            _curve(
                index,
                [0.0, 1.0],
                [0, 0],
                [0, 0],
                total=100,
                objects=0,
            )
            for index in range(CYCLIC_QUERY_SIZE)
        ]
    )

    _assert_oracle_v2_equal_materialized(provider)
    oracle = provider.select_exact_oracle_rows_v2()
    assert oracle.thresholds.tolist() == [1.0] * len(BUDGET_KNOT_RATIONALS)
    assert oracle.detection_probability.tolist() == [0.0] * len(
        BUDGET_KNOT_RATIONALS
    )


def test_descending_tail_merge_stops_at_first_loose_infeasible_row() -> None:
    thresholds = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    fp = np.full(101, 100, dtype=np.int64)
    fp[-3:] = [20, 1, 0]
    tp = np.full(101, 3, dtype=np.int64)
    tp[-3:] = [2, 1, 0]
    curves = [
        _curve(
            index,
            thresholds,
            fp,
            tp,
            total=100_000,
            objects=3,
        )
        for index in range(CYCLIC_QUERY_SIZE)
    ]
    provider = _provider(curves)

    _assert_oracle_equal_materialized(provider)
    result = provider.select_exact_oracle_rows()
    assert result.allowed_false_positive_counts == (280, 28, 2)
    assert result.global_rows_examined == 3
    assert result.global_rows_examined < thresholds.size
    assert result.stopped_at_first_loose_infeasible is True
    assert result.thresholds.tolist() == [0.99, 0.99, 1.0]


def test_random_composition_matches_brute_force_materialization() -> None:
    rng = np.random.default_rng(20260717)
    shared_grid = np.asarray(
        [
            0.0,
            0.01,
            0.03,
            0.10,
            0.25,
            0.50,
            0.70,
            0.83,
            0.91,
            0.97,
            0.995,
            np.nextafter(1.0, 0.0),
            1.0,
        ],
        dtype=np.float64,
    )
    for trial in range(24):
        curves = []
        for ordinal in range(CYCLIC_QUERY_SIZE):
            interior_count = int(rng.integers(0, shared_grid.size - 2))
            chosen = np.sort(
                rng.choice(
                    np.arange(1, shared_grid.size - 1),
                    size=interior_count,
                    replace=False,
                )
            )
            thresholds = np.concatenate(
                (shared_grid[:1], shared_grid[chosen], shared_grid[-1:])
            )
            total = int(rng.integers(2_000_000, 5_000_001))
            objects = int(rng.integers(0, 16))
            fp = np.concatenate(
                (
                    np.sort(
                        rng.integers(0, 20_000, size=thresholds.size - 1)
                    )[::-1],
                    np.asarray([0]),
                )
            ).astype(np.int64)
            tp = (
                np.concatenate(
                    (
                        rng.integers(
                            0, objects + 1, size=thresholds.size - 1
                        ),
                        np.asarray([0]),
                    )
                ).astype(np.int64)
                if objects
                else np.zeros(thresholds.size, dtype=np.int64)
            )
            curves.append(
                _curve(
                    trial * 100 + ordinal,
                    thresholds,
                    fp,
                    tp,
                    total=total,
                    objects=objects,
                )
            )
        provider = _provider(curves)
        conceptual, _, _ = _materialize_conceptual_union(provider)
        prediction_count = int(rng.integers(1, 4))
        probabilities = rng.choice(
            np.concatenate((conceptual, rng.random(6))),
            size=prediction_count,
            replace=True,
        ).astype(np.float64)
        predictions = encode_probability_numpy(probabilities)

        _assert_brackets_equal_materialized(provider, predictions)
        _assert_oracle_equal_materialized(provider)


def test_provider_retains_only_per_image_curves_and_compact_outputs() -> None:
    curves = _mixed_curves()
    bank = build_per_image_exact_event_curve_bank(curves)
    provider = build_compositional_exact_curve_provider(
        curve_bank=bank,
        ordered_image_identities=[curve.image_identity_sha256 for curve in curves],
    )
    compact = provider.compact_brackets(
        encode_probability_numpy(np.asarray([0.2, 0.6, 0.99]))
    )
    oracle = provider.select_exact_oracle_rows()

    assert provider.curve_bank_id == bank.bank_id
    assert len(provider.curves) == CYCLIC_QUERY_SIZE
    assert not hasattr(provider, "thresholds")
    assert compact.thresholds.size <= 6
    assert oracle.thresholds.shape == (3,)
    assert COMPOSITIONAL_ORACLE_SELECTION_RULE.startswith(
        "descending-k-way-tail-merge-exact-rational"
    )
    assert BUDGET_COUNT_RULE == (
        "(numerator * total_native_pixels) // denominator"
    )
    assert_compositional_exact_curve_provider(provider)


def test_provider_assert_replays_retained_token_identity() -> None:
    provider = _provider(_mixed_curves())
    object.__setattr__(provider, "provider_id", "0" * 64)
    with pytest.raises(TypeError, match="identity replay"):
        assert_compositional_exact_curve_provider(provider)


@pytest.mark.parametrize(
    ("predictions", "error"),
    [
        (np.asarray([], dtype=np.float64), Stage2CompositionalCurveError),
        (np.zeros(4, dtype=np.float64), Stage2CompositionalCurveError),
        (np.zeros(3, dtype=np.float32), Stage2CompositionalCurveError),
        (
            np.asarray([0.0, MAX_INTERIOR_COORDINATE + 0.25]),
            Stage2CompositionalCurveError,
        ),
        ([0.0], TypeError),
    ],
)
def test_live_prediction_contract_fails_closed(
    predictions: object, error: type[Exception]
) -> None:
    provider = _provider(_mixed_curves())
    with pytest.raises(error):
        provider.compact_brackets(predictions)  # type: ignore[arg-type]


def test_bank_and_query_identity_contracts_fail_closed() -> None:
    curves = _mixed_curves()
    with pytest.raises(Stage2CompositionalCurveError, match="duplicate identity"):
        build_per_image_exact_event_curve_bank([curves[0], curves[0]])
    with pytest.raises(Stage2CompositionalCurveError, match="nonempty"):
        build_per_image_exact_event_curve_bank([])

    bank = build_per_image_exact_event_curve_bank(curves)
    identities = [curve.image_identity_sha256 for curve in curves]
    with pytest.raises(Stage2CompositionalCurveError, match="exactly Q"):
        build_compositional_exact_curve_provider(
            curve_bank=bank, ordered_image_identities=identities[:-1]
        )
    duplicate = identities.copy()
    duplicate[-1] = duplicate[0]
    with pytest.raises(Stage2CompositionalCurveError, match="duplicate"):
        build_compositional_exact_curve_provider(
            curve_bank=bank, ordered_image_identities=duplicate
        )
    missing = identities.copy()
    missing[-1] = _identity(9999)
    with pytest.raises(Stage2CompositionalCurveError, match="absent"):
        build_compositional_exact_curve_provider(
            curve_bank=bank, ordered_image_identities=missing
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "negative_zero",
        "duplicate_threshold",
        "fp_increase",
        "fp_exceeds_total",
        "tp_exceeds_objects",
        "endpoint_predicts",
        "float_count",
    ],
)
def test_per_image_curve_validation_rejects_nonexact_or_invalid_rows(
    mutation: str,
) -> None:
    thresholds = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    fp = np.asarray([10, 1, 0], dtype=np.int64)
    tp = np.asarray([2, 1, 0], dtype=np.int64)
    total: object = 100
    if mutation == "negative_zero":
        thresholds[0] = -0.0
    elif mutation == "duplicate_threshold":
        thresholds[1] = 0.0
    elif mutation == "fp_increase":
        fp[1] = 11
    elif mutation == "fp_exceeds_total":
        fp[0] = 101
    elif mutation == "tp_exceeds_objects":
        tp[0] = 11
    elif mutation == "endpoint_predicts":
        tp[-1] = 1
    elif mutation == "float_count":
        total = 100.0
    with pytest.raises((Stage2CompositionalCurveError, TypeError)):
        build_per_image_exact_event_curve(
            image_identity_sha256=_identity(1000),
            thresholds=thresholds,
            false_positive_pixels=fp,
            matched_objects=tp,
            total_native_pixels=total,  # type: ignore[arg-type]
            ground_truth_objects=10,
        )


def test_curve_assert_replays_retained_token_content() -> None:
    curve = _curve(2000, [0.0, 0.5, 1.0], [10, 1, 0], [2, 1, 0])
    object.__setattr__(
        curve,
        "false_positive_pixels",
        np.asarray([9, 1, 0], dtype=np.int64),
    )
    with pytest.raises(TypeError, match="content replay"):
        assert_per_image_exact_event_curve(curve)


def test_bank_assert_replays_retained_token_identity() -> None:
    bank = build_per_image_exact_event_curve_bank(
        [_curve(2001, [0.0, 0.5, 1.0], [10, 1, 0], [2, 1, 0])]
    )
    object.__setattr__(bank, "bank_id", "0" * 64)
    with pytest.raises(TypeError, match="identity replay"):
        assert_per_image_exact_event_curve_bank(bank)

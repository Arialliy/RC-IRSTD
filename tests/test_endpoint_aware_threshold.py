from __future__ import annotations

import numpy as np
import pytest
import torch

from model.endpoint_aware_threshold import (
    INTERIOR_KIND,
    MAX_INTERIOR_COORDINATE,
    MAX_INTERIOR_PROBABILITY,
    PIECEWISE_SPLIT_COORDINATE,
    PIECEWISE_SPLIT_PROBABILITY,
    RAW_COORDINATE_MAX,
    RAW_COORDINATE_MIN,
    UPPER_ENDPOINT_COORDINATE,
    UPPER_ENDPOINT_KIND,
    UPPER_ENDPOINT_SPLIT,
    EndpointAwareThresholdError,
    assert_monotone_coordinate_decision,
    canonicalize_raw_numpy,
    canonicalize_raw_torch,
    decode_coordinate_numpy,
    decode_coordinate_scalar,
    decode_coordinate_torch,
    encode_probability_numpy,
    endpoint_kinds_numpy,
    representation_contract,
    resolve_strict_threshold_event_row,
)


def _nonnegative_float64_ulp_distance(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left_bits = np.asarray(left, dtype=np.float64).view(np.uint64)
    right_bits = np.asarray(right, dtype=np.float64).view(np.uint64)
    return np.maximum(left_bits, right_bits) - np.minimum(left_bits, right_bits)


def test_hard_canonicalization_is_bitwise_idempotent_for_encoded_events() -> None:
    probabilities = np.asarray(
        [
            0.0,
            np.nextafter(np.float64(0.0), np.float64(1.0)),
            0.1,
            0.25,
            0.5,
            1.0 - 1e-10,
            MAX_INTERIOR_PROBABILITY,
            1.0,
        ],
        dtype=np.float64,
    )

    coordinates = encode_probability_numpy(probabilities)
    np.testing.assert_array_equal(
        canonicalize_raw_numpy(coordinates), coordinates
    )
    np.testing.assert_array_equal(
        canonicalize_raw_torch(torch.from_numpy(coordinates.copy()))
        .detach()
        .cpu()
        .numpy(),
        coordinates,
    )


def test_piecewise_lower_half_is_an_exact_bitwise_identity() -> None:
    rng = np.random.default_rng(917)
    half_bits = int(
        np.asarray([PIECEWISE_SPLIT_PROBABILITY], dtype=np.float64).view(
            np.uint64
        )[0]
    )
    random_lower = rng.integers(
        0, half_bits + 1, size=8192, dtype=np.uint64
    ).view(np.float64)
    lower = np.unique(
        np.concatenate(
            [
                random_lower,
                np.asarray(
                    [
                        0.0,
                        np.nextafter(np.float64(0.0), np.float64(1.0)),
                        0.1,
                        0.25,
                        np.nextafter(np.float64(0.5), np.float64(0.0)),
                        0.5,
                    ],
                    dtype=np.float64,
                ),
            ]
        )
    )

    np.testing.assert_array_equal(encode_probability_numpy(lower), lower)
    np.testing.assert_array_equal(decode_coordinate_numpy(lower), lower)
    assert PIECEWISE_SPLIT_COORDINATE == PIECEWISE_SPLIT_PROBABILITY == 0.5


def test_piecewise_encoder_is_strict_on_large_adjacent_float64_sample() -> None:
    rng = np.random.default_rng(20260717)
    one_bits = int(
        np.asarray([1.0], dtype=np.float64).view(np.uint64)[0]
    )
    half_bits = int(
        np.asarray([0.5], dtype=np.float64).view(np.uint64)[0]
    )
    max_interior_bits = int(
        np.asarray([MAX_INTERIOR_PROBABILITY], dtype=np.float64).view(
            np.uint64
        )[0]
    )
    random_left_bits = rng.integers(
        0, one_bits - 1, size=131_072, dtype=np.uint64
    )
    near_split = np.arange(
        half_bits - 8192, half_bits + 8192, dtype=np.uint64
    ).view(np.float64)
    near_upper_endpoint = np.arange(
        max_interior_bits - 16_384,
        max_interior_bits,
        dtype=np.uint64,
    ).view(np.float64)
    left = np.unique(
        np.concatenate(
            [
                random_left_bits.view(np.float64),
                near_split,
                near_upper_endpoint,
                np.asarray(
                    [
                        0.0,
                        np.nextafter(np.float64(0.0), np.float64(1.0)),
                        np.nextafter(np.float64(0.5), np.float64(0.0)),
                        0.5,
                        np.nextafter(np.float64(0.5), np.float64(1.0)),
                        np.nextafter(MAX_INTERIOR_PROBABILITY, 0.0),
                    ],
                    dtype=np.float64,
                ),
            ]
        )
    )
    right = np.nextafter(left, np.float64(1.0))

    assert np.all(right <= MAX_INTERIOR_PROBABILITY)
    assert np.all(
        encode_probability_numpy(right) > encode_probability_numpy(left)
    )


def test_sampled_piecewise_codec_probability_error_is_at_most_one_ulp() -> None:
    rng = np.random.default_rng(20260717)
    random_interior = rng.random(8192, dtype=np.float64)
    near_zero = np.arange(1, 2049, dtype=np.uint64).view(np.float64)
    max_interior_bits = np.asarray(
        [MAX_INTERIOR_PROBABILITY], dtype=np.float64
    ).view(np.uint64)[0]
    near_one = (
        max_interior_bits - np.arange(0, 2048, dtype=np.uint64)
    ).view(np.float64)
    anchors = np.asarray(
        [
            0.0,
            np.nextafter(np.float64(0.0), np.float64(1.0)),
            0.1,
            0.25,
            0.5,
            1.0 - 1e-10,
            float.fromhex("0x1.76185e7716c9bp-1"),
            float.fromhex("0x1.8befe5d13a1fbp-1"),
            MAX_INTERIOR_PROBABILITY,
        ],
        dtype=np.float64,
    )
    probabilities = np.unique(
        np.concatenate([random_interior, near_zero, near_one, anchors])
    )

    coordinates = encode_probability_numpy(probabilities)
    recovered = decode_coordinate_numpy(coordinates)
    reencoded = encode_probability_numpy(recovered)
    probability_ulp = _nonnegative_float64_ulp_distance(
        probabilities, recovered
    )
    coordinate_ulp = _nonnegative_float64_ulp_distance(
        coordinates, reencoded
    )

    assert int(probability_ulp.max()) <= 1
    # Coordinate re-encoding is not a schema guarantee.  Four ULPs is the
    # conservative observed ceiling of the frozen 1,899,438-point audit.
    assert int(coordinate_ulp.max()) <= 4

    quarter_index = int(np.flatnonzero(probabilities == np.float64(0.25))[0])
    assert int(probability_ulp[quarter_index]) == 0
    assert coordinates[quarter_index] == recovered[quarter_index] == 0.25

    probability_witness = int(
        np.flatnonzero(
            probabilities == float.fromhex("0x1.76185e7716c9bp-1")
        )[0]
    )
    coordinate_witness = int(
        np.flatnonzero(
            probabilities == float.fromhex("0x1.8befe5d13a1fbp-1")
        )[0]
    )
    assert int(probability_ulp[probability_witness]) == 1
    assert 0 < int(coordinate_ulp[coordinate_witness]) <= 4


def test_upper_endpoint_is_distinct_from_nextafter_one() -> None:
    probabilities = np.asarray(
        [MAX_INTERIOR_PROBABILITY, np.float64(1.0)], dtype=np.float64
    )
    coordinates = encode_probability_numpy(probabilities)

    assert coordinates[0] == MAX_INTERIOR_COORDINATE
    assert coordinates[1] == UPPER_ENDPOINT_COORDINATE
    assert coordinates[1] - coordinates[0] == 1.0
    assert endpoint_kinds_numpy(coordinates) == (
        INTERIOR_KIND,
        UPPER_ENDPOINT_KIND,
    )
    np.testing.assert_array_equal(decode_coordinate_numpy(coordinates), probabilities)

    contract = representation_contract()
    assert contract["coordinate"] == "piecewise_identity_tail_log_survival"
    assert contract["piecewise_split_probability_hex"] == float(0.5).hex()
    assert contract["endpoint_gap_coordinate_units_hex"] == float(1.0).hex()
    assert contract["max_interior_probability_hex"] == probabilities[0].hex()
    assert contract["upper_endpoint_coordinate_hex"] == coordinates[1].hex()
    assert decode_coordinate_scalar(UPPER_ENDPOINT_COORDINATE) == 1.0
def test_numpy_and_torch_canonicalization_share_exact_boundary_behavior() -> None:
    raw = np.asarray(
        [
            RAW_COORDINATE_MIN,
            np.nextafter(np.float64(0.0), np.float64(-np.inf)),
            0.0,
            0.25,
            MAX_INTERIOR_COORDINATE,
            np.nextafter(UPPER_ENDPOINT_SPLIT, -np.inf),
            UPPER_ENDPOINT_SPLIT,
            RAW_COORDINATE_MAX,
        ],
        dtype=np.float64,
    )
    expected = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            0.25,
            MAX_INTERIOR_COORDINATE,
            MAX_INTERIOR_COORDINATE,
            UPPER_ENDPOINT_COORDINATE,
            UPPER_ENDPOINT_COORDINATE,
        ],
        dtype=np.float64,
    )

    numpy_result = canonicalize_raw_numpy(raw)
    torch_result = canonicalize_raw_torch(torch.from_numpy(raw.copy()))

    np.testing.assert_array_equal(numpy_result, expected)
    assert torch_result.dtype == torch.float64
    np.testing.assert_array_equal(torch_result.detach().cpu().numpy(), expected)


@pytest.mark.parametrize(
    "invalid",
    [
        np.nan,
        np.nextafter(RAW_COORDINATE_MIN, -np.inf),
        np.nextafter(RAW_COORDINATE_MAX, np.inf),
    ],
)
def test_numpy_and_torch_canonicalization_reject_the_same_invalid_raw_values(
    invalid: float,
) -> None:
    with pytest.raises(EndpointAwareThresholdError):
        canonicalize_raw_numpy(np.asarray([invalid], dtype=np.float64))
    with pytest.raises(EndpointAwareThresholdError):
        canonicalize_raw_torch(torch.tensor([invalid], dtype=torch.float64))


def test_torch_canonicalization_has_finite_identity_ste_gradients() -> None:
    raw = torch.tensor(
        [
            -0.5,
            0.25,
            MAX_INTERIOR_COORDINATE + 0.25,
            UPPER_ENDPOINT_SPLIT,
            RAW_COORDINATE_MAX,
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    weights = torch.tensor([1.0, -2.0, 0.5, 3.0, -4.0], dtype=torch.float64)

    canonical = canonicalize_raw_torch(raw)
    (canonical * weights).sum().backward()

    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    torch.testing.assert_close(raw.grad, weights, rtol=0.0, atol=0.0)

    decoded_raw = raw.detach().clone().requires_grad_(True)
    decoded = decode_coordinate_torch(canonicalize_raw_torch(decoded_raw))
    decoded.sum().backward()
    assert decoded_raw.grad is not None
    assert torch.isfinite(decoded_raw.grad).all()


@pytest.mark.parametrize("_family", ("T7", "T8"), ids=("T7", "T8"))
def test_t7_t8_canonical_decisions_are_monotone_with_endpoint_suffix(
    _family: str,
) -> None:
    raw = np.asarray(
        [
            -0.25,
            0.25,
            MAX_INTERIOR_COORDINATE + 0.25,
            UPPER_ENDPOINT_SPLIT,
            RAW_COORDINATE_MAX,
        ],
        dtype=np.float64,
    )
    coordinates = canonicalize_raw_numpy(raw)
    thresholds = decode_coordinate_numpy(coordinates)

    assert np.all(np.diff(coordinates) >= 0.0)
    assert np.all(np.diff(thresholds) >= 0.0)
    assert endpoint_kinds_numpy(coordinates) == (
        INTERIOR_KIND,
        INTERIOR_KIND,
        INTERIOR_KIND,
        UPPER_ENDPOINT_KIND,
        UPPER_ENDPOINT_KIND,
    )
    assert_monotone_coordinate_decision(coordinates, thresholds)


@pytest.mark.parametrize("_family", ("T7", "T8"), ids=("T7", "T8"))
def test_t7_t8_reject_an_endpoint_followed_by_an_interior_coordinate(
    _family: str,
) -> None:
    coordinates = np.asarray(
        [0.0, UPPER_ENDPOINT_COORDINATE, MAX_INTERIOR_COORDINATE],
        dtype=np.float64,
    )
    thresholds = decode_coordinate_numpy(coordinates)

    with pytest.raises(EndpointAwareThresholdError, match="coordinates decrease"):
        assert_monotone_coordinate_decision(coordinates, thresholds)


@pytest.mark.parametrize("_consumer", ("T7", "T8"), ids=("T7", "T8"))
def test_shared_strict_event_resolver_matches_brute_force_semantics(
    _consumer: str,
) -> None:
    scores = np.asarray(
        [
            0.0,
            0.125,
            0.125,
            0.5,
            0.75,
            MAX_INTERIOR_PROBABILITY,
            1.0,
            1.0,
        ],
        dtype=np.float64,
    )
    curve_thresholds = np.unique(scores)
    interior_events = curve_thresholds[1:-1]
    midpoints = (curve_thresholds[:-1] + curve_thresholds[1:]) / 2.0
    queries = np.unique(
        np.concatenate(
            [
                curve_thresholds,
                midpoints,
                np.nextafter(interior_events, -np.inf),
                np.nextafter(interior_events, np.inf),
            ]
        )
    )

    for query in queries:
        resolved = resolve_strict_threshold_event_row(curve_thresholds, float(query))
        brute_row = max(
            index
            for index, event in enumerate(curve_thresholds)
            if event <= query
        )

        assert resolved == brute_row
        np.testing.assert_array_equal(
            scores > curve_thresholds[resolved],
            scores > query,
            err_msg=f"strict-event replay mismatch at threshold {query.hex()}",
        )

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, laplace, maximum_filter

import rc.domain_statistics as domain_statistics


def _legacy_base_features(
    probabilities: list[np.ndarray],
    grayscale_images: list[np.ndarray],
) -> np.ndarray:
    """Small-input reference for the pre-streaming concatenate implementation."""

    probability_images = [
        np.clip(image.astype(np.float64, copy=False), 0.0, 1.0)
        for image in probabilities
    ]
    flat_probabilities = np.concatenate(
        [image.reshape(-1) for image in probability_images]
    )
    probability_histogram = np.histogram(
        flat_probabilities,
        bins=domain_statistics.PROBABILITY_HISTOGRAM_BINS,
        range=(0.0, 1.0),
    )[0].astype(np.float64)
    probability_histogram /= probability_histogram.sum()
    probability_quantiles = np.quantile(
        flat_probabilities, domain_statistics.QUANTILES
    )

    peak_arrays: list[np.ndarray] = []
    for image in probability_images:
        pooled = maximum_filter(image, size=3, mode="nearest")
        candidates = (image == pooled) & (image >= 0.05)
        peak_arrays.append(
            domain_statistics._plateau_representative_values(
                image, candidates, kernel_size=3
            )
        )
    nonempty_peaks = [values for values in peak_arrays if values.size]
    peaks = np.concatenate(nonempty_peaks)
    peak_histogram = np.histogram(
        peaks,
        bins=domain_statistics.PEAK_HISTOGRAM_BINS,
        range=(0.0, 1.0),
    )[0].astype(np.float64)
    peak_histogram /= peak_histogram.sum()

    gray_images = [
        domain_statistics._normalise_grayscale(image) for image in grayscale_images
    ]
    gray_values = np.concatenate([image.reshape(-1) for image in gray_images])
    gradients: list[np.ndarray] = []
    laplacians: list[np.ndarray] = []
    high_frequency_energy: list[float] = []
    for image in gray_images:
        grad_y, grad_x = np.gradient(image)
        gradients.append(np.hypot(grad_x, grad_y).reshape(-1))
        laplacian_image = laplace(image, mode="nearest")
        laplacians.append(
            np.abs(laplacian_image - np.median(laplacian_image)).reshape(-1)
        )
        high = image - gaussian_filter(image, sigma=1.0, mode="nearest")
        high_frequency_energy.append(
            float(
                np.mean(np.square(high))
                / max(float(np.mean(np.square(image))), 1e-12)
            )
        )
    gradient_values = np.concatenate(gradients)
    laplacian_values = np.concatenate(laplacians)
    return np.concatenate(
        [
            probability_histogram,
            np.quantile(flat_probabilities, domain_statistics.QUANTILES),
            peak_histogram,
            np.quantile(peaks, domain_statistics.QUANTILES),
            np.asarray(
                [
                    peaks.size
                    / (sum(image.size for image in probability_images) / 1_000_000.0),
                    1.0,
                    gray_values.mean(),
                    gray_values.std(),
                    np.median(np.abs(gray_values - np.median(gray_values))),
                    gradient_values.mean(),
                    np.quantile(gradient_values, 0.95),
                    np.median(laplacian_values),
                    np.mean(high_frequency_energy),
                ]
            ),
        ]
    ).astype(np.float32)


def test_streaming_statistics_match_legacy_exact_small_window() -> None:
    rng = np.random.default_rng(20260714)
    shapes = ((7, 9), (5, 8), (9, 6))
    probabilities = [rng.random(shape, dtype=np.float32) for shape in shapes]
    grayscale = [
        rng.integers(0, 256, size=shape, dtype=np.uint8) for shape in shapes
    ]

    result = domain_statistics.extract_unlabeled_statistics(
        (image for image in probabilities),
        (image for image in grayscale),
    )
    expected = _legacy_base_features(probabilities, grayscale)

    np.testing.assert_allclose(
        result.vector[: domain_statistics.BASE_FEATURE_DIM],
        expected,
        rtol=1e-6,
        atol=1e-7,
    )
    assert result.metadata["num_images"] == len(probabilities)
    assert all(result.metadata["quantiles_exact"].values())


@pytest.mark.parametrize(
    "empty_probabilities",
    [[], iter(()), np.empty((0, 4, 4), dtype=np.float32)],
)
def test_streaming_statistics_reject_empty_probability_input(
    empty_probabilities: object,
) -> None:
    with pytest.raises(ValueError, match="probabilities must contain at least one image"):
        domain_statistics.extract_unlabeled_statistics(empty_probabilities)


def test_streaming_statistics_reject_mismatched_grayscale_length() -> None:
    probability = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="must have the same length"):
        domain_statistics.extract_unlabeled_statistics([probability], iter(()))
    with pytest.raises(ValueError, match="must have the same length"):
        domain_statistics.extract_unlabeled_statistics(
            [probability], [probability, probability]
        )


def test_streaming_statistics_bound_samples_and_consume_generators_in_lockstep() -> None:
    events: list[str] = []

    def probabilities():
        for index in range(80):
            events.append(f"p{index}")
            values = np.arange(42, dtype=np.float64).reshape(6, 7)
            yield np.mod(values * 0.037 + index * 0.011, 1.0)

    def grayscale():
        for index in range(80):
            events.append(f"g{index}")
            values = np.arange(42, dtype=np.float64).reshape(6, 7)
            yield np.mod(values * 0.021 + index * 0.017, 1.0)

    config = domain_statistics.StatisticsConfig(
        peak_kernel_size=3,
        peak_min_score=0.05,
        quantile_sample_limit=17,
    )
    with mock.patch.object(domain_statistics, "STREAM_CHUNK_SIZE", 5):
        first = domain_statistics.extract_unlabeled_statistics(
            probabilities(), grayscale(), statistics_config=config
        )
    with mock.patch.object(domain_statistics, "STREAM_CHUNK_SIZE", 13):
        second = domain_statistics.extract_unlabeled_statistics(
            probabilities(), grayscale(), statistics_config=config
        )

    expected_first_run = [
        event for index in range(80) for event in (f"p{index}", f"g{index}")
    ]
    assert events[:160] == expected_first_run
    assert events[160:] == expected_first_run
    assert max(first.metadata["quantile_sample_counts"].values()) <= 17
    assert not any(first.metadata["quantiles_exact"].values())
    assert first.metadata["streaming_memory_contract"] == {
        "mode": "single_image_plus_bounded_deterministic_priority_samples",
        "quantile_sample_limit_per_family": 17,
        "quantile_estimator": "splitmix64_priority_bottom_k_stream_index_v1",
        "stream_chunk_size": 5,
        "auxiliary_memory_independent_of_domain_pixel_count": True,
    }
    np.testing.assert_array_equal(first.vector, second.vector)


def test_statistics_v3_contract_rejects_explicit_v2_artifacts() -> None:
    config = domain_statistics.StatisticsConfig(
        peak_kernel_size=3,
        peak_min_score=0.05,
    )
    payload = config.to_dict()
    assert payload["algorithm_version"] == (
        "rc-domain-statistics-v3-bounded-quantiles"
    )
    assert payload["quantile_sample_limit"] == 262_144
    assert payload["quantile_estimator"] == (
        "splitmix64_priority_bottom_k_stream_index_v1"
    )

    payload["algorithm_version"] = "rc-domain-statistics-v2"
    payload.pop("quantile_sample_limit")
    payload.pop("quantile_estimator")
    with pytest.raises(ValueError, match="unsupported statistics algorithm_version"):
        domain_statistics.StatisticsConfig.from_dict(payload)

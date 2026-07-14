from __future__ import annotations

import gc
import weakref
from pathlib import Path
from unittest import mock

import numpy as np

from data_ext.score_manifest_artifacts import VerifiedScoreItem
from rc.build_source_reference import _load_domain_inputs
from rc.domain_statistics import extract_unlabeled_statistics
from rc.schema import StatisticsConfig


def _verified_items(root: Path, count: int) -> tuple[VerifiedScoreItem, ...]:
    items: list[VerifiedScoreItem] = []
    gray_path = root / "placeholder-gray.png"
    gray_path.touch()
    for index in range(count):
        image_id = f"image-{index:04d}"
        score_path = root / f"{image_id}.npz"
        # _load_domain_inputs independently rechecks the embedded image ID.
        np.savez(score_path, image_id=np.asarray(image_id))
        items.append(
            VerifiedScoreItem(
                manifest_index=index,
                image_id=image_id,
                record={},
                score_path=score_path,
                gray_path=gray_path,
                original_hw=(32, 32),
            )
        )
    return tuple(items)


def test_source_domain_inputs_are_lazy_and_consumed_in_lockstep(tmp_path: Path) -> None:
    items = _verified_items(tmp_path, 5)
    calls: list[str] = []

    def load_one(
        score_path: str | Path,
        gray_path: str | Path | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del gray_path
        calls.append(Path(score_path).stem)
        value = float(len(calls)) / 10.0
        return (
            np.full((32, 32), value, dtype=np.float64),
            np.full((32, 32), value, dtype=np.float64),
        )

    with mock.patch(
        "rc.build_source_reference.load_probability_and_grayscale",
        side_effect=load_one,
    ):
        probabilities, grayscale = _load_domain_inputs(
            {"target_dataset": "source-A"}, items
        )
        assert grayscale is not None
        assert calls == []

        probability_iterator = iter(probabilities)
        grayscale_iterator = iter(grayscale)
        np.testing.assert_array_equal(next(probability_iterator), 0.1)
        assert calls == ["image-0000"]
        np.testing.assert_array_equal(next(grayscale_iterator), 0.1)
        # The gray half comes from the explicit one-pair handoff; it must not
        # reopen the artifact or prefetch any later image.
        assert calls == ["image-0000"]

        for expected_index, (probability, gray) in enumerate(
            zip(probability_iterator, grayscale_iterator), start=1
        ):
            expected = float(expected_index + 1) / 10.0
            np.testing.assert_array_equal(probability, expected)
            np.testing.assert_array_equal(gray, expected)
            assert len(calls) == expected_index + 1

    assert calls == [f"image-{index:04d}" for index in range(5)]


def test_source_reference_stream_has_domain_size_independent_pixel_memory(
    tmp_path: Path,
) -> None:
    maxima: list[int] = []
    config = StatisticsConfig(
        peak_kernel_size=3,
        peak_min_score=0.05,
        quantile_sample_limit=19,
    )

    for item_count in (6, 60):
        case_root = tmp_path / str(item_count)
        case_root.mkdir()
        items = _verified_items(case_root, item_count)
        live_arrays: list[weakref.ReferenceType[np.ndarray]] = []
        maximum_live = 0

        def load_one(
            score_path: str | Path,
            gray_path: str | Path | None,
        ) -> tuple[np.ndarray, np.ndarray]:
            nonlocal maximum_live
            del gray_path
            index = int(Path(score_path).stem.rsplit("-", 1)[1])
            probability = np.full(
                (32, 32), (index + 1) / (item_count + 1), dtype=np.float64
            )
            grayscale = np.full((32, 32), index % 255, dtype=np.uint8)
            live_arrays[:] = [
                reference for reference in live_arrays if reference() is not None
            ]
            live_arrays.extend((weakref.ref(probability), weakref.ref(grayscale)))
            maximum_live = max(
                maximum_live,
                sum(reference() is not None for reference in live_arrays),
            )
            return probability, grayscale

        with mock.patch(
            "rc.build_source_reference.load_probability_and_grayscale",
            side_effect=load_one,
        ):
            probabilities, grayscale = _load_domain_inputs(
                {"target_dataset": "source-A"}, items
            )
            assert grayscale is not None
            statistics = extract_unlabeled_statistics(
                probabilities,
                grayscale,
                statistics_config=config,
            )

        gc.collect()
        maxima.append(maximum_live)
        assert not any(reference() is not None for reference in live_arrays)
        assert statistics.metadata["num_images"] == item_count
        assert max(statistics.metadata["quantile_sample_counts"].values()) <= 19
        assert statistics.metadata["statistics_config"]["algorithm_version"] == (
            "rc-domain-statistics-v3-bounded-quantiles"
        )

    # A materialising implementation reaches 2 * item_count live source
    # arrays.  The paired stream retains only a constant number of image
    # arrays, independent of the number of files in the domain.
    assert maxima[0] <= 4
    assert maxima[1] <= 4

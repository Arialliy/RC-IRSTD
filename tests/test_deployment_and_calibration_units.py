from __future__ import annotations

from pathlib import Path

import numpy as np

from rc_irstd.calibration.samples import image_calibration_samples
from rc_irstd.deployment.monitor import feature_ood_score, score_drift
from rc_irstd.deployment.session import DeploymentState, ThresholdUpdate
from rc_irstd.episodes.dataset import EpisodeArrays
from rc_irstd.models.risk_curve import FeatureNormaliser


def _episode_arrays() -> EpisodeArrays:
    thresholds = np.asarray([0.0, 0.5, 1.000001], dtype=np.float32)
    pixel_risk = np.asarray([[0.2, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32)
    peak_risk = np.asarray([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    pd = np.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    return EpisodeArrays(
        features=np.zeros((2, 4), dtype=np.float32),
        pixel_log_risk=np.log10(np.maximum(pixel_risk, 1e-12)),
        peak_log_risk=np.log10(np.maximum(peak_risk, 1e-6)),
        pixel_risk=pixel_risk,
        peak_risk=peak_risk,
        pd=pd,
        context_pixel_upper=np.zeros_like(pixel_risk),
        context_peak_upper=np.zeros_like(peak_risk),
        thresholds=thresholds,
        domains=np.asarray(["A", "A"]),
        sequences=np.asarray(["iid0", "iid1"]),
        context_ids=np.asarray(['["c0"]', '["c1"]']),
        future_ids=np.asarray(['["f0"]', '["f1"]']),
        feature_names=("a", "b", "c", "d"),
        protocols=np.asarray(["iid", "iid"]),
        future_pixel_risk=np.asarray(
            [[[0.2, 0.0, 0.0]], [[0.3, 0.0, 0.0]]], dtype=np.float32
        ),
        future_peak_risk=np.asarray(
            [[[2.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]], dtype=np.float32
        ),
        future_pd=np.asarray(
            [[[1.0, 1.0, 0.0]], [[1.0, 1.0, 0.0]]], dtype=np.float32
        ),
        future_gt_count=np.ones((2, 1), dtype=np.int32),
    )


def test_image_calibration_samples_count_exact_images() -> None:
    arrays = _episode_arrays()
    samples = image_calibration_samples(
        arrays,
        base_indices=np.asarray([1, 1]),
        base_rejected=np.asarray([False, False]),
    )
    assert samples.unit == "image"
    assert samples.num_samples == 2
    assert samples.label_count_per_sample.tolist() == [1, 1]
    assert samples.sample_ids.tolist() == ["f0", "f1"]


def test_deployment_state_and_monitor() -> None:
    normaliser = FeatureNormaliser(
        mean=np.asarray([1.0, 2.0], dtype=np.float32),
        std=np.asarray([2.0, 4.0], dtype=np.float32),
    )
    assert feature_ood_score(np.asarray([1.0, 2.0]), normaliser) == 0.0
    assert score_drift(np.asarray([1.0, 1.0]), np.asarray([2.0, 1.0])) > 0.0

    state = DeploymentState(
        detector_checkpoint="det.pt",
        curve_checkpoint="curve.pt",
        score_directory="scores",
        pixel_budget=1e-6,
        peak_budget_per_mp=1.0,
        warmup_size=32,
    )
    state.add(
        ThresholdUpdate(
            sequence_id="default",
            update_index=32,
            warmup_ids=("a", "b"),
            base_threshold_index=3,
            offset_index=1,
            final_threshold_index=4,
            threshold=0.8,
            predicted_pixel_risk=1e-7,
            predicted_peak_risk_per_mp=0.5,
            rejected=False,
            feature_ood_score=1.2,
        )
    )
    payload = state.to_dict()
    assert payload["updates"][0]["threshold"] == 0.8
    assert payload["updates"][0]["warmup_ids"] == ["a", "b"]


def test_apply_operating_point_serialises_single_peak_coordinates(tmp_path: Path) -> None:
    import csv

    from rc_irstd.data.score_records import ScoreRecord, save_score_record
    from rc_irstd.pipelines.apply_operating_point import main as apply_main

    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    probability = np.zeros((7, 9), dtype=np.float32)
    probability[3, 5] = 0.9
    save_score_record(
        ScoreRecord(
            probability=probability,
            mask=None,
            image_stats=np.zeros(2, dtype=np.float32),
            image_stat_names=("mean", "std"),
            image_id="single",
            dataset_name="D",
            sequence_id="default",
            frame_index=0,
            original_hw=(7, 9),
        ),
        score_dir / "00000000.npz",
    )
    output_dir = tmp_path / "applied"
    apply_main(
        [
            "--score-dir",
            str(score_dir),
            "--threshold",
            "0.5",
            "--peak-min-distance",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )
    with (output_dir / "candidates.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert (int(rows[0]["y"]), int(rows[0]["x"])) == (3, 5)

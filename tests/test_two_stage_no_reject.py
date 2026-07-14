from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rc_irstd.data.score_records import ScoreRecord, load_score_record, save_score_record
from rc_irstd.episodes.meta_dataset import (
    MetaEpisodeBuildConfig,
    build_meta_episode_file,
    load_meta_episode_file,
)
from rc_irstd.losses.calibrator import risk_aligned_calibrator_loss
from rc_irstd.losses.target_background_margin import (
    background_local_peak_mask,
    domain_tail_separation_loss,
)
from rc_irstd.models.monotone_pixel_calibrator import MonotonePixelCalibrator


def _toy_tail_inputs():
    logits = torch.zeros((4, 1, 9, 9), dtype=torch.float32, requires_grad=True)
    masks = torch.zeros_like(logits)
    masks[0, 0, 4, 4] = 1
    masks[2, 0, 2, 2] = 1
    with torch.no_grad():
        logits[0, 0, 4, 4] = 3.0
        logits[2, 0, 2, 2] = 2.5
        logits[1, 0, 7, 7] = 2.0  # no-target image still contributes background
        logits[3, 0, 6, 6] = 1.5
    domains = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    return logits, masks, domains


def test_domain_tail_separation_is_shift_invariant_and_uses_no_target_images():
    logits, masks, domains = _toy_tail_inputs()
    first = domain_tail_separation_loss(
        logits,
        masks,
        domains,
        margin=1.0,
        background_tail_fraction=0.5,
        object_top_fraction=1.0,
        hard_object_fraction=1.0,
        peak_kernel_size=3,
        exclusion_radius=1,
    )
    shifted = domain_tail_separation_loss(
        logits + 7.0,
        masks,
        domains,
        margin=1.0,
        background_tail_fraction=0.5,
        object_top_fraction=1.0,
        hard_object_fraction=1.0,
        peak_kernel_size=3,
        exclusion_radius=1,
    )
    assert torch.allclose(first.loss, shifted.loss, atol=1e-6)
    assert first.image_background_tail[1] >= 1.9
    assert first.image_background_tail[3] >= 1.4
    first.loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_background_plateau_is_one_candidate():
    logits = torch.zeros((1, 1, 11, 11))
    target = torch.zeros_like(logits)
    peaks, _ = background_local_peak_mask(logits, target, kernel_size=3)
    assert int(peaks.sum()) == 1


def test_target_is_excluded_before_background_peak_pooling():
    logits = torch.zeros((1, 1, 7, 7))
    target = torch.zeros_like(logits)
    target[0, 0, 3, 3] = 1
    logits[0, 0, 3, 3] = 100.0
    logits[0, 0, 3, 4] = 5.0

    peaks, valid = background_local_peak_mask(
        logits,
        target,
        kernel_size=3,
        exclusion_radius=0,
    )

    assert not bool(valid[0, 0, 3, 3])
    assert bool(valid[0, 0, 3, 4])
    assert bool(peaks[0, 0, 3, 4])


def test_monotone_inverse_curve_has_no_reject_and_is_source_permutation_invariant():
    torch.manual_seed(4)
    model = MonotonePixelCalibrator(7, [1e-4, 1e-5, 1e-6], dropout=0.0)
    model.eval()
    features = torch.randn(2, 7)
    distances = torch.tensor([[0.2, 1.0, 0.4], [0.5, 0.1, 0.7]])
    output = model(features, distances)
    permuted = model(features, distances[:, [2, 0, 1]])
    assert set(output) == {"threshold_logit", "threshold"}
    assert torch.all(torch.diff(output["threshold_logit"], dim=1) >= 0)
    assert torch.allclose(output["threshold_logit"], permuted["threshold_logit"], atol=1e-6)
    interpolated = model.interpolate_logit(output["threshold_logit"], [5e-5, 5e-6])
    assert interpolated.shape == (2, 2)
    with pytest.raises(ValueError, match="extrapolation"):
        model.interpolate_logit(output["threshold_logit"], [1e-7])


def test_query_risk_aligned_loss_backpropagates():
    eta = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    budgets = torch.tensor([0.2, 0.1, 0.05])
    oracle = torch.tensor([[0.2, 1.1, 2.1]])
    background = torch.tensor([[2.5, 1.5, 0.5, -1.0]])
    background_valid = torch.ones_like(background, dtype=torch.bool)
    background_fraction = torch.tensor([0.95])
    objects = torch.tensor([[3.0, 2.2]])
    object_valid = torch.ones_like(objects, dtype=torch.bool)
    loss = risk_aligned_calibrator_loss(
        eta,
        budgets,
        oracle,
        background,
        background_valid,
        background_fraction,
        objects,
        object_valid,
    )
    assert loss.violation > 0
    assert 0 <= loss.surrogate_pd.mean() <= 1
    loss.total.backward()
    assert eta.grad is not None and torch.isfinite(eta.grad).all()


def _write_score(path: Path, index: int) -> None:
    probability = np.full((8, 8), 0.02, dtype=np.float32)
    probability[(index + 1) % 8, (2 * index + 1) % 8] = 0.8
    mask = np.zeros((8, 8), dtype=np.uint8)
    if index % 2 == 0:
        mask[(index + 1) % 8, (2 * index + 1) % 8] = 1
    save_score_record(
        ScoreRecord(
            probability=probability,
            mask=mask,
            image_stats=np.asarray([0.5, 0.1], dtype=np.float32),
            image_stat_names=("mean", "std"),
            image_id=f"image_{index:03d}",
            dataset_name="PseudoTarget",
            sequence_id="iid",
            frame_index=index,
            original_hw=(8, 8),
            source_checkpoint="detector-sha",
            dataset_type="iid_images",
        ),
        path,
    )


def test_meta_dataset_groups_budget_vector_and_support_loader_is_label_free(tmp_path: Path):
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    for index in range(8):
        _write_score(score_dir / f"{index:03d}.npz", index)
    assert load_score_record(score_dir / "000.npz", load_mask=False).mask is None

    output = tmp_path / "meta.npz"
    build_meta_episode_file(
        score_dir,
        output,
        budgets=[0.20, 0.10, 0.05],
        config=MetaEpisodeBuildConfig(
            context_size=2,
            horizon=2,
            stride=4,
            protocol="iid",
            background_sample_limit=128,
            seed=3,
        ),
    )
    arrays = load_meta_episode_file(output)
    assert arrays.features.shape[0] == 2
    assert arrays.oracle_threshold_logit.shape == (2, 3)
    assert arrays.budgets.tolist() == pytest.approx([0.20, 0.10, 0.05])
    assert arrays.background_logits.shape[0] == 2
    assert arrays.hard_pixel_risk.shape[0] == 2
    assert all(set(__import__("json").loads(s)).isdisjoint(__import__("json").loads(q)) for s, q in zip(arrays.support_ids, arrays.query_ids, strict=True))

from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from data_ext.stage2_threshold_decision import PIXEL_BUDGET_GRID
from evaluation.stage2_source_threshold_pool import (
    build_source_threshold_reference_from_verified_collection,
    pool_source_validation_safe_rows,
)
from evaluation.stage2_threshold_family import (
    Stage2ThresholdFamilyError,
    select_source_safe_threshold,
)
from rc.stage2_crossfit_dataset import (
    fit_stage2_context_standardizer,
    make_stage2_trainer_replay_capability,
)
from rc.stage2_crossfit_schema import (
    COLLECTION_TRAIN,
    COLLECTION_VALIDATION,
    STAGE2_OOF_FIT,
    Stage2CrossfitEpisode,
    VerifiedEpisodeArtifacts,
    make_verified_collection,
)
from data_ext.stage2_label_attachment import SOURCE_DIAGNOSTIC_VALIDATION


def _rows(
    threshold: list[float], tp: list[int], fp: list[int], *, gt: int, pixels: int
) -> list[dict[str, int | float]]:
    return [
        {
            "threshold": value,
            "tp_objects": detected,
            "gt_objects": gt,
            "fp_pixels": false_pixels,
            "total_pixels": pixels,
        }
        for value, detected, false_pixels in zip(threshold, tp, fp, strict=True)
    ]


def _brute_pool(curves: list[list[dict[str, int | float]]]) -> list[dict[str, int | float]]:
    events = sorted({float(row["threshold"]) for curve in curves for row in curve})
    pooled: list[dict[str, int | float]] = []
    for threshold in events:
        local = []
        for curve in curves:
            eligible = [row for row in curve if float(row["threshold"]) <= threshold]
            local.append(eligible[-1])
        pooled.append(
            {
                "threshold": threshold,
                "tp_objects": sum(int(row["tp_objects"]) for row in local),
                "gt_objects": sum(int(row["gt_objects"]) for row in local),
                "fp_pixels": sum(int(row["fp_pixels"]) for row in local),
                "total_pixels": sum(int(row["total_pixels"]) for row in local),
            }
        )
    return pooled


def _expected(curves: list[list[dict[str, int | float]]]) -> list[dict[str, object]]:
    pooled = _brute_pool(curves)
    return [select_source_safe_threshold(pooled, budget) for budget in PIXEL_BUDGET_GRID]


def test_streaming_pool_matches_materialized_union_for_random_curves() -> None:
    rng = np.random.default_rng(20260717)
    for trial in range(40):
        by_domain: dict[str, list[list[dict[str, int | float]]]] = {
            "IRSTD-1K": [],
            "NUDT-SIRST": [],
        }
        for domain in by_domain:
            for curve_index in range(3):
                interior = np.sort(rng.choice(np.arange(1, 999), size=9, replace=False)) / 1000.0
                threshold = np.concatenate(([0.0], interior, [1.0])).tolist()
                gt = 5 + trial + curve_index
                tp = np.sort(rng.integers(0, gt + 1, size=len(threshold)))[::-1]
                fp = np.sort(rng.integers(0, 500, size=len(threshold)))[::-1]
                tp[-1] = 0
                fp[-1] = 0
                by_domain[domain].append(
                    _rows(threshold, tp.tolist(), fp.tolist(), gt=gt, pixels=1_000_000)
                )
        pooled, per_domain = pool_source_validation_safe_rows(by_domain)
        assert pooled == _expected([curve for curves in by_domain.values() for curve in curves])
        for domain, curves in by_domain.items():
            assert per_domain[domain] == _expected(curves)


def test_streaming_pool_uses_fp_then_larger_threshold_exact_ties() -> None:
    left = _rows(
        [0.0, 0.2, 0.7, 0.8, 1.0],
        [8, 8, 8, 8, 0],
        [200, 100, 80, 80, 0],
        gt=10,
        pixels=1_000_000,
    )
    right = _rows(
        [0.0, 0.3, 0.7, 0.8, 1.0],
        [9, 9, 9, 9, 0],
        [180, 100, 20, 20, 0],
        gt=10,
        pixels=1_000_000,
    )
    pooled, _ = pool_source_validation_safe_rows(
        {"IRSTD-1K": [left], "NUDT-SIRST": [right]}
    )
    # At 1e-4, t=0.7 and t=0.8 have identical Pd and FP.  The frozen B3
    # rank chooses the larger threshold.
    assert pooled[0]["threshold"] == 0.8
    assert pooled[0]["tp_objects"] == 17
    assert Fraction(int(pooled[0]["fp_pixels"]), int(pooled[0]["total_pixels"])) <= Fraction.from_float(1e-4)


def test_streaming_pool_accepts_legitimate_nonmonotone_object_tp_counts() -> None:
    # Connected components may split as the threshold rises, so maximum
    # one-to-one matched target count is not a monotone curve even though
    # foreground/false-positive pixels are monotone.
    split_curve = _rows(
        [0.0, 0.4, 0.7, 1.0],
        [1, 3, 2, 0],
        [200, 90, 9, 0],
        gt=5,
        pixels=1_000_000,
    )
    other = _rows(
        [0.0, 0.5, 0.8, 1.0],
        [2, 2, 1, 0],
        [180, 80, 8, 0],
        gt=5,
        pixels=1_000_000,
    )
    pooled, per_domain = pool_source_validation_safe_rows(
        {"IRSTD-1K": [split_curve], "NUDT-SIRST": [other]}
    )
    assert len(pooled) == 3
    assert set(per_domain) == {"IRSTD-1K", "NUDT-SIRST"}


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _episode(
    index: int,
    role: str,
    domain: str,
    *,
    collection_tag: str,
) -> Stage2CrossfitEpisode:
    training = role == COLLECTION_TRAIN
    values = np.full(93, index / 10.0, dtype=np.float64)
    payload = {
        "episode_id": _sha(f"{collection_tag}:episode:{index}"),
        "episode_index": index,
        "collection_role": role,
        "episode_role": STAGE2_OOF_FIT if training else SOURCE_DIAGNOSTIC_VALIDATION,
        "outer_fold_id": "outer_leave_nuaa_sirst",
        "outer_target": "NUAA-SIRST",
        "source_domain": domain,
        "base_seed": 42,
        "derived_seed": 42001,
        "window_binding": {
            "path": f"synthetic/{collection_tag}-window-{index}.json",
            "window_id": f"{collection_tag}-window-{index}",
        },
        "context_statistics": {
            "values": values.tolist(),
            "vector_sha256": _sha(f"{collection_tag}:vector:{index}"),
        },
        "context_full_identity_sha256": _sha(
            f"{collection_tag}:context:{index}"
        ),
        "source_ordered_query_identity_sha256": _sha(
            f"{collection_tag}:query:{index}"
        ),
        "detector_identity": {
            "outer_fold_id": "outer_leave_nuaa_sirst",
            "outer_target": "NUAA-SIRST",
            "base_seed": 42,
            "derived_seed": 42001,
            "detector_role": "detector_oof" if training else "detector_full_fit",
            "oof_fold_index": index % 2 if training else None,
            "checkpoint_sha256": _sha(
                f"{collection_tag}:oof:{index % 2}"
                if training
                else "validation-full-fit-checkpoint"
            ),
        },
        "context_records": [],
        "query_records": [],
    }
    return Stage2CrossfitEpisode(MappingProxyType(payload))


def _verified_collection(
    root: Path,
    role: str,
    *,
    collection_tag: str,
):
    count = 26 if role == COLLECTION_TRAIN else 6
    domains = ("NUDT-SIRST", "IRSTD-1K")
    episodes = [
        _episode(
            index,
            role,
            domains[index % 2],
            collection_tag=collection_tag,
        )
        for index in range(count)
    ]
    curve = tuple(
        _rows(
            [0.0, 0.4, 0.7, 1.0],
            [1, 3, 2, 0],
            [200, 90, 9, 0],
            gt=5,
            pixels=1_000_000,
        )
    )
    artifacts = tuple(
        VerifiedEpisodeArtifacts(
            SimpleNamespace(),
            SimpleNamespace(),
            curve if role == COLLECTION_VALIDATION else (),
        )
        for _ in episodes
    )
    path = root / f"{collection_tag}.jsonl"
    manifest_path = root / f"{collection_tag}.collection.json"
    commit_path = root / f"{collection_tag}.commit.json"
    for item in (path, manifest_path, commit_path):
        item.write_text(f"{collection_tag}\n", encoding="ascii")
    collection_sha = _sha(f"{collection_tag}:collection")
    return make_verified_collection(
        path=path,
        manifest_path=manifest_path,
        commit_path=commit_path,
        episodes=episodes,
        artifacts=artifacts,
        collection_sha256=collection_sha,
        manifest_sha256=_sha(f"{collection_tag}:manifest"),
        commit_sha256=_sha(f"{collection_tag}:commit"),
        manifest={
            "collection_role": role,
            "outer_fold_id": "outer_leave_nuaa_sirst",
            "outer_target": "NUAA-SIRST",
            "base_seed": 42,
            "ordered_record_sha256": _sha(f"{collection_tag}:ordered-records"),
            "records": [
                {"record_sha256": _sha(f"{collection_tag}:record:{index}")}
                for index in range(count)
            ],
        },
    )


def test_complete_verified_source_bridge_binds_exact_training_standardizer(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    train = _verified_collection(root, COLLECTION_TRAIN, collection_tag="train-a")
    validation = _verified_collection(
        root,
        COLLECTION_VALIDATION,
        collection_tag="validation",
    )
    standardizer = fit_stage2_context_standardizer(train)
    replay = make_stage2_trainer_replay_capability(
        train,
        validation,
        standardizer,
    )
    reference = build_source_threshold_reference_from_verified_collection(
        validation,
        standardizer,
        replay,
        repository_root=root,
    )
    assert reference["source_domains"] == ["IRSTD-1K", "NUDT-SIRST"]
    assert reference["collection_binding"]["collection_identity_sha256"] == (
        validation.manifest["ordered_record_sha256"]
    )
    assert reference["standardizer_binding"] == {
        "fit_manifest_sha256": standardizer.fit_manifest_sha256,
        "train_collection_sha256": train.collection_sha256,
    }

    other_train = _verified_collection(
        root,
        COLLECTION_TRAIN,
        collection_tag="train-b",
    )
    other_standardizer = fit_stage2_context_standardizer(other_train)
    with pytest.raises(Stage2ThresholdFamilyError, match="standardizer fit mismatch"):
        build_source_threshold_reference_from_verified_collection(
            validation,
            other_standardizer,
            replay,
            repository_root=root,
        )

    class ForgedTransformer:
        def transform(self, values: np.ndarray) -> np.ndarray:
            return values

    with pytest.raises(TypeError, match="standardizer capability"):
        build_source_threshold_reference_from_verified_collection(
            validation,
            ForgedTransformer(),
            replay,
            repository_root=root,
        )

    standardizer.mean.setflags(write=True)
    standardizer.mean[0] += 1.0
    standardizer.mean.setflags(write=False)
    with pytest.raises(
        Stage2ThresholdFamilyError,
        match="transform state/fit manifest mismatch",
    ):
        build_source_threshold_reference_from_verified_collection(
            validation,
            standardizer,
            replay,
            repository_root=root,
        )

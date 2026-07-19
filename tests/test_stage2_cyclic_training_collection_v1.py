from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from model.endpoint_aware_threshold import encode_probability_numpy
from rc.stage2_compositional_curve_provider import build_per_image_exact_event_curve
from rc.stage2_cyclic_training_collection_v1 import (
    ARRAY_FILENAMES,
    EPISODES_FILENAME,
    Stage2CyclicTrainingCollectionError,
    build_synthetic_cyclic_source_role_material,
    publish_cyclic_training_collection_v1,
    verify_cyclic_training_collection_v1,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _role(domain: str, fold: int):
    identities = tuple(_digest(f"{domain}-{fold}-{index}") for index in range(42))
    features = np.arange(42 * 93, dtype=np.float32).reshape(42, 93)
    features = features + np.float32(fold + (1 if domain == "IRSTD-1K" else 0))
    anchor_row = encode_probability_numpy(
        np.asarray([0.2, 0.5, 0.8], dtype=np.float64)
    )
    anchors = np.repeat(anchor_row[None, :], 42, axis=0)
    curves = tuple(
        build_per_image_exact_event_curve(
            image_identity_sha256=identity,
            thresholds=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
            false_positive_pixels=np.asarray([100, 0, 0], dtype=np.int64),
            matched_objects=np.asarray([1, 1, 0], dtype=np.int64),
            total_native_pixels=1000,
            ground_truth_objects=1,
        )
        for identity in identities
    )
    keys = (
        "score_attestation_sha256",
        "score_manifest_metadata_sha256",
        "score_records_content_sha256",
        "run_complete_identity_sha256",
        "run_complete_artifact_sha256",
        "cyclic_context_collection_sha256",
        "seed_manifest_sha256",
        "statistics_config_sha256",
        "source_reference_sha256",
        "source_release_sha256",
    )
    bindings = {key: _digest(f"{domain}-{fold}-{key}") for key in keys}
    bindings["statistics_config_sha256"] = _digest("shared-statistics-config")
    bindings["source_release_sha256"] = _digest("shared-source-release")
    return build_synthetic_cyclic_source_role_material(
        outer_fold_id="outer_leave_nuaa_sirst",
        source_domain=domain,
        oof_fold=fold,
        image_identities=identities,
        context_features=features,
        anchor_coordinates=anchors,
        per_image_curves=curves,
        upstream_bindings=bindings,
    )


@pytest.fixture()
def collection(tmp_path: Path):
    return publish_cyclic_training_collection_v1(
        tmp_path / "collection",
        [_role(domain, fold) for domain in ("NUDT-SIRST", "IRSTD-1K") for fold in (0, 1)],
    )


def test_publish_verify_mmap_identity_only_and_live_q28(collection) -> None:
    assert collection.artifact_scope == "synthetic_cpu_contract_test"
    assert collection.manifest["episode_count"] == 168
    assert collection.manifest["aggregate_curve_materialization"] is False
    assert all(isinstance(value, np.memmap) for value in collection.arrays.values())
    assert not any("aggregate" in name for name in ARRAY_FILENAMES.values())
    first_line = (collection.path / EPISODES_FILENAME).read_text().splitlines()[0]
    assert "context_features" not in first_line
    assert "curve_thresholds" not in first_line
    assert "false_positive" not in first_line
    episode = collection.episode_for_domain("NUDT-SIRST", 0)
    assert len(episode["ordered_context_image_identity_sha256"]) == 14
    assert len(episode["ordered_query_image_identity_sha256"]) == 28
    provider = collection.provider_for_episode("NUDT-SIRST", 0)
    assert len(provider.curves) == 28
    assert provider.select_exact_oracle_rows().coordinates.shape == (3,)
    mean, scale = collection.fit_training_standardizer()
    assert mean.shape == scale.shape == (93,)
    assert np.all(scale >= 1e-8)


def test_external_commit_sha_is_mandatory(collection) -> None:
    with pytest.raises(Stage2CyclicTrainingCollectionError, match="external commit"):
        verify_cyclic_training_collection_v1(collection.path, "0" * 64)


def test_rejects_target_domain() -> None:
    with pytest.raises(Stage2CyclicTrainingCollectionError, match="source-only"):
        _role("NUAA-SIRST", 0)

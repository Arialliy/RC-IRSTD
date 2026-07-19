from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from model.endpoint_aware_threshold import encode_probability_numpy
from rc.stage2_calibrator_generation_v2 import INPUT_BINDING_NAMES
from rc.stage2_compositional_curve_provider import build_per_image_exact_event_curve
from rc.stage2_cyclic_training_collection_v1 import (
    build_synthetic_cyclic_source_role_material,
    publish_cyclic_training_collection_v1,
)
from rc.stage2_source_validation_views import (
    build_synthetic_source_validation_cyclic_selection_view,
    build_synthetic_source_variable_query_sanity_view,
    source_validation_collection_identity_sha256,
)
from rc.train_stage2_rc5_cyclic import (
    Stage2RC5CyclicTrainerError,
    build_synthetic_rc5_training_execution_spec,
    train_stage2_rc5_cyclic,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _curve(identity: str, total: int = 100_000):
    return build_per_image_exact_event_curve(
        image_identity_sha256=identity,
        thresholds=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        false_positive_pixels=np.asarray([10, 0, 0], dtype=np.int64),
        matched_objects=np.asarray([1, 1, 0], dtype=np.int64),
        total_native_pixels=total,
        ground_truth_objects=1,
    )


def _training_role(domain: str, fold: int):
    identities = tuple(_sha(f"train-{domain}-{fold}-{index}") for index in range(42))
    generator = np.random.default_rng(100 + fold + (10 if domain == "IRSTD-1K" else 0))
    features = generator.normal(size=(42, 93)).astype(np.float32)
    anchor = encode_probability_numpy(np.asarray([0.2, 0.5, 0.8], dtype=np.float64))
    keys = ("score_attestation_sha256", "score_manifest_metadata_sha256",
            "score_records_content_sha256", "run_complete_identity_sha256",
            "run_complete_artifact_sha256", "statistics_config_sha256",
            "cyclic_context_collection_sha256",
            "seed_manifest_sha256",
            "source_reference_sha256", "source_release_sha256")
    bindings = {key: _sha(f"{domain}-{fold}-{key}") for key in keys}
    bindings["statistics_config_sha256"] = _sha("shared-statistics-config")
    bindings["source_release_sha256"] = _sha("shared-source-release")
    return build_synthetic_cyclic_source_role_material(
        outer_fold_id="outer_leave_nuaa_sirst", source_domain=domain, oof_fold=fold,
        image_identities=identities, context_features=features,
        anchor_coordinates=np.repeat(anchor[None, :], 42, axis=0),
        per_image_curves=tuple(_curve(identity) for identity in identities),
        upstream_bindings=bindings)


def _inputs(tmp_path: Path):
    collection = publish_cyclic_training_collection_v1(
        tmp_path / "collection",
        [_training_role(domain, fold)
         for domain in ("NUDT-SIRST", "IRSTD-1K") for fold in (0, 1)])
    anchor = encode_probability_numpy(np.asarray([0.2, 0.5, 0.8], dtype=np.float64))
    validation = {}
    for domain in ("NUDT-SIRST", "IRSTD-1K"):
        identities = tuple(_sha(f"validation-{domain}-{index}") for index in range(42))
        validation[domain] = {
            "image_identities": identities,
            "context_features": np.zeros((42, 93), dtype=np.float32),
            "anchor_coordinates": np.repeat(anchor[None, :], 42, axis=0),
            "per_image_curves": tuple(_curve(identity) for identity in identities),
        }
    selection = build_synthetic_source_validation_cyclic_selection_view(
        outer_fold_id="outer_leave_nuaa_sirst", domain_materials=validation)
    sanity_rows = [{
        "source_domain": domain, "query_size": 43 + index,
        "context_features": np.zeros(93, dtype=np.float32),
        "anchor_coordinates": anchor,
        "aggregate_curve": _curve(_sha(f"sanity-{domain}"), total=4_300_000),
    } for index, domain in enumerate(("NUDT-SIRST", "IRSTD-1K"))]
    sanity = build_synthetic_source_variable_query_sanity_view(
        outer_fold_id="outer_leave_nuaa_sirst", rows=sanity_rows)
    return collection, selection, sanity


def _bindings(collection, selection, sanity, spec):
    actual = collection.manifest["actual_input_binding_identities"]
    digests = {
        "rc5_config": spec.training_contract_sha256,
        "training_collection": collection.commit_sha256,
        "validation_collection": source_validation_collection_identity_sha256(
            selection, sanity),
        "statistics_config": actual["statistics_config"],
        "source_reference": actual["source_reference"],
        "per_image_curve_bank": collection.curve_bank_id,
        "detector_run_complete_set": actual["detector_run_complete_set"],
        "seed_manifest": actual["seed_manifest"],
        "source_release": actual["source_release"],
    }
    result = {name: {"path": f"inputs/{name}.bin", "sha256": digests[name]}
              for name in INPUT_BINDING_NAMES}
    result["training_collection"]["path"] = (
        f"{collection.path.name}/COLLECTION_COMMIT.json"
    )
    return result


def _assert_tree_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right)
        for key in left: _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)) and len(left) == len(right)
        for l_item, r_item in zip(left, right, strict=True):
            _assert_tree_equal(l_item, r_item)
    else:
        assert left == right


def test_t6_t7_t8_publish_generation_and_mandatory_sanity(tmp_path: Path) -> None:
    collection, selection, sanity = _inputs(tmp_path)
    spec = build_synthetic_rc5_training_execution_spec(max_epochs=1)
    bindings = _bindings(collection, selection, sanity, spec)
    for method in ("T6", "T7", "T8"):
        outcome = train_stage2_rc5_cyclic(
            method=method, collection=collection, selection_view=selection,
            sanity_view=sanity, execution_spec=spec,
            run_root=tmp_path / f"run-{method}", run_id=f"synthetic-{method}",
            base_seed=42, derived_seed=12345, input_bindings=bindings)
        assert outcome.run is not None and len(outcome.generations) == 1
        row = outcome.history[0]
        assert row["source_variable_query_sanity"]["excluded_from_epoch_ranking"] is True
        assert row["selection_record"]["schema_version"].endswith(".v2")


def test_stale_curve_bank_binding_fails_before_run_artifact_creation(
    tmp_path: Path,
) -> None:
    collection, selection, sanity = _inputs(tmp_path)
    spec = build_synthetic_rc5_training_execution_spec(max_epochs=1)
    bindings = {
        name: dict(binding)
        for name, binding in _bindings(collection, selection, sanity, spec).items()
    }
    bindings["per_image_curve_bank"]["sha256"] = _sha("stale-curve-bank")
    run_root = tmp_path / "invalid-binding-run"

    with pytest.raises(
        Stage2RC5CyclicTrainerError,
        match=r"input_bindings\.per_image_curve_bank.*verified capability",
    ):
        train_stage2_rc5_cyclic(
            method="T8",
            collection=collection,
            selection_view=selection,
            sanity_view=sanity,
            execution_spec=spec,
            run_root=run_root,
            run_id="synthetic-invalid-binding",
            base_seed=42,
            derived_seed=12345,
            input_bindings=bindings,
        )
    assert not run_root.exists()


def test_two_epoch_uninterrupted_equals_external_commit_resume(tmp_path: Path) -> None:
    collection, selection, sanity = _inputs(tmp_path)
    spec = build_synthetic_rc5_training_execution_spec(max_epochs=2)
    bindings = _bindings(collection, selection, sanity, spec)
    common = dict(method="T8", collection=collection, selection_view=selection,
                  sanity_view=sanity, execution_spec=spec, run_id="synthetic-T8-equality",
                  base_seed=42, derived_seed=90210, input_bindings=bindings)
    uninterrupted = train_stage2_rc5_cyclic(
        **common, run_root=tmp_path / "uninterrupted")
    interrupted = train_stage2_rc5_cyclic(
        **common, run_root=tmp_path / "resumed", synthetic_interrupt_after_epoch=0)
    assert interrupted.run is None and interrupted.interrupted_after_epoch == 0
    generation0 = interrupted.generations[-1]
    resumed = train_stage2_rc5_cyclic(
        **common, run_root=tmp_path / "resumed",
        resume_generation_path=generation0.path,
        resume_generation_commit_sha256=generation0.commit_sha256)
    assert uninterrupted.run is not None and resumed.run is not None
    left = uninterrupted.generations[-1].resume_state
    right = resumed.generations[-1].resume_state
    _assert_tree_equal(dict(left["model_state_dict"]), dict(right["model_state_dict"]))
    _assert_tree_equal(dict(left["optimizer_state_dict"]), dict(right["optimizer_state_dict"]))
    _assert_tree_equal(left["python_rng_state"], right["python_rng_state"])
    _assert_tree_equal(left["numpy_rng_state"], right["numpy_rng_state"])
    _assert_tree_equal(left["torch_cpu_rng_state"], right["torch_cpu_rng_state"])
    _assert_tree_equal(left["dataloader_rng_state"], right["dataloader_rng_state"])
    for left_row, right_row in zip(uninterrupted.history, resumed.history, strict=True):
        assert left_row["sampler_ordered_selection_sha256"] == right_row[
            "sampler_ordered_selection_sha256"]
        assert left_row["mean_loss_hex"] == right_row["mean_loss_hex"]
        assert left_row["selection_record"] == right_row["selection_record"]
    assert uninterrupted.run.payload["selected_generation"]["epoch"] == \
        resumed.run.payload["selected_generation"]["epoch"]

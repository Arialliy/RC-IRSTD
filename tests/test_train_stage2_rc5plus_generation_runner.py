from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rc.stage2_rc5_feature_mask import build_stage2_rc5_feature_mask
from rc.stage2_rc5plus_calibrator_generation_v3 import (
    INPUT_BINDING_NAMES,
    build_selection_record_v3,
)
from rc.stage2_rc5plus_frozen_config import (
    verify_stage2_rc5plus_frozen_config_file,
)
from rc.stage2_rc5plus_training_core import RC5PLUS_LOSS_METRIC_NAMES
import rc.train_stage2_rc5plus_cyclic as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/aaai27_stage2_crossfit_rc5plus_v1.json"
SHA = {
    "view": "1" * 64,
    "selection": "2" * 64,
    "sanity": "3" * 64,
    "source_reference": "4" * 64,
    "curve_bank": "5" * 64,
    "detector_runs": "6" * 64,
    "seed_manifest": "7" * 64,
    "source_release": "8" * 64,
}
BOUNDARY_FIELDS = (
    "canonical_id",
    "original_image_sha256",
    "near_duplicate_cluster_id_or_unique_sentinel",
    "exclusion_group_id",
)


def _boundaries(prefix: str) -> dict[str, frozenset[str]]:
    return {
        field: frozenset({f"{prefix}-{field}"}) for field in BOUNDARY_FIELDS
    }


class _TrainView:
    artifact_scope = "synthetic_cpu_contract_test"
    view_identity_sha256 = SHA["view"]
    curve_bank_id = SHA["curve_bank"]
    boundary_values = _boundaries("train")
    domain_episode_indices = {
        "NUDT-SIRST": tuple(range(84)),
        "IRSTD-1K": tuple(range(84)),
    }
    manifest = {
        "outer_fold_id": "outer_leave_nuaa_sirst",
        "outer_target": "NUAA-SIRST",
        "source_domains": ["NUDT-SIRST", "IRSTD-1K"],
        "actual_input_binding_identities": {
            "source_reference": SHA["source_reference"],
            "per_image_curve_bank": SHA["curve_bank"],
            "detector_run_complete_set": SHA["detector_runs"],
            "seed_manifest": SHA["seed_manifest"],
            "source_release": SHA["source_release"],
        },
    }

    @staticmethod
    def fit_training_standardizer() -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros(93, dtype=np.float64),
            np.ones(93, dtype=np.float64),
        )


def _validation_views():
    selection_base = SimpleNamespace(
        outer_fold_id="outer_leave_nuaa_sirst",
        boundary_values=_boundaries("validation"),
        upstream_bindings={},
    )
    sanity_base = SimpleNamespace(
        outer_fold_id="outer_leave_nuaa_sirst",
        boundary_values=_boundaries("validation"),
        upstream_bindings={},
    )
    selection = SimpleNamespace(
        base_view=selection_base,
        artifact_scope="synthetic_cpu_contract_test",
        identity_sha256=SHA["selection"],
    )
    sanity = SimpleNamespace(
        base_view=sanity_base,
        artifact_scope="synthetic_cpu_contract_test",
        identity_sha256=SHA["sanity"],
    )
    return selection, sanity


def _patch_capabilities_and_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "assert_verified_stage2_rc5plus_cyclic_training_view",
        lambda value: value,
    )
    monkeypatch.setattr(
        runner,
        "assert_verified_stage2_rc5plus_source_validation_view",
        lambda value: value,
    )
    monkeypatch.setattr(
        runner,
        "assert_verified_stage2_rc5plus_variable_query_sanity_view",
        lambda value: value,
    )
    payload = {
        "epoch_size": 2,
        "ordered_selection": [
            {"source_domain": "NUDT-SIRST", "domain_episode_index": 0},
            {"source_domain": "IRSTD-1K", "domain_episode_index": 0},
        ],
        "ordered_selection_sha256": "9" * 64,
    }
    monkeypatch.setattr(
        runner, "build_domain_balanced_cyclic_epoch", lambda **_kwargs: payload
    )
    monkeypatch.setattr(
        runner,
        "verify_domain_balanced_cyclic_epoch",
        lambda value: SimpleNamespace(payload=value),
    )
    monkeypatch.setattr(
        runner,
        "assert_verified_domain_balanced_cyclic_epoch",
        lambda value: value,
    )
    monkeypatch.setattr(
        runner, "collate_rc5plus_cyclic_batch", lambda **_kwargs: {}
    )

    def deterministic_step(*, model, optimizer, **_kwargs):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(parameter.square().mean() for parameter in model.parameters())
        loss.backward()
        optimizer.step()
        detached = loss.detach()
        losses = {
            name: detached.clone() for name in RC5PLUS_LOSS_METRIC_NAMES
        }
        return None, losses

    monkeypatch.setattr(
        runner, "rc5plus_cyclic_optimization_step", deterministic_step
    )
    selection_record = build_selection_record_v3(
        macro_source_bsr=0.5,
        macro_source_log_excess=0.25,
        macro_source_pd=0.75,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_stage2_rc5plus_source_validation_view",
        lambda **_kwargs: {
            "selection_record": selection_record,
            "selection_geometry": selection_record["selection_geometry"],
            "domain_metrics": {"synthetic": {"checked": True}},
        },
    )
    monkeypatch.setattr(
        runner,
        "evaluate_stage2_rc5plus_variable_query_sanity_view",
        lambda **_kwargs: {
            "excluded_from_epoch_ranking": True,
            "selection_record_present": False,
            "all_records_consumed_once": True,
        },
    )


def _bindings(spec, train, selection, sanity, mask):
    mean, scale = train.fit_training_standardizer()
    validation_identity = runner.rc5plus_source_validation_identity_sha256(
        selection, sanity
    )
    digests = {
        "rc5plus_config": spec.config_source_sha256,
        "training_view": train.view_identity_sha256,
        "source_validation_view": validation_identity,
        "feature_mask": mask.identity_sha256,
        "standardizer": runner.rc5plus_standardizer_identity_sha256(
            mean, scale
        ),
        "source_reference": SHA["source_reference"],
        "per_image_curve_bank": SHA["curve_bank"],
        "detector_run_complete_set": SHA["detector_runs"],
        "seed_manifest": SHA["seed_manifest"],
        "source_release": SHA["source_release"],
    }
    assert set(digests) == set(INPUT_BINDING_NAMES)
    return {
        name: {"path": f"inputs/{name}.json", "sha256": digests[name]}
        for name in INPUT_BINDING_NAMES
    }


def _run_kwargs(tmp_path, monkeypatch):
    _patch_capabilities_and_epoch(monkeypatch)
    train = _TrainView()
    selection, sanity = _validation_views()
    mask = build_stage2_rc5_feature_mask("C3")
    spec = runner.build_synthetic_rc5plus_training_execution_spec(
        max_epochs=2
    )
    return {
        "method": "T7_PLUS",
        "collection": train,
        "selection_view": selection,
        "sanity_view": sanity,
        "feature_mask": mask,
        "execution_spec": spec,
        "run_id": "synthetic-resume-equivalence",
        "base_seed": 42,
        "derived_seed": 123456,
        "input_bindings": _bindings(
            spec, train, selection, sanity, mask
        ),
        "device": "cpu",
    }


def _assert_tree_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
        return
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert tuple(left) == tuple(right)
        for key in left:
            _assert_tree_equal(left[key], right[key])
        return
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
    ):
        assert isinstance(right, Sequence)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_tree_equal(left_item, right_item)
        return
    assert left == right


def test_production_execution_spec_is_bound_to_unique_config_file() -> None:
    config = verify_stage2_rc5plus_frozen_config_file(CONFIG)
    spec = runner.build_rc5plus_training_execution_spec_from_verified_config(
        config
    )
    assert spec.artifact_scope == "production"
    assert spec.training_contract_sha256 == config.canonical_sha256
    assert spec.config_source_sha256 == config.source_bytes_sha256
    assert spec.payload["loss"]["risk_epsilon"] == 1e-12
    assert spec.payload["optimizer"]["data_iteration"] == (
        "custom_loop_verified_sampler_no_dataloader"
    )


def test_interrupted_resume_matches_uninterrupted_generation_v3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _run_kwargs(tmp_path, monkeypatch)
    uninterrupted = runner.train_stage2_rc5plus_cyclic(
        **kwargs, run_root=tmp_path / "uninterrupted"
    )
    interrupted = runner.train_stage2_rc5plus_cyclic(
        **kwargs,
        run_root=tmp_path / "resumed",
        synthetic_interrupt_after_epoch=0,
    )
    assert interrupted.run is None
    assert interrupted.interrupted_after_epoch == 0
    resumed = runner.train_stage2_rc5plus_cyclic(
        **kwargs,
        run_root=tmp_path / "resumed",
        resume_generation_path=interrupted.generations[-1].path,
        resume_generation_commit_sha256=(
            interrupted.generations[-1].commit_sha256
        ),
    )
    assert uninterrupted.run is not None
    assert resumed.run is not None
    left = uninterrupted.generations[-1]
    right = resumed.generations[-1]
    assert left.deployment_checkpoint.sha256 == right.deployment_checkpoint.sha256
    for field in (
        "model_state_dict",
        "optimizer_state_dict",
        "history",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "dataloader_rng_state",
    ):
        _assert_tree_equal(left.resume_state[field], right.resume_state[field])
    assert resumed.run.payload["selected_generation"]["epoch"] == (
        uninterrupted.run.payload["selected_generation"]["epoch"]
    )
    assert resumed.run.selected_generation.deployment_checkpoint.sha256 == (
        uninterrupted.run.selected_generation.deployment_checkpoint.sha256
    )


def test_wrong_verified_input_sha_fails_before_run_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _run_kwargs(tmp_path, monkeypatch)
    bad = deepcopy(kwargs["input_bindings"])
    bad["feature_mask"]["sha256"] = "f" * 64
    run_root = tmp_path / "must-not-exist"
    with pytest.raises(
        runner.Stage2RC5PlusCyclicTrainerError,
        match=r"input_bindings\.feature_mask",
    ):
        runner.train_stage2_rc5plus_cyclic(
            **{**kwargs, "input_bindings": bad}, run_root=run_root
        )
    assert not run_root.exists()

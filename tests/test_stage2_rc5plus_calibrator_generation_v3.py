from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from model.budget_conditioned_residual_transport_calibrator import (
    BudgetConditionedMonotoneNoTargetAnchorCalibrator,
    BudgetConditionedMonotoneResidualTransportCalibrator,
)
from model.endpoint_aware_threshold import encode_probability_numpy
from rc.stage2_calibrator_checkpoint_v8 import (
    make_calibrator_checkpoint_v8,
    serialize_calibrator_checkpoint_v8,
)
from rc.stage2_rc5plus_calibrator_generation_v3 import (
    COMMIT_FILENAME,
    INPUT_BINDING_NAMES,
    RUN_COMMIT_FILENAME,
    Stage2RC5PlusCalibratorGenerationV3Error,
    VerifiedRC5PlusCalibratorGenerationV3,
    build_resume_state_v3,
    build_selection_record_v3,
    normalize_input_bindings_v3,
    publish_rc5plus_calibrator_generation_v3,
    publish_rc5plus_calibrator_run_v3,
    verify_rc5plus_calibrator_generation_v3,
    verify_rc5plus_calibrator_run_v3,
)


TRAINING_SHA = hashlib.sha256(b"rc5plus-generation-v3-training").hexdigest()
VIEW_SHA = hashlib.sha256(b"rc5plus-generation-v3-view").hexdigest()


def _bindings() -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": f"inputs/{name}.json",
            "sha256": hashlib.sha256(name.encode("ascii")).hexdigest(),
        }
        for name in INPUT_BINDING_NAMES
    }


def _model(method: str = "T8_PLUS") -> torch.nn.Module:
    model_type = (
        BudgetConditionedMonotoneNoTargetAnchorCalibrator
        if method == "T8_PLUS_NO_ANCHOR"
        else BudgetConditionedMonotoneResidualTransportCalibrator
    )
    return model_type(
        context_feature_dim=93,
        hidden_dims=(32,),
        dropout=0.1,
        minimum_residual_increment=1e-6,
    )


def _checkpoint(model: torch.nn.Module, method: str = "T8_PLUS") -> bytes:
    payload = make_calibrator_checkpoint_v8(
        method=method,
        model=model,
        standardizer_mean=np.linspace(-0.5, 0.5, 93, dtype=np.float64),
        standardizer_scale=np.linspace(0.5, 2.0, 93, dtype=np.float64),
        training_contract_sha256=TRAINING_SHA,
        training_view_identity_sha256=VIEW_SHA,
    )
    return serialize_calibrator_checkpoint_v8(payload)


def _selection(
    *, bsr: float = 0.5, log_excess: float = 0.2, pd: float = 0.8
) -> dict[str, object]:
    return build_selection_record_v3(
        macro_source_bsr=bsr,
        macro_source_log_excess=log_excess,
        macro_source_pd=pd,
    )


def _resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    method: str = "T8_PLUS",
    bsr: float = 0.5,
    log_excess: float = 0.2,
    pd: float = 0.8,
    torch_rng_state: torch.Tensor | None = None,
    dataloader_rng_state: torch.Tensor | None = None,
) -> dict[str, object]:
    loader_state = (
        torch.Generator().manual_seed(301).get_state()
        if dataloader_rng_state is None
        else dataloader_rng_state
    )
    return build_resume_state_v3(
        method=method,
        run_id=f"outer_leave_nuaa_sirst__s42__{method.lower()}",
        outer_fold_id="outer_leave_nuaa_sirst",
        outer_target_domain="NUAA-SIRST",
        base_seed=42,
        derived_seed=420031,
        epoch=epoch,
        process_rank=0,
        world_size=1,
        training_contract_sha256=TRAINING_SHA,
        training_view_identity_sha256=VIEW_SHA,
        input_bindings=_bindings(),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        history=[
            {"epoch": index, "loss_hex": float(1.0 / (index + 1)).hex()}
            for index in range(epoch + 1)
        ],
        selection_record=_selection(
            bsr=bsr, log_excess=log_excess, pd=pd
        ),
        python_rng_state={"version": 3, "state": [1, 2, 3], "gauss": None},
        numpy_rng_state={"bit_generator": "PCG64", "state": {"state": 1}},
        torch_cpu_rng_state=(
            torch.get_rng_state() if torch_rng_state is None else torch_rng_state
        ),
        torch_cuda_rng_states=[],
        dataloader_rng_state=loader_state,
    )


def _publish(
    root: Path,
    epoch: int,
    *,
    method: str = "T8_PLUS",
    bsr: float = 0.5,
    log_excess: float = 0.2,
    pd: float = 0.8,
) -> VerifiedRC5PlusCalibratorGenerationV3:
    torch.manual_seed(9000 + epoch)
    model = _model(method)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return publish_rc5plus_calibrator_generation_v3(
        root,
        resume_state=_resume(
            model,
            optimizer,
            epoch=epoch,
            method=method,
            bsr=bsr,
            log_excess=log_excess,
            pd=pd,
        ),
        deployment_checkpoint_bytes=_checkpoint(model, method),
        input_bindings=_bindings(),
    )


@pytest.mark.parametrize("method", ("T8_PLUS", "T8_PLUS_NO_ANCHOR"))
def test_generation_v3_round_trip_binds_checkpoint_v8_and_resume_state(
    tmp_path: Path, method: str
) -> None:
    verified = _publish(tmp_path / method, 0, method=method)
    assert verified.deployment_checkpoint.method == method
    assert verified.manifest["deployment_checkpoint"]["schema_version"] == (
        "rc-irstd.calibrator.v8"
    )
    assert verified.manifest["training_view_identity_sha256"] == VIEW_SHA
    assert verified.manifest["nonprimary_budget_epoch_rescue"] is False
    assert verified.resume_state["official_test_accessed"] is False
    replayed = verify_rc5plus_calibrator_generation_v3(
        verified.path, verified.commit_sha256
    )
    assert replayed.commit_sha256 == verified.commit_sha256
    with pytest.raises(FileExistsError, match=r"immutable RC5\+"):
        _publish(tmp_path / method, 0, method=method)
    with pytest.raises(TypeError, match="verifier-issued"):
        VerifiedRC5PlusCalibratorGenerationV3()


def _training_step(
    model: BudgetConditionedMonotoneResidualTransportCalibrator,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    anchors: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    output = model(features, anchor_coordinates=anchors)
    loss = output.grid_raw_coordinates.square().mean()
    loss.backward()
    optimizer.step()


def test_interrupted_resume_matches_uninterrupted_next_dropout_step(
    tmp_path: Path,
) -> None:
    torch.manual_seed(811)
    model = _model("T8_PLUS")
    assert isinstance(model, BudgetConditionedMonotoneResidualTransportCalibrator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    features = torch.linspace(-1.0, 1.0, 3 * 93).reshape(3, 93)
    anchor_probability = np.tile(np.linspace(0.1, 0.9, 9), (3, 1))
    anchors = torch.from_numpy(encode_probability_numpy(anchor_probability))
    _training_step(model, optimizer, features, anchors)
    saved_torch_rng = torch.get_rng_state().clone()
    saved_loader_rng = torch.Generator().manual_seed(501).get_state()
    state = _resume(
        model,
        optimizer,
        epoch=0,
        torch_rng_state=saved_torch_rng,
        dataloader_rng_state=saved_loader_rng,
    )
    generation = publish_rc5plus_calibrator_generation_v3(
        tmp_path / "resume-equivalence",
        resume_state=state,
        deployment_checkpoint_bytes=_checkpoint(model),
        input_bindings=_bindings(),
    )

    torch.set_rng_state(saved_torch_rng.clone())
    _training_step(model, optimizer, features, anchors)
    uninterrupted = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }

    resumed = generation.deployment_checkpoint.model()
    assert isinstance(resumed, BudgetConditionedMonotoneResidualTransportCalibrator)
    resumed.train()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(generation.resume_state["optimizer_state_dict"])
    torch.set_rng_state(generation.resume_state["torch_cpu_rng_state"].clone())
    _training_step(resumed, resumed_optimizer, features, anchors)
    assert all(
        torch.equal(tensor, resumed.state_dict()[name])
        for name, tensor in uninterrupted.items()
    )


def test_generation_v3_rejects_checkpoint_state_view_and_method_mismatch(
    tmp_path: Path,
) -> None:
    torch.manual_seed(17)
    resume_model = _model("T8_PLUS")
    optimizer = torch.optim.AdamW(resume_model.parameters(), lr=1e-3)
    different_model = _model("T8_PLUS")
    with torch.no_grad():
        next(different_model.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="model-state digests differ"):
        publish_rc5plus_calibrator_generation_v3(
            tmp_path / "state-mismatch",
            resume_state=_resume(resume_model, optimizer, epoch=0),
            deployment_checkpoint_bytes=_checkpoint(different_model),
            input_bindings=_bindings(),
        )

    no_anchor = _model("T8_PLUS_NO_ANCHOR")
    with pytest.raises(ValueError, match="method"):
        publish_rc5plus_calibrator_generation_v3(
            tmp_path / "method-mismatch",
            resume_state=_resume(resume_model, optimizer, epoch=0),
            deployment_checkpoint_bytes=_checkpoint(
                no_anchor, "T8_PLUS_NO_ANCHOR"
            ),
            input_bindings=_bindings(),
        )


def test_generation_v3_external_hash_member_and_symlink_fail_closed(
    tmp_path: Path,
) -> None:
    first = _publish(tmp_path / "hash", 0)
    with pytest.raises(ValueError, match="external commit SHA"):
        verify_rc5plus_calibrator_generation_v3(first.path, "0" * 64)
    resume_path = first.path / "resume_state_v3.pt"
    resume_path.chmod(0o644)
    resume_path.write_bytes(resume_path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="resume-state SHA"):
        verify_rc5plus_calibrator_generation_v3(
            first.path, first.commit_sha256
        )

    second = _publish(tmp_path / "symlink", 0)
    resume_path = second.path / "resume_state_v3.pt"
    target = second.path / "resume_state_v3.target.pt"
    resume_path.rename(target)
    resume_path.symlink_to(target.name)
    with pytest.raises(ValueError, match="symlink"):
        verify_rc5plus_calibrator_generation_v3(
            second.path, second.commit_sha256
        )


def test_resume_and_selection_contracts_reject_target_access_and_drift(
    tmp_path: Path,
) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = _resume(model, optimizer, epoch=0)
    state["outer_target_accessed"] = True
    with pytest.raises(ValueError, match="outer_target_accessed"):
        publish_rc5plus_calibrator_generation_v3(
            tmp_path / "unused-after-state-rejection",
            resume_state=state,
            deployment_checkpoint_bytes=_checkpoint(model),
            input_bindings=_bindings(),
        )
    injected = _bindings()
    injected["target_labels"] = {
        "path": "inputs/target_labels.json",
        "sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="key closure"):
        normalize_input_bindings_v3(injected)
    record = _selection()
    record["nonprimary_budgets_can_rescue_epoch_selection"] = True
    with pytest.raises(ValueError, match="contract drifted"):
        build_resume_state_v3(
            **{
                **{
                    key: value
                    for key, value in _resume(model, optimizer, epoch=0).items()
                    if key
                    not in {
                        "format_version",
                        "input_identity_sha256",
                        "official_test_accessed",
                        "outer_target_accessed",
                        "query_labels_accessed",
                    }
                },
                "input_bindings": _bindings(),
                "selection_record": record,
            }
        )


def test_run_v3_recomputes_primary_rank_and_earliest_exact_tie(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    metrics = (
        (0.50, 0.30, 0.80),
        (0.60, 0.40, 0.70),
        (0.60, 0.20, 0.60),
        (0.60, 0.20, 0.90),
        (0.60, 0.20, 0.90),
    )
    generations = [
        _publish(
            root,
            epoch,
            bsr=bsr,
            log_excess=log_excess,
            pd=pd,
        )
        for epoch, (bsr, log_excess, pd) in enumerate(metrics)
    ]
    run = publish_rc5plus_calibrator_run_v3(root, generations)
    assert run.selected_generation.manifest["epoch"] == 3
    assert run.payload["selection_budget"] == {
        "numerator": 1,
        "denominator": 100_000,
        "grid_index": 4,
    }
    assert run.payload["nonprimary_budget_epoch_rescue"] is False
    replayed = verify_rc5plus_calibrator_run_v3(root, run.sha256)
    assert replayed.selected_generation.commit_sha256 == generations[3].commit_sha256
    with pytest.raises(FileExistsError, match="immutable run-v3"):
        publish_rc5plus_calibrator_run_v3(root, generations)

    path = root / RUN_COMMIT_FILENAME
    path.chmod(0o644)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_generation"] = copy.deepcopy(
        payload["generation_inventory"][0]
    )
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="self-hash|selected generation"):
        verify_rc5plus_calibrator_run_v3(
            root, hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_generation_v3_commit_is_written_last(tmp_path: Path) -> None:
    verified = _publish(tmp_path / "mtime", 0)
    commit = verified.path / COMMIT_FILENAME
    members = [
        item
        for item in verified.path.iterdir()
        if item.name != COMMIT_FILENAME
    ]
    assert all(
        commit.stat(follow_symlinks=False).st_mtime_ns
        >= item.stat(follow_symlinks=False).st_mtime_ns
        for item in members
    )
    assert os.stat(commit, follow_symlinks=False).st_nlink == 1

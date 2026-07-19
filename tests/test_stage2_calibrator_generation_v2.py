from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from model.endpoint_aware_pixel_calibrator import (
    MonotoneEndpointAwarePixelCalibrator,
)
from rc.stage2_calibrator_checkpoint_v7 import (
    make_calibrator_checkpoint_v7,
    serialize_calibrator_checkpoint_v7,
)
from rc.stage2_calibrator_generation_v2 import (
    COMMIT_FILENAME,
    INPUT_BINDING_NAMES,
    RUN_COMMIT_FILENAME,
    Stage2CalibratorGenerationV2Error,
    VerifiedCalibratorGenerationV2,
    build_resume_state_v2,
    build_selection_record,
    publish_calibrator_generation_v2,
    publish_calibrator_run_v2,
    verify_calibrator_generation_v2,
    verify_calibrator_run_v2,
)


TRAINING_SHA = hashlib.sha256(b"generation-v2-training-contract").hexdigest()


def _bindings() -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": f"inputs/{name}.json",
            "sha256": hashlib.sha256(name.encode("ascii")).hexdigest(),
        }
        for name in INPUT_BINDING_NAMES
    }


def _model() -> MonotoneEndpointAwarePixelCalibrator:
    return MonotoneEndpointAwarePixelCalibrator(
        context_feature_dim=93,
        pixel_budget_grid=[1e-4, 1e-5, 1e-6],
        hidden_dims=[32],
        dropout=0.1,
        minimum_raw_coordinate_gap=0.001,
    )


def _deployment(model: torch.nn.Module | None = None) -> bytes:
    payload = make_calibrator_checkpoint_v7(
        method="T8",
        model=_model() if model is None else model,
        standardizer_mean=np.zeros(93, dtype=np.float64),
        standardizer_scale=np.ones(93, dtype=np.float64),
        training_contract_sha256=TRAINING_SHA,
    )
    return serialize_calibrator_checkpoint_v7(payload)


def _resume(
    epoch: int,
    *,
    bsr: float,
    log_excess: float,
    pd: float,
    model: torch.nn.Module | None = None,
) -> dict[str, object]:
    current = _model() if model is None else model
    optimizer = torch.optim.Adam(current.parameters(), lr=1e-3)
    return build_resume_state_v2(
        method="T8",
        run_id="outer_leave_nuaa_sirst__s42__t8",
        outer_fold_id="outer_leave_nuaa_sirst",
        outer_target_domain="NUAA-SIRST",
        base_seed=42,
        derived_seed=123456,
        epoch=epoch,
        process_rank=0,
        world_size=1,
        training_contract_sha256=TRAINING_SHA,
        input_bindings=_bindings(),
        model_state_dict=current.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        history=[{"epoch": index, "loss_hex": float(1.0 / (index + 1)).hex()} for index in range(epoch + 1)],
        selection_record=build_selection_record(
            macro_source_bsr=bsr,
            macro_source_log_excess=log_excess,
            macro_source_pd=pd,
        ),
        python_rng_state={"version": 3, "state": [1, 2, 3], "gauss": None},
        numpy_rng_state={"bit_generator": "PCG64", "state": {"state": 1, "inc": 3}},
        torch_cpu_rng_state=torch.get_rng_state(),
        torch_cuda_rng_states=[],
        dataloader_rng_state=torch.Generator().manual_seed(99).get_state(),
    )


def _publish(root: Path, epoch: int, *, bsr: float, log_excess: float, pd: float):
    model = _model()
    return publish_calibrator_generation_v2(
        root,
        resume_state=_resume(
            epoch, bsr=bsr, log_excess=log_excess, pd=pd, model=model
        ),
        deployment_checkpoint_bytes=_deployment(model),
        input_bindings=_bindings(),
    )


def test_generation_round_trip_is_immutable_external_hash_bound_and_weights_only(
    tmp_path: Path,
) -> None:
    verified = _publish(tmp_path / "run", 0, bsr=0.4, log_excess=0.2, pd=0.8)
    assert isinstance(verified, VerifiedCalibratorGenerationV2)
    assert verified.resume_state["epoch"] == 0
    assert verified.deployment_checkpoint.method == "T8"
    replay = torch.load(
        io.BytesIO((verified.path / "resume_state.pt").read_bytes()),
        map_location="cpu",
        weights_only=True,
    )
    assert replay["optimizer_state_dict"] == verified.resume_state["optimizer_state_dict"]
    assert verify_calibrator_generation_v2(
        verified.path, verified.commit_sha256
    ).commit_sha256 == verified.commit_sha256
    with pytest.raises(FileExistsError, match="immutable generation"):
        _publish(tmp_path / "run", 0, bsr=0.4, log_excess=0.2, pd=0.8)


def test_generation_requires_external_commit_hash_and_rejects_member_drift(
    tmp_path: Path,
) -> None:
    verified = _publish(tmp_path / "run", 0, bsr=0.4, log_excess=0.2, pd=0.8)
    with pytest.raises(Stage2CalibratorGenerationV2Error, match="external commit"):
        verify_calibrator_generation_v2(verified.path, "0" * 64)
    state = verified.path / "resume_state.pt"
    state.chmod(0o644)
    state.write_bytes(state.read_bytes() + b"drift")
    with pytest.raises(Stage2CalibratorGenerationV2Error, match="SHA mismatch"):
        verify_calibrator_generation_v2(verified.path, verified.commit_sha256)


def test_generation_rejects_resume_and_deployment_model_state_mismatch(
    tmp_path: Path,
) -> None:
    resume_model = _model()
    deployment_model = _model()
    with torch.no_grad():
        next(deployment_model.parameters()).add_(1.0)
    with pytest.raises(
        Stage2CalibratorGenerationV2Error,
        match="model-state content digests differ",
    ):
        publish_calibrator_generation_v2(
            tmp_path / "run",
            resume_state=_resume(
                0,
                bsr=0.4,
                log_excess=0.2,
                pd=0.8,
                model=resume_model,
            ),
            deployment_checkpoint_bytes=_deployment(deployment_model),
            input_bindings=_bindings(),
        )


def test_generation_rejects_symlink_member_and_forged_capability(tmp_path: Path) -> None:
    verified = _publish(tmp_path / "run", 0, bsr=0.4, log_excess=0.2, pd=0.8)
    state = verified.path / "resume_state.pt"
    target = verified.path / "resume_state.target.pt"
    state.rename(target)
    state.symlink_to(target.name)
    with pytest.raises(Stage2CalibratorGenerationV2Error, match="symlink"):
        verify_calibrator_generation_v2(verified.path, verified.commit_sha256)
    with pytest.raises(TypeError, match="verifier-issued"):
        VerifiedCalibratorGenerationV2()


def test_resume_state_rejects_short_history_nonfinite_and_target_access() -> None:
    state = _resume(1, bsr=0.4, log_excess=0.2, pd=0.8)
    state["history"] = [{"epoch": 0}]
    with pytest.raises(Stage2CalibratorGenerationV2Error, match="history"):
        build_resume_state_v2(
            method="T8",
            run_id="x",
            outer_fold_id="f",
            outer_target_domain="d",
            base_seed=0,
            derived_seed=1,
            epoch=1,
            process_rank=0,
            world_size=1,
            training_contract_sha256=TRAINING_SHA,
            input_bindings=_bindings(),
            model_state_dict=state["model_state_dict"],
            optimizer_state_dict=state["optimizer_state_dict"],
            history=state["history"],
            selection_record=state["selection_record"],
            python_rng_state=state["python_rng_state"],
            numpy_rng_state=state["numpy_rng_state"],
            torch_cpu_rng_state=state["torch_cpu_rng_state"],
            torch_cuda_rng_states=state["torch_cuda_rng_states"],
            dataloader_rng_state=state["dataloader_rng_state"],
        )
    with pytest.raises(Stage2CalibratorGenerationV2Error, match="finite"):
        build_selection_record(
            macro_source_bsr=float("nan"),
            macro_source_log_excess=0.0,
            macro_source_pd=0.0,
        )


def test_run_commit_recomputes_multimetric_rank_and_earliest_tie(tmp_path: Path) -> None:
    root = tmp_path / "run"
    generations = [
        _publish(root, 0, bsr=0.50, log_excess=0.30, pd=0.80),
        _publish(root, 1, bsr=0.60, log_excess=0.40, pd=0.70),
        _publish(root, 2, bsr=0.60, log_excess=0.20, pd=0.60),
        _publish(root, 3, bsr=0.60, log_excess=0.20, pd=0.90),
        _publish(root, 4, bsr=0.60, log_excess=0.20, pd=0.90),
    ]
    run = publish_calibrator_run_v2(root, generations)
    assert run.selected_generation.manifest["epoch"] == 3
    assert run.payload["selected_generation"]["epoch"] == 3
    replay = verify_calibrator_run_v2(root, run.sha256)
    assert replay.selected_generation.commit_sha256 == generations[3].commit_sha256
    with pytest.raises(FileExistsError, match="run commit"):
        publish_calibrator_run_v2(root, generations)


def test_run_verifier_rejects_forged_selected_epoch_and_missing_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    generations = [
        _publish(root, 0, bsr=0.4, log_excess=0.2, pd=0.8),
        _publish(root, 1, bsr=0.5, log_excess=0.2, pd=0.8),
    ]
    run = publish_calibrator_run_v2(root, generations)
    path = root / RUN_COMMIT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_generation"] = copy.deepcopy(
        payload["generation_inventory"][0]
    )
    payload["selected_generation"] = {
        "epoch": 0,
        "commit_sha256": generations[0].commit_sha256,
        "deployment_checkpoint_sha256": generations[0].manifest[
            "deployment_checkpoint"
        ]["sha256"],
    }
    projection = dict(payload)
    projection.pop("run_identity_sha256")
    from rc.stage2_calibrator_generation_v2 import canonical_json_bytes, canonical_json_sha256

    payload["run_identity_sha256"] = canonical_json_sha256(projection)
    path.chmod(0o644)
    data = canonical_json_bytes(payload)
    path.write_bytes(data)
    with pytest.raises(Stage2CalibratorGenerationV2Error, match="not the recomputed"):
        verify_calibrator_run_v2(root, hashlib.sha256(data).hexdigest())

    # Restore the committed bytes, then remove one generation: the inventory
    # cannot be used as an existence-free declaration.
    path.write_bytes(canonical_json_bytes(dict(run.payload)))
    missing = generations[1].path / COMMIT_FILENAME
    missing.chmod(0o644)
    missing.unlink()
    with pytest.raises((FileNotFoundError, Stage2CalibratorGenerationV2Error)):
        verify_calibrator_run_v2(root, run.sha256)


def test_run_requires_contiguous_epochs_and_identical_inputs(tmp_path: Path) -> None:
    root = tmp_path / "run"
    first = _publish(root, 0, bsr=0.4, log_excess=0.2, pd=0.8)
    third = _publish(root, 2, bsr=0.5, log_excess=0.2, pd=0.8)
    with pytest.raises(Stage2CalibratorGenerationV2Error, match="contiguous"):
        publish_calibrator_run_v2(root, [first, third])

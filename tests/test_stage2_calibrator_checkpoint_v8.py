from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
import torch

from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from model.budget_conditioned_residual_transport_calibrator import (
    BudgetConditionedDirectResidualTransportCalibrator,
    BudgetConditionedMonotoneNoTargetAnchorCalibrator,
    BudgetConditionedMonotoneResidualTransportCalibrator,
)
from rc.stage2_calibrator_checkpoint_v8 import (
    ARTIFACT_KIND,
    CHECKPOINT_SCHEMA,
    EXPECTED_PARAMETER_COUNTS,
    Stage2CalibratorCheckpointV8Error,
    make_calibrator_checkpoint_v8,
    serialize_calibrator_checkpoint_v8,
    tensor_tree_content_sha256,
    verify_calibrator_checkpoint_v8_bytes,
    verify_calibrator_checkpoint_v8_file,
)


TRAINING_SHA = hashlib.sha256(b"synthetic-v8-training-contract").hexdigest()
VIEW_SHA = hashlib.sha256(b"synthetic-v8-training-view").hexdigest()


def _model(method: str) -> torch.nn.Module:
    if method == "T6_PLUS":
        model_type = BudgetConditionedDirectResidualTransportCalibrator
    elif method == "T8_PLUS_NO_ANCHOR":
        model_type = BudgetConditionedMonotoneNoTargetAnchorCalibrator
    else:
        model_type = BudgetConditionedMonotoneResidualTransportCalibrator
    return model_type(
        context_feature_dim=93,
        hidden_dims=(32,),
        dropout=0.1,
        minimum_residual_increment=1e-6,
    )


def _payload(method: str = "T8_PLUS") -> dict[str, object]:
    return make_calibrator_checkpoint_v8(
        method=method,
        model=_model(method),
        standardizer_mean=np.linspace(-1.0, 1.0, 93, dtype=np.float64),
        standardizer_scale=np.linspace(1e-8, 2.0, 93, dtype=np.float64),
        training_contract_sha256=TRAINING_SHA,
        training_view_identity_sha256=VIEW_SHA,
    )


def _raw_bytes(payload: object) -> bytes:
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


@pytest.mark.parametrize("method", ("T6_PLUS", "T7_PLUS", "T8_PLUS"))
def test_v8_round_trip_binds_nine_rationals_view_and_exact_model(method: str) -> None:
    payload = _payload(method)
    data = serialize_calibrator_checkpoint_v8(payload)
    digest = hashlib.sha256(data).hexdigest()
    restricted = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)

    assert restricted["format_version"] == CHECKPOINT_SCHEMA
    assert restricted["artifact_kind"] == ARTIFACT_KIND
    assert restricted["budget_knot_rationals"] == [
        list(row) for row in BUDGET_KNOT_RATIONALS
    ]
    assert restricted["training_view_identity_sha256"] == VIEW_SHA
    assert restricted["anchor_overlay_required"] is True
    assert restricted["inference_contract"]["anchor_input_shape"] == "[batch,9]"
    assert restricted["inference_contract"]["float_budget_authority_forbidden"] is True
    assert restricted["inference_contract"]["caller_threshold_injection_forbidden"] is True

    verified = verify_calibrator_checkpoint_v8_bytes(
        data,
        digest,
        expected_method=method,
        expected_training_contract_sha256=TRAINING_SHA,
        expected_training_view_identity_sha256=VIEW_SHA,
    )
    replay = verified.model()
    assert type(replay) is type(_model(method))
    assert sum(parameter.numel() for parameter in replay.parameters()) == 3339
    assert EXPECTED_PARAMETER_COUNTS[method] == 3339


def test_v8_no_anchor_ablation_round_trip_forbids_anchor_overlay() -> None:
    method = "T8_PLUS_NO_ANCHOR"
    payload = _payload(method)
    assert payload["anchor_overlay_required"] is False
    contract = payload["inference_contract"]
    assert contract["anchor_overlay_required"] is False
    assert contract["anchor_input_shape"] == "not_applicable"
    assert contract["anchor_source"] == "none_target_anchor_forbidden"
    assert "anchor_coordinates" not in contract["output_fields"]
    data = serialize_calibrator_checkpoint_v8(payload)
    verified = verify_calibrator_checkpoint_v8_bytes(
        data,
        hashlib.sha256(data).hexdigest(),
        expected_method=method,
        expected_training_contract_sha256=TRAINING_SHA,
        expected_training_view_identity_sha256=VIEW_SHA,
    )
    assert type(verified.model()) is BudgetConditionedMonotoneNoTargetAnchorCalibrator

    drifted = copy.deepcopy(payload)
    drifted["anchor_overlay_required"] = True
    with pytest.raises(Stage2CalibratorCheckpointV8Error, match="ablation contract"):
        verify_calibrator_checkpoint_v8_bytes(_raw_bytes(drifted))


def test_v8_rejects_method_class_swaps_and_v7_schema() -> None:
    direct = _model("T6_PLUS")
    monotone = _model("T8_PLUS")
    with pytest.raises(TypeError, match="exact BudgetConditionedDirect"):
        make_calibrator_checkpoint_v8(
            method="T6_PLUS",
            model=monotone,
            standardizer_mean=np.zeros(93),
            standardizer_scale=np.ones(93),
            training_contract_sha256=TRAINING_SHA,
            training_view_identity_sha256=VIEW_SHA,
        )
    with pytest.raises(TypeError, match="exact BudgetConditionedMonotone"):
        make_calibrator_checkpoint_v8(
            method="T8_PLUS",
            model=direct,
            standardizer_mean=np.zeros(93),
            standardizer_scale=np.ones(93),
            training_contract_sha256=TRAINING_SHA,
            training_view_identity_sha256=VIEW_SHA,
        )
    legacy = copy.deepcopy(_payload())
    legacy["format_version"] = "rc-irstd.calibrator.v7"
    with pytest.raises(Stage2CalibratorCheckpointV8Error, match="schema"):
        verify_calibrator_checkpoint_v8_bytes(_raw_bytes(legacy))


def test_v8_rejects_state_budget_and_view_identity_tamper() -> None:
    changed = copy.deepcopy(_payload("T6_PLUS"))
    changed["model_state_dict"]["encoder.0.bias"][0] += 1.0
    with pytest.raises(Stage2CalibratorCheckpointV8Error, match="content digest"):
        verify_calibrator_checkpoint_v8_bytes(_raw_bytes(changed))

    missing_state = copy.deepcopy(_payload("T6_PLUS"))
    missing_state["model_state_dict"].pop("encoder.0.bias")
    missing_state["model_state_content_sha256"] = tensor_tree_content_sha256(
        missing_state["model_state_dict"]
    )
    with pytest.raises(Stage2CalibratorCheckpointV8Error, match="strict"):
        verify_calibrator_checkpoint_v8_bytes(_raw_bytes(missing_state))

    budget = copy.deepcopy(_payload())
    budget["budget_knot_rationals"][4][1] += 1
    with pytest.raises(Stage2CalibratorCheckpointV8Error, match="budget lattice"):
        verify_calibrator_checkpoint_v8_bytes(_raw_bytes(budget))

    view = copy.deepcopy(_payload())
    view["training_view_identity_sha256"] = "0" * 64
    data = _raw_bytes(view)
    with pytest.raises(Stage2CalibratorCheckpointV8Error, match="expected training_view"):
        verify_calibrator_checkpoint_v8_bytes(
            data, expected_training_view_identity_sha256=VIEW_SHA
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reject_head", True),
        ("missing_episode_fallback", True),
        ("anchor_overlay_required", False),
        ("official_test_accessed", True),
    ],
)
def test_v8_rejects_reject_fallback_overlay_and_test_access_drift(field, value) -> None:
    payload = copy.deepcopy(_payload())
    payload[field] = value
    with pytest.raises(Stage2CalibratorCheckpointV8Error):
        verify_calibrator_checkpoint_v8_bytes(_raw_bytes(payload))


def test_v8_rejects_inference_contract_and_exact_field_injection() -> None:
    payload = copy.deepcopy(_payload())
    payload["inference_contract"]["anchor_algorithm"] = "caller_freeform"
    with pytest.raises(Stage2CalibratorCheckpointV8Error, match="inference contract"):
        verify_calibrator_checkpoint_v8_bytes(_raw_bytes(payload))

    payload = copy.deepcopy(_payload())
    payload["pixel_budgets"] = [1e-4, 1e-5, 1e-6]
    with pytest.raises(Stage2CalibratorCheckpointV8Error, match="field closure"):
        verify_calibrator_checkpoint_v8_bytes(_raw_bytes(payload))


def test_v8_file_hash_and_symlink_fail_closed(tmp_path: Path) -> None:
    data = serialize_calibrator_checkpoint_v8(_payload())
    target = tmp_path / "checkpoint.v8.pt"
    target.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    verified = verify_calibrator_checkpoint_v8_file(target, digest)
    assert verified.sha256 == digest
    with pytest.raises(Stage2CalibratorCheckpointV8Error, match="SHA-256"):
        verify_calibrator_checkpoint_v8_file(target, "0" * 64)
    link = tmp_path / "link.v8.pt"
    link.symlink_to(target)
    with pytest.raises(Stage2CalibratorCheckpointV8Error, match="stable direct"):
        verify_calibrator_checkpoint_v8_file(link, digest)

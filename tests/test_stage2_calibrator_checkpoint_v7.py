from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
import torch

from model.direct_no_reject_pixel_calibrator import DirectNoRejectPixelCalibrator
from model.endpoint_aware_pixel_calibrator import (
    DirectEndpointAwarePixelCalibrator,
    MonotoneEndpointAwarePixelCalibrator,
)
from model.endpoint_aware_threshold import representation_contract
from rc.domain_statistics import FEATURE_NAMES
from rc.stage2_calibrator_checkpoint_v7 import (
    ARTIFACT_KIND,
    CHECKPOINT_SCHEMA,
    EXPECTED_PARAMETER_COUNTS,
    STANDARDIZER_SCHEMA,
    Stage2CalibratorCheckpointV7Error,
    cpu_inference_contract,
    make_calibrator_checkpoint_v7,
    serialize_calibrator_checkpoint_v7,
    tensor_tree_content_sha256,
    verify_calibrator_checkpoint_v7_bytes,
    verify_calibrator_checkpoint_v7_file,
)


TRAINING_SHA = hashlib.sha256(b"synthetic-v7-training-contract").hexdigest()


def _model(method: str) -> torch.nn.Module:
    common = {
        "context_feature_dim": 93,
        "pixel_budget_grid": [1e-4, 1e-5, 1e-6],
        "hidden_dims": [32],
        "dropout": 0.1,
    }
    if method == "T6":
        return DirectEndpointAwarePixelCalibrator(**common)
    return MonotoneEndpointAwarePixelCalibrator(
        **common, minimum_raw_coordinate_gap=0.001
    )


def _payload(method: str = "T8") -> dict[str, object]:
    return make_calibrator_checkpoint_v7(
        method=method,
        model=_model(method),
        standardizer_mean=np.linspace(-1.0, 1.0, 93, dtype=np.float64),
        standardizer_scale=np.linspace(1e-8, 2.0, 93, dtype=np.float64),
        training_contract_sha256=TRAINING_SHA,
    )


def _raw_bytes(payload: object) -> bytes:
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


@pytest.mark.parametrize(
    ("method", "expected_parameters"),
    (("T6", 3108), ("T7", 3141), ("T8", 3141)),
)
def test_v7_round_trip_bytes_file_and_strict_model(
    tmp_path: Path, method: str, expected_parameters: int
) -> None:
    payload = _payload(method)
    data = serialize_calibrator_checkpoint_v7(payload)
    digest = hashlib.sha256(data).hexdigest()

    # The production bytes must use only the restricted unpickler surface.
    restricted = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    assert restricted["format_version"] == CHECKPOINT_SCHEMA
    assert restricted["representation_contract"] == representation_contract()
    assert restricted["standardizer"]["schema_version"] == STANDARDIZER_SCHEMA
    assert restricted["standardizer"]["feature_names"] == list(FEATURE_NAMES)
    assert restricted["inference_contract"] == cpu_inference_contract(
        restricted["inference_contract"]["anchor_mix_alpha_hex"]
    )
    assert restricted["inference_contract"]["threshold_decode"] == (
        "endpoint_aware_piecewise_tail_coordinate_v2"
    )
    assert restricted["inference_contract"][
        "sigmoid_logit_interpretation_forbidden"
    ] is True

    verified = verify_calibrator_checkpoint_v7_bytes(
        data,
        digest,
        expected_method=method,
        expected_training_contract_sha256=TRAINING_SHA,
    )
    model = verified.model()
    assert type(model) is type(_model(method))
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == expected_parameters
    )
    assert EXPECTED_PARAMETER_COUNTS[method] == expected_parameters

    path = tmp_path / f"{method}.v7.pt"
    path.write_bytes(data)
    verified_file = verify_calibrator_checkpoint_v7_file(
        path, digest, expected_method=method
    )
    assert verified_file.sha256 == digest
    assert verified_file.training_contract_sha256 == TRAINING_SHA


def test_v7_rejects_legacy_model_and_method_class_swaps() -> None:
    legacy = DirectNoRejectPixelCalibrator(
        93, [1e-4, 1e-5, 1e-6], hidden_dims=[32], dropout=0.1
    )
    with pytest.raises(TypeError, match="DirectEndpointAware"):
        make_calibrator_checkpoint_v7(
            method="T6",
            model=legacy,
            standardizer_mean=np.zeros(93),
            standardizer_scale=np.ones(93),
            training_contract_sha256=TRAINING_SHA,
        )
    with pytest.raises(TypeError, match="MonotoneEndpointAware"):
        make_calibrator_checkpoint_v7(
            method="T8",
            model=_model("T6"),
            standardizer_mean=np.zeros(93),
            standardizer_scale=np.ones(93),
            training_contract_sha256=TRAINING_SHA,
        )


def test_v7_rejects_v6_and_exact_field_drift() -> None:
    payload = _payload()
    legacy = copy.deepcopy(payload)
    legacy["format_version"] = "rc-irstd.calibrator.v6"
    with pytest.raises(Stage2CalibratorCheckpointV7Error, match="not schema"):
        verify_calibrator_checkpoint_v7_bytes(_raw_bytes(legacy))

    extra = copy.deepcopy(payload)
    extra["unexpected"] = False
    with pytest.raises(Stage2CalibratorCheckpointV7Error, match="fields mismatch"):
        verify_calibrator_checkpoint_v7_bytes(_raw_bytes(extra))

    missing = copy.deepcopy(payload)
    missing.pop("representation_contract")
    with pytest.raises(Stage2CalibratorCheckpointV7Error, match="fields mismatch"):
        verify_calibrator_checkpoint_v7_bytes(_raw_bytes(missing))


def test_v7_is_deployment_only_and_rejects_old_or_training_state_identity() -> None:
    assert ARTIFACT_KIND == "immutable_endpoint_aware_deployment_state"
    payload = _payload()
    assert payload["artifact_kind"] == ARTIFACT_KIND

    old_identity = copy.deepcopy(payload)
    old_identity["artifact_kind"] = (
        "immutable_endpoint_aware_training_and_deployment_state"
    )
    with pytest.raises(Stage2CalibratorCheckpointV7Error, match="artifact_kind"):
        verify_calibrator_checkpoint_v7_bytes(_raw_bytes(old_identity))

    for forbidden_field in (
        "optimizer",
        "epoch",
        "rank",
        "history",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "dataloader_rng_state",
    ):
        training_state = copy.deepcopy(payload)
        training_state[forbidden_field] = {}
        with pytest.raises(Stage2CalibratorCheckpointV7Error, match="fields mismatch"):
            verify_calibrator_checkpoint_v7_bytes(_raw_bytes(training_state))


def test_v7_rejects_weight_tamper_and_strict_state_drift() -> None:
    payload = _payload("T6")
    changed = copy.deepcopy(payload)
    state = changed["model_state_dict"]
    state["encoder.0.bias"][0] += 1.0
    with pytest.raises(Stage2CalibratorCheckpointV7Error, match="content digest"):
        verify_calibrator_checkpoint_v7_bytes(_raw_bytes(changed))

    missing = copy.deepcopy(payload)
    missing_state = missing["model_state_dict"]
    missing_state.pop("encoder.0.bias")
    missing["model_state_content_sha256"] = tensor_tree_content_sha256(
        missing_state
    )
    with pytest.raises(Stage2CalibratorCheckpointV7Error, match="strict"):
        verify_calibrator_checkpoint_v7_bytes(_raw_bytes(missing))


def test_v7_tensor_digest_is_sorted_and_binds_dtype() -> None:
    first = {
        "z": torch.tensor([1.0], dtype=torch.float64),
        "a": torch.tensor([2.0], dtype=torch.float64),
    }
    reordered = {"a": first["a"], "z": first["z"]}
    changed_dtype = {"a": first["a"].to(torch.float32), "z": first["z"]}
    assert tensor_tree_content_sha256(first) == tensor_tree_content_sha256(reordered)
    assert tensor_tree_content_sha256(first) != tensor_tree_content_sha256(
        changed_dtype
    )


def test_v7_inference_alpha_is_derived_from_strict_state() -> None:
    payload = _payload("T6")
    state = payload["model_state_dict"]
    assert state["anchor_mix_logit"].dtype == torch.float64
    observed_alpha_hex = float(torch.sigmoid(state["anchor_mix_logit"]).item()).hex()
    assert payload["inference_contract"]["anchor_mix_alpha_hex"] == observed_alpha_hex

    changed = copy.deepcopy(payload)
    changed_state = changed["model_state_dict"]
    changed_state["anchor_mix_logit"] += 0.25
    changed["model_state_content_sha256"] = tensor_tree_content_sha256(changed_state)
    with pytest.raises(Stage2CalibratorCheckpointV7Error, match="inference contract"):
        verify_calibrator_checkpoint_v7_bytes(_raw_bytes(changed))


@pytest.mark.parametrize("mutation", ("mean", "scale", "features", "dtype", "shape"))
def test_v7_rejects_standardizer_tamper(mutation: str) -> None:
    payload = copy.deepcopy(_payload())
    standardizer = payload["standardizer"]
    if mutation == "mean":
        standardizer["mean"][0] += 0.25
    elif mutation == "scale":
        standardizer["scale"][0] = 0.0
    elif mutation == "features":
        standardizer["feature_names"][0], standardizer["feature_names"][1] = (
            standardizer["feature_names"][1],
            standardizer["feature_names"][0],
        )
    elif mutation == "dtype":
        standardizer["mean"] = standardizer["mean"].to(torch.float32)
    else:
        standardizer["scale"] = standardizer["scale"][:-1]
    with pytest.raises((Stage2CalibratorCheckpointV7Error, TypeError)):
        verify_calibrator_checkpoint_v7_bytes(_raw_bytes(payload))


@pytest.mark.parametrize("location", ("top", "capability", "model_config"))
def test_v7_rejects_representation_contract_tamper(location: str) -> None:
    payload = copy.deepcopy(_payload())
    if location == "top":
        payload["representation_contract"]["threshold_semantics"] = (
            "prediction = probability >= threshold"
        )
    elif location == "capability":
        payload["capability_contract"]["threshold_representation"][
            "threshold_semantics"
        ] = "sigmoid(logit)"
    else:
        payload["model_config"]["threshold_representation_schema"] = (
            "rc-irstd.endpoint-aware-tail-coordinate.v0"
        )
    with pytest.raises(Stage2CalibratorCheckpointV7Error):
        verify_calibrator_checkpoint_v7_bytes(_raw_bytes(payload))


def test_v7_rejects_inference_anchor_and_alpha_tamper() -> None:
    payload = copy.deepcopy(_payload())
    payload["inference_contract"]["anchor_algorithm"] = "caller_supplied_freeform"
    with pytest.raises(Stage2CalibratorCheckpointV7Error, match="inference contract"):
        verify_calibrator_checkpoint_v7_bytes(_raw_bytes(payload))

    payload = copy.deepcopy(_payload())
    payload["inference_contract"]["anchor_mix_alpha_hex"] = float(0.9).hex()
    with pytest.raises(Stage2CalibratorCheckpointV7Error, match="inference contract"):
        verify_calibrator_checkpoint_v7_bytes(_raw_bytes(payload))


def test_v7_rejects_bad_training_hash_and_external_file_hash(tmp_path: Path) -> None:
    with pytest.raises(Stage2CalibratorCheckpointV7Error, match="SHA-256"):
        make_calibrator_checkpoint_v7(
            method="T6",
            model=_model("T6"),
            standardizer_mean=np.zeros(93),
            standardizer_scale=np.ones(93),
            training_contract_sha256="not-a-digest",
        )

    data = serialize_calibrator_checkpoint_v7(_payload("T7"))
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(data)
    with pytest.raises(Stage2CalibratorCheckpointV7Error, match="SHA-256 mismatch"):
        verify_calibrator_checkpoint_v7_file(path, "0" * 64)


def test_v7_file_verifier_rejects_symlink(tmp_path: Path) -> None:
    data = serialize_calibrator_checkpoint_v7(_payload())
    target = tmp_path / "target.pt"
    target.write_bytes(data)
    link = tmp_path / "link.pt"
    link.symlink_to(target)
    with pytest.raises(Stage2CalibratorCheckpointV7Error, match="symlink"):
        verify_calibrator_checkpoint_v7_file(
            link, hashlib.sha256(data).hexdigest()
        )

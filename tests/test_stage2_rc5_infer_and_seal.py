from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pytest
import torch

from model.endpoint_aware_pixel_calibrator import (
    DirectEndpointAwarePixelCalibrator,
    MonotoneEndpointAwarePixelCalibrator,
)
from model.endpoint_aware_threshold import (
    UPPER_ENDPOINT_KIND,
    representation_contract,
)
from rc.build_stage2_rc5_context import (
    BUNDLE_CAPABILITY_SCHEMA,
    COMMIT_SCHEMA,
    PRODUCER_MANIFEST_SCHEMA,
)
from rc.stage2_calibrator_checkpoint_v7 import (
    make_calibrator_checkpoint_v7,
    serialize_calibrator_checkpoint_v7,
    verify_calibrator_checkpoint_v7_bytes,
)
from rc.stage2_context_tail_anchor import (
    BUDGET_RATIONALS,
    build_context_tail_anchor,
    verify_context_tail_anchor,
)
from rc.stage2_crossfit_schema_v6 import (
    FEATURE_DIM,
    OUTER_TARGETS,
    SOURCE_DIAGNOSTIC_VALIDATION,
    VerifiedContextInferenceMaterialV2,
    VerifiedStage2ContextV2,
    build_context_payload_v2,
    context_inference_material_v2,
    verify_context_payload_v2,
)
from rc.stage2_rc5_infer_and_seal import (
    CONTEXT_ADAPTER,
    DECISION_SCHEMA,
    TRANSCRIPT_SCHEMA,
    Stage2RC5InferenceSealError,
    VerifiedStage2RC5InferenceSeal,
    assert_verified_stage2_rc5_inference_seal,
    canonical_json_bytes,
    canonical_json_sha256,
    _infer_and_seal_stage2_rc5_material_core,
    _verify_stage2_rc5_inference_seal_material_core,
    infer_and_seal_stage2_rc5 as public_infer_and_seal_stage2_rc5,
    verify_stage2_rc5_inference_seal as public_verify_stage2_rc5_inference_seal,
)
from rc.stage2_variable_query_geometry import (
    build_stage2_variable_query_geometry,
)


def _context_maps() -> tuple[np.ndarray, ...]:
    maps: list[np.ndarray] = []
    base = np.linspace(0.0, 0.99, 64, dtype=np.float64).reshape(8, 8)
    for index in range(14):
        array = np.array(base, dtype=np.float64, order="C", copy=True)
        array += index * 1e-6
        array[:] = np.minimum(array, 0.999999)
        maps.append(array)
    maps[-1][-1, -1] = 1.0
    return tuple(maps)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _producer_bundle_binding() -> dict[str, str]:
    return {
        "capability_schema": BUNDLE_CAPABILITY_SCHEMA,
        "producer_manifest_schema": PRODUCER_MANIFEST_SCHEMA,
        "commit_schema": COMMIT_SCHEMA,
        "producer_identity_sha256": _sha("test-producer"),
        "bundle_identity_sha256": _sha("test-bundle"),
        "producer_manifest_sha256": _sha("test-producer-manifest"),
        "commit_sha256": _sha("test-bundle-commit"),
    }


def infer_and_seal_stage2_rc5(
    *,
    checkpoint: Any,
    context: VerifiedStage2ContextV2,
    anchor: Any,
) -> bytes:
    return _infer_and_seal_stage2_rc5_material_core(
        checkpoint=checkpoint,
        context=context,
        anchor=anchor,
        producer_bundle_binding=_producer_bundle_binding(),
    )


def verify_stage2_rc5_inference_seal(
    data: bytes,
    *,
    checkpoint: Any,
    context: VerifiedStage2ContextV2,
    anchor: Any,
) -> VerifiedStage2RC5InferenceSeal:
    return _verify_stage2_rc5_inference_seal_material_core(
        data,
        checkpoint=checkpoint,
        context=context,
        anchor=anchor,
        producer_bundle_binding=_producer_bundle_binding(),
    )


def _records(count: int, tag: str) -> list[dict[str, Any]]:
    return [
        {
            "canonical_id": f"canonical:{tag}:{index}",
            "image_id": f"image:{tag}:{index}",
            "original_image_sha256": _sha(f"image:{tag}:{index}"),
            "exclusion_group_id": f"exclusion:{tag}:{index}",
            "near_duplicate_cluster_id_or_unique_sentinel": (
                f"unique:{tag}:{index}"
            ),
            "source_role_record_index": index,
        }
        for index in range(count)
    ]


def _verified_context_v2(
    *,
    count: int = 43,
    tag: str = "rc5-inference",
    feature_values: list[float] | None = None,
) -> VerifiedStage2ContextV2:
    geometry = build_stage2_variable_query_geometry(count)
    window = geometry["windows"][0]
    records = _records(count, tag)
    payload = build_context_payload_v2(
        expected_role=SOURCE_DIAGNOSTIC_VALIDATION,
        outer_fold_id="outer_leave_nuaa_sirst",
        outer_target=OUTER_TARGETS["outer_leave_nuaa_sirst"],
        source_domain="NUDT-SIRST",
        base_seed=42,
        derived_seed=91_713,
        geometry=geometry,
        window_index=0,
        window_id=f"{tag}:window-0",
        context_records=records[
            window["context_start"] : window["context_stop"]
        ],
        query_identity_records=records[
            window["query_start"] : window["query_stop"]
        ],
        context_feature_values=(
            feature_values
            if feature_values is not None
            else [float(np.float32(index / 128.0)) for index in range(FEATURE_DIM)]
        ),
    )
    return verify_context_payload_v2(payload)


@pytest.fixture(scope="module")
def verified_context() -> VerifiedStage2ContextV2:
    return _verified_context_v2()


@pytest.fixture(scope="module")
def verified_anchor(verified_context: Any) -> Any:
    maps = _context_maps()
    identity = verified_context.payload["context_full_identity_sha256"]
    payload = build_context_tail_anchor(
        context_probability_maps=maps,
        context_identity_sha256=identity,
    )
    return verify_context_tail_anchor(
        payload,
        context_probability_maps=maps,
        expected_context_identity_sha256=identity,
    )


def _model(method: str) -> torch.nn.Module:
    torch.manual_seed(713)
    common = {
        "context_feature_dim": 93,
        "pixel_budget_grid": [1e-4, 1e-5, 1e-6],
        "hidden_dims": [32],
        "dropout": 0.1,
    }
    if method == "T6":
        return DirectEndpointAwarePixelCalibrator(**common)
    return MonotoneEndpointAwarePixelCalibrator(
        **common,
        minimum_raw_coordinate_gap=0.001,
    )


def _checkpoint(
    method: str,
    *,
    weight_delta: float = 0.0,
    scale_delta: float = 0.0,
    force_nonmonotone_t6: bool = False,
) -> Any:
    model = _model(method)
    with torch.no_grad():
        if weight_delta:
            next(model.parameters()).reshape(-1)[0].add_(weight_delta)
        if force_nonmonotone_t6:
            assert isinstance(model, DirectEndpointAwarePixelCalibrator)
            model.anchor_mix_logit.fill_(math.log(999.0))
            model.coordinate_head.weight.zero_()
            model.coordinate_head.bias.copy_(
                torch.tensor([20.0, -20.0, 0.0], dtype=torch.float32)
            )
    training_sha = hashlib.sha256(
        f"rc5-infer-training:{method}".encode("ascii")
    ).hexdigest()
    payload = make_calibrator_checkpoint_v7(
        method=method,
        model=model,
        standardizer_mean=np.linspace(-0.5, 0.5, 93, dtype=np.float64),
        standardizer_scale=(
            np.linspace(0.5, 2.0, 93, dtype=np.float64) + scale_delta
        ),
        training_contract_sha256=training_sha,
    )
    data = serialize_calibrator_checkpoint_v7(payload)
    return verify_calibrator_checkpoint_v7_bytes(
        data,
        hashlib.sha256(data).hexdigest(),
        expected_method=method,
        expected_training_contract_sha256=training_sha,
    )


def _payload(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    assert isinstance(value, dict)
    return value


def _refresh_self_hashes(payload: dict[str, Any]) -> bytes:
    decision = payload["decision"]
    decision.pop("decision_identity_sha256", None)
    decision["decision_identity_sha256"] = canonical_json_sha256(decision)
    payload.pop("transcript_identity_sha256", None)
    payload["transcript_identity_sha256"] = canonical_json_sha256(payload)
    return canonical_json_bytes(payload)


def test_public_api_rejects_bare_context_and_anchor(
    verified_context: Any,
    verified_anchor: Any,
) -> None:
    checkpoint = _checkpoint("T8")
    with pytest.raises(TypeError, match="context producer bundle"):
        public_infer_and_seal_stage2_rc5(
            checkpoint=checkpoint,
            producer_bundle=verified_context,
        )
    material = infer_and_seal_stage2_rc5(
        checkpoint=checkpoint,
        context=verified_context,
        anchor=verified_anchor,
    )
    with pytest.raises(TypeError, match="context producer bundle"):
        public_verify_stage2_rc5_inference_seal(
            material,
            checkpoint=checkpoint,
            producer_bundle=verified_context,
        )


@pytest.mark.parametrize("method", ("T6", "T7", "T8"))
def test_rc5_infer_seal_round_trip_binds_full_causal_chain(
    verified_context: Any,
    verified_anchor: Any,
    method: str,
) -> None:
    checkpoint = _checkpoint(method)
    data = infer_and_seal_stage2_rc5(
        checkpoint=checkpoint,
        context=verified_context,
        anchor=verified_anchor,
    )
    parsed = _payload(data)
    assert parsed["schema_version"] == TRANSCRIPT_SCHEMA
    assert parsed["decision"]["schema_version"] == DECISION_SCHEMA
    assert parsed["method"] == method
    assert parsed["checkpoint_binding"]["checkpoint_bytes_sha256"] == (
        checkpoint.sha256
    )
    assert parsed["checkpoint_binding"]["training_contract_sha256"] == (
        checkpoint.training_contract_sha256
    )
    assert parsed["producer_bundle_binding"] == _producer_bundle_binding()
    assert parsed["context_binding"]["adapter"] == CONTEXT_ADAPTER
    assert parsed["context_binding"]["context_schema"].endswith(
        "context-package.v2"
    )
    assert parsed["context_binding"]["context_payload_sha256"] == (
        verified_context.payload_sha256
    )
    assert parsed["context_binding"]["context_full_identity_sha256"] == (
        verified_anchor.payload["context_identity_sha256"]
    )
    assert parsed["context_binding"]["context_feature_vector_sha256"] == (
        verified_context.payload["context_statistics"]["vector_sha256"]
    )
    assert parsed["anchor_binding"]["anchor_identity_sha256"] == (
        verified_anchor.payload["anchor_identity_sha256"]
    )
    assert parsed["anchor_binding"][
        "context_probability_content_sha256"
    ] == verified_anchor.payload["context_probability_content_sha256"]
    assert parsed["standardizer_binding"]["standardizer_content_sha256"]
    assert parsed["threshold_representation"] == representation_contract()
    assert parsed["budget_rationals"] == [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in BUDGET_RATIONALS
    ]
    assert all(value is False for value in parsed["guardrails"].values())
    assert all(
        parsed["decision"][field] is False
        for field in ("labels_accessed", "query_accessed", "reject", "fallback")
    )
    for row in parsed["decision"]["rows"]:
        for field in (
            "anchor_threshold_probability_hex",
            "anchor_coordinate_hex",
            "learned_raw_coordinate_hex",
            "final_raw_coordinate_hex",
            "canonical_coordinate_hex",
            "decoded_threshold_hex",
        ):
            value = float.fromhex(row[field])
            assert math.isfinite(value)
            assert value.hex() == row[field]

    verified = verify_stage2_rc5_inference_seal(
        data,
        checkpoint=checkpoint,
        context=verified_context,
        anchor=verified_anchor,
    )
    assert verified.method == method
    assert verified.transcript_bytes == data
    assert verified.transcript_bytes_sha256 == hashlib.sha256(data).hexdigest()
    assert (
        infer_and_seal_stage2_rc5(
            checkpoint=checkpoint,
            context=verified_context,
            anchor=verified_anchor,
        )
        == data
    )
    assert_verified_stage2_rc5_inference_seal(verified)
    with pytest.raises(TypeError, match="verifier"):
        VerifiedStage2RC5InferenceSeal()

    rows = parsed["decision"]["rows"]
    final_raw = np.asarray(
        [float.fromhex(row["final_raw_coordinate_hex"]) for row in rows]
    )
    canonical = np.asarray(
        [float.fromhex(row["canonical_coordinate_hex"]) for row in rows]
    )
    thresholds = np.asarray(
        [float.fromhex(row["decoded_threshold_hex"]) for row in rows]
    )
    if method in {"T7", "T8"}:
        assert np.all(np.diff(final_raw) > 0.0)
        assert np.all(np.diff(canonical) >= 0.0)
        assert np.all(np.diff(thresholds) >= 0.0)
        kinds = [row["threshold_kind"] for row in rows]
        if UPPER_ENDPOINT_KIND in kinds:
            first = kinds.index(UPPER_ENDPOINT_KIND)
            assert kinds[first:] == [UPPER_ENDPOINT_KIND] * (3 - first)


def test_t6_explicitly_allows_nonmonotone_threshold_curve(
    verified_context: Any,
    verified_anchor: Any,
) -> None:
    checkpoint = _checkpoint("T6", force_nonmonotone_t6=True)
    data = infer_and_seal_stage2_rc5(
        checkpoint=checkpoint,
        context=verified_context,
        anchor=verified_anchor,
    )
    rows = _payload(data)["decision"]["rows"]
    final_raw = [
        float.fromhex(row["final_raw_coordinate_hex"]) for row in rows
    ]
    assert not all(right >= left for left, right in zip(final_raw, final_raw[1:]))
    verified = verify_stage2_rc5_inference_seal(
        data,
        checkpoint=checkpoint,
        context=verified_context,
        anchor=verified_anchor,
    )
    assert verified.decision["monotonicity_contract"] == (
        "not_structurally_required"
    )


def test_rc5_inference_accepts_no_free_features_thresholds_or_fake_capabilities(
    verified_context: Any,
    verified_anchor: Any,
) -> None:
    checkpoint = _checkpoint("T8")
    with pytest.raises(TypeError, match="VerifiedCalibrator"):
        infer_and_seal_stage2_rc5(
            checkpoint={},
            context=verified_context,
            anchor=verified_anchor,
        )
    with pytest.raises(TypeError, match="context-v2 capability"):
        infer_and_seal_stage2_rc5(
            checkpoint=checkpoint,
            context={},
            anchor=verified_anchor,
        )
    with pytest.raises(TypeError, match="VerifiedContextTailAnchor"):
        infer_and_seal_stage2_rc5(
            checkpoint=checkpoint,
            context=verified_context,
            anchor={},
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        infer_and_seal_stage2_rc5(
            checkpoint=checkpoint,
            context=verified_context,
            anchor=verified_anchor,
            context_features=np.zeros(93, dtype=np.float32),
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        infer_and_seal_stage2_rc5(
            checkpoint=checkpoint,
            context=verified_context,
            anchor=verified_anchor,
            thresholds=[0.5, 0.75, 1.0],
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        infer_and_seal_stage2_rc5(
            checkpoint=checkpoint,
            context=verified_context,
            anchor=verified_anchor,
            query=[{"caller": "supplied"}],
        )
    with pytest.raises(TypeError, match="verifier-only"):
        VerifiedStage2ContextV2(
            payload={},
            canonical_payload=b"{}",
            payload_sha256="0" * 64,
            _capability=object(),
        )
    with pytest.raises(TypeError, match="verifier-only"):
        VerifiedContextInferenceMaterialV2(
            context_package_id="0" * 64,
            context_full_identity_sha256="0" * 64,
            feature_names=tuple(),
            feature_values=tuple(),
            feature_vector_sha256="0" * 64,
            _capability=object(),
        )
    with pytest.raises(TypeError, match="verifier-issued"):
        assert_verified_stage2_rc5_inference_seal({})


def test_rc5_rejects_context_identity_mismatch(
    verified_context: Any,
) -> None:
    maps = _context_maps()
    wrong_identity = "f" * 64
    payload = build_context_tail_anchor(
        context_probability_maps=maps,
        context_identity_sha256=wrong_identity,
    )
    wrong_anchor = verify_context_tail_anchor(
        payload,
        context_probability_maps=maps,
        expected_context_identity_sha256=wrong_identity,
    )
    with pytest.raises(Stage2RC5InferenceSealError, match="identity"):
        infer_and_seal_stage2_rc5(
            checkpoint=_checkpoint("T8"),
            context=verified_context,
            anchor=wrong_anchor,
        )


def test_rc5_rejects_weight_standardizer_feature_and_anchor_tamper(
    verified_context: Any,
    verified_anchor: Any,
) -> None:
    checkpoint = _checkpoint("T8")
    data = infer_and_seal_stage2_rc5(
        checkpoint=checkpoint,
        context=verified_context,
        anchor=verified_anchor,
    )

    changed_weight = _checkpoint("T8", weight_delta=0.125)
    with pytest.raises(Stage2RC5InferenceSealError, match="causal replay"):
        verify_stage2_rc5_inference_seal(
            data,
            checkpoint=changed_weight,
            context=verified_context,
            anchor=verified_anchor,
        )

    changed_standardizer = _checkpoint("T8", scale_delta=0.25)
    with pytest.raises(Stage2RC5InferenceSealError, match="causal replay"):
        verify_stage2_rc5_inference_seal(
            data,
            checkpoint=changed_standardizer,
            context=verified_context,
            anchor=verified_anchor,
        )

    original_payload_sha = verified_context.payload_sha256
    object.__setattr__(verified_context, "payload_sha256", "f" * 64)
    try:
        with pytest.raises(Stage2RC5InferenceSealError, match="context capability"):
            verify_stage2_rc5_inference_seal(
                data,
                checkpoint=checkpoint,
                context=verified_context,
                anchor=verified_anchor,
            )
    finally:
        object.__setattr__(
            verified_context, "payload_sha256", original_payload_sha
        )

    original_coordinates = verified_anchor.coordinates
    object.__setattr__(
        verified_anchor,
        "coordinates",
        (
            original_coordinates[0] + 0.25,
            original_coordinates[1],
            original_coordinates[2],
        ),
    )
    try:
        with pytest.raises(Stage2RC5InferenceSealError, match="anchor capability"):
            verify_stage2_rc5_inference_seal(
                data,
                checkpoint=checkpoint,
                context=verified_context,
                anchor=verified_anchor,
            )
    finally:
        object.__setattr__(
            verified_anchor,
            "coordinates",
            original_coordinates,
        )


def test_rc5_rejects_threshold_and_transcript_tamper_even_with_fresh_self_hashes(
    verified_context: Any,
    verified_anchor: Any,
) -> None:
    checkpoint = _checkpoint("T7")
    data = infer_and_seal_stage2_rc5(
        checkpoint=checkpoint,
        context=verified_context,
        anchor=verified_anchor,
    )

    threshold_tamper = _payload(data)
    threshold_tamper["decision"]["rows"][0][
        "decoded_threshold_hex"
    ] = float(0.125).hex()
    threshold_bytes = _refresh_self_hashes(threshold_tamper)
    with pytest.raises(Stage2RC5InferenceSealError, match="causal replay"):
        verify_stage2_rc5_inference_seal(
            threshold_bytes,
            checkpoint=checkpoint,
            context=verified_context,
            anchor=verified_anchor,
        )

    transcript_tamper = _payload(data)
    transcript_tamper["context_binding"]["adapter"] = "freeform_context"
    transcript_bytes = _refresh_self_hashes(transcript_tamper)
    with pytest.raises(Stage2RC5InferenceSealError, match="causal replay"):
        verify_stage2_rc5_inference_seal(
            transcript_bytes,
            checkpoint=checkpoint,
            context=verified_context,
            anchor=verified_anchor,
        )

    broken_self_hash = _payload(data)
    broken_self_hash["decision"]["rows"][1][
        "canonical_coordinate_hex"
    ] = float(0.5).hex()
    with pytest.raises(Stage2RC5InferenceSealError, match="self-hash"):
        verify_stage2_rc5_inference_seal(
            canonical_json_bytes(broken_self_hash),
            checkpoint=checkpoint,
            context=verified_context,
            anchor=verified_anchor,
        )


def test_schema_v6_dynamic_q_is_absent_from_inference_material_and_decision() -> None:
    features = [float(np.float32(index / 128.0)) for index in range(FEATURE_DIM)]
    q29 = _verified_context_v2(
        count=43,
        tag="dynamic-q",
        feature_values=features,
    )
    q39 = _verified_context_v2(
        count=53,
        tag="dynamic-q",
        feature_values=features,
    )
    assert len(q29.payload["query_identity_records"]) == 29
    assert len(q39.payload["query_identity_records"]) == 39
    assert q29.payload["context_full_identity_sha256"] == (
        q39.payload["context_full_identity_sha256"]
    )

    material29 = context_inference_material_v2(q29)
    material39 = context_inference_material_v2(q39)
    assert material29.feature_values == material39.feature_values
    assert material29.feature_vector_sha256 == material39.feature_vector_sha256
    assert material29.source_query_consumed is False
    assert material39.source_query_consumed is False
    for material in (material29, material39):
        assert not hasattr(material, "query_size")
        assert not hasattr(material, "query_identity_records")

    maps = _context_maps()
    identity = q29.payload["context_full_identity_sha256"]
    anchor_payload = build_context_tail_anchor(
        context_probability_maps=maps,
        context_identity_sha256=identity,
    )
    anchor = verify_context_tail_anchor(
        anchor_payload,
        context_probability_maps=maps,
        expected_context_identity_sha256=identity,
    )
    checkpoint = _checkpoint("T8")
    first = infer_and_seal_stage2_rc5(
        checkpoint=checkpoint, context=q29, anchor=anchor
    )
    second = infer_and_seal_stage2_rc5(
        checkpoint=checkpoint, context=q39, anchor=anchor
    )
    first_payload = _payload(first)
    second_payload = _payload(second)
    assert canonical_json_bytes(first_payload["decision"]) == (
        canonical_json_bytes(second_payload["decision"])
    )
    assert first_payload["guardrails"]["query_accessed"] is False
    assert second_payload["guardrails"]["query_accessed"] is False
    assert first_payload["context_binding"]["source_query_consumed"] is False
    assert second_payload["context_binding"]["source_query_consumed"] is False

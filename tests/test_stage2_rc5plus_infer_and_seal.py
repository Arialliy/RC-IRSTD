from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np
import pytest
import torch

import data_ext.stage2_rc5plus_atomic_learned_decision_set as atomic
from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from model.budget_conditioned_residual_transport_calibrator import (
    BudgetConditionedDirectResidualTransportCalibrator,
    BudgetConditionedMonotoneNoTargetAnchorCalibrator,
    BudgetConditionedMonotoneResidualTransportCalibrator,
)
from rc.build_stage2_rc5_context import (
    BUNDLE_CAPABILITY_SCHEMA,
    COMMIT_SCHEMA,
    PRODUCER_MANIFEST_SCHEMA,
)
from rc.stage2_calibrator_checkpoint_v8 import (
    make_calibrator_checkpoint_v8,
    serialize_calibrator_checkpoint_v8,
    verify_calibrator_checkpoint_v8_bytes,
)
from rc.stage2_context_tail_anchor import (
    build_context_tail_anchor,
    verify_context_tail_anchor,
)
from rc.stage2_context_tail_anchor_v2 import (
    build_context_tail_anchor_v2,
    verify_context_tail_anchor_v2,
)
from rc.stage2_crossfit_schema_v6 import (
    FEATURE_DIM,
    OUTER_TARGETS,
    SOURCE_DIAGNOSTIC_VALIDATION,
    build_context_payload_v2,
    verify_context_payload_v2,
)
from rc.stage2_rc5_feature_mask import build_stage2_rc5_feature_mask
from rc.stage2_rc5plus_infer_and_seal import (
    DECISION_SCHEMA,
    TRANSCRIPT_SCHEMA,
    Stage2RC5PlusInferenceSealError,
    VerifiedStage2RC5PlusInferenceSeal,
    _recompute_material,
    _verify_material,
    assert_verified_stage2_rc5plus_inference_seal,
    canonical_json_bytes,
    canonical_json_sha256,
    infer_and_seal_stage2_rc5plus,
)
import rc.stage2_rc5plus_no_anchor_infer_and_seal as no_anchor_seal
from rc.stage2_variable_query_geometry import build_stage2_variable_query_geometry


REQUESTED = ((1, 20_000), (1, 100_000), (1, 250_000))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _binding() -> dict[str, str]:
    return {
        "capability_schema": BUNDLE_CAPABILITY_SCHEMA,
        "producer_manifest_schema": PRODUCER_MANIFEST_SCHEMA,
        "commit_schema": COMMIT_SCHEMA,
        "producer_identity_sha256": _sha("producer"),
        "bundle_identity_sha256": _sha("bundle"),
        "producer_manifest_sha256": _sha("manifest"),
        "commit_sha256": _sha("commit"),
    }


def _records(count: int) -> list[dict[str, Any]]:
    return [
        {
            "canonical_id": f"canonical:rc5plus:{index}",
            "image_id": f"image:rc5plus:{index}",
            "original_image_sha256": _sha(f"image:{index}"),
            "exclusion_group_id": f"exclusion:{index}",
            "near_duplicate_cluster_id_or_unique_sentinel": f"unique:{index}",
            "source_role_record_index": index,
        }
        for index in range(count)
    ]


def _context():
    count = 43
    geometry = build_stage2_variable_query_geometry(count)
    window = geometry["windows"][0]
    records = _records(count)
    payload = build_context_payload_v2(
        expected_role=SOURCE_DIAGNOSTIC_VALIDATION,
        outer_fold_id="outer_leave_nuaa_sirst",
        outer_target=OUTER_TARGETS["outer_leave_nuaa_sirst"],
        source_domain="NUDT-SIRST",
        base_seed=42,
        derived_seed=91_713,
        geometry=geometry,
        window_index=0,
        window_id="rc5plus-seal:window-0",
        context_records=records[
            window["context_start"] : window["context_stop"]
        ],
        query_identity_records=records[
            window["query_start"] : window["query_stop"]
        ],
        context_feature_values=[
            float(np.float32((index - 17) / 64.0)) for index in range(FEATURE_DIM)
        ],
    )
    return verify_context_payload_v2(payload)


def _maps(*, changed: bool = False) -> tuple[np.ndarray, ...]:
    values = np.linspace(0.0, 1.0, 140_000, dtype=np.float64)
    if changed:
        values[0] = 0.123456789
    return tuple(
        np.ascontiguousarray(chunk.reshape(100, 100), dtype=np.float64)
        for chunk in np.array_split(values, 14)
    )


def _anchors(context, *, requested=REQUESTED, changed: bool = False):
    maps = _maps(changed=changed)
    identity = context.payload["context_full_identity_sha256"]
    v1_payload = build_context_tail_anchor(
        context_probability_maps=maps,
        context_identity_sha256=identity,
    )
    v1 = verify_context_tail_anchor(
        v1_payload,
        context_probability_maps=maps,
        expected_context_identity_sha256=identity,
    )
    v2_payload = build_context_tail_anchor_v2(
        context_probability_maps=maps,
        context_identity_sha256=identity,
        requested_budget_rationals=requested,
    )
    v2 = verify_context_tail_anchor_v2(
        v2_payload,
        context_probability_maps=maps,
        expected_context_identity_sha256=identity,
        expected_requested_budget_rationals=requested,
    )
    return v1, v2


def _checkpoint(method: str = "T8_PLUS", *, feature_variant: str = "C4"):
    torch.manual_seed(1701)
    kwargs = {
        "context_feature_dim": 93,
        "hidden_dims": (32,),
        "dropout": 0.1,
        "minimum_residual_increment": 1e-6,
    }
    if method == "T6_PLUS":
        model = BudgetConditionedDirectResidualTransportCalibrator(**kwargs)
    elif method == "T8_PLUS_NO_ANCHOR":
        model = BudgetConditionedMonotoneNoTargetAnchorCalibrator(**kwargs)
    else:
        model = BudgetConditionedMonotoneResidualTransportCalibrator(**kwargs)
    payload = make_calibrator_checkpoint_v8(
        method=method,
        model=model,
        standardizer_mean=np.linspace(-0.5, 0.5, 93, dtype=np.float64),
        standardizer_scale=np.linspace(0.5, 2.0, 93, dtype=np.float64),
        training_contract_sha256=_sha(f"training:{method}"),
        training_view_identity_sha256=_sha("four-role-view"),
        feature_mask=build_stage2_rc5_feature_mask(feature_variant),
    )
    data = serialize_calibrator_checkpoint_v8(payload)
    return verify_calibrator_checkpoint_v8_bytes(
        data,
        hashlib.sha256(data).hexdigest(),
        expected_method=method,
        expected_training_contract_sha256=_sha(f"training:{method}"),
        expected_training_view_identity_sha256=_sha("four-role-view"),
    )


def _verified_seal_for(
    *,
    method: str,
    context,
    v1,
    v2,
    checkpoint=None,
):
    checkpoint = checkpoint or _checkpoint(method)
    _, data = _recompute_material(
        checkpoint=checkpoint,
        context=context,
        producer_anchor=v1,
        anchor_v2=v2,
        producer_bundle_binding=_binding(),
    )
    seal = _verify_material(
        data,
        checkpoint=checkpoint,
        context=context,
        producer_anchor=v1,
        anchor_v2=v2,
        producer_bundle_binding=_binding(),
    )
    return checkpoint, seal


def _atomic_material(*, requested=REQUESTED):
    context = _context()
    v1, v2 = _anchors(context, requested=requested)
    checkpoints = {}
    seals = {}
    for method in atomic.METHOD_IDS:
        checkpoints[method], seals[method] = _verified_seal_for(
            method=method,
            context=context,
            v1=v1,
            v2=v2,
        )
    payload = atomic._build_material_payload(
        inference_seals=seals,
        producer_bundle_binding=_binding(),
    )
    return context, v1, v2, checkpoints, seals, payload


def _seal(method: str = "T8_PLUS", *, requested=REQUESTED):
    context = _context()
    v1, v2 = _anchors(context, requested=requested)
    checkpoint = _checkpoint(method)
    transcript, data = _recompute_material(
        checkpoint=checkpoint,
        context=context,
        producer_anchor=v1,
        anchor_v2=v2,
        producer_bundle_binding=_binding(),
    )
    return context, v1, v2, checkpoint, transcript, data


@pytest.mark.parametrize("method", ["T6_PLUS", "T7_PLUS", "T8_PLUS"])
def test_v8_seal_round_trip_binds_grid_requests_mask_and_transport(method) -> None:
    context, v1, v2, checkpoint, transcript, data = _seal(method)

    assert transcript["schema_version"] == TRANSCRIPT_SCHEMA
    assert transcript["decision"]["schema_version"] == DECISION_SCHEMA
    assert transcript["method"] == method
    assert transcript["checkpoint_binding"]["checkpoint_bytes_sha256"] == checkpoint.sha256
    assert transcript["checkpoint_binding"]["training_view_identity_sha256"] == (
        checkpoint.training_view_identity_sha256
    )
    assert transcript["context_binding"]["context_full_identity_sha256"] == (
        context.payload["context_full_identity_sha256"]
    )
    assert transcript["anchor_v2_binding"]["anchor_identity_sha256"] == (
        v2.payload["anchor_identity_sha256"]
    )
    assert transcript["anchor_v2_binding"]["primary_budget_cross_generation_match"]
    assert transcript["standardizer_binding"]["feature_mask_variant"] == "C4"
    assert transcript["model_input_binding"]["masked_standardized_float32_sha256"]
    assert transcript["grid_budget_rationals"] == [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in BUDGET_KNOT_RATIONALS
    ]
    decision = transcript["decision"]
    assert decision["deployed_rows_source"] == "requested"
    assert len(decision["grid_rows"]) == 9
    assert len(decision["requested_rows"]) == len(REQUESTED)
    assert not any(transcript["guardrails"].values())
    assert all(
        decision[field] is False
        for field in (
            "labels_accessed",
            "query_accessed",
            "caller_float_budget_authority",
            "caller_threshold_injection",
            "reject",
            "fallback",
        )
    )
    verified = _verify_material(
        data,
        checkpoint=checkpoint,
        context=context,
        producer_anchor=v1,
        anchor_v2=v2,
        producer_bundle_binding=_binding(),
    )
    assert verified.transcript_bytes == data
    assert verified.method == method
    assert assert_verified_stage2_rc5plus_inference_seal(verified) is verified


def test_no_requested_budget_seals_the_complete_nine_knot_grid() -> None:
    _, _, _, _, transcript, _ = _seal(requested=())
    assert transcript["decision"]["deployed_rows_source"] == "grid"
    assert transcript["decision"]["requested_rows"] == []
    assert transcript["requested_budget_rationals"] == []
    assert len(transcript["decision"]["grid_rows"]) == 9


def test_anchor_v2_from_different_maps_is_rejected_even_with_same_context_id() -> None:
    context = _context()
    producer_v1, _ = _anchors(context)
    _, changed_v2 = _anchors(context, changed=True)
    with pytest.raises(Stage2RC5PlusInferenceSealError, match="current context maps"):
        _recompute_material(
            checkpoint=_checkpoint(),
            context=context,
            producer_anchor=producer_v1,
            anchor_v2=changed_v2,
            producer_bundle_binding=_binding(),
        )


def _resign(payload: dict[str, Any]) -> bytes:
    decision = payload["decision"]
    decision.pop("decision_identity_sha256", None)
    decision["decision_identity_sha256"] = canonical_json_sha256(decision)
    payload.pop("transcript_identity_sha256", None)
    payload["transcript_identity_sha256"] = canonical_json_sha256(payload)
    return canonical_json_bytes(payload)


def test_resigned_threshold_tampering_fails_full_causal_replay() -> None:
    context, v1, v2, checkpoint, _, data = _seal()
    tampered = json.loads(data.decode("utf-8"))
    tampered["decision"]["grid_rows"][0]["decoded_threshold_hex"] = 0.5.hex()
    resigned = _resign(tampered)
    with pytest.raises(Stage2RC5PlusInferenceSealError, match="causal replay"):
        _verify_material(
            resigned,
            checkpoint=checkpoint,
            context=context,
            producer_anchor=v1,
            anchor_v2=v2,
            producer_bundle_binding=_binding(),
        )


def test_no_anchor_seal_has_no_anchor_authority_and_replays_byte_exactly() -> None:
    context = _context()
    checkpoint = _checkpoint("T8_PLUS_NO_ANCHOR")
    transcript, data = no_anchor_seal._recompute_material(
        checkpoint=checkpoint,
        context=context,
        producer_bundle_binding=_binding(),
    )
    verified = no_anchor_seal._verify_material(
        data,
        checkpoint=checkpoint,
        context=context,
        producer_bundle_binding=_binding(),
    )
    assert transcript["schema_version"] == no_anchor_seal.TRANSCRIPT_SCHEMA
    assert transcript["decision"]["schema_version"] == no_anchor_seal.DECISION_SCHEMA
    assert transcript["anchor_binding"] == {
        "anchor_schema": "not_applicable",
        "anchor_identity_sha256": "not_applicable",
        "target_anchor_accessed": False,
        "caller_anchor_injection": False,
    }
    assert len(transcript["decision"]["grid_rows"]) == 9
    assert transcript["decision"]["target_anchor_accessed"] is False
    assert verified.transcript_bytes == data
    assert verified.decision_identity_sha256 == transcript["decision"][
        "decision_identity_sha256"
    ]


def test_no_anchor_seal_rejects_anchored_checkpoint_and_resigned_tamper() -> None:
    context = _context()
    anchored = _checkpoint("T8_PLUS")
    with pytest.raises(
        no_anchor_seal.Stage2RC5PlusNoAnchorInferenceSealError,
        match="exact no-anchor",
    ):
        no_anchor_seal._recompute_material(
            checkpoint=anchored,
            context=context,
            producer_bundle_binding=_binding(),
        )

    checkpoint = _checkpoint("T8_PLUS_NO_ANCHOR")
    _, data = no_anchor_seal._recompute_material(
        checkpoint=checkpoint,
        context=context,
        producer_bundle_binding=_binding(),
    )
    tampered = json.loads(data)
    tampered["decision"]["target_anchor_accessed"] = True
    resigned = _resign(tampered)
    with pytest.raises(
        no_anchor_seal.Stage2RC5PlusNoAnchorInferenceSealError,
        match="causal replay",
    ):
        no_anchor_seal._verify_material(
            resigned,
            checkpoint=checkpoint,
            context=context,
            producer_bundle_binding=_binding(),
        )


def test_no_anchor_seal_capability_is_unforgeable() -> None:
    with pytest.raises(TypeError, match="verifier-issued"):
        no_anchor_seal.VerifiedStage2RC5PlusNoAnchorInferenceSeal()
    forged = object.__new__(
        no_anchor_seal.VerifiedStage2RC5PlusNoAnchorInferenceSeal
    )
    with pytest.raises(TypeError, match="verifier-issued"):
        no_anchor_seal.assert_verified_stage2_rc5plus_no_anchor_inference_seal(
            forged
        )


def test_checkpoint_change_changes_sealed_decision_and_replay_rejects_swap() -> None:
    context, v1, v2, checkpoint, _, data = _seal()
    changed = _checkpoint("T7_PLUS")
    with pytest.raises(Stage2RC5PlusInferenceSealError, match="causal replay"):
        _verify_material(
            data,
            checkpoint=changed,
            context=context,
            producer_anchor=v1,
            anchor_v2=v2,
            producer_bundle_binding=_binding(),
        )


def test_public_api_rejects_bare_context_instead_of_producer_authority() -> None:
    context = _context()
    _, v2 = _anchors(context)
    with pytest.raises(TypeError, match="producer bundle"):
        infer_and_seal_stage2_rc5plus(
            checkpoint=_checkpoint(),
            producer_bundle=context,  # type: ignore[arg-type]
            anchor_v2=v2,
        )


def test_verified_seal_capability_cannot_be_constructed() -> None:
    with pytest.raises(TypeError, match="verifier-issued"):
        VerifiedStage2RC5PlusInferenceSeal()
    with pytest.raises(TypeError, match="verifier-issued"):
        assert_verified_stage2_rc5plus_inference_seal(
            object.__new__(VerifiedStage2RC5PlusInferenceSeal)
        )


def test_atomic_learned_set_binds_all_three_routes_before_labels() -> None:
    _, _, _, checkpoints, seals, payload = _atomic_material()

    assert payload["schema_version"] == atomic.DECISION_SET_SCHEMA
    assert payload["method_ids"] == list(atomic.METHOD_IDS)
    assert len(payload["decisions"]) == 3
    assert payload["grid_budget_rationals"] == [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in BUDGET_KNOT_RATIONALS
    ]
    assert not any(
        payload[field]
        for field in (
            "labels_accessed",
            "query_members_opened",
            "reject",
            "fallback",
        )
    )
    for method, decision in zip(atomic.METHOD_IDS, payload["decisions"], strict=True):
        assert decision["method_id"] == method
        assert decision["outcome"] == "complete"
        assert len(decision["grid_rows"]) == 9
        assert len(decision["requested_rows"]) == len(REQUESTED)
        assert decision["authority"]["checkpoint_bytes_sha256"] == (
            checkpoints[method].sha256
        )
        assert decision["authority"]["transcript_bytes_sha256"] == (
            seals[method].transcript_bytes_sha256
        )
        assert all(
            decision[field] is False
            for field in (
                "labels_accessed",
                "query_members_opened",
                "caller_float_budget_authority",
                "caller_threshold_injection",
                "reject",
                "fallback",
            )
        )
    assert atomic._parse_payload(copy.deepcopy(payload)) == payload


def test_atomic_learned_set_rejects_missing_or_reordered_method_routes() -> None:
    _, _, _, _, seals, _ = _atomic_material()
    missing = dict(seals)
    missing.pop("T7_PLUS")
    with pytest.raises(atomic.Stage2RC5PlusAtomicDecisionSetError, match="keys/order"):
        atomic._build_material_payload(
            inference_seals=missing,
            producer_bundle_binding=_binding(),
        )
    reordered = {
        "T8_PLUS": seals["T8_PLUS"],
        "T7_PLUS": seals["T7_PLUS"],
        "T6_PLUS": seals["T6_PLUS"],
    }
    with pytest.raises(atomic.Stage2RC5PlusAtomicDecisionSetError, match="keys/order"):
        atomic._build_material_payload(
            inference_seals=reordered,
            producer_bundle_binding=_binding(),
        )


def test_atomic_learned_set_rejects_different_masked_input_contract() -> None:
    context, v1, v2, _, seals, _ = _atomic_material()
    changed_checkpoint = _checkpoint("T8_PLUS", feature_variant="C3")
    _, changed_seal = _verified_seal_for(
        method="T8_PLUS",
        context=context,
        v1=v1,
        v2=v2,
        checkpoint=changed_checkpoint,
    )
    changed = dict(seals)
    changed["T8_PLUS"] = changed_seal
    with pytest.raises(atomic.Stage2RC5PlusAtomicDecisionSetError, match="standardizer"):
        atomic._build_material_payload(
            inference_seals=changed,
            producer_bundle_binding=_binding(),
        )


def test_atomic_learned_set_rejects_different_requested_budget_identity() -> None:
    context, v1, _, _, seals, _ = _atomic_material()
    _, no_request_v2 = _anchors(context, requested=())
    _, changed_seal = _verified_seal_for(
        method="T8_PLUS",
        context=context,
        v1=v1,
        v2=no_request_v2,
    )
    changed = dict(seals)
    changed["T8_PLUS"] = changed_seal
    with pytest.raises(atomic.Stage2RC5PlusAtomicDecisionSetError, match="anchor_v2_binding"):
        atomic._build_material_payload(
            inference_seals=changed,
            producer_bundle_binding=_binding(),
        )


def test_atomic_payload_resigned_row_tampering_fails_material_replay() -> None:
    _, _, _, _, seals, payload = _atomic_material()
    tampered = copy.deepcopy(payload)
    tampered["decisions"][0]["grid_rows"][0]["decoded_threshold_hex"] = 0.5.hex()
    decision = tampered["decisions"][0]
    decision.pop("decision_identity_sha256")
    decision["decision_identity_sha256"] = atomic._self_hash(
        decision, "decision_identity_sha256"
    )
    tampered.pop("decision_set_identity_sha256")
    tampered["decision_set_identity_sha256"] = atomic._self_hash(
        tampered, "decision_set_identity_sha256"
    )
    assert atomic._parse_payload(tampered) == tampered
    expected = atomic._build_material_payload(
        inference_seals=seals,
        producer_bundle_binding=_binding(),
    )
    assert canonical_json_bytes(tampered) != canonical_json_bytes(expected)


def test_atomic_verified_capability_cannot_be_constructed() -> None:
    with pytest.raises(TypeError, match="verifier-issued"):
        atomic.VerifiedStage2RC5PlusAtomicLearnedDecisionSet()
    forged = object.__new__(atomic.VerifiedStage2RC5PlusAtomicLearnedDecisionSet)
    with pytest.raises(TypeError, match="verifier-issued"):
        atomic.assert_verified_stage2_rc5plus_atomic_learned_decision_set(forged)

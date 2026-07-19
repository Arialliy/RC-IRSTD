"""Pre-label sealed inference for the capacity-matched T8+ no-anchor ablation.

This path intentionally has no anchor or threshold argument.  It replays a
checkpoint-v8 whose method is exactly ``T8_PLUS_NO_ANCHOR``, replays the
label-blind producer bundle for context features, applies the checkpoint-bound
standardizer and feature mask, and seals the complete nine-budget curve.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
from typing import Any

import numpy as np
import torch

from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from model.endpoint_aware_threshold import (
    decode_coordinate_torch,
    endpoint_kinds_numpy,
    representation_contract,
)
from rc.build_stage2_rc5_context import VerifiedStage2RC5ContextBundle
from rc.stage2_calibrator_checkpoint_v8 import (
    CHECKPOINT_SCHEMA,
    VerifiedCalibratorCheckpointV8,
)
from rc.stage2_rc5_feature_mask import FEATURE_MASK_APPLICATION
from rc.stage2_rc5plus_infer_and_seal import (
    MODEL_INPUT_DIGEST_ALGORITHM,
    SELF_HASH_ALGORITHM,
    _context_inference_material,
    _freeze,
    _output_array,
    _producer_bundle_binding,
    _reverify_checkpoint,
    _reverify_producer_bundle,
    _self_hash,
    _standardized_masked_input,
    _validated_producer_binding,
    canonical_json_bytes,
    canonical_json_sha256,
)


TRANSCRIPT_SCHEMA = "rc-irstd.stage2-rc5plus-no-anchor-inference-transcript.v1"
DECISION_SCHEMA = "rc-irstd.stage2-rc5plus-no-anchor-threshold-decision.v1"
TRANSCRIPT_ARTIFACT_TYPE = (
    "rc_irstd_stage2_rc5plus_no_target_anchor_inference_transcript"
)
DECISION_ARTIFACT_TYPE = (
    "rc_irstd_stage2_rc5plus_no_target_anchor_threshold_decision"
)
CAUSAL_CHAIN = (
    "verified_checkpoint_v8_no_anchor_bytes->reverified_label_blind_producer_"
    "bundle->verified_query_free_schema_v6_context->float64_standardize->"
    "float32_feature_mask->anchor_free_residual_cpu_eval->exact_rational_"
    "threshold_curve->sealed_no_reject_ablation_decision"
)
METHOD = "T8_PLUS_NO_ANCHOR"
_CAPABILITY = object()
_GUARDRAILS = (
    "caller_anchor_injection",
    "caller_curve_injection",
    "caller_feature_injection",
    "caller_float_budget_authority",
    "caller_threshold_injection",
    "fallback",
    "labels_accessed",
    "official_test_accessed",
    "query_accessed",
    "reject",
    "target_anchor_accessed",
)


class Stage2RC5PlusNoAnchorInferenceSealError(ValueError):
    """The no-anchor ablation seal failed exact causal replay."""


def _float_hex(value: float) -> str:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise Stage2RC5PlusNoAnchorInferenceSealError(
            "no-anchor output contains a non-finite scalar"
        )
    return numeric.hex()


def _rows(
    *,
    residual: np.ndarray,
    latent: np.ndarray,
    raw: np.ndarray,
    coordinates: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict[str, Any]]:
    kinds = endpoint_kinds_numpy(coordinates)
    return [
        {
            "budget_numerator": numerator,
            "budget_denominator": denominator,
            "context_residual_hex": _float_hex(context_residual),
            "transport_latent_hex": _float_hex(transport_latent),
            "raw_coordinate_hex": _float_hex(raw_coordinate),
            "coordinate_hex": _float_hex(coordinate),
            "decoded_threshold_hex": _float_hex(threshold),
            "threshold_kind": kind,
        }
        for (
            (numerator, denominator),
            context_residual,
            transport_latent,
            raw_coordinate,
            coordinate,
            threshold,
            kind,
        ) in zip(
            BUDGET_KNOT_RATIONALS,
            residual,
            latent,
            raw,
            coordinates,
            thresholds,
            kinds,
            strict=True,
        )
    ]


def _recompute_material(
    *,
    checkpoint: VerifiedCalibratorCheckpointV8,
    context: Any,
    producer_bundle_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    replayed_checkpoint, checkpoint_payload, model = _reverify_checkpoint(checkpoint)
    if (
        replayed_checkpoint.method != METHOD
        or checkpoint_payload["anchor_overlay_required"] is not False
        or checkpoint_payload["inference_contract"]["anchor_overlay_required"]
        is not False
    ):
        raise Stage2RC5PlusNoAnchorInferenceSealError(
            "no-anchor inference requires the exact no-anchor checkpoint-v8"
        )
    context_material = _context_inference_material(context)
    model_input, model_input_sha, feature_mask = _standardized_masked_input(
        checkpoint_payload, context_material.values
    )
    model = model.to(device="cpu").eval()
    with torch.inference_mode():
        output = model(model_input)
    width = len(BUDGET_KNOT_RATIONALS)
    residual = _output_array(output, "grid_residual", width)
    latent = _output_array(output, "grid_transport_latent", width)
    raw = _output_array(output, "grid_raw_coordinates", width)
    coordinates = _output_array(output, "grid_coordinates", width)
    thresholds = _output_array(output, "grid_thresholds", width)
    if not torch.equal(
        decode_coordinate_torch(torch.from_numpy(coordinates)),
        torch.from_numpy(thresholds),
    ):
        raise Stage2RC5PlusNoAnchorInferenceSealError(
            "no-anchor thresholds do not decode from EATC-v2"
        )
    alpha = getattr(output, "correction_strength", None)
    beta = getattr(output, "context_scale", None)
    if (
        not isinstance(alpha, torch.Tensor)
        or alpha.dtype != torch.float64
        or alpha.shape != torch.Size([])
        or not isinstance(beta, torch.Tensor)
        or beta.dtype != torch.float64
        or beta.shape != (1, 1)
    ):
        raise Stage2RC5PlusNoAnchorInferenceSealError(
            "no-anchor transport scalar outputs are invalid"
        )
    decision: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA,
        "artifact_type": DECISION_ARTIFACT_TYPE,
        "artifact_status": "complete_prelabel_ablation",
        "method": METHOD,
        "decision_kind": "sealed_complete_nine_budget_no_target_anchor_curve",
        "grid_budget_rationals": [
            {"numerator": row[0], "denominator": row[1]}
            for row in BUDGET_KNOT_RATIONALS
        ],
        "correction_strength_hex": _float_hex(alpha.item()),
        "context_scale_hex": _float_hex(beta.item()),
        "grid_rows": _rows(
            residual=residual,
            latent=latent,
            raw=raw,
            coordinates=coordinates,
            thresholds=thresholds,
        ),
        "threshold_semantics": "prediction = probability > threshold",
        "target_anchor_accessed": False,
        "labels_accessed": False,
        "query_accessed": False,
        "caller_anchor_injection": False,
        "caller_threshold_injection": False,
        "reject": False,
        "fallback": False,
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    decision["decision_identity_sha256"] = _self_hash(
        decision, "decision_identity_sha256"
    )
    standardizer = checkpoint_payload["standardizer"]
    transcript: dict[str, Any] = {
        "schema_version": TRANSCRIPT_SCHEMA,
        "artifact_type": TRANSCRIPT_ARTIFACT_TYPE,
        "artifact_status": "complete_prelabel_ablation",
        "causal_chain": CAUSAL_CHAIN,
        "method": METHOD,
        "checkpoint_binding": {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "checkpoint_bytes_sha256": replayed_checkpoint.sha256,
            "model_state_content_sha256": checkpoint_payload[
                "model_state_content_sha256"
            ],
            "training_contract_sha256": replayed_checkpoint.training_contract_sha256,
            "training_view_identity_sha256": (
                replayed_checkpoint.training_view_identity_sha256
            ),
            "inference_contract_sha256": canonical_json_sha256(
                checkpoint_payload["inference_contract"]
            ),
            "anchor_overlay_required": False,
        },
        "producer_bundle_binding": dict(
            _validated_producer_binding(producer_bundle_binding)
        ),
        "context_binding": {
            "context_payload_sha256": context_material.context_payload_sha256,
            "context_package_id": context_material.context_package_id,
            "context_full_identity_sha256": context_material.full_identity_sha256,
            "context_feature_vector_sha256": context_material.vector_sha256,
            "query_free_projection": True,
        },
        "anchor_binding": {
            "anchor_schema": "not_applicable",
            "anchor_identity_sha256": "not_applicable",
            "target_anchor_accessed": False,
            "caller_anchor_injection": False,
        },
        "standardizer_binding": {
            "schema_version": standardizer["schema_version"],
            "standardizer_content_sha256": checkpoint_payload[
                "standardizer_content_sha256"
            ],
            "mean_content_sha256": standardizer["mean_content_sha256"],
            "scale_content_sha256": standardizer["scale_content_sha256"],
            "transformation": standardizer["transformation"],
            "feature_mask_variant": feature_mask.variant,
            "feature_mask_identity_sha256": feature_mask.identity_sha256,
            "feature_mask_application": FEATURE_MASK_APPLICATION,
        },
        "model_input_binding": {
            "source_feature_vector_sha256": context_material.vector_sha256,
            "masked_standardized_float32_sha256": model_input_sha,
            "digest_algorithm": MODEL_INPUT_DIGEST_ALGORITHM,
            "dtype": "float32",
            "shape": [1, 93],
            "feature_mask_identity_sha256": feature_mask.identity_sha256,
        },
        "threshold_representation": representation_contract(),
        "threshold_representation_sha256": canonical_json_sha256(
            representation_contract()
        ),
        "decision": decision,
        "guardrails": {key: False for key in _GUARDRAILS},
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    transcript["transcript_identity_sha256"] = _self_hash(
        transcript, "transcript_identity_sha256"
    )
    return transcript, canonical_json_bytes(transcript)


def infer_and_seal_stage2_rc5plus_no_anchor(
    *,
    checkpoint: VerifiedCalibratorCheckpointV8,
    producer_bundle: VerifiedStage2RC5ContextBundle,
) -> bytes:
    bundle = _reverify_producer_bundle(producer_bundle)
    _, data = _recompute_material(
        checkpoint=checkpoint,
        context=bundle.context,
        producer_bundle_binding=_producer_bundle_binding(bundle),
    )
    return data


def _parse(data: bytes) -> Mapping[str, Any]:
    if type(data) is not bytes or not data:
        raise TypeError("no-anchor transcript must be nonempty bytes")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2RC5PlusNoAnchorInferenceSealError(
            "no-anchor transcript is not JSON"
        ) from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != data:
        raise Stage2RC5PlusNoAnchorInferenceSealError(
            "no-anchor transcript is not canonical JSON"
        )
    decision = value.get("decision")
    if (
        value.get("schema_version") != TRANSCRIPT_SCHEMA
        or value.get("artifact_type") != TRANSCRIPT_ARTIFACT_TYPE
        or value.get("artifact_status") != "complete_prelabel_ablation"
        or value.get("causal_chain") != CAUSAL_CHAIN
        or value.get("method") != METHOD
        or value.get("self_hash_algorithm") != SELF_HASH_ALGORITHM
        or value.get("transcript_identity_sha256")
        != _self_hash(value, "transcript_identity_sha256")
        or not isinstance(decision, Mapping)
        or decision.get("schema_version") != DECISION_SCHEMA
        or decision.get("artifact_type") != DECISION_ARTIFACT_TYPE
        or decision.get("method") != METHOD
        or decision.get("decision_identity_sha256")
        != _self_hash(decision, "decision_identity_sha256")
    ):
        raise Stage2RC5PlusNoAnchorInferenceSealError(
            "no-anchor transcript identity contract drifted"
        )
    guardrails = value.get("guardrails")
    if (
        not isinstance(guardrails, Mapping)
        or tuple(guardrails) != tuple(sorted(_GUARDRAILS))
        or any(type(item) is not bool or item for item in guardrails.values())
    ):
        raise Stage2RC5PlusNoAnchorInferenceSealError(
            "no-anchor transcript guardrails drifted"
        )
    return value


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5PlusNoAnchorInferenceSeal:
    transcript_bytes: bytes
    transcript_bytes_sha256: str
    transcript_identity_sha256: str
    decision_identity_sha256: str
    transcript: Mapping[str, Any]
    decision: Mapping[str, Any]
    _capability: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "VerifiedStage2RC5PlusNoAnchorInferenceSeal is verifier-issued only"
        )


def _issue(
    data: bytes, transcript: Mapping[str, Any]
) -> VerifiedStage2RC5PlusNoAnchorInferenceSeal:
    frozen = _freeze(transcript)
    result = object.__new__(VerifiedStage2RC5PlusNoAnchorInferenceSeal)
    for name, value in {
        "transcript_bytes": bytes(data),
        "transcript_bytes_sha256": hashlib.sha256(data).hexdigest(),
        "transcript_identity_sha256": transcript["transcript_identity_sha256"],
        "decision_identity_sha256": transcript["decision"]["decision_identity_sha256"],
        "transcript": frozen,
        "decision": frozen["decision"],
        "_capability": _CAPABILITY,
    }.items():
        object.__setattr__(result, name, value)
    return result


def _verify_material(
    data: bytes,
    *,
    checkpoint: VerifiedCalibratorCheckpointV8,
    context: Any,
    producer_bundle_binding: Mapping[str, Any],
) -> VerifiedStage2RC5PlusNoAnchorInferenceSeal:
    supplied = _parse(data)
    expected, expected_bytes = _recompute_material(
        checkpoint=checkpoint,
        context=context,
        producer_bundle_binding=producer_bundle_binding,
    )
    if not hmac.compare_digest(data, expected_bytes) or supplied != expected:
        raise Stage2RC5PlusNoAnchorInferenceSealError(
            "no-anchor transcript differs from full causal replay"
        )
    return _issue(data, expected)


def verify_stage2_rc5plus_no_anchor_inference_seal(
    data: bytes,
    *,
    checkpoint: VerifiedCalibratorCheckpointV8,
    producer_bundle: VerifiedStage2RC5ContextBundle,
) -> VerifiedStage2RC5PlusNoAnchorInferenceSeal:
    bundle = _reverify_producer_bundle(producer_bundle)
    return _verify_material(
        data,
        checkpoint=checkpoint,
        context=bundle.context,
        producer_bundle_binding=_producer_bundle_binding(bundle),
    )


def assert_verified_stage2_rc5plus_no_anchor_inference_seal(
    value: object,
) -> VerifiedStage2RC5PlusNoAnchorInferenceSeal:
    if (
        type(value) is not VerifiedStage2RC5PlusNoAnchorInferenceSeal
        or getattr(value, "_capability", None) is not _CAPABILITY
    ):
        raise TypeError("a verifier-issued RC5+ no-anchor inference seal is required")
    return value


__all__ = [
    "CAUSAL_CHAIN",
    "DECISION_SCHEMA",
    "METHOD",
    "Stage2RC5PlusNoAnchorInferenceSealError",
    "TRANSCRIPT_SCHEMA",
    "VerifiedStage2RC5PlusNoAnchorInferenceSeal",
    "assert_verified_stage2_rc5plus_no_anchor_inference_seal",
    "infer_and_seal_stage2_rc5plus_no_anchor",
    "verify_stage2_rc5plus_no_anchor_inference_seal",
]

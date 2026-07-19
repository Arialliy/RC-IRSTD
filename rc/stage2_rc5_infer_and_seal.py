"""Verifier-capability-only RC5 inference and threshold sealing.

This module closes the causal chain

checkpoint -> verified schema-v6 context -> query-free inference material ->
verified T4 anchor -> threshold curve.

It is deliberately independent from the existing deployment implementation.
No inference core API accepts free feature vectors, query data, or caller
thresholds. Checkpoint bytes and the recursively immutable context-v2 payload
are reverified before every replay. The context is then reduced by the
schema-v6 verifier to VerifiedContextInferenceMaterialV2, which contains no
query cardinality or query identities. The anchor capability is independently
checked against its identity, content, budget, coordinate, and live
representation contracts.

A bare VerifiedStage2ContextV2 proves only canonical context payload identity.
It does not by itself prove the semantic provenance of score maps to the 93D
feature vector. Final public authority is supplied by the label-blind RC5
producer bundle; this module's context projection remains the inference core
used beneath that authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from model.endpoint_aware_threshold import (
    UPPER_ENDPOINT_COORDINATE,
    canonicalize_raw_torch,
    decode_coordinate_torch,
    encode_probability_numpy,
    endpoint_kinds_numpy,
    representation_contract,
)
from rc.domain_statistics import FEATURE_NAMES
from rc.build_stage2_rc5_context import (
    BUNDLE_CAPABILITY_SCHEMA,
    COMMIT_SCHEMA,
    PRODUCER_MANIFEST_SCHEMA,
    VerifiedStage2RC5ContextBundle,
    assert_verified_stage2_rc5_context_bundle,
    replay_verified_stage2_rc5_context_bundle,
)
from rc.stage2_calibrator_checkpoint_v7 import (
    CHECKPOINT_SCHEMA,
    TENSOR_CONTENT_DIGEST_ALGORITHM,
    VerifiedCalibratorCheckpointV7,
    verify_calibrator_checkpoint_v7_bytes,
)
from rc.stage2_context_tail_anchor import (
    BUDGET_RATIONALS,
    CONTEXT_SIZE,
    CONTEXT_PROBABILITY_CONTENT_ALGORITHM,
    CONTEXT_TAIL_ANCHOR_ALGORITHM,
    CONTEXT_TAIL_ANCHOR_ARTIFACT_TYPE,
    CONTEXT_TAIL_ANCHOR_SCHEMA,
    STRICT_THRESHOLD_SEMANTICS,
    VerifiedContextTailAnchor,
    assert_verified_context_tail_anchor,
    canonical_json_bytes as anchor_canonical_json_bytes,
    canonical_json_sha256 as anchor_canonical_json_sha256,
)
from rc.stage2_crossfit_schema_v6 import (
    CONTEXT_SCHEMA,
    FLOAT32_VECTOR_ALGORITHM,
    VerifiedContextInferenceMaterialV2,
    VerifiedStage2ContextV2,
    assert_verified_context_inference_material_v2,
    assert_verified_context_v2,
    context_inference_material_v2,
    verify_context_payload_v2,
)


TRANSCRIPT_SCHEMA = "rc-irstd.stage2-rc5-inference-transcript.v4"
DECISION_SCHEMA = "rc-irstd.stage2-rc5-threshold-decision.v2"
TRANSCRIPT_ARTIFACT_TYPE = "rc_irstd_stage2_rc5_inference_transcript"
DECISION_ARTIFACT_TYPE = "rc_irstd_stage2_rc5_threshold_decision"
CAUSAL_CHAIN = (
    "verified_checkpoint_bytes->reverified_label_blind_producer_bundle->"
    "verified_schema_v6_context->"
    "verified_query_free_context_inference_material_float32_93->"
    "verified_T4_float64_anchor->standardize_float64_cast_float32->"
    "endpoint_aware_model_eval_cpu->strict_gt_threshold_curve"
)
CONTEXT_ADAPTER = (
    "VerifiedStage2ContextV2_to_"
    "VerifiedContextInferenceMaterialV2_schema_v6_query_free_projection"
)
STANDARDIZED_INPUT_DIGEST_ALGORITHM = (
    "sha256-little-endian-float32-c-order-v1"
)
SELF_HASH_ALGORITHM = "sha256-canonical-json-with-self-field-omitted-v1"

_SHA256_HEX = frozenset("0123456789abcdef")
_VERIFIED_CAPABILITY = object()

_TRANSCRIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "causal_chain",
        "method",
        "checkpoint_binding",
        "producer_bundle_binding",
        "context_binding",
        "anchor_binding",
        "standardizer_binding",
        "model_input_binding",
        "threshold_representation",
        "threshold_representation_sha256",
        "checkpoint_inference_contract",
        "budget_rationals",
        "guardrails",
        "decision",
        "self_hash_algorithm",
        "transcript_identity_sha256",
    }
)
_PRODUCER_BUNDLE_BINDING_FIELDS = frozenset(
    {
        "capability_schema",
        "producer_manifest_schema",
        "commit_schema",
        "producer_identity_sha256",
        "bundle_identity_sha256",
        "producer_manifest_sha256",
        "commit_sha256",
    }
)
_CHECKPOINT_BINDING_FIELDS = frozenset(
    {
        "checkpoint_bytes_sha256",
        "checkpoint_schema",
        "training_contract_sha256",
        "method",
        "calibrator_model",
        "expected_trainable_parameters",
        "model_state_content_sha256",
        "model_state_content_digest_algorithm",
    }
)
_CONTEXT_BINDING_FIELDS = frozenset(
    {
        "adapter",
        "context_schema",
        "context_payload_sha256",
        "context_package_id",
        "context_full_identity_sha256",
        "context_feature_vector_sha256",
        "context_feature_vector_digest_algorithm",
        "feature_schema_sha256",
        "feature_dim",
        "source_query_consumed",
    }
)
_ANCHOR_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "selection_algorithm",
        "anchor_identity_sha256",
        "anchor_payload_sha256",
        "context_identity_sha256",
        "context_probability_content_sha256",
        "context_size",
        "total_context_pixels",
        "budget_rationals",
        "anchor_threshold_probability_hex",
        "anchor_coordinate_hex",
    }
)
_STANDARDIZER_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "standardizer_content_sha256",
        "mean_content_sha256",
        "scale_content_sha256",
        "tensor_content_digest_algorithm",
        "feature_schema_sha256",
        "calculation_dtype",
        "model_input_dtype",
        "scale_floor_hex",
        "transformation",
    }
)
_MODEL_INPUT_BINDING_FIELDS = frozenset(
    {
        "source",
        "feature_dim",
        "source_dtype",
        "source_vector_sha256",
        "standardized_dtype",
        "standardized_shape",
        "standardized_content_sha256",
        "standardized_content_digest_algorithm",
    }
)
_GUARDRAIL_FIELDS = frozenset(
    {
        "labels_accessed",
        "context_labels_accessed",
        "query_accessed",
        "query_scores_accessed",
        "query_labels_accessed",
        "caller_features_accepted",
        "caller_thresholds_accepted",
        "reject",
        "fallback",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "decision_kind",
        "method",
        "budget_order",
        "budget_rationals",
        "anchor_mix_alpha_hex",
        "threshold_representation_schema",
        "threshold_semantics",
        "monotonicity_contract",
        "rows",
        "labels_accessed",
        "query_accessed",
        "reject",
        "fallback",
        "self_hash_algorithm",
        "decision_identity_sha256",
    }
)
_DECISION_ROW_FIELDS = frozenset(
    {
        "budget_numerator",
        "budget_denominator",
        "anchor_threshold_probability_hex",
        "anchor_coordinate_hex",
        "learned_raw_coordinate_hex",
        "final_raw_coordinate_hex",
        "canonical_coordinate_hex",
        "decoded_threshold_hex",
        "threshold_kind",
    }
)
_ANCHOR_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "context_identity_sha256",
        "context_size",
        "total_context_pixels",
        "context_probability_content_algorithm",
        "context_probability_content_sha256",
        "context_map_bindings",
        "budget_order",
        "budget_rationals",
        "threshold_rows",
        "threshold_representation",
        "threshold_semantics",
        "selection_algorithm",
        "guardrails",
        "anchor_identity_sha256",
    }
)
_ANCHOR_ROW_FIELDS = frozenset(
    {
        "budget_numerator",
        "budget_denominator",
        "allowed_strict_exceedances",
        "observed_strict_exceedances",
        "order_statistic_rank_zero_based",
        "threshold_probability_hex",
        "threshold_coordinate_hex",
        "threshold_kind",
    }
)
_ANCHOR_MAP_BINDING_FIELDS = frozenset(
    {"ordinal", "height", "width", "pixel_count", "content_sha256"}
)
_ANCHOR_GUARDRAIL_FIELDS = frozenset(
    {
        "context_labels_accessed",
        "query_scores_accessed",
        "query_labels_accessed",
        "postlabel_statistics_accessed",
    }
)


class Stage2RC5InferenceSealError(ValueError):
    """The RC5 causal inference transcript failed closed."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole byte representation accepted for RC5 transcripts."""

    try:
        return json.dumps(
            _plain(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2RC5InferenceSealError(
            "transcript contains a non-canonical JSON value"
        ) from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _SHA256_HEX
    ):
        raise Stage2RC5InferenceSealError(
            f"{name} must be one lowercase SHA-256 digest"
        )
    return value


def _exact_fields(
    value: Any, fields: frozenset[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise Stage2RC5InferenceSealError(
            f"{name} fields mismatch; missing={sorted(fields-actual)}, "
            f"extra={sorted(actual-fields)}"
        )
    return value


def _exact_false(value: Any, name: str) -> None:
    if type(value) is not bool or value is not False:
        raise Stage2RC5InferenceSealError(f"{name} must be exact false")


def _strict_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise Stage2RC5InferenceSealError(f"{name} must be a positive integer")
    return value


def _canonical_float_hex(value: Any, name: str) -> float:
    if not isinstance(value, str):
        raise Stage2RC5InferenceSealError(
            f"{name} must be canonical float.hex text"
        )
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise Stage2RC5InferenceSealError(
            f"{name} is not hexadecimal float64"
        ) from error
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise Stage2RC5InferenceSealError(
            f"{name} is not canonical finite float64"
        )
    return parsed


def _float_hex_values(values: Sequence[float] | np.ndarray) -> list[str]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise Stage2RC5InferenceSealError(
            "float.hex transcript values must be one finite vector"
        )
    return [float(value).hex() for value in array]


def _budget_payload() -> list[dict[str, int]]:
    return [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in BUDGET_RATIONALS
    ]


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    preimage = {key: item for key, item in value.items() if key != field}
    return canonical_json_sha256(preimage)


def _reverify_checkpoint(
    value: VerifiedCalibratorCheckpointV7,
) -> VerifiedCalibratorCheckpointV7:
    if type(value) is not VerifiedCalibratorCheckpointV7:
        raise TypeError(
            "a verifier-issued VerifiedCalibratorCheckpointV7 is required"
        )
    try:
        checkpoint_bytes = value.checkpoint_bytes
        expected_sha256 = value.sha256
        expected_method = value.method
        expected_training = value.training_contract_sha256
    except AttributeError as error:
        raise TypeError(
            "a complete verifier-issued checkpoint capability is required"
        ) from error
    verified = verify_calibrator_checkpoint_v7_bytes(
        checkpoint_bytes,
        expected_sha256,
        expected_method=expected_method,
        expected_training_contract_sha256=expected_training,
    )
    if (
        verified.checkpoint_bytes != checkpoint_bytes
        or verified.sha256 != expected_sha256
        or verified.method != expected_method
        or verified.training_contract_sha256 != expected_training
    ):
        raise Stage2RC5InferenceSealError(
            "checkpoint capability differs from public-verifier replay"
        )
    return verified


def _reverify_producer_bundle(
    value: VerifiedStage2RC5ContextBundle,
) -> VerifiedStage2RC5ContextBundle:
    """Replay the commit-last producer authority from its current files."""

    bundle = assert_verified_stage2_rc5_context_bundle(value)
    try:
        return replay_verified_stage2_rc5_context_bundle(bundle)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Stage2RC5InferenceSealError(
            "producer bundle differs from full current-state replay"
        ) from error


def _producer_bundle_binding(
    value: VerifiedStage2RC5ContextBundle,
) -> dict[str, str]:
    bundle = assert_verified_stage2_rc5_context_bundle(value)
    manifest = bundle.producer_manifest
    commit = bundle.commit
    producer_identity = _sha256(
        manifest.get("producer_identity_sha256"),
        "producer_identity_sha256",
    )
    if (
        manifest.get("schema_version") != PRODUCER_MANIFEST_SCHEMA
        or commit.get("schema_version") != COMMIT_SCHEMA
        or bundle.capability_schema != BUNDLE_CAPABILITY_SCHEMA
        or commit.get("producer_identity_sha256") != producer_identity
        or commit.get("bundle_identity_sha256")
        != bundle.bundle_identity_sha256
    ):
        raise Stage2RC5InferenceSealError(
            "producer bundle identity closure drifted"
        )
    return {
        "capability_schema": BUNDLE_CAPABILITY_SCHEMA,
        "producer_manifest_schema": PRODUCER_MANIFEST_SCHEMA,
        "commit_schema": COMMIT_SCHEMA,
        "producer_identity_sha256": producer_identity,
        "bundle_identity_sha256": _sha256(
            bundle.bundle_identity_sha256, "bundle_identity_sha256"
        ),
        "producer_manifest_sha256": _sha256(
            bundle.producer_manifest_sha256,
            "producer_manifest_sha256",
        ),
        "commit_sha256": _sha256(bundle.commit_sha256, "commit_sha256"),
    }


def _validated_producer_bundle_binding(
    value: Mapping[str, Any],
) -> dict[str, str]:
    binding = _exact_fields(
        value,
        _PRODUCER_BUNDLE_BINDING_FIELDS,
        "producer_bundle_binding",
    )
    if (
        binding["capability_schema"] != BUNDLE_CAPABILITY_SCHEMA
        or binding["producer_manifest_schema"] != PRODUCER_MANIFEST_SCHEMA
        or binding["commit_schema"] != COMMIT_SCHEMA
    ):
        raise Stage2RC5InferenceSealError(
            "producer bundle binding schema drifted"
        )
    for field in (
        "producer_identity_sha256",
        "bundle_identity_sha256",
        "producer_manifest_sha256",
        "commit_sha256",
    ):
        _sha256(binding[field], f"producer_bundle_binding.{field}")
    return dict(binding)


def _reverify_context(
    value: VerifiedStage2ContextV2,
) -> VerifiedStage2ContextV2:
    context = assert_verified_context_v2(value)
    try:
        payload = context.payload
        canonical_payload = context.canonical_payload
        payload_sha256 = context.payload_sha256
    except AttributeError as error:
        raise Stage2RC5InferenceSealError(
            "context capability lacks its replay dependencies"
        ) from error
    try:
        replay_payload = json.loads(canonical_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2RC5InferenceSealError(
            "context capability canonical bytes are invalid"
        ) from error
    replay = verify_context_payload_v2(replay_payload)
    if (
        replay.canonical_payload != canonical_payload
        or replay.payload_sha256 != payload_sha256
        or hashlib.sha256(canonical_payload).hexdigest() != payload_sha256
        or canonical_json_bytes(replay.payload) != canonical_payload
        or canonical_json_bytes(payload) != canonical_payload
    ):
        raise Stage2RC5InferenceSealError(
            "context capability differs from canonical schema-v6 replay"
        )
    return replay


@dataclass(frozen=True)
class _ContextInferenceMaterial:
    context_capability: VerifiedStage2ContextV2
    capability: VerifiedContextInferenceMaterialV2
    values: np.ndarray
    vector_sha256: str
    full_identity_sha256: str
    context_payload_sha256: str
    context_package_id: str


def _context_inference_material(
    value: VerifiedStage2ContextV2,
) -> _ContextInferenceMaterial:
    """Project canonical context-v2 to verifier-issued query-free material."""

    context = _reverify_context(value)
    material = assert_verified_context_inference_material_v2(
        context_inference_material_v2(context)
    )
    identity = _sha256(
        material.context_full_identity_sha256,
        "context_full_identity_sha256",
    )
    if (
        material.feature_names != tuple(FEATURE_NAMES)
        or type(material.source_query_consumed) is not bool
        or material.source_query_consumed is not False
    ):
        raise Stage2RC5InferenceSealError(
            "schema-v6 query-free inference material contract drifted"
        )
    values = np.asarray(material.feature_values, dtype="<f4")
    if values.shape != (93,) or not np.isfinite(values).all():
        raise Stage2RC5InferenceSealError(
            "verified context must supply one finite float32[93] vector"
        )
    values = np.array(values, dtype="<f4", order="C", copy=True)
    observed = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    expected = _sha256(
        material.feature_vector_sha256,
        "context inference material feature_vector_sha256",
    )
    if observed != expected:
        raise Stage2RC5InferenceSealError(
            "context feature vector content digest mismatch"
        )
    values.setflags(write=False)
    return _ContextInferenceMaterial(
        context_capability=context,
        capability=material,
        context_payload_sha256=_sha256(
            context.payload_sha256, "context.payload_sha256"
        ),
        context_package_id=_sha256(
            material.context_package_id, "context_package_id"
        ),
        values=values,
        vector_sha256=observed,
        full_identity_sha256=identity,
    )


@dataclass(frozen=True)
class _AnchorInferenceMaterial:
    capability: VerifiedContextTailAnchor
    payload: Mapping[str, Any]
    coordinates: np.ndarray
    thresholds: np.ndarray
    anchor_identity_sha256: str
    anchor_payload_sha256: str
    context_probability_content_sha256: str


def _validated_anchor(
    value: VerifiedContextTailAnchor,
    *,
    expected_context_identity_sha256: str,
) -> _AnchorInferenceMaterial:
    anchor = assert_verified_context_tail_anchor(value)
    payload = _exact_fields(anchor.payload, _ANCHOR_TOP_FIELDS, "anchor")
    if (
        payload["schema_version"] != CONTEXT_TAIL_ANCHOR_SCHEMA
        or payload["artifact_type"] != CONTEXT_TAIL_ANCHOR_ARTIFACT_TYPE
        or payload["artifact_status"] != "complete"
        or payload["selection_algorithm"] != CONTEXT_TAIL_ANCHOR_ALGORITHM
        or payload["context_probability_content_algorithm"]
        != CONTEXT_PROBABILITY_CONTENT_ALGORITHM
        or payload["budget_order"]
        != "strictly_descending_loose_to_strict"
        or payload["threshold_semantics"] != STRICT_THRESHOLD_SEMANTICS
        or payload["threshold_representation"] != representation_contract()
    ):
        raise Stage2RC5InferenceSealError("anchor contract drifted")
    context_identity = _sha256(
        payload["context_identity_sha256"],
        "anchor.context_identity_sha256",
    )
    if context_identity != expected_context_identity_sha256:
        raise Stage2RC5InferenceSealError(
            "context_full_identity_sha256 does not match anchor context identity"
        )
    if payload["context_size"] != CONTEXT_SIZE:
        raise Stage2RC5InferenceSealError("anchor context size drifted")
    _strict_positive_int(
        payload["total_context_pixels"], "anchor.total_context_pixels"
    )
    content_sha = _sha256(
        payload["context_probability_content_sha256"],
        "anchor.context_probability_content_sha256",
    )
    anchor_identity = _sha256(
        payload["anchor_identity_sha256"],
        "anchor.anchor_identity_sha256",
    )
    anchor_preimage = {
        key: item
        for key, item in payload.items()
        if key != "anchor_identity_sha256"
    }
    if anchor_canonical_json_sha256(anchor_preimage) != anchor_identity:
        raise Stage2RC5InferenceSealError("anchor identity self-hash mismatch")
    anchor_payload_sha = hashlib.sha256(
        anchor_canonical_json_bytes(payload)
    ).hexdigest()
    map_bindings = payload["context_map_bindings"]
    if (
        isinstance(map_bindings, (str, bytes))
        or not isinstance(map_bindings, Sequence)
        or len(map_bindings) != CONTEXT_SIZE
    ):
        raise Stage2RC5InferenceSealError(
            "anchor context map bindings are invalid"
        )
    total_bound_pixels = 0
    for index, raw in enumerate(map_bindings):
        binding = _exact_fields(
            raw,
            _ANCHOR_MAP_BINDING_FIELDS,
            f"anchor.context_map_bindings[{index}]",
        )
        if type(binding["ordinal"]) is not int or binding["ordinal"] != index:
            raise Stage2RC5InferenceSealError(
                "anchor context map ordinals are not contiguous"
            )
        height = _strict_positive_int(
            binding["height"],
            f"anchor.context_map_bindings[{index}].height",
        )
        width = _strict_positive_int(
            binding["width"],
            f"anchor.context_map_bindings[{index}].width",
        )
        pixel_count = _strict_positive_int(
            binding["pixel_count"],
            f"anchor.context_map_bindings[{index}].pixel_count",
        )
        if pixel_count != height * width:
            raise Stage2RC5InferenceSealError(
                "anchor context map pixel count/shape mismatch"
            )
        _sha256(
            binding["content_sha256"],
            f"anchor.context_map_bindings[{index}].content_sha256",
        )
        total_bound_pixels += pixel_count
    if (
        total_bound_pixels != payload["total_context_pixels"]
        or anchor_canonical_json_sha256(map_bindings) != content_sha
    ):
        raise Stage2RC5InferenceSealError(
            "anchor context map content closure mismatch"
        )

    raw_budgets = payload["budget_rationals"]
    if (
        isinstance(raw_budgets, (str, bytes))
        or not isinstance(raw_budgets, Sequence)
        or len(raw_budgets) != 3
    ):
        raise Stage2RC5InferenceSealError("anchor budget rationals are invalid")
    parsed_budgets: list[tuple[int, int]] = []
    for index, raw in enumerate(raw_budgets):
        row = _exact_fields(
            raw,
            frozenset({"numerator", "denominator"}),
            f"anchor.budget_rationals[{index}]",
        )
        parsed_budgets.append((row["numerator"], row["denominator"]))
    if tuple(parsed_budgets) != BUDGET_RATIONALS:
        raise Stage2RC5InferenceSealError("anchor budget grid drifted")

    threshold_rows = payload["threshold_rows"]
    if (
        isinstance(threshold_rows, (str, bytes))
        or not isinstance(threshold_rows, Sequence)
        or len(threshold_rows) != 3
    ):
        raise Stage2RC5InferenceSealError("anchor threshold rows are invalid")
    row_thresholds: list[float] = []
    row_coordinates: list[float] = []
    for index, raw in enumerate(threshold_rows):
        row = _exact_fields(
            raw, _ANCHOR_ROW_FIELDS, f"anchor.threshold_rows[{index}]"
        )
        if (
            row["budget_numerator"],
            row["budget_denominator"],
        ) != BUDGET_RATIONALS[index]:
            raise Stage2RC5InferenceSealError(
                "anchor threshold row budget drifted"
            )
        for field in (
            "allowed_strict_exceedances",
            "observed_strict_exceedances",
            "order_statistic_rank_zero_based",
        ):
            if type(row[field]) is not int or row[field] < 0:
                raise Stage2RC5InferenceSealError(
                    "anchor threshold integer fields are invalid"
                )
        expected_allowed = (
            BUDGET_RATIONALS[index][0] * payload["total_context_pixels"]
        ) // BUDGET_RATIONALS[index][1]
        expected_rank = max(
            0,
            payload["total_context_pixels"] - expected_allowed - 1,
        )
        if (
            row["allowed_strict_exceedances"] != expected_allowed
            or row["order_statistic_rank_zero_based"] != expected_rank
            or row["observed_strict_exceedances"]
            > row["allowed_strict_exceedances"]
        ):
            raise Stage2RC5InferenceSealError(
                "anchor strict-exceedance row is inconsistent"
            )
        row_thresholds.append(
            _canonical_float_hex(
                row["threshold_probability_hex"],
                f"anchor.threshold_rows[{index}].threshold_probability_hex",
            )
        )
        row_coordinates.append(
            _canonical_float_hex(
                row["threshold_coordinate_hex"],
                f"anchor.threshold_rows[{index}].threshold_coordinate_hex",
            )
        )

    if type(anchor.thresholds) is not tuple or type(anchor.coordinates) is not tuple:
        raise Stage2RC5InferenceSealError(
            "anchor capability values must remain frozen tuples"
        )
    thresholds = np.asarray(anchor.thresholds, dtype=np.float64)
    coordinates = np.asarray(anchor.coordinates, dtype=np.float64)
    if (
        thresholds.shape != (3,)
        or coordinates.shape != (3,)
        or not np.isfinite(thresholds).all()
        or not np.isfinite(coordinates).all()
        or _float_hex_values(thresholds)
        != [float(value).hex() for value in row_thresholds]
        or _float_hex_values(coordinates)
        != [float(value).hex() for value in row_coordinates]
        or not np.array_equal(encode_probability_numpy(thresholds), coordinates)
    ):
        raise Stage2RC5InferenceSealError(
            "anchor capability coordinates differ from its sealed rows"
        )
    expected_kinds = endpoint_kinds_numpy(coordinates)
    if tuple(
        row["threshold_kind"] for row in threshold_rows
    ) != expected_kinds:
        raise Stage2RC5InferenceSealError(
            "anchor threshold kinds differ from canonical coordinates"
        )
    if np.any(coordinates[1:] < coordinates[:-1]):
        raise Stage2RC5InferenceSealError(
            "anchor coordinates must be nondecreasing"
        )
    guardrails = _exact_fields(
        payload["guardrails"],
        _ANCHOR_GUARDRAIL_FIELDS,
        "anchor.guardrails",
    )
    if any(
        type(item) is not bool or item
        for item in guardrails.values()
    ):
        raise Stage2RC5InferenceSealError(
            "anchor records forbidden label/query access"
        )
    coordinates = np.array(coordinates, dtype=np.float64, copy=True)
    thresholds = np.array(thresholds, dtype=np.float64, copy=True)
    coordinates.setflags(write=False)
    thresholds.setflags(write=False)
    return _AnchorInferenceMaterial(
        capability=anchor,
        payload=payload,
        coordinates=coordinates,
        thresholds=thresholds,
        anchor_identity_sha256=anchor_identity,
        anchor_payload_sha256=anchor_payload_sha,
        context_probability_content_sha256=content_sha,
    )


def _standardized_model_input(
    checkpoint_payload: Mapping[str, Any],
    context: _ContextInferenceMaterial,
) -> tuple[torch.Tensor, str]:
    standardizer = checkpoint_payload["standardizer"]
    mean = standardizer["mean"]
    scale = standardizer["scale"]
    if (
        not isinstance(mean, torch.Tensor)
        or not isinstance(scale, torch.Tensor)
        or mean.device.type != "cpu"
        or scale.device.type != "cpu"
        or mean.dtype != torch.float64
        or scale.dtype != torch.float64
        or mean.shape != (93,)
        or scale.shape != (93,)
    ):
        raise Stage2RC5InferenceSealError(
            "checkpoint standardizer is not CPU float64[93]"
        )
    source = torch.from_numpy(
        np.array(context.values, dtype=np.float32, copy=True)
    ).to(dtype=torch.float64)
    standardized64 = (source - mean) / scale
    if not bool(torch.isfinite(standardized64).all().item()):
        raise Stage2RC5InferenceSealError(
            "float64 standardization produced non-finite values"
        )
    model_input = standardized64.to(dtype=torch.float32).reshape(1, 93)
    if not bool(torch.isfinite(model_input).all().item()):
        raise Stage2RC5InferenceSealError(
            "float32 model input overflowed or became non-finite"
        )
    little_endian = np.asarray(
        model_input.detach().cpu().numpy().reshape(93), dtype="<f4"
    )
    digest = hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()
    return model_input.contiguous(), digest


@dataclass(frozen=True)
class _InferenceOutput:
    alpha: float
    learned_raw: np.ndarray
    final_raw: np.ndarray
    canonical: np.ndarray
    thresholds: np.ndarray


def _run_model_inference(
    checkpoint: VerifiedCalibratorCheckpointV7,
    checkpoint_payload: Mapping[str, Any],
    model_input: torch.Tensor,
    anchor: _AnchorInferenceMaterial,
) -> _InferenceOutput:
    model = checkpoint.model().to(device="cpu").eval()
    anchor_tensor = torch.from_numpy(
        np.array(anchor.coordinates, dtype=np.float64, copy=True)
    ).reshape(1, 3)
    with torch.inference_mode():
        output = model(
            model_input,
            anchor_coordinates=anchor_tensor,
        )
    if (
        output.pixel_budget_grid.device.type != "cpu"
        or output.pixel_budget_grid.dtype != torch.float64
        or output.pixel_budget_grid.tolist()
        != [numerator / denominator for numerator, denominator in BUDGET_RATIONALS]
    ):
        raise Stage2RC5InferenceSealError(
            "model output budget grid differs from exact RC5 rationals"
        )
    for name in (
        "anchor_coordinates",
        "grid_learned_raw_coordinates",
        "grid_raw_coordinates",
        "grid_coordinates",
        "grid_thresholds",
    ):
        tensor = getattr(output, name, None)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float64
            or tensor.shape != (1, 3)
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise Stage2RC5InferenceSealError(
                f"model produced invalid {name}"
            )
    if not torch.equal(output.anchor_coordinates, anchor_tensor):
        raise Stage2RC5InferenceSealError(
            "model output changed the verified anchor coordinates"
        )
    alpha_tensor = getattr(output, "anchor_mix_weight", None)
    if (
        not isinstance(alpha_tensor, torch.Tensor)
        or alpha_tensor.device.type != "cpu"
        or alpha_tensor.dtype != torch.float64
        or alpha_tensor.shape != torch.Size([])
        or not bool(torch.isfinite(alpha_tensor).item())
    ):
        raise Stage2RC5InferenceSealError(
            "model produced an invalid anchor mix alpha"
        )
    alpha = float(alpha_tensor.item())
    if not 0.0 < alpha < 1.0:
        raise Stage2RC5InferenceSealError(
            "anchor mix alpha must lie strictly in (0,1)"
        )
    expected_alpha_hex = checkpoint_payload["inference_contract"][
        "anchor_mix_alpha_hex"
    ]
    if alpha.hex() != expected_alpha_hex:
        raise Stage2RC5InferenceSealError(
            "live anchor mix alpha differs from checkpoint state contract"
        )
    expected_raw = (
        (1.0 - alpha_tensor) * output.anchor_coordinates
        + alpha_tensor * output.grid_learned_raw_coordinates
    )
    if not torch.equal(expected_raw, output.grid_raw_coordinates):
        raise Stage2RC5InferenceSealError(
            "live output did not apply the pre-canonical anchor convex mix"
        )
    canonical = canonicalize_raw_torch(output.grid_raw_coordinates)
    if not torch.equal(canonical, output.grid_coordinates):
        raise Stage2RC5InferenceSealError(
            "live canonical coordinates do not replay from final raw values"
        )
    decoded = decode_coordinate_torch(output.grid_coordinates)
    if not torch.equal(decoded, output.grid_thresholds):
        raise Stage2RC5InferenceSealError(
            "live thresholds do not decode from canonical coordinates"
        )

    method = checkpoint.method
    if method in {"T7", "T8"}:
        if not bool(
            (
                output.grid_raw_coordinates[:, 1:]
                > output.grid_raw_coordinates[:, :-1]
            )
            .all()
            .item()
        ):
            raise Stage2RC5InferenceSealError(
                "T7/T8 final raw coordinates are not strictly increasing"
            )
        if not bool(
            (
                output.grid_coordinates[:, 1:]
                >= output.grid_coordinates[:, :-1]
            )
            .all()
            .item()
        ) or not bool(
            (
                output.grid_thresholds[:, 1:]
                >= output.grid_thresholds[:, :-1]
            )
            .all()
            .item()
        ):
            raise Stage2RC5InferenceSealError(
                "T7/T8 canonical or decoded curve decreases"
            )
        endpoint = output.grid_coordinates == UPPER_ENDPOINT_COORDINATE
        if bool((endpoint[:, :-1] & ~endpoint[:, 1:]).any().item()):
            raise Stage2RC5InferenceSealError(
                "T7/T8 upper endpoints are not suffix closed"
            )

    def vector(name: str) -> np.ndarray:
        raw = getattr(output, name).detach().cpu().numpy().reshape(3)
        result = np.array(raw, dtype=np.float64, copy=True)
        result.setflags(write=False)
        return result

    return _InferenceOutput(
        alpha=alpha,
        learned_raw=vector("grid_learned_raw_coordinates"),
        final_raw=vector("grid_raw_coordinates"),
        canonical=vector("grid_coordinates"),
        thresholds=vector("grid_thresholds"),
    )


def _make_decision(
    *,
    method: str,
    anchor: _AnchorInferenceMaterial,
    output: _InferenceOutput,
) -> dict[str, Any]:
    kinds = endpoint_kinds_numpy(output.canonical)
    rows: list[dict[str, Any]] = []
    for index, (numerator, denominator) in enumerate(BUDGET_RATIONALS):
        rows.append(
            {
                "budget_numerator": numerator,
                "budget_denominator": denominator,
                "anchor_threshold_probability_hex": float(
                    anchor.thresholds[index]
                ).hex(),
                "anchor_coordinate_hex": float(
                    anchor.coordinates[index]
                ).hex(),
                "learned_raw_coordinate_hex": float(
                    output.learned_raw[index]
                ).hex(),
                "final_raw_coordinate_hex": float(
                    output.final_raw[index]
                ).hex(),
                "canonical_coordinate_hex": float(
                    output.canonical[index]
                ).hex(),
                "decoded_threshold_hex": float(
                    output.thresholds[index]
                ).hex(),
                "threshold_kind": kinds[index],
            }
        )
    decision: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA,
        "artifact_type": DECISION_ARTIFACT_TYPE,
        "artifact_status": "complete",
        "decision_kind": "sealed_complete_threshold_curve",
        "method": method,
        "budget_order": "strictly_descending_loose_to_strict",
        "budget_rationals": _budget_payload(),
        "anchor_mix_alpha_hex": output.alpha.hex(),
        "threshold_representation_schema": representation_contract()[
            "schema_version"
        ],
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "monotonicity_contract": (
            "strict_raw_nondecreasing_canonical_endpoint_suffix"
            if method in {"T7", "T8"}
            else "not_structurally_required"
        ),
        "rows": rows,
        "labels_accessed": False,
        "query_accessed": False,
        "reject": False,
        "fallback": False,
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    decision["decision_identity_sha256"] = _self_hash(
        decision, "decision_identity_sha256"
    )
    return decision


def _recompute_transcript_material_core(
    *,
    checkpoint: VerifiedCalibratorCheckpointV7,
    context: VerifiedStage2ContextV2,
    anchor: VerifiedContextTailAnchor,
    producer_bundle_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    producer_binding = _validated_producer_bundle_binding(
        producer_bundle_binding
    )
    checked_checkpoint = _reverify_checkpoint(checkpoint)
    checkpoint_payload = checked_checkpoint.payload()
    if (
        checkpoint_payload["format_version"] != CHECKPOINT_SCHEMA
        or checkpoint_payload["model_state_content_digest_algorithm"]
        != TENSOR_CONTENT_DIGEST_ALGORITHM
    ):
        raise Stage2RC5InferenceSealError(
            "checkpoint schema or tensor digest algorithm drifted"
        )
    context_material = _context_inference_material(context)
    anchor_material = _validated_anchor(
        anchor,
        expected_context_identity_sha256=(
            context_material.full_identity_sha256
        ),
    )
    model_input, model_input_sha = _standardized_model_input(
        checkpoint_payload, context_material
    )
    output = _run_model_inference(
        checked_checkpoint,
        checkpoint_payload,
        model_input,
        anchor_material,
    )
    decision = _make_decision(
        method=checked_checkpoint.method,
        anchor=anchor_material,
        output=output,
    )
    standardizer = checkpoint_payload["standardizer"]
    threshold_contract = representation_contract()
    guardrails = {
        "labels_accessed": False,
        "context_labels_accessed": False,
        "query_accessed": False,
        "query_scores_accessed": False,
        "query_labels_accessed": False,
        "caller_features_accepted": False,
        "caller_thresholds_accepted": False,
        "reject": False,
        "fallback": False,
    }
    transcript: dict[str, Any] = {
        "schema_version": TRANSCRIPT_SCHEMA,
        "artifact_type": TRANSCRIPT_ARTIFACT_TYPE,
        "artifact_status": "complete",
        "causal_chain": CAUSAL_CHAIN,
        "method": checked_checkpoint.method,
        "checkpoint_binding": {
            "checkpoint_bytes_sha256": checked_checkpoint.sha256,
            "checkpoint_schema": checkpoint_payload["format_version"],
            "training_contract_sha256": checked_checkpoint.training_contract_sha256,
            "method": checked_checkpoint.method,
            "calibrator_model": checkpoint_payload["calibrator_model"],
            "expected_trainable_parameters": checkpoint_payload[
                "expected_trainable_parameters"
            ],
            "model_state_content_sha256": checkpoint_payload[
                "model_state_content_sha256"
            ],
            "model_state_content_digest_algorithm": checkpoint_payload[
                "model_state_content_digest_algorithm"
            ],
        },
        "producer_bundle_binding": producer_binding,
        "context_binding": {
            "adapter": CONTEXT_ADAPTER,
            "context_schema": CONTEXT_SCHEMA,
            "context_payload_sha256": (
                context_material.context_payload_sha256
            ),
            "context_package_id": context_material.context_package_id,
            "context_full_identity_sha256": (
                context_material.full_identity_sha256
            ),
            "context_feature_vector_sha256": context_material.vector_sha256,
            "context_feature_vector_digest_algorithm": (
                FLOAT32_VECTOR_ALGORITHM
            ),
            "feature_schema_sha256": standardizer[
                "feature_schema_sha256"
            ],
            "feature_dim": 93,
            "source_query_consumed": (
                context_material.capability.source_query_consumed
            ),
        },
        "anchor_binding": {
            "schema_version": anchor_material.payload["schema_version"],
            "artifact_type": anchor_material.payload["artifact_type"],
            "selection_algorithm": anchor_material.payload[
                "selection_algorithm"
            ],
            "anchor_identity_sha256": (
                anchor_material.anchor_identity_sha256
            ),
            "anchor_payload_sha256": anchor_material.anchor_payload_sha256,
            "context_identity_sha256": anchor_material.payload[
                "context_identity_sha256"
            ],
            "context_probability_content_sha256": (
                anchor_material.context_probability_content_sha256
            ),
            "context_size": anchor_material.payload["context_size"],
            "total_context_pixels": anchor_material.payload[
                "total_context_pixels"
            ],
            "budget_rationals": _budget_payload(),
            "anchor_threshold_probability_hex": _float_hex_values(
                anchor_material.thresholds
            ),
            "anchor_coordinate_hex": _float_hex_values(
                anchor_material.coordinates
            ),
        },
        "standardizer_binding": {
            "schema_version": standardizer["schema_version"],
            "standardizer_content_sha256": checkpoint_payload[
                "standardizer_content_sha256"
            ],
            "mean_content_sha256": standardizer["mean_content_sha256"],
            "scale_content_sha256": standardizer["scale_content_sha256"],
            "tensor_content_digest_algorithm": standardizer[
                "tensor_content_digest_algorithm"
            ],
            "feature_schema_sha256": standardizer[
                "feature_schema_sha256"
            ],
            "calculation_dtype": standardizer["calculation_dtype"],
            "model_input_dtype": standardizer["model_input_dtype"],
            "scale_floor_hex": float(standardizer["scale_floor"]).hex(),
            "transformation": standardizer["transformation"],
        },
        "model_input_binding": {
            "source": "VerifiedContextInferenceMaterialV2.feature_values",
            "feature_dim": 93,
            "source_dtype": "float32",
            "source_vector_sha256": context_material.vector_sha256,
            "standardized_dtype": "float32",
            "standardized_shape": "[1,93]",
            "standardized_content_sha256": model_input_sha,
            "standardized_content_digest_algorithm": (
                STANDARDIZED_INPUT_DIGEST_ALGORITHM
            ),
        },
        "threshold_representation": threshold_contract,
        "threshold_representation_sha256": canonical_json_sha256(
            threshold_contract
        ),
        "checkpoint_inference_contract": checkpoint_payload[
            "inference_contract"
        ],
        "budget_rationals": _budget_payload(),
        "guardrails": guardrails,
        "decision": decision,
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    transcript["transcript_identity_sha256"] = _self_hash(
        transcript, "transcript_identity_sha256"
    )
    return transcript, canonical_json_bytes(transcript)


def _recompute_transcript(
    *,
    checkpoint: VerifiedCalibratorCheckpointV7,
    producer_bundle: VerifiedStage2RC5ContextBundle,
) -> tuple[dict[str, Any], bytes]:
    bundle = _reverify_producer_bundle(producer_bundle)
    return _recompute_transcript_material_core(
        checkpoint=checkpoint,
        context=bundle.context,
        anchor=bundle.anchor,
        producer_bundle_binding=_producer_bundle_binding(bundle),
    )


def _infer_and_seal_stage2_rc5_material_core(
    *,
    checkpoint: VerifiedCalibratorCheckpointV7,
    context: VerifiedStage2ContextV2,
    anchor: VerifiedContextTailAnchor,
    producer_bundle_binding: Mapping[str, Any],
) -> bytes:
    """Private unit-testable material core; it is not producer authority."""

    _, data = _recompute_transcript_material_core(
        checkpoint=checkpoint,
        context=context,
        anchor=anchor,
        producer_bundle_binding=producer_bundle_binding,
    )
    return data


def infer_and_seal_stage2_rc5(
    *,
    checkpoint: VerifiedCalibratorCheckpointV7,
    producer_bundle: VerifiedStage2RC5ContextBundle,
) -> bytes:
    """Infer only from a checkpoint and reverified producer bundle."""

    _, data = _recompute_transcript(
        checkpoint=checkpoint,
        producer_bundle=producer_bundle,
    )
    return data


def _duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2RC5InferenceSealError(
                f"duplicate transcript JSON key: {key!r}"
            )
        result[key] = value
    return result


def _nonfinite_guard(value: str) -> None:
    raise Stage2RC5InferenceSealError(
        f"non-finite transcript JSON number is forbidden: {value}"
    )


def _parse_transcript(data: bytes) -> Mapping[str, Any]:
    if type(data) is not bytes or not data:
        raise TypeError("transcript must be non-empty bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_duplicate_guard,
            parse_constant=_nonfinite_guard,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2RC5InferenceSealError(
            "transcript is not canonical UTF-8 JSON"
        ) from error
    transcript = _exact_fields(value, _TRANSCRIPT_FIELDS, "transcript")
    if canonical_json_bytes(transcript) != data:
        raise Stage2RC5InferenceSealError(
            "transcript bytes are not the canonical representation"
        )
    if (
        transcript["schema_version"] != TRANSCRIPT_SCHEMA
        or transcript["artifact_type"] != TRANSCRIPT_ARTIFACT_TYPE
        or transcript["artifact_status"] != "complete"
        or transcript["causal_chain"] != CAUSAL_CHAIN
        or transcript["self_hash_algorithm"] != SELF_HASH_ALGORITHM
    ):
        raise Stage2RC5InferenceSealError(
            "transcript top-level contract drifted"
        )
    _exact_fields(
        transcript["checkpoint_binding"],
        _CHECKPOINT_BINDING_FIELDS,
        "checkpoint_binding",
    )
    _validated_producer_bundle_binding(
        transcript["producer_bundle_binding"]
    )
    _exact_fields(
        transcript["context_binding"],
        _CONTEXT_BINDING_FIELDS,
        "context_binding",
    )
    _exact_fields(
        transcript["anchor_binding"],
        _ANCHOR_BINDING_FIELDS,
        "anchor_binding",
    )
    _exact_fields(
        transcript["standardizer_binding"],
        _STANDARDIZER_BINDING_FIELDS,
        "standardizer_binding",
    )
    _exact_fields(
        transcript["model_input_binding"],
        _MODEL_INPUT_BINDING_FIELDS,
        "model_input_binding",
    )
    guardrails = _exact_fields(
        transcript["guardrails"], _GUARDRAIL_FIELDS, "guardrails"
    )
    for key, item in guardrails.items():
        _exact_false(item, f"guardrails.{key}")
    if transcript["threshold_representation"] != representation_contract():
        raise Stage2RC5InferenceSealError(
            "transcript EATC live contract drifted"
        )
    if transcript["threshold_representation_sha256"] != canonical_json_sha256(
        representation_contract()
    ):
        raise Stage2RC5InferenceSealError(
            "transcript EATC contract digest mismatch"
        )
    if transcript["budget_rationals"] != _budget_payload():
        raise Stage2RC5InferenceSealError(
            "transcript budget rationals drifted"
        )

    decision = _exact_fields(
        transcript["decision"], _DECISION_FIELDS, "decision"
    )
    if (
        decision["schema_version"] != DECISION_SCHEMA
        or decision["artifact_type"] != DECISION_ARTIFACT_TYPE
        or decision["artifact_status"] != "complete"
        or decision["decision_kind"] != "sealed_complete_threshold_curve"
        or decision["budget_order"]
        != "strictly_descending_loose_to_strict"
        or decision["budget_rationals"] != _budget_payload()
        or decision["threshold_semantics"] != STRICT_THRESHOLD_SEMANTICS
        or decision["self_hash_algorithm"] != SELF_HASH_ALGORITHM
    ):
        raise Stage2RC5InferenceSealError("decision contract drifted")
    for key in ("labels_accessed", "query_accessed", "reject", "fallback"):
        _exact_false(decision[key], f"decision.{key}")
    _canonical_float_hex(
        decision["anchor_mix_alpha_hex"],
        "decision.anchor_mix_alpha_hex",
    )
    rows = decision["rows"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise Stage2RC5InferenceSealError(
            "decision must contain exactly three threshold rows"
        )
    for index, raw in enumerate(rows):
        row = _exact_fields(
            raw, _DECISION_ROW_FIELDS, f"decision.rows[{index}]"
        )
        if (
            row["budget_numerator"],
            row["budget_denominator"],
        ) != BUDGET_RATIONALS[index]:
            raise Stage2RC5InferenceSealError(
                "decision row budget drifted"
            )
        for field in (
            "anchor_threshold_probability_hex",
            "anchor_coordinate_hex",
            "learned_raw_coordinate_hex",
            "final_raw_coordinate_hex",
            "canonical_coordinate_hex",
            "decoded_threshold_hex",
        ):
            _canonical_float_hex(
                row[field], f"decision.rows[{index}].{field}"
            )
    _sha256(
        decision["decision_identity_sha256"],
        "decision.decision_identity_sha256",
    )
    if decision["decision_identity_sha256"] != _self_hash(
        decision, "decision_identity_sha256"
    ):
        raise Stage2RC5InferenceSealError("decision self-hash mismatch")
    _sha256(
        transcript["transcript_identity_sha256"],
        "transcript_identity_sha256",
    )
    if transcript["transcript_identity_sha256"] != _self_hash(
        transcript, "transcript_identity_sha256"
    ):
        raise Stage2RC5InferenceSealError("transcript self-hash mismatch")
    return transcript


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5InferenceSeal:
    transcript_bytes: bytes
    transcript_bytes_sha256: str
    transcript_identity_sha256: str
    decision_identity_sha256: str
    producer_identity_sha256: str
    producer_bundle_identity_sha256: str
    producer_manifest_sha256: str
    producer_commit_sha256: str
    method: str
    transcript: Mapping[str, Any]
    decision: Mapping[str, Any]
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "VerifiedStage2RC5InferenceSeal is public-verifier-issued only"
        )


def _verified_seal(
    data: bytes, transcript: Mapping[str, Any]
) -> VerifiedStage2RC5InferenceSeal:
    value = object.__new__(VerifiedStage2RC5InferenceSeal)
    frozen = _freeze(transcript)
    object.__setattr__(value, "transcript_bytes", bytes(data))
    object.__setattr__(
        value,
        "transcript_bytes_sha256",
        hashlib.sha256(data).hexdigest(),
    )
    object.__setattr__(
        value,
        "transcript_identity_sha256",
        transcript["transcript_identity_sha256"],
    )
    object.__setattr__(
        value,
        "decision_identity_sha256",
        transcript["decision"]["decision_identity_sha256"],
    )
    producer = transcript["producer_bundle_binding"]
    object.__setattr__(
        value,
        "producer_identity_sha256",
        producer["producer_identity_sha256"],
    )
    object.__setattr__(
        value,
        "producer_bundle_identity_sha256",
        producer["bundle_identity_sha256"],
    )
    object.__setattr__(
        value,
        "producer_manifest_sha256",
        producer["producer_manifest_sha256"],
    )
    object.__setattr__(
        value,
        "producer_commit_sha256",
        producer["commit_sha256"],
    )
    object.__setattr__(value, "method", transcript["method"])
    object.__setattr__(value, "transcript", frozen)
    object.__setattr__(value, "decision", frozen["decision"])
    object.__setattr__(value, "_capability", _VERIFIED_CAPABILITY)
    return value


def _verify_stage2_rc5_inference_seal_material_core(
    data: bytes,
    *,
    checkpoint: VerifiedCalibratorCheckpointV7,
    context: VerifiedStage2ContextV2,
    anchor: VerifiedContextTailAnchor,
    producer_bundle_binding: Mapping[str, Any],
) -> VerifiedStage2RC5InferenceSeal:
    """Private material-core replay used by focused unit tests."""

    supplied = _parse_transcript(data)
    expected, expected_bytes = _recompute_transcript_material_core(
        checkpoint=checkpoint,
        context=context,
        anchor=anchor,
        producer_bundle_binding=producer_bundle_binding,
    )
    if not hmac.compare_digest(data, expected_bytes):
        raise Stage2RC5InferenceSealError(
            "transcript differs byte-for-byte from full causal replay"
        )
    if supplied != expected:
        raise Stage2RC5InferenceSealError(
            "parsed transcript differs from full causal replay"
        )
    return _verified_seal(data, expected)


def verify_stage2_rc5_inference_seal(
    data: bytes,
    *,
    checkpoint: VerifiedCalibratorCheckpointV7,
    producer_bundle: VerifiedStage2RC5ContextBundle,
) -> VerifiedStage2RC5InferenceSeal:
    """Reverify the producer bundle, recompute inference, and compare bytes."""

    supplied = _parse_transcript(data)
    expected, expected_bytes = _recompute_transcript(
        checkpoint=checkpoint,
        producer_bundle=producer_bundle,
    )
    if not hmac.compare_digest(data, expected_bytes):
        raise Stage2RC5InferenceSealError(
            "transcript differs byte-for-byte from full causal replay"
        )
    if supplied != expected:
        raise Stage2RC5InferenceSealError(
            "parsed transcript differs from full causal replay"
        )
    return _verified_seal(data, expected)


def assert_verified_stage2_rc5_inference_seal(
    value: VerifiedStage2RC5InferenceSeal,
) -> VerifiedStage2RC5InferenceSeal:
    if (
        type(value) is not VerifiedStage2RC5InferenceSeal
        or getattr(value, "_capability", None) is not _VERIFIED_CAPABILITY
    ):
        raise TypeError(
            "a verifier-issued VerifiedStage2RC5InferenceSeal is required"
        )
    return value


__all__ = [
    "CAUSAL_CHAIN",
    "CONTEXT_ADAPTER",
    "DECISION_ARTIFACT_TYPE",
    "DECISION_SCHEMA",
    "SELF_HASH_ALGORITHM",
    "STANDARDIZED_INPUT_DIGEST_ALGORITHM",
    "Stage2RC5InferenceSealError",
    "TRANSCRIPT_ARTIFACT_TYPE",
    "TRANSCRIPT_SCHEMA",
    "VerifiedStage2RC5InferenceSeal",
    "assert_verified_stage2_rc5_inference_seal",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "infer_and_seal_stage2_rc5",
    "verify_stage2_rc5_inference_seal",
]

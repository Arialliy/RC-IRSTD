"""Capability-only sealed inference for the RC5+ residual-transport model.

The public API accepts exactly three authorities: checkpoint-v8, the
commit-last label-blind RC5 producer bundle, and a verifier-issued anchor-v2
computed from the *same* fourteen context maps.  It accepts no free feature
vector, floating-point budget, threshold, query score, label, reject or
fallback input.

The producer bundle is replayed from current files before every inference.
Anchor-v2 is then cross-generation-bound to the producer's independently
replayed v1 anchor by context identity, every map-content binding, total pixel
count, and the three shared primary-budget rows.  The learned model therefore
receives only the checkpoint-bound masked context vector and direct
same-budget analytic anchors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from model.budget_conditioned_endpoint_calibrator import (
    BUDGET_KNOT_RATIONALS,
    PRIMARY_BUDGET_KNOT_INDICES,
)
from model.endpoint_aware_threshold import (
    decode_coordinate_torch,
    encode_probability_numpy,
    endpoint_kinds_numpy,
    representation_contract,
)
from rc.build_stage2_rc5_context import (
    BUNDLE_CAPABILITY_SCHEMA,
    COMMIT_SCHEMA,
    PRODUCER_MANIFEST_SCHEMA,
    VerifiedStage2RC5ContextBundle,
    assert_verified_stage2_rc5_context_bundle,
    replay_verified_stage2_rc5_context_bundle,
)
from rc.stage2_calibrator_checkpoint_v8 import (
    CHECKPOINT_SCHEMA,
    VerifiedCalibratorCheckpointV8,
    verify_calibrator_checkpoint_v8_bytes,
)
from rc.stage2_context_tail_anchor import (
    CONTEXT_PROBABILITY_CONTENT_ALGORITHM,
    CONTEXT_SIZE,
    STRICT_THRESHOLD_SEMANTICS,
    VerifiedContextTailAnchor,
    canonical_json_bytes as anchor_canonical_json_bytes,
    canonical_json_sha256 as anchor_canonical_json_sha256,
)
from rc.stage2_context_tail_anchor_v2 import (
    BUDGET_CURVE_COORDINATE_ALGORITHM,
    CONTEXT_TAIL_ANCHOR_V2_ALGORITHM,
    CONTEXT_TAIL_ANCHOR_V2_ARTIFACT_TYPE,
    CONTEXT_TAIL_ANCHOR_V2_SCHEMA,
    VerifiedContextTailAnchorV2,
    assert_verified_context_tail_anchor_v2,
)
from rc.stage2_rc5_feature_mask import (
    FEATURE_MASK_APPLICATION,
    apply_stage2_rc5_feature_mask_torch,
    verify_stage2_rc5_feature_mask_payload,
)
from rc.stage2_rc5_infer_and_seal import (
    _context_inference_material,
    _producer_bundle_binding,
    _validated_anchor,
)


TRANSCRIPT_SCHEMA = "rc-irstd.stage2-rc5-inference-transcript.v5"
DECISION_SCHEMA = "rc-irstd.stage2-rc5-threshold-decision.v3"
TRANSCRIPT_ARTIFACT_TYPE = "rc_irstd_stage2_rc5plus_inference_transcript"
DECISION_ARTIFACT_TYPE = "rc_irstd_stage2_rc5plus_threshold_decision"
CAUSAL_CHAIN = (
    "verified_checkpoint_v8_bytes->reverified_label_blind_producer_bundle->"
    "verified_query_free_schema_v6_context->verified_same-map_anchor_v2->"
    "float64_standardize->float32_feature_mask->residual_transport_cpu_eval->"
    "exact_rational_threshold_curve->sealed_no_reject_decision"
)
CONTEXT_ADAPTER = (
    "VerifiedStage2ContextV2_to_"
    "VerifiedContextInferenceMaterialV2_schema_v6_query_free_projection"
)
MODEL_INPUT_DIGEST_ALGORITHM = (
    "sha256-little-endian-masked-float32-c-order-v1"
)
SELF_HASH_ALGORITHM = "sha256-canonical-json-with-self-field-omitted-v1"

_SHA256_HEX = frozenset("0123456789abcdef")
_CAPABILITY = object()
_ANCHOR_V2_FIELDS = frozenset(
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
        "grid_budget_rationals",
        "requested_budget_rationals",
        "grid_threshold_rows",
        "requested_threshold_rows",
        "requested_anchor_source",
        "budget_curve_coordinate_algorithm",
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
_MAP_BINDING_FIELDS = frozenset(
    {"ordinal", "height", "width", "pixel_count", "content_sha256"}
)
_ANCHOR_GUARDRAILS = frozenset(
    {
        "context_labels_accessed",
        "query_scores_accessed",
        "query_labels_accessed",
        "postlabel_statistics_accessed",
        "anchor_interpolation_used",
    }
)
_TRANSCRIPT_GUARDRAILS = (
    "context_labels_accessed",
    "query_scores_accessed",
    "query_labels_accessed",
    "postlabel_statistics_accessed",
    "caller_feature_injection",
    "caller_float_budget_authority",
    "caller_threshold_injection",
    "anchor_interpolation_used",
    "reject_used",
    "fallback_used",
    "official_test_accessed",
)


class Stage2RC5PlusInferenceSealError(ValueError):
    """One edge of the RC5+ sealed causal chain failed replay."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2RC5PlusInferenceSealError(
            "value is not finite canonical JSON"
        ) from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _SHA256_HEX
    ):
        raise Stage2RC5PlusInferenceSealError(f"{name} must be lowercase SHA-256")
    return value


def _fields(value: Any, expected: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Stage2RC5PlusInferenceSealError(f"{name} field closure mismatch")
    return value


def _float_hex(value: Any, name: str) -> float:
    if not isinstance(value, str):
        raise Stage2RC5PlusInferenceSealError(f"{name} must be float.hex text")
    try:
        result = float.fromhex(value)
    except ValueError as error:
        raise Stage2RC5PlusInferenceSealError(f"{name} is invalid") from error
    if not math.isfinite(result) or result.hex() != value:
        raise Stage2RC5PlusInferenceSealError(f"{name} is not canonical binary64")
    return result


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({key: item for key, item in value.items() if key != field})


def _budget_payload(values: Sequence[tuple[int, int]]) -> list[dict[str, int]]:
    return [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in values
    ]


def _parse_budget_payload(value: Any, name: str) -> tuple[tuple[int, int], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Stage2RC5PlusInferenceSealError(f"{name} must be an ordered sequence")
    rows: list[tuple[int, int]] = []
    for index, raw in enumerate(value):
        row = _fields(
            raw,
            frozenset({"numerator", "denominator"}),
            f"{name}[{index}]",
        )
        numerator, denominator = row["numerator"], row["denominator"]
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or numerator <= 0
            or denominator <= numerator
        ):
            raise Stage2RC5PlusInferenceSealError(f"{name}[{index}] is invalid")
        reduced = Fraction(numerator, denominator)
        if (reduced.numerator, reduced.denominator) != (numerator, denominator):
            raise Stage2RC5PlusInferenceSealError(
                f"{name}[{index}] is not a lowest-term rational"
            )
        rows.append((numerator, denominator))
    fractions = tuple(Fraction(*row) for row in rows)
    if not all(left > right for left, right in zip(fractions, fractions[1:])):
        raise Stage2RC5PlusInferenceSealError(
            f"{name} is not strictly descending from loose to strict"
        )
    return tuple(rows)


def _reverify_checkpoint(
    value: VerifiedCalibratorCheckpointV8,
) -> tuple[VerifiedCalibratorCheckpointV8, Mapping[str, Any], torch.nn.Module]:
    if type(value) is not VerifiedCalibratorCheckpointV8:
        raise TypeError("a verifier-issued checkpoint-v8 is required")
    try:
        replay = verify_calibrator_checkpoint_v8_bytes(
            value.checkpoint_bytes,
            value.sha256,
            expected_method=value.method,
            expected_training_contract_sha256=value.training_contract_sha256,
            expected_training_view_identity_sha256=(
                value.training_view_identity_sha256
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Stage2RC5PlusInferenceSealError(
            "checkpoint-v8 differs from strict byte replay"
        ) from error
    if (
        replay.checkpoint_bytes != value.checkpoint_bytes
        or replay.sha256 != value.sha256
        or replay.method != value.method
        or replay.training_contract_sha256 != value.training_contract_sha256
        or replay.training_view_identity_sha256
        != value.training_view_identity_sha256
    ):
        raise Stage2RC5PlusInferenceSealError(
            "retained checkpoint-v8 capability differs from replay"
        )
    return replay, replay.payload(), replay.model()


def _reverify_producer_bundle(
    value: VerifiedStage2RC5ContextBundle,
) -> VerifiedStage2RC5ContextBundle:
    try:
        bundle = assert_verified_stage2_rc5_context_bundle(value)
        return replay_verified_stage2_rc5_context_bundle(bundle)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise TypeError(
            "a current verifier-issued label-blind context producer bundle is required"
        ) from error


@dataclass(frozen=True)
class _AnchorV2Material:
    capability: VerifiedContextTailAnchorV2
    payload: Mapping[str, Any]
    identity_sha256: str
    payload_sha256: str
    grid_thresholds: np.ndarray
    grid_coordinates: np.ndarray
    requested_thresholds: np.ndarray
    requested_coordinates: np.ndarray
    requested_budgets: tuple[tuple[int, int], ...]


def _validate_row_set(
    rows: Any,
    budgets: tuple[tuple[int, int], ...],
    *,
    total_pixels: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or len(rows) != len(budgets):
        raise Stage2RC5PlusInferenceSealError(f"{name} cardinality mismatch")
    thresholds: list[float] = []
    coordinates: list[float] = []
    for index, (raw, budget) in enumerate(zip(rows, budgets, strict=True)):
        row = _fields(raw, _ANCHOR_ROW_FIELDS, f"{name}[{index}]")
        if (row["budget_numerator"], row["budget_denominator"]) != budget:
            raise Stage2RC5PlusInferenceSealError(f"{name} budget mismatch")
        allowed = (budget[0] * total_pixels) // budget[1]
        rank = total_pixels - allowed - 1
        for field in (
            "allowed_strict_exceedances",
            "observed_strict_exceedances",
            "order_statistic_rank_zero_based",
        ):
            if type(row[field]) is not int or row[field] < 0:
                raise Stage2RC5PlusInferenceSealError(f"{name} integer row is invalid")
        if (
            row["allowed_strict_exceedances"] != allowed
            or row["order_statistic_rank_zero_based"] != rank
            or row["observed_strict_exceedances"] > allowed
        ):
            raise Stage2RC5PlusInferenceSealError(
                f"{name} strict-exceedance geometry mismatch"
            )
        thresholds.append(_float_hex(row["threshold_probability_hex"], f"{name}.threshold"))
        coordinates.append(_float_hex(row["threshold_coordinate_hex"], f"{name}.coordinate"))
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    if (
        not np.array_equal(encode_probability_numpy(threshold_array), coordinate_array)
        or tuple(row["threshold_kind"] for row in rows)
        != endpoint_kinds_numpy(coordinate_array)
        or (coordinate_array.size > 1 and np.any(coordinate_array[1:] < coordinate_array[:-1]))
    ):
        raise Stage2RC5PlusInferenceSealError(f"{name} EATC-v2 closure mismatch")
    return threshold_array, coordinate_array


def _validated_anchor_v2(
    value: VerifiedContextTailAnchorV2,
    *,
    context_identity_sha256: str,
    producer_anchor: VerifiedContextTailAnchor,
) -> _AnchorV2Material:
    anchor = assert_verified_context_tail_anchor_v2(value)
    payload = _fields(anchor.payload, _ANCHOR_V2_FIELDS, "anchor_v2")
    if (
        payload["schema_version"] != CONTEXT_TAIL_ANCHOR_V2_SCHEMA
        or payload["artifact_type"] != CONTEXT_TAIL_ANCHOR_V2_ARTIFACT_TYPE
        or payload["artifact_status"] != "complete"
        or payload["selection_algorithm"] != CONTEXT_TAIL_ANCHOR_V2_ALGORITHM
        or payload["context_probability_content_algorithm"]
        != CONTEXT_PROBABILITY_CONTENT_ALGORITHM
        or payload["budget_curve_coordinate_algorithm"]
        != BUDGET_CURVE_COORDINATE_ALGORITHM
        or payload["budget_order"] != "strictly_descending_loose_to_strict"
        or payload["threshold_representation"] != representation_contract()
        or payload["threshold_semantics"] != STRICT_THRESHOLD_SEMANTICS
        or payload["requested_anchor_source"]
        != "direct_same_budget_context_order_statistic_not_grid_interpolation"
    ):
        raise Stage2RC5PlusInferenceSealError("anchor-v2 frozen contract drifted")
    identity = _sha256(payload["anchor_identity_sha256"], "anchor-v2 identity")
    if identity != anchor_canonical_json_sha256(
        {key: item for key, item in payload.items() if key != "anchor_identity_sha256"}
    ):
        raise Stage2RC5PlusInferenceSealError("anchor-v2 self-hash mismatch")
    if payload["context_identity_sha256"] != context_identity_sha256:
        raise Stage2RC5PlusInferenceSealError("anchor-v2 context identity mismatch")
    if type(payload["context_size"]) is not int or payload["context_size"] != CONTEXT_SIZE:
        raise Stage2RC5PlusInferenceSealError("anchor-v2 context size mismatch")
    total = payload["total_context_pixels"]
    if type(total) is not int or total <= 0:
        raise Stage2RC5PlusInferenceSealError("anchor-v2 total pixels is invalid")
    maps = payload["context_map_bindings"]
    if isinstance(maps, (str, bytes)) or not isinstance(maps, Sequence) or len(maps) != CONTEXT_SIZE:
        raise Stage2RC5PlusInferenceSealError("anchor-v2 map bindings are invalid")
    total_bound = 0
    for index, raw in enumerate(maps):
        row = _fields(raw, _MAP_BINDING_FIELDS, f"anchor_v2.maps[{index}]")
        if row["ordinal"] != index or type(row["ordinal"]) is not int:
            raise Stage2RC5PlusInferenceSealError("anchor-v2 map ordinal mismatch")
        height, width, count = row["height"], row["width"], row["pixel_count"]
        if (
            any(type(item) is not int or item <= 0 for item in (height, width, count))
            or count != height * width
        ):
            raise Stage2RC5PlusInferenceSealError("anchor-v2 map geometry mismatch")
        _sha256(row["content_sha256"], "anchor-v2 map content")
        total_bound += count
    content = _sha256(
        payload["context_probability_content_sha256"],
        "anchor-v2 map-binding digest",
    )
    if total_bound != total or anchor_canonical_json_sha256(maps) != content:
        raise Stage2RC5PlusInferenceSealError("anchor-v2 map content closure mismatch")
    grid = _parse_budget_payload(payload["grid_budget_rationals"], "grid budgets")
    requested = _parse_budget_payload(
        payload["requested_budget_rationals"], "requested budgets"
    )
    if grid != BUDGET_KNOT_RATIONALS:
        raise Stage2RC5PlusInferenceSealError("anchor-v2 grid lattice mismatch")
    if requested:
        loose = Fraction(*BUDGET_KNOT_RATIONALS[0])
        strict = Fraction(*BUDGET_KNOT_RATIONALS[-1])
        if any(not strict <= Fraction(*row) <= loose for row in requested):
            raise Stage2RC5PlusInferenceSealError(
                "anchor-v2 requested budget lies outside the trained knot range"
            )
        log_positions = tuple(
            math.log(numerator) - math.log(denominator)
            for numerator, denominator in requested
        )
        if not all(
            left > right for left, right in zip(log_positions, log_positions[1:])
        ):
            raise Stage2RC5PlusInferenceSealError(
                "anchor-v2 requested budgets collide in float64 log-budget space"
            )
    grid_thresholds, grid_coordinates = _validate_row_set(
        payload["grid_threshold_rows"], grid, total_pixels=total, name="grid rows"
    )
    requested_thresholds, requested_coordinates = _validate_row_set(
        payload["requested_threshold_rows"],
        requested,
        total_pixels=total,
        name="requested rows",
    )
    if (
        tuple(anchor.grid_budget_rationals) != grid
        or tuple(anchor.requested_budget_rationals) != requested
        or tuple(float(value).hex() for value in anchor.grid_thresholds)
        != tuple(float(value).hex() for value in grid_thresholds)
        or tuple(float(value).hex() for value in anchor.grid_coordinates)
        != tuple(float(value).hex() for value in grid_coordinates)
        or tuple(float(value).hex() for value in anchor.requested_thresholds)
        != tuple(float(value).hex() for value in requested_thresholds)
        or tuple(float(value).hex() for value in anchor.requested_coordinates)
        != tuple(float(value).hex() for value in requested_coordinates)
    ):
        raise Stage2RC5PlusInferenceSealError(
            "anchor-v2 capability differs from its immutable payload"
        )
    guardrails = _fields(payload["guardrails"], _ANCHOR_GUARDRAILS, "anchor-v2 guardrails")
    if any(type(item) is not bool or item for item in guardrails.values()):
        raise Stage2RC5PlusInferenceSealError("anchor-v2 records forbidden access")

    # The freshly replayed v1 producer anchor is the current-state raw-map
    # authority.  Cross-generation equality prevents a free v2 anchor from
    # entering through this otherwise additive schema.
    producer = producer_anchor.payload
    if (
        payload["context_identity_sha256"] != producer["context_identity_sha256"]
        or total != producer["total_context_pixels"]
        or content != producer["context_probability_content_sha256"]
        or canonical_json_bytes(maps)
        != canonical_json_bytes(producer["context_map_bindings"])
    ):
        raise Stage2RC5PlusInferenceSealError(
            "anchor-v2 is not derived from the producer bundle's current context maps"
        )
    for v1_index, grid_index in enumerate(PRIMARY_BUDGET_KNOT_INDICES):
        if canonical_json_bytes(payload["grid_threshold_rows"][grid_index]) != canonical_json_bytes(
            producer["threshold_rows"][v1_index]
        ):
            raise Stage2RC5PlusInferenceSealError(
                "anchor-v2 primary row differs from current producer replay"
            )
    return _AnchorV2Material(
        capability=anchor,
        payload=payload,
        identity_sha256=identity,
        payload_sha256=hashlib.sha256(anchor_canonical_json_bytes(payload)).hexdigest(),
        grid_thresholds=np.array(grid_thresholds, copy=True),
        grid_coordinates=np.array(grid_coordinates, copy=True),
        requested_thresholds=np.array(requested_thresholds, copy=True),
        requested_coordinates=np.array(requested_coordinates, copy=True),
        requested_budgets=requested,
    )


def _standardized_masked_input(
    checkpoint_payload: Mapping[str, Any],
    context_values: np.ndarray,
) -> tuple[torch.Tensor, str, Any]:
    standardizer = checkpoint_payload["standardizer"]
    mean, scale = standardizer["mean"], standardizer["scale"]
    if (
        not isinstance(mean, torch.Tensor)
        or not isinstance(scale, torch.Tensor)
        or mean.dtype != torch.float64
        or scale.dtype != torch.float64
        or mean.device.type != "cpu"
        or scale.device.type != "cpu"
        or mean.shape != (93,)
        or scale.shape != (93,)
    ):
        raise Stage2RC5PlusInferenceSealError("checkpoint standardizer is invalid")
    mask = verify_stage2_rc5_feature_mask_payload(standardizer["feature_mask"])
    source = torch.from_numpy(np.array(context_values, dtype=np.float32, copy=True)).to(
        dtype=torch.float64
    )
    standardized = ((source - mean) / scale).to(dtype=torch.float32).reshape(1, 93)
    if not bool(torch.isfinite(standardized).all().item()):
        raise Stage2RC5PlusInferenceSealError("standardized input is non-finite")
    model_input = apply_stage2_rc5_feature_mask_torch(standardized, mask)
    little_endian = np.asarray(model_input.numpy().reshape(93), dtype="<f4")
    digest = hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()
    return model_input.contiguous(), digest, mask


def _output_array(output: Any, field: str, width: int) -> np.ndarray:
    value = getattr(output, field, None)
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.shape != (1, width)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Stage2RC5PlusInferenceSealError(f"model output {field} is invalid")
    return np.array(value.detach().numpy().reshape(width), dtype=np.float64, copy=True)


def _decision_rows(
    *,
    budgets: tuple[tuple[int, int], ...],
    anchor_thresholds: np.ndarray,
    anchor_coordinates: np.ndarray,
    anchor_latent: np.ndarray,
    residual: np.ndarray,
    transport_latent: np.ndarray,
    raw: np.ndarray,
    coordinates: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict[str, Any]]:
    kinds = endpoint_kinds_numpy(coordinates)
    return [
        {
            "budget_numerator": budget[0],
            "budget_denominator": budget[1],
            "anchor_threshold_probability_hex": float(anchor_threshold).hex(),
            "anchor_coordinate_hex": float(anchor_coordinate).hex(),
            "anchor_latent_hex": float(anchor_z).hex(),
            "context_residual_hex": float(context_residual).hex(),
            "transport_latent_hex": float(transport_z).hex(),
            "raw_coordinate_hex": float(raw_coordinate).hex(),
            "canonical_coordinate_hex": float(coordinate).hex(),
            "decoded_threshold_hex": float(threshold).hex(),
            "threshold_kind": kind,
        }
        for (
            budget,
            anchor_threshold,
            anchor_coordinate,
            anchor_z,
            context_residual,
            transport_z,
            raw_coordinate,
            coordinate,
            threshold,
            kind,
        ) in zip(
            budgets,
            anchor_thresholds,
            anchor_coordinates,
            anchor_latent,
            residual,
            transport_latent,
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
    producer_anchor: VerifiedContextTailAnchor,
    anchor_v2: VerifiedContextTailAnchorV2,
    producer_bundle_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    replayed_checkpoint, checkpoint_payload, model = _reverify_checkpoint(checkpoint)
    context_material = _context_inference_material(context)
    replayed_v1_anchor = _validated_anchor(
        producer_anchor,
        expected_context_identity_sha256=context_material.full_identity_sha256,
    )
    anchor = _validated_anchor_v2(
        anchor_v2,
        context_identity_sha256=context_material.full_identity_sha256,
        producer_anchor=replayed_v1_anchor.capability,
    )
    model_input, model_input_sha, feature_mask = _standardized_masked_input(
        checkpoint_payload, context_material.values
    )
    grid_anchor = torch.from_numpy(anchor.grid_coordinates.copy()).reshape(1, 9)
    model = model.to(device="cpu").eval()
    call: dict[str, Any] = {"anchor_coordinates": grid_anchor}
    if anchor.requested_budgets:
        call.update(
            {
                "budget_numerators": torch.tensor(
                    [row[0] for row in anchor.requested_budgets], dtype=torch.int64
                ),
                "budget_denominators": torch.tensor(
                    [row[1] for row in anchor.requested_budgets], dtype=torch.int64
                ),
                "requested_anchor_coordinates": torch.from_numpy(
                    anchor.requested_coordinates.copy()
                ).reshape(1, -1),
            }
        )
    with torch.inference_mode():
        output = model(model_input, **call)
    if not torch.equal(output.anchor_coordinates, grid_anchor):
        raise Stage2RC5PlusInferenceSealError("model changed grid anchor coordinates")
    grid_width = len(BUDGET_KNOT_RATIONALS)
    grid_anchor_latent = _output_array(output, "grid_anchor_latent", grid_width)
    grid_residual = _output_array(output, "grid_residual", grid_width)
    grid_transport = _output_array(output, "grid_transport_latent", grid_width)
    grid_raw = _output_array(output, "grid_raw_coordinates", grid_width)
    grid_coordinates = _output_array(output, "grid_coordinates", grid_width)
    grid_thresholds = _output_array(output, "grid_thresholds", grid_width)
    if not torch.equal(
        decode_coordinate_torch(torch.from_numpy(grid_coordinates)),
        torch.from_numpy(grid_thresholds),
    ):
        raise Stage2RC5PlusInferenceSealError("grid threshold decode mismatch")
    grid_rows = _decision_rows(
        budgets=BUDGET_KNOT_RATIONALS,
        anchor_thresholds=anchor.grid_thresholds,
        anchor_coordinates=anchor.grid_coordinates,
        anchor_latent=grid_anchor_latent,
        residual=grid_residual,
        transport_latent=grid_transport,
        raw=grid_raw,
        coordinates=grid_coordinates,
        thresholds=grid_thresholds,
    )
    requested_rows: list[dict[str, Any]] = []
    if anchor.requested_budgets:
        width = len(anchor.requested_budgets)
        if not torch.equal(
            output.requested_anchor_coordinates,
            torch.from_numpy(anchor.requested_coordinates.copy()).reshape(1, -1),
        ):
            raise Stage2RC5PlusInferenceSealError("model changed requested anchors")
        requested_coordinates = _output_array(output, "requested_coordinates", width)
        requested_thresholds = _output_array(output, "requested_thresholds", width)
        if not torch.equal(
            decode_coordinate_torch(torch.from_numpy(requested_coordinates)),
            torch.from_numpy(requested_thresholds),
        ):
            raise Stage2RC5PlusInferenceSealError("requested threshold decode mismatch")
        requested_rows = _decision_rows(
            budgets=anchor.requested_budgets,
            anchor_thresholds=anchor.requested_thresholds,
            anchor_coordinates=anchor.requested_coordinates,
            anchor_latent=_output_array(output, "requested_anchor_latent", width),
            residual=_output_array(output, "requested_residual", width),
            transport_latent=_output_array(output, "requested_transport_latent", width),
            raw=_output_array(output, "requested_raw_coordinates", width),
            coordinates=requested_coordinates,
            thresholds=requested_thresholds,
        )
    alpha = output.correction_strength
    beta = output.anchor_slope
    if (
        not isinstance(alpha, torch.Tensor)
        or alpha.dtype != torch.float64
        or alpha.shape != torch.Size([])
        or not isinstance(beta, torch.Tensor)
        or beta.dtype != torch.float64
        or beta.shape != (1, 1)
    ):
        raise Stage2RC5PlusInferenceSealError("transport scalar outputs are invalid")

    decision: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA,
        "artifact_type": DECISION_ARTIFACT_TYPE,
        "artifact_status": "complete",
        "decision_kind": "sealed_complete_exact_rational_threshold_function",
        "method": replayed_checkpoint.method,
        "deployed_rows_source": "requested" if requested_rows else "grid",
        "grid_budget_rationals": _budget_payload(BUDGET_KNOT_RATIONALS),
        "requested_budget_rationals": _budget_payload(anchor.requested_budgets),
        "correction_strength_hex": float(alpha.item()).hex(),
        "anchor_slope_hex": float(beta.item()).hex(),
        "grid_rows": grid_rows,
        "requested_rows": requested_rows,
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "labels_accessed": False,
        "query_accessed": False,
        "caller_float_budget_authority": False,
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
        "artifact_status": "complete",
        "causal_chain": CAUSAL_CHAIN,
        "method": replayed_checkpoint.method,
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
        },
        "producer_bundle_binding": dict(producer_bundle_binding),
        "context_binding": {
            "adapter": CONTEXT_ADAPTER,
            "context_payload_sha256": context_material.context_payload_sha256,
            "context_package_id": context_material.context_package_id,
            "context_full_identity_sha256": context_material.full_identity_sha256,
            "context_feature_vector_sha256": context_material.vector_sha256,
            "query_free_projection": True,
        },
        "anchor_v2_binding": {
            "anchor_schema": CONTEXT_TAIL_ANCHOR_V2_SCHEMA,
            "anchor_identity_sha256": anchor.identity_sha256,
            "anchor_payload_sha256": anchor.payload_sha256,
            "context_identity_sha256": anchor.payload["context_identity_sha256"],
            "context_probability_content_sha256": anchor.payload[
                "context_probability_content_sha256"
            ],
            "total_context_pixels": anchor.payload["total_context_pixels"],
            "primary_budget_cross_generation_match": True,
            "requested_anchor_source": anchor.payload["requested_anchor_source"],
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
        "grid_budget_rationals": _budget_payload(BUDGET_KNOT_RATIONALS),
        "requested_budget_rationals": _budget_payload(anchor.requested_budgets),
        "decision": decision,
        "guardrails": {key: False for key in _TRANSCRIPT_GUARDRAILS},
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    transcript["transcript_identity_sha256"] = _self_hash(
        transcript, "transcript_identity_sha256"
    )
    return transcript, canonical_json_bytes(transcript)


def _validated_producer_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = frozenset(
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
    binding = _fields(value, expected, "producer_bundle_binding")
    if (
        binding["capability_schema"] != BUNDLE_CAPABILITY_SCHEMA
        or binding["producer_manifest_schema"] != PRODUCER_MANIFEST_SCHEMA
        or binding["commit_schema"] != COMMIT_SCHEMA
    ):
        raise Stage2RC5PlusInferenceSealError("producer binding schema mismatch")
    for field in expected - {
        "capability_schema",
        "producer_manifest_schema",
        "commit_schema",
    }:
        _sha256(binding[field], f"producer_binding.{field}")
    return dict(binding)


def _recompute_public(
    *,
    checkpoint: VerifiedCalibratorCheckpointV8,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    anchor_v2: VerifiedContextTailAnchorV2,
) -> tuple[dict[str, Any], bytes]:
    bundle = _reverify_producer_bundle(producer_bundle)
    return _recompute_material(
        checkpoint=checkpoint,
        context=bundle.context,
        producer_anchor=bundle.anchor,
        anchor_v2=anchor_v2,
        producer_bundle_binding=_producer_bundle_binding(bundle),
    )


def infer_and_seal_stage2_rc5plus(
    *,
    checkpoint: VerifiedCalibratorCheckpointV8,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    anchor_v2: VerifiedContextTailAnchorV2,
) -> bytes:
    """Replay all authorities and emit one canonical sealed transcript."""

    _, data = _recompute_public(
        checkpoint=checkpoint,
        producer_bundle=producer_bundle,
        anchor_v2=anchor_v2,
    )
    return data


def _duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2RC5PlusInferenceSealError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _nonfinite_guard(value: str) -> None:
    raise Stage2RC5PlusInferenceSealError(f"non-finite JSON number: {value}")


def _parse_transcript(data: bytes) -> Mapping[str, Any]:
    if type(data) is not bytes or not data:
        raise TypeError("transcript must be nonempty bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_duplicate_guard,
            parse_constant=_nonfinite_guard,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2RC5PlusInferenceSealError("transcript is not canonical JSON") from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != data:
        raise Stage2RC5PlusInferenceSealError("transcript bytes are not canonical")
    if (
        value.get("schema_version") != TRANSCRIPT_SCHEMA
        or value.get("artifact_type") != TRANSCRIPT_ARTIFACT_TYPE
        or value.get("artifact_status") != "complete"
        or value.get("causal_chain") != CAUSAL_CHAIN
        or value.get("self_hash_algorithm") != SELF_HASH_ALGORITHM
        or value.get("transcript_identity_sha256")
        != _self_hash(value, "transcript_identity_sha256")
    ):
        raise Stage2RC5PlusInferenceSealError("transcript identity contract drifted")
    _validated_producer_binding(value.get("producer_bundle_binding"))
    guardrails = value.get("guardrails")
    if (
        not isinstance(guardrails, Mapping)
        or tuple(guardrails) != tuple(sorted(_TRANSCRIPT_GUARDRAILS))
        or any(type(item) is not bool or item for item in guardrails.values())
    ):
        raise Stage2RC5PlusInferenceSealError("transcript guardrails drifted")
    decision = value.get("decision")
    if (
        not isinstance(decision, Mapping)
        or decision.get("schema_version") != DECISION_SCHEMA
        or decision.get("artifact_type") != DECISION_ARTIFACT_TYPE
        or decision.get("artifact_status") != "complete"
        or decision.get("self_hash_algorithm") != SELF_HASH_ALGORITHM
        or decision.get("decision_identity_sha256")
        != _self_hash(decision, "decision_identity_sha256")
    ):
        raise Stage2RC5PlusInferenceSealError("decision identity contract drifted")
    return value


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5PlusInferenceSeal:
    transcript_bytes: bytes
    transcript_bytes_sha256: str
    transcript_identity_sha256: str
    decision_identity_sha256: str
    method: str
    transcript: Mapping[str, Any]
    decision: Mapping[str, Any]
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("VerifiedStage2RC5PlusInferenceSeal is verifier-issued only")


def _issue_verified(
    data: bytes, transcript: Mapping[str, Any]
) -> VerifiedStage2RC5PlusInferenceSeal:
    value = object.__new__(VerifiedStage2RC5PlusInferenceSeal)
    frozen = _freeze(transcript)
    for name, item in {
        "transcript_bytes": bytes(data),
        "transcript_bytes_sha256": hashlib.sha256(data).hexdigest(),
        "transcript_identity_sha256": transcript["transcript_identity_sha256"],
        "decision_identity_sha256": transcript["decision"]["decision_identity_sha256"],
        "method": transcript["method"],
        "transcript": frozen,
        "decision": frozen["decision"],
        "_capability": _CAPABILITY,
    }.items():
        object.__setattr__(value, name, item)
    return value


def _verify_material(
    data: bytes,
    *,
    checkpoint: VerifiedCalibratorCheckpointV8,
    context: Any,
    producer_anchor: VerifiedContextTailAnchor,
    anchor_v2: VerifiedContextTailAnchorV2,
    producer_bundle_binding: Mapping[str, Any],
) -> VerifiedStage2RC5PlusInferenceSeal:
    supplied = _parse_transcript(data)
    expected, expected_bytes = _recompute_material(
        checkpoint=checkpoint,
        context=context,
        producer_anchor=producer_anchor,
        anchor_v2=anchor_v2,
        producer_bundle_binding=_validated_producer_binding(producer_bundle_binding),
    )
    if not hmac.compare_digest(data, expected_bytes) or supplied != expected:
        raise Stage2RC5PlusInferenceSealError(
            "transcript differs byte-for-byte from full causal replay"
        )
    return _issue_verified(data, expected)


def verify_stage2_rc5plus_inference_seal(
    data: bytes,
    *,
    checkpoint: VerifiedCalibratorCheckpointV8,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    anchor_v2: VerifiedContextTailAnchorV2,
) -> VerifiedStage2RC5PlusInferenceSeal:
    supplied = _parse_transcript(data)
    expected, expected_bytes = _recompute_public(
        checkpoint=checkpoint,
        producer_bundle=producer_bundle,
        anchor_v2=anchor_v2,
    )
    if not hmac.compare_digest(data, expected_bytes) or supplied != expected:
        raise Stage2RC5PlusInferenceSealError(
            "transcript differs byte-for-byte from full causal replay"
        )
    return _issue_verified(data, expected)


def assert_verified_stage2_rc5plus_inference_seal(
    value: VerifiedStage2RC5PlusInferenceSeal,
) -> VerifiedStage2RC5PlusInferenceSeal:
    if (
        type(value) is not VerifiedStage2RC5PlusInferenceSeal
        or getattr(value, "_capability", None) is not _CAPABILITY
    ):
        raise TypeError("a verifier-issued RC5+ inference seal is required")
    return value


__all__ = [
    "CAUSAL_CHAIN",
    "CONTEXT_ADAPTER",
    "DECISION_ARTIFACT_TYPE",
    "DECISION_SCHEMA",
    "MODEL_INPUT_DIGEST_ALGORITHM",
    "SELF_HASH_ALGORITHM",
    "Stage2RC5PlusInferenceSealError",
    "TRANSCRIPT_ARTIFACT_TYPE",
    "TRANSCRIPT_SCHEMA",
    "VerifiedStage2RC5PlusInferenceSeal",
    "assert_verified_stage2_rc5plus_inference_seal",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "infer_and_seal_stage2_rc5plus",
    "verify_stage2_rc5plus_inference_seal",
]

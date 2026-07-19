"""RC5-only atomic T0--T8 pre-label threshold-decision authority.

This additive module never imports the legacy RC4 threshold-family or
threshold-decision modules.  Every complete row carries the frozen exact
budget rational plus EATC-v2 probability/coordinate hexadecimal values and
the endpoint kind.  T5 is either a verifier-issued complete EVT curve or an
explicitly sealed missing method with no fallback.  T9 has a separate
post-label schema and can never enter the pre-label method set.

Publication consists of one canonical decision-set member followed by one
commit marker written last.  Public verification re-derives the complete set
from verifier capabilities and compares canonical bytes; self-hashes alone
are never semantic authority.  The guarded label-resolver entry point invokes
its callback only after this replay succeeds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType
from typing import Any

import numpy as np

from model.endpoint_aware_threshold import (
    decode_coordinate_numpy,
    encode_probability_numpy,
    endpoint_kinds_numpy,
    representation_contract,
)
from rc.build_stage2_rc5_context import (
    VerifiedStage2RC5ContextBundle,
    assert_verified_stage2_rc5_context_bundle,
    replay_verified_stage2_rc5_context_bundle,
)
from rc.stage2_context_tail_anchor import (
    BUDGET_RATIONALS,
    STRICT_THRESHOLD_SEMANTICS,
    assert_verified_context_tail_anchor,
)
from rc.stage2_crossfit_schema_v6 import context_inference_material_v2
from rc.stage2_exact_oracle_v2 import select_exact_oracle_v2
from rc.stage2_rc5_infer_and_seal import (
    TRANSCRIPT_SCHEMA as INFERENCE_TRANSCRIPT_SCHEMA,
    VerifiedStage2RC5InferenceSeal,
    assert_verified_stage2_rc5_inference_seal,
)
from rc.stage2_rc5_source_reference_v3 import (
    VerifiedStage2RC5SourceReferenceV3,
    assert_verified_stage2_rc5_source_reference_v3,
    replay_verified_stage2_rc5_source_reference_v3,
)


SOURCE_CURVE_SCHEMA = "rc-irstd.stage2-exact-source-domain-curve.v2"
SOURCE_REFERENCE_SCHEMA = "rc-irstd.stage2-exact-source-threshold-reference.v3"
EVT_SEAL_SCHEMA = "rc-irstd.stage2-rc5-evt-threshold-seal.v1"
DECISION_SCHEMA = "rc-irstd.stage2-rc5-prelabel-threshold-decision.v3"
DECISION_SET_SCHEMA = "rc-irstd.stage2-rc5-atomic-decision-set.v3"
DECISION_SET_COMMIT_SCHEMA = "rc-irstd.stage2-rc5-atomic-decision-set-commit.v3"
T9_DIAGNOSTIC_SCHEMA = "rc-irstd.stage2-rc5-postlabel-t9-diagnostic.v2"

DECISION_SET_ARTIFACT_TYPE = "rc_irstd_stage2_rc5_atomic_t0_t8_decision_set"
DECISION_SET_COMMIT_ARTIFACT_TYPE = (
    "rc_irstd_stage2_rc5_atomic_t0_t8_decision_set_commit"
)
EVT_SEAL_ARTIFACT_TYPE = "rc_irstd_stage2_rc5_evt_threshold_seal"
SOURCE_REFERENCE_ARTIFACT_TYPE = (
    "rc_irstd_stage2_exact_source_threshold_reference"
)

DECISION_SET_FILENAME = "rc5_t0_t8_decision_set.json"
DECISION_SET_COMMIT_FILENAME = "rc5_t0_t8_decision_set.commit.json"
PUBLICATION_ORDER = "canonical_decision_set_then_commit_last"
SELF_HASH_ALGORITHM = "sha256-canonical-json-with-self-field-omitted-v1"
EXACT_BUDGET_COUNT_RULE = "(numerator * total_native_pixels) // denominator"
METHOD_IDS = tuple(f"T{index}" for index in range(9))
LEARNED_METHOD_IDS = ("T6", "T7", "T8")
METHOD_NAMES = {
    "T0": "fixed_0.5",
    "T1": "pooled_exact_source_safe",
    "T2": "safer_of_two_exact_source_thresholds",
    "T3": "nearest_source_exact_safe",
    "T4": "verified_context_tail_anchor",
    "T5": "prelabel_evt_gpd_optional",
    "T6": "direct_endpoint_aware_calibrator",
    "T7": "monotone_endpoint_aware_calibrator",
    "T8": "risk_aligned_monotone_endpoint_aware_calibrator",
}
SHARED_IDENTITY_SCHEMA = "rc-irstd.stage2-rc5-shared-prelabel-identity.v2"

_SHA_CHARS = frozenset("0123456789abcdef")
_SOURCE_CURVE_TOKEN = object()
_SOURCE_REFERENCE_TOKEN = object()
_EVT_TOKEN = object()
_DECISION_SET_TOKEN = object()
_INT64_MAX = int(np.iinfo(np.int64).max)


class Stage2RC5AtomicDecisionSetError(ValueError):
    """An RC5 exact source reference or atomic decision set failed closed."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
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
        raise Stage2RC5AtomicDecisionSetError(
            f"value is not canonical JSON: {error}"
        ) from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def _sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _SHA_CHARS
    ):
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} must be one lowercase SHA-256 digest"
        )
    return value


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} must be nonempty canonical text"
        )
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if (
        type(value) is not int
        or value < minimum
        or value > _INT64_MAX
    ):
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} must be an exact int in [{minimum}, {_INT64_MAX}]"
        )
    return value


def _exact_false(value: Any, name: str) -> None:
    if value is not False:
        raise Stage2RC5AtomicDecisionSetError(f"{name} must be exact false")


def _canonical_float_hex(value: Any, name: str) -> float:
    if type(value) is not str:
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} must be canonical float.hex text"
        )
    try:
        result = float.fromhex(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} is not float.hex"
        ) from error
    if not math.isfinite(result) or result.hex() != value:
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} must be finite canonical float.hex"
        )
    return result


def _budget_payload() -> list[dict[str, int]]:
    return [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in BUDGET_RATIONALS
    ]


def _readonly_array(value: Any, dtype: Any) -> np.ndarray:
    owned = np.array(value, dtype=dtype, order="C", copy=True)
    result = np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype)
    result.setflags(write=False)
    return result


def _float64_vector(value: Any, name: str, *, minimum_size: int = 2) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an explicit numpy array")
    if value.dtype != np.float64 or value.ndim != 1 or value.size < minimum_size:
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} must be one float64 vector with at least {minimum_size} values"
        )
    result = _readonly_array(value, np.float64)
    if not np.isfinite(result).all():
        raise Stage2RC5AtomicDecisionSetError(f"{name} must be finite")
    return result


def _int64_vector(value: Any, name: str, *, size: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an explicit numpy array")
    if value.dtype != np.int64 or value.ndim != 1 or value.size != size:
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} must be one aligned int64 vector"
        )
    result = _readonly_array(value, np.int64)
    if np.any(result < 0):
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} must contain nonnegative exact counts"
        )
    return result


def _digest_frame(digest: Any, value: bytes | memoryview) -> None:
    raw = memoryview(value).cast("B")
    digest.update(int(raw.nbytes).to_bytes(8, "big"))
    digest.update(raw)


def _detector_identity(value: Any) -> tuple[dict[str, Any], str]:
    required = {
        "run_id",
        "outer_fold_id",
        "outer_target",
        "base_seed",
        "derived_seed",
        "detector_role",
        "oof_fold_index",
        "checkpoint_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise Stage2RC5AtomicDecisionSetError(
            "detector_identity fields do not match the frozen source contract"
        )
    result = {
        "run_id": _text(value["run_id"], "detector_identity.run_id"),
        "outer_fold_id": _text(
            value["outer_fold_id"], "detector_identity.outer_fold_id"
        ),
        "outer_target": _text(
            value["outer_target"], "detector_identity.outer_target"
        ),
        "base_seed": _strict_int(
            value["base_seed"], "detector_identity.base_seed"
        ),
        "derived_seed": _strict_int(
            value["derived_seed"],
            "detector_identity.derived_seed",
            minimum=1,
        ),
        "detector_role": _text(
            value["detector_role"], "detector_identity.detector_role"
        ),
        "oof_fold_index": value["oof_fold_index"],
        "checkpoint_sha256": _sha256(
            value["checkpoint_sha256"],
            "detector_identity.checkpoint_sha256",
        ),
    }
    if result["detector_role"] == "detector_oof":
        if type(result["oof_fold_index"]) is not int or result[
            "oof_fold_index"
        ] not in {0, 1}:
            raise Stage2RC5AtomicDecisionSetError(
                "detector_oof requires oof_fold_index 0 or 1"
            )
    elif result["oof_fold_index"] is not None:
        raise Stage2RC5AtomicDecisionSetError(
            "non-OOF detector must have null oof_fold_index"
        )
    return result, canonical_json_sha256(result)


@dataclass(frozen=True, init=False)
class VerifiedExactSourceDomainCurveV2:
    source_domain: str
    detector_identity_sha256: str
    thresholds: np.ndarray
    false_positive_pixels: np.ndarray
    matched_objects: np.ndarray
    total_native_pixels: int
    ground_truth_objects: int
    curve_identity_sha256: str
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("VerifiedExactSourceDomainCurveV2 is verifier-issued only")


def build_exact_source_domain_curve_v2(
    *,
    source_domain: str,
    detector_identity_sha256: str,
    thresholds: np.ndarray,
    false_positive_pixels: np.ndarray,
    matched_objects: np.ndarray,
    total_native_pixels: int,
    ground_truth_objects: int,
) -> VerifiedExactSourceDomainCurveV2:
    """Validate one source-only exact event curve and issue a capability."""

    domain = _text(source_domain, "source_domain")
    detector_sha = _sha256(
        detector_identity_sha256, "detector_identity_sha256"
    )
    threshold = _float64_vector(thresholds, "thresholds")
    fp = _int64_vector(
        false_positive_pixels,
        "false_positive_pixels",
        size=threshold.size,
    )
    tp = _int64_vector(
        matched_objects, "matched_objects", size=threshold.size
    )
    total = _strict_int(
        total_native_pixels, "total_native_pixels", minimum=1
    )
    objects = _strict_int(
        ground_truth_objects, "ground_truth_objects"
    )
    if threshold[0] == 0.0 and np.signbit(threshold[0]):
        raise Stage2RC5AtomicDecisionSetError(
            "source curve lower endpoint must be exact positive zero"
        )
    # The public exact-oracle core performs the complete strict-curve audit.
    select_exact_oracle_v2(
        thresholds=threshold,
        false_positive_pixels=fp,
        matched_objects=tp,
        total_native_pixels=total,
        ground_truth_objects=objects,
    )
    header = canonical_json_bytes(
        {
            "schema_version": SOURCE_CURVE_SCHEMA,
            "source_domain": domain,
            "detector_identity_sha256": detector_sha,
            "row_count": int(threshold.size),
            "total_native_pixels": total,
            "ground_truth_objects": objects,
            "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        }
    )
    digest = hashlib.sha256()
    _digest_frame(digest, header)
    for name, values, dtype in (
        (b"thresholds", threshold, "<f8"),
        (b"false_positive_pixels", fp, "<i8"),
        (b"matched_objects", tp, "<i8"),
    ):
        _digest_frame(digest, name)
        _digest_frame(
            digest,
            memoryview(np.ascontiguousarray(values, dtype=dtype)),
        )
    value = object.__new__(VerifiedExactSourceDomainCurveV2)
    object.__setattr__(value, "source_domain", domain)
    object.__setattr__(value, "detector_identity_sha256", detector_sha)
    object.__setattr__(value, "thresholds", threshold)
    object.__setattr__(value, "false_positive_pixels", fp)
    object.__setattr__(value, "matched_objects", tp)
    object.__setattr__(value, "total_native_pixels", total)
    object.__setattr__(value, "ground_truth_objects", objects)
    object.__setattr__(value, "curve_identity_sha256", digest.hexdigest())
    object.__setattr__(value, "_capability", _SOURCE_CURVE_TOKEN)
    return value


def assert_verified_exact_source_domain_curve_v2(
    value: Any,
) -> VerifiedExactSourceDomainCurveV2:
    if (
        type(value) is not VerifiedExactSourceDomainCurveV2
        or getattr(value, "_capability", None) is not _SOURCE_CURVE_TOKEN
    ):
        raise TypeError("a verifier-issued exact source-domain curve is required")
    return value


def _threshold_row(
    *,
    index: int,
    probability: float,
    coordinate: float | None = None,
    allowed_fp: int | None = None,
    observed_fp: int | None = None,
    matched_objects: int | None = None,
    total_native_pixels: int | None = None,
    ground_truth_objects: int | None = None,
    relation: str = "encode_probability",
) -> dict[str, Any]:
    numerator, denominator = BUDGET_RATIONALS[index]
    probability_array = np.asarray([probability], dtype=np.float64)
    if (
        not np.isfinite(probability_array).all()
        or np.any((probability_array < 0.0) | (probability_array > 1.0))
        or (probability_array[0] == 0.0 and np.signbit(probability_array[0]))
    ):
        raise Stage2RC5AtomicDecisionSetError("threshold probability is invalid")
    if relation not in {"encode_probability", "decode_coordinate"}:
        raise Stage2RC5AtomicDecisionSetError(
            "threshold probability/coordinate relation is invalid"
        )
    if coordinate is None:
        if relation != "encode_probability":
            raise Stage2RC5AtomicDecisionSetError(
                "decode-authoritative threshold row requires a coordinate"
            )
        canonical = float(encode_probability_numpy(probability_array)[0])
    else:
        canonical = float(coordinate)
        if relation == "encode_probability":
            expected_coordinate = float(
                encode_probability_numpy(probability_array)[0]
            )
            if canonical.hex() != expected_coordinate.hex():
                raise Stage2RC5AtomicDecisionSetError(
                    "threshold coordinate does not encode its probability"
                )
        else:
            try:
                decoded = float(
                    decode_coordinate_numpy(
                        np.asarray([canonical], dtype=np.float64)
                    )[0]
                )
            except ValueError as error:
                raise Stage2RC5AtomicDecisionSetError(
                    "learned threshold coordinate is not canonical EATC-v2"
                ) from error
            if decoded.hex() != float(probability).hex():
                raise Stage2RC5AtomicDecisionSetError(
                    "learned threshold probability does not decode from coordinate"
                )
    kind = endpoint_kinds_numpy(np.asarray([canonical], dtype=np.float64))[0]
    row: dict[str, Any] = {
        "budget_numerator": numerator,
        "budget_denominator": denominator,
        "threshold_probability_hex": float(probability).hex(),
        "threshold_coordinate_hex": canonical.hex(),
        "threshold_kind": kind,
        "probability_coordinate_relation": relation,
    }
    exact_fields = (
        allowed_fp,
        observed_fp,
        matched_objects,
        total_native_pixels,
        ground_truth_objects,
    )
    if any(value is not None for value in exact_fields):
        if any(value is None for value in exact_fields):
            raise Stage2RC5AtomicDecisionSetError(
                "exact source row count fields must be all present or all absent"
            )
        row.update(
            {
                "allowed_false_positive_pixels": _strict_int(
                    allowed_fp, "allowed_false_positive_pixels"
                ),
                "observed_false_positive_pixels": _strict_int(
                    observed_fp, "observed_false_positive_pixels"
                ),
                "matched_objects": _strict_int(
                    matched_objects, "matched_objects"
                ),
                "total_native_pixels": _strict_int(
                    total_native_pixels, "total_native_pixels", minimum=1
                ),
                "ground_truth_objects": _strict_int(
                    ground_truth_objects, "ground_truth_objects"
                ),
            }
        )
    return row


def _oracle_rows_from_selection(selection: Any) -> list[dict[str, Any]]:
    rows = []
    for index in range(3):
        rows.append(
            _threshold_row(
                index=index,
                probability=float(selection.thresholds[index]),
                coordinate=float(selection.coordinates[index]),
                allowed_fp=int(selection.allowed_false_positive_counts[index]),
                observed_fp=int(selection.false_positive_pixels[index]),
                matched_objects=int(selection.matched_objects[index]),
                total_native_pixels=int(selection.total_native_pixels),
                ground_truth_objects=int(selection.ground_truth_objects),
            )
        )
    return rows


def _pooled_exact_oracle_rows(
    curves: Sequence[VerifiedExactSourceDomainCurveV2],
) -> list[dict[str, Any]]:
    """Select pooled rows with a descending exact-count k-way tail merge."""

    total = sum(curve.total_native_pixels for curve in curves)
    objects = sum(curve.ground_truth_objects for curve in curves)
    _strict_int(total, "pooled total_native_pixels", minimum=1)
    _strict_int(objects, "pooled ground_truth_objects")
    allowed = tuple(
        (numerator * total) // denominator
        for numerator, denominator in BUDGET_RATIONALS
    )
    current_fp = sum(int(curve.false_positive_pixels[-1]) for curve in curves)
    current_tp = sum(int(curve.matched_objects[-1]) for curve in curves)
    best: list[tuple[tuple[int, int, float], float, int, int] | None] = [
        None,
        None,
        None,
    ]

    def consider(threshold: float) -> None:
        key = (current_tp, -current_fp, threshold)
        for index, maximum in enumerate(allowed):
            if current_fp <= maximum and (
                best[index] is None or key > best[index][0]
            ):
                best[index] = (key, threshold, current_fp, current_tp)

    consider(1.0)
    heap: list[tuple[float, int, int]] = []
    for curve_index, curve in enumerate(curves):
        row_index = int(curve.thresholds.size) - 2
        heapq.heappush(
            heap,
            (-float(curve.thresholds[row_index]), curve_index, row_index),
        )
    while heap:
        threshold = -heap[0][0]
        events: list[tuple[int, int]] = []
        while heap and -heap[0][0] == threshold:
            _, curve_index, row_index = heapq.heappop(heap)
            events.append((curve_index, row_index))
        next_events: list[tuple[float, int, int]] = []
        for curve_index, row_index in events:
            curve = curves[curve_index]
            previous = row_index + 1
            current_fp += int(curve.false_positive_pixels[row_index])
            current_fp -= int(curve.false_positive_pixels[previous])
            current_tp += int(curve.matched_objects[row_index])
            current_tp -= int(curve.matched_objects[previous])
            if row_index > 0:
                next_events.append(
                    (
                        -float(curve.thresholds[row_index - 1]),
                        curve_index,
                        row_index - 1,
                    )
                )
        if current_fp > allowed[0]:
            break
        consider(threshold)
        for event in next_events:
            heapq.heappush(heap, event)
    if any(row is None for row in best):
        raise RuntimeError("exact threshold=1 did not satisfy pooled budgets")
    selected = [row for row in best if row is not None]
    thresholds = [float(row[1]) for row in selected]
    if any(right < left for left, right in zip(thresholds, thresholds[1:])):
        raise RuntimeError("pooled exact source thresholds decreased")
    return [
        _threshold_row(
            index=index,
            probability=thresholds[index],
            allowed_fp=allowed[index],
            observed_fp=int(selected[index][2]),
            matched_objects=int(selected[index][3]),
            total_native_pixels=total,
            ground_truth_objects=objects,
        )
        for index in range(3)
    ]


@dataclass(frozen=True, init=False)
class VerifiedExactSourceThresholdReferenceV3:
    payload: Mapping[str, Any]
    canonical_bytes: bytes
    reference_identity_sha256: str
    detector_identity_sha256: str
    source_reference_identity_sha256: str
    source_reference_attestation_sha256: str
    source_reference_v3: VerifiedStage2RC5SourceReferenceV3
    source_domains: tuple[str, str]
    source_centers: np.ndarray
    source_scale: np.ndarray
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "VerifiedExactSourceThresholdReferenceV3 is verifier-issued only"
        )


def build_exact_source_threshold_reference_v3(
    *,
    domain_curves: Mapping[str, VerifiedExactSourceDomainCurveV2],
    source_reference: VerifiedStage2RC5SourceReferenceV3,
) -> VerifiedExactSourceThresholdReferenceV3:
    """Build T1/T2/T3 only from fresh-replayed RC5 source authority."""

    source = replay_verified_stage2_rc5_source_reference_v3(
        assert_verified_stage2_rc5_source_reference_v3(source_reference)
    )
    detector, detector_sha = _detector_identity(source.detector_identity)
    source_reference_sha = _sha256(
        source.attestation["attestation_identity_sha256"],
        "source_reference.attestation_identity_sha256",
    )
    source_attestation_sha = _sha256(
        source.attestation_sha256,
        "source_reference.attestation_sha256",
    )
    base = source.source_reference_v2.source_reference_bundle
    if not isinstance(domain_curves, Mapping) or len(domain_curves) != 2:
        raise Stage2RC5AtomicDecisionSetError(
            "exact source reference requires exactly two domain curves"
        )
    authoritative_domains = tuple(str(domain) for domain in base.domains)
    domains = tuple(sorted(authoritative_domains))
    if len(set(domains)) != 2 or detector["outer_target"] in domains:
        raise Stage2RC5AtomicDecisionSetError(
            "source domains must be two unique non-outer domains"
        )
    if set(domain_curves) != set(authoritative_domains):
        raise Stage2RC5AtomicDecisionSetError(
            "exact source curves do not cover the v3 source domains"
        )
    curves = tuple(
        assert_verified_exact_source_domain_curve_v2(domain_curves[domain])
        for domain in domains
    )
    for domain, curve in zip(domains, curves, strict=True):
        if (
            curve.source_domain != domain
            or curve.detector_identity_sha256 != detector_sha
        ):
            raise Stage2RC5AtomicDecisionSetError(
                "source curve domain/detector identity mismatch"
            )
    authoritative_centers = {
        domain: np.asarray(center, dtype=np.float32)
        for domain, center in zip(
            authoritative_domains, base.centers, strict=True
        )
    }
    centers = []
    for domain in domains:
        raw = authoritative_centers[domain]
        if (
            not isinstance(raw, np.ndarray)
            or raw.dtype != np.float32
            or raw.shape != (87,)
            or not np.isfinite(raw).all()
        ):
            raise Stage2RC5AtomicDecisionSetError(
                f"source_centers.{domain} must be finite float32[87]"
            )
        centers.append(np.array(raw, dtype=np.float32, copy=True))
    source_scale = np.asarray(base.scale, dtype=np.float32)
    if source_scale.shape != (87,) or not np.isfinite(source_scale).all() or np.any(
        source_scale <= 0.0
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "source_scale must be positive finite float32[87]"
        )
    center_matrix = _readonly_array(np.stack(centers), np.float32).reshape(2, 87)
    center_matrix.setflags(write=False)
    scale = _readonly_array(source_scale, np.float32)
    domain_rows: dict[str, list[dict[str, Any]]] = {}
    for domain, curve in zip(domains, curves, strict=True):
        selection = select_exact_oracle_v2(
            thresholds=curve.thresholds,
            false_positive_pixels=curve.false_positive_pixels,
            matched_objects=curve.matched_objects,
            total_native_pixels=curve.total_native_pixels,
            ground_truth_objects=curve.ground_truth_objects,
        )
        domain_rows[domain] = _oracle_rows_from_selection(selection)
    pooled_rows = _pooled_exact_oracle_rows(curves)
    safer_rows = []
    for index in range(3):
        probabilities = [
            float.fromhex(domain_rows[domain][index]["threshold_probability_hex"])
            for domain in domains
        ]
        safer_rows.append(
            _threshold_row(index=index, probability=max(probabilities))
        )
    payload: dict[str, Any] = {
        "schema_version": SOURCE_REFERENCE_SCHEMA,
        "artifact_type": SOURCE_REFERENCE_ARTIFACT_TYPE,
        "artifact_status": "source_only_exact_complete",
        "outer_target_present": False,
        "official_test_accessed": False,
        "detector_identity": detector,
        "detector_identity_sha256": detector_sha,
        "source_reference_identity_sha256": source_reference_sha,
        "source_reference_authority": {
            "capability_schema": source.capability_schema,
            "attestation_sha256": source_attestation_sha,
            "attestation_identity_sha256": source_reference_sha,
            "base_npz_sha256": source.npz_sha256,
            "base_audit_sha256": source.audit_sha256,
            "source_score_attestations": [
                {
                    "source_domain": row["source_domain"],
                    "sha256": row["score_attestation"]["sha256"],
                }
                for row in source.attestation["source_score_bundles"]
            ],
            "shared_run_complete": _plain(
                source.attestation["source_score_bundles"][0]["run_complete"]
            ),
        },
        "source_domains": list(domains),
        "source_curve_bindings": {
            domain: {
                "curve_identity_sha256": curve.curve_identity_sha256,
                "total_native_pixels": curve.total_native_pixels,
                "ground_truth_objects": curve.ground_truth_objects,
                "row_count": int(curve.thresholds.size),
            }
            for domain, curve in zip(domains, curves, strict=True)
        },
        "budget_order": "strictly_descending_loose_to_strict",
        "budget_rationals": _budget_payload(),
        "budget_count_rule": EXACT_BUDGET_COUNT_RULE,
        "selection_rule": "max_tp_then_min_fp_then_max_threshold",
        "pooled_safe_rows": pooled_rows,
        "domain_safe_rows": domain_rows,
        "safer_safe_rows": safer_rows,
        "nearest_source_rule": (
            "argmin_domain_l2((context_first87-source_center)/source_scale)_"
            "lexicographic_tie"
        ),
        "source_centers_float32_hex": {
            domain: [float(value).hex() for value in center_matrix[index]]
            for index, domain in enumerate(domains)
        },
        "source_scale_float32_hex": [float(value).hex() for value in scale],
        "threshold_representation": representation_contract(),
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "guardrails": {
            "outer_target_scores_accessed": False,
            "outer_target_labels_accessed": False,
            "postlabel_statistics_accessed": False,
            "float_budget_count_logic_used": False,
        },
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    payload["reference_identity_sha256"] = _self_hash(
        payload, "reference_identity_sha256"
    )
    data = canonical_json_bytes(payload)
    value = object.__new__(VerifiedExactSourceThresholdReferenceV3)
    object.__setattr__(value, "payload", _freeze(payload))
    object.__setattr__(value, "canonical_bytes", data)
    object.__setattr__(
        value, "reference_identity_sha256", payload["reference_identity_sha256"]
    )
    object.__setattr__(value, "detector_identity_sha256", detector_sha)
    object.__setattr__(
        value, "source_reference_identity_sha256", source_reference_sha
    )
    object.__setattr__(
        value,
        "source_reference_attestation_sha256",
        source_attestation_sha,
    )
    object.__setattr__(value, "source_reference_v3", source)
    object.__setattr__(value, "source_domains", domains)
    object.__setattr__(value, "source_centers", center_matrix)
    object.__setattr__(value, "source_scale", scale)
    object.__setattr__(value, "_capability", _SOURCE_REFERENCE_TOKEN)
    return value


def assert_verified_exact_source_threshold_reference_v3(
    value: Any,
) -> VerifiedExactSourceThresholdReferenceV3:
    if (
        type(value) is not VerifiedExactSourceThresholdReferenceV3
        or getattr(value, "_capability", None) is not _SOURCE_REFERENCE_TOKEN
        or canonical_json_bytes(value.payload) != value.canonical_bytes
        or value.payload["reference_identity_sha256"]
        != _self_hash(value.payload, "reference_identity_sha256")
        or value.reference_identity_sha256
        != value.payload["reference_identity_sha256"]
        or value.detector_identity_sha256
        != value.payload["detector_identity_sha256"]
        or value.source_reference_identity_sha256
        != value.payload["source_reference_identity_sha256"]
        or value.source_reference_attestation_sha256
        != value.payload["source_reference_authority"]["attestation_sha256"]
    ):
        raise TypeError(
            "a verifier-issued exact source threshold reference-v2 is required"
        )
    assert_verified_stage2_rc5_source_reference_v3(value.source_reference_v3)
    if tuple(value.payload["source_domains"]) != value.source_domains:
        raise TypeError("exact source threshold reference domain state drifted")
    payload_centers = np.asarray(
        [
            [float.fromhex(item) for item in value.payload[
                "source_centers_float32_hex"
            ][domain]]
            for domain in value.source_domains
        ],
        dtype=np.float64,
    )
    payload_scale = np.asarray(
        [float.fromhex(item) for item in value.payload["source_scale_float32_hex"]],
        dtype=np.float64,
    )
    if (
        value.source_centers.flags.writeable
        or value.source_scale.flags.writeable
        or not np.array_equal(
            np.asarray(value.source_centers, dtype=np.float64), payload_centers
        )
        or not np.array_equal(
            np.asarray(value.source_scale, dtype=np.float64), payload_scale
        )
    ):
        raise TypeError("exact source threshold reference numeric state drifted")
    return value


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5EVTSeal:
    payload: Mapping[str, Any]
    seal_identity_sha256: str
    thresholds: tuple[float, float, float]
    coordinates: tuple[float, float, float]
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("VerifiedStage2RC5EVTSeal is verifier-issued only")


def build_stage2_rc5_evt_seal_complete(
    *,
    producer_identity_sha256: str,
    context_full_identity_sha256: str,
    anchor_identity_sha256: str,
    thresholds: np.ndarray,
    fit_identity_sha256: str,
) -> VerifiedStage2RC5EVTSeal:
    """Seal one complete pre-label EVT curve; no fallback is representable."""

    values = _float64_vector(thresholds, "EVT thresholds", minimum_size=3)
    if values.shape != (3,) or np.any(values[1:] < values[:-1]):
        raise Stage2RC5AtomicDecisionSetError(
            "EVT thresholds must be one nondecreasing float64[3] curve"
        )
    coordinates = encode_probability_numpy(values)
    rows = [
        _threshold_row(
            index=index,
            probability=float(values[index]),
            coordinate=float(coordinates[index]),
        )
        for index in range(3)
    ]
    payload: dict[str, Any] = {
        "schema_version": EVT_SEAL_SCHEMA,
        "artifact_type": EVT_SEAL_ARTIFACT_TYPE,
        "artifact_status": "prelabel_complete",
        "outcome": "complete",
        "producer_identity_sha256": _sha256(
            producer_identity_sha256, "producer_identity_sha256"
        ),
        "context_full_identity_sha256": _sha256(
            context_full_identity_sha256,
            "context_full_identity_sha256",
        ),
        "anchor_identity_sha256": _sha256(
            anchor_identity_sha256, "anchor_identity_sha256"
        ),
        "fit_identity_sha256": _sha256(
            fit_identity_sha256, "fit_identity_sha256"
        ),
        "budget_order": "strictly_descending_loose_to_strict",
        "budget_rationals": _budget_payload(),
        "threshold_representation": representation_contract(),
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "rows": rows,
        "labels_accessed": False,
        "query_accessed": False,
        "reject": False,
        "fallback": False,
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    payload["seal_identity_sha256"] = _self_hash(
        payload, "seal_identity_sha256"
    )
    value = object.__new__(VerifiedStage2RC5EVTSeal)
    object.__setattr__(value, "payload", _freeze(payload))
    object.__setattr__(
        value, "seal_identity_sha256", payload["seal_identity_sha256"]
    )
    object.__setattr__(value, "thresholds", tuple(float(item) for item in values))
    object.__setattr__(
        value,
        "coordinates",
        tuple(float(item) for item in coordinates),
    )
    object.__setattr__(value, "_capability", _EVT_TOKEN)
    return value


def assert_verified_stage2_rc5_evt_seal(
    value: Any,
) -> VerifiedStage2RC5EVTSeal:
    if (
        type(value) is not VerifiedStage2RC5EVTSeal
        or getattr(value, "_capability", None) is not _EVT_TOKEN
        or value.payload["seal_identity_sha256"]
        != _self_hash(value.payload, "seal_identity_sha256")
    ):
        raise TypeError("a verifier-issued complete RC5 EVT seal is required")
    return value


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage2RC5AtomicDecisionSetError(f"{name} must be a mapping")
    return value


def _shared_identity_from_bundle(
    producer_bundle: VerifiedStage2RC5ContextBundle,
) -> tuple[dict[str, Any], np.ndarray]:
    """Close producer/window/query/score/detector identities in one hash."""

    bundle = replay_verified_stage2_rc5_context_bundle(
        assert_verified_stage2_rc5_context_bundle(producer_bundle)
    )
    manifest = _required_mapping(bundle.producer_manifest, "producer_manifest")
    commit = _required_mapping(bundle.commit, "context bundle commit")
    context = bundle.context
    context_payload = _required_mapping(context.payload, "context payload")
    anchor = assert_verified_context_tail_anchor(bundle.anchor)
    anchor_payload = _required_mapping(anchor.payload, "anchor payload")
    window = bundle.variable_query_window
    score = bundle.score_manifest_metadata
    source_reference = bundle.source_reference

    producer_identity = _sha256(
        manifest.get("producer_identity_sha256"),
        "producer_manifest.producer_identity_sha256",
    )
    if producer_identity != _self_hash(
        manifest, "producer_identity_sha256"
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "producer manifest identity self-hash mismatch"
        )
    if canonical_json_sha256(manifest) != _sha256(
        bundle.producer_manifest_sha256,
        "producer_manifest_sha256",
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "producer manifest capability digest mismatch"
        )
    if canonical_json_sha256(commit) != _sha256(
        bundle.commit_sha256, "context bundle commit_sha256"
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "context bundle commit capability digest mismatch"
        )
    if commit.get("producer_identity_sha256") != producer_identity:
        raise Stage2RC5AtomicDecisionSetError(
            "context bundle commit/producer identity mismatch"
        )
    bundle_identity = _sha256(
        commit.get("bundle_identity_sha256"),
        "context bundle_identity_sha256",
    )
    if bundle_identity != _sha256(
        bundle.bundle_identity_sha256,
        "bundle capability identity",
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "context bundle identity capability mismatch"
        )

    window_index = _strict_int(
        manifest.get("window_index"), "producer_manifest.window_index"
    )
    if window_index >= len(window.windows):
        raise Stage2RC5AtomicDecisionSetError(
            "producer window index exceeds verified variable-Q windows"
        )
    raw_window = _required_mapping(
        window.windows[window_index], "selected variable-Q window"
    )
    query_records = raw_window.get("query_records")
    if (
        isinstance(query_records, (str, bytes))
        or not isinstance(query_records, Sequence)
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "selected variable-Q query records are invalid"
        )
    query_size = _strict_int(
        raw_window.get("query_size"), "selected window query_size", minimum=28
    )
    if query_size != len(query_records) or manifest.get("query_size") != query_size:
        raise Stage2RC5AtomicDecisionSetError(
            "producer/variable-Q dynamic query size mismatch"
        )
    context_query = context_payload.get("query_identity_records")
    if (
        isinstance(context_query, (str, bytes))
        or not isinstance(context_query, Sequence)
        or len(context_query) != query_size
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "context/query dynamic-Q cardinality mismatch"
        )
    if manifest.get("window_id") != raw_window.get("window_id"):
        raise Stage2RC5AtomicDecisionSetError(
            "producer/variable-Q window identity mismatch"
        )

    inputs = _required_mapping(manifest.get("inputs"), "producer inputs")
    window_input = _required_mapping(
        inputs.get("variable_query_window"),
        "producer variable_query_window input",
    )
    score_input = _required_mapping(
        inputs.get("score_manifest_metadata"),
        "producer score_manifest_metadata input",
    )
    score_bundle_input = _required_mapping(
        inputs.get("score_bundle"), "producer score_bundle input"
    )
    source_input = _required_mapping(
        inputs.get("source_reference"), "producer source_reference input"
    )
    if (
        window_input.get("sha256") != window.manifest_sha256
        or window_input.get("window_id") != raw_window.get("window_id")
        or window_input.get("schema_version")
        != window.payload.get("schema_version")
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "producer variable-Q input binding mismatch"
        )
    if (
        score_input.get("sha256") != score.manifest_sha256
        or score_input.get("records_content_sha256")
        != score.records_content_sha256
        or score_input.get("role") != score.role
        or score_input.get("member_content_verified") is not False
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "producer score metadata input binding mismatch"
        )
    query_score_bundle = bundle.score_bundle
    query_attestation = query_score_bundle.attestation
    query_run = query_attestation["run_complete"]
    if (
        score_bundle_input.get("sha256")
        != query_score_bundle.attestation_sha256
        or score_bundle_input.get("run_complete_sha256")
        != query_score_bundle.run_complete.sha256
        or score_bundle_input.get("run_complete_identity_sha256")
        != query_run["identity"]["identity_sha256"]
        or score_bundle_input.get("restricted_checkpoint_sha256")
        != query_attestation["restricted_checkpoint"]["sha256"]
        or score_bundle_input.get("current_state_replayed") is not True
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "producer query score-bundle authority mismatch"
        )

    source_attestation = source_reference.attestation
    source_attestation_binding = _required_mapping(
        source_input.get("attestation"), "producer source v3 attestation"
    )
    source_base_binding = _required_mapping(
        source_input.get("base_reference"), "producer base source reference"
    )
    source_run_binding = _required_mapping(
        source_input.get("shared_run_complete"),
        "producer source RUN_COMPLETE",
    )
    source_score_rows = source_attestation["source_score_bundles"]
    expected_source_score_attestations = [
        {
            "source_domain": row["source_domain"],
            "path": row["score_attestation"]["path"],
            "sha256": row["score_attestation"]["sha256"],
            "capability_schema": row["score_attestation"][
                "capability_schema"
            ],
        }
        for row in source_score_rows
    ]
    if (
        source_attestation_binding.get("sha256")
        != source_reference.attestation_sha256
        or source_attestation_binding.get("capability_schema")
        != source_reference.capability_schema
        or source_base_binding.get("npz_sha256")
        != source_reference.npz_sha256
        or source_base_binding.get("audit_sha256")
        != source_reference.audit_sha256
        or _plain(source_input.get("source_score_attestations"))
        != expected_source_score_attestations
        or _plain(source_run_binding)
        != _plain(source_score_rows[0]["run_complete"])
        or source_input.get("current_state_replayed") is not True
        or source_input.get("mixed_consumer_schemas_allowed") is not False
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "producer source-reference v3 authority mismatch"
        )
    query_run_summary = {
        "path": query_run["path"],
        "sha256": query_score_bundle.run_complete.sha256,
        "identity_sha256": query_run["identity"]["identity_sha256"],
    }
    if query_run_summary != _plain(source_run_binding):
        raise Stage2RC5AtomicDecisionSetError(
            "producer source/query score authorities use different RUN_COMPLETE"
        )

    detector, detector_sha = _detector_identity(manifest.get("detector_identity"))
    if dict(source_reference.detector_identity) != detector:
        raise Stage2RC5AtomicDecisionSetError(
            "producer/source-reference detector identity mismatch"
        )
    outputs = _required_mapping(manifest.get("outputs"), "producer outputs")
    context_output = _required_mapping(outputs.get("context"), "context output")
    anchor_output = _required_mapping(outputs.get("anchor"), "anchor output")
    context_full_identity = _sha256(
        context_payload.get("context_full_identity_sha256"),
        "context_full_identity_sha256",
    )
    query_full_identity = _sha256(
        context_payload.get("query_full_identity_sha256"),
        "query_full_identity_sha256",
    )
    window_identity = _sha256(
        context_payload.get("window_identity_sha256"),
        "window_identity_sha256",
    )
    context_package_id = _sha256(
        context_payload.get("context_package_id"), "context_package_id"
    )
    feature_binding = _required_mapping(
        context_payload.get("context_statistics"), "context_statistics"
    )
    feature_vector_sha = _sha256(
        feature_binding.get("vector_sha256"), "context feature_vector_sha256"
    )
    expected_context_output = {
        "sha256": bundle.context_sha256,
        "context_package_id": context_package_id,
        "context_full_identity_sha256": context_full_identity,
        "query_full_identity_sha256": query_full_identity,
        "window_identity_sha256": window_identity,
        "feature_vector_sha256": feature_vector_sha,
    }
    for field, expected in expected_context_output.items():
        if context_output.get(field) != expected:
            raise Stage2RC5AtomicDecisionSetError(
                f"producer context output {field} mismatch"
            )
    anchor_identity = _sha256(
        anchor_payload.get("anchor_identity_sha256"), "anchor_identity_sha256"
    )
    expected_anchor_output = {
        "sha256": bundle.anchor_sha256,
        "anchor_identity_sha256": anchor_identity,
        "context_identity_sha256": context_full_identity,
        "context_probability_content_sha256": anchor_payload.get(
            "context_probability_content_sha256"
        ),
    }
    for field, expected in expected_anchor_output.items():
        if anchor_output.get(field) != expected:
            raise Stage2RC5AtomicDecisionSetError(
                f"producer anchor output {field} mismatch"
            )
    if anchor_payload.get("context_identity_sha256") != context_full_identity:
        raise Stage2RC5AtomicDecisionSetError(
            "anchor/context identity mismatch"
        )
    access = _required_mapping(manifest.get("access_audit"), "access_audit")
    for field in (
        "query_score_member_open_count",
        "query_image_member_open_count",
    ):
        if access.get(field) != 0:
            raise Stage2RC5AtomicDecisionSetError(
                "producer opened a query member before decision sealing"
            )
    for field in (
        "context_labels_accessed",
        "query_labels_accessed",
        "observed_results_accessed",
    ):
        _exact_false(access.get(field), f"access_audit.{field}")

    material = context_inference_material_v2(context)
    features = np.asarray(material.feature_values, dtype=np.float64)
    if features.shape != (93,) or not np.isfinite(features).all():
        raise Stage2RC5AtomicDecisionSetError(
            "producer context does not expose one finite 93D feature vector"
        )
    if material.feature_vector_sha256 != feature_vector_sha:
        raise Stage2RC5AtomicDecisionSetError(
            "context inference material feature digest mismatch"
        )
    source_domains = tuple(
        sorted(
            str(item)
            for item in source_reference.source_reference_v2.source_reference_bundle.domains
        )
    )
    if len(source_domains) != 2 or len(set(source_domains)) != 2:
        raise Stage2RC5AtomicDecisionSetError(
            "producer source reference must contain two source domains"
        )
    shared: dict[str, Any] = {
        "schema_version": SHARED_IDENTITY_SCHEMA,
        "producer_identity_sha256": producer_identity,
        "producer_bundle_identity_sha256": bundle_identity,
        "producer_bundle_capability_schema": _text(
            bundle.capability_schema, "producer bundle capability_schema"
        ),
        "producer_manifest_schema": _text(
            manifest.get("schema_version"), "producer manifest schema"
        ),
        "producer_commit_schema": _text(
            commit.get("schema_version"), "producer commit schema"
        ),
        "producer_manifest_sha256": bundle.producer_manifest_sha256,
        "producer_commit_sha256": bundle.commit_sha256,
        "window_manifest_sha256": window.manifest_sha256,
        "window_schema": window.payload["schema_version"],
        "window_id": raw_window["window_id"],
        "window_index": window_index,
        "window_identity_sha256": window_identity,
        "query_size": query_size,
        "query_full_identity_sha256": query_full_identity,
        "score_manifest_sha256": score.manifest_sha256,
        "score_records_content_sha256": score.records_content_sha256,
        "score_role": score.role,
        "query_score_attestation_sha256": (
            query_score_bundle.attestation_sha256
        ),
        "query_run_complete_artifact_sha256": (
            query_score_bundle.run_complete.sha256
        ),
        "query_run_complete_identity_sha256": query_run[
            "identity"
        ]["identity_sha256"],
        "detector_identity_sha256": detector_sha,
        "detector_checkpoint_sha256": detector["checkpoint_sha256"],
        "source_reference_identity_sha256": source_attestation[
            "attestation_identity_sha256"
        ],
        "source_reference_attestation_sha256": (
            source_reference.attestation_sha256
        ),
        "source_reference_base_npz_sha256": source_reference.npz_sha256,
        "source_score_attestations": [
            {
                "source_domain": row["source_domain"],
                "sha256": row["score_attestation"]["sha256"],
            }
            for row in source_score_rows
        ],
        "source_run_complete_artifact_sha256": source_run_binding["sha256"],
        "source_run_complete_identity_sha256": source_run_binding[
            "identity_sha256"
        ],
        "source_domains": list(source_domains),
        "outer_fold_id": manifest["outer_fold_id"],
        "outer_target": manifest["outer_target"],
        "source_domain": manifest["source_domain"],
        "base_seed": manifest["base_seed"],
        "derived_seed": manifest["derived_seed"],
        "context_payload_sha256": context.payload_sha256,
        "context_package_id": context_package_id,
        "context_full_identity_sha256": context_full_identity,
        "context_feature_vector_sha256": feature_vector_sha,
        "anchor_identity_sha256": anchor_identity,
        "anchor_payload_sha256": bundle.anchor_sha256,
    }
    shared["shared_identity_sha256"] = _self_hash(
        shared, "shared_identity_sha256"
    )
    return shared, features


def _plain_threshold_rows(rows: Any, name: str) -> list[dict[str, Any]]:
    if (
        isinstance(rows, (str, bytes))
        or not isinstance(rows, Sequence)
        or len(rows) != 3
    ):
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} must contain three threshold rows"
        )
    result = []
    for index, raw in enumerate(rows):
        row = _required_mapping(raw, f"{name}[{index}]")
        if (
            row.get("budget_numerator"),
            row.get("budget_denominator"),
        ) != BUDGET_RATIONALS[index]:
            raise Stage2RC5AtomicDecisionSetError(
                f"{name}[{index}] budget rational mismatch"
            )
        probability = _canonical_float_hex(
            row.get("threshold_probability_hex"),
            f"{name}[{index}].threshold_probability_hex",
        )
        coordinate = _canonical_float_hex(
            row.get("threshold_coordinate_hex"),
            f"{name}[{index}].threshold_coordinate_hex",
        )
        relation = row.get(
            "probability_coordinate_relation", "encode_probability"
        )
        expected = _threshold_row(
            index=index,
            probability=probability,
            coordinate=coordinate,
            relation=relation,
        )
        if row.get("threshold_kind") != expected["threshold_kind"]:
            raise Stage2RC5AtomicDecisionSetError(
                f"{name}[{index}] endpoint kind mismatch"
            )
        result.append(expected)
    return result


def _nearest_source_domain(
    reference: VerifiedExactSourceThresholdReferenceV3,
    context_features: np.ndarray,
) -> str:
    base = np.asarray(context_features[:87], dtype=np.float64)
    centers = np.asarray(reference.source_centers, dtype=np.float64)
    scale = np.asarray(reference.source_scale, dtype=np.float64)
    distances = np.linalg.norm((centers - base[None, :]) / scale[None, :], axis=1)
    return min(
        (float(distances[index]), domain)
        for index, domain in enumerate(reference.source_domains)
    )[1]


def _complete_decision(
    *,
    method_id: str,
    rows: Sequence[Mapping[str, Any]],
    authority: Mapping[str, Any],
    shared_identity_sha256: str,
) -> dict[str, Any]:
    normalized_rows = _plain_threshold_rows(rows, f"{method_id}.rows")
    payload: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA,
        "artifact_status": "prelabel_complete",
        "method_id": method_id,
        "method_name": METHOD_NAMES[method_id],
        "outcome": "complete",
        "budget_order": "strictly_descending_loose_to_strict",
        "budget_rationals": _budget_payload(),
        "threshold_representation_schema": representation_contract()[
            "schema_version"
        ],
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "rows": normalized_rows,
        "authority": _plain(authority),
        "missing_reason_code": None,
        "shared_prelabel_identity_sha256": shared_identity_sha256,
        "labels_accessed": False,
        "query_members_opened": False,
        "reject": False,
        "fallback": False,
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    payload["decision_identity_sha256"] = _self_hash(
        payload, "decision_identity_sha256"
    )
    return payload


def _missing_t5_decision(shared_identity_sha256: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA,
        "artifact_status": "prelabel_sealed_missing",
        "method_id": "T5",
        "method_name": METHOD_NAMES["T5"],
        "outcome": "sealed_missing",
        "budget_order": "strictly_descending_loose_to_strict",
        "budget_rationals": _budget_payload(),
        "threshold_representation_schema": representation_contract()[
            "schema_version"
        ],
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "rows": [],
        "authority": {
            "authority_kind": "atomic_missing_seal",
            "fallback_method": None,
        },
        "missing_reason_code": "not_complete_before_atomic_prelabel_seal",
        "shared_prelabel_identity_sha256": shared_identity_sha256,
        "labels_accessed": False,
        "query_members_opened": False,
        "reject": False,
        "fallback": False,
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    payload["decision_identity_sha256"] = _self_hash(
        payload, "decision_identity_sha256"
    )
    return payload


def _learned_rows_and_authority(
    *,
    method: str,
    seal: VerifiedStage2RC5InferenceSeal,
    shared: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verified = assert_verified_stage2_rc5_inference_seal(seal)
    if verified.method != method:
        raise Stage2RC5AtomicDecisionSetError(
            f"{method} inference seal method mismatch"
        )
    transcript = _required_mapping(verified.transcript, f"{method} transcript")
    if transcript.get("schema_version") != INFERENCE_TRANSCRIPT_SCHEMA:
        raise Stage2RC5AtomicDecisionSetError(
            f"{method} inference transcript schema is not the frozen RC5 schema"
        )
    producer_binding = _required_mapping(
        transcript.get("producer_bundle_binding"),
        f"{method} producer bundle binding",
    )
    expected_producer_binding = {
        "capability_schema": shared["producer_bundle_capability_schema"],
        "producer_manifest_schema": shared["producer_manifest_schema"],
        "commit_schema": shared["producer_commit_schema"],
        "producer_identity_sha256": shared["producer_identity_sha256"],
        "bundle_identity_sha256": shared[
            "producer_bundle_identity_sha256"
        ],
        "producer_manifest_sha256": shared["producer_manifest_sha256"],
        "commit_sha256": shared["producer_commit_sha256"],
    }
    if dict(producer_binding) != expected_producer_binding:
        raise Stage2RC5AtomicDecisionSetError(
            f"{method} seal uses a different producer bundle"
        )
    capability_identity_fields = {
        "producer_identity_sha256": "producer_identity_sha256",
        "producer_bundle_identity_sha256": "bundle_identity_sha256",
        "producer_manifest_sha256": "producer_manifest_sha256",
        "producer_commit_sha256": "commit_sha256",
    }
    for attribute, binding_field in capability_identity_fields.items():
        if getattr(verified, attribute, None) != expected_producer_binding[
            binding_field
        ]:
            raise Stage2RC5AtomicDecisionSetError(
                f"{method} verifier capability producer identity drifted"
            )
    context_binding = _required_mapping(
        transcript.get("context_binding"), f"{method} context binding"
    )
    anchor_binding = _required_mapping(
        transcript.get("anchor_binding"), f"{method} anchor binding"
    )
    if (
        context_binding.get("context_payload_sha256")
        != shared["context_payload_sha256"]
        or context_binding.get("context_full_identity_sha256")
        != shared["context_full_identity_sha256"]
        or context_binding.get("context_feature_vector_sha256")
        != shared["context_feature_vector_sha256"]
    ):
        raise Stage2RC5AtomicDecisionSetError(
            f"{method} seal uses a different producer context"
        )
    if (
        anchor_binding.get("anchor_identity_sha256")
        != shared["anchor_identity_sha256"]
        or anchor_binding.get("anchor_payload_sha256")
        != shared["anchor_payload_sha256"]
    ):
        raise Stage2RC5AtomicDecisionSetError(
            f"{method} seal uses a different verified anchor"
        )
    decision = _required_mapping(verified.decision, f"{method} decision")
    if decision.get("method") != method:
        raise Stage2RC5AtomicDecisionSetError(
            f"{method} sealed decision method drifted"
        )
    raw_rows = decision.get("rows")
    if (
        isinstance(raw_rows, (str, bytes))
        or not isinstance(raw_rows, Sequence)
        or len(raw_rows) != 3
    ):
        raise Stage2RC5AtomicDecisionSetError(
            f"{method} sealed decision must contain three rows"
        )
    rows = []
    for index, raw in enumerate(raw_rows):
        row = _required_mapping(raw, f"{method}.decision.rows[{index}]")
        probability = _canonical_float_hex(
            row.get("decoded_threshold_hex"),
            f"{method}.rows[{index}].decoded_threshold_hex",
        )
        coordinate = _canonical_float_hex(
            row.get("canonical_coordinate_hex"),
            f"{method}.rows[{index}].canonical_coordinate_hex",
        )
        normalized = _threshold_row(
            index=index,
            probability=probability,
            coordinate=coordinate,
            relation="decode_coordinate",
        )
        if row.get("threshold_kind") != normalized["threshold_kind"]:
            raise Stage2RC5AtomicDecisionSetError(
                f"{method} sealed endpoint kind mismatch"
            )
        rows.append(normalized)
    checkpoint_binding = _required_mapping(
        transcript.get("checkpoint_binding"), f"{method} checkpoint binding"
    )
    authority = {
        "authority_kind": "VerifiedStage2RC5InferenceSeal",
        "transcript_schema": transcript.get("schema_version"),
        "producer_bundle_binding": expected_producer_binding,
        "transcript_bytes_sha256": verified.transcript_bytes_sha256,
        "transcript_identity_sha256": verified.transcript_identity_sha256,
        "decision_identity_sha256": verified.decision_identity_sha256,
        "calibrator_checkpoint_sha256": checkpoint_binding.get(
            "checkpoint_bytes_sha256"
        ),
        "training_contract_sha256": checkpoint_binding.get(
            "training_contract_sha256"
        ),
    }
    for field, value in authority.items():
        if field.endswith("sha256"):
            _sha256(value, f"{method}.authority.{field}")
    return rows, authority


BASELINE_PREFIX_SCHEMA = "rc-irstd.stage2-rc5-baseline-decision-prefix.v1"


def build_stage2_rc5_baseline_decision_prefix_payload(
    *,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    source_threshold_reference: VerifiedExactSourceThresholdReferenceV3,
    evt_seal: VerifiedStage2RC5EVTSeal | None = None,
) -> dict[str, Any]:
    """Recompute T0--T5 only from current verifier capabilities.

    This additive bridge lets RC5+ retain the frozen analytic/source/EVT
    baselines without constructing obsolete RC5 learned checkpoints.  It is
    still a pre-label three-primary-budget object; T9 remains post-label only.
    """

    shared, features = _shared_identity_from_bundle(producer_bundle)
    shared_sha = shared["shared_identity_sha256"]
    source = assert_verified_exact_source_threshold_reference_v3(
        source_threshold_reference
    )
    replayed_source_reference = replay_verified_stage2_rc5_source_reference_v3(
        source.source_reference_v3
    )
    source_authority = source.payload["source_reference_authority"]
    actual_source_authority = {
        "capability_schema": replayed_source_reference.capability_schema,
        "attestation_sha256": replayed_source_reference.attestation_sha256,
        "attestation_identity_sha256": replayed_source_reference.attestation[
            "attestation_identity_sha256"
        ],
        "base_npz_sha256": replayed_source_reference.npz_sha256,
        "base_audit_sha256": replayed_source_reference.audit_sha256,
        "source_score_attestations": [
            {
                "source_domain": row["source_domain"],
                "sha256": row["score_attestation"]["sha256"],
            }
            for row in replayed_source_reference.attestation[
                "source_score_bundles"
            ]
        ],
        "shared_run_complete": _plain(
            replayed_source_reference.attestation[
                "source_score_bundles"
            ][0]["run_complete"]
        ),
    }
    if (
        source.detector_identity_sha256 != shared["detector_identity_sha256"]
        or source.source_reference_identity_sha256
        != shared["source_reference_identity_sha256"]
        or source.source_reference_attestation_sha256
        != shared["source_reference_attestation_sha256"]
        or source.source_domains != tuple(shared["source_domains"])
        or _plain(source_authority) != actual_source_authority
        or _plain(source_authority["source_score_attestations"])
        != _plain(shared["source_score_attestations"])
        or source_authority["shared_run_complete"]["sha256"]
        != shared["source_run_complete_artifact_sha256"]
        or source_authority["shared_run_complete"]["identity_sha256"]
        != shared["source_run_complete_identity_sha256"]
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "exact source threshold reference does not share producer identity"
        )
    source_detector = source.payload["detector_identity"]
    for field in (
        "outer_fold_id",
        "outer_target",
        "base_seed",
        "derived_seed",
    ):
        if source_detector[field] != shared[field]:
            raise Stage2RC5AtomicDecisionSetError(
                f"source reference/producer {field} mismatch"
            )

    decisions: list[dict[str, Any]] = [
        _complete_decision(
            method_id="T0",
            rows=[
                _threshold_row(index=index, probability=0.5)
                for index in range(3)
            ],
            authority={
                "authority_kind": "frozen_constant",
                "value_hex": (0.5).hex(),
            },
            shared_identity_sha256=shared_sha,
        ),
        _complete_decision(
            method_id="T1",
            rows=source.payload["pooled_safe_rows"],
            authority={
                "authority_kind": "VerifiedExactSourceThresholdReferenceV3",
                "reference_identity_sha256": source.reference_identity_sha256,
                "selection": "pooled_exact_source_safe",
            },
            shared_identity_sha256=shared_sha,
        ),
        _complete_decision(
            method_id="T2",
            rows=source.payload["safer_safe_rows"],
            authority={
                "authority_kind": "VerifiedExactSourceThresholdReferenceV3",
                "reference_identity_sha256": source.reference_identity_sha256,
                "selection": "coordinatewise_max_of_two_source_safe_thresholds",
            },
            shared_identity_sha256=shared_sha,
        ),
    ]
    nearest = _nearest_source_domain(source, features)
    decisions.append(
        _complete_decision(
            method_id="T3",
            rows=source.payload["domain_safe_rows"][nearest],
            authority={
                "authority_kind": "VerifiedExactSourceThresholdReferenceV3",
                "reference_identity_sha256": source.reference_identity_sha256,
                "selection": "label_free_nearest_source",
                "nearest_source_domain": nearest,
                "context_feature_vector_sha256": shared[
                    "context_feature_vector_sha256"
                ],
            },
            shared_identity_sha256=shared_sha,
        )
    )
    anchor = assert_verified_context_tail_anchor(producer_bundle.anchor)
    decisions.append(
        _complete_decision(
            method_id="T4",
            rows=[
                _threshold_row(
                    index=index,
                    probability=float(anchor.thresholds[index]),
                    coordinate=float(anchor.coordinates[index]),
                )
                for index in range(3)
            ],
            authority={
                "authority_kind": "VerifiedContextTailAnchor",
                "anchor_identity_sha256": shared["anchor_identity_sha256"],
                "anchor_payload_sha256": shared["anchor_payload_sha256"],
            },
            shared_identity_sha256=shared_sha,
        )
    )
    if evt_seal is None:
        decisions.append(_missing_t5_decision(shared_sha))
    else:
        evt = assert_verified_stage2_rc5_evt_seal(evt_seal)
        if (
            evt.payload["producer_identity_sha256"]
            != shared["producer_identity_sha256"]
            or evt.payload["context_full_identity_sha256"]
            != shared["context_full_identity_sha256"]
            or evt.payload["anchor_identity_sha256"]
            != shared["anchor_identity_sha256"]
        ):
            raise Stage2RC5AtomicDecisionSetError(
                "EVT seal does not share producer/context/anchor identity"
            )
        decisions.append(
            _complete_decision(
                method_id="T5",
                rows=evt.payload["rows"],
                authority={
                    "authority_kind": "VerifiedStage2RC5EVTSeal",
                    "seal_identity_sha256": evt.seal_identity_sha256,
                    "fit_identity_sha256": evt.payload["fit_identity_sha256"],
                },
                shared_identity_sha256=shared_sha,
            )
        )
    if tuple(row["method_id"] for row in decisions) != tuple(
        f"T{index}" for index in range(6)
    ):
        raise RuntimeError("baseline prefix order is not exact T0--T5")
    payload: dict[str, Any] = {
        "schema_version": BASELINE_PREFIX_SCHEMA,
        "artifact_status": "prelabel_baseline_prefix_complete",
        "method_ids": [f"T{index}" for index in range(6)],
        "budget_rationals": _budget_payload(),
        "shared_prelabel_identity": shared,
        "decisions": decisions,
        "t9_included": False,
        "labels_accessed": False,
        "query_members_opened": False,
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    payload["prefix_identity_sha256"] = _self_hash(
        payload, "prefix_identity_sha256"
    )
    return payload


def build_stage2_rc5_atomic_decision_set_payload(
    *,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    source_threshold_reference: VerifiedExactSourceThresholdReferenceV3,
    inference_seals: Mapping[str, VerifiedStage2RC5InferenceSeal],
    evt_seal: VerifiedStage2RC5EVTSeal | None = None,
) -> dict[str, Any]:
    """Rebuild one canonical T0--T8 set from verifier capabilities only."""

    shared, features = _shared_identity_from_bundle(producer_bundle)
    shared_sha = shared["shared_identity_sha256"]
    source = assert_verified_exact_source_threshold_reference_v3(
        source_threshold_reference
    )
    replayed_source_reference = replay_verified_stage2_rc5_source_reference_v3(
        source.source_reference_v3
    )
    source_authority = source.payload["source_reference_authority"]
    actual_source_authority = {
        "capability_schema": replayed_source_reference.capability_schema,
        "attestation_sha256": replayed_source_reference.attestation_sha256,
        "attestation_identity_sha256": replayed_source_reference.attestation[
            "attestation_identity_sha256"
        ],
        "base_npz_sha256": replayed_source_reference.npz_sha256,
        "base_audit_sha256": replayed_source_reference.audit_sha256,
        "source_score_attestations": [
            {
                "source_domain": row["source_domain"],
                "sha256": row["score_attestation"]["sha256"],
            }
            for row in replayed_source_reference.attestation[
                "source_score_bundles"
            ]
        ],
        "shared_run_complete": _plain(
            replayed_source_reference.attestation[
                "source_score_bundles"
            ][0]["run_complete"]
        ),
    }
    if (
        source.detector_identity_sha256 != shared["detector_identity_sha256"]
        or source.source_reference_identity_sha256
        != shared["source_reference_identity_sha256"]
        or source.source_reference_attestation_sha256
        != shared["source_reference_attestation_sha256"]
        or source.source_domains != tuple(shared["source_domains"])
        or _plain(source_authority) != actual_source_authority
        or _plain(source_authority["source_score_attestations"])
        != _plain(shared["source_score_attestations"])
        or source_authority["shared_run_complete"]["sha256"]
        != shared["source_run_complete_artifact_sha256"]
        or source_authority["shared_run_complete"]["identity_sha256"]
        != shared["source_run_complete_identity_sha256"]
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "exact source threshold reference does not share producer identity"
        )
    source_detector = source.payload["detector_identity"]
    for field in (
        "outer_fold_id",
        "outer_target",
        "base_seed",
        "derived_seed",
    ):
        if source_detector[field] != shared[field]:
            raise Stage2RC5AtomicDecisionSetError(
                f"source reference/producer {field} mismatch"
            )
    if not isinstance(inference_seals, Mapping) or set(inference_seals) != set(
        LEARNED_METHOD_IDS
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "inference_seals keys must be exactly T6,T7,T8"
        )

    decisions: list[dict[str, Any]] = []
    decisions.append(
        _complete_decision(
            method_id="T0",
            rows=[
                _threshold_row(index=index, probability=0.5)
                for index in range(3)
            ],
            authority={"authority_kind": "frozen_constant", "value_hex": (0.5).hex()},
            shared_identity_sha256=shared_sha,
        )
    )
    decisions.append(
        _complete_decision(
            method_id="T1",
            rows=source.payload["pooled_safe_rows"],
            authority={
                "authority_kind": "VerifiedExactSourceThresholdReferenceV3",
                "reference_identity_sha256": source.reference_identity_sha256,
                "selection": "pooled_exact_source_safe",
            },
            shared_identity_sha256=shared_sha,
        )
    )
    decisions.append(
        _complete_decision(
            method_id="T2",
            rows=source.payload["safer_safe_rows"],
            authority={
                "authority_kind": "VerifiedExactSourceThresholdReferenceV3",
                "reference_identity_sha256": source.reference_identity_sha256,
                "selection": "coordinatewise_max_of_two_source_safe_thresholds",
            },
            shared_identity_sha256=shared_sha,
        )
    )
    nearest = _nearest_source_domain(source, features)
    decisions.append(
        _complete_decision(
            method_id="T3",
            rows=source.payload["domain_safe_rows"][nearest],
            authority={
                "authority_kind": "VerifiedExactSourceThresholdReferenceV3",
                "reference_identity_sha256": source.reference_identity_sha256,
                "selection": "label_free_nearest_source",
                "nearest_source_domain": nearest,
                "context_feature_vector_sha256": shared[
                    "context_feature_vector_sha256"
                ],
            },
            shared_identity_sha256=shared_sha,
        )
    )
    anchor = assert_verified_context_tail_anchor(producer_bundle.anchor)
    anchor_rows = [
        _threshold_row(
            index=index,
            probability=float(anchor.thresholds[index]),
            coordinate=float(anchor.coordinates[index]),
        )
        for index in range(3)
    ]
    decisions.append(
        _complete_decision(
            method_id="T4",
            rows=anchor_rows,
            authority={
                "authority_kind": "VerifiedContextTailAnchor",
                "anchor_identity_sha256": shared["anchor_identity_sha256"],
                "anchor_payload_sha256": shared["anchor_payload_sha256"],
            },
            shared_identity_sha256=shared_sha,
        )
    )
    if evt_seal is None:
        decisions.append(_missing_t5_decision(shared_sha))
    else:
        evt = assert_verified_stage2_rc5_evt_seal(evt_seal)
        if (
            evt.payload["producer_identity_sha256"]
            != shared["producer_identity_sha256"]
            or evt.payload["context_full_identity_sha256"]
            != shared["context_full_identity_sha256"]
            or evt.payload["anchor_identity_sha256"]
            != shared["anchor_identity_sha256"]
        ):
            raise Stage2RC5AtomicDecisionSetError(
                "EVT seal does not share producer/context/anchor identity"
            )
        decisions.append(
            _complete_decision(
                method_id="T5",
                rows=evt.payload["rows"],
                authority={
                    "authority_kind": "VerifiedStage2RC5EVTSeal",
                    "seal_identity_sha256": evt.seal_identity_sha256,
                    "fit_identity_sha256": evt.payload["fit_identity_sha256"],
                },
                shared_identity_sha256=shared_sha,
            )
        )
    for method in LEARNED_METHOD_IDS:
        rows, authority = _learned_rows_and_authority(
            method=method,
            seal=inference_seals[method],
            shared=shared,
        )
        decisions.append(
            _complete_decision(
                method_id=method,
                rows=rows,
                authority=authority,
                shared_identity_sha256=shared_sha,
            )
        )
    if tuple(decision["method_id"] for decision in decisions) != METHOD_IDS:
        raise RuntimeError("internal RC5 decision order is not exact T0--T8")
    payload: dict[str, Any] = {
        "schema_version": DECISION_SET_SCHEMA,
        "artifact_type": DECISION_SET_ARTIFACT_TYPE,
        "artifact_status": "prelabel_atomic_complete",
        "method_ids": list(METHOD_IDS),
        "budget_order": "strictly_descending_loose_to_strict",
        "budget_rationals": _budget_payload(),
        "budget_count_rule": EXACT_BUDGET_COUNT_RULE,
        "threshold_representation": representation_contract(),
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "shared_prelabel_identity": shared,
        "decisions": decisions,
        "t9_included": False,
        "publication_contract": PUBLICATION_ORDER,
        "guardrails": {
            "labels_accessed": False,
            "query_scores_opened": False,
            "query_images_opened": False,
            "query_labels_opened": False,
            "postlabel_statistics_accessed": False,
            "legacy_rc4_decision_authority_used": False,
            "float_budget_count_logic_used": False,
            "fallback_used": False,
            "reject_used": False,
        },
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    payload["decision_set_identity_sha256"] = _self_hash(
        payload, "decision_set_identity_sha256"
    )
    return payload


_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_status",
        "method_id",
        "method_name",
        "outcome",
        "budget_order",
        "budget_rationals",
        "threshold_representation_schema",
        "threshold_semantics",
        "rows",
        "authority",
        "missing_reason_code",
        "shared_prelabel_identity_sha256",
        "labels_accessed",
        "query_members_opened",
        "reject",
        "fallback",
        "self_hash_algorithm",
        "decision_identity_sha256",
    }
)
_ROW_FIELDS = frozenset(
    {
        "budget_numerator",
        "budget_denominator",
        "threshold_probability_hex",
        "threshold_coordinate_hex",
        "threshold_kind",
        "probability_coordinate_relation",
    }
)
_SET_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "method_ids",
        "budget_order",
        "budget_rationals",
        "budget_count_rule",
        "threshold_representation",
        "threshold_semantics",
        "shared_prelabel_identity",
        "decisions",
        "t9_included",
        "publication_contract",
        "guardrails",
        "self_hash_algorithm",
        "decision_set_identity_sha256",
    }
)
_COMMIT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "publication_order",
        "decision_set_filename",
        "decision_set_sha256",
        "decision_set_identity_sha256",
        "shared_prelabel_identity_sha256",
        "t9_included",
        "self_hash_algorithm",
        "commit_identity_sha256",
    }
)


def _exact_fields(
    value: Any, fields: frozenset[str], name: str
) -> Mapping[str, Any]:
    mapping = _required_mapping(value, name)
    if set(mapping) != fields:
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} fields differ from the frozen schema"
        )
    return mapping


def _validate_decision_set_payload(value: Any) -> dict[str, Any]:
    top = _exact_fields(value, _SET_FIELDS, "decision set")
    if (
        top["schema_version"] != DECISION_SET_SCHEMA
        or top["artifact_type"] != DECISION_SET_ARTIFACT_TYPE
        or top["artifact_status"] != "prelabel_atomic_complete"
        or top["method_ids"] != list(METHOD_IDS)
        or top["budget_order"] != "strictly_descending_loose_to_strict"
        or top["budget_rationals"] != _budget_payload()
        or top["budget_count_rule"] != EXACT_BUDGET_COUNT_RULE
        or top["threshold_representation"] != representation_contract()
        or top["threshold_semantics"] != STRICT_THRESHOLD_SEMANTICS
        or top["t9_included"] is not False
        or top["publication_contract"] != PUBLICATION_ORDER
        or top["self_hash_algorithm"] != SELF_HASH_ALGORITHM
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "decision-set top-level contract drifted"
        )
    shared = _required_mapping(
        top["shared_prelabel_identity"], "shared_prelabel_identity"
    )
    if (
        shared.get("schema_version") != SHARED_IDENTITY_SCHEMA
        or shared.get("shared_identity_sha256")
        != _self_hash(shared, "shared_identity_sha256")
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "shared prelabel identity self-hash mismatch"
        )
    shared_sha = _sha256(
        shared["shared_identity_sha256"], "shared_identity_sha256"
    )
    query_size = _strict_int(
        shared.get("query_size"), "shared query_size", minimum=28
    )
    if query_size in {0, 14}:
        raise Stage2RC5AtomicDecisionSetError(
            "shared query_size is not a dynamic query partition"
        )
    decisions = top["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(METHOD_IDS):
        raise Stage2RC5AtomicDecisionSetError(
            "decision set must contain exactly nine T0--T8 decisions"
        )
    for index, raw in enumerate(decisions):
        method = METHOD_IDS[index]
        decision = _exact_fields(raw, _DECISION_FIELDS, f"decisions[{index}]")
        if (
            decision["schema_version"] != DECISION_SCHEMA
            or decision["method_id"] != method
            or decision["method_name"] != METHOD_NAMES[method]
            or decision["budget_order"]
            != "strictly_descending_loose_to_strict"
            or decision["budget_rationals"] != _budget_payload()
            or decision["threshold_representation_schema"]
            != representation_contract()["schema_version"]
            or decision["threshold_semantics"] != STRICT_THRESHOLD_SEMANTICS
            or decision["shared_prelabel_identity_sha256"] != shared_sha
            or decision["self_hash_algorithm"] != SELF_HASH_ALGORITHM
        ):
            raise Stage2RC5AtomicDecisionSetError(
                f"{method} decision contract drifted"
            )
        for field in (
            "labels_accessed",
            "query_members_opened",
            "reject",
            "fallback",
        ):
            _exact_false(decision[field], f"{method}.{field}")
        rows = decision["rows"]
        if method == "T5" and decision["outcome"] == "sealed_missing":
            if (
                decision["artifact_status"] != "prelabel_sealed_missing"
                or rows != []
                or decision["missing_reason_code"]
                != "not_complete_before_atomic_prelabel_seal"
                or decision["authority"]
                != {
                    "authority_kind": "atomic_missing_seal",
                    "fallback_method": None,
                }
            ):
                raise Stage2RC5AtomicDecisionSetError(
                    "T5 missing seal is not exact or attempted a fallback"
                )
        else:
            if (
                decision["artifact_status"] != "prelabel_complete"
                or decision["outcome"] != "complete"
                or decision["missing_reason_code"] is not None
                or not isinstance(rows, list)
                or len(rows) != 3
            ):
                raise Stage2RC5AtomicDecisionSetError(
                    f"{method} complete outcome is invalid"
                )
            for row_index, raw_row in enumerate(rows):
                row = _exact_fields(
                    raw_row,
                    _ROW_FIELDS,
                    f"{method}.rows[{row_index}]",
                )
                probability = _canonical_float_hex(
                    row["threshold_probability_hex"],
                    f"{method}.rows[{row_index}].threshold_probability_hex",
                )
                coordinate = _canonical_float_hex(
                    row["threshold_coordinate_hex"],
                    f"{method}.rows[{row_index}].threshold_coordinate_hex",
                )
                expected = _threshold_row(
                    index=row_index,
                    probability=probability,
                    coordinate=coordinate,
                    relation=row["probability_coordinate_relation"],
                )
                if dict(row) != expected:
                    raise Stage2RC5AtomicDecisionSetError(
                        f"{method}.rows[{row_index}] is not canonical EATC-v2"
                    )
        if decision["decision_identity_sha256"] != _self_hash(
            decision, "decision_identity_sha256"
        ):
            raise Stage2RC5AtomicDecisionSetError(
                f"{method} decision self-hash mismatch"
            )
    guardrails = _required_mapping(top["guardrails"], "decision-set guardrails")
    if not guardrails or any(value is not False for value in guardrails.values()):
        raise Stage2RC5AtomicDecisionSetError(
            "decision-set guardrails record forbidden access/fallback"
        )
    if top["decision_set_identity_sha256"] != _self_hash(
        top, "decision_set_identity_sha256"
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "decision-set identity self-hash mismatch"
        )
    return _plain(top)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_canonical_json(data: bytes, name: str) -> dict[str, Any]:
    if type(data) is not bytes or not data:
        raise TypeError(f"{name} must be nonempty bytes")
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} is not strict JSON: {error}"
        ) from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != data:
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} is not one canonical JSON object"
        )
    return value


def _repository_root(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute() or raw.is_symlink():
        raise Stage2RC5AtomicDecisionSetError(
            "repository_root must be absolute and non-symlink"
        )
    root = raw.resolve(strict=True)
    if root != raw or not root.is_dir():
        raise Stage2RC5AtomicDecisionSetError(
            "repository_root must be a canonical existing directory"
        )
    return root


def _direct_path(
    value: str | Path,
    root: Path,
    name: str,
    *,
    require_file: bool,
) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} is outside repository_root"
        ) from error
    if not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise Stage2RC5AtomicDecisionSetError(
            f"{name} path is empty or unsafe"
        )
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as error:
            raise Stage2RC5AtomicDecisionSetError(
                f"{name} does not exist"
            ) from error
        if stat.S_ISLNK(info.st_mode):
            raise Stage2RC5AtomicDecisionSetError(
                f"{name} contains a symlink component"
            )
    final = candidate.stat(follow_symlinks=False)
    if require_file and not stat.S_ISREG(final.st_mode):
        raise Stage2RC5AtomicDecisionSetError(f"{name} is not a regular file")
    if not require_file and not stat.S_ISDIR(final.st_mode):
        raise Stage2RC5AtomicDecisionSetError(f"{name} is not a directory")
    return candidate


def _stable_read(path: Path, expected_sha256: str, root: Path, name: str) -> bytes:
    expected = _sha256(expected_sha256, f"{name}.sha256")
    candidate = _direct_path(path, root, name, require_file=True)
    descriptor = os.open(
        candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = candidate.stat(follow_symlinks=False)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(current):
        raise RuntimeError(f"{name} changed during stable read")
    if digest.hexdigest() != expected:
        raise Stage2RC5AtomicDecisionSetError(f"{name} SHA-256 mismatch")
    return b"".join(chunks)


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_payload(
    *, decision_set_sha256: str, decision_set: Mapping[str, Any]
) -> dict[str, Any]:
    shared = decision_set["shared_prelabel_identity"]
    payload: dict[str, Any] = {
        "schema_version": DECISION_SET_COMMIT_SCHEMA,
        "artifact_type": DECISION_SET_COMMIT_ARTIFACT_TYPE,
        "artifact_status": "committed_prelabel_t0_t8",
        "publication_order": PUBLICATION_ORDER,
        "decision_set_filename": DECISION_SET_FILENAME,
        "decision_set_sha256": decision_set_sha256,
        "decision_set_identity_sha256": decision_set[
            "decision_set_identity_sha256"
        ],
        "shared_prelabel_identity_sha256": shared[
            "shared_identity_sha256"
        ],
        "t9_included": False,
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    payload["commit_identity_sha256"] = _self_hash(
        payload, "commit_identity_sha256"
    )
    return payload


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5AtomicDecisionSet:
    decision_set_path: Path
    commit_path: Path
    decision_set_sha256: str
    commit_sha256: str
    decision_set_identity_sha256: str
    shared_prelabel_identity_sha256: str
    payload: Mapping[str, Any]
    decisions: tuple[Mapping[str, Any], ...]
    decision_by_method: Mapping[str, Mapping[str, Any]]
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "VerifiedStage2RC5AtomicDecisionSet is public-verifier-issued only"
        )

    def thresholds(self, method_id: str) -> tuple[float, float, float] | None:
        method = _text(method_id, "method_id")
        if method not in self.decision_by_method:
            raise KeyError(method)
        rows = self.decision_by_method[method]["rows"]
        if not rows:
            return None
        return tuple(
            float.fromhex(row["threshold_probability_hex"])
            for row in rows
        )  # type: ignore[return-value]


def _issue_decision_set(
    *,
    set_path: Path,
    commit_path: Path,
    set_sha: str,
    commit_sha: str,
    payload: Mapping[str, Any],
) -> VerifiedStage2RC5AtomicDecisionSet:
    frozen = _freeze(payload)
    decisions = tuple(frozen["decisions"])
    value = object.__new__(VerifiedStage2RC5AtomicDecisionSet)
    object.__setattr__(value, "decision_set_path", set_path)
    object.__setattr__(value, "commit_path", commit_path)
    object.__setattr__(value, "decision_set_sha256", set_sha)
    object.__setattr__(value, "commit_sha256", commit_sha)
    object.__setattr__(
        value,
        "decision_set_identity_sha256",
        payload["decision_set_identity_sha256"],
    )
    object.__setattr__(
        value,
        "shared_prelabel_identity_sha256",
        payload["shared_prelabel_identity"]["shared_identity_sha256"],
    )
    object.__setattr__(value, "payload", frozen)
    object.__setattr__(value, "decisions", decisions)
    object.__setattr__(
        value,
        "decision_by_method",
        MappingProxyType(
            {decision["method_id"]: decision for decision in decisions}
        ),
    )
    object.__setattr__(value, "_capability", _DECISION_SET_TOKEN)
    return value


def assert_verified_stage2_rc5_atomic_decision_set(
    value: Any,
) -> VerifiedStage2RC5AtomicDecisionSet:
    if (
        type(value) is not VerifiedStage2RC5AtomicDecisionSet
        or getattr(value, "_capability", None) is not _DECISION_SET_TOKEN
        or tuple(value.decision_by_method) != METHOD_IDS
    ):
        raise TypeError(
            "a verifier-issued RC5 atomic T0--T8 decision set is required"
        )
    return value


def verify_stage2_rc5_atomic_decision_set(
    *,
    decision_set_path: str | Path,
    commit_path: str | Path,
    expected_commit_sha256: str,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    source_threshold_reference: VerifiedExactSourceThresholdReferenceV3,
    inference_seals: Mapping[str, VerifiedStage2RC5InferenceSeal],
    evt_seal: VerifiedStage2RC5EVTSeal | None = None,
    repository_root: str | Path,
) -> VerifiedStage2RC5AtomicDecisionSet:
    """Reopen commit-first, then semantically replay every T0--T8 byte."""

    root = _repository_root(repository_root)
    commit_candidate = _direct_path(
        commit_path, root, "decision-set commit", require_file=True
    )
    if commit_candidate.name != DECISION_SET_COMMIT_FILENAME:
        raise Stage2RC5AtomicDecisionSetError(
            "decision-set commit does not use the canonical filename"
        )
    commit_sha = _sha256(expected_commit_sha256, "expected_commit_sha256")
    commit_bytes = _stable_read(
        commit_candidate, commit_sha, root, "decision-set commit"
    )
    commit = _exact_fields(
        _parse_canonical_json(commit_bytes, "decision-set commit"),
        _COMMIT_FIELDS,
        "decision-set commit",
    )
    if (
        commit["schema_version"] != DECISION_SET_COMMIT_SCHEMA
        or commit["artifact_type"] != DECISION_SET_COMMIT_ARTIFACT_TYPE
        or commit["artifact_status"] != "committed_prelabel_t0_t8"
        or commit["publication_order"] != PUBLICATION_ORDER
        or commit["decision_set_filename"] != DECISION_SET_FILENAME
        or commit["t9_included"] is not False
        or commit["self_hash_algorithm"] != SELF_HASH_ALGORITHM
        or commit["commit_identity_sha256"]
        != _self_hash(commit, "commit_identity_sha256")
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "decision-set commit contract drifted"
        )
    # Do not inspect the set member until the externally hashed, canonical
    # commit has passed.  This makes the verifier genuinely commit-first,
    # including on label-resolver failure paths.
    set_candidate = _direct_path(
        decision_set_path, root, "decision set", require_file=True
    )
    if (
        set_candidate.name != DECISION_SET_FILENAME
        or commit_candidate.parent != set_candidate.parent
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "decision set/commit do not use the canonical same-directory layout"
        )
    set_sha = _sha256(
        commit["decision_set_sha256"], "commit.decision_set_sha256"
    )
    set_bytes = _stable_read(set_candidate, set_sha, root, "decision set")
    if commit_candidate.stat(follow_symlinks=False).st_mtime_ns < set_candidate.stat(
        follow_symlinks=False
    ).st_mtime_ns:
        raise Stage2RC5AtomicDecisionSetError(
            "decision-set commit was not published last"
        )
    supplied = _validate_decision_set_payload(
        _parse_canonical_json(set_bytes, "decision set")
    )
    if (
        commit["decision_set_identity_sha256"]
        != supplied["decision_set_identity_sha256"]
        or commit["shared_prelabel_identity_sha256"]
        != supplied["shared_prelabel_identity"]["shared_identity_sha256"]
    ):
        raise Stage2RC5AtomicDecisionSetError(
            "decision-set commit identity binding mismatch"
        )
    expected_commit = _commit_payload(
        decision_set_sha256=set_sha, decision_set=supplied
    )
    if canonical_json_bytes(expected_commit) != commit_bytes:
        raise Stage2RC5AtomicDecisionSetError(
            "decision-set commit does not exactly replay"
        )
    expected = build_stage2_rc5_atomic_decision_set_payload(
        producer_bundle=producer_bundle,
        source_threshold_reference=source_threshold_reference,
        inference_seals=inference_seals,
        evt_seal=evt_seal,
    )
    expected_bytes = canonical_json_bytes(expected)
    if expected_bytes != set_bytes or expected != supplied:
        raise Stage2RC5AtomicDecisionSetError(
            "decision set differs from full verifier-capability replay"
        )
    return _issue_decision_set(
        set_path=set_candidate,
        commit_path=commit_candidate,
        set_sha=set_sha,
        commit_sha=commit_sha,
        payload=expected,
    )


def publish_stage2_rc5_atomic_decision_set(
    output_directory: str | Path,
    *,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    source_threshold_reference: VerifiedExactSourceThresholdReferenceV3,
    inference_seals: Mapping[str, VerifiedStage2RC5InferenceSeal],
    evt_seal: VerifiedStage2RC5EVTSeal | None = None,
    repository_root: str | Path,
) -> VerifiedStage2RC5AtomicDecisionSet:
    """Publish canonical set then commit last, without replacement."""

    root = _repository_root(repository_root)
    output = _direct_path(
        output_directory, root, "decision-set output directory", require_file=False
    )
    final_set = output / DECISION_SET_FILENAME
    final_commit = output / DECISION_SET_COMMIT_FILENAME
    for path in (final_set, final_commit):
        if os.path.lexists(path):
            raise FileExistsError(f"immutable decision-set target exists: {path}")
    payload = build_stage2_rc5_atomic_decision_set_payload(
        producer_bundle=producer_bundle,
        source_threshold_reference=source_threshold_reference,
        inference_seals=inference_seals,
        evt_seal=evt_seal,
    )
    set_bytes = canonical_json_bytes(payload)
    set_sha = hashlib.sha256(set_bytes).hexdigest()
    commit = _commit_payload(
        decision_set_sha256=set_sha, decision_set=payload
    )
    commit_bytes = canonical_json_bytes(commit)
    commit_sha = hashlib.sha256(commit_bytes).hexdigest()
    staging = Path(tempfile.mkdtemp(prefix=".rc5-decision-staging-", dir=output))
    staged_set = staging / DECISION_SET_FILENAME
    staged_commit = staging / DECISION_SET_COMMIT_FILENAME
    published: list[Path] = []
    try:
        _write_exclusive(staged_set, set_bytes)
        _write_exclusive(staged_commit, commit_bytes)
        _fsync_directory(staging)
        for source, destination in (
            (staged_set, final_set),
            (staged_commit, final_commit),
        ):
            os.link(source, destination, follow_symlinks=False)
            published.append(destination)
            _fsync_directory(output)
        return verify_stage2_rc5_atomic_decision_set(
            decision_set_path=final_set,
            commit_path=final_commit,
            expected_commit_sha256=commit_sha,
            producer_bundle=producer_bundle,
            source_threshold_reference=source_threshold_reference,
            inference_seals=inference_seals,
            evt_seal=evt_seal,
            repository_root=root,
        )
    except BaseException:
        for path in reversed(published):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(output)
        raise
    finally:
        for path in (staged_commit, staged_set):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            staging.rmdir()
        except FileNotFoundError:
            pass


def guarded_invoke_stage2_rc5_label_resolver(
    *,
    decision_set_path: str | Path,
    commit_path: str | Path,
    expected_commit_sha256: str,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    source_threshold_reference: VerifiedExactSourceThresholdReferenceV3,
    inference_seals: Mapping[str, VerifiedStage2RC5InferenceSeal],
    label_resolver: Callable[..., Any],
    evt_seal: VerifiedStage2RC5EVTSeal | None = None,
    repository_root: str | Path,
    resolver_args: Sequence[Any] = (),
    resolver_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Invoke label resolution only after full atomic semantic verification."""

    if not callable(label_resolver):
        raise TypeError("label_resolver must be callable")
    if isinstance(resolver_args, (str, bytes)) or not isinstance(
        resolver_args, Sequence
    ):
        raise TypeError("resolver_args must be a sequence")
    if resolver_kwargs is not None and not isinstance(resolver_kwargs, Mapping):
        raise TypeError("resolver_kwargs must be a mapping")
    verified = verify_stage2_rc5_atomic_decision_set(
        decision_set_path=decision_set_path,
        commit_path=commit_path,
        expected_commit_sha256=expected_commit_sha256,
        producer_bundle=producer_bundle,
        source_threshold_reference=source_threshold_reference,
        inference_seals=inference_seals,
        evt_seal=evt_seal,
        repository_root=repository_root,
    )
    return label_resolver(
        verified,
        *tuple(resolver_args),
        **dict(resolver_kwargs or {}),
    )


def build_stage2_rc5_t9_postlabel_diagnostic(
    *,
    prelabel_decision_set: VerifiedStage2RC5AtomicDecisionSet,
    thresholds: np.ndarray,
    false_positive_pixels: np.ndarray,
    matched_objects: np.ndarray,
    total_native_pixels: int,
    ground_truth_objects: int,
) -> Mapping[str, Any]:
    """Build a separate post-label T9 diagnostic that is never prelabel authority."""

    sealed = assert_verified_stage2_rc5_atomic_decision_set(
        prelabel_decision_set
    )
    selection = select_exact_oracle_v2(
        thresholds=thresholds,
        false_positive_pixels=false_positive_pixels,
        matched_objects=matched_objects,
        total_native_pixels=total_native_pixels,
        ground_truth_objects=ground_truth_objects,
    )
    payload: dict[str, Any] = {
        "schema_version": T9_DIAGNOSTIC_SCHEMA,
        "artifact_status": "postlabel_diagnostic_only",
        "method_id": "T9",
        "prelabel_eligible": False,
        "prelabel_decision_set_identity_sha256": (
            sealed.decision_set_identity_sha256
        ),
        "budget_rationals": _budget_payload(),
        "budget_count_rule": EXACT_BUDGET_COUNT_RULE,
        "rows": _oracle_rows_from_selection(selection),
        "target_labels_accessed": True,
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    payload["diagnostic_identity_sha256"] = _self_hash(
        payload, "diagnostic_identity_sha256"
    )
    return _freeze(payload)


__all__ = [
    "BASELINE_PREFIX_SCHEMA",
    "DECISION_SCHEMA",
    "DECISION_SET_COMMIT_FILENAME",
    "DECISION_SET_COMMIT_SCHEMA",
    "DECISION_SET_FILENAME",
    "DECISION_SET_SCHEMA",
    "EVT_SEAL_SCHEMA",
    "EXACT_BUDGET_COUNT_RULE",
    "METHOD_IDS",
    "PUBLICATION_ORDER",
    "SOURCE_CURVE_SCHEMA",
    "SOURCE_REFERENCE_SCHEMA",
    "Stage2RC5AtomicDecisionSetError",
    "T9_DIAGNOSTIC_SCHEMA",
    "VerifiedExactSourceDomainCurveV2",
    "VerifiedExactSourceThresholdReferenceV3",
    "VerifiedStage2RC5AtomicDecisionSet",
    "VerifiedStage2RC5EVTSeal",
    "assert_verified_exact_source_domain_curve_v2",
    "assert_verified_exact_source_threshold_reference_v3",
    "assert_verified_stage2_rc5_atomic_decision_set",
    "assert_verified_stage2_rc5_evt_seal",
    "build_exact_source_domain_curve_v2",
    "build_exact_source_threshold_reference_v3",
    "build_stage2_rc5_atomic_decision_set_payload",
    "build_stage2_rc5_baseline_decision_prefix_payload",
    "build_stage2_rc5_evt_seal_complete",
    "build_stage2_rc5_t9_postlabel_diagnostic",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "guarded_invoke_stage2_rc5_label_resolver",
    "publish_stage2_rc5_atomic_decision_set",
    "verify_stage2_rc5_atomic_decision_set",
]

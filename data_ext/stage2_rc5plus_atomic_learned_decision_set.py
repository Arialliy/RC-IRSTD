"""Commit-last pre-label authority for the three RC5+ learned routes.

This additive artifact atomically seals T6+/T7+/T8+ from verifier-issued v8
inference seals before any query member or label resolver may run.  It closes
the learned-method branch; the final pre-label authority is T0--T8, while the
label-dependent T9 oracle remains a separate post-label diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any

from data_ext.stage2_rc5_atomic_decision_set import (
    _direct_path,
    _fsync_directory,
    _parse_canonical_json,
    _repository_root,
    _stable_read,
    _write_exclusive,
)
from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from rc.build_stage2_rc5_context import (
    VerifiedStage2RC5ContextBundle,
    replay_verified_stage2_rc5_context_bundle,
)
from rc.stage2_calibrator_checkpoint_v8 import VerifiedCalibratorCheckpointV8
from rc.stage2_context_tail_anchor_v2 import VerifiedContextTailAnchorV2
from rc.stage2_rc5_infer_and_seal import _producer_bundle_binding
from rc.stage2_rc5plus_infer_and_seal import (
    DECISION_SCHEMA as INFERENCE_DECISION_SCHEMA,
    TRANSCRIPT_SCHEMA as INFERENCE_TRANSCRIPT_SCHEMA,
    VerifiedStage2RC5PlusInferenceSeal,
    assert_verified_stage2_rc5plus_inference_seal,
    canonical_json_bytes,
    verify_stage2_rc5plus_inference_seal,
)


DECISION_SET_SCHEMA = "rc-irstd.stage2-rc5plus-atomic-learned-decision-set.v1"
COMMIT_SCHEMA = "rc-irstd.stage2-rc5plus-atomic-learned-decision-set-commit.v1"
DECISION_SCHEMA = "rc-irstd.stage2-rc5plus-prelabel-learned-decision.v1"
DECISION_SET_ARTIFACT_TYPE = (
    "rc_irstd_stage2_rc5plus_atomic_t6plus_t8plus_decision_set"
)
COMMIT_ARTIFACT_TYPE = (
    "rc_irstd_stage2_rc5plus_atomic_t6plus_t8plus_decision_set_commit"
)
DECISION_SET_FILENAME = "rc5plus_t6plus_t8plus_decision_set.json"
COMMIT_FILENAME = "rc5plus_t6plus_t8plus_decision_set.commit.json"
PUBLICATION_ORDER = "canonical_decision_set_then_commit_last_before_labels"
SELF_HASH_ALGORITHM = "sha256-canonical-json-with-self-field-omitted-v1"
METHOD_IDS = ("T6_PLUS", "T7_PLUS", "T8_PLUS")

_SHA_HEX = frozenset("0123456789abcdef")
_TOKEN = object()
_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "publication_order",
        "method_ids",
        "shared_prelabel_identity",
        "grid_budget_rationals",
        "requested_budget_rationals",
        "decisions",
        "labels_accessed",
        "query_members_opened",
        "reject",
        "fallback",
        "self_hash_algorithm",
        "decision_set_identity_sha256",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_status",
        "method_id",
        "outcome",
        "deployed_rows_source",
        "grid_budget_rationals",
        "requested_budget_rationals",
        "grid_rows",
        "requested_rows",
        "authority",
        "shared_prelabel_identity_sha256",
        "labels_accessed",
        "query_members_opened",
        "caller_float_budget_authority",
        "caller_threshold_injection",
        "reject",
        "fallback",
        "self_hash_algorithm",
        "decision_identity_sha256",
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
        "method_ids",
        "labels_accessed",
        "query_members_opened",
        "self_hash_algorithm",
        "commit_identity_sha256",
    }
)


class Stage2RC5PlusAtomicDecisionSetError(ValueError):
    """The learned-route pre-label atomic set failed semantic replay."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _SHA_HEX
    ):
        raise Stage2RC5PlusAtomicDecisionSetError(
            f"{name} must be lowercase SHA-256"
        )
    return value


def _fields(value: Any, fields: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Stage2RC5PlusAtomicDecisionSetError(f"{name} field closure mismatch")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {key: item for key, item in value.items() if key != field}
        )
    ).hexdigest()


def _budgets() -> list[dict[str, int]]:
    return [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in BUDGET_KNOT_RATIONALS
    ]


def _assert_false_fields(value: Mapping[str, Any], fields: Sequence[str], name: str) -> None:
    for field in fields:
        if type(value.get(field)) is not bool or value[field] is not False:
            raise Stage2RC5PlusAtomicDecisionSetError(
                f"{name}.{field} must be exact false"
            )


def _verified_seal_material(
    value: VerifiedStage2RC5PlusInferenceSeal,
    *,
    method: str,
    expected_producer_binding: Mapping[str, Any],
) -> tuple[VerifiedStage2RC5PlusInferenceSeal, Mapping[str, Any], Mapping[str, Any]]:
    seal = assert_verified_stage2_rc5plus_inference_seal(value)
    if seal.method != method:
        raise Stage2RC5PlusAtomicDecisionSetError(f"{method} seal method mismatch")
    transcript = seal.transcript
    decision = seal.decision
    if (
        transcript.get("schema_version") != INFERENCE_TRANSCRIPT_SCHEMA
        or decision.get("schema_version") != INFERENCE_DECISION_SCHEMA
        or transcript.get("method") != method
        or decision.get("method") != method
    ):
        raise Stage2RC5PlusAtomicDecisionSetError(
            f"{method} inference seal schema/method mismatch"
        )
    if _plain(transcript.get("producer_bundle_binding")) != _plain(
        expected_producer_binding
    ):
        raise Stage2RC5PlusAtomicDecisionSetError(
            f"{method} uses a different producer bundle"
        )
    if _plain(transcript.get("grid_budget_rationals")) != _budgets() or _plain(
        decision.get("grid_budget_rationals")
    ) != _budgets():
        raise Stage2RC5PlusAtomicDecisionSetError(
            f"{method} grid budget lattice mismatch"
        )
    if _plain(transcript.get("requested_budget_rationals")) != _plain(
        decision.get("requested_budget_rationals")
    ):
        raise Stage2RC5PlusAtomicDecisionSetError(
            f"{method} requested budget identity mismatch"
        )
    guardrails = transcript.get("guardrails")
    if not isinstance(guardrails, Mapping) or any(
        type(item) is not bool or item for item in guardrails.values()
    ):
        raise Stage2RC5PlusAtomicDecisionSetError(
            f"{method} seal records forbidden access"
        )
    _assert_false_fields(
        decision,
        (
            "labels_accessed",
            "query_accessed",
            "caller_float_budget_authority",
            "caller_threshold_injection",
            "reject",
            "fallback",
        ),
        f"{method}.decision",
    )
    if (
        not isinstance(decision.get("grid_rows"), Sequence)
        or len(decision["grid_rows"]) != len(BUDGET_KNOT_RATIONALS)
        or not isinstance(decision.get("requested_rows"), Sequence)
        or len(decision["requested_rows"])
        != len(decision["requested_budget_rationals"])
        or decision.get("deployed_rows_source")
        != ("requested" if decision["requested_rows"] else "grid")
    ):
        raise Stage2RC5PlusAtomicDecisionSetError(
            f"{method} sealed row cardinality/source mismatch"
        )
    return seal, transcript, decision


def _build_material_payload(
    *,
    inference_seals: Mapping[str, VerifiedStage2RC5PlusInferenceSeal],
    producer_bundle_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(inference_seals, Mapping) or tuple(inference_seals) != METHOD_IDS:
        raise Stage2RC5PlusAtomicDecisionSetError(
            "inference_seals keys/order must be exactly T6_PLUS,T7_PLUS,T8_PLUS"
        )
    material = {
        method: _verified_seal_material(
            inference_seals[method],
            method=method,
            expected_producer_binding=producer_bundle_binding,
        )
        for method in METHOD_IDS
    }
    transcripts = [material[method][1] for method in METHOD_IDS]
    shared_fields = (
        "context_binding",
        "anchor_v2_binding",
        "standardizer_binding",
        "model_input_binding",
        "grid_budget_rationals",
        "requested_budget_rationals",
    )
    reference = transcripts[0]
    for method, transcript in zip(METHOD_IDS[1:], transcripts[1:], strict=True):
        for field in shared_fields:
            if _plain(transcript.get(field)) != _plain(reference.get(field)):
                raise Stage2RC5PlusAtomicDecisionSetError(
                    f"{method} does not share the same pre-label {field}"
                )
    checkpoint_bindings = [
        transcript.get("checkpoint_binding") for transcript in transcripts
    ]
    if any(not isinstance(item, Mapping) for item in checkpoint_bindings):
        raise Stage2RC5PlusAtomicDecisionSetError("checkpoint binding is missing")
    training_view_ids = {
        item.get("training_view_identity_sha256")  # type: ignore[union-attr]
        for item in checkpoint_bindings
    }
    if len(training_view_ids) != 1:
        raise Stage2RC5PlusAtomicDecisionSetError(
            "learned methods do not share one frozen training-view identity"
        )
    shared: dict[str, Any] = {
        "producer_bundle_binding": _plain(producer_bundle_binding),
        "context_binding": _plain(reference["context_binding"]),
        "anchor_v2_binding": _plain(reference["anchor_v2_binding"]),
        "standardizer_binding": _plain(reference["standardizer_binding"]),
        "model_input_binding": _plain(reference["model_input_binding"]),
        "training_view_identity_sha256": next(iter(training_view_ids)),
    }
    _sha(shared["training_view_identity_sha256"], "training view identity")
    shared["shared_identity_sha256"] = _self_hash(
        shared, "shared_identity_sha256"
    )
    shared_sha = shared["shared_identity_sha256"]

    decisions: list[dict[str, Any]] = []
    for method in METHOD_IDS:
        seal, transcript, sealed_decision = material[method]
        checkpoint = transcript["checkpoint_binding"]
        authority = {
            "authority_kind": "VerifiedStage2RC5PlusInferenceSeal",
            "transcript_schema": transcript["schema_version"],
            "transcript_bytes_sha256": seal.transcript_bytes_sha256,
            "transcript_identity_sha256": seal.transcript_identity_sha256,
            "sealed_decision_identity_sha256": seal.decision_identity_sha256,
            "checkpoint_bytes_sha256": checkpoint["checkpoint_bytes_sha256"],
            "training_contract_sha256": checkpoint["training_contract_sha256"],
            "training_view_identity_sha256": checkpoint[
                "training_view_identity_sha256"
            ],
        }
        for field, item in authority.items():
            if field.endswith("sha256"):
                _sha(item, f"{method}.authority.{field}")
        decision: dict[str, Any] = {
            "schema_version": DECISION_SCHEMA,
            "artifact_status": "prelabel_complete",
            "method_id": method,
            "outcome": "complete",
            "deployed_rows_source": sealed_decision["deployed_rows_source"],
            "grid_budget_rationals": _budgets(),
            "requested_budget_rationals": _plain(
                sealed_decision["requested_budget_rationals"]
            ),
            "grid_rows": _plain(sealed_decision["grid_rows"]),
            "requested_rows": _plain(sealed_decision["requested_rows"]),
            "authority": authority,
            "shared_prelabel_identity_sha256": shared_sha,
            "labels_accessed": False,
            "query_members_opened": False,
            "caller_float_budget_authority": False,
            "caller_threshold_injection": False,
            "reject": False,
            "fallback": False,
            "self_hash_algorithm": SELF_HASH_ALGORITHM,
        }
        decision["decision_identity_sha256"] = _self_hash(
            decision, "decision_identity_sha256"
        )
        decisions.append(decision)
    payload: dict[str, Any] = {
        "schema_version": DECISION_SET_SCHEMA,
        "artifact_type": DECISION_SET_ARTIFACT_TYPE,
        "artifact_status": "atomic_prelabel_complete",
        "publication_order": PUBLICATION_ORDER,
        "method_ids": list(METHOD_IDS),
        "shared_prelabel_identity": shared,
        "grid_budget_rationals": _budgets(),
        "requested_budget_rationals": _plain(
            reference["requested_budget_rationals"]
        ),
        "decisions": decisions,
        "labels_accessed": False,
        "query_members_opened": False,
        "reject": False,
        "fallback": False,
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    payload["decision_set_identity_sha256"] = _self_hash(
        payload, "decision_set_identity_sha256"
    )
    return payload


def _replay_public_inputs(
    *,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    checkpoints: Mapping[str, VerifiedCalibratorCheckpointV8],
    inference_seals: Mapping[str, VerifiedStage2RC5PlusInferenceSeal],
    anchor_v2: VerifiedContextTailAnchorV2,
) -> tuple[
    VerifiedStage2RC5ContextBundle,
    dict[str, VerifiedStage2RC5PlusInferenceSeal],
]:
    if (
        not isinstance(checkpoints, Mapping)
        or tuple(checkpoints) != METHOD_IDS
        or not isinstance(inference_seals, Mapping)
        or tuple(inference_seals) != METHOD_IDS
    ):
        raise Stage2RC5PlusAtomicDecisionSetError(
            "checkpoint/seal keys/order must equal the three learned methods"
        )
    try:
        bundle = replay_verified_stage2_rc5_context_bundle(producer_bundle)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Stage2RC5PlusAtomicDecisionSetError(
            "producer bundle failed current-state replay"
        ) from error
    replayed: dict[str, VerifiedStage2RC5PlusInferenceSeal] = {}
    for method in METHOD_IDS:
        checkpoint = checkpoints[method]
        if checkpoint.method != method:
            raise Stage2RC5PlusAtomicDecisionSetError(
                f"{method} checkpoint method mismatch"
            )
        supplied = assert_verified_stage2_rc5plus_inference_seal(
            inference_seals[method]
        )
        verified = verify_stage2_rc5plus_inference_seal(
            supplied.transcript_bytes,
            checkpoint=checkpoint,
            producer_bundle=bundle,
            anchor_v2=anchor_v2,
        )
        if (
            verified.transcript_bytes_sha256 != supplied.transcript_bytes_sha256
            or verified.transcript_identity_sha256
            != supplied.transcript_identity_sha256
            or verified.decision_identity_sha256
            != supplied.decision_identity_sha256
        ):
            raise Stage2RC5PlusAtomicDecisionSetError(
                f"{method} retained seal differs from full replay"
            )
        replayed[method] = verified
    return bundle, replayed


def build_stage2_rc5plus_atomic_learned_decision_set_payload(
    *,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    checkpoints: Mapping[str, VerifiedCalibratorCheckpointV8],
    inference_seals: Mapping[str, VerifiedStage2RC5PlusInferenceSeal],
    anchor_v2: VerifiedContextTailAnchorV2,
) -> dict[str, Any]:
    bundle, replayed = _replay_public_inputs(
        producer_bundle=producer_bundle,
        checkpoints=checkpoints,
        inference_seals=inference_seals,
        anchor_v2=anchor_v2,
    )
    return _build_material_payload(
        inference_seals=replayed,
        producer_bundle_binding=_producer_bundle_binding(bundle),
    )


def _parse_payload(value: Any) -> dict[str, Any]:
    top = _fields(value, _TOP_FIELDS, "decision set")
    if (
        top["schema_version"] != DECISION_SET_SCHEMA
        or top["artifact_type"] != DECISION_SET_ARTIFACT_TYPE
        or top["artifact_status"] != "atomic_prelabel_complete"
        or top["publication_order"] != PUBLICATION_ORDER
        or top["method_ids"] != list(METHOD_IDS)
        or top["grid_budget_rationals"] != _budgets()
        or top["self_hash_algorithm"] != SELF_HASH_ALGORITHM
        or top["decision_set_identity_sha256"]
        != _self_hash(top, "decision_set_identity_sha256")
    ):
        raise Stage2RC5PlusAtomicDecisionSetError("decision-set contract drifted")
    _assert_false_fields(
        top,
        ("labels_accessed", "query_members_opened", "reject", "fallback"),
        "decision set",
    )
    shared = top["shared_prelabel_identity"]
    if (
        not isinstance(shared, Mapping)
        or shared.get("shared_identity_sha256")
        != _self_hash(shared, "shared_identity_sha256")
    ):
        raise Stage2RC5PlusAtomicDecisionSetError("shared identity mismatch")
    decisions = top["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(METHOD_IDS):
        raise Stage2RC5PlusAtomicDecisionSetError("decision cardinality mismatch")
    for method, raw in zip(METHOD_IDS, decisions, strict=True):
        decision = _fields(raw, _DECISION_FIELDS, f"decision {method}")
        if (
            decision["schema_version"] != DECISION_SCHEMA
            or decision["artifact_status"] != "prelabel_complete"
            or decision["method_id"] != method
            or decision["outcome"] != "complete"
            or decision["grid_budget_rationals"] != _budgets()
            or decision["requested_budget_rationals"]
            != top["requested_budget_rationals"]
            or decision["shared_prelabel_identity_sha256"]
            != shared["shared_identity_sha256"]
            or decision["self_hash_algorithm"] != SELF_HASH_ALGORITHM
            or decision["decision_identity_sha256"]
            != _self_hash(decision, "decision_identity_sha256")
        ):
            raise Stage2RC5PlusAtomicDecisionSetError(
                f"decision {method} contract drifted"
            )
        _assert_false_fields(
            decision,
            (
                "labels_accessed",
                "query_members_opened",
                "caller_float_budget_authority",
                "caller_threshold_injection",
                "reject",
                "fallback",
            ),
            f"decision {method}",
        )
    return _plain(top)


def _commit_payload(payload: Mapping[str, Any], set_sha: str) -> dict[str, Any]:
    commit: dict[str, Any] = {
        "schema_version": COMMIT_SCHEMA,
        "artifact_type": COMMIT_ARTIFACT_TYPE,
        "artifact_status": "committed_prelabel_learned_methods",
        "publication_order": PUBLICATION_ORDER,
        "decision_set_filename": DECISION_SET_FILENAME,
        "decision_set_sha256": _sha(set_sha, "decision_set_sha256"),
        "decision_set_identity_sha256": payload[
            "decision_set_identity_sha256"
        ],
        "shared_prelabel_identity_sha256": payload[
            "shared_prelabel_identity"
        ]["shared_identity_sha256"],
        "method_ids": list(METHOD_IDS),
        "labels_accessed": False,
        "query_members_opened": False,
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    commit["commit_identity_sha256"] = _self_hash(
        commit, "commit_identity_sha256"
    )
    return commit


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5PlusAtomicLearnedDecisionSet:
    decision_set_path: Path
    commit_path: Path
    decision_set_sha256: str
    commit_sha256: str
    decision_set_identity_sha256: str
    shared_prelabel_identity_sha256: str
    payload: Mapping[str, Any]
    decision_by_method: Mapping[str, Mapping[str, Any]]
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "VerifiedStage2RC5PlusAtomicLearnedDecisionSet is verifier-issued only"
        )

    def thresholds(
        self, method: str, *, requested: bool = False
    ) -> tuple[float, ...]:
        if method not in self.decision_by_method:
            raise KeyError(method)
        rows = self.decision_by_method[method][
            "requested_rows" if requested else "grid_rows"
        ]
        return tuple(float.fromhex(row["decoded_threshold_hex"]) for row in rows)


def _issue(
    *,
    set_path: Path,
    commit_path: Path,
    set_sha: str,
    commit_sha: str,
    payload: Mapping[str, Any],
) -> VerifiedStage2RC5PlusAtomicLearnedDecisionSet:
    frozen = _freeze(payload)
    value = object.__new__(VerifiedStage2RC5PlusAtomicLearnedDecisionSet)
    for name, item in {
        "decision_set_path": set_path,
        "commit_path": commit_path,
        "decision_set_sha256": set_sha,
        "commit_sha256": commit_sha,
        "decision_set_identity_sha256": payload["decision_set_identity_sha256"],
        "shared_prelabel_identity_sha256": payload[
            "shared_prelabel_identity"
        ]["shared_identity_sha256"],
        "payload": frozen,
        "decision_by_method": MappingProxyType(
            {row["method_id"]: row for row in frozen["decisions"]}
        ),
        "_capability": _TOKEN,
    }.items():
        object.__setattr__(value, name, item)
    return value


def assert_verified_stage2_rc5plus_atomic_learned_decision_set(
    value: VerifiedStage2RC5PlusAtomicLearnedDecisionSet,
) -> VerifiedStage2RC5PlusAtomicLearnedDecisionSet:
    if (
        type(value) is not VerifiedStage2RC5PlusAtomicLearnedDecisionSet
        or getattr(value, "_capability", None) is not _TOKEN
        or tuple(value.decision_by_method) != METHOD_IDS
    ):
        raise TypeError("a verifier-issued RC5+ learned atomic decision set is required")
    return value


def verify_stage2_rc5plus_atomic_learned_decision_set(
    *,
    decision_set_path: str | Path,
    commit_path: str | Path,
    expected_commit_sha256: str,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    checkpoints: Mapping[str, VerifiedCalibratorCheckpointV8],
    inference_seals: Mapping[str, VerifiedStage2RC5PlusInferenceSeal],
    anchor_v2: VerifiedContextTailAnchorV2,
    repository_root: str | Path,
) -> VerifiedStage2RC5PlusAtomicLearnedDecisionSet:
    root = _repository_root(repository_root)
    commit_candidate = _direct_path(commit_path, root, "RC5+ commit", require_file=True)
    if commit_candidate.name != COMMIT_FILENAME:
        raise Stage2RC5PlusAtomicDecisionSetError("commit filename mismatch")
    commit_sha = _sha(expected_commit_sha256, "expected_commit_sha256")
    commit_bytes = _stable_read(commit_candidate, commit_sha, root, "RC5+ commit")
    commit = _fields(
        _parse_canonical_json(commit_bytes, "RC5+ commit"),
        _COMMIT_FIELDS,
        "commit",
    )
    if (
        commit["schema_version"] != COMMIT_SCHEMA
        or commit["artifact_type"] != COMMIT_ARTIFACT_TYPE
        or commit["artifact_status"] != "committed_prelabel_learned_methods"
        or commit["publication_order"] != PUBLICATION_ORDER
        or commit["decision_set_filename"] != DECISION_SET_FILENAME
        or commit["method_ids"] != list(METHOD_IDS)
        or commit["self_hash_algorithm"] != SELF_HASH_ALGORITHM
        or commit["commit_identity_sha256"]
        != _self_hash(commit, "commit_identity_sha256")
    ):
        raise Stage2RC5PlusAtomicDecisionSetError("commit contract drifted")
    _assert_false_fields(commit, ("labels_accessed", "query_members_opened"), "commit")
    set_candidate = _direct_path(
        decision_set_path, root, "RC5+ decision set", require_file=True
    )
    if (
        set_candidate.name != DECISION_SET_FILENAME
        or set_candidate.parent != commit_candidate.parent
    ):
        raise Stage2RC5PlusAtomicDecisionSetError("set/commit layout mismatch")
    set_sha = _sha(commit["decision_set_sha256"], "commit set SHA")
    set_bytes = _stable_read(set_candidate, set_sha, root, "RC5+ decision set")
    if commit_candidate.stat(follow_symlinks=False).st_mtime_ns < set_candidate.stat(
        follow_symlinks=False
    ).st_mtime_ns:
        raise Stage2RC5PlusAtomicDecisionSetError("commit was not published last")
    supplied = _parse_payload(_parse_canonical_json(set_bytes, "RC5+ decision set"))
    expected = build_stage2_rc5plus_atomic_learned_decision_set_payload(
        producer_bundle=producer_bundle,
        checkpoints=checkpoints,
        inference_seals=inference_seals,
        anchor_v2=anchor_v2,
    )
    if canonical_json_bytes(expected) != set_bytes or supplied != expected:
        raise Stage2RC5PlusAtomicDecisionSetError(
            "decision set differs from full verifier-capability replay"
        )
    if (
        canonical_json_bytes(_commit_payload(expected, set_sha)) != commit_bytes
        or commit["decision_set_identity_sha256"]
        != expected["decision_set_identity_sha256"]
        or commit["shared_prelabel_identity_sha256"]
        != expected["shared_prelabel_identity"]["shared_identity_sha256"]
    ):
        raise Stage2RC5PlusAtomicDecisionSetError("commit identity replay mismatch")
    return _issue(
        set_path=set_candidate,
        commit_path=commit_candidate,
        set_sha=set_sha,
        commit_sha=commit_sha,
        payload=expected,
    )


def publish_stage2_rc5plus_atomic_learned_decision_set(
    output_directory: str | Path,
    *,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    checkpoints: Mapping[str, VerifiedCalibratorCheckpointV8],
    inference_seals: Mapping[str, VerifiedStage2RC5PlusInferenceSeal],
    anchor_v2: VerifiedContextTailAnchorV2,
    repository_root: str | Path,
) -> VerifiedStage2RC5PlusAtomicLearnedDecisionSet:
    root = _repository_root(repository_root)
    output = _direct_path(
        output_directory, root, "RC5+ decision-set output", require_file=False
    )
    final_set = output / DECISION_SET_FILENAME
    final_commit = output / COMMIT_FILENAME
    if any(os.path.lexists(path) for path in (final_set, final_commit)):
        raise FileExistsError("immutable RC5+ atomic decision-set target exists")
    payload = build_stage2_rc5plus_atomic_learned_decision_set_payload(
        producer_bundle=producer_bundle,
        checkpoints=checkpoints,
        inference_seals=inference_seals,
        anchor_v2=anchor_v2,
    )
    set_bytes = canonical_json_bytes(payload)
    set_sha = hashlib.sha256(set_bytes).hexdigest()
    commit = _commit_payload(payload, set_sha)
    commit_bytes = canonical_json_bytes(commit)
    commit_sha = hashlib.sha256(commit_bytes).hexdigest()
    staging = Path(tempfile.mkdtemp(prefix=".rc5plus-atomic-", dir=output))
    staged_set = staging / DECISION_SET_FILENAME
    staged_commit = staging / COMMIT_FILENAME
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
        return verify_stage2_rc5plus_atomic_learned_decision_set(
            decision_set_path=final_set,
            commit_path=final_commit,
            expected_commit_sha256=commit_sha,
            producer_bundle=producer_bundle,
            checkpoints=checkpoints,
            inference_seals=inference_seals,
            anchor_v2=anchor_v2,
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


def guarded_invoke_stage2_rc5plus_label_resolver(
    *,
    decision_set_path: str | Path,
    commit_path: str | Path,
    expected_commit_sha256: str,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    checkpoints: Mapping[str, VerifiedCalibratorCheckpointV8],
    inference_seals: Mapping[str, VerifiedStage2RC5PlusInferenceSeal],
    anchor_v2: VerifiedContextTailAnchorV2,
    label_resolver: Callable[..., Any],
    repository_root: str | Path,
    resolver_args: Sequence[Any] = (),
    resolver_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    if not callable(label_resolver):
        raise TypeError("label_resolver must be callable")
    if isinstance(resolver_args, (str, bytes)) or not isinstance(resolver_args, Sequence):
        raise TypeError("resolver_args must be a sequence")
    if resolver_kwargs is not None and not isinstance(resolver_kwargs, Mapping):
        raise TypeError("resolver_kwargs must be a mapping")
    verified = verify_stage2_rc5plus_atomic_learned_decision_set(
        decision_set_path=decision_set_path,
        commit_path=commit_path,
        expected_commit_sha256=expected_commit_sha256,
        producer_bundle=producer_bundle,
        checkpoints=checkpoints,
        inference_seals=inference_seals,
        anchor_v2=anchor_v2,
        repository_root=repository_root,
    )
    return label_resolver(
        verified,
        *tuple(resolver_args),
        **dict(resolver_kwargs or {}),
    )


__all__ = [
    "COMMIT_FILENAME",
    "COMMIT_SCHEMA",
    "DECISION_SCHEMA",
    "DECISION_SET_FILENAME",
    "DECISION_SET_SCHEMA",
    "METHOD_IDS",
    "Stage2RC5PlusAtomicDecisionSetError",
    "VerifiedStage2RC5PlusAtomicLearnedDecisionSet",
    "assert_verified_stage2_rc5plus_atomic_learned_decision_set",
    "build_stage2_rc5plus_atomic_learned_decision_set_payload",
    "guarded_invoke_stage2_rc5plus_label_resolver",
    "publish_stage2_rc5plus_atomic_learned_decision_set",
    "verify_stage2_rc5plus_atomic_learned_decision_set",
]

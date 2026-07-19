"""Baseline-inclusive RC5+ T0--T8 atomic pre-label decision authority.

T0--T5 are recomputed from current RC5 verifier capabilities.  T6--T8 are
the primary-budget projections of the complete nine-budget T6+/T7+/T8+
sealed curves.  T9 is deliberately absent because it is a post-label oracle
diagnostic; including it in a pre-label atomic set would be target-label
leakage, not completeness.
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
    EXACT_BUDGET_COUNT_RULE,
    METHOD_IDS,
    SELF_HASH_ALGORITHM,
    VerifiedExactSourceThresholdReferenceV3,
    VerifiedStage2RC5EVTSeal,
    _complete_decision,
    _plain,
    _self_hash,
    _threshold_row,
    build_stage2_rc5_baseline_decision_prefix_payload,
    canonical_json_bytes,
)
from data_ext.stage2_rc5plus_atomic_learned_decision_set import (
    METHOD_IDS as LEARNED_METHOD_IDS,
    _direct_path,
    _fsync_directory,
    _parse_canonical_json,
    _repository_root,
    _stable_read,
    _write_exclusive,
    build_stage2_rc5plus_atomic_learned_decision_set_payload,
)
from model.budget_conditioned_endpoint_calibrator import (
    BUDGET_KNOT_RATIONALS,
    PRIMARY_BUDGET_KNOT_INDICES,
)
from model.endpoint_aware_threshold import representation_contract
from rc.build_stage2_rc5_context import VerifiedStage2RC5ContextBundle
from rc.stage2_calibrator_checkpoint_v8 import (
    VerifiedCalibratorCheckpointV8,
)
from rc.stage2_context_tail_anchor_v2 import VerifiedContextTailAnchorV2
from rc.stage2_rc5plus_infer_and_seal import (
    VerifiedStage2RC5PlusInferenceSeal,
)


DECISION_SET_SCHEMA = "rc-irstd.stage2-rc5plus-atomic-full-decision-set.v1"
COMMIT_SCHEMA = "rc-irstd.stage2-rc5plus-atomic-full-decision-set-commit.v1"
ARTIFACT_TYPE = "rc_irstd_stage2_rc5plus_atomic_t0_t8_decision_set"
COMMIT_ARTIFACT_TYPE = (
    "rc_irstd_stage2_rc5plus_atomic_t0_t8_decision_set_commit"
)
DECISION_SET_FILENAME = "rc5plus_t0_t8_decision_set.json"
COMMIT_FILENAME = "rc5plus_t0_t8_decision_set.commit.json"
PUBLICATION_ORDER = "canonical_t0_t8_decision_set_then_commit_last_before_labels"
METHOD_MAP = {
    "T6_PLUS": "T6",
    "T7_PLUS": "T7",
    "T8_PLUS": "T8",
}
METHOD_NAMES = {
    "T6": "budget_conditioned_direct_residual_transport_calibrator",
    "T7": "budget_conditioned_monotone_residual_transport_calibrator",
    "T8": "risk_aligned_budget_conditioned_monotone_residual_transport_calibrator",
}
_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "publication_order",
        "method_ids",
        "primary_budget_rationals",
        "learned_complete_grid_budget_rationals",
        "budget_count_rule",
        "threshold_representation",
        "threshold_semantics",
        "shared_prelabel_identity",
        "baseline_prefix_identity_sha256",
        "learned_set_identity_sha256",
        "learned_shared_identity_sha256",
        "decisions",
        "t9_included",
        "t9_policy",
        "guardrails",
        "self_hash_algorithm",
        "decision_set_identity_sha256",
    }
)
_GUARDRAIL_FIELDS = frozenset(
    {
        "labels_accessed",
        "query_scores_opened",
        "query_images_opened",
        "query_labels_opened",
        "postlabel_statistics_accessed",
        "caller_threshold_injection",
        "float_budget_count_logic_used",
        "fallback_used",
        "reject_used",
    }
)
_TOKEN = object()
_PRIMARY_BUDGETS = tuple(
    BUDGET_KNOT_RATIONALS[index] for index in PRIMARY_BUDGET_KNOT_INDICES
)


class Stage2RC5PlusAtomicFullDecisionSetError(ValueError):
    """The full T0--T8 pre-label set failed exact capability replay."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _primary_budget_payload() -> list[dict[str, int]]:
    return [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in _PRIMARY_BUDGETS
    ]


def _grid_budget_payload() -> list[dict[str, int]]:
    return [
        {"numerator": numerator, "denominator": denominator}
        for numerator, denominator in BUDGET_KNOT_RATIONALS
    ]


def _assert_shared_identity(
    baseline: Mapping[str, Any], learned: Mapping[str, Any]
) -> None:
    old = baseline["shared_prelabel_identity"]
    new = learned["shared_prelabel_identity"]
    producer = new["producer_bundle_binding"]
    context = new["context_binding"]
    for old_field, new_value in (
        ("producer_identity_sha256", producer["producer_identity_sha256"]),
        ("producer_bundle_identity_sha256", producer["bundle_identity_sha256"]),
        ("producer_manifest_sha256", producer["producer_manifest_sha256"]),
        ("producer_commit_sha256", producer["commit_sha256"]),
        ("context_full_identity_sha256", context["context_full_identity_sha256"]),
        ("context_payload_sha256", context["context_payload_sha256"]),
        ("context_feature_vector_sha256", context["context_feature_vector_sha256"]),
    ):
        if old.get(old_field) != new_value:
            raise Stage2RC5PlusAtomicFullDecisionSetError(
                f"baseline/learned shared identity mismatch: {old_field}"
            )


def _learned_primary_decision(
    *,
    learned_method: str,
    learned_decision: Mapping[str, Any],
    shared_identity_sha256: str,
) -> dict[str, Any]:
    old_method = METHOD_MAP[learned_method]
    rows = learned_decision["grid_rows"]
    if not isinstance(rows, Sequence) or len(rows) != len(BUDGET_KNOT_RATIONALS):
        raise Stage2RC5PlusAtomicFullDecisionSetError(
            f"{learned_method} complete nine-budget rows are missing"
        )
    primary_rows = []
    for primary_index, grid_index in enumerate(PRIMARY_BUDGET_KNOT_INDICES):
        row = rows[grid_index]
        if (
            not isinstance(row, Mapping)
            or (
                row.get("budget_numerator"),
                row.get("budget_denominator"),
            )
            != BUDGET_KNOT_RATIONALS[grid_index]
        ):
            raise Stage2RC5PlusAtomicFullDecisionSetError(
                f"{learned_method} primary grid row identity mismatch"
            )
        primary_rows.append(
            _threshold_row(
                index=primary_index,
                probability=float.fromhex(row["decoded_threshold_hex"]),
                coordinate=float.fromhex(row["canonical_coordinate_hex"]),
                relation="decode_coordinate",
            )
        )
    authority = {
        "authority_kind": "VerifiedStage2RC5PlusInferenceSealPrimaryProjection",
        "learned_method_id": learned_method,
        "complete_grid_budget_rationals": _grid_budget_payload(),
        "primary_grid_indices": list(PRIMARY_BUDGET_KNOT_INDICES),
        "learned_decision_identity_sha256": learned_decision[
            "decision_identity_sha256"
        ],
        "transcript_bytes_sha256": learned_decision["authority"][
            "transcript_bytes_sha256"
        ],
        "checkpoint_bytes_sha256": learned_decision["authority"][
            "checkpoint_bytes_sha256"
        ],
        "training_contract_sha256": learned_decision["authority"][
            "training_contract_sha256"
        ],
        "training_view_identity_sha256": learned_decision["authority"][
            "training_view_identity_sha256"
        ],
    }
    decision = _complete_decision(
        method_id=old_method,
        rows=primary_rows,
        authority=authority,
        shared_identity_sha256=shared_identity_sha256,
    )
    decision["method_name"] = METHOD_NAMES[old_method]
    decision["decision_identity_sha256"] = _self_hash(
        decision, "decision_identity_sha256"
    )
    return decision


def build_stage2_rc5plus_atomic_full_decision_set_payload(
    *,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    source_threshold_reference: VerifiedExactSourceThresholdReferenceV3,
    checkpoints: Mapping[str, VerifiedCalibratorCheckpointV8],
    inference_seals: Mapping[str, VerifiedStage2RC5PlusInferenceSeal],
    anchor_v2: VerifiedContextTailAnchorV2,
    evt_seal: VerifiedStage2RC5EVTSeal | None = None,
) -> dict[str, Any]:
    baseline = build_stage2_rc5_baseline_decision_prefix_payload(
        producer_bundle=producer_bundle,
        source_threshold_reference=source_threshold_reference,
        evt_seal=evt_seal,
    )
    learned = build_stage2_rc5plus_atomic_learned_decision_set_payload(
        producer_bundle=producer_bundle,
        checkpoints=checkpoints,
        inference_seals=inference_seals,
        anchor_v2=anchor_v2,
    )
    _assert_shared_identity(baseline, learned)
    shared = baseline["shared_prelabel_identity"]
    shared_sha = shared["shared_identity_sha256"]
    learned_by_method = {
        decision["method_id"]: decision for decision in learned["decisions"]
    }
    if tuple(learned_by_method) != LEARNED_METHOD_IDS:
        raise Stage2RC5PlusAtomicFullDecisionSetError(
            "learned decision order is not exact T6_PLUS,T7_PLUS,T8_PLUS"
        )
    decisions = [_plain(row) for row in baseline["decisions"]]
    decisions.extend(
        _learned_primary_decision(
            learned_method=method,
            learned_decision=learned_by_method[method],
            shared_identity_sha256=shared_sha,
        )
        for method in LEARNED_METHOD_IDS
    )
    if tuple(row["method_id"] for row in decisions) != METHOD_IDS:
        raise RuntimeError("full RC5+ decision order is not exact T0--T8")
    payload: dict[str, Any] = {
        "schema_version": DECISION_SET_SCHEMA,
        "artifact_type": ARTIFACT_TYPE,
        "artifact_status": "prelabel_atomic_t0_t8_complete",
        "publication_order": PUBLICATION_ORDER,
        "method_ids": list(METHOD_IDS),
        "primary_budget_rationals": _primary_budget_payload(),
        "learned_complete_grid_budget_rationals": _grid_budget_payload(),
        "budget_count_rule": EXACT_BUDGET_COUNT_RULE,
        "threshold_representation": representation_contract(),
        "threshold_semantics": "prediction = probability > threshold",
        "shared_prelabel_identity": _plain(shared),
        "baseline_prefix_identity_sha256": baseline["prefix_identity_sha256"],
        "learned_set_identity_sha256": learned["decision_set_identity_sha256"],
        "learned_shared_identity_sha256": learned[
            "shared_prelabel_identity"
        ]["shared_identity_sha256"],
        "decisions": decisions,
        "t9_included": False,
        "t9_policy": "separate_postlabel_oracle_diagnostic_only",
        "guardrails": {
            "labels_accessed": False,
            "query_scores_opened": False,
            "query_images_opened": False,
            "query_labels_opened": False,
            "postlabel_statistics_accessed": False,
            "caller_threshold_injection": False,
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


def _commit_payload(payload: Mapping[str, Any], set_sha: str) -> dict[str, Any]:
    commit: dict[str, Any] = {
        "schema_version": COMMIT_SCHEMA,
        "artifact_type": COMMIT_ARTIFACT_TYPE,
        "artifact_status": "committed_prelabel_t0_t8",
        "publication_order": PUBLICATION_ORDER,
        "decision_set_filename": DECISION_SET_FILENAME,
        "decision_set_sha256": set_sha,
        "decision_set_identity_sha256": payload[
            "decision_set_identity_sha256"
        ],
        "method_ids": list(METHOD_IDS),
        "t9_included": False,
        "labels_accessed": False,
        "query_members_opened": False,
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    commit["commit_identity_sha256"] = _self_hash(
        commit, "commit_identity_sha256"
    )
    return commit


def _parse_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage2RC5PlusAtomicFullDecisionSetError(
            "full decision set must be a mapping"
        )
    payload = _plain(value)
    if set(payload) != _TOP_FIELDS:
        raise Stage2RC5PlusAtomicFullDecisionSetError(
            "full decision-set field closure drifted"
        )
    if (
        payload.get("schema_version") != DECISION_SET_SCHEMA
        or payload.get("artifact_type") != ARTIFACT_TYPE
        or payload.get("artifact_status") != "prelabel_atomic_t0_t8_complete"
        or payload.get("publication_order") != PUBLICATION_ORDER
        or payload.get("method_ids") != list(METHOD_IDS)
        or payload.get("primary_budget_rationals") != _primary_budget_payload()
        or payload.get("learned_complete_grid_budget_rationals")
        != _grid_budget_payload()
        or payload.get("t9_included") is not False
        or payload.get("t9_policy")
        != "separate_postlabel_oracle_diagnostic_only"
        or payload.get("self_hash_algorithm") != SELF_HASH_ALGORITHM
        or payload.get("decision_set_identity_sha256")
        != _self_hash(payload, "decision_set_identity_sha256")
    ):
        raise Stage2RC5PlusAtomicFullDecisionSetError(
            "full decision-set identity contract drifted"
        )
    decisions = payload.get("decisions")
    if (
        not isinstance(decisions, list)
        or len(decisions) != len(METHOD_IDS)
        or any(not isinstance(row, Mapping) for row in decisions)
        or tuple(row["method_id"] for row in decisions) != METHOD_IDS
    ):
        raise Stage2RC5PlusAtomicFullDecisionSetError(
            "full decision-set method order drifted"
        )
    guardrails = payload.get("guardrails")
    if (
        not isinstance(guardrails, Mapping)
        or set(guardrails) != _GUARDRAIL_FIELDS
        or any(
        type(value) is not bool or value for value in guardrails.values()
        )
    ):
        raise Stage2RC5PlusAtomicFullDecisionSetError(
            "full decision-set guardrails drifted"
        )
    return payload


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5PlusAtomicFullDecisionSet:
    decision_set_path: Path
    commit_path: Path
    decision_set_sha256: str
    commit_sha256: str
    decision_set_identity_sha256: str
    payload: Mapping[str, Any]
    decision_by_method: Mapping[str, Mapping[str, Any]]
    _capability: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "VerifiedStage2RC5PlusAtomicFullDecisionSet is verifier-issued only"
        )

    def thresholds(self, method: str) -> tuple[float, ...] | None:
        if method not in self.decision_by_method:
            raise KeyError(method)
        decision = self.decision_by_method[method]
        if decision["outcome"] != "complete":
            return None
        return tuple(
            float.fromhex(row["threshold_probability_hex"])
            for row in decision["rows"]
        )


def _issue(
    *,
    set_path: Path,
    commit_path: Path,
    set_sha: str,
    commit_sha: str,
    payload: Mapping[str, Any],
) -> VerifiedStage2RC5PlusAtomicFullDecisionSet:
    frozen = _freeze(payload)
    result = object.__new__(VerifiedStage2RC5PlusAtomicFullDecisionSet)
    for name, value in {
        "decision_set_path": set_path,
        "commit_path": commit_path,
        "decision_set_sha256": set_sha,
        "commit_sha256": commit_sha,
        "decision_set_identity_sha256": payload[
            "decision_set_identity_sha256"
        ],
        "payload": frozen,
        "decision_by_method": MappingProxyType(
            {row["method_id"]: row for row in frozen["decisions"]}
        ),
        "_capability": _TOKEN,
    }.items():
        object.__setattr__(result, name, value)
    return result


def assert_verified_stage2_rc5plus_atomic_full_decision_set(
    value: VerifiedStage2RC5PlusAtomicFullDecisionSet,
) -> VerifiedStage2RC5PlusAtomicFullDecisionSet:
    if (
        type(value) is not VerifiedStage2RC5PlusAtomicFullDecisionSet
        or getattr(value, "_capability", None) is not _TOKEN
        or tuple(value.decision_by_method) != METHOD_IDS
    ):
        raise TypeError(
            "a verifier-issued RC5+ full atomic decision set is required"
        )
    parsed = _parse_payload(value.payload)
    if (
        value.decision_set_identity_sha256
        != parsed["decision_set_identity_sha256"]
        or tuple(value.decision_by_method) != tuple(
            row["method_id"] for row in parsed["decisions"]
        )
    ):
        raise TypeError("RC5+ full atomic decision capability drifted")
    return value


def verify_stage2_rc5plus_atomic_full_decision_set(
    *,
    decision_set_path: str | Path,
    commit_path: str | Path,
    expected_commit_sha256: str,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    source_threshold_reference: VerifiedExactSourceThresholdReferenceV3,
    checkpoints: Mapping[str, VerifiedCalibratorCheckpointV8],
    inference_seals: Mapping[str, VerifiedStage2RC5PlusInferenceSeal],
    anchor_v2: VerifiedContextTailAnchorV2,
    repository_root: str | Path,
    evt_seal: VerifiedStage2RC5EVTSeal | None = None,
) -> VerifiedStage2RC5PlusAtomicFullDecisionSet:
    root = _repository_root(repository_root)
    commit_candidate = _direct_path(commit_path, root, "full commit", require_file=True)
    set_candidate = _direct_path(
        decision_set_path, root, "full decision set", require_file=True
    )
    if (
        commit_candidate.name != COMMIT_FILENAME
        or set_candidate.name != DECISION_SET_FILENAME
        or commit_candidate.parent != set_candidate.parent
    ):
        raise Stage2RC5PlusAtomicFullDecisionSetError(
            "full set/commit layout mismatch"
        )
    commit_bytes = _stable_read(
        commit_candidate, expected_commit_sha256, root, "full commit"
    )
    commit = _parse_canonical_json(commit_bytes, "full commit")
    if not isinstance(commit, Mapping):
        raise Stage2RC5PlusAtomicFullDecisionSetError("full commit is invalid")
    set_sha = commit.get("decision_set_sha256")
    if type(set_sha) is not str or len(set_sha) != 64:
        raise Stage2RC5PlusAtomicFullDecisionSetError("full set SHA is invalid")
    set_bytes = _stable_read(set_candidate, set_sha, root, "full decision set")
    if commit_candidate.stat(follow_symlinks=False).st_mtime_ns < set_candidate.stat(
        follow_symlinks=False
    ).st_mtime_ns:
        raise Stage2RC5PlusAtomicFullDecisionSetError("full commit was not last")
    supplied = _parse_payload(_parse_canonical_json(set_bytes, "full decision set"))
    expected = build_stage2_rc5plus_atomic_full_decision_set_payload(
        producer_bundle=producer_bundle,
        source_threshold_reference=source_threshold_reference,
        checkpoints=checkpoints,
        inference_seals=inference_seals,
        anchor_v2=anchor_v2,
        evt_seal=evt_seal,
    )
    expected_commit = _commit_payload(expected, set_sha)
    if (
        canonical_json_bytes(expected) != set_bytes
        or supplied != expected
        or canonical_json_bytes(expected_commit) != commit_bytes
    ):
        raise Stage2RC5PlusAtomicFullDecisionSetError(
            "full decision set differs from capability replay"
        )
    return _issue(
        set_path=set_candidate,
        commit_path=commit_candidate,
        set_sha=set_sha,
        commit_sha=expected_commit_sha256,
        payload=expected,
    )


def publish_stage2_rc5plus_atomic_full_decision_set(
    output_directory: str | Path,
    *,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    source_threshold_reference: VerifiedExactSourceThresholdReferenceV3,
    checkpoints: Mapping[str, VerifiedCalibratorCheckpointV8],
    inference_seals: Mapping[str, VerifiedStage2RC5PlusInferenceSeal],
    anchor_v2: VerifiedContextTailAnchorV2,
    repository_root: str | Path,
    evt_seal: VerifiedStage2RC5EVTSeal | None = None,
) -> VerifiedStage2RC5PlusAtomicFullDecisionSet:
    root = _repository_root(repository_root)
    output = _direct_path(output_directory, root, "full output", require_file=False)
    final_set = output / DECISION_SET_FILENAME
    final_commit = output / COMMIT_FILENAME
    if any(os.path.lexists(path) for path in (final_set, final_commit)):
        raise FileExistsError("immutable full decision-set target exists")
    payload = build_stage2_rc5plus_atomic_full_decision_set_payload(
        producer_bundle=producer_bundle,
        source_threshold_reference=source_threshold_reference,
        checkpoints=checkpoints,
        inference_seals=inference_seals,
        anchor_v2=anchor_v2,
        evt_seal=evt_seal,
    )
    set_bytes = canonical_json_bytes(payload)
    set_sha = hashlib.sha256(set_bytes).hexdigest()
    commit_bytes = canonical_json_bytes(_commit_payload(payload, set_sha))
    commit_sha = hashlib.sha256(commit_bytes).hexdigest()
    staging = Path(tempfile.mkdtemp(prefix=".rc5plus-full-", dir=output))
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
        return verify_stage2_rc5plus_atomic_full_decision_set(
            decision_set_path=final_set,
            commit_path=final_commit,
            expected_commit_sha256=commit_sha,
            producer_bundle=producer_bundle,
            source_threshold_reference=source_threshold_reference,
            checkpoints=checkpoints,
            inference_seals=inference_seals,
            anchor_v2=anchor_v2,
            repository_root=root,
            evt_seal=evt_seal,
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


def guarded_invoke_stage2_rc5plus_full_label_resolver(
    *,
    label_resolver: Callable[..., Any],
    resolver_args: Sequence[Any] = (),
    resolver_kwargs: Mapping[str, Any] | None = None,
    **verification_kwargs: Any,
) -> Any:
    if not callable(label_resolver):
        raise TypeError("label_resolver must be callable")
    if isinstance(resolver_args, (str, bytes)) or not isinstance(
        resolver_args, Sequence
    ):
        raise TypeError("resolver_args must be a sequence")
    if resolver_kwargs is not None and not isinstance(
        resolver_kwargs, Mapping
    ):
        raise TypeError("resolver_kwargs must be a mapping")
    verified = verify_stage2_rc5plus_atomic_full_decision_set(
        **verification_kwargs
    )
    return label_resolver(
        verified,
        *tuple(resolver_args),
        **dict(resolver_kwargs or {}),
    )


__all__ = [
    "COMMIT_FILENAME",
    "COMMIT_SCHEMA",
    "DECISION_SET_FILENAME",
    "DECISION_SET_SCHEMA",
    "PUBLICATION_ORDER",
    "Stage2RC5PlusAtomicFullDecisionSetError",
    "VerifiedStage2RC5PlusAtomicFullDecisionSet",
    "assert_verified_stage2_rc5plus_atomic_full_decision_set",
    "build_stage2_rc5plus_atomic_full_decision_set_payload",
    "guarded_invoke_stage2_rc5plus_full_label_resolver",
    "publish_stage2_rc5plus_atomic_full_decision_set",
    "verify_stage2_rc5plus_atomic_full_decision_set",
]

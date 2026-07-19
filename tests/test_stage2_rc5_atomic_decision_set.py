from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from data_ext import stage2_rc5_atomic_decision_set as atomic
from model.endpoint_aware_threshold import (
    UPPER_ENDPOINT_COORDINATE,
    decode_coordinate_numpy,
    encode_probability_numpy,
    endpoint_kinds_numpy,
)
from rc.stage2_context_tail_anchor import (
    build_context_tail_anchor,
    verify_context_tail_anchor,
)
from rc.stage2_exact_oracle_v2 import select_exact_oracle_v2


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _self_hash(payload: dict[str, Any], field: str) -> str:
    return atomic.canonical_json_sha256(
        {key: value for key, value in payload.items() if key != field}
    )


def _detector_identity() -> dict[str, Any]:
    return {
        "run_id": "outer_leave_C__seed_42__full",
        "outer_fold_id": "outer_leave_C",
        "outer_target": "C",
        "base_seed": 42,
        "derived_seed": 424242,
        "detector_role": "detector_full_fit",
        "oof_fold_index": None,
        "checkpoint_sha256": _sha("detector-checkpoint"),
    }


def _fake_source_reference_v3() -> Any:
    detector = _detector_identity()
    run_complete = {
        "path": "runs/synthetic/RUN_COMPLETE.v2.json",
        "sha256": _sha("run-complete-artifact"),
        "identity_sha256": _sha("run-complete-identity"),
    }
    score_rows = [
        {
            "source_domain": domain,
            "score_attestation": {
                "path": f"scores/{domain}/RC5_SCORE_BUNDLE_ATTESTATION.json",
                "sha256": _sha(f"{domain}-source-score-attestation"),
                "capability_schema": "synthetic-score-capability.v2",
            },
            "run_complete": run_complete,
        }
        for domain in ("A", "B")
    ]
    base = SimpleNamespace(
        domains=("A", "B"),
        centers=(
            tuple(float(value) for value in np.zeros(87, dtype=np.float32)),
            tuple(float(value) for value in np.ones(87, dtype=np.float32)),
        ),
        scale=tuple(float(value) for value in np.ones(87, dtype=np.float32)),
    )
    return SimpleNamespace(
        detector_identity=detector,
        attestation={
            "attestation_identity_sha256": _sha("source-v3-identity"),
            "source_score_bundles": score_rows,
        },
        attestation_sha256=_sha("source-v3-attestation"),
        capability_schema="synthetic-source-reference-capability.v3",
        npz_sha256=_sha("source-reference-npz"),
        audit_sha256=_sha("source-reference-audit"),
        source_reference_v2=SimpleNamespace(source_reference_bundle=base),
    )


def _source_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, tuple[Any, Any]]:
    detector = _detector_identity()
    detector_sha = atomic.canonical_json_sha256(detector)
    thresholds = np.asarray([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float64)
    curve_a = atomic.build_exact_source_domain_curve_v2(
        source_domain="A",
        detector_identity_sha256=detector_sha,
        thresholds=thresholds,
        false_positive_pixels=np.asarray(
            [5_000, 1_000, 100, 10, 0], dtype=np.int64
        ),
        matched_objects=np.asarray([10, 9, 8, 7, 0], dtype=np.int64),
        total_native_pixels=10_000_000,
        ground_truth_objects=10,
    )
    curve_b = atomic.build_exact_source_domain_curve_v2(
        source_domain="B",
        detector_identity_sha256=detector_sha,
        thresholds=thresholds,
        false_positive_pixels=np.asarray(
            [6_000, 1_500, 150, 15, 0], dtype=np.int64
        ),
        matched_objects=np.asarray([10, 9, 8, 7, 0], dtype=np.int64),
        total_native_pixels=10_000_000,
        ground_truth_objects=10,
    )
    source_reference = _fake_source_reference_v3()
    monkeypatch.setattr(
        atomic,
        "assert_verified_stage2_rc5_source_reference_v3",
        lambda value: value,
    )
    monkeypatch.setattr(
        atomic,
        "replay_verified_stage2_rc5_source_reference_v3",
        lambda value: value,
    )
    reference = atomic.build_exact_source_threshold_reference_v3(
        domain_curves={"B": curve_b, "A": curve_a},
        source_reference=source_reference,
    )
    return reference, (curve_a, curve_b)


def _learned_seal(
    method: str,
    *,
    producer_binding: dict[str, Any],
    context_payload_sha: str,
    context_full_identity: str,
    feature_sha: str,
    anchor_identity: str,
    anchor_payload_sha: str,
) -> Any:
    coordinates = np.asarray(
        [0.25, 2.0, UPPER_ENDPOINT_COORDINATE], dtype=np.float64
    )
    probabilities = decode_coordinate_numpy(coordinates)
    kinds = endpoint_kinds_numpy(coordinates)
    decision = {
        "method": method,
        "rows": [
            {
                "decoded_threshold_hex": float(probabilities[index]).hex(),
                "canonical_coordinate_hex": float(coordinates[index]).hex(),
                "threshold_kind": kinds[index],
            }
            for index in range(3)
        ],
    }
    transcript = {
        "schema_version": atomic.INFERENCE_TRANSCRIPT_SCHEMA,
        "producer_bundle_binding": producer_binding,
        "context_binding": {
            "context_payload_sha256": context_payload_sha,
            "context_full_identity_sha256": context_full_identity,
            "context_feature_vector_sha256": feature_sha,
        },
        "anchor_binding": {
            "anchor_identity_sha256": anchor_identity,
            "anchor_payload_sha256": anchor_payload_sha,
        },
        "checkpoint_binding": {
            "checkpoint_bytes_sha256": _sha(f"{method}-checkpoint"),
            "training_contract_sha256": _sha(f"{method}-training-contract"),
        },
        "decision": decision,
    }
    return SimpleNamespace(
        method=method,
        transcript=transcript,
        decision=decision,
        producer_identity_sha256=producer_binding[
            "producer_identity_sha256"
        ],
        producer_bundle_identity_sha256=producer_binding[
            "bundle_identity_sha256"
        ],
        producer_manifest_sha256=producer_binding[
            "producer_manifest_sha256"
        ],
        producer_commit_sha256=producer_binding["commit_sha256"],
        transcript_bytes_sha256=_sha(f"{method}-transcript-bytes"),
        transcript_identity_sha256=_sha(f"{method}-transcript-identity"),
        decision_identity_sha256=_sha(f"{method}-decision-identity"),
    )


def _capabilities(monkeypatch: pytest.MonkeyPatch, query_size: int) -> dict[str, Any]:
    reference, curves = _source_reference(monkeypatch)
    detector = _detector_identity()
    context_full_identity = _sha(f"context-full-{query_size}")
    query_full_identity = _sha(f"query-full-{query_size}")
    window_identity = _sha(f"window-identity-{query_size}")
    context_package_id = _sha(f"context-package-{query_size}")
    feature_sha = _sha(f"feature-vector-{query_size}")
    features = np.zeros(93, dtype=np.float64)
    features[:87] = 0.1
    query_records = [
        {"ordinal": index, "identity_sha256": _sha(f"query-{query_size}-{index}")}
        for index in range(query_size)
    ]
    context_payload = {
        "context_package_id": context_package_id,
        "context_full_identity_sha256": context_full_identity,
        "query_full_identity_sha256": query_full_identity,
        "window_identity_sha256": window_identity,
        "query_identity_records": query_records,
        "context_statistics": {"vector_sha256": feature_sha},
    }
    context_payload_sha = atomic.canonical_json_sha256(context_payload)
    context = SimpleNamespace(
        payload=context_payload,
        payload_sha256=context_payload_sha,
    )

    maps = [
        np.asarray(
            [[0.01 + index * 0.001, 0.20], [0.50, 0.90 - index * 0.001]],
            dtype=np.float64,
        )
        for index in range(14)
    ]
    anchor_payload = build_context_tail_anchor(
        context_probability_maps=maps,
        context_identity_sha256=context_full_identity,
    )
    anchor = verify_context_tail_anchor(
        anchor_payload,
        context_probability_maps=maps,
        expected_context_identity_sha256=context_full_identity,
    )
    anchor_payload_sha = atomic.canonical_json_sha256(anchor.payload)
    anchor_identity = anchor.payload["anchor_identity_sha256"]

    window_id = f"dynamic-window-q{query_size}"
    window_manifest_sha = _sha(f"window-manifest-{query_size}")
    window_schema = "rc-irstd.synthetic-variable-query-window.v1"
    window = SimpleNamespace(
        windows=(
            {
                "window_id": window_id,
                "query_size": query_size,
                "query_records": query_records,
            },
        ),
        manifest_sha256=window_manifest_sha,
        payload={"schema_version": window_schema},
    )
    score = SimpleNamespace(
        manifest_sha256=_sha(f"score-manifest-{query_size}"),
        records_content_sha256=_sha(f"score-records-{query_size}"),
        role="source_full_fit",
    )
    producer_source_reference = reference.source_reference_v3
    source_rows = producer_source_reference.attestation["source_score_bundles"]
    shared_run_complete = source_rows[0]["run_complete"]
    query_score_bundle = SimpleNamespace(
        attestation_sha256=_sha(f"query-score-attestation-{query_size}"),
        run_complete=SimpleNamespace(sha256=shared_run_complete["sha256"]),
        attestation={
            "run_complete": {
                "path": shared_run_complete["path"],
                "identity": {
                    "identity_sha256": shared_run_complete[
                        "identity_sha256"
                    ]
                },
            },
            "restricted_checkpoint": {
                "sha256": detector["checkpoint_sha256"]
            },
        },
    )
    context_sha = context_payload_sha
    manifest: dict[str, Any] = {
        "schema_version": "synthetic-rc5-producer-manifest.v1",
        "outer_fold_id": detector["outer_fold_id"],
        "outer_target": detector["outer_target"],
        "source_domain": "A",
        "base_seed": detector["base_seed"],
        "derived_seed": detector["derived_seed"],
        "window_index": 0,
        "window_id": window_id,
        "query_size": query_size,
        "detector_identity": detector,
        "inputs": {
            "variable_query_window": {
                "sha256": window.manifest_sha256,
                "window_id": window_id,
                "schema_version": window_schema,
            },
            "score_manifest_metadata": {
                "sha256": score.manifest_sha256,
                "records_content_sha256": score.records_content_sha256,
                "role": score.role,
                "member_content_verified": False,
            },
            "score_bundle": {
                "sha256": query_score_bundle.attestation_sha256,
                "run_complete_sha256": shared_run_complete["sha256"],
                "run_complete_identity_sha256": shared_run_complete[
                    "identity_sha256"
                ],
                "restricted_checkpoint_sha256": detector[
                    "checkpoint_sha256"
                ],
                "current_state_replayed": True,
            },
            "source_reference": {
                "attestation": {
                    "sha256": producer_source_reference.attestation_sha256,
                    "capability_schema": (
                        producer_source_reference.capability_schema
                    ),
                },
                "base_reference": {
                    "npz_sha256": producer_source_reference.npz_sha256,
                    "audit_sha256": producer_source_reference.audit_sha256,
                },
                "source_score_attestations": [
                    {
                        "source_domain": row["source_domain"],
                        "path": row["score_attestation"]["path"],
                        "sha256": row["score_attestation"]["sha256"],
                        "capability_schema": row["score_attestation"][
                            "capability_schema"
                        ],
                    }
                    for row in source_rows
                ],
                "shared_run_complete": shared_run_complete,
                "current_state_replayed": True,
                "mixed_consumer_schemas_allowed": False,
            },
        },
        "outputs": {
            "context": {
                "sha256": context_sha,
                "context_package_id": context_package_id,
                "context_full_identity_sha256": context_full_identity,
                "query_full_identity_sha256": query_full_identity,
                "window_identity_sha256": window_identity,
                "feature_vector_sha256": feature_sha,
            },
            "anchor": {
                "sha256": anchor_payload_sha,
                "anchor_identity_sha256": anchor_identity,
                "context_identity_sha256": context_full_identity,
                "context_probability_content_sha256": anchor.payload[
                    "context_probability_content_sha256"
                ],
            },
        },
        "access_audit": {
            "query_score_member_open_count": 0,
            "query_image_member_open_count": 0,
            "context_labels_accessed": False,
            "query_labels_accessed": False,
            "observed_results_accessed": False,
        },
    }
    manifest["producer_identity_sha256"] = _self_hash(
        manifest, "producer_identity_sha256"
    )
    manifest_sha = atomic.canonical_json_sha256(manifest)
    bundle_identity = _sha(f"producer-bundle-{query_size}")
    commit = {
        "schema_version": "synthetic-rc5-producer-commit.v1",
        "producer_identity_sha256": manifest["producer_identity_sha256"],
        "bundle_identity_sha256": bundle_identity,
    }
    bundle_capability_schema = "synthetic-verified-rc5-context-bundle.v1"
    bundle = SimpleNamespace(
        producer_manifest=manifest,
        commit=commit,
        context=context,
        anchor=anchor,
        variable_query_window=window,
        score_bundle=query_score_bundle,
        score_manifest_metadata=score,
        source_reference=producer_source_reference,
        producer_manifest_sha256=manifest_sha,
        commit_sha256=atomic.canonical_json_sha256(commit),
        bundle_identity_sha256=bundle_identity,
        capability_schema=bundle_capability_schema,
        context_sha256=context_sha,
        anchor_sha256=anchor_payload_sha,
    )
    producer_binding = {
        "capability_schema": bundle_capability_schema,
        "producer_manifest_schema": manifest["schema_version"],
        "commit_schema": commit["schema_version"],
        "producer_identity_sha256": manifest["producer_identity_sha256"],
        "bundle_identity_sha256": bundle_identity,
        "producer_manifest_sha256": manifest_sha,
        "commit_sha256": bundle.commit_sha256,
    }
    seals = {
        method: _learned_seal(
            method,
            producer_binding=producer_binding,
            context_payload_sha=context_payload_sha,
            context_full_identity=context_full_identity,
            feature_sha=feature_sha,
            anchor_identity=anchor_identity,
            anchor_payload_sha=anchor_payload_sha,
        )
        for method in ("T6", "T7", "T8")
    }

    monkeypatch.setattr(
        atomic, "assert_verified_stage2_rc5_context_bundle", lambda value: value
    )
    monkeypatch.setattr(
        atomic, "replay_verified_stage2_rc5_context_bundle", lambda value: value
    )
    monkeypatch.setattr(
        atomic, "assert_verified_stage2_rc5_inference_seal", lambda value: value
    )

    def material(candidate: Any) -> Any:
        assert candidate is context
        return SimpleNamespace(
            feature_values=tuple(float(value) for value in features),
            feature_vector_sha256=feature_sha,
        )

    monkeypatch.setattr(atomic, "context_inference_material_v2", material)
    return {
        "bundle": bundle,
        "reference": reference,
        "curves": curves,
        "seals": seals,
        "context_full_identity": context_full_identity,
        "anchor_identity": anchor_identity,
    }


def _publish(
    root: Path,
    capabilities: dict[str, Any],
    *,
    evt_seal: Any | None = None,
) -> Any:
    output = root / "atomic"
    output.mkdir(parents=True)
    return atomic.publish_stage2_rc5_atomic_decision_set(
        output,
        producer_bundle=capabilities["bundle"],
        source_threshold_reference=capabilities["reference"],
        inference_seals=capabilities["seals"],
        evt_seal=evt_seal,
        repository_root=root,
    )


def _verify(root: Path, verified: Any, capabilities: dict[str, Any], **kwargs: Any) -> Any:
    return atomic.verify_stage2_rc5_atomic_decision_set(
        decision_set_path=kwargs.get("decision_set_path", verified.decision_set_path),
        commit_path=kwargs.get("commit_path", verified.commit_path),
        expected_commit_sha256=kwargs.get(
            "expected_commit_sha256", verified.commit_sha256
        ),
        producer_bundle=capabilities["bundle"],
        source_threshold_reference=capabilities["reference"],
        inference_seals=capabilities["seals"],
        evt_seal=kwargs.get("evt_seal"),
        repository_root=root,
    )


def test_exact_source_reference_uses_integer_count_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, curves = _source_reference(monkeypatch)
    payload = reference.payload
    assert reference.source_domains == ("A", "B")
    assert payload["budget_count_rule"] == atomic.EXACT_BUDGET_COUNT_RULE
    assert payload["guardrails"]["float_budget_count_logic_used"] is False

    thresholds = np.asarray(
        sorted(set(curves[0].thresholds.tolist() + curves[1].thresholds.tolist())),
        dtype=np.float64,
    )
    pooled_fp = []
    pooled_tp = []
    for threshold in thresholds:
        fp = 0
        tp = 0
        for curve in curves:
            row = int(np.flatnonzero(curve.thresholds <= threshold)[-1])
            fp += int(curve.false_positive_pixels[row])
            tp += int(curve.matched_objects[row])
        pooled_fp.append(fp)
        pooled_tp.append(tp)
    brute = select_exact_oracle_v2(
        thresholds=thresholds,
        false_positive_pixels=np.asarray(pooled_fp, dtype=np.int64),
        matched_objects=np.asarray(pooled_tp, dtype=np.int64),
        total_native_pixels=sum(curve.total_native_pixels for curve in curves),
        ground_truth_objects=sum(curve.ground_truth_objects for curve in curves),
    )
    rows = payload["pooled_safe_rows"]
    assert tuple(
        float.fromhex(row["threshold_probability_hex"]) for row in rows
    ) == tuple(float(value) for value in brute.thresholds)
    assert tuple(row["allowed_false_positive_pixels"] for row in rows) == (
        2_000,
        200,
        20,
    )
    assert tuple(row["observed_false_positive_pixels"] for row in rows) == tuple(
        int(value) for value in brute.false_positive_pixels
    )
    assert reference.source_centers.flags.writeable is False
    assert reference.source_scale.flags.writeable is False


@pytest.mark.parametrize("query_size", [28, 43])
def test_atomic_t0_t8_round_trip_is_dynamic_q_and_t9_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query_size: int,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    capabilities = _capabilities(monkeypatch, query_size)
    verified = _publish(root, capabilities)
    replay = _verify(root, verified, capabilities)

    assert tuple(replay.decision_by_method) == atomic.METHOD_IDS
    assert replay.payload["method_ids"] == atomic.METHOD_IDS
    assert replay.payload["shared_prelabel_identity"]["query_size"] == query_size
    assert replay.payload["t9_included"] is False
    assert b'"T9"' not in verified.decision_set_path.read_bytes()
    assert replay.decision_by_method["T3"]["authority"][
        "nearest_source_domain"
    ] == "A"
    missing = replay.decision_by_method["T5"]
    assert missing["outcome"] == "sealed_missing"
    assert missing["rows"] == ()
    assert missing["fallback"] is False
    for method in atomic.METHOD_IDS:
        decision = replay.decision_by_method[method]
        if method == "T5":
            continue
        assert len(decision["rows"]) == 3
        for row in decision["rows"]:
            assert {
                "threshold_probability_hex",
                "threshold_coordinate_hex",
                "threshold_kind",
            } <= set(row)
    for method in ("T6", "T7", "T8"):
        assert {
            row["probability_coordinate_relation"]
            for row in replay.decision_by_method[method]["rows"]
        } == {"decode_coordinate"}
        assert replay.decision_by_method[method]["authority"][
            "producer_bundle_binding"
        ]["bundle_identity_sha256"] == capabilities["bundle"].bundle_identity_sha256
    assert verified.commit_path.stat().st_mtime_ns >= verified.decision_set_path.stat().st_mtime_ns


def test_t5_is_complete_only_with_matching_verified_evt_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    capabilities = _capabilities(monkeypatch, 31)
    bundle = capabilities["bundle"]
    evt = atomic.build_stage2_rc5_evt_seal_complete(
        producer_identity_sha256=bundle.producer_manifest[
            "producer_identity_sha256"
        ],
        context_full_identity_sha256=capabilities["context_full_identity"],
        anchor_identity_sha256=capabilities["anchor_identity"],
        thresholds=np.asarray([0.60, 0.75, 1.0], dtype=np.float64),
        fit_identity_sha256=_sha("evt-fit"),
    )
    verified = _publish(root, capabilities, evt_seal=evt)
    replay = _verify(root, verified, capabilities, evt_seal=evt)
    t5 = replay.decision_by_method["T5"]
    assert t5["outcome"] == "complete"
    assert len(t5["rows"]) == 3
    assert t5["authority"]["authority_kind"] == "VerifiedStage2RC5EVTSeal"

    wrong = atomic.build_stage2_rc5_evt_seal_complete(
        producer_identity_sha256=_sha("wrong-producer"),
        context_full_identity_sha256=capabilities["context_full_identity"],
        anchor_identity_sha256=capabilities["anchor_identity"],
        thresholds=np.asarray([0.60, 0.75, 1.0], dtype=np.float64),
        fit_identity_sha256=_sha("evt-fit"),
    )
    with pytest.raises(atomic.Stage2RC5AtomicDecisionSetError, match="does not share"):
        atomic.build_stage2_rc5_atomic_decision_set_payload(
            producer_bundle=bundle,
            source_threshold_reference=capabilities["reference"],
            inference_seals=capabilities["seals"],
            evt_seal=wrong,
        )


def test_learned_seals_are_strictly_bound_to_the_same_producer_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    capabilities = _capabilities(monkeypatch, 33)
    original = capabilities["seals"]["T6"]
    transcript = copy.deepcopy(original.transcript)
    transcript["producer_bundle_binding"]["commit_sha256"] = _sha(
        "different-producer-commit"
    )
    forged = SimpleNamespace(
        **{
            **vars(original),
            "transcript": transcript,
            "producer_commit_sha256": transcript["producer_bundle_binding"][
                "commit_sha256"
            ],
        }
    )
    seals = dict(capabilities["seals"])
    seals["T6"] = forged
    with pytest.raises(
        atomic.Stage2RC5AtomicDecisionSetError,
        match="different producer bundle",
    ):
        atomic.build_stage2_rc5_atomic_decision_set_payload(
            producer_bundle=capabilities["bundle"],
            source_threshold_reference=capabilities["reference"],
            inference_seals=seals,
        )


def test_label_resolver_is_never_called_for_tampered_or_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    capabilities = _capabilities(monkeypatch, 35)
    verified = _publish(root, capabilities)
    valid_calls: list[str] = []
    failure_calls: list[str] = []

    def resolver(decision_set: Any, marker: str) -> str:
        valid_calls.append(decision_set.decision_set_identity_sha256)
        return marker

    def forbidden_resolver(decision_set: Any, marker: str = "") -> str:
        failure_calls.append(decision_set.decision_set_identity_sha256)
        return marker

    result = atomic.guarded_invoke_stage2_rc5_label_resolver(
        decision_set_path=verified.decision_set_path,
        commit_path=verified.commit_path,
        expected_commit_sha256=verified.commit_sha256,
        producer_bundle=capabilities["bundle"],
        source_threshold_reference=capabilities["reference"],
        inference_seals=capabilities["seals"],
        label_resolver=resolver,
        repository_root=root,
        resolver_args=("called",),
    )
    assert result == "called"
    assert len(valid_calls) == 1

    verified.decision_set_path.write_bytes(
        verified.decision_set_path.read_bytes() + b"tampered"
    )
    with pytest.raises(atomic.Stage2RC5AtomicDecisionSetError):
        atomic.guarded_invoke_stage2_rc5_label_resolver(
            decision_set_path=verified.decision_set_path,
            commit_path=verified.commit_path,
            expected_commit_sha256=verified.commit_sha256,
            producer_bundle=capabilities["bundle"],
            source_threshold_reference=capabilities["reference"],
            inference_seals=capabilities["seals"],
            label_resolver=forbidden_resolver,
            repository_root=root,
            resolver_args=("forbidden",),
        )
    assert failure_calls == []

    partial = root / "partial"
    partial.mkdir()
    with pytest.raises(atomic.Stage2RC5AtomicDecisionSetError, match="does not exist"):
        atomic.guarded_invoke_stage2_rc5_label_resolver(
            decision_set_path=partial / atomic.DECISION_SET_FILENAME,
            commit_path=partial / atomic.DECISION_SET_COMMIT_FILENAME,
            expected_commit_sha256=_sha("missing-commit"),
            producer_bundle=capabilities["bundle"],
            source_threshold_reference=capabilities["reference"],
            inference_seals=capabilities["seals"],
            label_resolver=forbidden_resolver,
            repository_root=root,
        )
    assert failure_calls == []


def test_symlink_and_commit_without_set_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    capabilities = _capabilities(monkeypatch, 37)
    verified = _publish(root, capabilities)
    target = verified.decision_set_path.with_name("decision.target.json")
    verified.decision_set_path.rename(target)
    verified.decision_set_path.symlink_to(target.name)
    with pytest.raises(atomic.Stage2RC5AtomicDecisionSetError, match="symlink"):
        _verify(root, verified, capabilities)

    partial = root / "commit-only"
    partial.mkdir()
    commit = partial / atomic.DECISION_SET_COMMIT_FILENAME
    commit.write_bytes(verified.commit_path.read_bytes())
    commit_sha = hashlib.sha256(commit.read_bytes()).hexdigest()
    with pytest.raises(atomic.Stage2RC5AtomicDecisionSetError, match="does not exist"):
        atomic.verify_stage2_rc5_atomic_decision_set(
            decision_set_path=partial / atomic.DECISION_SET_FILENAME,
            commit_path=commit,
            expected_commit_sha256=commit_sha,
            producer_bundle=capabilities["bundle"],
            source_threshold_reference=capabilities["reference"],
            inference_seals=capabilities["seals"],
            repository_root=root,
        )


def test_rehashed_semantic_mutation_still_fails_capability_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    capabilities = _capabilities(monkeypatch, 39)
    verified = _publish(root, capabilities)
    payload = json.loads(verified.decision_set_path.read_bytes())
    row = payload["decisions"][0]["rows"][0]
    row["threshold_probability_hex"] = (0.25).hex()
    row["threshold_coordinate_hex"] = (0.25).hex()
    row["threshold_kind"] = "interior"
    payload["decisions"][0]["decision_identity_sha256"] = _self_hash(
        payload["decisions"][0], "decision_identity_sha256"
    )
    payload["decision_set_identity_sha256"] = _self_hash(
        payload, "decision_set_identity_sha256"
    )
    set_bytes = atomic.canonical_json_bytes(payload)
    set_sha = hashlib.sha256(set_bytes).hexdigest()
    commit_payload = atomic._commit_payload(
        decision_set_sha256=set_sha, decision_set=payload
    )
    commit_bytes = atomic.canonical_json_bytes(commit_payload)
    commit_sha = hashlib.sha256(commit_bytes).hexdigest()
    verified.decision_set_path.write_bytes(set_bytes)
    verified.commit_path.write_bytes(commit_bytes)

    with pytest.raises(
        atomic.Stage2RC5AtomicDecisionSetError,
        match="full verifier-capability replay",
    ):
        _verify(
            root,
            verified,
            capabilities,
            expected_commit_sha256=commit_sha,
        )


def test_t9_has_separate_postlabel_schema_and_cannot_be_prelabel_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    capabilities = _capabilities(monkeypatch, 41)
    verified = _publish(root, capabilities)
    diagnostic = atomic.build_stage2_rc5_t9_postlabel_diagnostic(
        prelabel_decision_set=verified,
        thresholds=np.asarray([0.0, 0.5, 0.8, 1.0], dtype=np.float64),
        false_positive_pixels=np.asarray([100, 10, 0, 0], dtype=np.int64),
        matched_objects=np.asarray([1, 2, 1, 0], dtype=np.int64),
        total_native_pixels=1_000_000,
        ground_truth_objects=2,
    )
    assert diagnostic["schema_version"] == atomic.T9_DIAGNOSTIC_SCHEMA
    assert diagnostic["method_id"] == "T9"
    assert diagnostic["prelabel_eligible"] is False
    assert diagnostic["target_labels_accessed"] is True
    assert "T9" not in verified.decision_by_method


def test_public_capabilities_reject_direct_or_forged_construction() -> None:
    for capability in (
        atomic.VerifiedExactSourceDomainCurveV2,
        atomic.VerifiedExactSourceThresholdReferenceV3,
        atomic.VerifiedStage2RC5EVTSeal,
        atomic.VerifiedStage2RC5AtomicDecisionSet,
    ):
        with pytest.raises(TypeError, match="verifier-issued"):
            capability()
    with pytest.raises(TypeError, match="verifier-issued"):
        atomic.assert_verified_exact_source_domain_curve_v2(SimpleNamespace())
    with pytest.raises(TypeError, match="verifier-issued"):
        atomic.assert_verified_exact_source_threshold_reference_v3(
            SimpleNamespace()
        )
    with pytest.raises(TypeError, match="verifier-issued"):
        atomic.assert_verified_stage2_rc5_evt_seal(SimpleNamespace())
    with pytest.raises(TypeError, match="verifier-issued"):
        atomic.assert_verified_stage2_rc5_atomic_decision_set(SimpleNamespace())

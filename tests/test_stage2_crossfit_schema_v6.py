from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import data_ext.stage2_variable_query_window as variable_window
from rc.stage2_variable_query_geometry import (
    build_stage2_variable_query_geometry,
)
import rc.stage2_crossfit_schema_v6 as schema


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _records(domain: str, count: int, tag: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(count):
        token = f"{domain}:{tag}:{index}"
        result.append(
            {
                "canonical_id": f"canonical:{token}",
                "image_id": f"image:{token}",
                "original_image_sha256": _sha(f"image:{token}"),
                "exclusion_group_id": f"exclusion:{token}",
                "near_duplicate_cluster_id_or_unique_sentinel": (
                    f"unique:{token}"
                ),
                "source_role_record_index": index,
            }
        )
    return result


def _context_payloads(
    *,
    outer_fold: str,
    domain: str,
    expected_role: str,
    count: int,
    tag: str,
    records: list[dict[str, Any]] | None = None,
    base_seed: int = 42,
) -> list[dict[str, Any]]:
    geometry = build_stage2_variable_query_geometry(count)
    ordered = _records(domain, count, tag) if records is None else records
    payloads: list[dict[str, Any]] = []
    for window in geometry["windows"]:
        index = window["window_index"]
        payloads.append(
            schema.build_context_payload_v2(
                expected_role=expected_role,
                outer_fold_id=outer_fold,
                outer_target=schema.OUTER_TARGETS[outer_fold],
                source_domain=domain,
                base_seed=base_seed,
                derived_seed=90_000 + index,
                geometry=geometry,
                window_index=index,
                window_id=f"{tag}:{domain}:window-{index}",
                context_records=ordered[
                    window["context_start"] : window["context_stop"]
                ],
                query_identity_records=ordered[
                    window["query_start"] : window["query_stop"]
                ],
                context_feature_values=[float(index)] * schema.FEATURE_DIM,
            )
        )
    return payloads


def _episode(
    episode_index: int,
    context_payload: dict[str, Any],
) -> schema.VerifiedStage2EpisodeV6:
    context = schema.verify_context_payload_v2(context_payload)
    anchor = schema.make_anchor_binding_v6(
        context=context,
        path=f"synthetic/anchors/{context.payload['window_id']}.json",
        sha256=_sha(f"anchor-file:{context.payload_sha256}"),
        anchor_identity_sha256=_sha(
            f"anchor-identity:{context.payload_sha256}"
        ),
        anchor_payload_sha256=_sha(
            f"anchor-payload:{context.payload_sha256}"
        ),
        context_probability_content_sha256=_sha(
            f"context-probability:{context.payload_sha256}"
        ),
        total_context_pixels=14 * 100_000,
    )
    oracle = schema.make_oracle_curve_binding_v2(
        context=context,
        curve_path=f"synthetic/curves/{context.payload['window_id']}.csv",
        curve_sha256=_sha(f"curve:{context.payload_sha256}"),
        manifest_path=(
            f"synthetic/curves/{context.payload['window_id']}.json"
        ),
        manifest_sha256=_sha(f"curve-manifest:{context.payload_sha256}"),
        curve_rows_sha256=_sha(f"curve-rows:{context.payload_sha256}"),
        oracle_rows_sha256=_sha(f"oracle-rows:{context.payload_sha256}"),
        total_native_pixels=(
            len(context.payload["query_identity_records"]) * 100_000
        ),
    )
    decision = None
    if context.payload["episode_role"] == (
        schema.OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT
    ):
        decision = schema.make_prelabel_decision_binding_v1(
            context=context,
            path=(
                f"synthetic/decisions/{context.payload['window_id']}.json"
            ),
            sha256=_sha(f"decision:{context.payload_sha256}"),
            decision_set_content_sha256=_sha(
                f"decision-content:{context.payload_sha256}"
            ),
        )
    payload = schema.build_episode_payload_v6(
        episode_index=episode_index,
        context=context,
        anchor_binding=anchor,
        oracle_curve_binding=oracle,
        prelabel_decision_binding=decision,
    )
    return schema.verify_episode_payload_v6(payload)


def _source_collection(
    *,
    role: str = schema.OOF_HOLDOUT_STAGE2_FIT,
    first_count: int = 85,
    second_count: int = 127,
    first_records: list[dict[str, Any]] | None = None,
    second_records: list[dict[str, Any]] | None = None,
) -> tuple[schema.VerifiedStage2EpisodeV6, ...]:
    outer = "outer_leave_nuaa_sirst"
    contexts = _context_payloads(
        outer_fold=outer,
        domain="NUDT-SIRST",
        expected_role=role,
        count=first_count,
        tag=f"{role}:nudt",
        records=first_records,
    )
    contexts += _context_payloads(
        outer_fold=outer,
        domain="IRSTD-1K",
        expected_role=role,
        count=second_count,
        tag=f"{role}:irstd",
        records=second_records,
    )
    return tuple(_episode(index, context) for index, context in enumerate(contexts))


def _outer_collection(
    count: int = 159,
) -> tuple[schema.VerifiedStage2EpisodeV6, ...]:
    contexts = _context_payloads(
        outer_fold="outer_leave_nuaa_sirst",
        domain="NUAA-SIRST",
        expected_role=schema.OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
        count=count,
        tag="outer:nuaa",
    )
    return tuple(_episode(index, context) for index, context in enumerate(contexts))


def _publish_bundle(
    root: Path,
    episodes: tuple[schema.VerifiedStage2EpisodeV6, ...],
) -> dict[str, Any]:
    collection = root / "episodes.jsonl"
    manifest = schema.collection_manifest_path_v6(collection)
    commit = schema.collection_commit_path_v2(collection)
    collection_data = schema.collection_jsonl_bytes_v6(episodes)
    collection.write_bytes(collection_data)
    collection_sha = schema.sha256_bytes(collection_data)
    manifest_payload = schema.build_collection_manifest_payload_v6(
        episodes,
        collection_path=collection,
        collection_sha256=collection_sha,
        repository_root=root,
    )
    manifest_data = schema.canonical_json_document_bytes(manifest_payload)
    manifest.write_bytes(manifest_data)
    manifest_sha = schema.sha256_bytes(manifest_data)
    commit_payload = schema.build_collection_commit_payload_v2(
        collection_path=collection,
        collection_sha256=collection_sha,
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        repository_root=root,
    )
    commit_data = schema.canonical_json_document_bytes(commit_payload)
    commit.write_bytes(commit_data)
    commit_sha = schema.sha256_bytes(commit_data)
    return {
        "collection": collection,
        "manifest": manifest,
        "commit": commit,
        "collection_sha": collection_sha,
        "manifest_sha": manifest_sha,
        "commit_sha": commit_sha,
        "manifest_payload": manifest_payload,
        "commit_payload": commit_payload,
    }


def _verified_variable_window(
    root: Path,
) -> variable_window.VerifiedStage2VariableQueryWindow:
    contracts = root / "contracts"
    contracts.mkdir()

    def binding(name: str) -> dict[str, str]:
        relative = f"contracts/{name}.json"
        data = json.dumps({"name": name}, sort_keys=True).encode("utf-8")
        (root / relative).write_bytes(data)
        return {"path": relative, "sha256": schema.sha256_bytes(data)}

    role_binding = binding("role")
    bound_inputs = {
        name: binding(name)
        for name in sorted(variable_window.BOUND_INPUT_NAMES)
    }
    records: list[dict[str, Any]] = []
    for row in _records("NUDT-SIRST", 85, "verified-window"):
        records.append(
            {
                **row,
                "canonical_id": f"NUDT-SIRST::{row['image_id']}",
                "original_image_path": (
                    f"datasets/NUDT-SIRST/images/{row['image_id']}.png"
                ),
                "source_role": "detector_diagnostic",
                "outer_fold_id": "outer_leave_nuaa_sirst",
                "episode_role": schema.SOURCE_DIAGNOSTIC_VALIDATION,
                "oof_fold_index": None,
            }
        )
    payload = variable_window.build_stage2_variable_query_window_payload(
        ordered_role_records=records,
        outer_fold_id="outer_leave_nuaa_sirst",
        outer_target_domain="NUAA-SIRST",
        domain="NUDT-SIRST",
        source_role="detector_diagnostic",
        episode_role=schema.SOURCE_DIAGNOSTIC_VALIDATION,
        oof_fold_index=None,
        role_binding=role_binding,
        bound_inputs=bound_inputs,
    )
    path = contracts / "variable-q-windows.json"
    data = variable_window.canonical_json_bytes(payload)
    path.write_bytes(data)
    return variable_window.verify_stage2_variable_query_window(
        path,
        schema.sha256_bytes(data),
        repository_root=root,
    )


@pytest.mark.parametrize(
    ("count", "expected_queries"),
    [
        (43, [29]),
        (85, [29, 28]),
        (127, [29, 28, 28]),
        (159, [39, 39, 39]),
        (254, [29, 29, 28, 28, 28, 28]),
        (255, [29, 29, 29, 28, 28, 28]),
        (319, [32, 32, 32, 32, 31, 31, 31]),
    ],
)
def test_context_v2_replays_boundary_geometry_and_dynamic_q(
    count: int, expected_queries: list[int]
) -> None:
    payloads = _context_payloads(
        outer_fold="outer_leave_nuaa_sirst",
        domain="NUAA-SIRST",
        expected_role=schema.OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
        count=count,
        tag=f"boundary-{count}",
    )
    verified = [schema.verify_context_payload_v2(item) for item in payloads]
    assert len(verified) == count // 42
    assert [
        len(item.payload["query_identity_records"]) for item in verified
    ] == expected_queries
    consumed = [
        row["source_role_record_index"]
        for item in verified
        for row in (
            *item.payload["context_records"],
            *item.payload["query_identity_records"],
        )
    ]
    assert sorted(consumed) == list(range(count))


def test_context_capability_is_recursive_immutable_and_inference_is_query_free(
) -> None:
    payload = _context_payloads(
        outer_fold="outer_leave_nuaa_sirst",
        domain="NUDT-SIRST",
        expected_role=schema.SOURCE_DIAGNOSTIC_VALIDATION,
        count=43,
        tag="inference",
    )[0]
    context = schema.verify_context_payload_v2(payload)
    with pytest.raises(TypeError):
        context.payload["context_records"][0]["canonical_id"] = "forged"
    material = schema.context_inference_material_v2(context)
    assert len(material.feature_values) == 93
    assert material.source_query_consumed is False
    assert not hasattr(material, "query_identity_records")
    assert not hasattr(material, "query_size")
    with pytest.raises(TypeError):
        schema.assert_verified_context_v2(SimpleNamespace(payload=payload))
    with pytest.raises(TypeError, match="verifier-only"):
        schema.VerifiedStage2ContextV2(
            payload=payload,
            canonical_payload=b"{}",
            payload_sha256=_sha("fake"),
            _capability=object(),
        )
    missing_identity = deepcopy(payload)
    del missing_identity["query_identity_records"][0][
        "exclusion_group_id"
    ]
    with pytest.raises(
        schema.Stage2CrossfitSchemaV6Error, match="fields differ"
    ):
        schema.verify_context_payload_v2(missing_identity)


def test_context_adapter_requires_verified_variable_q_capability(
    tmp_path: Path,
) -> None:
    windows = _verified_variable_window(tmp_path)
    context = schema.context_from_verified_variable_query_window_v2(
        windows,
        expected_role=schema.SOURCE_DIAGNOSTIC_VALIDATION,
        base_seed=42,
        derived_seed=91_001,
        window_index=1,
        context_feature_values=[1.0] * schema.FEATURE_DIM,
    )
    assert context.variable_query_window is windows
    assert len(context.payload["context_records"]) == 14
    assert len(context.payload["query_identity_records"]) == 28
    material = schema.context_inference_material_v2(context)
    assert (
        schema.assert_verified_context_inference_material_v2(material)
        is material
    )
    assert not hasattr(material, "variable_query_window")
    with pytest.raises(TypeError, match="variable-Q window capability"):
        schema.context_from_verified_variable_query_window_v2(
            SimpleNamespace(payload=windows.payload),
            expected_role=schema.SOURCE_DIAGNOSTIC_VALIDATION,
            base_seed=42,
            derived_seed=91_001,
            window_index=1,
            context_feature_values=[1.0] * schema.FEATURE_DIM,
        )
    with pytest.raises(
        schema.Stage2CrossfitSchemaV6Error,
        match="episode_role/context role mismatch",
    ):
        schema.context_from_verified_variable_query_window_v2(
            windows,
            expected_role=schema.OOF_HOLDOUT_STAGE2_FIT,
            base_seed=42,
            derived_seed=91_001,
            window_index=1,
            context_feature_values=[1.0] * schema.FEATURE_DIM,
        )


def test_episode_v6_binds_anchor_eatc_exact_rationals_and_prelabel_gate() -> None:
    source_context = _context_payloads(
        outer_fold="outer_leave_nuaa_sirst",
        domain="NUDT-SIRST",
        expected_role=schema.OOF_HOLDOUT_STAGE2_FIT,
        count=43,
        tag="source-episode",
    )[0]
    source = _episode(0, source_context)
    assert source.payload["prelabel_decision_binding"] is None
    assert source.payload["threshold_semantics"] == (
        "prediction = probability > threshold"
    )
    assert source.payload["anchor_binding"][
        "threshold_representation_schema"
    ] == schema.EATC_V2_SCHEMA
    assert source.payload["oracle_curve_binding"][
        "threshold_representation_schema"
    ] == schema.EATC_V2_SCHEMA
    assert source.payload["budget_rationals"] == tuple(
        {key: value for key, value in row.items()}
        for row in schema.BUDGET_RATIONALS
    )

    tampered = deepcopy(schema._plain(source.payload))
    tampered["oracle_curve_binding"]["budget_counts"][1][
        "allowed_false_positive_pixels"
    ] += 1
    with pytest.raises(
        schema.Stage2CrossfitSchemaV6Error, match="exact integer replay"
    ):
        schema.verify_episode_payload_v6(tampered)

    outer_context = _context_payloads(
        outer_fold="outer_leave_nuaa_sirst",
        domain="NUAA-SIRST",
        expected_role=schema.OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
        count=43,
        tag="outer-episode",
    )[0]
    outer = _episode(0, outer_context)
    assert outer.payload["prelabel_decision_binding"] is not None
    assert outer.payload["prelabel_decision_binding"]["derived_seed"] == (
        outer.payload["derived_seed"]
    )
    missing = deepcopy(schema._plain(outer.payload))
    missing["prelabel_decision_binding"] = None
    with pytest.raises(
        schema.Stage2CrossfitSchemaV6Error,
        match="prelabel_decision_binding",
    ):
        schema.verify_episode_payload_v6(missing)


@pytest.mark.parametrize(
    "role",
    [schema.OOF_HOLDOUT_STAGE2_FIT, schema.SOURCE_DIAGNOSTIC_VALIDATION],
)
def test_source_collection_requires_both_sources_and_replays_mixed_q(
    role: str,
) -> None:
    episodes = _source_collection(role=role)
    summaries = schema.verify_episode_collection_completeness_v6(episodes)
    assert len(episodes) == 5
    assert {item["source_domain"] for item in summaries} == {
        "NUDT-SIRST",
        "IRSTD-1K",
    }
    assert [episode.payload["query_size"] for episode in episodes] == [
        29,
        28,
        29,
        28,
        28,
    ]
    assert all(
        episode.payload["episode_weighting"] == "equal_window"
        for episode in episodes
    )


def test_outer_collection_contains_only_target_and_bundle_is_canonical(
    tmp_path: Path,
) -> None:
    episodes = _outer_collection()
    summaries = schema.verify_episode_collection_completeness_v6(episodes)
    assert summaries == (
        {
            "source_domain": "NUAA-SIRST",
            "ordered_record_count": 159,
            "window_count": 3,
            "geometry_sha256": schema.canonical_json_sha256(
                build_stage2_variable_query_geometry(159)
            ),
        },
    )
    published = _publish_bundle(tmp_path, episodes)
    verified = schema.verify_stage2_collection_bundle_v6(
        published["collection"],
        published["collection_sha"],
        published["manifest"],
        published["manifest_sha"],
        published["commit"],
        published["commit_sha"],
        repository_root=tmp_path,
    )
    assert len(verified) == 3
    assert verified.manifest["episode_weighting"] == "equal_window"
    assert verified.manifest["all_records_consumed_once"] is True
    assert verified.commit["publication_order"].endswith("commit_last")
    with pytest.raises(TypeError):
        verified.manifest["episode_count"] = 999
    with pytest.raises(TypeError):
        schema.assert_verified_collection_v6(
            SimpleNamespace(episodes=verified.episodes)
        )


def test_missing_window_duplicate_identity_and_outer_leakage_fail_closed() -> None:
    episodes = _source_collection()
    with pytest.raises(
        schema.Stage2CrossfitSchemaV6Error, match="omits"
    ):
        schema.verify_episode_collection_completeness_v6(episodes[:-1])

    for field in (
        "canonical_id",
        "original_image_sha256",
        "near_duplicate_cluster_id_or_unique_sentinel",
        "exclusion_group_id",
    ):
        first_records = _records("NUDT-SIRST", 85, f"duplicate-{field}")
        second_records = _records("IRSTD-1K", 127, f"other-{field}")
        # Duplicate across two otherwise-valid windows, not within one.
        first_records[43][field] = first_records[0][field]
        duplicate_episodes = _source_collection(
            first_records=first_records,
            second_records=second_records,
        )
        with pytest.raises(
            schema.Stage2CrossfitSchemaV6Error,
            match=f"duplicate identity at {field}",
        ):
            schema.verify_episode_collection_completeness_v6(
                duplicate_episodes
            )

    with pytest.raises(
        schema.Stage2CrossfitSchemaV6Error, match="outer target"
    ):
        _context_payloads(
            outer_fold="outer_leave_nuaa_sirst",
            domain="NUAA-SIRST",
            expected_role=schema.OOF_HOLDOUT_STAGE2_FIT,
            count=43,
            tag="source-leak",
        )
    with pytest.raises(
        schema.Stage2CrossfitSchemaV6Error, match="outer context"
    ):
        _context_payloads(
            outer_fold="outer_leave_nuaa_sirst",
            domain="NUDT-SIRST",
            expected_role=schema.OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
            count=43,
            tag="outer-leak",
        )


def test_collection_rejects_role_and_base_seed_mixing() -> None:
    train = _context_payloads(
        outer_fold="outer_leave_nuaa_sirst",
        domain="NUDT-SIRST",
        expected_role=schema.OOF_HOLDOUT_STAGE2_FIT,
        count=43,
        tag="train-role",
    )[0]
    validation = _context_payloads(
        outer_fold="outer_leave_nuaa_sirst",
        domain="IRSTD-1K",
        expected_role=schema.SOURCE_DIAGNOSTIC_VALIDATION,
        count=43,
        tag="validation-role",
    )[0]
    with pytest.raises(
        schema.Stage2CrossfitSchemaV6Error,
        match="mixes role/fold/target/base seed",
    ):
        schema.verify_episode_collection_completeness_v6(
            (_episode(0, train), _episode(1, validation))
        )

    other_seed = _context_payloads(
        outer_fold="outer_leave_nuaa_sirst",
        domain="IRSTD-1K",
        expected_role=schema.OOF_HOLDOUT_STAGE2_FIT,
        count=43,
        tag="other-base-seed",
        base_seed=123,
    )[0]
    with pytest.raises(
        schema.Stage2CrossfitSchemaV6Error,
        match="mixes role/fold/target/base seed",
    ):
        schema.verify_episode_collection_completeness_v6(
            (_episode(0, train), _episode(1, other_seed))
        )


def test_bundle_rejects_commit_manifest_and_canonical_byte_tampering(
    tmp_path: Path,
) -> None:
    published = _publish_bundle(tmp_path, _outer_collection(43))
    bad_commit = deepcopy(published["commit_payload"])
    bad_commit["collection_manifest"]["sha256"] = _sha("wrong-manifest")
    bad_commit_data = schema.canonical_json_document_bytes(bad_commit)
    published["commit"].write_bytes(bad_commit_data)
    bad_commit_sha = schema.sha256_bytes(bad_commit_data)
    with pytest.raises(
        schema.Stage2CrossfitSchemaV6Error, match="member binding"
    ):
        schema.verify_stage2_collection_bundle_v6(
            published["collection"],
            published["collection_sha"],
            published["manifest"],
            published["manifest_sha"],
            published["commit"],
            bad_commit_sha,
            repository_root=tmp_path,
        )

    published["commit"].write_bytes(
        schema.canonical_json_document_bytes(published["commit_payload"])
    )
    noncanonical_manifest = (
        b" "
        + schema.canonical_json_document_bytes(
            published["manifest_payload"]
        )
    )
    published["manifest"].write_bytes(noncanonical_manifest)
    newer = published["manifest"].stat().st_mtime_ns + 1_000_000
    os.utime(published["commit"], ns=(newer, newer))
    noncanonical_sha = schema.sha256_bytes(noncanonical_manifest)
    with pytest.raises(
        schema.Stage2CrossfitSchemaV6Error, match="canonical JSON"
    ):
        schema.verify_stage2_collection_bundle_v6(
            published["collection"],
            published["collection_sha"],
            published["manifest"],
            noncanonical_sha,
            published["commit"],
            schema.sha256_bytes(published["commit"].read_bytes()),
            repository_root=tmp_path,
        )


def test_commit_last_mtime_is_enforced(tmp_path: Path) -> None:
    published = _publish_bundle(tmp_path, _outer_collection(43))
    old = published["collection"].stat().st_mtime_ns - 1_000_000
    os.utime(published["commit"], ns=(old, old))
    with pytest.raises(
        schema.Stage2CrossfitSchemaV6Error, match="not published last"
    ):
        schema.verify_stage2_collection_bundle_v6(
            published["collection"],
            published["collection_sha"],
            published["manifest"],
            published["manifest_sha"],
            published["commit"],
            published["commit_sha"],
            repository_root=tmp_path,
        )

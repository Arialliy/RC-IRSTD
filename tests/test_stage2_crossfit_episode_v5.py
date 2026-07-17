from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data_ext.stage2_label_attachment import (
    OOF_HOLDOUT_STAGE2_FIT,
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
    SOURCE_DIAGNOSTIC_VALIDATION,
    canonical_json_sha256 as w05_canonical_json_sha256,
    stage2_ordered_query_identity,
)
from evaluation.stage2_threshold_family import make_shared_input_bindings
from rc import build_stage2_crossfit_episodes as builder
from rc import stage2_crossfit_schema as schema
from rc.domain_statistics import FEATURE_NAMES
from rc.schema import StatisticsConfig


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _row(
    prefix: str,
    partition: str,
    ordinal: int,
    source_domain: str,
    outer_fold: str,
    *,
    oof_fold_index: int | None,
) -> dict[str, object]:
    token = f"{prefix}-{partition}-{ordinal}"
    return {
        "ordinal": ordinal,
        "partition": partition,
        "score_record_index": ordinal,
        "canonical_id": f"canonical-{token}",
        "image_id": f"image-{token}",
        "source_domain": source_domain,
        "original_image_path": f"synthetic/images/{token}.png",
        "original_image_sha256": _sha(f"image:{token}"),
        "exclusion_group_id": f"exclusion-{token}",
        "near_duplicate_cluster_id_or_unique_sentinel": f"unique-{token}",
        "source_role_record_index": ordinal + (0 if partition == "context" else 14),
        "source_role": "synthetic_source",
        "outer_fold_id": outer_fold,
        "oof_fold_index": oof_fold_index,
        "score_file": f"synthetic/scores/{token}.npz",
        "score_file_sha256": _sha(f"score:{token}"),
        "original_hw": [4, 5],
        "input_hw": [4, 5],
        "resized_hw": [4, 5],
        "padding_ltrb": [0, 0, 0, 0],
        "resize_mode": "native",
    }


def _w05_query_sha(rows: list[dict[str, object]]) -> str:
    projected = [
        {
            key: row[key]
            for key in (
                "canonical_id",
                "image_id",
                "original_image_sha256",
                "exclusion_group_id",
                "near_duplicate_cluster_id_or_unique_sentinel",
                "source_role_record_index",
            )
        }
        for row in rows
    ]
    return w05_canonical_json_sha256(stage2_ordered_query_identity(projected))


def _episode(
    index: int,
    *,
    collection_role: str,
    episode_role: str,
    outer_fold: str,
    source_domain: str,
    detector_role: str,
    oof_fold_index: int | None,
    feature_value: float = 0.0,
) -> schema.Stage2CrossfitEpisode:
    prefix = f"{collection_role}-{outer_fold}-{index}"
    context = [
        _row(
            prefix,
            "context",
            ordinal,
            source_domain,
            outer_fold,
            oof_fold_index=oof_fold_index,
        )
        for ordinal in range(14)
    ]
    query = [
        _row(
            prefix,
            "query",
            ordinal,
            source_domain,
            outer_fold,
            oof_fold_index=oof_fold_index,
        )
        for ordinal in range(28)
    ]
    values = np.full(93, feature_value, dtype=np.float32)
    payload = {
        "schema_version": schema.EPISODE_SCHEMA,
        "artifact_type": schema.EPISODE_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY_VERIFIED_COMPLETE",
        "development_only": True,
        "official_test_accessed": False,
        "episode_id": _sha(f"episode:{prefix}"),
        "episode_index": index,
        "collection_role": collection_role,
        "episode_role": episode_role,
        "outer_fold_id": outer_fold,
        "outer_target": schema.OUTER_TARGETS[outer_fold],
        "source_domain": source_domain,
        "base_seed": 42,
        "derived_seed": 42001,
        "detector_identity": {
            "detector_role": detector_role,
            "oof_fold_index": oof_fold_index,
            "checkpoint_sha256": _sha(f"checkpoint:{prefix}"),
        },
        "geometry": dict(schema.GEOMETRY),
        "context_package_binding": {
            "path": f"synthetic/{prefix}/context.json",
            "sha256": _sha(f"context:{prefix}"),
            "commit_path": f"synthetic/{prefix}/context.commit.json",
            "commit_sha256": _sha(f"context-commit:{prefix}"),
        },
        "window_binding": {
            "path": f"synthetic/{prefix}/window.json",
            "sha256": _sha(f"window-manifest:{prefix}"),
            "window_id": f"window-{index}",
            "window_identity_sha256": _sha(f"window:{prefix}"),
        },
        "score_manifest_binding": {
            "path": f"synthetic/{prefix}/scores.json",
            "sha256": _sha(f"scores:{prefix}"),
            "records_content_sha256": _sha(f"score-records:{prefix}"),
            "role": "synthetic",
        },
        "score_bindings": {},
        "source_reference_binding": {},
        "statistics_config_binding": {},
        "partition_bindings": {},
        "seed_binding": {},
        "governance_bindings": {},
        "context_records": context,
        "query_records": query,
        "context_full_identity_sha256": schema.full_identity_sha256(context),
        "source_query_full_identity_sha256": schema.full_identity_sha256(query),
        "source_ordered_query_identity_sha256": _w05_query_sha(query),
        "context_statistics": {
            "feature_names": list(FEATURE_NAMES),
            "feature_dim": 93,
            "dtype": "float32",
            "values": values.tolist(),
            "vector_sha256_algorithm": schema.FLOAT32_VECTOR_ALGORITHM,
            "vector_sha256": hashlib.sha256(values.astype("<f4").tobytes()).hexdigest(),
            "metadata": {},
        },
        "decision_seal_binding": (
            {}
            if episode_role == OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT
            else None
        ),
        "label_manifest_binding": {},
        "curve_binding": {},
        "supervision_contract": {},
        "guardrails": {
            "context_labels_loaded": False,
            "query_labels_loaded": True,
            "official_test_accessed": False,
            "reject_supported": False,
            "fallback_used": False,
        },
    }
    return schema.Stage2CrossfitEpisode.from_dict(payload)


@pytest.mark.parametrize(
    ("collection_role", "episode_role", "detector_role", "oof_index"),
    [
        (schema.COLLECTION_TRAIN, schema.STAGE2_OOF_FIT, "detector_oof", 0),
        (
            schema.COLLECTION_VALIDATION,
            SOURCE_DIAGNOSTIC_VALIDATION,
            "detector_full_fit",
            None,
        ),
        (
            schema.COLLECTION_OUTER,
            OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
            "detector_full_fit",
            None,
        ),
    ],
)
@pytest.mark.parametrize("outer_fold", tuple(schema.OUTER_TARGETS))
def test_collection_completeness_exact_frozen_matrix(
    collection_role: str,
    episode_role: str,
    detector_role: str,
    oof_index: int | None,
    outer_fold: str,
) -> None:
    count = schema.EXPECTED_COLLECTION_COUNTS[collection_role][outer_fold]
    target = schema.OUTER_TARGETS[outer_fold]
    sources = [domain for domain in schema.ALL_DOMAINS if domain != target]
    episodes = [
        _episode(
            index,
            collection_role=collection_role,
            episode_role=episode_role,
            outer_fold=outer_fold,
            source_domain=(
                target
                if collection_role == schema.COLLECTION_OUTER
                else sources[index % 2]
            ),
            detector_role=detector_role,
            oof_fold_index=oof_index,
        )
        for index in range(count)
    ]
    schema.verify_episode_collection_completeness(episodes)
    with pytest.raises(schema.Stage2CrossfitContractError, match="empty|count mismatch"):
        schema.verify_episode_collection_completeness(episodes[:-1])


def _context_payload(tmp_path: Path) -> tuple[dict[str, object], object, object, object]:
    outer = "outer_leave_nuaa_sirst"
    source = "NUDT-SIRST"
    context = [
        _row("context-package", "context", i, source, outer, oof_fold_index=None)
        for i in range(14)
    ]
    query = [
        _row("context-package", "query", i, source, outer, oof_fold_index=None)
        for i in range(28)
    ]
    selected = {
        "window_index": 0,
        "window_id": "synthetic-window-0",
        "context_records": [
            {
                key: row[key]
                for key in (
                    "canonical_id",
                    "image_id",
                    "original_image_sha256",
                    "exclusion_group_id",
                    "near_duplicate_cluster_id_or_unique_sentinel",
                    "source_role_record_index",
                )
            }
            for row in context
        ],
        "query_records": [
            {
                key: row[key]
                for key in (
                    "canonical_id",
                    "image_id",
                    "original_image_sha256",
                    "exclusion_group_id",
                    "near_duplicate_cluster_id_or_unique_sentinel",
                    "source_role_record_index",
                )
            }
            for row in query
        ],
    }
    window_sha = w05_canonical_json_sha256(selected)
    values = np.zeros(93, dtype=np.float32)
    checkpoint_sha = _sha("context-checkpoint")
    payload = {
        "schema_version": schema.CONTEXT_PACKAGE_SCHEMA,
        "artifact_type": schema.CONTEXT_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY_VERIFIED_UNLABELED",
        "development_only": True,
        "official_test_accessed": False,
        "path_anchor": "repository_root",
        "context_package_id": _sha("context-package-id"),
        "expected_role": SOURCE_DIAGNOSTIC_VALIDATION,
        "episode_role": SOURCE_DIAGNOSTIC_VALIDATION,
        "outer_fold_id": outer,
        "outer_target": schema.OUTER_TARGETS[outer],
        "source_domain": source,
        "base_seed": 42,
        "derived_seed": 42001,
        "detector_identity": {"checkpoint_sha256": checkpoint_sha},
        "geometry": dict(schema.GEOMETRY),
        "window_binding": {
            "path": "synthetic/window.json",
            "sha256": _sha("window-manifest"),
            "window_id": selected["window_id"],
            "window_identity_sha256": window_sha,
        },
        "score_manifest_binding": {
            "path": "synthetic/scores.json",
            "sha256": _sha("score-manifest"),
            "records_content_sha256": _sha("score-records"),
            "role": SOURCE_DIAGNOSTIC_VALIDATION,
        },
        "score_bindings": {},
        "source_reference_binding": {
            "path": "synthetic/reference.npz",
            "sha256": _sha("reference"),
            "audit_path": "synthetic/reference.audit.json",
            "audit_sha256": _sha("reference-audit"),
            "reference_role": "synthetic",
        },
        "statistics_config_binding": {},
        "extractor_binding": {},
        "partition_bindings": {},
        "seed_binding": {},
        "governance_bindings": {},
        "context_records": context,
        "query_identity_records": query,
        "context_full_identity_sha256": schema.full_identity_sha256(context),
        "source_query_full_identity_sha256": schema.full_identity_sha256(query),
        "source_ordered_query_identity_sha256": _w05_query_sha(query),
        "context_statistics": {
            "feature_names": list(FEATURE_NAMES),
            "feature_dim": 93,
            "dtype": "float32",
            "values": values.tolist(),
            "vector_sha256_algorithm": schema.FLOAT32_VECTOR_ALGORITHM,
            "vector_sha256": hashlib.sha256(values.astype("<f4").tobytes()).hexdigest(),
            "metadata": {},
        },
        "guardrails": {
            "context_labels_loaded": False,
            "query_labels_loaded": False,
            "mask_or_label_paths_resolved": False,
            "curve_artifacts_accessed": False,
            "official_test_accessed": False,
        },
    }
    window = SimpleNamespace(
        window=selected,
        query_records=tuple(selected["query_records"]),
        window_id=selected["window_id"],
        window_identity_sha256=window_sha,
    )
    score = SimpleNamespace(
        repository_root=tmp_path,
        manifest_sha256=payload["score_manifest_binding"]["sha256"],
        records_content_sha256=payload["score_manifest_binding"][
            "records_content_sha256"
        ],
    )
    reference = SimpleNamespace(
        stage2_contract={
            "detector_identity": {"checkpoint_sha256": checkpoint_sha}
        }
    )
    return payload, window, score, reference


def test_context_bundle_public_verifier_atomic_and_shared_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, window, score, reference = _context_payload(tmp_path)
    statistics_config = StatisticsConfig(peak_kernel_size=3, peak_min_score=0.05)

    def fake_build(**_: object) -> tuple[dict[str, object], object, object, object]:
        return deepcopy(payload), window, score, reference

    monkeypatch.setattr(schema, "build_context_payload", fake_build)
    monkeypatch.setattr(builder, "build_context_payload", fake_build)
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    synthetic = tmp_path / "synthetic"
    synthetic.mkdir()
    for name in ("window.json", "scores.json", "reference.npz", "reference.audit.json"):
        (synthetic / name).write_bytes(name.encode("ascii"))
    output = output_dir / "context.json"
    result = builder.build_stage2_context_package(
        window_manifest="ignored-window",
        window_manifest_sha256=_sha("ignored-window"),
        window_id="synthetic-window-0",
        expected_role=SOURCE_DIAGNOSTIC_VALIDATION,
        score_manifest="ignored-score",
        score_manifest_sha256=_sha("ignored-score"),
        source_reference="ignored-reference",
        source_reference_sha256=_sha("ignored-reference"),
        source_reference_audit_sha256=_sha("ignored-audit"),
        statistics_config=statistics_config,
        output=output,
        repository_root_value=tmp_path,
    )
    verified = schema.verify_stage2_context_package(
        output,
        result["context_sha256"],
        result["commit_sha256"],
        statistics_config=statistics_config,
        repository_root=tmp_path,
    )
    assert output.is_file()
    assert schema.sidecar_path(output).is_file()
    assert verified.commit_path.is_file()
    assert schema.sidecar_path(verified.commit_path).is_file()
    assert not (output.parent / f".{output.name}.lock").exists()
    assert not tuple(output.parent.glob(f".{output.name}.staging-*"))
    frozen_bytes = {
        path: path.read_bytes()
        for path in (
            output,
            schema.sidecar_path(output),
            verified.commit_path,
            schema.sidecar_path(verified.commit_path),
        )
    }
    with pytest.raises(FileExistsError):
        builder.build_stage2_context_package(
            window_manifest="ignored-window",
            window_manifest_sha256=_sha("ignored-window"),
            window_id="synthetic-window-0",
            expected_role=SOURCE_DIAGNOSTIC_VALIDATION,
            score_manifest="ignored-score",
            score_manifest_sha256=_sha("ignored-score"),
            source_reference="ignored-reference",
            source_reference_sha256=_sha("ignored-reference"),
            source_reference_audit_sha256=_sha("ignored-audit"),
            statistics_config=statistics_config,
            output=output,
            repository_root_value=tmp_path,
        )
    assert {path: path.read_bytes() for path in frozen_bytes} == frozen_bytes
    with pytest.raises(Exception, match="SHA|sha|mismatch"):
        schema.verify_stage2_context_package(
            output,
            _sha("wrong"),
            result["commit_sha256"],
            statistics_config=statistics_config,
            repository_root=tmp_path,
        )

    shared = schema.make_stage2_shared_input_bindings(verified)
    assert shared["window_id"] == window.window_id
    original_identity = shared["shared_input_identity_sha256"]
    # A shallow nested-payload mutation cannot influence the bridge: it
    # projects from independent verifier objects.
    verified.payload["window_binding"]["window_id"] = "forged-window"
    assert (
        schema.make_stage2_shared_input_bindings(verified)[
            "shared_input_identity_sha256"
        ]
        == original_identity
    )
    mutated = make_shared_input_bindings(
        context_package_path=shared["context_package"]["path"],
        context_package_sha256=shared["context_package"]["sha256"],
        context_package_commit_path=shared["context_package_commit"]["path"],
        context_package_commit_sha256=shared["context_package_commit"]["sha256"],
        window_id="mutated-window",
        window_identity_sha256=shared["window_identity_sha256"],
        ordered_query_identity_sha256=shared["ordered_query_identity_sha256"],
        score_manifest_sha256=shared["score_manifest_sha256"],
        score_records_content_sha256=shared["score_records_content_sha256"],
        detector_checkpoint_sha256=shared["detector_checkpoint_sha256"],
    )
    assert mutated["shared_input_identity_sha256"] != original_identity


def test_statistics_config_public_verifier_is_strict(tmp_path: Path) -> None:
    config = StatisticsConfig(
        peak_kernel_size=3, peak_min_score=0.05, quantile_sample_limit=128
    )
    path = tmp_path / "statistics.json"
    data = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode() + b"\n"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    assert schema.verify_stage2_statistics_config(
        path, digest, repository_root=tmp_path
    ) == config
    with pytest.raises(Exception, match="SHA|sha|mismatch"):
        schema.verify_stage2_statistics_config(
            path, _sha("wrong-config"), repository_root=tmp_path
        )
    text = data.decode("utf-8")
    duplicate = text.replace("{", '{"peak_kernel_size":3,', 1).encode()
    path.write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        schema.verify_stage2_statistics_config(
            path, hashlib.sha256(duplicate).hexdigest(), repository_root=tmp_path
        )


def test_shared_real_synthetic_context_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stage2_crossfit_fixtures import publish_synthetic_verified_context
    from test_stage2_label_curve_contract import _synthetic_workspace

    workspace = _synthetic_workspace(
        tmp_path, monkeypatch, role=SOURCE_DIAGNOSTIC_VALIDATION
    )
    result = publish_synthetic_verified_context(workspace, monkeypatch)
    verified = result["verified"]
    assert verified.path == result["path"]
    assert verified.context_sha256 == result["sha256"]
    assert verified.commit_sha256 == result["commit_sha256"]
    assert verified.payload["guardrails"] == {
        "context_labels_loaded": False,
        "query_labels_loaded": False,
        "mask_or_label_paths_resolved": False,
        "curve_artifacts_accessed": False,
        "official_test_accessed": False,
    }
    assert len(verified.payload["context_records"]) == 14
    assert len(verified.payload["query_identity_records"]) == 28

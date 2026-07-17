from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from evaluation.stage2_paired_bootstrap import (
    AUTHORIZATION_AMENDMENT_SHA256,
    BASE_SEED_ORDER,
    BOOTSTRAP_ROLE,
    DOMAIN_ORDER,
    IMAGE_COUNTS_SCHEMA_VERSION,
    INDEX_MANIFEST_SCHEMA_VERSION,
    PAIR_MANIFEST_SCHEMA_VERSION,
    PRIMARY_RESAMPLES,
    PROTOCOL_ID,
    QUERY_IMAGES_PER_WINDOW,
    QUERY_INDEX_TAG,
    SEED_INDEX_TAG,
    SEED_MANIFEST_SCHEMA_VERSION,
    SOURCE_THAW_SHA256,
    THRESHOLD_SEMANTICS,
    WINDOW_COUNT_BY_DOMAIN,
    WINDOW_INDEX_TAG,
    WORK_BREAKDOWN_SHA256,
    Stage2BootstrapContractError,
    _transactional_publish_bundle,
    canonical_json_bytes,
    evaluate_paired_bootstrap,
    extract_bootstrap_root_seeds,
    generate_paired_hierarchical_indices,
    indices_for_method,
    sha256_bytes,
    stateless_query_indices,
    stateless_seed_index,
    stateless_window_index,
    type7_quantile,
    validate_primary_pair_manifest,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_bytes(payload: object) -> bytes:
    return canonical_json_bytes(payload) + b"\n"


def _seed_manifest() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    root = 1000
    for base_seed in BASE_SEED_ORDER:
        for outer_fold in DOMAIN_ORDER:
            root += 1
            rows.append(
                {
                    "base_seed": base_seed,
                    "outer_fold_id": outer_fold,
                    "derived_seeds_by_role": {BOOTSTRAP_ROLE: root},
                }
            )
    return {
        "schema_version": SEED_MANIFEST_SCHEMA_VERSION,
        "dimensions": {
            "base_seeds": list(BASE_SEED_ORDER),
            "outer_folds": list(DOMAIN_ORDER),
        },
        "derived_seed_table": rows,
    }


def _query_counts(
    outer_fold: str,
    base_seed: int,
    window_index: int,
    method_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for query_index in range(QUERY_IMAGES_PER_WINDOW):
        rows.append(
            {
                "image_id": (
                    f"{outer_fold}:{base_seed}:w{window_index}:q{query_index:02d}"
                ),
                "original_image_sha256": _sha(
                    f"{outer_fold}:{base_seed}:w{window_index}:image:{query_index}"
                ),
                "false_positive_pixels": (
                    0 if method_id == "T8" else (3 if query_index % 2 == 0 else 0)
                ),
                "total_pixels": 100_000,
                "background_pixels": 99_999,
                "matched_targets": 1 if method_id == "T8" else query_index % 2,
                "ground_truth_targets": 1,
            }
        )
    return rows


def _pair_manifest() -> dict[str, object]:
    domains: list[dict[str, object]] = []
    targets = ("nuaa-sirst", "nudt-sirst", "irstd-1k")
    for outer_fold, target in zip(DOMAIN_ORDER, targets):
        window_count = WINDOW_COUNT_BY_DOMAIN[outer_fold]
        cells: list[dict[str, object]] = []
        for base_seed in BASE_SEED_ORDER:
            methods: dict[str, object] = {}
            common_window_identities: list[dict[str, str]] = []
            for method_id in ("T8", "T4"):
                windows: list[dict[str, object]] = []
                for window_index in range(window_count):
                    counts = _query_counts(
                        outer_fold, base_seed, window_index, method_id
                    )
                    query_identity = [
                        {
                            "image_id": row["image_id"],
                            "original_image_sha256": row[
                                "original_image_sha256"
                            ],
                        }
                        for row in counts
                    ]
                    query_sha = sha256_bytes(canonical_json_bytes(query_identity))
                    context_sha = _sha(
                        f"{outer_fold}:{base_seed}:w{window_index}:context"
                    )
                    window_id = f"{outer_fold}:{base_seed}:window-{window_index}"
                    common = {
                        "window_id": window_id,
                        "context_identity_sha256": context_sha,
                        "ordered_query_identity_sha256": query_sha,
                    }
                    window_sha = sha256_bytes(canonical_json_bytes(common))
                    if method_id == "T8":
                        common_window_identities.append(
                            {
                                "window_id": window_id,
                                "window_identity_sha256": window_sha,
                                "context_identity_sha256": context_sha,
                                "ordered_query_identity_sha256": query_sha,
                            }
                        )
                    windows.append(
                        {
                            **common,
                            "window_identity_sha256": window_sha,
                            "decision_sha256": _sha(
                                f"{outer_fold}:{base_seed}:{window_index}:{method_id}:decision"
                            ),
                            "decision_sealed": True,
                            "threshold": 0.7 if method_id == "T8" else 0.6,
                            "threshold_semantics": THRESHOLD_SEMANTICS,
                            "online_update_count": 0,
                            "threshold_reselected": False,
                            "query_counts": counts,
                        }
                    )
                methods[method_id] = {
                    "schema_version": IMAGE_COUNTS_SCHEMA_VERSION,
                    "method_id": method_id,
                    "detector_checkpoint_sha256": _sha(
                        f"{outer_fold}:{base_seed}:detector"
                    ),
                    "method_checkpoint_sha256": _sha(
                        f"{outer_fold}:{base_seed}:{method_id}:checkpoint"
                    ),
                    "windows": windows,
                }
            cells.append(
                {
                    "base_seed": base_seed,
                    "ordered_window_identity_sha256": sha256_bytes(
                        canonical_json_bytes(common_window_identities)
                    ),
                    "methods": methods,
                }
            )
        domains.append(
            {
                "outer_fold_id": outer_fold,
                "target_dataset": target,
                "window_count": window_count,
                "cells": cells,
            }
        )
    return {
        "schema_version": PAIR_MANIFEST_SCHEMA_VERSION,
        "source_thaw_sha256": SOURCE_THAW_SHA256,
        "work_breakdown_sha256": WORK_BREAKDOWN_SHA256,
        "authorization_amendment_sha256": AUTHORIZATION_AMENDMENT_SHA256,
        "comparison": {
            "left_method": "T8",
            "right_method": "T4",
            "difference_order": "T8_minus_T4",
        },
        "primary_budget": 1e-5,
        "threshold_semantics": THRESHOLD_SEMANTICS,
        "domain_weighting": "fixed_equal_one_third",
        "seed_weighting": "equal_one_third_within_domain",
        "official_test_used": False,
        "domains": domains,
    }


@pytest.fixture(scope="module")
def artifact_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    root = tmp_path_factory.mktemp("stage2-bootstrap-v2")
    pair = _pair_manifest()
    seed = _seed_manifest()
    pair_path = root / "pair.json"
    seed_path = root / "seed.json"
    pair_data = _file_bytes(pair)
    seed_data = _file_bytes(seed)
    pair_path.write_bytes(pair_data)
    seed_path.write_bytes(seed_data)
    seed_sha = sha256_bytes(seed_data)
    indices = generate_paired_hierarchical_indices(
        pair, seed, seed_manifest_sha256=seed_sha
    )
    index_path = root / "indices.json"
    index_data = _file_bytes(indices)
    index_path.write_bytes(index_data)
    return {
        "root": root,
        "pair": pair,
        "seed": seed,
        "indices": indices,
        "pair_path": pair_path,
        "seed_path": seed_path,
        "index_path": index_path,
        "pair_sha": sha256_bytes(pair_data),
        "seed_sha": seed_sha,
        "index_sha": sha256_bytes(index_data),
    }


def _evaluate(fixture: dict[str, object]) -> dict[str, object]:
    return evaluate_paired_bootstrap(
        fixture["pair_path"],
        fixture["pair_sha"],
        fixture["index_path"],
        fixture["index_sha"],
        fixture["seed_path"],
        fixture["seed_sha"],
        repository_root=fixture["root"],
    )


def test_pair_manifest_is_exact_seed_window_query_v2() -> None:
    pair = validate_primary_pair_manifest(_pair_manifest())
    assert [domain["window_count"] for domain in pair["domains"]] == [1, 3, 3]
    assert all(
        len(window["query_counts"]) == 28
        for domain in pair["domains"]
        for cell in domain["cells"]
        for method in cell["methods"].values()
        for window in method["windows"]
    )
    assert pair["authorization_amendment_sha256"] == AUTHORIZATION_AMENDMENT_SHA256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("window_count", "window_count mismatch"),
        ("query_count", "exactly 28"),
        ("query_identity", "query identity hash mismatch"),
        ("paired_denominator", "denominator mismatch"),
        ("inestimable", "inestimable"),
        ("zero_gt", "zero GT"),
    ],
)
def test_pair_contract_zero_missing_tolerance(mutation: str, message: str) -> None:
    pair = _pair_manifest()
    domain = pair["domains"][1]  # type: ignore[index]
    cell = domain["cells"][0]
    if mutation == "window_count":
        domain["window_count"] = 2
    elif mutation == "query_count":
        cell["methods"]["T4"]["windows"][0]["query_counts"].pop()
    elif mutation == "query_identity":
        cell["methods"]["T4"]["windows"][0]["query_counts"][0][
            "image_id"
        ] = "replacement"
    elif mutation == "paired_denominator":
        cell["methods"]["T4"]["windows"][0]["query_counts"][0][
            "total_pixels"
        ] += 1
    elif mutation == "inestimable":
        for method_id in ("T8", "T4"):
            for window in cell["methods"][method_id]["windows"]:
                for row in window["query_counts"]:
                    row["background_pixels"] = 1
    elif mutation == "zero_gt":
        for method_id in ("T8", "T4"):
            for window in cell["methods"][method_id]["windows"]:
                for row in window["query_counts"]:
                    row["matched_targets"] = 0
                    row["ground_truth_targets"] = 0
    with pytest.raises(Stage2BootstrapContractError, match=message):
        validate_primary_pair_manifest(pair)


def test_three_stateless_v2_preimages_match_manual_sha256() -> None:
    root = 1065371701
    replicate = 17
    slot = 2
    seed_preimage = json.dumps(
        [SEED_INDEX_TAG, root, replicate, slot], separators=(",", ":")
    ).encode()
    assert stateless_seed_index(root, replicate, slot) == (
        int.from_bytes(hashlib.sha256(seed_preimage).digest()[:8], "big") % 3
    )
    window_preimage = json.dumps(
        [WINDOW_INDEX_TAG, root, replicate, slot, 1], separators=(",", ":")
    ).encode()
    assert stateless_window_index(root, replicate, slot, 1, 3) == (
        int.from_bytes(hashlib.sha256(window_preimage).digest()[:8], "big") % 3
    )
    window_id = "frozen-window"
    query = stateless_query_indices(root, replicate, slot, 1, window_id)
    assert len(query) == 28
    manual = json.dumps(
        [QUERY_INDEX_TAG, root, replicate, slot, 1, window_id, 0],
        separators=(",", ":"),
    ).encode()
    assert query[0] == int.from_bytes(hashlib.sha256(manual).digest()[:8], "big") % 28


def test_10000_manifest_has_exact_three_level_method_agnostic_indices(
    artifact_fixture: dict[str, object],
) -> None:
    indices = artifact_fixture["indices"]
    assert indices["schema_version"] == INDEX_MANIFEST_SCHEMA_VERSION
    assert indices["protocol_id"] == PROTOCOL_ID
    assert len(indices["replicates"]) == PRIMARY_RESAMPLES
    assert indices["window_counts"] == [1, 3, 3]
    assert indices["query_images_per_window"] == 28
    assert indices["method_id_present_in_draw_preimages"] is False
    assert indices_for_method(indices, "T8") == indices_for_method(indices, "T4")
    assert b'"T8"' not in canonical_json_bytes(indices)
    assert b'"T4"' not in canonical_json_bytes(indices)
    first = indices["replicates"][0]
    for domain_index, domain in enumerate(first["domains"]):
        expected_windows = (1, 3, 3)[domain_index]
        assert len(domain["seed_slots"]) == 3
        for seed_slot in domain["seed_slots"]:
            assert len(seed_slot["windows"]) == expected_windows
            assert all(len(window["query_indices"]) == 28 for window in seed_slot["windows"])
            if domain_index == 0:
                assert all(window["selected_window_index"] == 0 for window in seed_slot["windows"])


def test_public_evaluator_rehashes_files_replays_all_draws_and_reports_estimand(
    artifact_fixture: dict[str, object],
) -> None:
    report = _evaluate(artifact_fixture)
    assert report["resample_count"] == 10_000
    assert report["window_counts"] == {
        "nuaa-sirst": 1,
        "nudt-sirst": 3,
        "irstd-1k": 3,
    }
    assert report["fa_pixel_denominator"] == "all_native_resolution_pixels"
    assert report["bsr_aggregation"].startswith("equal_window")
    assert report["pd_aggregation"].startswith("pooled_within_seed")
    assert report["T8_T4_index_bytes_identical"] is True
    assert report["missing_primary_pair_count"] == 0
    assert report["confidence_interval"]["method"].endswith("type_7")
    assert report["nuaa_one_window_degeneracy"] == {
        "window_resampling_deterministic": True,
        "between_window_variance_estimable": False,
        "domain_weight_remains_one_third": True,
        "alternate_context_synthesized_or_replaced": False,
    }


@pytest.mark.parametrize("which", ["pair_sha", "index_sha", "seed_sha"])
def test_public_evaluator_rejects_caller_supplied_fake_artifact_hash(
    artifact_fixture: dict[str, object], which: str
) -> None:
    fixture = dict(artifact_fixture)
    fixture[which] = "0" * 64
    with pytest.raises(Stage2BootstrapContractError, match="SHA-256 mismatch"):
        _evaluate(fixture)


def test_self_consistent_selector_window_query_and_identity_tampering_fails_replay(
    artifact_fixture: dict[str, object], tmp_path: Path
) -> None:
    mutated = copy.deepcopy(artifact_fixture["indices"])
    slot = mutated["replicates"][0]["domains"][1]["seed_slots"][0]
    slot["selector_slot_index"] = 1
    slot["selected_seed_index"] = (slot["selected_seed_index"] + 1) % 3
    slot["selected_base_seed"] = BASE_SEED_ORDER[slot["selected_seed_index"]]
    window = slot["windows"][0]
    window["selected_window_index"] = (window["selected_window_index"] + 1) % 3
    window["selected_window_id"] = "self-consistent-attacker-window"
    window["query_indices"][0] = (window["query_indices"][0] + 1) % 28
    mutated["common_geometry"][1]["cells"][0]["windows"][0][
        "window_id"
    ] = "self-consistent-attacker-window"
    mutated["pairing_geometry_sha256"] = sha256_bytes(
        canonical_json_bytes(mutated["common_geometry"])
    )
    path = tmp_path / "mutated-index.json"
    data = _file_bytes(mutated)
    path.write_bytes(data)
    fixture = dict(artifact_fixture)
    fixture["root"] = tmp_path
    for key in ("pair_path", "seed_path"):
        copied = tmp_path / Path(fixture[key]).name
        copied.write_bytes(Path(fixture[key]).read_bytes())
        fixture[key] = copied
    fixture["index_path"] = path
    fixture["index_sha"] = sha256_bytes(data)
    with pytest.raises(Stage2BootstrapContractError, match="stateless"):
        _evaluate(fixture)


def test_type7_quantile_exact_linear_definition() -> None:
    values = [0.0, 10.0, 20.0, 30.0]
    assert type7_quantile(values, 0.025) == pytest.approx(0.75)
    assert type7_quantile(values, 0.5) == 15.0
    assert type7_quantile(values, 0.975) == pytest.approx(29.25)


def test_seed_root_extraction_accepts_frozen_layout_and_rejects_missing() -> None:
    roots = extract_bootstrap_root_seeds(_seed_manifest())
    assert set(roots) == set(DOMAIN_ORDER)
    assert all(set(roots[domain]) == set(BASE_SEED_ORDER) for domain in DOMAIN_ORDER)
    missing = _seed_manifest()
    missing["derived_seed_table"].pop()
    with pytest.raises(Stage2BootstrapContractError, match="missing bootstrap roots"):
        extract_bootstrap_root_seeds(missing)


@pytest.mark.parametrize("occupied", ["result.json", "result.json.sha256"])
def test_atomic_bundle_preoccupied_member_leaves_zero_new_outputs(
    tmp_path: Path, occupied: str
) -> None:
    json_path = tmp_path / "result.json"
    sidecar = tmp_path / "result.json.sha256"
    occupied_path = tmp_path / occupied
    occupied_path.write_bytes(b"owned")
    with pytest.raises(Stage2BootstrapContractError, match="already exists"):
        _transactional_publish_bundle({json_path: b"{}\n", sidecar: b"digest\n"})
    assert occupied_path.read_bytes() == b"owned"
    other = sidecar if occupied_path == json_path else json_path
    assert not other.exists()


def test_atomic_bundle_rejects_symlink_without_overwrite(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.write_bytes(b"victim")
    output = tmp_path / "result.json"
    output.symlink_to(victim)
    sidecar = tmp_path / "result.json.sha256"
    with pytest.raises(Stage2BootstrapContractError, match="symlink"):
        _transactional_publish_bundle({output: b"{}\n", sidecar: b"digest\n"})
    assert victim.read_bytes() == b"victim"
    assert not sidecar.exists()


def test_atomic_bundle_rolls_back_json_if_sidecar_link_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    sidecar = tmp_path / "result.json.sha256"
    real_link = os.link
    calls = 0

    def failing_link(source: Path, target: Path, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic sidecar-stage failure")
        real_link(source, target, **kwargs)

    monkeypatch.setattr(os, "link", failing_link)
    with pytest.raises(OSError, match="sidecar-stage"):
        _transactional_publish_bundle({output: b"{}\n", sidecar: b"digest\n"})
    assert not output.exists()
    assert not sidecar.exists()
    assert not list(tmp_path.glob(".*.tmp"))

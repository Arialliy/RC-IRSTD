from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

import rc.stage2_deployment as deployment
from scripts.freeze_stage2_confirmatory_plan import (
    B2_AUTHORIZATION_SHA256,
    build_pre_open_plan,
    pretty_json_bytes as w12_pretty_json_bytes,
)
from scripts.materialize_stage2_confirmatory_identity import IDENTITY_SCHEMA_VERSION
from rc.stage2_deployment import (
    AUTHORIZATION_AMENDMENT_SHA256,
    CONTEXT_RULE,
    DECISION_SCHEMA_VERSION,
    MAX_DERIVED_SEED,
    PARTITION_RULE,
    PIXEL_BUDGET_GRID,
    PROTOCOL_SCHEMA_VERSION,
    QUERY_RULE,
    SCORE_IDENTITY_SCHEMA_VERSION,
    SOURCE_THAW_SHA256,
    THRESHOLD_SEMANTICS,
    WORK_BREAKDOWN_SHA256,
    Stage2DeploymentContractError,
    canonical_json_bytes,
    load_verified_sealed_decision,
    main,
    partition_first_c_all_remaining,
    seal_no_reject_curve_decision,
    sha256_bytes,
    validate_deployment_protocol_v2,
    validate_input_file,
    verify_sealed_no_reject_decision,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _protocol() -> dict[str, object]:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "source_thaw_sha256": SOURCE_THAW_SHA256,
        "work_breakdown_sha256": WORK_BREAKDOWN_SHA256,
        "authorization_amendment_sha256": AUTHORIZATION_AMENDMENT_SHA256,
        "context_size": 14,
        "context_rule": CONTEXT_RULE,
        "query_rule": QUERY_RULE,
        "partition_rule": PARTITION_RULE,
        "pixel_budget_grid": list(PIXEL_BUDGET_GRID),
        "threshold_semantics": THRESHOLD_SEMANTICS,
        "no_reject": True,
        "context_labels_loaded": False,
        "decision_sealed_before_query_labels": True,
        "online_updates_after_decision": False,
        "context_replacement_allowed": False,
        "query_truncation_allowed": False,
        "query_subsampling_allowed": False,
        "threshold_reselection_after_query_access": False,
        "config_sha256": _sha("config"),
        "split_sha256": _sha("split"),
        "context_rule_sha256": _sha("context-rule"),
        "geometry_sha256": _sha("geometry"),
        "detector_checkpoint_sha256": _sha("detector"),
        "pre_open_plan_sha256": _sha("pre-open-plan"),
        "confirmatory_identity_sha256": _sha("confirmatory-identity"),
    }


def _records(count: int = 17) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index in range(count):
        is_context = index < 14
        result.append(
            {
                "image_id": f"image-{index:02d}",
                "original_image_sha256": _sha(f"image-bytes-{index}"),
                "score_sha256": _sha(f"score-{index}") if is_context else None,
                "score_opened": is_context,
            }
        )
    return result


def _curve() -> list[dict[str, float]]:
    return [
        {"pixel_budget": 1e-4, "threshold": 0.60},
        {"pixel_budget": 1e-5, "threshold": 0.70},
        {"pixel_budget": 1e-6, "threshold": 0.80},
    ]


def _decision(**overrides: object) -> dict[str, object]:
    protocol = _protocol()
    arguments: dict[str, object] = {
        "protocol": protocol,
        "protocol_sha256": sha256_bytes(canonical_json_bytes(protocol)),
        "records": _records(),
        "threshold_curve": _curve(),
        "score_manifest_sha256": _sha("score-manifest"),
        "calibrator_checkpoint_sha256": _sha("calibrator"),
        "outer_fold_id": "outer_leave_nuaa_sirst",
        "target_dataset": "nuaa-sirst",
        "method_id": "T8",
        "base_seed": 42,
        "derived_seed": 214361673,
        "decision_timestamp_utc": "2026-07-16T00:00:00Z",
    }
    arguments.update(overrides)
    return seal_no_reject_curve_decision(**arguments)  # type: ignore[arg-type]


def _canonical_artifact_sha(decision: object) -> str:
    return sha256_bytes(canonical_json_bytes(decision))


def _refresh_internal_hashes(decision: dict[str, object]) -> None:
    decision["ordered_context_identity_sha256"] = sha256_bytes(
        canonical_json_bytes(decision["ordered_context_identity"])
    )
    decision["ordered_query_identity_sha256"] = sha256_bytes(
        canonical_json_bytes(decision["ordered_query_identity"])
    )
    decision["threshold_curve_sha256"] = sha256_bytes(
        canonical_json_bytes(decision["threshold_curve"])
    )
    decision.pop("decision_payload_sha256", None)
    decision["decision_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(decision)
    )


def _verify_semantics(decision: dict[str, object]) -> dict[str, object]:
    return verify_sealed_no_reject_decision(
        decision, _canonical_artifact_sha(decision)
    )


def _w12_artifacts(
    tmp_path: Path, records: list[dict[str, object]]
) -> tuple[Path, str, Path, str]:
    metadata = {
        "split_repository_relative_path": (
            "datasets/NUAA-SIRST/img_idx/test_NUAA-SIRST.txt"
        ),
        "split_expected_sha256": _sha("split"),
        "split_expected_record_count": len(records),
    }
    plan = build_pre_open_plan(
        metadata, split_metadata_sha256=_sha("split-metadata-artifact")
    )
    plan_path = tmp_path / "pre-open-plan.json"
    plan_bytes = w12_pretty_json_bytes(plan)
    plan_path.write_bytes(plan_bytes)
    plan_sha = sha256_bytes(plan_bytes)
    identity_rows = [
        {
            "position": index,
            "canonical_id": record["image_id"],
            "image_repository_relative_path": (
                f"datasets/NUAA-SIRST/images/{record['image_id']}.png"
            ),
            "original_image_sha256": record["original_image_sha256"],
        }
        for index, record in enumerate(records)
    ]
    context_rows = identity_rows[:14]
    query_rows = identity_rows[14:]
    identity = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "b2_authorization_sha256": B2_AUTHORIZATION_SHA256,
        "pre_open_plan_sha256": plan_sha,
        "s2_dgo_go_authorization_sha256": _sha("s2-dgo-authorization"),
        "development_gate_result_sha256": _sha("development-gate-result"),
        "target_dataset": "nuaa-sirst",
        "split_repository_relative_path": metadata[
            "split_repository_relative_path"
        ],
        "split_sha256": metadata["split_expected_sha256"],
        "split_record_count": len(records),
        "context_size": 14,
        "context_rule": CONTEXT_RULE,
        "query_rule": QUERY_RULE,
        "ordered_context_identity": context_rows,
        "ordered_context_identity_sha256": sha256_bytes(
            canonical_json_bytes(context_rows)
        ),
        "ordered_query_identity": query_rows,
        "ordered_query_identity_sha256": sha256_bytes(
            canonical_json_bytes(query_rows)
        ),
        "official_test_accessed": True,
        "official_test_split_opened": True,
        "official_test_images_opened": True,
        "official_test_masks_opened": False,
        "official_test_labels_opened": False,
        "inference_run": False,
        "metric_computed": False,
        "threshold_decision_sealed": False,
        "result_based_rerun": False,
    }
    identity_path = tmp_path / "confirmatory-identity.json"
    identity_bytes = w12_pretty_json_bytes(identity)
    identity_path.write_bytes(identity_bytes)
    return plan_path, plan_sha, identity_path, sha256_bytes(identity_bytes)


def _cli_fixture(tmp_path: Path, *, output_name: str = "decision.json") -> tuple[list[str], Path]:
    protocol = _protocol()
    records = _records()
    plan_path, plan_sha, identity_path, identity_sha = _w12_artifacts(
        tmp_path, records
    )
    protocol["pre_open_plan_sha256"] = plan_sha
    protocol["confirmatory_identity_sha256"] = identity_sha
    protocol_bytes = json.dumps(protocol, sort_keys=True, indent=2).encode("utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(protocol_bytes)
    checkpoint = tmp_path / "calibrator.pt"
    checkpoint.write_bytes(b"synthetic-opaque-checkpoint")
    manifest = {
        "schema_version": SCORE_IDENTITY_SCHEMA_VERSION,
        "ordered_records": records,
        "decision_input": {
            "threshold_curve": _curve(),
            "outer_fold_id": "outer_leave_nuaa_sirst",
            "target_dataset": "nuaa-sirst",
            "method_id": "T8",
            "base_seed": 42,
            "derived_seed": 214361673,
            "decision_timestamp_utc": "2026-07-16T00:00:00Z",
            "query_labels_attached": False,
            "threshold_reselected": False,
            "online_update_count": 0,
        },
        "official_test_labels_accessed": False,
        "official_test_query_scores_accessed": False,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path = tmp_path / "score-identity.json"
    manifest_path.write_bytes(manifest_bytes)
    output = tmp_path / output_name
    return [
        "--repository-root",
        str(tmp_path),
        "--protocol",
        str(protocol_path),
        "--protocol-sha256",
        sha256_bytes(protocol_bytes),
        "--score-manifest",
        str(manifest_path),
        "--score-manifest-sha256",
        sha256_bytes(manifest_bytes),
        "--calibrator-checkpoint",
        str(checkpoint),
        "--calibrator-checkpoint-sha256",
        sha256_bytes(checkpoint.read_bytes()),
        "--pre-open-plan",
        str(plan_path),
        "--pre-open-plan-sha256",
        plan_sha,
        "--confirmatory-identity",
        str(identity_path),
        "--confirmatory-identity-sha256",
        identity_sha,
        "--output",
        str(output),
    ], output


def test_partition_is_exact_first_14_and_complete_unmodified_suffix() -> None:
    records = _records(19)
    context, query = partition_first_c_all_remaining(records)
    assert len(context) == 14
    assert len(query) == 5
    assert tuple(context) + tuple(query) == tuple(records)
    assert all(left is right for left, right in zip(context + query, records))


@pytest.mark.parametrize("context_size", [13, 15, True, 14.0])
def test_partition_rejects_any_nonfrozen_context_size(context_size: object) -> None:
    with pytest.raises((TypeError, Stage2DeploymentContractError)):
        partition_first_c_all_remaining(_records(), context_size=context_size)  # type: ignore[arg-type]


def test_partition_rejects_empty_suffix() -> None:
    with pytest.raises(Stage2DeploymentContractError, match="at least one query"):
        partition_first_c_all_remaining(_records(14))


def test_protocol_v2_is_amendment_bound_strict_and_has_no_query_size() -> None:
    protocol = validate_deployment_protocol_v2(_protocol())
    assert protocol["authorization_amendment_sha256"] == AUTHORIZATION_AMENDMENT_SHA256
    assert protocol["query_rule"] == "all_remaining_suffix"
    assert "query_size" not in protocol

    mutated = _protocol()
    mutated["no_reject"] = 1
    with pytest.raises(TypeError, match="exact JSON boolean"):
        validate_deployment_protocol_v2(mutated)

    mutated = _protocol()
    mutated["query_size"] = 28
    with pytest.raises(Stage2DeploymentContractError, match="extra=.*query_size"):
        validate_deployment_protocol_v2(mutated)

    mutated = _protocol()
    mutated["authorization_amendment_sha256"] = "0" * 64
    with pytest.raises(Stage2DeploymentContractError, match="authorization amendment"):
        validate_deployment_protocol_v2(mutated)

    mutated = _protocol()
    mutated["pixel_budget_grid"] = tuple(PIXEL_BUDGET_GRID)
    with pytest.raises(TypeError, match="exact JSON array"):
        validate_deployment_protocol_v2(mutated)


def test_sealed_decision_binds_complete_suffix_curve_and_external_sha() -> None:
    decision = _decision()
    assert decision["schema_version"] == DECISION_SCHEMA_VERSION
    assert decision["pre_open_plan_sha256"] == _protocol()["pre_open_plan_sha256"]
    assert decision["confirmatory_identity_sha256"] == _protocol()["confirmatory_identity_sha256"]
    assert decision["authorization_amendment_sha256"] == AUTHORIZATION_AMENDMENT_SHA256
    assert decision["query_count"] == 3
    assert [row["image_id"] for row in decision["ordered_query_identity"]] == [
        "image-14", "image-15", "image-16"
    ]
    assert decision["threshold_curve"] == _curve()
    assert decision["no_reject"] is True
    assert decision["decision_sealed"] is True
    assert _verify_semantics(decision) == decision
    serialized = canonical_json_bytes(decision).decode("utf-8")
    for forbidden in ('"reject_cutoff"', '"reject_probability"', '"p_min"', '"query_size"'):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"query_labels_attached": True}, "labels"),
        ({"threshold_reselected": True}, "reselection"),
        ({"online_update_count": 1}, "online threshold updates"),
    ],
)
def test_sealer_rejects_label_access_reselection_and_online_update(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(Stage2DeploymentContractError, match=message):
        _decision(**override)


def test_sealer_rejects_opened_query_score_and_query_label_field() -> None:
    records = _records()
    records[14]["score_opened"] = True
    records[14]["score_sha256"] = _sha("forbidden-query-score")
    with pytest.raises(Stage2DeploymentContractError, match="query scores"):
        _decision(records=records)

    records = _records()
    records[15]["mask_path"] = "/forbidden/label.png"
    with pytest.raises(Stage2DeploymentContractError, match="query-label fields"):
        _decision(records=records)


@pytest.mark.parametrize(
    "curve",
    [
        [
            {"pixel_budget": 1e-4, "threshold": 0.6},
            {"pixel_budget": 1e-5, "threshold": 0.5},
            {"pixel_budget": 1e-6, "threshold": 0.8},
        ],
        [
            {"pixel_budget": 1e-4, "threshold": 0.6},
            {"pixel_budget": 1e-6, "threshold": 0.7},
            {"pixel_budget": 1e-5, "threshold": 0.8},
        ],
        [
            {"pixel_budget": 1e-4, "threshold": 0.6},
            {"pixel_budget": 1e-5, "threshold": 0.7},
        ],
    ],
)
def test_sealer_rejects_incomplete_reordered_or_nonmonotone_curve(
    curve: list[dict[str, float]],
) -> None:
    with pytest.raises(Stage2DeploymentContractError):
        _decision(threshold_curve=curve)


@pytest.mark.parametrize(
    "override",
    [
        {"base_seed": True},
        {"base_seed": 0},
        {"derived_seed": True},
        {"derived_seed": MAX_DERIVED_SEED + 1},
        {"decision_timestamp_utc": "2026-07-16T00:00:00+00:00"},
        {"decision_timestamp_utc": "2026-02-30T00:00:00Z"},
        {"target_dataset": "nudt-sirst"},
        {"method_id": "T10"},
    ],
)
def test_sealer_rejects_type_confused_or_unfrozen_scalars(override: dict[str, object]) -> None:
    with pytest.raises((TypeError, Stage2DeploymentContractError)):
        _decision(**override)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_seed", True),
        ("derived_seed", 1.0),
        ("query_count", True),
        ("decision_timestamp_utc", "2026-07-16T00:00:00.0Z"),
        ("method_id", 8),
        ("no_reject", 1),
    ],
)
def test_public_verifier_rejects_semantic_scalar_type_confusion(
    field: str, value: object
) -> None:
    decision = _decision()
    decision[field] = value
    _refresh_internal_hashes(decision)
    with pytest.raises((TypeError, Stage2DeploymentContractError)):
        _verify_semantics(decision)


@pytest.mark.parametrize("mutation", ["extra", "missing", "bool", "row_tuple", "array_tuple"])
def test_public_verifier_rejects_identity_row_schema_bypass(mutation: str) -> None:
    decision = _decision()
    rows = decision["ordered_context_identity"]
    assert isinstance(rows, list)
    if mutation == "extra":
        rows[0]["unexpected"] = "bypass"
    elif mutation == "missing":
        rows[0].pop("score_sha256")
    elif mutation == "bool":
        rows[0]["image_id"] = True
    elif mutation == "row_tuple":
        rows[0] = tuple(rows[0].items())
    else:
        decision["ordered_context_identity"] = tuple(rows)
    _refresh_internal_hashes(decision)
    with pytest.raises((TypeError, Stage2DeploymentContractError)):
        _verify_semantics(decision)


@pytest.mark.parametrize("boundary", ["image_id", "original_image_sha256", "score_sha256"])
def test_public_verifier_rejects_duplicate_identity_boundaries(boundary: str) -> None:
    decision = _decision()
    rows = decision["ordered_context_identity"]
    assert isinstance(rows, list)
    rows[1][boundary] = rows[0][boundary]
    _refresh_internal_hashes(decision)
    with pytest.raises(Stage2DeploymentContractError, match="duplicate"):
        _verify_semantics(decision)


def test_external_sha_rejects_self_consistent_identity_replacement() -> None:
    decision = _decision()
    trusted_expected = _canonical_artifact_sha(decision)
    mutated = copy.deepcopy(decision)
    mutated["ordered_query_identity"][0]["image_id"] = "replacement"
    mutated["ordered_query_identity"][0]["original_image_sha256"] = _sha("replacement")
    _refresh_internal_hashes(mutated)
    with pytest.raises(Stage2DeploymentContractError, match="artifact SHA-256 mismatch"):
        verify_sealed_no_reject_decision(mutated, trusted_expected)


def test_public_verifier_rejects_self_consistent_online_mutation() -> None:
    decision = _decision()
    decision["online_update_count"] = 1
    _refresh_internal_hashes(decision)
    with pytest.raises(Stage2DeploymentContractError, match="online_update_count"):
        _verify_semantics(decision)


def test_absolute_canonical_path_guard_rejects_relative_and_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    assert validate_input_file(real, name="fixture") == real
    with pytest.raises(Stage2DeploymentContractError, match="absolute"):
        validate_input_file(Path("relative.json"), name="fixture")
    alias = tmp_path / "alias.json"
    alias.symlink_to(real)
    with pytest.raises(Stage2DeploymentContractError, match="symlink"):
        validate_input_file(alias, name="fixture")


def test_verified_loader_binds_exact_file_bytes_and_repository_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    decision = _decision()
    data = json.dumps(decision, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path = root / "decision.json"
    path.write_bytes(data)
    digest = sha256_bytes(data)
    assert load_verified_sealed_decision(path, digest, root) == decision
    with pytest.raises(Stage2DeploymentContractError, match="external expectation"):
        load_verified_sealed_decision(path, "0" * 64, root)
    with pytest.raises(Stage2DeploymentContractError, match="absolute"):
        load_verified_sealed_decision(Path("decision.json"), digest, root)

    alias = root / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(Stage2DeploymentContractError, match="non-symlink"):
        load_verified_sealed_decision(alias, digest, root)

    outside = tmp_path / "outside.json"
    outside.write_bytes(data)
    with pytest.raises(Stage2DeploymentContractError, match="escapes repository_root"):
        load_verified_sealed_decision(outside, digest, root)


def test_cli_seals_synthetic_identity_as_atomic_json_sidecar_bundle(tmp_path: Path) -> None:
    argv, output = _cli_fixture(tmp_path)
    assert main(argv) == 0
    data = output.read_bytes()
    digest = sha256_bytes(data)
    sidecar = output.with_name(output.name + ".sha256")
    assert sidecar.read_text(encoding="utf-8") == f"{digest}  {output.name}\n"
    payload = load_verified_sealed_decision(output, digest, tmp_path)
    assert payload["query_scores_opened_at_seal"] is False
    assert payload["query_labels_attached_at_seal"] is False


@pytest.mark.parametrize("occupied", ["json", "sidecar"])
def test_cli_preoccupied_bundle_member_creates_no_other_output(
    tmp_path: Path, occupied: str
) -> None:
    argv, output = _cli_fixture(tmp_path)
    sidecar = output.with_name(output.name + ".sha256")
    occupied_path = output if occupied == "json" else sidecar
    occupied_path.write_bytes(b"preexisting")
    with pytest.raises(Stage2DeploymentContractError, match="already exists"):
        main(argv)
    assert occupied_path.read_bytes() == b"preexisting"
    other = sidecar if occupied == "json" else output
    assert not other.exists() and not other.is_symlink()


def test_cli_symlink_target_is_rejected_without_orphan(tmp_path: Path) -> None:
    argv, output = _cli_fixture(tmp_path)
    destination = tmp_path / "preexisting-target"
    destination.write_bytes(b"keep")
    output.symlink_to(destination)
    sidecar = output.with_name(output.name + ".sha256")
    with pytest.raises(Stage2DeploymentContractError, match="non-symlink"):
        main(argv)
    assert output.is_symlink()
    assert destination.read_bytes() == b"keep"
    assert not sidecar.exists()


def test_cli_sidecar_link_error_rolls_back_json_and_tempfiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, output = _cli_fixture(tmp_path)
    sidecar = output.with_name(output.name + ".sha256")
    real_link = os.link

    def failing_link(source: object, destination: object, *args: object, **kwargs: object) -> None:
        if Path(destination) == sidecar:
            raise OSError("simulated sidecar publication failure")
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(deployment.os, "link", failing_link)
    with pytest.raises(OSError, match="simulated sidecar"):
        main(argv)
    assert not output.exists() and not output.is_symlink()
    assert not sidecar.exists() and not sidecar.is_symlink()
    assert not list(tmp_path.glob(".*.tmp"))


def test_protocol_requires_both_w12_provenance_hashes() -> None:
    for field in ("pre_open_plan_sha256", "confirmatory_identity_sha256"):
        protocol = _protocol()
        protocol.pop(field)
        with pytest.raises(Stage2DeploymentContractError, match="missing"):
            validate_deployment_protocol_v2(protocol)


def test_cli_rejects_wrong_external_pre_open_sha_without_output(tmp_path: Path) -> None:
    argv, output = _cli_fixture(tmp_path)
    argv[argv.index("--pre-open-plan-sha256") + 1] = "0" * 64
    with pytest.raises(Stage2DeploymentContractError, match="external expectation"):
        main(argv)
    assert not output.exists()
    assert not output.with_name(output.name + ".sha256").exists()


def test_cli_rejects_self_consistent_identity_replacement_against_score_replay(
    tmp_path: Path,
) -> None:
    argv, output = _cli_fixture(tmp_path)
    identity_path = Path(argv[argv.index("--confirmatory-identity") + 1])
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    row = identity["ordered_query_identity"][0]
    row["canonical_id"] = "replacement-id"
    row["image_repository_relative_path"] = (
        "datasets/NUAA-SIRST/images/replacement-id.png"
    )
    identity["ordered_query_identity_sha256"] = sha256_bytes(
        canonical_json_bytes(identity["ordered_query_identity"])
    )
    identity_bytes = w12_pretty_json_bytes(identity)
    identity_path.write_bytes(identity_bytes)
    identity_sha = sha256_bytes(identity_bytes)
    argv[argv.index("--confirmatory-identity-sha256") + 1] = identity_sha

    protocol_path = Path(argv[argv.index("--protocol") + 1])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["confirmatory_identity_sha256"] = identity_sha
    protocol_bytes = json.dumps(protocol, sort_keys=True, indent=2).encode("utf-8")
    protocol_path.write_bytes(protocol_bytes)
    argv[argv.index("--protocol-sha256") + 1] = sha256_bytes(protocol_bytes)

    with pytest.raises(
        Stage2DeploymentContractError,
        match="differs from confirmatory identity",
    ):
        main(argv)
    assert not output.exists()
    assert not output.with_name(output.name + ".sha256").exists()

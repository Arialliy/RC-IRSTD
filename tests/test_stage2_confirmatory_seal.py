from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

import scripts.freeze_stage2_confirmatory_plan as freeze
import scripts.materialize_stage2_confirmatory_identity as materialize
from scripts.freeze_stage2_confirmatory_plan import (
    B2_AUTHORIZATION_SHA256,
    CONTEXT_RULE,
    PLAN_SCHEMA_VERSION,
    QUERY_RULE,
    Stage2ConfirmatoryContractError,
    build_pre_open_plan,
    pretty_json_bytes,
    sha256_bytes,
    validate_pre_open_plan,
    validate_split_metadata,
    verify_pre_open_plan,
)
from scripts.materialize_stage2_confirmatory_identity import (
    DGO_AUTHORIZATION_SCHEMA_VERSION,
    IDENTITY_SCHEMA_VERSION,
    load_verified_confirmatory_identity,
    validate_confirmatory_identity,
    validate_dgo_opening_authorization,
    verify_confirmatory_identity,
)


def _write_json(path: Path, payload: object) -> str:
    data = pretty_json_bytes(payload)
    path.write_bytes(data)
    return sha256_bytes(data)


def _layout(
    root: Path,
    *,
    count: int = 17,
    materialize_split: bool = True,
) -> dict[str, object]:
    split_relative = "datasets/NUAA-SIRST/img_idx/test_NUAA-SIRST.txt"
    split = root / split_relative
    image_dir = root / "datasets/NUAA-SIRST/images"
    split.parent.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    ids = [f"synthetic-{index:03d}" for index in range(count)]
    split_bytes = ("\n".join(ids) + "\n").encode("utf-8")
    if materialize_split:
        split.write_bytes(split_bytes)
        for index, canonical_id in enumerate(ids):
            (image_dir / f"{canonical_id}.png").write_bytes(
                b"synthetic-image\x00" + index.to_bytes(2, "big")
            )
    metadata = {
        "split_repository_relative_path": split_relative,
        "split_expected_sha256": sha256_bytes(split_bytes),
        "split_expected_record_count": count,
    }
    metadata_path = root / "split-metadata.json"
    metadata_sha = _write_json(metadata_path, metadata)
    plan = build_pre_open_plan(metadata, split_metadata_sha256=metadata_sha)
    plan_path = root / "pre-open-plan.json"
    plan_sha = _write_json(plan_path, plan)
    authorization = {
        "schema_version": DGO_AUTHORIZATION_SCHEMA_VERSION,
        "gate_id": "S2_DGO",
        "decision": "GO",
        "development_gate_result_sha256": sha256_bytes(b"development-gate-result"),
        "pre_open_plan_sha256": plan_sha,
        "confirmatory_identity_materialization_authorized": True,
        "official_test_split_open_authorized": True,
        "official_test_image_open_authorized": True,
        "official_test_label_open_authorized": False,
        "result_based_rerun_authorized": False,
        "official_test_accessed": False,
    }
    authorization_path = root / "s2-dgo-authorization.json"
    authorization_sha = _write_json(authorization_path, authorization)
    return {
        "split": split,
        "ids": ids,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "metadata_sha": metadata_sha,
        "plan": plan,
        "plan_path": plan_path,
        "plan_sha": plan_sha,
        "authorization": authorization,
        "authorization_path": authorization_path,
        "authorization_sha": authorization_sha,
    }


def _freeze_args(root: Path, layout: dict[str, object], output: Path) -> list[str]:
    return [
        "--repository-root",
        str(root),
        "--split-metadata",
        str(layout["metadata_path"]),
        "--split-metadata-sha256",
        str(layout["metadata_sha"]),
        "--output",
        str(output),
    ]


def _identity_args(root: Path, layout: dict[str, object], output: Path) -> list[str]:
    return [
        "--repository-root",
        str(root),
        "--pre-open-plan",
        str(layout["plan_path"]),
        "--pre-open-plan-sha256",
        str(layout["plan_sha"]),
        "--s2-dgo-authorization",
        str(layout["authorization_path"]),
        "--s2-dgo-authorization-sha256",
        str(layout["authorization_sha"]),
        "--output",
        str(output),
    ]


def test_pre_open_plan_exact_schema_and_result_free_semantics(tmp_path: Path) -> None:
    layout = _layout(tmp_path, materialize_split=False)
    plan = validate_pre_open_plan(layout["plan"])  # type: ignore[arg-type]
    assert plan["schema_version"] == PLAN_SCHEMA_VERSION
    assert plan["b2_authorization_sha256"] == B2_AUTHORIZATION_SHA256
    assert plan["context_size"] == 14
    assert plan["context_rule"] == CONTEXT_RULE
    assert plan["query_rule"] == QUERY_RULE
    assert plan["official_test_accessed"] is False
    assert plan["official_test_ids_materialized"] is False
    assert plan["official_test_images_opened"] is False
    assert plan["official_test_masks_opened"] is False
    assert plan["official_test_labels_opened"] is False
    assert "ordered_context_identity" not in plan
    assert "ordered_query_identity" not in plan


def test_pre_open_cli_never_stats_or_opens_referenced_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path, materialize_split=False)
    split = layout["split"]
    assert isinstance(split, Path)
    calls: list[str] = []
    real_open = os.open
    real_stat = os.stat
    real_lstat = os.lstat

    def guarded_open(path: object, *args: object, **kwargs: object) -> int:
        if Path(path) == split:
            calls.append("open")
            raise AssertionError("official split open forbidden at S2_I0")
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    def guarded_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        if Path(path) == split:
            calls.append("stat")
            raise AssertionError("official split stat forbidden at S2_I0")
        return real_stat(path, *args, **kwargs)  # type: ignore[arg-type]

    def guarded_lstat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        if Path(path) == split:
            calls.append("lstat")
            raise AssertionError("official split lstat forbidden at S2_I0")
        return real_lstat(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(freeze.os, "open", guarded_open)
    monkeypatch.setattr(freeze.os, "stat", guarded_stat)
    monkeypatch.setattr(freeze.os, "lstat", guarded_lstat)
    output = tmp_path / "frozen-plan.json"
    assert freeze.main(_freeze_args(tmp_path, layout, output)) == 0
    assert calls == []
    payload = json.loads(output.read_text(encoding="utf-8"))
    digest = sha256_bytes(output.read_bytes())
    assert verify_pre_open_plan(payload, digest, artifact_bytes=output.read_bytes()) == payload
    assert output.with_name(output.name + ".sha256").read_text(encoding="ascii") == (
        f"{digest}  {output.name}\n"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"split_expected_record_count": True},
        {"split_expected_record_count": 14},
        {"split_expected_sha256": "0" * 63},
        {"split_repository_relative_path": "../test.txt"},
        {"split_repository_relative_path": "/absolute/test.txt"},
        {"unexpected": "field"},
    ],
)
def test_split_metadata_is_closed_and_type_strict(mutation: dict[str, object]) -> None:
    metadata: dict[str, object] = {
        "split_repository_relative_path": "datasets/NUAA-SIRST/img_idx/test_NUAA-SIRST.txt",
        "split_expected_sha256": "a" * 64,
        "split_expected_record_count": 17,
    }
    metadata.update(mutation)
    with pytest.raises((TypeError, Stage2ConfirmatoryContractError)):
        validate_split_metadata(metadata)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_size", True),
        ("context_size", 15),
        ("official_test_accessed", 0),
        ("official_test_ids_materialized", True),
        ("execution_authorized", True),
        ("query_rule", "first_28"),
    ],
)
def test_pre_open_plan_rejects_type_confusion_and_access_claims(
    tmp_path: Path, field: str, value: object
) -> None:
    layout = _layout(tmp_path, materialize_split=False)
    plan = copy.deepcopy(layout["plan"])
    assert isinstance(plan, dict)
    plan[field] = value
    with pytest.raises((TypeError, Stage2ConfirmatoryContractError)):
        validate_pre_open_plan(plan)


def test_pre_open_external_sha_rejects_self_consistent_mutation(tmp_path: Path) -> None:
    layout = _layout(tmp_path, materialize_split=False)
    trusted_sha = str(layout["plan_sha"])
    plan = copy.deepcopy(layout["plan"])
    assert isinstance(plan, dict)
    plan["split_expected_sha256"] = "b" * 64
    data = pretty_json_bytes(plan)
    with pytest.raises(Stage2ConfirmatoryContractError, match="external expectation"):
        verify_pre_open_plan(plan, trusted_sha, artifact_bytes=data)


@pytest.mark.parametrize("decision", ["HOLD", "PASS", True, 1])
def test_post_go_requires_exact_go_before_split_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: object,
) -> None:
    layout = _layout(tmp_path)
    authorization = copy.deepcopy(layout["authorization"])
    assert isinstance(authorization, dict)
    authorization["decision"] = decision
    authorization_sha = _write_json(
        layout["authorization_path"], authorization  # type: ignore[arg-type]
    )
    layout["authorization_sha"] = authorization_sha
    split = layout["split"]
    assert isinstance(split, Path)
    real_open = os.open
    split_opened = False

    def guarded_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal split_opened
        if Path(path) == split:
            split_opened = True
            raise AssertionError("split opened before valid GO")
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(materialize.os, "open", guarded_open)
    output = tmp_path / "identity.json"
    with pytest.raises((TypeError, Stage2ConfirmatoryContractError)):
        materialize.main(_identity_args(tmp_path, layout, output))
    assert split_opened is False
    assert not output.exists()
    assert not output.with_name(output.name + ".sha256").exists()


def test_post_go_authorization_must_bind_exact_plan_before_split_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    authorization = copy.deepcopy(layout["authorization"])
    assert isinstance(authorization, dict)
    authorization["pre_open_plan_sha256"] = "f" * 64
    layout["authorization_sha"] = _write_json(
        layout["authorization_path"], authorization  # type: ignore[arg-type]
    )
    split = layout["split"]
    assert isinstance(split, Path)
    real_open = os.open

    def guarded_open(path: object, *args: object, **kwargs: object) -> int:
        if Path(path) == split:
            raise AssertionError("split opened before plan binding check")
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(materialize.os, "open", guarded_open)
    with pytest.raises(Stage2ConfirmatoryContractError, match="does not bind"):
        materialize.main(
            _identity_args(tmp_path, layout, tmp_path / "identity.json")
        )


def test_synthetic_post_go_identity_is_exact_first14_complete_suffix_and_label_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path, count=18)
    forbidden = tmp_path / "datasets/NUAA-SIRST/masks"
    real_open = os.open
    real_stat = os.stat

    def guarded_open(path: object, *args: object, **kwargs: object) -> int:
        if Path(path).is_relative_to(forbidden):
            raise AssertionError("mask/label open is forbidden during identity freeze")
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    def guarded_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        if Path(path).is_relative_to(forbidden):
            raise AssertionError("mask/label stat is forbidden during identity freeze")
        return real_stat(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(materialize.os, "open", guarded_open)
    monkeypatch.setattr(materialize.os, "stat", guarded_stat)
    output = tmp_path / "identity.json"
    assert materialize.main(_identity_args(tmp_path, layout, output)) == 0
    digest = sha256_bytes(output.read_bytes())
    identity = load_verified_confirmatory_identity(output, digest, tmp_path)
    assert identity["schema_version"] == IDENTITY_SCHEMA_VERSION
    assert identity["pre_open_plan_sha256"] == layout["plan_sha"]
    assert identity["s2_dgo_go_authorization_sha256"] == layout["authorization_sha"]
    assert len(identity["ordered_context_identity"]) == 14
    assert len(identity["ordered_query_identity"]) == 4
    assert [row["canonical_id"] for row in identity["ordered_context_identity"]] == layout[
        "ids"
    ][:14]
    assert [row["canonical_id"] for row in identity["ordered_query_identity"]] == layout[
        "ids"
    ][14:]
    assert identity["official_test_accessed"] is True
    assert identity["official_test_masks_opened"] is False
    assert identity["official_test_labels_opened"] is False
    assert identity["inference_run"] is False
    assert identity["metric_computed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confirmatory_identity_materialization_authorized", 1),
        ("official_test_split_open_authorized", False),
        ("official_test_image_open_authorized", False),
        ("official_test_label_open_authorized", True),
        ("result_based_rerun_authorized", True),
        ("official_test_accessed", True),
    ],
)
def test_dgo_authorization_is_closed_and_exact_boolean(
    tmp_path: Path, field: str, value: object
) -> None:
    authorization = copy.deepcopy(_layout(tmp_path)["authorization"])
    assert isinstance(authorization, dict)
    authorization[field] = value
    with pytest.raises((TypeError, Stage2ConfirmatoryContractError)):
        validate_dgo_opening_authorization(authorization)


@pytest.mark.parametrize(
    "mutation",
    ["extra", "position_bool", "path_traversal", "label_flag", "query_reorder"],
)
def test_identity_verifier_rejects_closed_schema_and_identity_mutations(
    tmp_path: Path, mutation: str
) -> None:
    layout = _layout(tmp_path)
    output = tmp_path / "identity.json"
    materialize.main(_identity_args(tmp_path, layout, output))
    identity = json.loads(output.read_text(encoding="utf-8"))
    if mutation == "extra":
        identity["unexpected"] = "field"
    elif mutation == "position_bool":
        identity["ordered_context_identity"][0]["position"] = True
    elif mutation == "path_traversal":
        identity["ordered_query_identity"][0][
            "image_repository_relative_path"
        ] = "../mask.png"
    elif mutation == "label_flag":
        identity["official_test_labels_opened"] = True
    else:
        identity["ordered_query_identity"].reverse()
    with pytest.raises((TypeError, Stage2ConfirmatoryContractError)):
        validate_confirmatory_identity(identity)


def test_identity_external_sha_rejects_self_consistent_replacement(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    output = tmp_path / "identity.json"
    materialize.main(_identity_args(tmp_path, layout, output))
    trusted_sha = sha256_bytes(output.read_bytes())
    identity = json.loads(output.read_text(encoding="utf-8"))
    identity["ordered_query_identity"][0]["canonical_id"] = "replacement"
    identity["ordered_query_identity_sha256"] = sha256_bytes(
        freeze.canonical_json_bytes(identity["ordered_query_identity"])
    )
    with pytest.raises(Stage2ConfirmatoryContractError, match="external expectation"):
        verify_confirmatory_identity(
            identity,
            trusted_sha,
            artifact_bytes=pretty_json_bytes(identity),
        )


@pytest.mark.parametrize("occupied", ["json", "sidecar"])
def test_pre_open_no_replace_preflight_leaves_zero_orphan(
    tmp_path: Path, occupied: str
) -> None:
    layout = _layout(tmp_path, materialize_split=False)
    output = tmp_path / "plan-output.json"
    sidecar = output.with_name(output.name + ".sha256")
    occupied_path = output if occupied == "json" else sidecar
    occupied_path.write_bytes(b"keep")
    with pytest.raises(Stage2ConfirmatoryContractError, match="already exists"):
        freeze.main(_freeze_args(tmp_path, layout, output))
    assert occupied_path.read_bytes() == b"keep"
    other = sidecar if occupied == "json" else output
    assert not other.exists()


def test_identity_bundle_link_failure_rolls_back_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    output = tmp_path / "identity.json"
    sidecar = output.with_name(output.name + ".sha256")
    real_link = os.link

    def failing_link(source: object, destination: object, *args: object, **kwargs: object) -> None:
        if Path(destination) == sidecar:
            raise OSError("simulated sidecar failure")
        real_link(source, destination, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(freeze.os, "link", failing_link)
    with pytest.raises(OSError, match="simulated sidecar"):
        materialize.main(_identity_args(tmp_path, layout, output))
    assert not output.exists()
    assert not sidecar.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_post_go_rejects_count_mismatch_duplicate_ids_and_image_symlink(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    split = layout["split"]
    assert isinstance(split, Path)
    ids = layout["ids"]
    assert isinstance(ids, list)

    split.write_text("\n".join(ids[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(Stage2ConfirmatoryContractError, match="SHA-256"):
        materialize.main(_identity_args(tmp_path, layout, tmp_path / "count.json"))

    split.write_text("\n".join(ids[:-1] + [ids[0]]) + "\n", encoding="utf-8")
    plan = copy.deepcopy(layout["plan"])
    assert isinstance(plan, dict)
    plan["split_expected_sha256"] = sha256_bytes(split.read_bytes())
    layout["plan_sha"] = _write_json(layout["plan_path"], plan)  # type: ignore[arg-type]
    authorization = copy.deepcopy(layout["authorization"])
    assert isinstance(authorization, dict)
    authorization["pre_open_plan_sha256"] = layout["plan_sha"]
    layout["authorization_sha"] = _write_json(
        layout["authorization_path"], authorization  # type: ignore[arg-type]
    )
    with pytest.raises(Stage2ConfirmatoryContractError, match="duplicate IDs"):
        materialize.main(_identity_args(tmp_path, layout, tmp_path / "duplicate.json"))

    # Restore unique split and bind the new bytes, then replace one image by a
    # symlink.  O_NOFOLLOW/canonical path validation must fail closed.
    split.write_text("\n".join(ids) + "\n", encoding="utf-8")
    plan["split_expected_sha256"] = sha256_bytes(split.read_bytes())
    layout["plan_sha"] = _write_json(layout["plan_path"], plan)  # type: ignore[arg-type]
    authorization["pre_open_plan_sha256"] = layout["plan_sha"]
    layout["authorization_sha"] = _write_json(
        layout["authorization_path"], authorization  # type: ignore[arg-type]
    )
    image = tmp_path / f"datasets/NUAA-SIRST/images/{ids[0]}.png"
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(b"replacement")
    image.unlink()
    image.symlink_to(replacement)
    with pytest.raises(Stage2ConfirmatoryContractError, match="canonical"):
        materialize.main(_identity_args(tmp_path, layout, tmp_path / "symlink.json"))


def test_bundle_write_failure_leaves_zero_target_or_tempfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.json"
    sidecar = tmp_path / "output.json.sha256"

    def failing_fsync(descriptor: int) -> None:
        raise OSError(f"simulated fsync failure on {descriptor}")

    monkeypatch.setattr(freeze.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="simulated fsync"):
        freeze.transactional_publish_bundle(
            {output: b"{}\n", sidecar: b"0" * 64 + b"  output.json\n"}
        )
    assert not output.exists()
    assert not sidecar.exists()
    assert not list(tmp_path.glob(".*.tmp"))

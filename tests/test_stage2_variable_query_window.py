from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import data_ext.stage2_score_manifest as score_manifest
import data_ext.stage2_variable_query_window as variable_window
from data_ext.stage2_score_manifest import SOURCE_DIAGNOSTIC_VALIDATION
from data_ext.stage2_variable_query_window import (
    ARTIFACT_TYPE,
    BOUND_INPUT_NAMES,
    IDENTITY_BOUNDARIES,
    SCHEMA_VERSION,
    Stage2VariableQueryWindowContractError,
    VerifiedStage2VariableQueryWindow,
    assert_verified_stage2_variable_query_window,
    build_stage2_variable_query_window_payload,
    canonical_json_bytes,
    stage2_variable_query_record_identity,
    validate_stage2_variable_query_window_payload,
    verify_stage2_variable_query_window,
)
from rc.stage2_variable_query_geometry import (
    CONTEXT_SIZE,
    MINIMUM_QUERY_SIZE,
    build_stage2_variable_query_geometry,
)


GOLDEN_QUERY_SIZES = {
    43: (29,),
    85: (29, 28),
    127: (29, 28, 28),
    159: (39, 39, 39),
    254: (29, 29, 28, 28, 28, 28),
    255: (29, 29, 29, 28, 28, 28),
    319: (32, 32, 32, 32, 31, 31, 31),
}
OUTER_FOLD = "outer_leave_nuaa_sirst"
OUTER_TARGET = "NUAA-SIRST"
DOMAIN = "NUDT-SIRST"
SOURCE_ROLE = "detector_diagnostic"
EPISODE_ROLE = SOURCE_DIAGNOSTIC_VALIDATION


def _sha(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _records(count: int) -> list[dict[str, Any]]:
    return [
        {
            "canonical_id": f"{DOMAIN}::image_{index:04d}",
            "image_id": f"image_{index:04d}",
            "original_image_path": (
                f"datasets/{DOMAIN}/images/image_{index:04d}.png"
            ),
            "original_image_sha256": _sha(f"image-content-{index}"),
            "exclusion_group_id": f"exclusion-{index:04d}",
            "near_duplicate_cluster_id_or_unique_sentinel": (
                f"unique-{index:04d}"
            ),
            "source_role_record_index": index,
            "source_role": SOURCE_ROLE,
            "outer_fold_id": OUTER_FOLD,
            "episode_role": EPISODE_ROLE,
            "oof_fold_index": None,
        }
        for index in range(count)
    ]


def _write_binding_files(
    root: Path,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    directory = root / "contracts"
    directory.mkdir(parents=True, exist_ok=True)

    def binding(name: str) -> dict[str, str]:
        relative = f"contracts/{name}.json"
        content = json.dumps(
            {"artifact_type": "synthetic_result_free_metadata", "name": name},
            sort_keys=True,
        ).encode("utf-8")
        (root / relative).write_bytes(content)
        return {"path": relative, "sha256": _sha(content)}

    return (
        binding("ordered_role_binding"),
        {name: binding(name) for name in sorted(BOUND_INPUT_NAMES)},
    )


def _dummy_bindings() -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    role = {
        "path": "contracts/ordered_role_binding.json",
        "sha256": _sha("role"),
    }
    inputs = {
        name: {"path": f"contracts/{name}.json", "sha256": _sha(name)}
        for name in sorted(BOUND_INPUT_NAMES)
    }
    return role, inputs


def _payload(
    count: int,
    *,
    root: Path | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    role, inputs = _write_binding_files(root) if root is not None else _dummy_bindings()
    return build_stage2_variable_query_window_payload(
        ordered_role_records=_records(count) if records is None else records,
        outer_fold_id=OUTER_FOLD,
        outer_target_domain=OUTER_TARGET,
        domain=DOMAIN,
        source_role=SOURCE_ROLE,
        episode_role=EPISODE_ROLE,
        oof_fold_index=None,
        role_binding=role,
        bound_inputs=inputs,
    )


def _write_manifest(
    root: Path, payload: dict[str, Any], name: str = "windows.json"
) -> tuple[Path, str]:
    path = root / "contracts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload)
    path.write_bytes(data)
    return path, _sha(data)


@pytest.mark.parametrize(
    ("record_count", "expected_query_sizes"), GOLDEN_QUERY_SIZES.items()
)
def test_golden_variable_query_payload_replays_geometry_and_verifies_stably(
    tmp_path: Path,
    record_count: int,
    expected_query_sizes: tuple[int, ...],
) -> None:
    payload = _payload(record_count, root=tmp_path)
    path, digest = _write_manifest(tmp_path, payload)
    verified = verify_stage2_variable_query_window(
        path, digest, repository_root=tmp_path
    )
    geometry = build_stage2_variable_query_geometry(record_count)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["artifact_type"] == ARTIFACT_TYPE
    assert payload["geometry"] == geometry
    assert payload["complete_window_count"] == record_count // 42
    assert payload["ordered_role_record_count"] == record_count
    assert payload["window_record_count"] == record_count
    assert payload["unused_suffix"] == {
        "record_count": 0,
        "records": [],
        "all_ordered_role_records_consumed_once": True,
    }
    assert tuple(window["query_size"] for window in payload["windows"]) == (
        expected_query_sizes
    )
    assert len(verified.ordered_records) == record_count
    assert verified.path == path
    assert verified.manifest_sha256 == digest
    assert assert_verified_stage2_variable_query_window(verified) is verified

    flattened: list[dict[str, Any]] = []
    for window, expected in zip(
        payload["windows"], geometry["windows"], strict=True
    ):
        for field in (
            "window_index",
            "context_start",
            "context_stop",
            "query_start",
            "query_stop",
            "context_size",
            "query_size",
        ):
            assert window[field] == expected[field]
        assert len(window["context_records"]) == CONTEXT_SIZE
        assert len(window["query_records"]) == window["query_size"]
        assert window["query_size"] >= MINIMUM_QUERY_SIZE
        flattened.extend(window["context_records"])
        flattened.extend(window["query_records"])
    assert [record["source_role_record_index"] for record in flattened] == list(
        range(record_count)
    )
    assert stage2_variable_query_record_identity(flattened) == (
        stage2_variable_query_record_identity(list(verified.ordered_records))
    )


def test_payload_is_directly_compatible_with_score_manifest_selection_flattening() -> None:
    payload = _payload(85)

    score_manifest._verify_development_selection_source(payload)
    flattened = score_manifest._selection_records(
        payload,
        role=SOURCE_DIAGNOSTIC_VALIDATION,
        oof_fold_index=None,
    )

    assert len(flattened) == 85
    assert [record["source_role_record_index"] for record in flattened] == list(
        range(85)
    )


def test_builder_is_pure_and_never_attempts_external_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise AssertionError("pure builder attempted filesystem access")

    monkeypatch.setattr(variable_window, "_stable_file_bytes", forbidden)
    payload = _payload(43)

    assert payload["guardrails"]["result_free"] is True
    assert payload["guardrails"]["mask_or_label_files_opened"] is False
    assert (
        payload["guardrails"][
            "predictions_scores_checkpoints_or_metrics_opened"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("context_start", 1),
        ("context_stop", 13),
        ("query_start", 15),
        ("query_stop", 42),
        ("context_size", 13),
        ("query_size", 28),
    ],
)
def test_validator_rejects_every_window_span_or_query_size_tamper(
    field: str, replacement: int
) -> None:
    payload = _payload(85)
    payload["windows"][0][field] = replacement

    with pytest.raises(Stage2VariableQueryWindowContractError, match="geometry replay"):
        validate_stage2_variable_query_window_payload(payload)


def test_validator_rejects_geometry_tamper_even_when_window_rows_are_unchanged() -> None:
    payload = _payload(85)
    payload["geometry"]["windows"][0]["query_stop"] -= 1

    with pytest.raises(Stage2VariableQueryWindowContractError, match="geometry"):
        validate_stage2_variable_query_window_payload(payload)


def test_validator_rejects_window_or_record_reordering_and_lost_tail_record() -> None:
    reversed_windows = _payload(85)
    reversed_windows["windows"].reverse()
    with pytest.raises(Stage2VariableQueryWindowContractError):
        validate_stage2_variable_query_window_payload(reversed_windows)

    reversed_records = _payload(43)
    context = reversed_records["windows"][0]["context_records"]
    context[0], context[1] = context[1], context[0]
    with pytest.raises(Stage2VariableQueryWindowContractError, match="out of strict"):
        validate_stage2_variable_query_window_payload(reversed_records)

    lost_tail = _payload(85)
    lost_tail["windows"][-1]["query_records"].pop()
    with pytest.raises(Stage2VariableQueryWindowContractError, match="query_size"):
        validate_stage2_variable_query_window_payload(lost_tail)


@pytest.mark.parametrize("boundary", IDENTITY_BOUNDARIES)
def test_builder_rejects_context_query_overlap_on_every_identity_boundary(
    boundary: str,
) -> None:
    records = _records(43)
    records[CONTEXT_SIZE][boundary] = records[0][boundary]
    if boundary == "canonical_id":
        records[CONTEXT_SIZE]["image_id"] = records[0]["image_id"]

    with pytest.raises(Stage2VariableQueryWindowContractError, match=f"duplicate {boundary}"):
        _payload(43, records=records)


@pytest.mark.parametrize("boundary", IDENTITY_BOUNDARIES)
def test_builder_rejects_identity_reuse_across_distant_windows(boundary: str) -> None:
    records = _records(85)
    records[-1][boundary] = records[0][boundary]
    if boundary == "canonical_id":
        records[-1]["image_id"] = records[0]["image_id"]

    with pytest.raises(Stage2VariableQueryWindowContractError, match=f"duplicate {boundary}"):
        _payload(85, records=records)


def test_strict_field_closure_and_exact_bool_int_types_fail_closed() -> None:
    top_extra = _payload(43)
    top_extra["query_label_count"] = 29
    with pytest.raises(Stage2VariableQueryWindowContractError, match="extra"):
        validate_stage2_variable_query_window_payload(top_extra)

    window_extra = _payload(43)
    window_extra["windows"][0]["query_labels"] = []
    with pytest.raises(Stage2VariableQueryWindowContractError, match="extra"):
        validate_stage2_variable_query_window_payload(window_extra)

    record_missing = _payload(43)
    record_missing["windows"][0]["query_records"][0].pop("original_image_sha256")
    with pytest.raises(Stage2VariableQueryWindowContractError, match="missing"):
        validate_stage2_variable_query_window_payload(record_missing)

    bool_as_int = _payload(43)
    bool_as_int["ordered_role_record_count"] = True
    with pytest.raises(Stage2VariableQueryWindowContractError, match="exact integer"):
        validate_stage2_variable_query_window_payload(bool_as_int)

    int_as_bool = _payload(43)
    int_as_bool["guardrails"]["result_free"] = 1
    with pytest.raises(Stage2VariableQueryWindowContractError, match="exactly true"):
        validate_stage2_variable_query_window_payload(int_as_bool)

    nested_bool_as_int = _payload(43)
    nested_bool_as_int["windows"][0]["query_size"] = True
    with pytest.raises(Stage2VariableQueryWindowContractError, match="exact integer"):
        validate_stage2_variable_query_window_payload(nested_bool_as_int)


def test_no_unused_suffix_is_an_exact_closed_contract() -> None:
    count = _payload(43)
    count["unused_suffix"]["record_count"] = 1
    with pytest.raises(Stage2VariableQueryWindowContractError, match="unused suffix"):
        validate_stage2_variable_query_window_payload(count)

    records = _payload(43)
    records["unused_suffix"]["records"] = [records["windows"][0]["query_records"][-1]]
    with pytest.raises(Stage2VariableQueryWindowContractError, match="empty list"):
        validate_stage2_variable_query_window_payload(records)

    flag = _payload(43)
    flag["unused_suffix"]["all_ordered_role_records_consumed_once"] = 1
    with pytest.raises(Stage2VariableQueryWindowContractError, match="exactly true"):
        validate_stage2_variable_query_window_payload(flag)


def test_verifier_rejects_manifest_or_external_binding_sha_mismatch(
    tmp_path: Path,
) -> None:
    payload = _payload(43, root=tmp_path)
    path, digest = _write_manifest(tmp_path, payload)

    with pytest.raises(Stage2VariableQueryWindowContractError, match="manifest SHA"):
        verify_stage2_variable_query_window(
            path, _sha("wrong-manifest"), repository_root=tmp_path
        )

    bad_binding = copy.deepcopy(payload)
    bad_binding["role_binding"]["sha256"] = _sha("wrong-role-binding")
    bad_path, bad_digest = _write_manifest(
        tmp_path, bad_binding, name="bad-binding-windows.json"
    )
    with pytest.raises(Stage2VariableQueryWindowContractError, match="role_binding SHA"):
        verify_stage2_variable_query_window(
            bad_path, bad_digest, repository_root=tmp_path
        )
    assert digest == _sha(path.read_bytes())


def test_verifier_rejects_manifest_and_binding_symlinks(tmp_path: Path) -> None:
    payload = _payload(43, root=tmp_path)
    real_path, digest = _write_manifest(tmp_path, payload)
    manifest_link = real_path.with_name("window-link.json")
    manifest_link.symlink_to(real_path.name)
    with pytest.raises(Stage2VariableQueryWindowContractError, match="symlink"):
        verify_stage2_variable_query_window(
            manifest_link, digest, repository_root=tmp_path
        )

    role_target = tmp_path / payload["role_binding"]["path"]
    role_link = role_target.with_name("role-link.json")
    role_link.symlink_to(role_target.name)
    linked_payload = copy.deepcopy(payload)
    linked_payload["role_binding"]["path"] = role_link.relative_to(tmp_path).as_posix()
    linked_path, linked_digest = _write_manifest(
        tmp_path, linked_payload, name="linked-binding-windows.json"
    )
    with pytest.raises(Stage2VariableQueryWindowContractError, match="symlink"):
        verify_stage2_variable_query_window(
            linked_path, linked_digest, repository_root=tmp_path
        )


def test_verifier_final_recheck_detects_manifest_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(43, root=tmp_path)
    path, digest = _write_manifest(tmp_path, payload)
    original = variable_window._stable_file_bytes

    def changed_on_final(candidate: Path, name: str) -> bytes:
        data = original(candidate, name)
        return data + b" " if name == "window manifest final recheck" else data

    monkeypatch.setattr(variable_window, "_stable_file_bytes", changed_on_final)
    with pytest.raises(RuntimeError, match="changed"):
        verify_stage2_variable_query_window(
            path, digest, repository_root=tmp_path
        )


def test_verifier_reads_only_manifest_and_declared_result_free_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(43, root=tmp_path)
    path, digest = _write_manifest(tmp_path, payload)
    original = variable_window._stable_file_bytes
    read_names: list[str] = []

    def record_reads(candidate: Path, name: str) -> bytes:
        read_names.append(name)
        return original(candidate, name)

    monkeypatch.setattr(variable_window, "_stable_file_bytes", record_reads)
    verify_stage2_variable_query_window(path, digest, repository_root=tmp_path)

    assert read_names == [
        "window manifest",
        "role_binding",
        *[f"bound_inputs.{name}" for name in sorted(BOUND_INPUT_NAMES)],
        "window manifest final recheck",
    ]
    assert not any(
        token in name.casefold()
        for name in read_names
        for token in ("label", "mask", "score", "checkpoint", "metric")
    )


def test_verified_capability_is_deeply_immutable_and_cannot_be_forged(
    tmp_path: Path,
) -> None:
    payload = _payload(43, root=tmp_path)
    path, digest = _write_manifest(tmp_path, payload)
    verified = verify_stage2_variable_query_window(
        path, digest, repository_root=tmp_path
    )

    with pytest.raises(TypeError):
        verified.payload["window_record_count"] = 1
    with pytest.raises(TypeError):
        verified.windows[0]["query_size"] = 28
    with pytest.raises(TypeError):
        verified.ordered_records[0]["canonical_id"] = "forged"
    with pytest.raises(TypeError, match="verifier-issued only"):
        VerifiedStage2VariableQueryWindow()

    forged = object.__new__(VerifiedStage2VariableQueryWindow)
    with pytest.raises(TypeError, match="verifier-issued"):
        assert_verified_stage2_variable_query_window(forged)


def test_builder_rejects_too_few_records_and_sensitive_external_paths() -> None:
    with pytest.raises(Stage2VariableQueryWindowContractError, match="C14/Qmin28"):
        _payload(41)

    role, inputs = _dummy_bindings()
    role["path"] = "contracts/query_scores.json"
    with pytest.raises(Stage2VariableQueryWindowContractError, match="score"):
        build_stage2_variable_query_window_payload(
            ordered_role_records=_records(43),
            outer_fold_id=OUTER_FOLD,
            outer_target_domain=OUTER_TARGET,
            domain=DOMAIN,
            source_role=SOURCE_ROLE,
            episode_role=EPISODE_ROLE,
            oof_fold_index=None,
            role_binding=role,
            bound_inputs=inputs,
        )

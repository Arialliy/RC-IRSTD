#!/usr/bin/env python3
"""Materialize the post-S2_DGO confirmatory identity (W12 schema v2).

The official split and image bytes are unreachable until a caller supplies an
externally hash-bound, exact-schema S2_DGO=GO opening authorization that binds
the already frozen pre-open plan.  This program reads IDs and image bytes only;
it has no label/mask argument, does not import inference code, and records that
labels, masks, predictions, metrics, and threshold decisions remain absent.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from scripts.freeze_stage2_confirmatory_plan import (
    B2_AUTHORIZATION_SHA256,
    CONTEXT_RULE,
    CONTEXT_SIZE,
    QUERY_RULE,
    Stage2ConfirmatoryContractError,
    _assert_exact_keys,
    _canonical_repository_root,
    _existing_artifact_within_root,
    _new_output_path,
    _nonempty_string,
    _strict_bool,
    _strict_int,
    canonical_json_bytes,
    load_verified_pre_open_plan,
    parse_json_bytes,
    pretty_json_bytes,
    sha256_bytes,
    stable_read_bytes,
    transactional_publish_bundle,
    validate_repository_relative_path,
    validate_sha256,
    verify_pre_open_plan,
)


IDENTITY_SCHEMA_VERSION = "rc-irstd.stage2-confirmatory-identity.v2"
DGO_AUTHORIZATION_SCHEMA_VERSION = "rc-irstd.stage2-dgo-opening-authorization.v1"

IMAGE_DIRECTORIES = {
    "nuaa-sirst": "datasets/NUAA-SIRST/images",
    "nudt-sirst": "datasets/NUDT-SIRST/images",
    "irstd-1k": "datasets/IRSTD-1K/images",
}
IMAGE_SUFFIX = ".png"


def validate_dgo_opening_authorization(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact future authorization before constructing any split path."""

    if type(payload) is not dict:
        raise TypeError("S2_DGO opening authorization must be an exact JSON object")
    required = {
        "schema_version",
        "gate_id",
        "decision",
        "development_gate_result_sha256",
        "pre_open_plan_sha256",
        "confirmatory_identity_materialization_authorized",
        "official_test_split_open_authorized",
        "official_test_image_open_authorized",
        "official_test_label_open_authorized",
        "result_based_rerun_authorized",
        "official_test_accessed",
    }
    _assert_exact_keys(payload, required, "S2_DGO opening authorization")
    expected_strings = {
        "schema_version": DGO_AUTHORIZATION_SCHEMA_VERSION,
        "gate_id": "S2_DGO",
        "decision": "GO",
    }
    for field, expected in expected_strings.items():
        if _nonempty_string(payload[field], f"authorization.{field}") != expected:
            raise Stage2ConfirmatoryContractError(
                f"authorization.{field} must be exactly {expected!r}"
            )
    expected_booleans = {
        "confirmatory_identity_materialization_authorized": True,
        "official_test_split_open_authorized": True,
        "official_test_image_open_authorized": True,
        "official_test_label_open_authorized": False,
        "result_based_rerun_authorized": False,
        "official_test_accessed": False,
    }
    for field, expected in expected_booleans.items():
        if _strict_bool(payload[field], f"authorization.{field}") is not expected:
            raise Stage2ConfirmatoryContractError(
                f"authorization.{field} must be exact {str(expected).lower()}"
            )
    result = dict(payload)
    result["development_gate_result_sha256"] = validate_sha256(
        payload["development_gate_result_sha256"],
        "authorization.development_gate_result_sha256",
    )
    result["pre_open_plan_sha256"] = validate_sha256(
        payload["pre_open_plan_sha256"], "authorization.pre_open_plan_sha256"
    )
    return result


def verify_dgo_opening_authorization(
    payload: Mapping[str, Any],
    expected_artifact_sha256: str,
    *,
    artifact_bytes: bytes | None = None,
) -> dict[str, Any]:
    expected = validate_sha256(
        expected_artifact_sha256, "expected_s2_dgo_authorization_sha256"
    )
    data = pretty_json_bytes(payload) if artifact_bytes is None else artifact_bytes
    if not isinstance(data, bytes):
        raise TypeError("artifact_bytes must be bytes")
    parsed = parse_json_bytes(data, name="S2_DGO opening authorization")
    if canonical_json_bytes(parsed) != canonical_json_bytes(payload):
        raise Stage2ConfirmatoryContractError(
            "authorization mapping differs from exact artifact bytes"
        )
    if sha256_bytes(data) != expected:
        raise Stage2ConfirmatoryContractError(
            "S2_DGO authorization SHA-256 differs from external expectation"
        )
    return validate_dgo_opening_authorization(payload)


def load_verified_dgo_opening_authorization(
    path: str | Path, expected_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    data = stable_read_bytes(
        path,
        expected_sha256,
        repository_root,
        name="s2_dgo_authorization",
    )
    payload = parse_json_bytes(data, name="S2_DGO opening authorization")
    return verify_dgo_opening_authorization(
        payload, expected_sha256, artifact_bytes=data
    )


def _stable_read_unbound_file(
    path: Path, repository_root: Path, *, name: str
) -> tuple[bytes, str]:
    """Read a newly identified image twice and bind the stable observed bytes."""

    checked = _existing_artifact_within_root(path, repository_root, name=name)
    descriptor = os.open(checked, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Stage2ConfirmatoryContractError(f"{name} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        data_before = b"".join(chunks)
        observed_sha = sha256_bytes(data_before)
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        data_after = b"".join(chunks)
        after = os.fstat(descriptor)
        path_after = os.stat(checked, follow_symlinks=False)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        path_identity = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if before_identity != after_identity or before_identity != path_identity:
            raise Stage2ConfirmatoryContractError(f"{name} changed during consumption")
        if data_before != data_after or sha256_bytes(data_after) != observed_sha:
            raise Stage2ConfirmatoryContractError(f"{name} bytes changed during consumption")
        return data_before, observed_sha
    finally:
        os.close(descriptor)


def _parse_split_ids(data: bytes, *, expected_count: int) -> list[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Stage2ConfirmatoryContractError("official split is not UTF-8") from error
    if "\r" in text or "\x00" in text:
        raise Stage2ConfirmatoryContractError(
            "official split must use LF text and contain no NUL"
        )
    lines = text.splitlines()
    if len(lines) != expected_count:
        raise Stage2ConfirmatoryContractError(
            f"official split record count mismatch: {len(lines)} != {expected_count}"
        )
    result: list[str] = []
    for index, line in enumerate(lines):
        canonical_id = _nonempty_string(line, f"split[{index}]")
        if "/" in canonical_id or "\\" in canonical_id or canonical_id in {".", ".."}:
            raise Stage2ConfirmatoryContractError(
                f"split[{index}] must be one basename identity, not a path"
            )
        result.append(canonical_id)
    if len(set(result)) != len(result):
        raise Stage2ConfirmatoryContractError("official split contains duplicate IDs")
    return result


def _image_filename(canonical_id: str) -> str:
    if canonical_id.endswith(IMAGE_SUFFIX):
        stem = canonical_id[: -len(IMAGE_SUFFIX)]
        if not stem or "." in stem and stem in {".", ".."}:
            raise Stage2ConfirmatoryContractError("invalid image identity")
        return canonical_id
    return f"{canonical_id}{IMAGE_SUFFIX}"


def _identity_row(
    *, position: int, canonical_id: str, image_path: str, image_sha256: str
) -> dict[str, Any]:
    return {
        "position": position,
        "canonical_id": canonical_id,
        "image_repository_relative_path": image_path,
        "original_image_sha256": image_sha256,
    }


def materialize_confirmatory_identity(
    *,
    repository_root: str | Path,
    pre_open_plan: Mapping[str, Any],
    pre_open_plan_sha256: str,
    pre_open_plan_artifact_bytes: bytes,
    dgo_authorization: Mapping[str, Any],
    dgo_authorization_sha256: str,
    dgo_authorization_artifact_bytes: bytes,
) -> dict[str, Any]:
    """Open the split/images only after both externally bound artifacts verify."""

    root = _canonical_repository_root(repository_root)
    plan = verify_pre_open_plan(
        pre_open_plan,
        pre_open_plan_sha256,
        artifact_bytes=pre_open_plan_artifact_bytes,
    )
    authorization = verify_dgo_opening_authorization(
        dgo_authorization,
        dgo_authorization_sha256,
        artifact_bytes=dgo_authorization_artifact_bytes,
    )
    if authorization["pre_open_plan_sha256"] != pre_open_plan_sha256:
        raise Stage2ConfirmatoryContractError(
            "S2_DGO authorization does not bind the supplied pre-open plan"
        )

    # This is intentionally the first construction/access of the official
    # split path in the post-GO code path.
    split_relative = validate_repository_relative_path(
        plan["split_repository_relative_path"], name="plan.split path"
    )
    split_path = root.joinpath(*split_relative.split("/"))
    split_bytes = stable_read_bytes(
        split_path,
        plan["split_expected_sha256"],
        root,
        name="official_test_split",
    )
    ordered_ids = _parse_split_ids(
        split_bytes, expected_count=plan["split_expected_record_count"]
    )
    if len(ordered_ids) <= CONTEXT_SIZE:
        raise Stage2ConfirmatoryContractError(
            "official split must contain a nonempty all_remaining_suffix query"
        )

    dataset = plan["target_dataset"]
    image_directory = IMAGE_DIRECTORIES.get(dataset)
    if image_directory is None:
        raise Stage2ConfirmatoryContractError("unsupported confirmatory target dataset")
    rows: list[dict[str, Any]] = []
    seen_image_paths: set[str] = set()
    seen_image_shas: set[str] = set()
    for position, canonical_id in enumerate(ordered_ids):
        filename = _image_filename(canonical_id)
        image_relative = f"{image_directory}/{filename}"
        validate_repository_relative_path(
            image_relative, name=f"image[{position}].repository_relative_path"
        )
        if image_relative in seen_image_paths:
            raise Stage2ConfirmatoryContractError("duplicate derived image path")
        image_path = root.joinpath(*image_relative.split("/"))
        _, image_sha = _stable_read_unbound_file(
            image_path, root, name=f"official_test_image_{position}"
        )
        if image_sha in seen_image_shas:
            raise Stage2ConfirmatoryContractError(
                "official split contains duplicate original-image content"
            )
        seen_image_paths.add(image_relative)
        seen_image_shas.add(image_sha)
        rows.append(
            _identity_row(
                position=position,
                canonical_id=canonical_id,
                image_path=image_relative,
                image_sha256=image_sha,
            )
        )

    context_rows = rows[:CONTEXT_SIZE]
    query_rows = rows[CONTEXT_SIZE:]
    result = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "b2_authorization_sha256": B2_AUTHORIZATION_SHA256,
        "pre_open_plan_sha256": validate_sha256(
            pre_open_plan_sha256, "pre_open_plan_sha256"
        ),
        "s2_dgo_go_authorization_sha256": validate_sha256(
            dgo_authorization_sha256, "s2_dgo_go_authorization_sha256"
        ),
        "development_gate_result_sha256": authorization[
            "development_gate_result_sha256"
        ],
        "target_dataset": dataset,
        "split_repository_relative_path": split_relative,
        "split_sha256": plan["split_expected_sha256"],
        "split_record_count": len(rows),
        "context_size": CONTEXT_SIZE,
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
    return validate_confirmatory_identity(result)


def _validate_identity_rows(rows: Any, *, name: str, start: int) -> list[dict[str, Any]]:
    if type(rows) is not list:
        raise TypeError(f"identity.{name} must be an exact JSON array")
    if not rows:
        raise Stage2ConfirmatoryContractError(f"identity.{name} must be nonempty")
    result: list[dict[str, Any]] = []
    required = {
        "position",
        "canonical_id",
        "image_repository_relative_path",
        "original_image_sha256",
    }
    for offset, row in enumerate(rows):
        row_name = f"identity.{name}[{offset}]"
        if type(row) is not dict:
            raise TypeError(f"{row_name} must be an exact JSON object")
        _assert_exact_keys(row, required, row_name)
        position = _strict_int(row["position"], f"{row_name}.position", minimum=0)
        if position != start + offset:
            raise Stage2ConfirmatoryContractError(f"{row_name}.position is not contiguous")
        canonical_id = _nonempty_string(row["canonical_id"], f"{row_name}.canonical_id")
        if "/" in canonical_id or "\\" in canonical_id:
            raise Stage2ConfirmatoryContractError(f"{row_name}.canonical_id is a path")
        result.append(
            {
                "position": position,
                "canonical_id": canonical_id,
                "image_repository_relative_path": validate_repository_relative_path(
                    row["image_repository_relative_path"],
                    name=f"{row_name}.image_repository_relative_path",
                ),
                "original_image_sha256": validate_sha256(
                    row["original_image_sha256"],
                    f"{row_name}.original_image_sha256",
                ),
            }
        )
    return result


def validate_confirmatory_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("confirmatory identity must be an exact JSON object")
    required = {
        "schema_version",
        "b2_authorization_sha256",
        "pre_open_plan_sha256",
        "s2_dgo_go_authorization_sha256",
        "development_gate_result_sha256",
        "target_dataset",
        "split_repository_relative_path",
        "split_sha256",
        "split_record_count",
        "context_size",
        "context_rule",
        "query_rule",
        "ordered_context_identity",
        "ordered_context_identity_sha256",
        "ordered_query_identity",
        "ordered_query_identity_sha256",
        "official_test_accessed",
        "official_test_split_opened",
        "official_test_images_opened",
        "official_test_masks_opened",
        "official_test_labels_opened",
        "inference_run",
        "metric_computed",
        "threshold_decision_sealed",
        "result_based_rerun",
    }
    _assert_exact_keys(payload, required, "confirmatory identity")
    if payload["schema_version"] != IDENTITY_SCHEMA_VERSION:
        raise Stage2ConfirmatoryContractError("unsupported confirmatory identity schema")
    if payload["b2_authorization_sha256"] != B2_AUTHORIZATION_SHA256:
        raise Stage2ConfirmatoryContractError("identity B2 authorization binding changed")
    for field in (
        "b2_authorization_sha256",
        "pre_open_plan_sha256",
        "s2_dgo_go_authorization_sha256",
        "development_gate_result_sha256",
        "split_sha256",
        "ordered_context_identity_sha256",
        "ordered_query_identity_sha256",
    ):
        validate_sha256(payload[field], f"identity.{field}")
    dataset = _nonempty_string(payload["target_dataset"], "identity.target_dataset")
    if dataset not in IMAGE_DIRECTORIES:
        raise Stage2ConfirmatoryContractError("identity target dataset is not frozen")
    split_path = validate_repository_relative_path(
        payload["split_repository_relative_path"],
        name="identity.split_repository_relative_path",
    )
    if _strict_int(payload["context_size"], "identity.context_size", minimum=1) != CONTEXT_SIZE:
        raise Stage2ConfirmatoryContractError("identity context_size must be 14")
    if payload["context_rule"] != CONTEXT_RULE or payload["query_rule"] != QUERY_RULE:
        raise Stage2ConfirmatoryContractError("identity partition rule changed")
    context = _validate_identity_rows(
        payload["ordered_context_identity"], name="ordered_context_identity", start=0
    )
    if len(context) != CONTEXT_SIZE:
        raise Stage2ConfirmatoryContractError("identity must have exactly 14 context rows")
    query = _validate_identity_rows(
        payload["ordered_query_identity"],
        name="ordered_query_identity",
        start=CONTEXT_SIZE,
    )
    count = _strict_int(payload["split_record_count"], "identity.split_record_count", minimum=15)
    if count != len(context) + len(query):
        raise Stage2ConfirmatoryContractError("identity split_record_count mismatch")
    all_rows = context + query
    image_directory = IMAGE_DIRECTORIES[dataset]
    for index, row in enumerate(all_rows):
        expected_image_path = (
            f"{image_directory}/{_image_filename(row['canonical_id'])}"
        )
        if row["image_repository_relative_path"] != expected_image_path:
            raise Stage2ConfirmatoryContractError(
                f"identity image path {index} does not match canonical ID"
            )
    if len({row["canonical_id"] for row in all_rows}) != len(all_rows):
        raise Stage2ConfirmatoryContractError("identity contains duplicate canonical IDs")
    if len({row["image_repository_relative_path"] for row in all_rows}) != len(all_rows):
        raise Stage2ConfirmatoryContractError("identity contains duplicate image paths")
    if len({row["original_image_sha256"] for row in all_rows}) != len(all_rows):
        raise Stage2ConfirmatoryContractError("identity contains duplicate image hashes")
    if sha256_bytes(canonical_json_bytes(context)) != payload[
        "ordered_context_identity_sha256"
    ]:
        raise Stage2ConfirmatoryContractError("context identity SHA-256 mismatch")
    if sha256_bytes(canonical_json_bytes(query)) != payload[
        "ordered_query_identity_sha256"
    ]:
        raise Stage2ConfirmatoryContractError("query identity SHA-256 mismatch")
    expected_booleans = {
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
    for field, expected in expected_booleans.items():
        if _strict_bool(payload[field], f"identity.{field}") is not expected:
            raise Stage2ConfirmatoryContractError(
                f"identity.{field} must be exact {str(expected).lower()}"
            )
    result = dict(payload)
    result["split_repository_relative_path"] = split_path
    result["ordered_context_identity"] = context
    result["ordered_query_identity"] = query
    return result


def verify_confirmatory_identity(
    payload: Mapping[str, Any],
    expected_artifact_sha256: str,
    *,
    artifact_bytes: bytes | None = None,
) -> dict[str, Any]:
    expected = validate_sha256(
        expected_artifact_sha256, "expected_confirmatory_identity_sha256"
    )
    data = pretty_json_bytes(payload) if artifact_bytes is None else artifact_bytes
    if not isinstance(data, bytes):
        raise TypeError("artifact_bytes must be bytes")
    parsed = parse_json_bytes(data, name="confirmatory identity")
    if canonical_json_bytes(parsed) != canonical_json_bytes(payload):
        raise Stage2ConfirmatoryContractError(
            "confirmatory identity mapping differs from artifact bytes"
        )
    if sha256_bytes(data) != expected:
        raise Stage2ConfirmatoryContractError(
            "confirmatory identity SHA-256 differs from external expectation"
        )
    return validate_confirmatory_identity(payload)


def load_verified_confirmatory_identity(
    path: str | Path, expected_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    data = stable_read_bytes(
        path,
        expected_sha256,
        repository_root,
        name="confirmatory_identity",
    )
    payload = parse_json_bytes(data, name="confirmatory identity")
    return verify_confirmatory_identity(payload, expected_sha256, artifact_bytes=data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--pre-open-plan", required=True)
    parser.add_argument("--pre-open-plan-sha256", required=True)
    parser.add_argument("--s2-dgo-authorization", required=True)
    parser.add_argument("--s2-dgo-authorization-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _canonical_repository_root(args.repository_root)
    output = _new_output_path(args.output, root, name="output")
    sidecar = _new_output_path(
        output.with_name(output.name + ".sha256"), root, name="output sidecar"
    )

    # Both trusted artifacts are loaded and fully verified before the split
    # repository-relative path is joined to the repository root.
    plan_bytes = stable_read_bytes(
        args.pre_open_plan,
        args.pre_open_plan_sha256,
        root,
        name="pre_open_plan",
    )
    plan = parse_json_bytes(plan_bytes, name="pre-open plan")
    verify_pre_open_plan(
        plan, args.pre_open_plan_sha256, artifact_bytes=plan_bytes
    )
    authorization_bytes = stable_read_bytes(
        args.s2_dgo_authorization,
        args.s2_dgo_authorization_sha256,
        root,
        name="s2_dgo_authorization",
    )
    authorization = parse_json_bytes(
        authorization_bytes, name="S2_DGO opening authorization"
    )
    verify_dgo_opening_authorization(
        authorization,
        args.s2_dgo_authorization_sha256,
        artifact_bytes=authorization_bytes,
    )

    identity = materialize_confirmatory_identity(
        repository_root=root,
        pre_open_plan=plan,
        pre_open_plan_sha256=args.pre_open_plan_sha256,
        pre_open_plan_artifact_bytes=plan_bytes,
        dgo_authorization=authorization,
        dgo_authorization_sha256=args.s2_dgo_authorization_sha256,
        dgo_authorization_artifact_bytes=authorization_bytes,
    )
    identity_bytes = pretty_json_bytes(identity)
    identity_sha = sha256_bytes(identity_bytes)
    verify_confirmatory_identity(
        identity, identity_sha, artifact_bytes=identity_bytes
    )

    # Reauthenticate the two authorization artifacts.  The output identity
    # itself binds the stable split and every observed image byte digest.
    if stable_read_bytes(
        args.pre_open_plan,
        args.pre_open_plan_sha256,
        root,
        name="pre_open_plan",
    ) != plan_bytes:
        raise Stage2ConfirmatoryContractError("pre-open plan changed before publication")
    if stable_read_bytes(
        args.s2_dgo_authorization,
        args.s2_dgo_authorization_sha256,
        root,
        name="s2_dgo_authorization",
    ) != authorization_bytes:
        raise Stage2ConfirmatoryContractError(
            "S2_DGO authorization changed before publication"
        )
    sidecar_bytes = f"{identity_sha}  {output.name}\n".encode("ascii")
    transactional_publish_bundle({output: identity_bytes, sidecar: sidecar_bytes})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

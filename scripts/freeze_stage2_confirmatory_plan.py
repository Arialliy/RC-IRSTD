#!/usr/bin/env python3
"""Freeze the result-free Stage-2 confirmatory pre-open plan (W12).

This program is intentionally incapable of opening the referenced official
split.  At S2_I0 it consumes only a separately hash-bound metadata JSON with
the split's repository-relative path, expected byte SHA-256, and record count.
The official split path is validated lexically against the three frozen
benchmark locations; it is never resolved, statted, opened, or parsed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any, Mapping, Sequence


PLAN_SCHEMA_VERSION = "rc-irstd.stage2-confirmatory-plan.v1"
SPLIT_METADATA_SCHEMA_VERSION = "rc-irstd.stage2-confirmatory-split-metadata.v1"

SOURCE_THAW_SHA256 = (
    "0e4f3e27026d5a2071a2c8f94f84c366d208f3789de17649aad926c64cd6b0b9"
)
WORK_BREAKDOWN_SHA256 = (
    "cc240f97aea6c99dde1e5c537a26c1b22e606b0f499ca495af71d15fa44c9d06"
)
SEMANTIC_AMENDMENT_SHA256 = (
    "c60e087116f98a3e59772792e16be389cc2961180b7a9c5de930e2b9cd9abef7"
)
B1_AUTHORIZATION_SHA256 = (
    "185b7e4cac7d7a23ca537641575a00c5e64c6a5d0783dc34f999ba402f174845"
)
B2_AUTHORIZATION_SHA256 = (
    "cc15832de4f85abfae84c4d49a5ac098cff253d0fecfa885d0d7735d3ef5aea6"
)

CONTEXT_SIZE = 14
CONTEXT_RULE = "first_C_in_frozen_split_order"
QUERY_RULE = "all_remaining_suffix"

# Exact lexical locations are part of the result-free protocol.  Looking up a
# value in this table does not perform any filesystem operation.
FROZEN_OFFICIAL_SPLIT_PATHS = {
    "datasets/NUAA-SIRST/img_idx/test_NUAA-SIRST.txt": "nuaa-sirst",
    "datasets/NUDT-SIRST/img_idx/test_NUDT-SIRST.txt": "nudt-sirst",
    "datasets/IRSTD-1K/img_idx/test_IRSTD-1K.txt": "irstd-1k",
}

_SHA256_HEX = frozenset("0123456789abcdef")


class Stage2ConfirmatoryContractError(ValueError):
    """Raised when a W12 artifact fails closed."""


def canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2ConfirmatoryContractError(
            f"payload is not finite canonical JSON: {error}"
        ) from error


def pretty_json_bytes(payload: Any) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise Stage2ConfirmatoryContractError(
            f"payload is not finite JSON: {error}"
        ) from error


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or value != value.lower() or not set(value) <= _SHA256_HEX:
        raise Stage2ConfirmatoryContractError(
            f"{name} must be a lowercase 64-character SHA-256 digest"
        )
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:  # noqa: E721 - exact JSON type is intentional
        raise TypeError(f"{name} must be an exact JSON boolean")
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise Stage2ConfirmatoryContractError(f"{name} must be >= {minimum}")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or "\x00" in value:
        raise Stage2ConfirmatoryContractError(
            f"{name} must be non-empty, trimmed, and contain no NUL"
        )
    return value


def _assert_exact_keys(payload: Mapping[str, Any], required: set[str], name: str) -> None:
    missing = required - set(payload)
    extra = set(payload) - required
    if missing or extra:
        raise Stage2ConfirmatoryContractError(
            f"{name} fields differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2ConfirmatoryContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise Stage2ConfirmatoryContractError(f"non-finite JSON number is forbidden: {value}")


def parse_json_bytes(data: bytes, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2ConfirmatoryContractError(f"invalid UTF-8 JSON in {name}: {error}") from error
    if type(payload) is not dict:
        raise TypeError(f"{name} must contain an exact JSON object")
    return payload


def validate_repository_relative_path(value: Any, *, name: str) -> str:
    """Validate a POSIX repository-relative path without touching the filesystem."""

    text = _nonempty_string(value, name)
    if "\\" in text or text.startswith("/") or text.endswith("/") or "//" in text:
        raise Stage2ConfirmatoryContractError(
            f"{name} must be a canonical POSIX repository-relative path"
        )
    raw_parts = text.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise Stage2ConfirmatoryContractError(
            f"{name} traversal or non-canonical components are forbidden"
        )
    parsed = PurePosixPath(text)
    if parsed.is_absolute() or parsed.as_posix() != text:
        raise Stage2ConfirmatoryContractError(
            f"{name} must be a canonical POSIX repository-relative path"
        )
    return text


def validate_split_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the only three pieces of official-split metadata S2_I0 may consume."""

    if type(payload) is not dict:
        raise TypeError("split metadata must be an exact JSON object")
    required = {
        "split_repository_relative_path",
        "split_expected_sha256",
        "split_expected_record_count",
    }
    _assert_exact_keys(payload, required, "split metadata")
    split_path = validate_repository_relative_path(
        payload["split_repository_relative_path"],
        name="split_metadata.split_repository_relative_path",
    )
    if split_path not in FROZEN_OFFICIAL_SPLIT_PATHS:
        raise Stage2ConfirmatoryContractError(
            "split metadata path is not one of the three frozen official-test split paths"
        )
    count = _strict_int(
        payload["split_expected_record_count"],
        "split_metadata.split_expected_record_count",
        minimum=CONTEXT_SIZE + 1,
    )
    return {
        "split_repository_relative_path": split_path,
        "split_expected_sha256": validate_sha256(
            payload["split_expected_sha256"],
            "split_metadata.split_expected_sha256",
        ),
        "split_expected_record_count": count,
    }


def build_pre_open_plan(
    split_metadata: Mapping[str, Any],
    *,
    split_metadata_sha256: str,
) -> dict[str, Any]:
    """Build a plan using metadata only; this function has no path/root argument."""

    metadata = validate_split_metadata(split_metadata)
    split_path = metadata["split_repository_relative_path"]
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_thaw_sha256": SOURCE_THAW_SHA256,
        "work_breakdown_sha256": WORK_BREAKDOWN_SHA256,
        "semantic_amendment_sha256": SEMANTIC_AMENDMENT_SHA256,
        "b1_authorization_sha256": B1_AUTHORIZATION_SHA256,
        "b2_authorization_sha256": B2_AUTHORIZATION_SHA256,
        "split_metadata_sha256": validate_sha256(
            split_metadata_sha256, "split_metadata_sha256"
        ),
        "target_dataset": FROZEN_OFFICIAL_SPLIT_PATHS[split_path],
        **metadata,
        "context_size": CONTEXT_SIZE,
        "context_rule": CONTEXT_RULE,
        "query_rule": QUERY_RULE,
        "official_test_accessed": False,
        "official_test_ids_materialized": False,
        "official_test_images_opened": False,
        "official_test_masks_opened": False,
        "official_test_labels_opened": False,
        "execution_authorized": False,
    }


def validate_pre_open_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    if type(payload) is not dict:
        raise TypeError("pre-open plan must be an exact JSON object")
    required = {
        "schema_version",
        "source_thaw_sha256",
        "work_breakdown_sha256",
        "semantic_amendment_sha256",
        "b1_authorization_sha256",
        "b2_authorization_sha256",
        "split_metadata_sha256",
        "target_dataset",
        "split_repository_relative_path",
        "split_expected_sha256",
        "split_expected_record_count",
        "context_size",
        "context_rule",
        "query_rule",
        "official_test_accessed",
        "official_test_ids_materialized",
        "official_test_images_opened",
        "official_test_masks_opened",
        "official_test_labels_opened",
        "execution_authorized",
    }
    _assert_exact_keys(payload, required, "pre-open plan")
    expected_bindings = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_thaw_sha256": SOURCE_THAW_SHA256,
        "work_breakdown_sha256": WORK_BREAKDOWN_SHA256,
        "semantic_amendment_sha256": SEMANTIC_AMENDMENT_SHA256,
        "b1_authorization_sha256": B1_AUTHORIZATION_SHA256,
        "b2_authorization_sha256": B2_AUTHORIZATION_SHA256,
        "context_rule": CONTEXT_RULE,
        "query_rule": QUERY_RULE,
    }
    for field, expected in expected_bindings.items():
        if _nonempty_string(payload[field], f"plan.{field}") != expected:
            raise Stage2ConfirmatoryContractError(f"plan.{field} binding changed")
    metadata = validate_split_metadata(
        {
            "split_repository_relative_path": payload[
                "split_repository_relative_path"
            ],
            "split_expected_sha256": payload["split_expected_sha256"],
            "split_expected_record_count": payload["split_expected_record_count"],
        }
    )
    if payload["target_dataset"] != FROZEN_OFFICIAL_SPLIT_PATHS[
        metadata["split_repository_relative_path"]
    ]:
        raise Stage2ConfirmatoryContractError("plan.target_dataset does not match split path")
    if _strict_int(payload["context_size"], "plan.context_size", minimum=1) != CONTEXT_SIZE:
        raise Stage2ConfirmatoryContractError("plan.context_size must be exactly 14")
    for field in (
        "official_test_accessed",
        "official_test_ids_materialized",
        "official_test_images_opened",
        "official_test_masks_opened",
        "official_test_labels_opened",
        "execution_authorized",
    ):
        if _strict_bool(payload[field], f"plan.{field}") is not False:
            raise Stage2ConfirmatoryContractError(f"plan.{field} must be exact false")
    result = dict(payload)
    result.update(metadata)
    result["split_metadata_sha256"] = validate_sha256(
        payload["split_metadata_sha256"], "plan.split_metadata_sha256"
    )
    return result


def verify_pre_open_plan(
    payload: Mapping[str, Any],
    expected_artifact_sha256: str,
    *,
    artifact_bytes: bytes | None = None,
) -> dict[str, Any]:
    expected = validate_sha256(expected_artifact_sha256, "expected_pre_open_plan_sha256")
    data = pretty_json_bytes(payload) if artifact_bytes is None else artifact_bytes
    if not isinstance(data, bytes):
        raise TypeError("artifact_bytes must be bytes")
    parsed = parse_json_bytes(data, name="pre-open plan artifact")
    if canonical_json_bytes(parsed) != canonical_json_bytes(payload):
        raise Stage2ConfirmatoryContractError(
            "pre-open plan mapping differs from exact artifact bytes"
        )
    if sha256_bytes(data) != expected:
        raise Stage2ConfirmatoryContractError(
            "pre-open plan artifact SHA-256 differs from external expectation"
        )
    return validate_pre_open_plan(payload)


def _canonical_repository_root(repository_root: str | Path) -> Path:
    raw = Path(repository_root)
    if not raw.is_absolute() or ".." in raw.parts or raw.is_symlink():
        raise Stage2ConfirmatoryContractError(
            "repository_root must be an absolute canonical non-symlink path"
        )
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2ConfirmatoryContractError("repository_root does not exist") from error
    if resolved != raw or not resolved.is_dir():
        raise Stage2ConfirmatoryContractError(
            "repository_root must be an absolute canonical directory"
        )
    return resolved


def _existing_artifact_within_root(
    path: str | Path, repository_root: Path, *, name: str
) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts or raw.is_symlink():
        raise Stage2ConfirmatoryContractError(
            f"{name} must be an absolute canonical non-symlink path"
        )
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2ConfirmatoryContractError(f"{name} does not exist") from error
    if resolved != raw or not resolved.is_file() or not resolved.is_relative_to(repository_root):
        raise Stage2ConfirmatoryContractError(
            f"{name} must be a canonical in-repository regular file"
        )
    return resolved


def stable_read_bytes(
    path: str | Path,
    expected_sha256: str,
    repository_root: str | Path,
    *,
    name: str,
) -> bytes:
    root = _canonical_repository_root(repository_root)
    checked = _existing_artifact_within_root(path, root, name=name)
    expected = validate_sha256(expected_sha256, f"expected_{name}_sha256")
    descriptor = os.open(checked, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Stage2ConfirmatoryContractError(f"{name} is not a regular file")
        os.lseek(descriptor, 0, os.SEEK_SET)
        data_before = b""
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            data_before += block
        if sha256_bytes(data_before) != expected:
            raise Stage2ConfirmatoryContractError(
                f"{name} SHA-256 differs from external expectation"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        data_after = b""
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            data_after += block
        after = os.fstat(descriptor)
        path_after = os.stat(checked, follow_symlinks=False)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        identity_path = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if identity_before != identity_after or identity_before != identity_path:
            raise Stage2ConfirmatoryContractError(f"{name} changed during consumption")
        if data_before != data_after or sha256_bytes(data_after) != expected:
            raise Stage2ConfirmatoryContractError(f"{name} bytes changed during consumption")
        return data_before
    finally:
        os.close(descriptor)


def load_verified_pre_open_plan(
    path: str | Path, expected_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    data = stable_read_bytes(
        path,
        expected_sha256,
        repository_root,
        name="pre_open_plan",
    )
    payload = parse_json_bytes(data, name="pre-open plan")
    return verify_pre_open_plan(payload, expected_sha256, artifact_bytes=data)


def _new_output_path(path: str | Path, repository_root: Path, *, name: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts:
        raise Stage2ConfirmatoryContractError(f"{name} must be an absolute canonical path")
    try:
        parent = raw.parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2ConfirmatoryContractError(f"{name} parent does not exist") from error
    if parent != raw.parent or not parent.is_dir() or not parent.is_relative_to(repository_root):
        raise Stage2ConfirmatoryContractError(
            f"{name} parent must be a canonical in-repository directory"
        )
    try:
        os.lstat(raw)
    except FileNotFoundError:
        return raw
    raise Stage2ConfirmatoryContractError(f"{name} already exists or is a symlink")


def transactional_publish_bundle(files: Mapping[Path, bytes]) -> None:
    """No-replace, same-parent atomic bundle publication with rollback."""

    if not files:
        raise Stage2ConfirmatoryContractError("empty output bundle")
    targets = list(files)
    if len(set(targets)) != len(targets):
        raise Stage2ConfirmatoryContractError("duplicate output target")
    parent = targets[0].parent
    if any(target.parent != parent for target in targets):
        raise Stage2ConfirmatoryContractError("all bundle outputs must share one parent")
    staged: list[tuple[Path, Path, tuple[int, int]]] = []
    linked: list[tuple[Path, tuple[int, int]]] = []
    try:
        for target, data in files.items():
            try:
                os.lstat(target)
            except FileNotFoundError:
                pass
            else:
                raise Stage2ConfirmatoryContractError(
                    f"bundle target already exists: {target.name}"
                )
            descriptor, temporary_name = tempfile.mkstemp(
                dir=parent, prefix=f".{target.name}.", suffix=".tmp"
            )
            temporary = Path(temporary_name)
            observed = os.fstat(descriptor)
            staged.append((target, temporary, (observed.st_dev, observed.st_ino)))
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        for target, _, _ in staged:
            try:
                os.lstat(target)
            except FileNotFoundError:
                continue
            raise Stage2ConfirmatoryContractError(
                f"bundle target appeared during staging: {target.name}"
            )
        for target, temporary, identity in staged:
            os.link(temporary, target, follow_symlinks=False)
            linked.append((target, identity))
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        for target, identity in reversed(linked):
            try:
                observed = os.stat(target, follow_symlinks=False)
                if (observed.st_dev, observed.st_ino) == identity:
                    os.unlink(target)
            except FileNotFoundError:
                pass
        raise
    finally:
        for _, temporary, _ in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--split-metadata", required=True)
    parser.add_argument("--split-metadata-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _canonical_repository_root(args.repository_root)
    output = _new_output_path(args.output, root, name="output")
    sidecar = _new_output_path(
        output.with_name(output.name + ".sha256"), root, name="output sidecar"
    )
    metadata_bytes = stable_read_bytes(
        args.split_metadata,
        args.split_metadata_sha256,
        root,
        name="split_metadata",
    )
    metadata = parse_json_bytes(metadata_bytes, name="split metadata")
    plan = build_pre_open_plan(
        metadata,
        split_metadata_sha256=args.split_metadata_sha256,
    )
    plan_bytes = pretty_json_bytes(plan)
    plan_sha = sha256_bytes(plan_bytes)
    verify_pre_open_plan(plan, plan_sha, artifact_bytes=plan_bytes)

    # Reauthenticate the only input immediately before publication.  There is
    # deliberately no construction of repository_root / split_path here.
    if stable_read_bytes(
        args.split_metadata,
        args.split_metadata_sha256,
        root,
        name="split_metadata",
    ) != metadata_bytes:
        raise Stage2ConfirmatoryContractError("split metadata changed before publication")
    sidecar_bytes = f"{plan_sha}  {output.name}\n".encode("ascii")
    transactional_publish_bundle({output: plan_bytes, sidecar: sidecar_bytes})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

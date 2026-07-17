"""Build and verify Stage-2 unlabeled context packages and episode-v5 bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

from rc.schema import StatisticsConfig

from rc.stage2_crossfit_schema import (
    COLLECTION_ARTIFACT_TYPE,
    COLLECTION_COMMIT_SCHEMA,
    COLLECTION_SCHEMA,
    COLLECTION_SPEC_SCHEMA,
    CONTEXT_PACKAGE_COMMIT_SCHEMA,
    EXPECTED_COLLECTION_COUNTS,
    Stage2CrossfitContractError,
    Stage2CrossfitEpisode,
    VerifiedEpisodeArtifacts,
    VerifiedStage2EpisodeCollection,
    build_context_payload,
    build_episode_payload,
    canonical_json_bytes,
    collection_commit_path,
    collection_manifest_path,
    context_commit_path,
    direct_file,
    make_verified_collection,
    parse_json_bytes,
    record_sha256,
    repo_relative,
    repository_root,
    sha256_file,
    sidecar_path,
    stable_read,
    verify_episode_collection_completeness,
    verify_governance,
    verify_stage2_context_package,
    verify_stage2_statistics_config,
)


SPEC_ARTIFACT_TYPE = "rc_irstd_stage2_crossfit_collection_build_spec"
COMMIT_ARTIFACT_TYPE = "rc_irstd_stage2_episode_collection_commit"

_SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "path_anchor",
        "collection_role",
        "outer_fold_id",
        "outer_target",
        "base_seed",
        "expected_episode_count",
        "result_based_window_selection",
        "all_manifest_windows_exactly_once",
        "governance_bindings",
        "entries",
    }
)
_ENTRY_FIELDS = frozenset(
    {
        "episode_index",
        "context_package",
        "label_manifest",
        "curve_file",
        "curve_manifest",
    }
)
_CONTEXT_ENTRY_FIELDS = frozenset(
    {"path", "sha256", "commit_path", "commit_sha256"}
)
_ARTIFACT_BINDING_FIELDS = frozenset({"path", "sha256"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "path_anchor",
        "collection_role",
        "outer_fold_id",
        "outer_target",
        "base_seed",
        "episode_count",
        "collection_spec_binding",
        "collection_file",
        "record_sha256_algorithm",
        "ordered_record_sha256_algorithm",
        "ordered_record_sha256",
        "records",
        "governance_bindings",
    }
)
_MANIFEST_RECORD_FIELDS = frozenset(
    {"episode_index", "episode_id", "window_id", "source_domain", "record_sha256"}
)
_COMMIT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "publication_complete",
        "official_test_accessed",
        "path_anchor",
        "collection_file",
        "collection_sidecar",
        "collection_manifest",
        "manifest_sidecar",
    }
)


def _exact_keys(value: Mapping[str, Any], fields: frozenset[str], name: str) -> None:
    if frozenset(value) != fields:
        raise Stage2CrossfitContractError(f"{name} field closure mismatch")


def _bool(value: object, name: str, expected: bool) -> None:
    if type(value) is not bool:  # noqa: E721
        raise TypeError(f"{name} must be an exact JSON boolean")
    if value is not expected:
        raise Stage2CrossfitContractError(f"{name} must be {expected}")


def _int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise Stage2CrossfitContractError(f"{name} must be >= {minimum}")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise TypeError(f"{name} must be a lowercase SHA-256")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be hexadecimal")
    return value


def _relative(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value or any(part in {"", ".", ".."} for part in pure.parts):
        raise Stage2CrossfitContractError(f"{name} must be canonical repository-relative")
    lowered = value.lower().replace("-", "_")
    if "official_test" in lowered or "officialtest" in lowered:
        raise Stage2CrossfitContractError(f"{name} may not reference official test")
    return value


def _artifact_binding(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    _exact_keys(value, _ARTIFACT_BINDING_FIELDS, name)
    return {"path": _relative(value["path"], f"{name}.path"), "sha256": _sha(value["sha256"], f"{name}.sha256")}


def _context_binding(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    _exact_keys(value, _CONTEXT_ENTRY_FIELDS, name)
    result = {
        "path": _relative(value["path"], f"{name}.path"),
        "sha256": _sha(value["sha256"], f"{name}.sha256"),
        "commit_path": _relative(value["commit_path"], f"{name}.commit_path"),
        "commit_sha256": _sha(value["commit_sha256"], f"{name}.commit_sha256"),
    }
    expected_commit = Path(result["path"]).with_name(f"{Path(result['path']).stem}.commit.json").as_posix()
    if result["commit_path"] != expected_commit:
        raise Stage2CrossfitContractError("context commit path is not canonically derived")
    return result


def _sidecar_bytes(digest: str, filename: str) -> bytes:
    return f"{digest}  {filename}\n".encode("ascii")


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise RuntimeError("fsync target is not a directory")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _future_output(path: str | Path, root: Path, name: str) -> Path:
    output = Path(path).expanduser().absolute()
    if root not in output.parents:
        raise Stage2CrossfitContractError(f"{name} must be below repository_root")
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise Stage2CrossfitContractError(f"{name} parent must be a real directory")
    if os.path.lexists(output):
        raise FileExistsError(f"{name} already exists")
    return output


def _publish_bundle(
    output: Path,
    members: Mapping[Path, bytes],
    *,
    post_verify: Any,
) -> None:
    parent = output.parent
    lock = parent / f".{output.name}.lock"
    if os.path.lexists(lock):
        raise FileExistsError("publication lock already exists")
    for path in members:
        if path.parent != parent or os.path.lexists(path):
            raise FileExistsError(f"bundle member already exists: {path}")
    lock_fd = -1
    staging: Path | None = None
    published: list[Path] = []
    try:
        lock_fd = os.open(
            lock,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
        os.close(lock_fd)
        lock_fd = -1
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
        os.chmod(staging, 0o700)
        staged: list[tuple[Path, Path]] = []
        for final, data in members.items():
            candidate = staging / final.name
            _write_exclusive(candidate, data)
            if hashlib.sha256(candidate.read_bytes()).digest() != hashlib.sha256(data).digest():
                raise RuntimeError("staged member verification failed")
            staged.append((candidate, final))
        _fsync_directory(staging)
        # Mapping insertion order places payload/sidecars before commit/sidecar.
        for candidate, final in staged:
            os.link(candidate, final, follow_symlinks=False)
            published.append(final)
        _fsync_directory(parent)
        post_verify(lock)
        lock.unlink()
        _fsync_directory(parent)
    except BaseException:
        for path in reversed(published):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        if lock_fd >= 0:
            os.close(lock_fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
        try:
            _fsync_directory(parent)
        except BaseException:
            pass
        raise
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _context_members(output: Path, payload: Mapping[str, Any], root: Path) -> tuple[dict[Path, bytes], str, str]:
    context_data = canonical_json_bytes(payload) + b"\n"
    context_sha = hashlib.sha256(context_data).hexdigest()
    context_sidecar = sidecar_path(output)
    context_sidecar_data = _sidecar_bytes(context_sha, output.name)
    commit_path = context_commit_path(output)
    commit_payload = {
        "schema_version": CONTEXT_PACKAGE_COMMIT_SCHEMA,
        "artifact_type": "rc_irstd_stage2_context_package_commit",
        "artifact_status": "COMPLETE",
        "publication_complete": True,
        "official_test_accessed": False,
        "path_anchor": "repository_root",
        "context_package": {"path": repo_relative(output, root), "sha256": context_sha},
        "context_sidecar": {
            "path": repo_relative(context_sidecar, root),
            "sha256": hashlib.sha256(context_sidecar_data).hexdigest(),
        },
    }
    commit_data = canonical_json_bytes(commit_payload) + b"\n"
    commit_sha = hashlib.sha256(commit_data).hexdigest()
    members = {
        output: context_data,
        context_sidecar: context_sidecar_data,
        commit_path: commit_data,
        sidecar_path(commit_path): _sidecar_bytes(commit_sha, commit_path.name),
    }
    return members, context_sha, commit_sha


def build_stage2_context_package(
    *,
    window_manifest: str | Path,
    window_manifest_sha256: str,
    window_id: str,
    expected_role: str,
    score_manifest: str | Path,
    score_manifest_sha256: str,
    source_reference: str | Path,
    source_reference_sha256: str,
    source_reference_audit_sha256: str,
    statistics_config: StatisticsConfig,
    output: str | Path,
    repository_root_value: str | Path | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one label-blind context package."""

    root = repository_root(repository_root_value)
    output_path = _future_output(output, root, "context output")
    for other in (
        sidecar_path(output_path), context_commit_path(output_path),
        sidecar_path(context_commit_path(output_path)),
    ):
        if os.path.lexists(other):
            raise FileExistsError(f"context bundle member already exists: {other}")
    payload, _, _, _ = build_context_payload(
        window_manifest=window_manifest,
        window_manifest_sha256=window_manifest_sha256,
        window_id=window_id,
        expected_role=expected_role,
        score_manifest=score_manifest,
        score_manifest_sha256=score_manifest_sha256,
        source_reference=source_reference,
        source_reference_sha256=source_reference_sha256,
        source_reference_audit_sha256=source_reference_audit_sha256,
        statistics_config=statistics_config,
        repository_root_value=root,
    )
    members, context_sha, commit_sha = _context_members(output_path, payload, root)

    def post_verify(lock: Path) -> None:
        if not lock.is_file():
            raise RuntimeError("owned publication lock disappeared")
        for path, data in members.items():
            if path.read_bytes() != data:
                raise RuntimeError("published context member mismatch")

    _publish_bundle(output_path, members, post_verify=post_verify)
    verified = verify_stage2_context_package(
        output_path,
        context_sha,
        commit_sha,
        statistics_config=statistics_config,
        repository_root=root,
    )
    return {
        "schema_version": CONTEXT_PACKAGE_COMMIT_SCHEMA,
        "context_package": repo_relative(verified.path, root),
        "context_sha256": verified.context_sha256,
        "commit": repo_relative(verified.commit_path, root),
        "commit_sha256": verified.commit_sha256,
        "official_test_accessed": False,
    }


def build_stage2_crossfit_episode(
    *,
    episode_index: int,
    context_package: str | Path,
    context_package_sha256: str,
    context_commit_sha256: str,
    label_manifest: str | Path,
    label_manifest_sha256: str,
    curve_file: str | Path,
    curve_file_sha256: str,
    curve_manifest: str | Path,
    curve_manifest_sha256: str,
    statistics_config: StatisticsConfig,
    repository_root_value: str | Path | None = None,
) -> tuple[Stage2CrossfitEpisode, VerifiedEpisodeArtifacts]:
    root = repository_root(repository_root_value)
    context = verify_stage2_context_package(
        context_package,
        context_package_sha256,
        context_commit_sha256,
        statistics_config=statistics_config,
        repository_root=root,
    )
    payload, attachment, rows = build_episode_payload(
        episode_index=episode_index,
        context_package=context,
        label_manifest=label_manifest,
        label_manifest_sha256=label_manifest_sha256,
        curve_file=curve_file,
        curve_file_sha256=curve_file_sha256,
        curve_manifest=curve_manifest,
        curve_manifest_sha256=curve_manifest_sha256,
        repository_root_value=root,
    )
    return (
        Stage2CrossfitEpisode.from_dict(payload),
        VerifiedEpisodeArtifacts(context=context, attachment=attachment, curve_rows=rows),
    )


def verify_collection_spec(
    path: str | Path,
    expected_sha256: str,
    *,
    repository_root_value: str | Path | None = None,
) -> tuple[Path, str, Mapping[str, Any], tuple[dict[str, Any], ...]]:
    root = repository_root(repository_root_value)
    spec_path = direct_file(path, root, "collection spec")
    digest = _sha(expected_sha256, "collection spec SHA")
    payload = parse_json_bytes(stable_read(spec_path, digest, "collection spec"), "collection spec")
    _exact_keys(payload, _SPEC_FIELDS, "collection spec")
    exact = {
        "schema_version": COLLECTION_SPEC_SCHEMA,
        "artifact_type": SPEC_ARTIFACT_TYPE,
        "artifact_status": "RESULT_FREE_FROZEN_BUILD_SPEC",
        "path_anchor": "repository_root",
    }
    for field, expected in exact.items():
        if payload[field] != expected:
            raise Stage2CrossfitContractError(f"collection spec {field} mismatch")
    _bool(payload["development_only"], "development_only", True)
    _bool(payload["official_test_accessed"], "official_test_accessed", False)
    _bool(payload["result_based_window_selection"], "result_based_window_selection", False)
    _bool(payload["all_manifest_windows_exactly_once"], "all_manifest_windows_exactly_once", True)
    role = payload["collection_role"]
    outer = payload["outer_fold_id"]
    expected_count = EXPECTED_COLLECTION_COUNTS.get(role, {}).get(outer)
    if expected_count is None or _int(payload["expected_episode_count"], "expected_episode_count", 1) != expected_count:
        raise Stage2CrossfitContractError("collection spec frozen count mismatch")
    if payload["outer_target"] not in {"NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"}:
        raise Stage2CrossfitContractError("collection spec outer target mismatch")
    if _int(payload["base_seed"], "base_seed") not in {42, 123, 3407}:
        raise Stage2CrossfitContractError("collection spec base seed mismatch")
    if payload["governance_bindings"] != verify_governance(root):
        raise Stage2CrossfitContractError("collection spec governance mismatch")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) != expected_count:
        raise Stage2CrossfitContractError("collection spec entry count mismatch")
    entries: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping):
            raise TypeError("collection entry must be an object")
        _exact_keys(raw, _ENTRY_FIELDS, f"entries[{index}]")
        if _int(raw["episode_index"], "episode_index") != index:
            raise Stage2CrossfitContractError("collection entry index mismatch")
        entries.append(
            {
                "episode_index": index,
                "context_package": _context_binding(raw["context_package"], "context_package"),
                "label_manifest": _artifact_binding(raw["label_manifest"], "label_manifest"),
                "curve_file": _artifact_binding(raw["curve_file"], "curve_file"),
                "curve_manifest": _artifact_binding(raw["curve_manifest"], "curve_manifest"),
            }
        )
    return spec_path, digest, payload, tuple(entries)


def _check_all_windows_once(artifacts: Sequence[VerifiedEpisodeArtifacts]) -> None:
    by_manifest: dict[tuple[str, str], set[str]] = {}
    available: dict[tuple[str, str], set[str]] = {}
    for artifact in artifacts:
        window = artifact.context.window
        key = (str(window.path), window.manifest_sha256)
        by_manifest.setdefault(key, set()).add(window.window_id)
        available.setdefault(key, {str(item["window_id"]) for item in window.payload["windows"]})
    for key in by_manifest:
        if by_manifest[key] != available[key]:
            raise Stage2CrossfitContractError(
                f"collection omits or duplicates windows from {key[0]}"
            )


def _collection_members(
    output: Path,
    episodes: Sequence[Stage2CrossfitEpisode],
    spec_path: Path,
    spec_sha: str,
    root: Path,
) -> tuple[dict[Path, bytes], str, str, str, Mapping[str, Any]]:
    jsonl_data = b"".join(canonical_json_bytes(item.payload) + b"\n" for item in episodes)
    collection_sha = hashlib.sha256(jsonl_data).hexdigest()
    output_sidecar = sidecar_path(output)
    output_sidecar_data = _sidecar_bytes(collection_sha, output.name)
    record_rows = [
        {
            "episode_index": index,
            "episode_id": item.episode_id,
            "window_id": item.payload["window_binding"]["window_id"],
            "source_domain": item.payload["source_domain"],
            "record_sha256": record_sha256(item.payload),
        }
        for index, item in enumerate(episodes)
    ]
    ordered_record_sha = hashlib.sha256(canonical_json_bytes(record_rows)).hexdigest()
    first = episodes[0].payload
    manifest_path = collection_manifest_path(output)
    manifest_payload = {
        "schema_version": COLLECTION_SCHEMA,
        "artifact_type": COLLECTION_ARTIFACT_TYPE,
        "artifact_status": "COMPLETE",
        "development_only": True,
        "official_test_accessed": False,
        "path_anchor": "repository_root",
        "collection_role": first["collection_role"],
        "outer_fold_id": first["outer_fold_id"],
        "outer_target": first["outer_target"],
        "base_seed": first["base_seed"],
        "episode_count": len(episodes),
        "collection_spec_binding": {"path": repo_relative(spec_path, root), "sha256": spec_sha},
        "collection_file": {"path": repo_relative(output, root), "sha256": collection_sha},
        "record_sha256_algorithm": "sha256-canonical-json-stage2-v5-record-v1",
        "ordered_record_sha256_algorithm": "sha256-canonical-json-ordered-stage2-v5-record-digests-v1",
        "ordered_record_sha256": ordered_record_sha,
        "records": record_rows,
        "governance_bindings": verify_governance(root),
    }
    manifest_data = canonical_json_bytes(manifest_payload) + b"\n"
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    manifest_sidecar = sidecar_path(manifest_path)
    manifest_sidecar_data = _sidecar_bytes(manifest_sha, manifest_path.name)
    commit_path = collection_commit_path(output)
    commit_payload = {
        "schema_version": COLLECTION_COMMIT_SCHEMA,
        "artifact_type": COMMIT_ARTIFACT_TYPE,
        "artifact_status": "COMPLETE",
        "publication_complete": True,
        "official_test_accessed": False,
        "path_anchor": "repository_root",
        "collection_file": {"path": repo_relative(output, root), "sha256": collection_sha},
        "collection_sidecar": {
            "path": repo_relative(output_sidecar, root),
            "sha256": hashlib.sha256(output_sidecar_data).hexdigest(),
        },
        "collection_manifest": {"path": repo_relative(manifest_path, root), "sha256": manifest_sha},
        "manifest_sidecar": {
            "path": repo_relative(manifest_sidecar, root),
            "sha256": hashlib.sha256(manifest_sidecar_data).hexdigest(),
        },
    }
    commit_data = canonical_json_bytes(commit_payload) + b"\n"
    commit_sha = hashlib.sha256(commit_data).hexdigest()
    members = {
        output: jsonl_data,
        output_sidecar: output_sidecar_data,
        manifest_path: manifest_data,
        manifest_sidecar: manifest_sidecar_data,
        commit_path: commit_data,
        sidecar_path(commit_path): _sidecar_bytes(commit_sha, commit_path.name),
    }
    return members, collection_sha, manifest_sha, commit_sha, manifest_payload


def build_stage2_crossfit_collection(
    *,
    collection_spec: str | Path,
    collection_spec_sha256: str,
    output: str | Path,
    statistics_config: StatisticsConfig,
    repository_root_value: str | Path | None = None,
) -> dict[str, Any]:
    root = repository_root(repository_root_value)
    output_path = _future_output(output, root, "episode JSONL output")
    if output_path.suffix != ".jsonl":
        raise Stage2CrossfitContractError("episode collection output must end in .jsonl")
    final_members = (
        output_path, sidecar_path(output_path), collection_manifest_path(output_path),
        sidecar_path(collection_manifest_path(output_path)), collection_commit_path(output_path),
        sidecar_path(collection_commit_path(output_path)),
    )
    if any(os.path.lexists(path) for path in final_members):
        raise FileExistsError("one or more collection bundle members already exist")
    spec_path, spec_sha, spec, entries = verify_collection_spec(
        collection_spec, collection_spec_sha256, repository_root_value=root
    )
    episodes: list[Stage2CrossfitEpisode] = []
    artifacts: list[VerifiedEpisodeArtifacts] = []
    for entry in entries:
        context_binding = entry["context_package"]
        episode, verified = build_stage2_crossfit_episode(
            episode_index=entry["episode_index"],
            context_package=root / context_binding["path"],
            context_package_sha256=context_binding["sha256"],
            context_commit_sha256=context_binding["commit_sha256"],
            label_manifest=root / entry["label_manifest"]["path"],
            label_manifest_sha256=entry["label_manifest"]["sha256"],
            curve_file=root / entry["curve_file"]["path"],
            curve_file_sha256=entry["curve_file"]["sha256"],
            curve_manifest=root / entry["curve_manifest"]["path"],
            curve_manifest_sha256=entry["curve_manifest"]["sha256"],
            statistics_config=statistics_config,
            repository_root_value=root,
        )
        episodes.append(episode)
        artifacts.append(verified)
    verify_episode_collection_completeness(episodes)
    _check_all_windows_once(artifacts)
    first = episodes[0].payload
    for field in ("collection_role", "outer_fold_id", "outer_target", "base_seed"):
        if first[field] != spec[field]:
            raise Stage2CrossfitContractError(f"collection spec/episode {field} mismatch")
    members, collection_sha, manifest_sha, commit_sha, _ = _collection_members(
        output_path, episodes, spec_path, spec_sha, root
    )

    def post_verify(lock: Path) -> None:
        if not lock.is_file():
            raise RuntimeError("collection publication lock disappeared")
        for path, data in members.items():
            if path.read_bytes() != data:
                raise RuntimeError("published collection member mismatch")

    _publish_bundle(output_path, members, post_verify=post_verify)
    verified = verify_stage2_episode_collection_bundle(
        output_path,
        collection_sha,
        collection_manifest_path(output_path),
        manifest_sha,
        collection_commit_path(output_path),
        commit_sha,
        statistics_config=statistics_config,
        repository_root_value=root,
    )
    return {
        "schema_version": COLLECTION_COMMIT_SCHEMA,
        "episode_count": len(verified),
        "collection_sha256": collection_sha,
        "manifest_sha256": manifest_sha,
        "commit_sha256": commit_sha,
        "official_test_accessed": False,
    }


def _verify_sidecar(path: Path, artifact: Path, digest: str) -> None:
    expected = _sidecar_bytes(digest, artifact.name)
    data = stable_read(path, sha256_file(path), f"{artifact.name} sidecar")
    if data != expected:
        raise Stage2CrossfitContractError(f"stale sidecar for {artifact.name}")


def verify_stage2_episode_collection_bundle(
    path: str | Path,
    expected_sha256: str,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    commit_path: str | Path,
    expected_commit_sha256: str,
    *,
    statistics_config: StatisticsConfig,
    repository_root_value: str | Path | None = None,
) -> VerifiedStage2EpisodeCollection:
    root = repository_root(repository_root_value)
    collection = direct_file(path, root, "episode JSONL")
    manifest = direct_file(manifest_path, root, "episode collection manifest")
    commit = direct_file(commit_path, root, "episode collection commit")
    if manifest != collection_manifest_path(collection) or commit != collection_commit_path(collection):
        raise Stage2CrossfitContractError("collection manifest/commit path derivation mismatch")
    lock = collection.parent / f".{collection.name}.lock"
    if os.path.lexists(lock):
        raise RuntimeError("episode collection publication lock is present")
    collection_sha = _sha(expected_sha256, "collection SHA")
    manifest_sha = _sha(expected_manifest_sha256, "manifest SHA")
    commit_sha = _sha(expected_commit_sha256, "commit SHA")
    collection_data = stable_read(collection, collection_sha, "episode JSONL")
    manifest_data = stable_read(manifest, manifest_sha, "episode manifest")
    commit_data = stable_read(commit, commit_sha, "episode commit")
    for artifact, digest in ((collection, collection_sha), (manifest, manifest_sha), (commit, commit_sha)):
        _verify_sidecar(sidecar_path(artifact), artifact, digest)
    commit_payload = parse_json_bytes(commit_data, "episode commit")
    _exact_keys(commit_payload, _COMMIT_FIELDS, "episode commit")
    if commit_payload["schema_version"] != COLLECTION_COMMIT_SCHEMA or commit_payload["artifact_type"] != COMMIT_ARTIFACT_TYPE:
        raise Stage2CrossfitContractError("episode commit schema mismatch")
    if commit_payload["artifact_status"] != "COMPLETE" or commit_payload["path_anchor"] != "repository_root":
        raise Stage2CrossfitContractError("episode commit status/path mismatch")
    _bool(commit_payload["publication_complete"], "publication_complete", True)
    _bool(commit_payload["official_test_accessed"], "official_test_accessed", False)
    expected_commit_members = {
        "collection_file": {"path": repo_relative(collection, root), "sha256": collection_sha},
        "collection_sidecar": {
            "path": repo_relative(sidecar_path(collection), root),
            "sha256": sha256_file(sidecar_path(collection)),
        },
        "collection_manifest": {"path": repo_relative(manifest, root), "sha256": manifest_sha},
        "manifest_sidecar": {
            "path": repo_relative(sidecar_path(manifest), root),
            "sha256": sha256_file(sidecar_path(manifest)),
        },
    }
    for field, expected in expected_commit_members.items():
        if commit_payload[field] != expected:
            raise Stage2CrossfitContractError(f"episode commit {field} mismatch")
    manifest_payload = parse_json_bytes(manifest_data, "episode collection manifest")
    _exact_keys(manifest_payload, _MANIFEST_FIELDS, "episode collection manifest")
    if manifest_payload["schema_version"] != COLLECTION_SCHEMA or manifest_payload["artifact_type"] != COLLECTION_ARTIFACT_TYPE:
        raise Stage2CrossfitContractError("collection manifest schema mismatch")
    if manifest_payload["artifact_status"] != "COMPLETE" or manifest_payload["path_anchor"] != "repository_root":
        raise Stage2CrossfitContractError("collection manifest status/path mismatch")
    _bool(manifest_payload["development_only"], "development_only", True)
    _bool(manifest_payload["official_test_accessed"], "official_test_accessed", False)
    if manifest_payload["collection_file"] != expected_commit_members["collection_file"]:
        raise Stage2CrossfitContractError("manifest/commit collection binding mismatch")
    if manifest_payload["governance_bindings"] != verify_governance(root):
        raise Stage2CrossfitContractError("collection governance mismatch")
    spec_binding = _artifact_binding(manifest_payload["collection_spec_binding"], "collection_spec_binding")
    spec_path, spec_sha, spec, entries = verify_collection_spec(
        root / spec_binding["path"], spec_binding["sha256"], repository_root_value=root
    )
    del spec_path, spec_sha
    raw_records = manifest_payload["records"]
    raw_episodes = _parse_jsonl_canonical(collection_data)
    if not isinstance(raw_records, list) or len(raw_records) != len(raw_episodes):
        raise Stage2CrossfitContractError("manifest/JSONL record count mismatch")
    if manifest_payload["episode_count"] != len(raw_episodes) or len(entries) != len(raw_episodes):
        raise Stage2CrossfitContractError("collection count closure mismatch")
    episodes: list[Stage2CrossfitEpisode] = []
    artifacts: list[VerifiedEpisodeArtifacts] = []
    expected_record_rows: list[dict[str, Any]] = []
    for index, (payload, entry) in enumerate(zip(raw_episodes, entries, strict=True)):
        episode = Stage2CrossfitEpisode.from_dict(payload)
        context_binding = entry["context_package"]
        rebuilt, verified = build_stage2_crossfit_episode(
            episode_index=index,
            context_package=root / context_binding["path"],
            context_package_sha256=context_binding["sha256"],
            context_commit_sha256=context_binding["commit_sha256"],
            label_manifest=root / entry["label_manifest"]["path"],
            label_manifest_sha256=entry["label_manifest"]["sha256"],
            curve_file=root / entry["curve_file"]["path"],
            curve_file_sha256=entry["curve_file"]["sha256"],
            curve_manifest=root / entry["curve_manifest"]["path"],
            curve_manifest_sha256=entry["curve_manifest"]["sha256"],
            statistics_config=statistics_config,
            repository_root_value=root,
        )
        if rebuilt.to_dict() != episode.to_dict():
            raise Stage2CrossfitContractError("episode JSONL differs from full replay")
        record_row = {
            "episode_index": index,
            "episode_id": episode.episode_id,
            "window_id": episode.payload["window_binding"]["window_id"],
            "source_domain": episode.payload["source_domain"],
            "record_sha256": record_sha256(episode.payload),
        }
        manifest_record = raw_records[index]
        if not isinstance(manifest_record, Mapping):
            raise TypeError("manifest record must be an object")
        _exact_keys(manifest_record, _MANIFEST_RECORD_FIELDS, "manifest record")
        if dict(manifest_record) != record_row:
            raise Stage2CrossfitContractError("manifest record summary mismatch")
        episodes.append(episode)
        artifacts.append(verified)
        expected_record_rows.append(record_row)
    if manifest_payload["ordered_record_sha256"] != hashlib.sha256(canonical_json_bytes(expected_record_rows)).hexdigest():
        raise Stage2CrossfitContractError("ordered record digest mismatch")
    verify_episode_collection_completeness(episodes)
    _check_all_windows_once(artifacts)
    first = episodes[0].payload
    for field in ("collection_role", "outer_fold_id", "outer_target", "base_seed"):
        if manifest_payload[field] != first[field] or spec[field] != first[field]:
            raise Stage2CrossfitContractError(f"collection {field} mismatch")
    if sha256_file(collection) != collection_sha or sha256_file(manifest) != manifest_sha or sha256_file(commit) != commit_sha:
        raise RuntimeError("collection bundle changed after verification")
    return make_verified_collection(
        path=collection,
        manifest_path=manifest,
        commit_path=commit,
        episodes=episodes,
        artifacts=artifacts,
        collection_sha256=collection_sha,
        manifest_sha256=manifest_sha,
        commit_sha256=commit_sha,
        manifest=manifest_payload,
    )


def _parse_jsonl_canonical(data: bytes) -> list[Mapping[str, Any]]:
    if not data or not data.endswith(b"\n") or b"\r" in data:
        raise Stage2CrossfitContractError("JSONL must use canonical LF records")
    result: list[Mapping[str, Any]] = []
    for index, line in enumerate(data.splitlines()):
        if not line:
            raise Stage2CrossfitContractError("JSONL contains empty record")
        payload = parse_json_bytes(line, f"JSONL[{index}]")
        if canonical_json_bytes(payload) != line:
            raise Stage2CrossfitContractError("JSONL record is not canonical")
        result.append(payload)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-spec", required=True)
    parser.add_argument("--collection-spec-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--statistics-config", required=True)
    parser.add_argument("--statistics-config-sha256", required=True)
    parser.add_argument("--repository-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    statistics_config = verify_stage2_statistics_config(
        args.statistics_config,
        args.statistics_config_sha256,
        repository_root=args.repository_root,
    )
    result = build_stage2_crossfit_collection(
        collection_spec=args.collection_spec,
        collection_spec_sha256=args.collection_spec_sha256,
        output=args.output,
        statistics_config=statistics_config,
        repository_root_value=args.repository_root,
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_arg_parser",
    "build_stage2_context_package",
    "build_stage2_crossfit_collection",
    "build_stage2_crossfit_episode",
    "main",
    "verify_collection_spec",
    "verify_stage2_episode_collection_bundle",
]

#!/usr/bin/env python3
"""Publish a deterministic repository-scoped RC-IRSTD model-design release.

This is the no-new-Stage2-results B4 substitute for committing/tagging a shared
parent worktree.  It freezes the complete in-scope source/protocol surface and
existing development metadata without claiming that the parent Git worktree is
clean and never authorizes execution.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = Path(
    "outputs/stage2_protocol/"
    "RC4_STAGE2_B4_SCOPED_RELEASE_SUBSTITUTION_AMENDMENT_20260717.json"
)
AMENDMENT_SHA = "ff1e575703318214f17d261fc583bd70744c125f53fbd31118a33a6de57e64a4"

TREE_RULES: Mapping[str, frozenset[str]] = {
    "configs": frozenset({".json", ".yaml", ".yml", ".txt"}),
    "data_ext": frozenset({".py"}),
    "evaluation": frozenset({".py"}),
    "losses": frozenset({".py"}),
    "model": frozenset({".py"}),
    "outputs/stage2_detector_contracts/rc4_k2_c14q28_w01_20260716": frozenset(
        {".json", ".sha256", ".txt"}
    ),
    "outputs/stage2_manifests/rc4_k2_c14q28_20260716": frozenset(
        {".json", ".sha256"}
    ),
    "rc": frozenset({".py"}),
    "rc_irstd": frozenset({".py"}),
    "scripts": frozenset({".py", ".sh"}),
    "tests": frozenset({".py"}),
    "utils": frozenset({".py"}),
}

TOP_LEVEL_FILES = (
    "RC-IRSTD_AAAI27_完整模型设计冻结与实验执行计划_20260717.md",
    "RC-IRSTD_AAAI27_预注册实验矩阵与结果表模板_20260717.md",
    "pyproject.toml",
    "requirements.txt",
)

SAFE_STATIC_METADATA_FILES = (
    "audits/aaai27/near_duplicates_effective_splits_v2.json",
    "splits/aaai27_v2/manifest.json",
)

GOVERNANCE_FILES = (
    "outputs/gate_evidence/G1_DEVELOPMENT_GATE_RESULT_RC4_20260716.json",
    "outputs/gate_evidence/G1_DEVELOPMENT_GATE_TERMINAL_AND_PUBLICATION_EVIDENCE_RC4_20260716.json",
    "outputs/experiment_design/RC4_AAAI27_RESULT_FREE_DESIGN_INTEGRITY_AUDIT_20260716.json",
    "outputs/experiment_design/RC4_AAAI27_RESULT_FREE_DESIGN_INTEGRITY_AUDIT_20260716.md",
    "outputs/stage2_detector_contracts/rc4_k2_c14q28_w01_20260716/run_contract_index.json",
    "outputs/stage2_manifests/rc4_k2_c14q28_20260716/materialization_index.json",
    "outputs/stage2_protocol/RC4_STAGE2_B1_CONTRACT_SPINE_INTEGRATION_PASS_20260716.json",
    "outputs/stage2_protocol/RC4_STAGE2_B1_INTEGRITY_AND_BOOTSTRAP_V2_AUTHORIZATION_AMENDMENT_20260716.json",
    "outputs/stage2_protocol/RC4_STAGE2_B2_EPISODE_REFERENCE_AUTHORIZATION_AMENDMENT_20260717.json",
    "outputs/stage2_protocol/RC4_STAGE2_B2_SAFE_CORE_AND_CAUSAL_PERFORMANCE_HOLD_20260717.json",
    "outputs/stage2_protocol/RC4_STAGE2_B3_MODEL_DATA_AND_DEPENDENCY_RESOLUTION_AUTHORIZATION_20260717.json",
    "outputs/stage2_protocol/RC4_STAGE2_CAPACITY_SEED_BOOTSTRAP_PREFREEZE_AUDIT_20260716.json",
    "outputs/stage2_protocol/RC4_STAGE2_IMPLEMENTATION_WORK_BREAKDOWN_HOLD_20260716.json",
    "outputs/stage2_protocol/RC4_STAGE2_K2_C14Q28_INDEPENDENT_AUDIT_20260716.json",
    "outputs/stage2_protocol/RC4_STAGE2_K2_C14Q28_MATERIALIZATION_REPORT_20260716.json",
    "outputs/stage2_protocol/RC4_STAGE2_PRE_G1_RESULT_FREE_ANALYSIS_PLAN_AMENDMENT_SEMANTICS_V1_20260716.json",
    "outputs/stage2_protocol/RC4_STAGE2_PRE_G1_RESULT_FREE_ANALYSIS_PLAN_AMENDMENT_SEMANTICS_V1_20260716.md",
    "outputs/stage2_protocol/RC4_STAGE2_SEED_DERIVATION_MANIFEST_V1_20260716.json",
    "outputs/stage2_protocol/RC4_STAGE2_SOURCE_THAW_AFTER_G1_PASS_20260716.json",
    AMENDMENT.as_posix(),
    AMENDMENT.as_posix() + ".sha256",
)

W12_PREOPEN_FILES = (
    "outputs/stage2_protocol/RC4_STAGE2_W12_THREE_DOMAIN_PREOPEN_PLAN_INDEX_20260717.json",
    "outputs/stage2_protocol/RC4_STAGE2_W12_NUAA_SIRST_OFFICIAL_SPLIT_METADATA_20260717.json",
    "outputs/stage2_protocol/RC4_STAGE2_W12_NUAA_SIRST_PREOPEN_PLAN_20260717.json",
    "outputs/stage2_protocol/RC4_STAGE2_W12_NUDT_SIRST_OFFICIAL_SPLIT_METADATA_20260717.json",
    "outputs/stage2_protocol/RC4_STAGE2_W12_NUDT_SIRST_PREOPEN_PLAN_20260717.json",
    "outputs/stage2_protocol/RC4_STAGE2_W12_IRSTD_1K_OFFICIAL_SPLIT_METADATA_20260717.json",
    "outputs/stage2_protocol/RC4_STAGE2_W12_IRSTD_1K_PREOPEN_PLAN_20260717.json",
)

MATERIALIZATION_INDEX = (
    "outputs/stage2_manifests/rc4_k2_c14q28_20260716/materialization_index.json"
)
RUN_CONTRACT_INDEX = (
    "outputs/stage2_detector_contracts/rc4_k2_c14q28_w01_20260716/"
    "run_contract_index.json"
)

AUTHORITATIVE_STAGE2_CONFIG = "configs/aaai27_stage2_crossfit_v2.json"
LEGACY_NON_AUTHORITATIVE_STAGE2_CONFIGS = (
    "configs/aaai27_analysis_plan.json",
    "configs/aaai27_calibrator_risk_aligned.json",
    "configs/aaai27_calibrator_risk_aligned.yaml",
)
FINAL_AUTHORITY_EXPECTED_SHA256 = {
    "RC-IRSTD_AAAI27_完整模型设计冻结与实验执行计划_20260717.md": (
        "49f83fd0389f7aa0406b161ca46f0343bd5aa8e5a991861d77b313c910589144"
    ),
    "RC-IRSTD_AAAI27_预注册实验矩阵与结果表模板_20260717.md": (
        "bdcf8826d1f72fd1bf8416ba809f5dac68c052f834bf255ed1efe021e4518ccf"
    ),
    AUTHORITATIVE_STAGE2_CONFIG: (
        "dd5e49c9633612e52c00091cfcb2543b48f5fd3f0d7fc5690f297ec0e7d9d963"
    ),
}

REQUIRED_W13 = (
    "scripts/orchestrate_stage2_crossfit.py",
    "outputs/audit_tools/audit_stage2_i0.py",
    "outputs/audit_tools/audit_stage2_development_completion.py",
    "tests/test_stage2_orchestrator_contract.py",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def stable_file_bytes(relative: str) -> bytes:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"non-canonical member path: {relative}")
    path = ROOT.joinpath(*pure.parts)
    if path.is_symlink() or path.resolve(strict=True) != path.absolute():
        raise ValueError(f"symlink/non-canonical member: {relative}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"non-regular member: {relative}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after) or identity(after) != identity(path.stat(follow_symlinks=False)):
        raise RuntimeError(f"member changed during read: {relative}")
    return b"".join(chunks)


def collect_members() -> tuple[str, ...]:
    members: set[str] = (
        set(TOP_LEVEL_FILES)
        | set(SAFE_STATIC_METADATA_FILES)
        | set(GOVERNANCE_FILES)
        | set(REQUIRED_W13)
        | set(W12_PREOPEN_FILES)
        | {path + ".sha256" for path in W12_PREOPEN_FILES}
    )
    for raw_root, extensions in TREE_RULES.items():
        tree = ROOT / raw_root
        if not tree.is_dir() or tree.is_symlink():
            raise FileNotFoundError(f"required source tree missing: {raw_root}")
        for path in tree.rglob("*"):
            if path.is_symlink():
                raise ValueError(
                    f"symlink is forbidden in scoped release tree: "
                    f"{path.relative_to(ROOT).as_posix()}"
                )
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(
                    f"non-regular entry in scoped release tree: "
                    f"{path.relative_to(ROOT).as_posix()}"
                )
            if "__pycache__" in path.parts or path.suffix.lower() not in extensions:
                continue
            members.add(path.relative_to(ROOT).as_posix())
    ordered = tuple(sorted(members))
    if len(ordered) != len(set(ordered)):
        raise RuntimeError("duplicate release member")
    for relative in ordered:
        stable_file_bytes(relative)
    for required in REQUIRED_W13:
        if required not in members:
            raise FileNotFoundError(f"W13 release member missing: {required}")
    for required in W12_PREOPEN_FILES:
        if required not in members or required + ".sha256" not in members:
            raise FileNotFoundError(f"W12 release member/sidecar missing: {required}")
    forbidden = [
        relative
        for relative in ordered
        if relative.startswith("datasets/")
        or (
            relative.startswith("audits/aaai27/")
            and relative not in SAFE_STATIC_METADATA_FILES
        )
        or (
            relative.startswith("splits/aaai27_v2/")
            and relative not in SAFE_STATIC_METADATA_FILES
        )
    ]
    if forbidden:
        raise RuntimeError(f"official-ID/data-bearing members are forbidden: {forbidden}")
    return ordered


def _member_json(member_data: Mapping[str, bytes], path: str) -> Mapping[str, Any]:
    if path not in member_data:
        raise RuntimeError(f"indexed artifact missing from release: {path}")
    value = json.loads(member_data[path].decode("utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"indexed JSON root is not an object: {path}")
    return value


def _verify_bound_member(
    member_data: Mapping[str, bytes], path: Any, digest: Any, *, label: str
) -> None:
    if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"invalid {label} path/SHA binding")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise RuntimeError(f"non-canonical {label} path: {path}")
    if path not in member_data:
        raise RuntimeError(f"{label} member omitted from release: {path}")
    observed = sha256_bytes(member_data[path])
    if observed != digest:
        raise RuntimeError(f"{label} member SHA mismatch: {path}")


def verify_index_and_sidecar_closure(member_data: Mapping[str, bytes]) -> dict[str, Any]:
    materialization = _member_json(member_data, MATERIALIZATION_INDEX)
    materialized = materialization.get("artifacts_excluding_this_index")
    if (
        materialization.get("artifact_count_excluding_this_index") != 52
        or not isinstance(materialized, Mapping)
        or len(materialized) != 52
    ):
        raise RuntimeError("materialization index is not the exact frozen 52+index set")
    for path, digest in materialized.items():
        _verify_bound_member(member_data, path, digest, label="materialization")

    run_index = _member_json(member_data, RUN_CONTRACT_INDEX)
    contracts = run_index.get("contracts")
    if run_index.get("artifact_status") != "DEVELOPMENT_ONLY_RESULT_FREE" or not isinstance(
        contracts, list
    ) or len(contracts) != 27:
        raise RuntimeError("run-contract index is not the exact frozen 27-contract set")
    selection_count = 0
    id_list_count = 0
    for row in contracts:
        if not isinstance(row, Mapping):
            raise RuntimeError("invalid run-contract row")
        _verify_bound_member(member_data, row.get("path"), row.get("sha256"), label="run contract")
        selections = row.get("selection_contracts")
        if not isinstance(selections, list):
            raise RuntimeError("run-contract selection list is absent")
        for selection in selections:
            if not isinstance(selection, Mapping):
                raise RuntimeError("invalid selection-contract row")
            _verify_bound_member(
                member_data,
                selection.get("path"),
                selection.get("sha256"),
                label="selection contract",
            )
            selection_payload = _member_json(member_data, selection.get("path"))
            id_list = selection_payload.get("id_list")
            if not isinstance(id_list, Mapping):
                raise RuntimeError("selection contract ID-list binding is absent")
            _verify_bound_member(
                member_data,
                id_list.get("path"),
                id_list.get("sha256"),
                label="selection ID list",
            )
            selection_count += 1
            id_list_count += 1

    w12_index = _member_json(member_data, W12_PREOPEN_FILES[0])
    rows = w12_index.get("datasets")
    if (
        w12_index.get("artifact_status") != "RESULT_FREE_PREOPEN_PLANS_FROZEN"
        or w12_index.get("contains_observed_results") is not False
        or w12_index.get("execution_authorized") is not False
        or not isinstance(rows, list)
        or len(rows) != 3
        or {row.get("dataset") for row in rows if isinstance(row, Mapping)}
        != {"nuaa-sirst", "nudt-sirst", "irstd-1k"}
    ):
        raise RuntimeError("W12 three-domain pre-open index is not exact/result-free")
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("invalid W12 pre-open index row")
        _verify_bound_member(
            member_data,
            row.get("metadata_path"),
            row.get("metadata_sha256"),
            label="W12 metadata",
        )
        _verify_bound_member(
            member_data,
            row.get("plan_path"),
            row.get("plan_sha256"),
            label="W12 pre-open plan",
        )

    sidecar_count = 0
    for sidecar_path, sidecar_data in member_data.items():
        if not sidecar_path.endswith(".sha256"):
            continue
        target_path = sidecar_path[: -len(".sha256")]
        if target_path not in member_data:
            raise RuntimeError(f"sidecar target omitted from release: {target_path}")
        digest = sha256_bytes(member_data[target_path])
        expected = f"{digest}  {PurePosixPath(target_path).name}\n".encode("ascii")
        if sidecar_data != expected:
            raise RuntimeError(f"external sidecar mismatch: {sidecar_path}")
        sidecar_count += 1
    for required in W12_PREOPEN_FILES:
        if required + ".sha256" not in member_data:
            raise RuntimeError(f"W12 sidecar omitted from release: {required}")
    return {
        "materialization_leaf_count": len(materialized),
        "run_contract_count": len(contracts),
        "selection_contract_count": selection_count,
        "selection_id_list_count": id_list_count,
        "w12_dataset_count": len(rows),
        "verified_external_sidecar_count": sidecar_count,
    }


def verify_final_authority(member_data: Mapping[str, bytes]) -> list[dict[str, str]]:
    required = set(FINAL_AUTHORITY_EXPECTED_SHA256) | set(
        LEGACY_NON_AUTHORITATIVE_STAGE2_CONFIGS
    )
    missing = sorted(required - set(member_data))
    if missing:
        raise RuntimeError(f"Stage-2 config/final authority inventory is incomplete: {missing}")
    rows: list[dict[str, str]] = []
    for path, expected in FINAL_AUTHORITY_EXPECTED_SHA256.items():
        observed = sha256_bytes(member_data[path])
        if observed != expected:
            raise RuntimeError(f"final authority file external SHA mismatch: {path}")
        rows.append({"path": path, "sha256": observed})
    return rows


def command_bytes(arguments: list[str]) -> bytes:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {arguments!r}\n"
            + completed.stdout.decode("utf-8", errors="replace")
        )
    return bytes(completed.stdout)


def environment_lock() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "Pillow", "pytest", "scipy", "torch", "torchvision"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": "rc-irstd.stage2-scoped-release-environment.v1",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "cuda_visible_devices_during_freeze": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "execution_authorized": False,
    }


def build_archive(member_data: Mapping[str, bytes], generated: Mapping[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        combined = {**member_data, **generated}
        for name in sorted(combined):
            pure = PurePosixPath(name)
            if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
                raise ValueError(f"unsafe archive member: {name}")
            data = combined[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def verify_archive(
    archive_data: bytes, expected: Mapping[str, bytes], extraction_root: Path
) -> dict[str, Any]:
    observed: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:") as archive:
        names = [member.name for member in archive.getmembers()]
        if names != sorted(expected) or len(names) != len(set(names)):
            raise RuntimeError("deterministic archive inventory mismatch")
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or any(part in {"", ".", ".."} for part in pure.parts)
                or not member.isfile()
                or member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.mode != 0o644
            ):
                raise RuntimeError(f"unsafe/non-deterministic archive member: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"archive member unreadable: {member.name}")
            data = handle.read()
            if data != expected[member.name]:
                raise RuntimeError(f"archive member differs: {member.name}")
            target = extraction_root.joinpath(*pure.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            observed[member.name] = data
    for name, data in expected.items():
        target = extraction_root.joinpath(*PurePosixPath(name).parts)
        if target.read_bytes() != data:
            raise RuntimeError(f"fresh extraction mismatch: {name}")
    return {
        "fresh_extraction_verified": True,
        "verified_member_count": len(observed),
        "member_order": "lexicographic_repository_relative_posix",
        "fixed_mtime": 0,
        "fixed_uid_gid": [0, 0],
        "fixed_mode": "0644",
    }


def write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing any target entry."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is required for atomic publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    target_parent = os.open(
        target.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        result = renameat2(
            -100,  # AT_FDCWD for the absolute source path.
            os.fsencode(source),
            target_parent,
            os.fsencode(target.name),
            1,  # RENAME_NOREPLACE
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(error, os.strerror(error), target)
            raise OSError(error, os.strerror(error), target)
    finally:
        os.close(target_parent)


def sidecar_bytes(digest: str, name: str) -> bytes:
    return f"{digest}  {name}\n".encode("ascii")


def freeze(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).expanduser()
    output = output if output.is_absolute() else ROOT / output
    output = output.absolute()
    if ROOT not in output.parents or output == ROOT or os.path.lexists(output):
        raise ValueError("output must be a new direct repository descendant")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("release output parent must be a real directory")

    members = collect_members()
    member_data = {relative: stable_file_bytes(relative) for relative in members}
    if collect_members() != members:
        raise RuntimeError("scoped release inventory changed while freezing")
    closure_verification = verify_index_and_sidecar_closure(member_data)
    final_authority_bindings = verify_final_authority(member_data)
    amendment_data = member_data[AMENDMENT.as_posix()]
    if sha256_bytes(amendment_data) != AMENDMENT_SHA:
        raise RuntimeError("B4 amendment external SHA mismatch")
    base_head = command_bytes(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    tracked_diff = command_bytes(["git", "diff", "--binary", "--", *members])
    worktree_status = command_bytes(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *members,
        ]
    )
    environment_data = pretty_bytes(environment_lock())
    environment_sha = sha256_bytes(environment_data)
    generated = {
        "_release/ENVIRONMENT.json": environment_data,
        "_release/TRACKED_DIFF.patch": tracked_diff,
        "_release/WORKTREE_STATUS.txt": worktree_status,
    }
    archive_data = build_archive(member_data, generated)
    extraction = Path(tempfile.mkdtemp(prefix="rc-irstd-stage2-release-verify-"))
    try:
        archive_verification = verify_archive(
            archive_data, {**member_data, **generated}, extraction
        )
    finally:
        shutil.rmtree(extraction, ignore_errors=True)

    source_rows = [
        {"path": name, "size_bytes": len(member_data[name]), "sha256": sha256_bytes(member_data[name])}
        for name in members
    ]
    generated_rows = [
        {"path": name, "size_bytes": len(generated[name]), "sha256": sha256_bytes(generated[name])}
        for name in sorted(generated)
    ]
    archive_sha = sha256_bytes(archive_data)
    manifest: dict[str, Any] = {
        "schema_version": "rc-irstd.stage2-scoped-source-release.v1",
        "artifact_type": "rc_irstd_stage2_complete_model_design_scoped_release",
        "artifact_status": "SCOPED_SOURCE_RELEASE_COMPLETE",
        "result_free": True,
        "result_free_scope": (
            "No new Stage-2 observed performance results are produced or consumed; "
            "the release hashes existing frozen development metadata and prior Stage-1 "
            "governance evidence."
        ),
        "contains_observed_stage2_results": False,
        "execution_authorized": False,
        "frozen_development_metadata_hashed": True,
        "official_test_id_artifacts_in_allowlist": False,
        "dataset_image_mask_checkpoint_files_in_allowlist": False,
        "official_test_execution_authorized": False,
        "system_level_file_access_instrumented": False,
        "path_anchor": "repository_root",
        "stage2_config_authority": {
            "authoritative_path": AUTHORITATIVE_STAGE2_CONFIG,
            "authoritative_sha256": sha256_bytes(
                member_data[AUTHORITATIVE_STAGE2_CONFIG]
            ),
            "legacy_non_authoritative_paths": list(
                LEGACY_NON_AUTHORITATIVE_STAGE2_CONFIGS
            ),
            "legacy_paths_must_be_rejected_by_stage2_launch": True,
        },
        "final_authority_bindings": final_authority_bindings,
        "repository_scope": str(ROOT),
        "base_git_head": base_head,
        "worktree_clean_claimed": False,
        "git_diff_and_status_scope": "verified_source_member_allowlist_only",
        "tracked_diff": {"size_bytes": len(tracked_diff), "sha256": sha256_bytes(tracked_diff)},
        "worktree_status": {"size_bytes": len(worktree_status), "sha256": sha256_bytes(worktree_status)},
        "b4_amendment": {"path": AMENDMENT.as_posix(), "sha256": AMENDMENT_SHA},
        "allowlist_algorithm": "explicit-safe-metadata-governance-w12-plus-source-trees-v3",
        "source_member_count": len(source_rows),
        "source_members": source_rows,
        "generated_members": generated_rows,
        "archive": {
            "name": "RC-IRSTD_STAGE2_MODEL_DESIGN_SCOPED_RELEASE.tar",
            "format": "pax-tar-uncompressed",
            "size_bytes": len(archive_data),
            "sha256": archive_sha,
        },
        "archive_verification": archive_verification,
        "index_and_sidecar_closure_verification": closure_verification,
        "external_environment_lock": {
            "name": "ENVIRONMENT.json",
            "size_bytes": len(environment_data),
            "sha256": environment_sha,
        },
        "required_w13_members": list(REQUIRED_W13),
        "release_scope_complete": True,
    }
    manifest["manifest_content_sha256_algorithm"] = "sha256-canonical-json-without-self-field-v1"
    manifest["manifest_content_sha256"] = sha256_bytes(canonical_bytes(manifest))
    manifest_data = pretty_bytes(manifest)
    manifest_sha = sha256_bytes(manifest_data)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        archive_path = staging / manifest["archive"]["name"]
        manifest_path = staging / "SCOPED_RELEASE_MANIFEST.json"
        environment_path = staging / "ENVIRONMENT.json"
        write_exclusive(archive_path, archive_data)
        write_exclusive(archive_path.with_name(archive_path.name + ".sha256"), sidecar_bytes(archive_sha, archive_path.name))
        write_exclusive(environment_path, environment_data)
        write_exclusive(
            environment_path.with_name(environment_path.name + ".sha256"),
            sidecar_bytes(environment_sha, environment_path.name),
        )
        write_exclusive(manifest_path, manifest_data)
        write_exclusive(manifest_path.with_name(manifest_path.name + ".sha256"), sidecar_bytes(manifest_sha, manifest_path.name))
        commit = {
            "schema_version": "rc-irstd.stage2-scoped-source-release-commit.v1",
            "artifact_status": "COMMITTED_COMPLETE",
            "publication_complete": True,
            "execution_authorized": False,
            "contains_observed_stage2_results": False,
            "frozen_development_metadata_hashed": True,
            "official_test_id_artifacts_in_allowlist": False,
            "official_test_execution_authorized": False,
            "system_level_file_access_instrumented": False,
            "manifest": {"path": manifest_path.name, "sha256": manifest_sha},
            "archive": {"path": archive_path.name, "sha256": archive_sha},
            "environment_lock": {
                "path": environment_path.name,
                "sha256": environment_sha,
            },
            "b4_amendment_sha256": AMENDMENT_SHA,
        }
        commit_data = pretty_bytes(commit)
        commit_sha = sha256_bytes(commit_data)
        commit_path = staging / "COMMIT.json"
        write_exclusive(commit_path.with_name(commit_path.name + ".sha256"), sidecar_bytes(commit_sha, commit_path.name))
        write_exclusive(commit_path, commit_data)
        descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        rename_noreplace(staging, output)
        descriptor = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {
            "output_dir": str(output),
            "manifest_sha256": manifest_sha,
            "archive_sha256": archive_sha,
            "environment_sha256": environment_sha,
            "commit_sha256": commit_sha,
            "source_member_count": len(source_rows),
            "execution_authorized": False,
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    print(json.dumps(freeze(arguments.output_dir), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

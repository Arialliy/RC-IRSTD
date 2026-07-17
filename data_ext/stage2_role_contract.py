"""Fail-closed Stage-2 detector selection and run contracts.

This module is deliberately separate from the legacy official split contract.
It consumes only result-free, official-train-derived manifest metadata.  It
never resolves an official-test split and never opens an image, mask, score,
checkpoint, or metric artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .dataset_identity import sha256_file


STAGE2_SELECTION_SCHEMA = "rc-irstd.stage2-detector-selection.v1"
STAGE2_RUN_CONTRACT_SCHEMA = "rc-irstd.stage2-detector-run-contract.v1"
STAGE2_RUN_INDEX_SCHEMA = "rc-irstd.stage2-detector-run-contract-index.v1"
SEED_MANIFEST_SCHEMA = "rc-irstd.stage2-seed-derivation-manifest.v1"
MATERIALIZATION_INDEX_SCHEMA = (
    "rc-irstd.stage2-k2-c14q28-materialization-index.v1"
)
SELECTION_ROLES = frozenset({"detector_oof_train", "detector_full_fit_train"})
DETECTOR_ROLES = frozenset({"detector_oof", "detector_full_fit"})
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

SOURCE_THAW_PATH = Path(
    "outputs/stage2_protocol/RC4_STAGE2_SOURCE_THAW_AFTER_G1_PASS_20260716.json"
)
SOURCE_THAW_SHA256 = (
    "0e4f3e27026d5a2071a2c8f94f84c366d208f3789de17649aad926c64cd6b0b9"
)
WORK_BREAKDOWN_PATH = Path(
    "outputs/stage2_protocol/RC4_STAGE2_IMPLEMENTATION_WORK_BREAKDOWN_HOLD_20260716.json"
)
WORK_BREAKDOWN_SHA256 = (
    "cc240f97aea6c99dde1e5c537a26c1b22e606b0f499ca495af71d15fa44c9d06"
)
SEED_DOMAIN_TAG = "rc-irstd.stage2.seed.v1"
RECORDS_CONTENT_ALGORITHM = "sha256-canonical-json-stage2-selection-records-v1"
FROZEN_BASE_SEEDS = (42, 123, 3407)
FROZEN_OUTER_FOLDS = (
    "outer_leave_nuaa_sirst",
    "outer_leave_nudt_sirst",
    "outer_leave_irstd_1k",
)
FROZEN_SEED_ROLES = (
    ("detector_oof::fold_0", "detector_oof", "fold_0"),
    ("detector_oof::fold_1", "detector_oof", "fold_1"),
    ("detector_full_fit::full_fit", "detector_full_fit", "full_fit"),
    (
        "stage2_calibrator_t8::not_applicable",
        "stage2_calibrator_t8",
        "not_applicable",
    ),
    (
        "baseline_t6_direct_mlp::not_applicable",
        "baseline_t6_direct_mlp",
        "not_applicable",
    ),
    (
        "baseline_t7_monotone_oracle::not_applicable",
        "baseline_t7_monotone_oracle",
        "not_applicable",
    ),
    (
        "paired_bootstrap_query_images::not_applicable",
        "paired_bootstrap_query_images",
        "not_applicable",
    ),
)


_SELECTION_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "execution_authorized",
        "official_test_accessed",
        "observed_results",
        "selection_id",
        "run_id",
        "selection_role",
        "detector_role",
        "outer_fold_id",
        "outer_target_domain",
        "source_domain",
        "source_domains",
        "base_seed",
        "derived_seed",
        "oof_fold_index",
        "dataset_root",
        "id_list",
        "records",
        "record_count",
        "records_content_algorithm",
        "records_content_sha256",
        "bindings",
    }
)
_SELECTION_RECORD_KEYS = frozenset(
    {
        "canonical_id",
        "image_id",
        "original_image_path",
        "original_image_sha256",
        "exclusion_group_id",
        "near_duplicate_cluster_id_or_unique_sentinel",
        "source_role_record_index",
    }
)
_RUN_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "execution_authorized",
        "official_test_accessed",
        "observed_results",
        "run_id",
        "outer_fold_id",
        "outer_target_domain",
        "source_domains",
        "base_seed",
        "derived_seed",
        "detector_role",
        "oof_fold_index",
        "selection_contracts",
        "training",
        "bindings",
    }
)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_stage2_seed(
    base_seed: int,
    outer_fold_id: str,
    artifact_role: str,
    oof_marker: str,
) -> int:
    """Replay the frozen SHA-256 domain-separated seed derivation."""

    values = [
        SEED_DOMAIN_TAG,
        _exact_int(base_seed, "base_seed", minimum=0),
        _nonempty(outer_fold_id, "outer_fold_id"),
        _nonempty(artifact_role, "artifact_role"),
        _nonempty(oof_marker, "oof_marker"),
    ]
    encoded = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )
    unsigned = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return 1 + (unsigned % 2147483646)


def load_stage2_selection(
    path: str | Path,
    expected_sha256: str,
    expected_role: str,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and fully replay one detector training selection contract."""

    root = _repository_root(repository_root)
    selection_path = _input_file(path, root, "selection contract")
    _reject_incomplete_ancestor(selection_path, root)
    payload, digest = _load_json_stable(selection_path, "selection contract")
    if digest != _sha256(expected_sha256, "expected selection SHA-256"):
        raise ValueError("selection contract SHA-256 mismatch")
    _exact_keys(payload, _SELECTION_KEYS, "selection contract")
    if payload["schema_version"] != STAGE2_SELECTION_SCHEMA:
        raise ValueError("unsupported Stage2 selection schema")
    if payload["artifact_type"] != "rc_irstd_stage2_detector_selection":
        raise ValueError("selection artifact_type mismatch")
    _common_result_free_guards(payload, "selection contract")

    role = _nonempty(payload["selection_role"], "selection_role")
    if role not in SELECTION_ROLES or role != expected_role:
        raise ValueError("selection role mismatch")
    detector_role = _nonempty(payload["detector_role"], "detector_role")
    expected_detector_role = (
        "detector_oof" if role == "detector_oof_train" else "detector_full_fit"
    )
    if detector_role != expected_detector_role:
        raise ValueError("selection detector_role does not match selection_role")

    _validate_identity_fields(payload, detector_role=detector_role)
    source_domain = _nonempty(payload["source_domain"], "source_domain")
    source_domains = _two_unique_strings(payload["source_domains"], "source_domains")
    if source_domain not in source_domains:
        raise ValueError("selection source_domain is absent from source_domains")
    if payload["outer_target_domain"] in source_domains:
        raise ValueError("outer target appears in detector source domains")

    dataset_root = _relative_path(payload["dataset_root"], "dataset_root")
    _assert_not_official_test_path(dataset_root, "dataset_root")
    id_binding = _binding(payload["id_list"], "id_list", allowed_extra={"format"})
    if id_binding.get("format") != "utf8_one_image_id_per_line_v1":
        raise ValueError("selection id_list format mismatch")
    id_path = _repo_file(root, id_binding["path"], "selection id_list")
    id_digest_before = sha256_file(id_path)
    if id_digest_before != id_binding["sha256"]:
        raise ValueError("selection id_list SHA-256 mismatch")
    ids = [line.strip() for line in id_path.read_text(encoding="utf-8").splitlines()]
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("selection id_list must contain unique non-empty IDs")
    if sha256_file(id_path) != id_digest_before:
        raise RuntimeError("selection id_list changed while read")

    raw_records = payload["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("selection records must be a non-empty list")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise TypeError(f"selection records[{index}] must be an object")
        _exact_keys(raw, _SELECTION_RECORD_KEYS, f"records[{index}]")
        record = dict(raw)
        _validate_selection_record(record, index, source_domain, dataset_root)
        records.append(record)
    if _exact_int(payload["record_count"], "record_count", minimum=1) != len(records):
        raise ValueError("selection record_count mismatch")
    if ids != [str(record["image_id"]) for record in records]:
        raise ValueError("selection id_list order differs from records")
    if payload["records_content_algorithm"] != RECORDS_CONTENT_ALGORITHM:
        raise ValueError("selection records-content algorithm mismatch")
    if _sha256(payload["records_content_sha256"], "records_content_sha256") != canonical_json_sha256(records):
        raise ValueError("selection records_content_sha256 mismatch")

    bindings = _selection_bindings(payload["bindings"], root)
    assignment = bindings["assignment"]
    assignment_payload, _ = _load_json_stable(
        _repo_file(root, assignment["path"], "assignment"), "assignment"
    )
    expected_records = _expected_records_from_assignment(
        assignment_payload,
        source_domain=source_domain,
        detector_role=detector_role,
        oof_fold_index=payload["oof_fold_index"],
    )
    if records != expected_records:
        raise ValueError("selection records are not the exact assignment-derived order")
    return dict(payload)


def verify_stage2_run_contract(
    path: str | Path,
    expected_sha256: str,
    seed_manifest_path: str | Path,
    materialization_index_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a complete two-source Stage-2 detector run contract."""

    root = _repository_root(repository_root)
    run_path = _input_file(path, root, "Stage2 run contract")
    _reject_incomplete_ancestor(run_path, root)
    payload, digest = _load_json_stable(run_path, "Stage2 run contract")
    if digest != _sha256(expected_sha256, "expected run-contract SHA-256"):
        raise ValueError("Stage2 run-contract SHA-256 mismatch")
    _exact_keys(payload, _RUN_KEYS, "Stage2 run contract")
    if payload["schema_version"] != STAGE2_RUN_CONTRACT_SCHEMA:
        raise ValueError("unsupported Stage2 run-contract schema")
    if payload["artifact_type"] != "rc_irstd_stage2_detector_run_contract":
        raise ValueError("Stage2 run-contract artifact_type mismatch")
    _common_result_free_guards(payload, "Stage2 run contract")
    detector_role = _nonempty(payload["detector_role"], "detector_role")
    if detector_role not in DETECTOR_ROLES:
        raise ValueError("unsupported detector_role")
    _validate_identity_fields(payload, detector_role=detector_role)
    source_domains = _two_unique_strings(payload["source_domains"], "source_domains")
    if payload["outer_target_domain"] in source_domains:
        raise ValueError("outer target appears in detector source domains")

    bindings = _run_bindings(payload["bindings"], root)
    supplied_seed = _input_file(seed_manifest_path, root, "seed manifest")
    supplied_index = _input_file(
        materialization_index_path, root, "materialization index"
    )
    if supplied_seed != _repo_file(root, bindings["seed_manifest"]["path"], "seed manifest"):
        raise ValueError("supplied seed manifest path differs from run binding")
    if supplied_index != _repo_file(
        root, bindings["materialization_index"]["path"], "materialization index"
    ):
        raise ValueError("supplied materialization-index path differs from run binding")
    seed_payload, seed_digest = _load_json_stable(supplied_seed, "seed manifest")
    if seed_digest != bindings["seed_manifest"]["sha256"]:
        raise ValueError("seed manifest SHA-256 mismatch")
    _verify_seed_manifest(seed_payload)
    materialization_payload, materialization_digest = _load_json_stable(
        supplied_index, "materialization index"
    )
    if materialization_digest != bindings["materialization_index"]["sha256"]:
        raise ValueError("materialization index SHA-256 mismatch")
    _verify_materialization_index(
        materialization_payload,
        root,
        bindings["materialization_artifacts_sha256"],
    )

    expected_seed = _lookup_seed(
        seed_payload,
        base_seed=payload["base_seed"],
        outer_fold_id=str(payload["outer_fold_id"]),
        detector_role=detector_role,
        oof_fold_index=payload["oof_fold_index"],
    )
    if payload["derived_seed"] != expected_seed:
        raise ValueError("run derived_seed differs from frozen 63-entry seed table")
    _verify_detector_training(payload["training"], bindings["detector_config"], root)

    raw_selections = payload["selection_contracts"]
    if not isinstance(raw_selections, list) or len(raw_selections) != 2:
        raise ValueError("run contract requires exactly two selection contracts")
    expected_role = (
        "detector_oof_train"
        if detector_role == "detector_oof"
        else "detector_full_fit_train"
    )
    selected_domains: list[str] = []
    for index, raw_binding in enumerate(raw_selections):
        binding = _binding(
            raw_binding,
            f"selection_contracts[{index}]",
            allowed_extra={"source_domain", "selection_role", "record_count"},
        )
        if binding.get("selection_role") != expected_role:
            raise ValueError("run selection_role mismatch")
        selection = load_stage2_selection(
            _repo_file(root, binding["path"], "selection contract"),
            binding["sha256"],
            expected_role,
            repository_root=root,
        )
        domain = _nonempty(binding.get("source_domain"), "selection source_domain")
        if selection["source_domain"] != domain:
            raise ValueError("selection binding source_domain mismatch")
        if _exact_int(binding.get("record_count"), "selection record_count", minimum=1) != selection["record_count"]:
            raise ValueError("selection binding record_count mismatch")
        for field in (
            "run_id",
            "outer_fold_id",
            "outer_target_domain",
            "source_domains",
            "base_seed",
            "derived_seed",
            "detector_role",
            "oof_fold_index",
        ):
            if selection[field] != payload[field]:
                raise ValueError(f"selection/run identity mismatch for {field}")
        selected_domains.append(domain)
    if selected_domains != source_domains:
        raise ValueError("selection order must equal the run source_domains order")
    return dict(payload)


def verify_stage2_run_contract_sidecar(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Verify a run through its adjacent SHA sidecar and declared bindings."""

    root = _repository_root(repository_root)
    run_path = _input_file(path, root, "Stage2 run contract")
    sidecar = run_path.with_suffix(run_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        raise FileNotFoundError("Stage2 run contract requires an adjacent .sha256 sidecar")
    lines = [line.strip() for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("Stage2 run-contract sidecar must contain exactly one line")
    fields = lines[0].split()
    if len(fields) != 2 or fields[1] != run_path.name:
        raise ValueError("Stage2 run-contract sidecar filename mismatch")
    expected = _sha256(fields[0], "run-contract sidecar SHA-256")
    raw, _ = _load_json_stable(run_path, "Stage2 run contract preflight")
    if not isinstance(raw.get("bindings"), Mapping):
        raise ValueError("Stage2 run contract has no bindings")
    seed = raw["bindings"].get("seed_manifest")
    index = raw["bindings"].get("materialization_index")
    if not isinstance(seed, Mapping) or not isinstance(index, Mapping):
        raise ValueError("Stage2 run contract lacks seed/materialization bindings")
    payload = verify_stage2_run_contract(
        run_path,
        expected,
        _repo_file(root, seed.get("path"), "seed manifest"),
        _repo_file(root, index.get("path"), "materialization index"),
        repository_root=root,
    )
    return payload, expected


def _validate_identity_fields(payload: Mapping[str, Any], *, detector_role: str) -> None:
    _nonempty(payload["run_id"], "run_id")
    _nonempty(payload["outer_fold_id"], "outer_fold_id")
    _nonempty(payload["outer_target_domain"], "outer_target_domain")
    _exact_int(payload["base_seed"], "base_seed", minimum=0)
    _exact_int(payload["derived_seed"], "derived_seed", minimum=1, maximum=2147483646)
    fold = payload["oof_fold_index"]
    if detector_role == "detector_oof":
        if _exact_int(fold, "oof_fold_index", minimum=0, maximum=1) not in (0, 1):
            raise ValueError("OOF fold index must be 0 or 1")
    elif fold is not None:
        raise ValueError("full-fit detector oof_fold_index must be null")


def _validate_selection_record(
    record: Mapping[str, Any], index: int, source_domain: str, dataset_root: str
) -> None:
    canonical = _nonempty(record["canonical_id"], f"records[{index}].canonical_id")
    image_id = _nonempty(record["image_id"], f"records[{index}].image_id")
    if canonical != f"{source_domain}::{image_id}":
        raise ValueError("selection canonical_id/domain/image_id mismatch")
    image_path = _relative_path(
        record["original_image_path"], f"records[{index}].original_image_path"
    )
    _assert_not_official_test_path(image_path, "original_image_path")
    expected_prefix = PurePosixPath(dataset_root) / "images"
    if expected_prefix not in PurePosixPath(image_path).parents:
        raise ValueError("selection image path is outside dataset_root/images")
    _sha256(record["original_image_sha256"], "original_image_sha256")
    _nonempty(record["exclusion_group_id"], "exclusion_group_id")
    _nonempty(
        record["near_duplicate_cluster_id_or_unique_sentinel"],
        "near_duplicate_cluster_id_or_unique_sentinel",
    )
    _exact_int(record["source_role_record_index"], "source_role_record_index", minimum=0)


def _selection_bindings(value: Any, root: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("selection bindings must be an object")
    required = {
        "assignment",
        "materialization_index",
        "seed_manifest",
        "detector_config",
        "release_artifact",
        "implementation_work_breakdown",
    }
    _exact_keys(value, required, "selection bindings")
    result: dict[str, Any] = {}
    for name in required - {"release_artifact"}:
        result[name] = _binding(value[name], f"bindings.{name}")
        path = _repo_file(root, result[name]["path"], f"bindings.{name}")
        if sha256_file(path) != result[name]["sha256"]:
            raise ValueError(f"selection binding SHA-256 mismatch: {name}")
    result["release_artifact"] = _release_binding(value["release_artifact"], root)
    _verify_frozen_governance(result, root)
    return result


def _run_bindings(value: Any, root: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("run bindings must be an object")
    required = {
        "detector_config",
        "seed_manifest",
        "materialization_index",
        "materialization_artifacts_sha256",
        "release_artifact",
        "implementation_work_breakdown",
    }
    _exact_keys(value, required, "run bindings")
    result: dict[str, Any] = {}
    for name in ("detector_config", "seed_manifest", "materialization_index", "implementation_work_breakdown"):
        result[name] = _binding(value[name], f"bindings.{name}")
        path = _repo_file(root, result[name]["path"], f"bindings.{name}")
        if sha256_file(path) != result[name]["sha256"]:
            raise ValueError(f"run binding SHA-256 mismatch: {name}")
    artifacts = value["materialization_artifacts_sha256"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("materialization_artifacts_sha256 must be a non-empty object")
    result["materialization_artifacts_sha256"] = {
        _relative_path(path, "materialization artifact path"): _sha256(digest, "materialization artifact SHA-256")
        for path, digest in artifacts.items()
    }
    result["release_artifact"] = _release_binding(value["release_artifact"], root)
    _verify_frozen_governance(result, root)
    return result


def _release_binding(value: Any, root: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("release_artifact must be an object")
    keys = {"path", "sha256", "git_commit", "tag", "source_archive"}
    _exact_keys(value, keys, "release_artifact")
    result = dict(value)
    result["path"] = _relative_path(result["path"], "release_artifact.path")
    result["sha256"] = _sha256(result["sha256"], "release_artifact.sha256")
    _nonempty(result["git_commit"], "release_artifact.git_commit")
    _nonempty(result["tag"], "release_artifact.tag")
    source_archive = _binding(result["source_archive"], "release_artifact.source_archive")
    result["source_archive"] = source_archive
    if sha256_file(_repo_file(root, result["path"], "release artifact")) != result["sha256"]:
        raise ValueError("release artifact SHA-256 mismatch")
    if sha256_file(_repo_file(root, source_archive["path"], "source archive")) != source_archive["sha256"]:
        raise ValueError("source archive SHA-256 mismatch")
    return result


def _verify_frozen_governance(bindings: Mapping[str, Any], root: Path) -> None:
    release = bindings["release_artifact"]
    if release["path"] != SOURCE_THAW_PATH.as_posix() or release["sha256"] != SOURCE_THAW_SHA256:
        raise ValueError("Stage2 contract is not bound to the authorized thaw artifact")
    work = bindings["implementation_work_breakdown"]
    if work["path"] != WORK_BREAKDOWN_PATH.as_posix() or work["sha256"] != WORK_BREAKDOWN_SHA256:
        raise ValueError("Stage2 contract is not bound to the frozen work breakdown")
    if sha256_file(root / SOURCE_THAW_PATH) != SOURCE_THAW_SHA256:
        raise ValueError("authorized Stage2 source-thaw bytes changed")
    if sha256_file(root / WORK_BREAKDOWN_PATH) != WORK_BREAKDOWN_SHA256:
        raise ValueError("Stage2 work-breakdown bytes changed")


def _expected_records_from_assignment(
    assignment: Mapping[str, Any],
    *,
    source_domain: str,
    detector_role: str,
    oof_fold_index: Any,
) -> list[dict[str, Any]]:
    if assignment.get("artifact_type") != "rc_irstd_stage2_detector_fit_group_assignment":
        raise ValueError("assignment artifact_type mismatch")
    if assignment.get("domain") != source_domain:
        raise ValueError("assignment domain mismatch")
    if assignment.get("number_of_folds_K") != 2:
        raise ValueError("assignment K must be exactly 2")
    _common_result_free_guards(assignment, "assignment")
    records = assignment.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("assignment records must be a non-empty list")
    selected: list[dict[str, Any]] = []
    for position, raw in enumerate(records):
        if not isinstance(raw, Mapping) or raw.get("source_role") != "detector_fit":
            raise ValueError("assignment record role mismatch")
        if raw.get("source_role_record_index") != position:
            raise ValueError("assignment record order/index mismatch")
        fold = _exact_int(raw.get("oof_fold_index"), "assignment oof_fold_index", minimum=0, maximum=1)
        include = detector_role == "detector_full_fit" or fold != oof_fold_index
        if include:
            selected.append({key: raw.get(key) for key in _SELECTION_RECORD_KEYS})
    if not selected:
        raise ValueError("assignment-derived detector selection is empty")
    return selected


def _verify_seed_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SEED_MANIFEST_SCHEMA:
        raise ValueError("unsupported Stage2 seed-manifest schema")
    if payload.get("execution_authorized") is not False:
        raise ValueError("seed manifest execution_authorized must be exactly false")
    algorithm = payload.get("derivation_algorithm")
    if not isinstance(algorithm, Mapping) or algorithm.get("algorithm_id") != "sha256_domain_separated_seed_v1" or algorithm.get("domain_tag") != SEED_DOMAIN_TAG:
        raise ValueError("seed derivation algorithm mismatch")
    dimensions = payload.get("dimensions")
    table = payload.get("derived_seed_table")
    if not isinstance(dimensions, Mapping) or not isinstance(table, list):
        raise ValueError("seed manifest dimensions/table malformed")
    base_seeds = dimensions.get("base_seeds")
    outer_folds = dimensions.get("outer_folds")
    role_order = dimensions.get("role_order")
    if base_seeds != list(FROZEN_BASE_SEEDS) or outer_folds != list(
        FROZEN_OUTER_FOLDS
    ):
        raise ValueError("seed dimensions differ from the frozen 3x3 design")
    if not isinstance(role_order, list) or len(role_order) != 7:
        raise ValueError("seed role registry must contain exactly seven roles")
    roles: list[tuple[str, str, str]] = []
    for raw in role_order:
        if not isinstance(raw, Mapping):
            raise TypeError("seed role registry entry must be an object")
        roles.append((str(raw.get("mapping_key")), str(raw.get("artifact_role")), str(raw.get("oof_marker"))))
    if roles != list(FROZEN_SEED_ROLES):
        raise ValueError("seed role registry differs from the frozen seven roles")
    expected_rows = {
        (base_seed, outer_fold_id)
        for base_seed in FROZEN_BASE_SEEDS
        for outer_fold_id in FROZEN_OUTER_FOLDS
    }
    seen_rows: set[tuple[int, str]] = set()
    seen: set[tuple[int, str, str]] = set()
    derived: set[int] = set()
    for row in table:
        if not isinstance(row, Mapping):
            raise TypeError("seed table row must be an object")
        base = _exact_int(row.get("base_seed"), "seed table base_seed", minimum=0)
        outer = _nonempty(row.get("outer_fold_id"), "seed table outer_fold_id")
        row_identity = (base, outer)
        if row_identity not in expected_rows or row_identity in seen_rows:
            raise ValueError("seed table row identity is unexpected or duplicated")
        seen_rows.add(row_identity)
        values = row.get("derived_seeds_by_role")
        if not isinstance(values, Mapping) or set(values) != {item[0] for item in roles}:
            raise ValueError("seed table role keys mismatch")
        for key, artifact_role, marker in roles:
            actual = _exact_int(values[key], "derived seed", minimum=1, maximum=2147483646)
            expected = derive_stage2_seed(base, outer, artifact_role, marker)
            if actual != expected:
                raise ValueError("seed table does not replay SHA-256 derivation")
            identity = (base, outer, key)
            if identity in seen or actual in derived:
                raise ValueError("seed table contains duplicate identity or collision")
            seen.add(identity)
            derived.add(actual)
    if seen_rows != expected_rows or len(seen) != 63:
        raise ValueError("seed table must contain exactly 63 mappings")


def _lookup_seed(
    payload: Mapping[str, Any],
    *,
    base_seed: Any,
    outer_fold_id: str,
    detector_role: str,
    oof_fold_index: Any,
) -> int:
    key = (
        f"detector_oof::fold_{oof_fold_index}"
        if detector_role == "detector_oof"
        else "detector_full_fit::full_fit"
    )
    matches = [
        row
        for row in payload["derived_seed_table"]
        if row.get("base_seed") == base_seed and row.get("outer_fold_id") == outer_fold_id
    ]
    if len(matches) != 1:
        raise ValueError("run identity is absent or duplicated in seed manifest")
    return int(matches[0]["derived_seeds_by_role"][key])


def _verify_materialization_index(
    payload: Mapping[str, Any], root: Path, bound_artifacts: Mapping[str, str]
) -> None:
    if payload.get("schema_version") != MATERIALIZATION_INDEX_SCHEMA:
        raise ValueError("materialization-index schema mismatch")
    _common_result_free_guards(payload, "materialization index")
    artifacts = payload.get("artifacts_excluding_this_index")
    if not isinstance(artifacts, Mapping) or len(artifacts) != 52:
        raise ValueError("materialization index must bind exactly 52 artifacts")
    normalized = {
        _relative_path(path, "materialization artifact path"): _sha256(digest, "materialization artifact SHA-256")
        for path, digest in artifacts.items()
    }
    if normalized != dict(bound_artifacts):
        raise ValueError("run materialization artifact map differs from index")
    for path, expected in normalized.items():
        artifact = _repo_file(root, path, "materialization artifact")
        if sha256_file(artifact) != expected:
            raise ValueError(f"materialization artifact SHA-256 mismatch: {path}")
    if payload.get("official_test_ids_materialized") is not None:
        raise ValueError("ambiguous top-level official-test field in materialization index")


def _verify_detector_training(value: Any, config_binding: Mapping[str, str], root: Path) -> None:
    if not isinstance(value, dict):
        raise TypeError("run training must be an object")
    expected_keys = {
        "stage1_variant",
        "risk_objective",
        "tail_mode",
        "lambda_margin",
        "target_background_margin",
        "tail_q",
        "miss_q",
        "object_pixel_q",
        "tail_gamma",
        "peak_kernel_size",
        "exclusion_radius",
        "peak_min_score",
        "plateau_atol",
        "warm_epoch",
        "risk_warmup_epochs",
        "risk_ramp_epochs",
        "optimizer",
        "lr",
        "checkpoint_selection",
    }
    _exact_keys(value, expected_keys, "run training")
    expected = {
        "stage1_variant": "D3",
        "risk_objective": "margin",
        "tail_mode": "local-peak",
        "lambda_margin": 0.2,
        "target_background_margin": 1.0,
        "tail_q": 0.05,
        "miss_q": 0.25,
        "object_pixel_q": 0.25,
        "tail_gamma": 10.0,
        "peak_kernel_size": 5,
        "exclusion_radius": 2,
        "peak_min_score": 0.05,
        "plateau_atol": 0.0,
        "warm_epoch": 5,
        "risk_warmup_epochs": 5,
        "risk_ramp_epochs": 10,
        "optimizer": "Adagrad",
        "lr": 0.05,
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
    }
    if value != expected:
        raise ValueError("run training is not the exact frozen D3 objective")
    config, digest = _load_json_stable(
        _repo_file(root, config_binding["path"], "detector config"), "detector config"
    )
    if digest != config_binding["sha256"]:
        raise ValueError("detector config SHA-256 mismatch")
    config_expectations = {
        "risk_objective": expected["risk_objective"],
        "lambda_margin": expected["lambda_margin"],
        "target_background_margin_logit": expected["target_background_margin"],
        "background_tail_fraction": expected["tail_q"],
        "hard_object_fraction": expected["miss_q"],
        "object_top_pixel_fraction": expected["object_pixel_q"],
        "peak_kernel_size": expected["peak_kernel_size"],
        "gt_exclusion_radius": expected["exclusion_radius"],
        "plateau_atol": expected["plateau_atol"],
        "smooth_worst_domain_gamma": expected["tail_gamma"],
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
    }
    for key, expected_value in config_expectations.items():
        if config.get(key) != expected_value:
            raise ValueError(f"detector config D3 identity mismatch: {key}")
    schedule = config.get("training_schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("detector config training_schedule missing")
    for config_key, training_key in (
        ("warm_epoch", "warm_epoch"),
        ("risk_warmup_epochs", "risk_warmup_epochs"),
        ("risk_ramp_epochs", "risk_ramp_epochs"),
        ("optimizer", "optimizer"),
        ("lr", "lr"),
    ):
        if schedule.get(config_key) != expected[training_key]:
            raise ValueError(f"detector config schedule mismatch: {config_key}")


def _common_result_free_guards(payload: Mapping[str, Any], label: str) -> None:
    exact = {
        "artifact_status": "DEVELOPMENT_ONLY_RESULT_FREE",
        "development_only": True,
        "execution_authorized": False,
        "official_test_accessed": False,
        "observed_results": None,
    }
    # Older materialization/assignment artifacts carry the booleans in a
    # strict guardrails object.  New W01 contracts carry them at top level.
    if all(key in payload for key in exact):
        for key, expected in exact.items():
            actual = payload[key]
            if isinstance(expected, bool):
                _exact_bool(actual, f"{label}.{key}")
            if actual is not expected and actual != expected:
                raise ValueError(f"{label}.{key} must be exactly {expected!r}")
        return
    guards = payload.get("guardrails")
    if not isinstance(guards, Mapping):
        raise ValueError(f"{label} lacks strict result-free guardrails")
    for key, expected in {
        "development_only": True,
        "execution_authorized": False,
        "official_test_ids_materialized": False,
        "official_test_images_opened": False,
        "official_test_split_files_opened": False,
        "mask_or_label_files_opened": False,
        "predictions_scores_checkpoints_or_metrics_opened": False,
        "result_free": True,
    }.items():
        actual = guards.get(key)
        _exact_bool(actual, f"{label}.guardrails.{key}")
        if actual is not expected:
            raise ValueError(f"{label}.guardrails.{key} mismatch")
    if payload.get("execution_authorized") is not False or payload.get("observed_results") is not None:
        raise ValueError(f"{label} is not execution-HOLD/result-free")


def _binding(value: Any, label: str, *, allowed_extra: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    allowed = {"path", "sha256"} | (allowed_extra or set())
    if set(value) != allowed:
        raise ValueError(f"{label} keys mismatch: {sorted(set(value) ^ allowed)}")
    result = dict(value)
    result["path"] = _relative_path(result["path"], f"{label}.path")
    _assert_not_official_test_path(result["path"], f"{label}.path")
    result["sha256"] = _sha256(result["sha256"], f"{label}.sha256")
    return result


def _load_json_stable(path: Path, label: str) -> tuple[dict[str, Any], str]:
    before = sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    after = sha256_file(path)
    if before != after:
        raise RuntimeError(f"{label} changed while read: {path}")
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload, after


def _repository_root(value: str | Path | None) -> Path:
    root = REPOSITORY_ROOT if value is None else Path(value).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("repository_root must be a real directory")
    return root


def _input_file(value: str | Path, root: Path, label: str) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes repository root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    _reject_symlink_components(candidate, root, label)
    return resolved


def _repo_file(root: Path, value: Any, label: str) -> Path:
    relative = _relative_path(value, label)
    return _input_file(root / PurePosixPath(relative), root, label)


def _reject_symlink_components(candidate: Path, root: Path, label: str) -> None:
    absolute = candidate if candidate.is_absolute() else root / candidate
    try:
        relative = absolute.absolute().relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes repository root") from error
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {cursor}")


def _reject_incomplete_ancestor(path: Path, root: Path) -> None:
    cursor = path.parent
    while cursor == root or root in cursor.parents:
        marker = cursor / ".stage2_contract_materialization_incomplete"
        if marker.exists():
            raise RuntimeError(f"Stage2 contract tree is incomplete: {marker}")
        if cursor == root:
            break
        cursor = cursor.parent


def _relative_path(value: Any, label: str) -> str:
    text = _nonempty(value, label).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or text.startswith("/") or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a normalized repository-relative path")
    if path.as_posix() != text:
        raise ValueError(f"{label} is not canonical POSIX form")
    return text


def _assert_not_official_test_path(value: str, label: str) -> None:
    path = PurePosixPath(value.lower())
    for part in path.parts:
        stem = PurePosixPath(part).stem
        if part in {"test", "official_test", "official-test"} or stem.startswith("test_") or stem.startswith("official_test"):
            raise ValueError(f"{label} names an official-test-like path")


def _exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{label} keys mismatch: {sorted(set(value) ^ set(expected))}")


def _exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact JSON boolean")
    return value


def _exact_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact JSON integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} is below its minimum")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds its maximum")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError(f"{label} must be a non-empty exact string")
    return value


def _two_unique_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two domains")
    result = [_nonempty(item, f"{label} item") for item in value]
    if len(set(result)) != 2:
        raise ValueError(f"{label} domains must be unique")
    return result

"""Materialize the frozen 18 OOF + 9 full-fit Stage-2 detector contracts.

Only JSON/text manifest metadata is consumed.  The implementation does not
resolve or open dataset images, masks, split files, scores, checkpoints, or
metrics, and it has no official-test code path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from data_ext.dataset_identity import sha256_file
from data_ext.stage2_role_contract import (
    MATERIALIZATION_INDEX_SCHEMA,
    RECORDS_CONTENT_ALGORITHM,
    REPOSITORY_ROOT,
    SEED_MANIFEST_SCHEMA,
    SOURCE_THAW_PATH,
    SOURCE_THAW_SHA256,
    STAGE2_RUN_CONTRACT_SCHEMA,
    STAGE2_RUN_INDEX_SCHEMA,
    STAGE2_SELECTION_SCHEMA,
    WORK_BREAKDOWN_PATH,
    WORK_BREAKDOWN_SHA256,
    _common_result_free_guards,
    _expected_records_from_assignment,
    _input_file,
    _load_json_stable,
    _lookup_seed,
    _relative_path,
    _repo_file,
    _repository_root,
    _verify_detector_training,
    _verify_materialization_index,
    _verify_seed_manifest,
    canonical_json_sha256,
    load_stage2_selection,
    verify_stage2_run_contract,
)


DEFAULT_MATERIALIZATION_INDEX = (
    "outputs/stage2_manifests/rc4_k2_c14q28_20260716/materialization_index.json"
)
DEFAULT_SEED_MANIFEST = (
    "outputs/stage2_protocol/RC4_STAGE2_SEED_DERIVATION_MANIFEST_V1_20260716.json"
)
DEFAULT_DETECTOR_CONFIG = "configs/aaai27_detector_tail_sep.json"
INCOMPLETE_MARKER = ".stage2_contract_materialization_incomplete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize result-free Stage2 detector run contracts"
    )
    parser.add_argument("--materialization-index", required=True)
    parser.add_argument("--materialization-index-sha256", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--seed-manifest-sha256", required=True)
    parser.add_argument("--detector-config", required=True)
    parser.add_argument("--detector-config-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def materialize_stage2_detector_run_contracts(
    *,
    materialization_index: str | Path,
    materialization_index_sha256: str,
    seed_manifest: str | Path,
    seed_manifest_sha256: str,
    detector_config: str | Path,
    detector_config_sha256: str,
    output_root: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _repository_root(repository_root)
    materialization_path = _input_file(
        materialization_index, root, "materialization index"
    )
    seed_path = _input_file(seed_manifest, root, "seed manifest")
    config_path = _input_file(detector_config, root, "detector config")
    materialization_payload, materialization_digest = _load_json_stable(
        materialization_path, "materialization index"
    )
    seed_payload, seed_digest = _load_json_stable(seed_path, "seed manifest")
    _, config_digest = _load_json_stable(config_path, "detector config")
    if materialization_digest != materialization_index_sha256:
        raise ValueError("materialization-index CLI SHA-256 mismatch")
    if seed_digest != seed_manifest_sha256:
        raise ValueError("seed-manifest CLI SHA-256 mismatch")
    if config_digest != detector_config_sha256:
        raise ValueError("detector-config CLI SHA-256 mismatch")
    if materialization_payload.get("schema_version") != MATERIALIZATION_INDEX_SCHEMA:
        raise ValueError("materialization-index schema mismatch")
    if seed_payload.get("schema_version") != SEED_MANIFEST_SCHEMA:
        raise ValueError("seed-manifest schema mismatch")
    _verify_seed_manifest(seed_payload)
    raw_artifacts = materialization_payload.get("artifacts_excluding_this_index")
    if not isinstance(raw_artifacts, Mapping):
        raise ValueError("materialization index lacks artifact map")
    artifacts = {str(path): str(digest) for path, digest in raw_artifacts.items()}
    _verify_materialization_index(materialization_payload, root, artifacts)

    thaw_path = _repo_file(root, SOURCE_THAW_PATH.as_posix(), "source thaw")
    work_path = _repo_file(root, WORK_BREAKDOWN_PATH.as_posix(), "work breakdown")
    thaw, thaw_digest = _load_json_stable(thaw_path, "source thaw")
    if thaw_digest != SOURCE_THAW_SHA256 or sha256_file(work_path) != WORK_BREAKDOWN_SHA256:
        raise ValueError("Stage2 governance artifact drift")
    authorization = thaw.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("source thaw lacks authorization")
    if authorization.get("tracked_source_thaw_released") is not True:
        raise ValueError("tracked-source thaw is not released")
    if authorization.get("authorized_batch") != "B1_CONTRACT_SPINE":
        raise ValueError("source thaw does not authorize B1_CONTRACT_SPINE")
    if authorization.get("authorized_work_items") != ["W01", "W02", "W03", "W10", "W11"]:
        raise ValueError("source thaw work-item authorization drift")
    for forbidden in (
        "claim_bearing_training_authorized",
        "development_metric_inspection_for_architecture_selection_authorized",
        "stage2_real_execution_authorized",
        "official_test_authorized",
        "official_test_used",
    ):
        if authorization.get(forbidden) is not False:
            raise ValueError(f"source thaw {forbidden} must be exactly false")
    release = thaw.get("release_baseline")
    if not isinstance(release, Mapping):
        raise ValueError("source thaw lacks release_baseline")
    release_binding = {
        "path": SOURCE_THAW_PATH.as_posix(),
        "sha256": SOURCE_THAW_SHA256,
        "git_commit": str(release.get("git_head")),
        "tag": str(release.get("tag")),
        "source_archive": {
            "path": _relative_path(release.get("source_archive"), "source archive"),
            "sha256": str(release.get("source_archive_sha256")),
        },
    }
    if sha256_file(_repo_file(root, release_binding["source_archive"]["path"], "source archive")) != release_binding["source_archive"]["sha256"]:
        raise ValueError("frozen source archive SHA-256 mismatch")

    # Validate the exact D3 identity against the bound config before producing
    # any output.  The helper consumes no data artifacts.
    detector_config_binding = {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": config_digest,
    }
    training = frozen_d3_training_contract()
    _verify_detector_training(training, detector_config_binding, root)

    output = _new_output_root(output_root, root)
    output.mkdir(parents=True, exist_ok=False)
    marker = output / INCOMPLETE_MARKER
    marker.write_text(
        "result-free Stage2 contract materialization is incomplete\n",
        encoding="utf-8",
    )
    (output / "selections").mkdir()
    (output / "runs").mkdir()

    common_bindings = {
        "materialization_index": {
            "path": materialization_path.relative_to(root).as_posix(),
            "sha256": materialization_digest,
        },
        "seed_manifest": {
            "path": seed_path.relative_to(root).as_posix(),
            "sha256": seed_digest,
        },
        "detector_config": detector_config_binding,
        "release_artifact": release_binding,
        "implementation_work_breakdown": {
            "path": WORK_BREAKDOWN_PATH.as_posix(),
            "sha256": WORK_BREAKDOWN_SHA256,
        },
    }
    outer_indexes = materialization_payload.get("outer_fold_indexes")
    if not isinstance(outer_indexes, Mapping) or list(outer_indexes) != [
        "outer_leave_irstd_1k",
        "outer_leave_nuaa_sirst",
        "outer_leave_nudt_sirst",
    ]:
        # JSON key sorting is not the scientific order.  Require the exact key
        # set, then use the seed manifest's frozen order below.
        if set(outer_indexes or {}) != {
            "outer_leave_nuaa_sirst",
            "outer_leave_nudt_sirst",
            "outer_leave_irstd_1k",
        }:
            raise ValueError("materialization index outer-fold inventory mismatch")

    contracts: list[dict[str, Any]] = []
    selection_count = 0
    base_seeds = seed_payload["dimensions"]["base_seeds"]
    outer_order = seed_payload["dimensions"]["outer_folds"]
    for outer_fold_id in outer_order:
        outer_binding = outer_indexes[outer_fold_id]
        outer_path = _repo_file(root, outer_binding["path"], "outer-fold index")
        if sha256_file(outer_path) != outer_binding["sha256"]:
            raise ValueError("outer-fold index SHA-256 mismatch")
        outer, _ = _load_json_stable(outer_path, "outer-fold index")
        _common_result_free_guards(outer, "outer-fold index")
        if outer.get("outer_fold_id") != outer_fold_id:
            raise ValueError("outer-fold identity mismatch")
        source_domains = outer.get("source_domains")
        if not isinstance(source_domains, list) or len(source_domains) != 2 or len(set(source_domains)) != 2:
            raise ValueError("outer fold requires exactly two unique source domains")
        outer_target = outer.get("outer_target_domain")
        if outer_target in source_domains:
            raise ValueError("outer target appears in source domains")
        assignments = outer.get("assignments")
        if not isinstance(assignments, Mapping) or set(assignments) != set(source_domains):
            raise ValueError("outer-fold assignments differ from source domains")

        for base_seed in base_seeds:
            for detector_role, oof_fold_index in (
                ("detector_oof", 0),
                ("detector_oof", 1),
                ("detector_full_fit", None),
            ):
                role_suffix = (
                    f"oof_fold_{oof_fold_index}"
                    if detector_role == "detector_oof"
                    else "full_fit"
                )
                run_id = f"{outer_fold_id}__s{base_seed}__{role_suffix}"
                derived_seed = _lookup_seed(
                    seed_payload,
                    base_seed=base_seed,
                    outer_fold_id=outer_fold_id,
                    detector_role=detector_role,
                    oof_fold_index=oof_fold_index,
                )
                selection_role = (
                    "detector_oof_train"
                    if detector_role == "detector_oof"
                    else "detector_full_fit_train"
                )
                selection_bindings: list[dict[str, Any]] = []
                for source_domain in source_domains:
                    assignment_binding = assignments[source_domain]
                    assignment_path = _repo_file(
                        root, assignment_binding["path"], "assignment"
                    )
                    if sha256_file(assignment_path) != assignment_binding["sha256"]:
                        raise ValueError("assignment SHA-256 mismatch")
                    assignment, _ = _load_json_stable(assignment_path, "assignment")
                    records = _expected_records_from_assignment(
                        assignment,
                        source_domain=source_domain,
                        detector_role=detector_role,
                        oof_fold_index=oof_fold_index,
                    )
                    dataset_root = _dataset_root_from_records(records)
                    slug = _domain_slug(source_domain)
                    selection_dir = output / "selections" / run_id
                    selection_dir.mkdir(parents=True, exist_ok=False) if not selection_dir.exists() else None
                    id_path = selection_dir / f"{slug}.ids.txt"
                    _atomic_write_text(
                        id_path,
                        "".join(f"{record['image_id']}\n" for record in records),
                    )
                    id_digest = sha256_file(id_path)
                    selection_id = f"{run_id}::{slug}"
                    selection = {
                        "schema_version": STAGE2_SELECTION_SCHEMA,
                        "artifact_type": "rc_irstd_stage2_detector_selection",
                        "artifact_status": "DEVELOPMENT_ONLY_RESULT_FREE",
                        "development_only": True,
                        "execution_authorized": False,
                        "official_test_accessed": False,
                        "observed_results": None,
                        "selection_id": selection_id,
                        "run_id": run_id,
                        "selection_role": selection_role,
                        "detector_role": detector_role,
                        "outer_fold_id": outer_fold_id,
                        "outer_target_domain": outer_target,
                        "source_domain": source_domain,
                        "source_domains": list(source_domains),
                        "base_seed": base_seed,
                        "derived_seed": derived_seed,
                        "oof_fold_index": oof_fold_index,
                        "dataset_root": dataset_root,
                        "id_list": {
                            "path": id_path.relative_to(root).as_posix(),
                            "sha256": id_digest,
                            "format": "utf8_one_image_id_per_line_v1",
                        },
                        "records": records,
                        "record_count": len(records),
                        "records_content_algorithm": RECORDS_CONTENT_ALGORITHM,
                        "records_content_sha256": canonical_json_sha256(records),
                        "bindings": {
                            "assignment": {
                                "path": assignment_path.relative_to(root).as_posix(),
                                "sha256": assignment_binding["sha256"],
                            },
                            **common_bindings,
                        },
                    }
                    selection_path = selection_dir / f"{slug}.selection.json"
                    _atomic_write_json(selection_path, selection)
                    selection_digest = sha256_file(selection_path)
                    _write_sha_sidecar(selection_path, selection_digest)
                    selection_bindings.append(
                        {
                            "path": selection_path.relative_to(root).as_posix(),
                            "sha256": selection_digest,
                            "source_domain": source_domain,
                            "selection_role": selection_role,
                            "record_count": len(records),
                        }
                    )
                    selection_count += 1

                run_contract = {
                    "schema_version": STAGE2_RUN_CONTRACT_SCHEMA,
                    "artifact_type": "rc_irstd_stage2_detector_run_contract",
                    "artifact_status": "DEVELOPMENT_ONLY_RESULT_FREE",
                    "development_only": True,
                    "execution_authorized": False,
                    "official_test_accessed": False,
                    "observed_results": None,
                    "run_id": run_id,
                    "outer_fold_id": outer_fold_id,
                    "outer_target_domain": outer_target,
                    "source_domains": list(source_domains),
                    "base_seed": base_seed,
                    "derived_seed": derived_seed,
                    "detector_role": detector_role,
                    "oof_fold_index": oof_fold_index,
                    "selection_contracts": selection_bindings,
                    "training": training,
                    "bindings": {
                        **common_bindings,
                        "materialization_artifacts_sha256": artifacts,
                    },
                }
                run_path = output / "runs" / f"{run_id}.json"
                _atomic_write_json(run_path, run_contract)
                run_digest = sha256_file(run_path)
                _write_sha_sidecar(run_path, run_digest)
                contracts.append(
                    {
                        "run_id": run_id,
                        "path": run_path.relative_to(root).as_posix(),
                        "sha256": run_digest,
                        "outer_fold_id": outer_fold_id,
                        "outer_target_domain": outer_target,
                        "base_seed": base_seed,
                        "derived_seed": derived_seed,
                        "detector_role": detector_role,
                        "oof_fold_index": oof_fold_index,
                        "source_domains": list(source_domains),
                        "selection_contracts": selection_bindings,
                    }
                )

    if len(contracts) != 27 or selection_count != 54:
        raise RuntimeError("materializer did not produce exact 27 runs / 54 selections")
    if sum(item["detector_role"] == "detector_oof" for item in contracts) != 18:
        raise RuntimeError("materializer did not produce exactly 18 OOF contracts")
    if sum(item["detector_role"] == "detector_full_fit" for item in contracts) != 9:
        raise RuntimeError("materializer did not produce exactly 9 full-fit contracts")
    index = {
        "schema_version": STAGE2_RUN_INDEX_SCHEMA,
        "artifact_type": "rc_irstd_stage2_detector_run_contract_index",
        "artifact_status": "DEVELOPMENT_ONLY_RESULT_FREE",
        "development_only": True,
        "execution_authorized": False,
        "official_test_accessed": False,
        "observed_results": None,
        "run_count": 27,
        "oof_run_count": 18,
        "full_fit_run_count": 9,
        "selection_count": 54,
        "run_order": "outer_fold_seed_oof0_oof1_fullfit",
        "contracts": contracts,
        "bindings": {
            **common_bindings,
            "materialization_artifacts_sha256": artifacts,
        },
    }
    index_path = output / "run_contract_index.json"
    _atomic_write_json(index_path, index)
    index_digest = sha256_file(index_path)
    _write_sha_sidecar(index_path, index_digest)
    marker.unlink()
    try:
        # Consume the public APIs only after the incomplete marker is removed;
        # any verification failure restores the marker before propagating.
        for contract in contracts:
            verify_stage2_run_contract(
                _repo_file(root, contract["path"], "materialized run contract"),
                contract["sha256"],
                seed_path,
                materialization_path,
                repository_root=root,
            )
    except Exception:
        marker.write_text(
            "result-free Stage2 contract verification failed\n",
            encoding="utf-8",
        )
        raise
    return {**index, "index_path": index_path.relative_to(root).as_posix(), "index_sha256": index_digest}


def frozen_d3_training_contract() -> dict[str, Any]:
    return {
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


def _dataset_root_from_records(records: list[dict[str, Any]]) -> str:
    roots: set[str] = set()
    for record in records:
        path = PurePosixPath(str(record["original_image_path"]))
        if len(path.parts) < 3 or path.parts[-2] != "images":
            raise ValueError("assignment image path is not dataset_root/images/file")
        roots.add(PurePosixPath(*path.parts[:-2]).as_posix())
    if len(roots) != 1:
        raise ValueError("selection records do not share one dataset root")
    return next(iter(roots))


def _domain_slug(domain: str) -> str:
    slug = domain.lower().replace("-", "_")
    if not slug.replace("_", "").isalnum():
        raise ValueError("domain cannot be rendered as one safe slug")
    return slug


def _new_output_root(value: str | Path, root: Path) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("output root escapes repository root") from error
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"output root already exists: {resolved}")
    cursor = root
    for part in resolved.relative_to(root).parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("output root contains a symlink ancestor")
    return resolved


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_text(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_sha_sidecar(path: Path, digest: str) -> None:
    _atomic_write_text(
        path.with_suffix(path.suffix + ".sha256"),
        f"{digest}  {path.name}\n",
    )


def main() -> None:
    args = parse_args()
    result = materialize_stage2_detector_run_contracts(
        materialization_index=args.materialization_index,
        materialization_index_sha256=args.materialization_index_sha256,
        seed_manifest=args.seed_manifest,
        seed_manifest_sha256=args.seed_manifest_sha256,
        detector_config=args.detector_config,
        detector_config_sha256=args.detector_config_sha256,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": "PASS_RESULT_FREE_CONTRACT_MATERIALIZATION",
                "index_path": result["index_path"],
                "index_sha256": result["index_sha256"],
                "run_count": result["run_count"],
                "oof_run_count": result["oof_run_count"],
                "full_fit_run_count": result["full_fit_run_count"],
                "selection_count": result["selection_count"],
                "official_test_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

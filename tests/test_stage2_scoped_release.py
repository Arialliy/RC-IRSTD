from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import freeze_stage2_scoped_release as release


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def _bind(member_data: dict[str, bytes], path: str, data: bytes) -> dict[str, str]:
    member_data[path] = data
    return {"path": path, "sha256": release.sha256_bytes(data)}


def _synthetic_closure() -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    materialized: dict[str, str] = {}
    for index in range(52):
        path = f"synthetic/materialization/{index:02d}.json"
        data = _json_bytes({"index": index})
        members[path] = data
        materialized[path] = release.sha256_bytes(data)
    members[release.MATERIALIZATION_INDEX] = _json_bytes(
        {
            "artifact_count_excluding_this_index": 52,
            "artifacts_excluding_this_index": materialized,
        }
    )

    contracts: list[dict[str, object]] = []
    for index in range(27):
        id_path = f"synthetic/selections/{index:02d}.ids.txt"
        members[id_path] = f"synthetic-{index}\n".encode("ascii")
        selection_path = f"synthetic/selections/{index:02d}.json"
        selection = _json_bytes(
            {
                "id_list": {
                    "path": id_path,
                    "sha256": release.sha256_bytes(members[id_path]),
                }
            }
        )
        members[selection_path] = selection
        run_path = f"synthetic/runs/{index:02d}.json"
        run_data = _json_bytes({"run": index})
        members[run_path] = run_data
        contracts.append(
            {
                **_bind(members, run_path, run_data),
                "selection_contracts": [
                    {
                        "path": selection_path,
                        "sha256": release.sha256_bytes(selection),
                    }
                ],
            }
        )
    members[release.RUN_CONTRACT_INDEX] = _json_bytes(
        {
            "artifact_status": "DEVELOPMENT_ONLY_RESULT_FREE",
            "contracts": contracts,
        }
    )

    datasets = ("nuaa-sirst", "nudt-sirst", "irstd-1k")
    w12_rows: list[dict[str, str]] = []
    for offset, dataset in enumerate(datasets):
        metadata_path = release.W12_PREOPEN_FILES[1 + 2 * offset]
        plan_path = release.W12_PREOPEN_FILES[2 + 2 * offset]
        metadata = _json_bytes({"split_expected_record_count": 42})
        plan = _json_bytes({"execution_authorized": False})
        members[metadata_path] = metadata
        members[plan_path] = plan
        w12_rows.append(
            {
                "dataset": dataset,
                "metadata_path": metadata_path,
                "metadata_sha256": release.sha256_bytes(metadata),
                "plan_path": plan_path,
                "plan_sha256": release.sha256_bytes(plan),
            }
        )
    members[release.W12_PREOPEN_FILES[0]] = _json_bytes(
        {
            "artifact_status": "RESULT_FREE_PREOPEN_PLANS_FROZEN",
            "contains_observed_results": False,
            "execution_authorized": False,
            "datasets": w12_rows,
        }
    )
    for path in release.W12_PREOPEN_FILES:
        digest = release.sha256_bytes(members[path])
        members[path + ".sha256"] = release.sidecar_bytes(
            digest, Path(path).name
        )
    return members


def test_scoped_release_policy_never_recurses_official_id_bearing_trees() -> None:
    assert "audits/aaai27" not in release.TREE_RULES
    assert "splits/aaai27_v2" not in release.TREE_RULES
    assert release.SAFE_STATIC_METADATA_FILES == (
        "audits/aaai27/near_duplicates_effective_splits_v2.json",
        "splits/aaai27_v2/manifest.json",
    )


def test_final_authority_three_hash_binding_is_exact_and_mutation_fails() -> None:
    paths = set(release.FINAL_AUTHORITY_EXPECTED_SHA256) | set(
        release.LEGACY_NON_AUTHORITATIVE_STAGE2_CONFIGS
    )
    members = {path: release.stable_file_bytes(path) for path in paths}
    rows = release.verify_final_authority(members)
    assert {row["path"]: row["sha256"] for row in rows} == dict(
        release.FINAL_AUTHORITY_EXPECTED_SHA256
    )

    mutated = dict(members)
    mutated[release.AUTHORITATIVE_STAGE2_CONFIG] += b"\n"
    with pytest.raises(RuntimeError, match="external SHA mismatch"):
        release.verify_final_authority(mutated)


def test_index_closure_is_exact_and_missing_leaf_fails_closed() -> None:
    members = _synthetic_closure()
    verification = release.verify_index_and_sidecar_closure(members)
    assert verification["materialization_leaf_count"] == 52
    assert verification["run_contract_count"] == 27
    assert verification["selection_id_list_count"] == 27
    assert verification["w12_dataset_count"] == 3

    missing = dict(members)
    missing.pop("synthetic/materialization/00.json")
    with pytest.raises(RuntimeError, match="omitted"):
        release.verify_index_and_sidecar_closure(missing)


def test_atomic_publication_never_replaces_existing_target(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    source.mkdir()
    (source / "COMMIT.json").write_text("{}\n", encoding="utf-8")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        release.rename_noreplace(source, occupied)
    assert source.is_dir()
    assert occupied.is_dir()

    published = tmp_path / "published"
    release.rename_noreplace(source, published)
    assert not source.exists()
    assert (published / "COMMIT.json").is_file()

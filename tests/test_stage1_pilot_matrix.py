from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.validate_stage1_pilot_matrix import validate_matrix


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs" / "aaai27_stage1_pilot_matrix.json"
PLAN = ROOT / "configs" / "aaai27_analysis_plan.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _isolated_contract_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repository"
    (root / "configs").mkdir(parents=True)
    (root / "splits" / "aaai27_v2").mkdir(parents=True)
    for domain in ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"):
        (root / "datasets" / domain).mkdir(parents=True)

    for relative in (
        "configs/aaai27_stage1_pilot_matrix.json",
        "configs/aaai27_analysis_plan.json",
        "configs/aaai27_detector_tail_sep.json",
        "splits/aaai27_v2/manifest.json",
        "splits/aaai27_v2/nuaa-sirst/detector_fit.txt",
        "splits/aaai27_v2/nuaa-sirst/detector_diagnostic.txt",
        "splits/aaai27_v2/nudt-sirst/detector_fit.txt",
        "splits/aaai27_v2/nudt-sirst/detector_diagnostic.txt",
        "splits/aaai27_v2/irstd-1k/detector_fit.txt",
        "splits/aaai27_v2/irstd-1k/detector_diagnostic.txt",
    ):
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return (
        root,
        root / "configs" / "aaai27_stage1_pilot_matrix.json",
        root / "configs" / "aaai27_analysis_plan.json",
    )


def _write_mutation(
    matrix_path: Path,
    plan_path: Path,
    mutate: object,
) -> None:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(matrix)
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["hash_contracts"]["stage1_pilot_matrix"]["sha256"] = _sha256(matrix_path)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def test_current_stage1_pilot_matrix_is_exact_and_argv_compatible() -> None:
    report = validate_matrix(MATRIX, ROOT, plan_path=PLAN)
    assert report["status"] == "PASS"
    assert report["contains_observed_results"] is False
    assert report["run_count"] == 8
    assert report["phase_count"] == 4
    assert report["run_ids"] == [
        "D0_all-three_s42",
        "D3_all-three_s42",
        "D0_leave-NUAA_s42",
        "D0_leave-NUDT_s42",
        "D0_leave-IRSTD1K_s42",
        "D3_leave-NUAA_s42",
        "D3_leave-NUDT_s42",
        "D3_leave-IRSTD1K_s42",
    ]

    invocations = report["normalized_invocations"]
    assert [item["environment"]["CUDA_VISIBLE_DEVICES"] for item in invocations] == [
        "0,1,2",
        "0,1,2",
        "0",
        "1",
        "2",
        "0",
        "1",
        "2",
    ]
    for invocation in invocations:
        argv = invocation["python_argv"]
        split_start = argv.index("--source-split-files") + 1
        split_end = argv.index("--source-names")
        selected_splits = argv[split_start:split_end]
        assert selected_splits
        assert all(Path(path).is_absolute() for path in selected_splits)
        assert all(Path(path).name == "detector_fit.txt" for path in selected_splits)
        assert all(
            Path(path).name == "detector_diagnostic.txt"
            for path in invocation["evaluation_diagnostic_files"]
        )
        assert "--epoch-steps" not in argv
        assert "--deterministic" in argv
    assert "--data-parallel" in invocations[0]["python_argv"]
    assert "--data-parallel" in invocations[1]["python_argv"]
    assert all(
        "--data-parallel" not in item["python_argv"] for item in invocations[2:]
    )


def test_matrix_rejects_a_sealed_evaluation_path(tmp_path: Path) -> None:
    root, matrix_path, plan_path = _isolated_contract_root(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        payload["runs"][0]["evaluation_diagnostic_files"][0] = (
            "datasets/NUAA-SIRST/img_idx/test_NUAA-SIRST.txt"
        )

    _write_mutation(matrix_path, plan_path, mutate)
    with pytest.raises(ValueError, match="sealed evaluation token"):
        validate_matrix(matrix_path, root, plan_path=plan_path)


def test_matrix_rejects_held_out_domain_in_detector_sources(tmp_path: Path) -> None:
    root, matrix_path, plan_path = _isolated_contract_root(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        payload["runs"][2]["sources"][0] = "NUAA-SIRST"

    _write_mutation(matrix_path, plan_path, mutate)
    with pytest.raises(ValueError, match="run contract drift"):
        validate_matrix(matrix_path, root, plan_path=plan_path)


def test_matrix_rejects_all_three_and_lodo_schedule_overlap(tmp_path: Path) -> None:
    root, matrix_path, plan_path = _isolated_contract_root(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        payload["scheduling"]["phases"][2]["after"] = []

    _write_mutation(matrix_path, plan_path, mutate)
    with pytest.raises(ValueError, match="four-phase contract"):
        validate_matrix(matrix_path, root, plan_path=plan_path)


def test_matrix_requires_release_artifacts_only_at_release_gate(tmp_path: Path) -> None:
    root, matrix_path, plan_path = _isolated_contract_root(tmp_path)
    with pytest.raises(FileNotFoundError, match="release archive/checksum"):
        validate_matrix(
            matrix_path,
            root,
            plan_path=plan_path,
            require_release_artifacts=True,
        )

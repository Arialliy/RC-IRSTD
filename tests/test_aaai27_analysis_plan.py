from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.validate_aaai27_analysis_plan as analysis_plan_validator
from scripts.validate_aaai27_analysis_plan import (
    _assert_no_placeholders,
    load_strict_json,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "aaai27_analysis_plan.json"


def test_current_analysis_plan_is_valid_and_runtime_gated() -> None:
    report = validate_plan(PLAN, ROOT)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["contains_observed_results"] is False
    assert report["gate_minus_1"] is (
        not report["runtime_repository_state_blockers"]
    )
    assert report["development_domains"] == [
        "NUAA-SIRST",
        "NUDT-SIRST",
        "IRSTD-1K",
    ]
    assert report["meta_train_window_counts"] == {
        "NUAA-SIRST": 1,
        "NUDT-SIRST": 5,
        "IRSTD-1K": 6,
    }
    assert report["near_duplicate_summary"] == {
        "source_confirmed_pair_count": 31,
        "quarantined_train_id_count": 30,
        "effective_confirmed_pair_count": 0,
    }
    assert report["stage1_pilot_matrix_summary"] == {
        "sha256": "cb0763c99691057f061806c1680e115bbda39fade0fbe8400f76583bf7ba91b9",
        "run_count": 8,
        "phase_count": 4,
    }
    assert "FOURTH_INDEPENDENT_DOMAIN_ABSENT" in report["blocker_codes"]
    assert "NEAR_DUPLICATE_AUDIT_NOT_RUN" not in report["blocker_codes"]
    assert "STAGE2_BASELINE_RUNNERS_INCOMPLETE" in report[
        "scientific_blocker_codes"
    ]
    assert plan["data_roles"][
        "outer_target_official_train_used_for_detector_fit"
    ] is False
    assert plan["data_roles"][
        "outer_target_detector_diagnostic_used_for_development_evaluation"
    ] is True
    assert plan["data_roles"][
        "outer_target_diagnostic_selects_checkpoint"
    ] is False
    assert "outer_target_official_train_allowed_in_same_outer_fold" not in plan[
        "data_roles"
    ]


def test_gate_minus_one_fails_closed_for_dirty_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_git_output = analysis_plan_validator._git_output

    def dirty_git_output(
        repository_root: Path, *args: str
    ) -> subprocess.CompletedProcess[str]:
        if args == ("status", "--porcelain=v1"):
            return subprocess.CompletedProcess(
                args=["git", *args],
                returncode=0,
                stdout=" M simulated-change\n",
                stderr="",
            )
        return original_git_output(repository_root, *args)

    monkeypatch.setattr(analysis_plan_validator, "_git_output", dirty_git_output)
    report = analysis_plan_validator.validate_plan(PLAN, ROOT)

    assert report["status"] == "PASS"
    assert report["gate_minus_1"] is False
    assert report["runtime_repository_state_blockers"] == ["WORKTREE_NOT_CLEAN"]


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_strict_json(path)


@pytest.mark.parametrize("value", [None, "TBD", "todo: decide later", float("nan")])
def test_frozen_plan_rejects_placeholders_and_nonfinite_values(value: object) -> None:
    with pytest.raises(ValueError):
        _assert_no_placeholders({"field": value})


def test_analysis_plan_rejects_hash_drift(tmp_path: Path) -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["hash_contracts"]["stage1_config"]["sha256"] = "0" * 64
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash contract drift"):
        validate_plan(path, ROOT)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("outer_target_official_train_used_for_detector_fit", True),
        (
            "outer_target_detector_diagnostic_used_for_development_evaluation",
            False,
        ),
        ("outer_target_diagnostic_selects_checkpoint", True),
    ),
)
def test_analysis_plan_rejects_outer_target_role_drift(
    tmp_path: Path,
    field: str,
    invalid_value: bool,
) -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["data_roles"][field] = invalid_value
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="outer-target"):
        validate_plan(path, ROOT)


@pytest.mark.parametrize(
    "deprecated_field",
    (
        "outer_target_official_train_used",
        "outer_target_official_train_allowed_in_same_outer_fold",
    ),
)
def test_analysis_plan_rejects_deprecated_ambiguous_role_fields(
    tmp_path: Path,
    deprecated_field: str,
) -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["data_roles"][deprecated_field] = False
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="deprecated ambiguous role fields"):
        validate_plan(path, ROOT)

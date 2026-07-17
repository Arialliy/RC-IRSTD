from __future__ import annotations

from pathlib import Path
import sys

import pytest

from scripts import audit_stage2_model_design_freeze as auditor


def test_s2_i0_static_governance_config_model_and_source_checks_pass() -> None:
    governance, bindings = auditor._validate_governance()
    prerequisites, prerequisite_bindings = auditor._validate_frozen_prerequisites()
    authority, authority_bindings = auditor._validate_final_authority_files()
    config, config_binding = auditor._validate_config()
    models = auditor._validate_models()
    sources, source_bindings = auditor._validate_sources()
    checks = [*governance, *prerequisites, *authority, *config, *models, *sources]
    assert checks
    assert all(item["status"] == "PASS" for item in checks)
    assert len(bindings) == 2
    assert len(prerequisite_bindings) == len(auditor.FROZEN_PREREQUISITES)
    assert len(authority_bindings) == len(auditor.FINAL_AUTHORITY_FILES) == 3
    assert config_binding["path"] == "configs/aaai27_stage2_crossfit_v2.json"
    assert source_bindings


def test_s2_i0_auditor_has_no_real_gpu_or_official_input_flags() -> None:
    parser_source = (auditor.REPOSITORY_ROOT / "scripts/audit_stage2_model_design_freeze.py").read_text(
        encoding="utf-8"
    )
    assert "--dataset" not in parser_source
    assert "--checkpoint" not in parser_source
    assert "--gpu" not in parser_source
    assert "--official-test" not in parser_source
    assert auditor.PYTEST_FILES


def test_fixed_suite_has_explicit_w06_through_w13_and_transitive_regressions() -> None:
    assert tuple(auditor.WORK_PACKAGE_REQUIRED_NODES) == tuple(
        f"W{number:02d}" for number in range(6, 14)
    )
    assert all(auditor.WORK_PACKAGE_REQUIRED_NODES.values())
    assert {
        "tests/test_deployment_and_calibration_units.py",
        "tests/test_source_reference_streaming.py",
        "tests/test_risk_losses.py",
    }.issubset(auditor.PYTEST_FILES)


def test_unrelated_test_volume_cannot_replace_a_required_work_package_node() -> None:
    required = {
        node
        for nodes in auditor.WORK_PACKAGE_REQUIRED_NODES.values()
        for node in nodes
    } | set(auditor.OFFICIAL_SENTINEL_REQUIRED_NODES)
    assert all(
        check["status"] == "PASS"
        for check in auditor._coverage_checks(frozenset(required))
    )

    removed = auditor.WORK_PACKAGE_REQUIRED_NODES["W06"][0]
    substituted = (required - {removed}) | {
        f"tests/test_unrelated.py::test_unrelated_{index}" for index in range(200)
    }
    checks = auditor._coverage_checks(frozenset(substituted))
    w06 = next(check for check in checks if check["name"].endswith("::W06"))
    assert w06["status"] == "FAIL"
    assert w06["detail"]["missing"] == [removed]


def test_parameterized_required_node_collection_is_accepted() -> None:
    required = auditor.WORK_PACKAGE_REQUIRED_NODES["W06"][0]
    assert auditor._node_is_collected(required, frozenset({required + "[case-0]"}))


def test_main_returns_nonzero_and_prints_status_for_hold(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        auditor,
        "audit",
        lambda output_dir, **kwargs: (
            Path(output_dir) / "S2_I0_REPORT.json",
            "a" * 64,
            "b" * 64,
            "HOLD",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_stage2_model_design_freeze.py",
            "--output-dir",
            "hold",
            "--scoped-release-dir",
            "release",
            "--scoped-release-manifest-sha256",
            "a" * 64,
            "--scoped-release-archive-sha256",
            "b" * 64,
            "--scoped-release-environment-sha256",
            "c" * 64,
            "--scoped-release-commit-sha256",
            "d" * 64,
        ],
    )
    assert auditor.main() == 2
    assert '"status": "HOLD"' in capsys.readouterr().out


def test_source_audit_atomic_publication_never_replaces_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "staging"
    source.mkdir()
    (source / "COMMIT.json").write_text("{}\n", encoding="utf-8")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        auditor._rename_noreplace(source, occupied)
    assert source.is_dir()
    assert occupied.is_dir()

    published = tmp_path / "published"
    auditor._rename_noreplace(source, published)
    assert not source.exists()
    assert (published / "COMMIT.json").is_file()

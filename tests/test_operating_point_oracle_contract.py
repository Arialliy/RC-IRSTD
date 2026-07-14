from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.operating_point import main as operating_point_main
from evaluation.operating_point import select_budget_grid, select_operating_point
from evaluation.threshold_sweep import (
    ORACLE_DEPLOYMENT_STATUS,
    ORACLE_SELECTION_SCOPE,
    write_curve_csv,
)


def _rows() -> list[dict[str, float | int]]:
    return [
        {
            "threshold": 0.25,
            "pd": 1.0,
            "fa_pixel": 0.25,
            "fa_component_mp": 2.0,
            "tp_objects": 1,
            "gt_objects": 1,
            "pred_components": 3,
            "fp_components": 2,
            "fp_pixels": 1,
            "total_pixels": 4,
            "num_images": 1,
        },
        {
            "threshold": 0.75,
            "pd": 1.0,
            "fa_pixel": 0.0,
            "fa_component_mp": 0.0,
            "tp_objects": 1,
            "gt_objects": 1,
            "pred_components": 1,
            "fp_components": 0,
            "fp_pixels": 0,
            "total_pixels": 4,
            "num_images": 1,
        },
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_oracle_contract(payload: dict[str, object]) -> None:
    assert payload["oracle_only"] is True
    assert payload["selection_scope"] == ORACLE_SELECTION_SCOPE
    assert payload["deployable"] is False
    assert payload["deployment_status"] == ORACLE_DEPLOYMENT_STATUS
    assert payload["selection_uses_ground_truth_labels"] is True


def test_programmatic_operating_points_cannot_claim_deployability() -> None:
    # Adversarial extra fields in a row must not override the oracle contract.
    rows = _rows()
    rows[1]["deployable"] = True  # type: ignore[assignment]
    rows[1]["selection_scope"] = "deployment"  # type: ignore[assignment]
    point = select_operating_point(rows, pixel_budget=0.1)
    assert point is not None
    _assert_oracle_contract(point)
    assert point["selection_provenance_bound"] is False
    assert point["curve_sha256"] is None
    assert point["curve_manifest_sha256"] is None

    grid = select_budget_grid(
        rows,
        pixel_budgets=[0.1],
        component_budgets=[None],
    )
    _assert_oracle_contract(grid[0])
    assert isinstance(grid[0]["operating_point"], dict)
    _assert_oracle_contract(grid[0]["operating_point"])


def test_cli_binds_verified_curve_and_manifest_hashes(tmp_path: Path) -> None:
    curve_path = tmp_path / "curve.csv"
    write_curve_csv(
        _rows(),
        curve_path,
        diagnostic_legacy_combined=True,
    )
    manifest_path = curve_path.with_suffix(".csv.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _assert_oracle_contract(manifest)

    output_path = tmp_path / "selected.json"
    assert operating_point_main(
        [
            "--curve",
            str(curve_path),
            "--pixel-budget",
            "0.1",
            "--output",
            str(output_path),
        ]
    ) == 0
    selected = json.loads(output_path.read_text(encoding="utf-8"))
    _assert_oracle_contract(selected)
    assert selected["selection_provenance_bound"] is True
    assert selected["curve_sha256"] == _sha256(curve_path)
    assert selected["curve_manifest_sha256"] == _sha256(manifest_path)

    # A stale sibling manifest must fail closed rather than silently bind a
    # selected threshold to the wrong curve bytes.
    curve_path.write_text(
        curve_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match curve manifest"):
        operating_point_main(
            [
                "--curve",
                str(curve_path),
                "--pixel-budget",
                "0.1",
            ]
        )

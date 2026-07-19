from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import data_ext.stage2_detector_run_complete_v2 as run_complete_module
from data_ext.stage2_detector_run_complete_v2 import (
    RUN_COMPLETE_NAME,
    RUN_COMPLETE_SIDECAR_NAME,
    Stage2DetectorRunCompleteV2Error,
)


ROOT = Path(__file__).resolve().parents[1]
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_stage2_detector_run_complete_commit_test_helper",
    ROOT / "tests/test_stage2_detector_run_complete_v2.py",
)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPERS)


def test_failed_final_verifier_removes_commit_last_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _HELPERS._synthetic_run(tmp_path)

    def fail_final_verifier(*args: object, **kwargs: object):
        raise Stage2DetectorRunCompleteV2Error(
            "synthetic final verifier failure"
        )

    monkeypatch.setattr(
        run_complete_module,
        "verify_stage2_detector_run_complete_v2",
        fail_final_verifier,
    )
    with pytest.raises(Stage2DetectorRunCompleteV2Error, match="final verifier"):
        _HELPERS._publish(fixture)
    assert (fixture["run_dir"] / RUN_COMPLETE_NAME).is_file()
    assert not (fixture["run_dir"] / RUN_COMPLETE_SIDECAR_NAME).exists()


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
    "_stage2_detector_run_complete_precommit_test_helper",
    ROOT / "tests/test_stage2_detector_run_complete_v2.py",
)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPERS)


def test_precommit_replay_rejects_input_change_without_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _HELPERS._synthetic_run(tmp_path)
    real_builder = run_complete_module._build_expected_payload
    completed_replays = 0

    def mutate_after_first_replay(*args: object, **kwargs: object):
        nonlocal completed_replays
        result = real_builder(*args, **kwargs)
        completed_replays += 1
        if completed_replays == 1:
            metrics = fixture["run_dir"] / "metrics.jsonl"
            metrics.write_bytes(metrics.read_bytes() + b" ")
        return result

    monkeypatch.setattr(
        run_complete_module,
        "_build_expected_payload",
        mutate_after_first_replay,
    )
    with pytest.raises(Stage2DetectorRunCompleteV2Error, match="SHA-256"):
        _HELPERS._publish(fixture)
    assert completed_replays == 1
    assert (fixture["run_dir"] / RUN_COMPLETE_NAME).is_file()
    assert not (fixture["run_dir"] / RUN_COMPLETE_SIDECAR_NAME).exists()


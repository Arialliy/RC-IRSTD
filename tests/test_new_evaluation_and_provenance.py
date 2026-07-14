from __future__ import annotations

from pathlib import Path

import numpy as np

from rc_irstd.evaluation.irstd_metrics import evaluate_irstd_at_threshold
from rc_irstd.provenance.fingerprint import command_fingerprint, source_tree_fingerprint


def test_irstd_metrics_perfect_and_false_component() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:6, 5:7] = 1
    probability = mask.astype(np.float32)
    perfect = evaluate_irstd_at_threshold([probability], [mask], threshold=0.5)
    assert perfect.iou == 1.0
    assert perfect.niou == 1.0
    assert perfect.pd == 1.0
    assert perfect.false_components == 0

    noisy = probability.copy()
    noisy[13, 13] = 1.0
    result = evaluate_irstd_at_threshold([noisy], [mask], threshold=0.5)
    assert result.pd == 1.0
    assert result.false_components == 1
    assert result.false_components_per_mp > 0
    assert 0 < result.iou < 1


def test_fingerprint_changes_with_source_and_input(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    module = source / "module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text('{"a": 1}\n', encoding="utf-8")
    command = ["python", "run.py", "--config", str(config)]
    first, _ = command_fingerprint(command, tmp_path, source)
    assert first == command_fingerprint(command, tmp_path, source)[0]

    config.write_text('{"a": 2}\n', encoding="utf-8")
    second, _ = command_fingerprint(command, tmp_path, source)
    assert second != first

    before_source = source_tree_fingerprint(source)
    module.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_fingerprint(source) != before_source

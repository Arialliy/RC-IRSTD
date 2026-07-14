import json
from pathlib import Path

import yaml

from rc_irstd.pipelines.run_lodo import main


def test_lodo_separates_training_and_evaluation_strides(tmp_path: Path) -> None:
    config = {
        "python": "python",
        "working_directory": ".",
        "output_root": "outputs",
        "datasets": {
            "A": {"path": "data/A"},
            "B": {"path": "data/B"},
            "C": {"path": "data/C"},
        },
        "outer_targets": ["C"],
        "detector": {"name": "tiny", "device": "cpu", "amp": False},
        "episodes": {
            "context_size": 2,
            "horizon": 1,
            "train_stride": 1,
            "eval_stride": 3,
        },
    }
    config_path = tmp_path / "lodo.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    main([
        "--config",
        str(config_path),
        "--outer-target",
        "C",
        "--stages",
        "detector",
        "episodes",
        "--dry-run",
    ])

    protocol = json.loads((tmp_path / "outputs" / "protocol.json").read_text())
    resolved = protocol["resolved_episode_protocol"]
    assert resolved["pseudo_train_stride"] == 1
    assert resolved["target_eval_stride"] == 3

    command_log = (
        tmp_path / "outputs" / "outer_C" / "commands.log"
    ).read_text(encoding="utf-8")
    assert "target_episodes.npz --context-size 2 --horizon 1 --stride 3" in command_log
    assert command_log.count("--context-size 2 --horizon 1 --stride 1") == 2
    assert command_log.count("--source-train-split train") >= 3
    assert command_log.count("--source-val-split test") >= 3

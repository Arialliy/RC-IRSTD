from __future__ import annotations

import argparse
import hashlib
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from rc_irstd.provenance.fingerprint import command_fingerprint
from rc_irstd.provenance.manifest import load_run_manifest, write_run_manifest
from rc_irstd.utils.config import load_yaml
from rc_irstd.utils.io import atomic_json_dump, ensure_dir


ALL_STAGES = (
    "detector",
    "export",
    "episodes",
    "curve",
    "zero",
    "calibrate",
    "baselines",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the nested leave-one-domain-out RC-IRSTD protocol."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--outer-target",
        action="append",
        default=None,
        help="Run only the named outer target; repeat for several targets.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=list(ALL_STAGES),
    )
    parser.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a command when its declared output artifact already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


class CommandRunner:
    def __init__(
        self,
        log_path: Path,
        dry_run: bool,
        resume_existing: bool,
        working_directory: Path,
        environment: dict[str, str],
    ) -> None:
        self.log_path = log_path
        self.dry_run = dry_run
        self.resume_existing = resume_existing
        self.working_directory = working_directory
        self.environment = environment
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, command: list[str], expected: Path | None = None) -> None:
        printable = shlex.join(command)
        source_root = Path(__file__).resolve().parents[1]
        fingerprint, provenance = command_fingerprint(
            command, self.working_directory, source_root
        )
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"# cwd: {self.working_directory}\n")
            handle.write(f"# fingerprint: {fingerprint}\n")
            handle.write(printable + "\n")
        if expected is not None and expected.exists() and self.resume_existing:
            manifest = load_run_manifest(expected)
            if manifest is not None and manifest.get("fingerprint") == fingerprint:
                print(f"[skip:fingerprint-match] {expected}")
                return
            print(f"[rerun:stale-or-untracked] {expected}")
        print(f"[run] (cd {self.working_directory} && {printable})")
        if self.dry_run:
            return
        subprocess.run(
            command,
            check=True,
            cwd=self.working_directory,
            env=self.environment,
        )
        if expected is not None:
            if not expected.exists():
                raise RuntimeError(
                    f"Command completed but expected artifact is missing: {expected}"
                )
            write_run_manifest(expected, fingerprint, provenance)


def _require(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing configuration key: {key}")
    return config[key]


def _resolve_path(value: str | Path, base_directory: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _normalise_config(
    raw: dict[str, Any],
    config_directory: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(raw)
    raw_datasets = _require(config, "datasets")
    if not isinstance(raw_datasets, dict):
        raise TypeError("datasets must be a mapping")
    datasets: dict[str, dict[str, Any]] = {}
    for name, raw_item in raw_datasets.items():
        if isinstance(raw_item, (str, Path)):
            item: dict[str, Any] = {"path": str(raw_item)}
        elif isinstance(raw_item, dict):
            item = dict(raw_item)
        else:
            raise TypeError(
                f"Dataset '{name}' must be a path string or mapping, got "
                f"{type(raw_item).__name__}"
            )
        if "path" not in item:
            raise KeyError(f"Dataset '{name}' is missing path")
        item["path"] = str(_resolve_path(item["path"], config_directory))
        item.setdefault("train_split", "train")
        item.setdefault("eval_split", "test")
        datasets[str(name)] = item
    config["datasets"] = datasets

    config["output_root"] = str(
        _resolve_path(config.get("output_root", "outputs/lodo"), config_directory)
    )
    config["working_directory"] = str(
        _resolve_path(config.get("working_directory", "."), config_directory)
    )
    return config


def _dataset(config: dict[str, Any], name: str) -> dict[str, Any]:
    datasets = _require(config, "datasets")
    if name not in datasets:
        raise KeyError(f"Unknown dataset '{name}'")
    return dict(datasets[name])


def _extend_optional(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _detector_cache_directory(
    output_root: Path,
    source_names: list[str],
    detector_cfg: dict[str, Any],
) -> Path:
    payload = json.dumps(
        {
            "sources": sorted(source_names),
            "detector": detector_cfg,
        },
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    key = hashlib.sha256(payload).hexdigest()[:16]
    directory = ensure_dir(output_root / "detector_cache" / key)
    atomic_json_dump(
        {"key": key, "sources": sorted(source_names), "detector": detector_cfg},
        directory / "source_set.json",
    )
    return directory


def _train_detector_command(
    python: str,
    detector_cfg: dict[str, Any],
    datasets_cfg: dict[str, Any],
    source_names: list[str],
    output_dir: Path,
) -> list[str]:
    if not source_names:
        raise ValueError("Detector training requires at least one source domain")
    command = [python, "-m", "rc_irstd.pipelines.train_detector"]
    for name in source_names:
        dataset_cfg = datasets_cfg[name]
        command.extend(
            [
                "--source-dataset",
                str(dataset_cfg["path"]),
                "--source-train-split",
                str(dataset_cfg.get("train_split", detector_cfg.get("train_split", "train"))),
                "--source-val-split",
                str(dataset_cfg.get("eval_split", detector_cfg.get("val_split", "test"))),
            ]
        )
    resize = detector_cfg.get("resize", [256, 256])
    command.extend(
        [
            "--detector",
            str(detector_cfg.get("name", "mshnet")),
            "--base-loss",
            str(detector_cfg.get("base_loss", "auto")),
            "--resize",
            str(resize[0]),
            str(resize[1]),
            "--batch-size",
            str(int(detector_cfg.get("per_domain_batch", 2)) * len(source_names)),
            "--epochs",
            str(detector_cfg.get("epochs", 400)),
            "--warm-epoch",
            str(detector_cfg.get("warm_epoch", 5)),
            "--optimizer",
            str(detector_cfg.get("optimizer", "adagrad")),
            "--lr",
            str(detector_cfg.get("lr", 0.05)),
            "--weight-decay",
            str(detector_cfg.get("weight_decay", 0.0)),
            "--lambda-tail",
            str(detector_cfg.get("lambda_tail", 0.1)),
            "--lambda-miss",
            str(detector_cfg.get("lambda_miss", 0.1)),
            "--tail-quantile",
            str(detector_cfg.get("tail_quantile", 0.95)),
            "--miss-quantile",
            str(detector_cfg.get("miss_quantile", 0.8)),
            "--peak-kernel",
            str(detector_cfg.get("peak_kernel", 5)),
            "--exclusion-radius",
            str(detector_cfg.get("exclusion_radius", 2)),
            "--worst-gamma",
            str(detector_cfg.get("worst_gamma", 10.0)),
            "--auxiliary-weight",
            str(detector_cfg.get("auxiliary_weight", 1.0)),
            "--pixel-budget",
            str(detector_cfg.get("pixel_budget", 1e-5)),
            "--peak-budget",
            str(detector_cfg.get("peak_budget", 5.0)),
            "--normalization",
            str(detector_cfg.get("normalization", "imagenet")),
            "--dataset-type",
            str(detector_cfg.get("dataset_type", "iid_images")),
            "--num-workers",
            str(detector_cfg.get("num_workers", 4)),
            "--device",
            str(detector_cfg.get("device", "auto")),
            "--seed",
            str(detector_cfg.get("seed", 42)),
            "--grad-clip",
            str(detector_cfg.get("grad_clip", 5.0)),
            "--val-every",
            str(detector_cfg.get("val_every", 1)),
            "--save-every",
            str(detector_cfg.get("save_every", 20)),
            "--output-dir",
            str(output_dir),
        ]
    )
    command.append("--amp" if bool(detector_cfg.get("amp", True)) else "--no-amp")
    command.append(
        "--deterministic"
        if bool(detector_cfg.get("deterministic", True))
        else "--no-deterministic"
    )
    return command


def _export_command(
    python: str,
    detector_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    checkpoint: Path,
    output_dir: Path,
) -> list[str]:
    resize = detector_cfg.get("resize", [256, 256])
    tile_size = detector_cfg.get("tile_size", [512, 512])
    command = [
        python,
        "-m",
        "rc_irstd.pipelines.export_scores",
        "--dataset-dir",
        str(dataset_cfg["path"]),
        "--split",
        str(dataset_cfg.get("eval_split", "test")),
        "--detector",
        str(detector_cfg.get("name", "mshnet")),
        "--checkpoint",
        str(checkpoint),
        "--inference-mode",
        str(detector_cfg.get("inference_mode", "resize")),
        "--normalization",
        str(detector_cfg.get("normalization", "imagenet")),
        "--dataset-type",
        str(dataset_cfg.get("dataset_type", detector_cfg.get("dataset_type", "iid_images"))),
        "--resize",
        str(resize[0]),
        str(resize[1]),
        "--stride-multiple",
        str(detector_cfg.get("stride_multiple", 32)),
        "--tile-size",
        str(tile_size[0]),
        str(tile_size[1]),
        "--tile-overlap",
        str(detector_cfg.get("tile_overlap", 64)),
        "--include-mask",
        "--num-workers",
        str(detector_cfg.get("num_workers", 4)),
        "--device",
        str(detector_cfg.get("device", "auto")),
        "--seed",
        str(detector_cfg.get("seed", 42)),
        "--output-dir",
        str(output_dir),
    ]
    command.append(
        "--restore-original"
        if bool(detector_cfg.get("restore_original", True))
        else "--no-restore-original"
    )
    return command


def _episode_command(
    python: str,
    episode_cfg: dict[str, Any],
    score_dir: Path,
    output: Path,
) -> list[str]:
    command = [
        python,
        "-m",
        "rc_irstd.pipelines.build_episodes",
        "--score-dir",
        str(score_dir),
        "--output",
        str(output),
        "--context-size",
        str(episode_cfg.get("context_size", 32)),
        "--horizon",
        str(episode_cfg.get("horizon", 16)),
        "--stride",
        str(episode_cfg.get("stride", 48)),
        "--protocol",
        str(episode_cfg.get("protocol", "auto")),
        "--seed",
        str(episode_cfg.get("seed", 0)),
        "--peak-min-distance",
        str(episode_cfg.get("peak_min_distance", 2)),
        "--peak-min-score",
        str(episode_cfg.get("peak_min_score", 0.0)),
        "--peak-border",
        str(episode_cfg.get("peak_border", 0)),
        "--peak-tolerance",
        str(episode_cfg.get("peak_tolerance", 2.0)),
        "--max-candidates",
        str(episode_cfg.get("max_candidates", 0)),
    ]
    _extend_optional(command, "--threshold-grid", episode_cfg.get("threshold_grid"))
    return command


def _curve_command(
    python: str,
    curve_cfg: dict[str, Any],
    budget_cfg: dict[str, Any],
    episode_files: list[Path],
    output_dir: Path,
) -> list[str]:
    command = [python, "-m", "rc_irstd.pipelines.train_curve"]
    for episode in episode_files:
        command.extend(["--train-episode", str(episode)])
    command.extend(
        [
            "--val-fraction",
            str(curve_cfg.get("val_fraction", 0.2)),
            "--quantile",
            str(curve_cfg.get("quantile", 0.9)),
            "--hidden-dim",
            str(curve_cfg.get("hidden_dim", 256)),
            "--dropout",
            str(curve_cfg.get("dropout", 0.1)),
            "--lambda-peak",
            str(curve_cfg.get("lambda_peak", 1.0)),
            "--lambda-crossing",
            str(curve_cfg.get("lambda_crossing", 0.25)),
            "--crossing-temperature",
            str(curve_cfg.get("crossing_temperature", 0.25)),
            "--focus-base-weight",
            str(curve_cfg.get("focus_base_weight", 1.0)),
            "--focus-weight",
            str(curve_cfg.get("focus_weight", 4.0)),
            "--focus-log-scale",
            str(curve_cfg.get("focus_log_scale", 1.0)),
            "--empty-action-weight",
            str(curve_cfg.get("empty_action_weight", 0.1)),
            "--batch-size",
            str(curve_cfg.get("batch_size", 64)),
            "--epochs",
            str(curve_cfg.get("epochs", 300)),
            "--lr",
            str(curve_cfg.get("lr", 1e-3)),
            "--weight-decay",
            str(curve_cfg.get("weight_decay", 1e-4)),
            "--patience",
            str(curve_cfg.get("patience", 40)),
            "--pixel-budget",
            str(budget_cfg.get("pixel", 1e-6)),
            "--peak-budget",
            str(budget_cfg.get("peak_per_mp", 1.0)),
            "--num-workers",
            str(curve_cfg.get("num_workers", 0)),
            "--device",
            str(curve_cfg.get("device", "auto")),
            "--seed",
            str(curve_cfg.get("seed", 42)),
            "--output-dir",
            str(output_dir),
        ]
    )
    return command


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = _normalise_config(load_yaml(config_path), config_path.parent)

    python = str(config.get("python", sys.executable))
    output_root = ensure_dir(config["output_root"])
    working_directory = Path(config["working_directory"])
    if not working_directory.is_dir():
        raise FileNotFoundError(
            f"working_directory does not exist: {working_directory}. Point it "
            "to the RC-IRSTD project root."
        )

    package_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    python_path_entries = [str(package_root), str(working_directory)]
    existing_pythonpath = environment.get("PYTHONPATH")
    if existing_pythonpath:
        python_path_entries.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(python_path_entries)

    datasets_cfg = _require(config, "datasets")
    all_domains = list(datasets_cfg)
    if len(all_domains) < 3:
        raise ValueError("Nested LODO requires at least three domains")
    outer_targets = args.outer_target or config.get("outer_targets", all_domains)
    unknown = set(outer_targets).difference(all_domains)
    if unknown:
        raise KeyError(f"Unknown outer targets: {sorted(unknown)}")

    detector_cfg = dict(config.get("detector", {}))
    episode_cfg = dict(config.get("episodes", {}))
    episode_total = int(episode_cfg.get("context_size", 32)) + int(
        episode_cfg.get("horizon", 16)
    )
    pseudo_episode_cfg = dict(episode_cfg)
    pseudo_episode_cfg["stride"] = int(
        episode_cfg.get("train_stride", episode_cfg.get("stride", 16))
    )
    target_episode_cfg = dict(episode_cfg)
    target_episode_cfg["stride"] = int(
        episode_cfg.get(
            "eval_stride",
            episode_cfg.get("target_stride", max(episode_total, int(episode_cfg.get("stride", episode_total)))),
        )
    )
    if target_episode_cfg["stride"] < episode_total:
        raise ValueError(
            "Formal target calibration/test episodes require eval_stride >= "
            "context_size + horizon so windows do not share images."
        )
    curve_cfg = dict(config.get("curve", {}))
    curve_cfg.setdefault("device", detector_cfg.get("device", "auto"))
    budget_cfg = dict(config.get("budgets", {}))
    detector_cfg.setdefault("pixel_budget", budget_cfg.get("pixel", 1e-5))
    detector_cfg.setdefault("peak_budget", budget_cfg.get("peak_per_mp", 5.0))
    calibration_cfg = dict(config.get("calibration", {}))
    stages = set(args.stages)

    protocol_manifest = {
        "protocol": "nested_leave_one_domain_out",
        "config_path": str(config_path),
        "working_directory": str(working_directory),
        "outer_targets": outer_targets,
        "all_domains": all_domains,
        "stages": list(args.stages),
        "leakage_rule": (
            "For outer target t and pseudo-target p, the episode detector is "
            "trained only on domains excluding both t and p. Warm-up context and "
            "future risk windows are disjoint. Calibration and test sequences are "
            "disjoint."
        ),
        "risk_definition": (
            "pixel false rate and threshold-independent fixed false local peaks "
            "per megapixel"
        ),
        "resolved_episode_protocol": {
            "pseudo_train_stride": pseudo_episode_cfg["stride"],
            "target_eval_stride": target_episode_cfg["stride"],
            "context_size": episode_cfg.get("context_size", 32),
            "horizon": episode_cfg.get("horizon", 16),
        },
        "config": config,
    }
    atomic_json_dump(protocol_manifest, output_root / "protocol.json")

    for outer in outer_targets:
        outer_dir = ensure_dir(output_root / f"outer_{outer}")
        runner = CommandRunner(
            outer_dir / "commands.log",
            dry_run=args.dry_run,
            resume_existing=args.resume_existing,
            working_directory=working_directory,
            environment=environment,
        )
        sources = [name for name in all_domains if name != outer]

        # Final detector: trained on every non-target source domain.
        final_detector_dir = _detector_cache_directory(
            output_root, sources, detector_cfg
        )
        final_checkpoint = final_detector_dir / "best.pt"
        if "detector" in stages:
            runner.run(
                _train_detector_command(
                    python,
                    detector_cfg,
                    datasets_cfg,
                    sources,
                    final_detector_dir,
                ),
                expected=final_checkpoint,
            )

        # Target score records and labelled episodes are used only for offline
        # evaluation. True deployment uses predict_unlabeled.py without masks.
        target_scores = outer_dir / "target_scores"
        if "export" in stages or "baselines" in stages:
            runner.run(
                _export_command(
                    python,
                    detector_cfg,
                    _dataset(config, outer),
                    final_checkpoint,
                    target_scores,
                ),
                expected=target_scores / "manifest.json",
            )
        target_episode = outer_dir / "target_episodes.npz"
        if "episodes" in stages or "baselines" in stages:
            runner.run(
                _episode_command(python, target_episode_cfg, target_scores, target_episode),
                expected=target_episode,
            )

        # Inner pseudo-target episodes. Excluding both outer and pseudo domains
        # prevents target identity or pseudo-target labels from entering detector
        # training for that episode family.
        pseudo_episode_files: list[Path] = []
        for pseudo in sources:
            pseudo_dir = ensure_dir(outer_dir / "pseudo" / pseudo)
            inner_sources = [
                name for name in all_domains if name not in {outer, pseudo}
            ]
            inner_detector_dir = _detector_cache_directory(
                output_root, inner_sources, detector_cfg
            )
            inner_checkpoint = inner_detector_dir / "best.pt"
            if "detector" in stages:
                runner.run(
                    _train_detector_command(
                        python,
                        detector_cfg,
                        datasets_cfg,
                        inner_sources,
                        inner_detector_dir,
                    ),
                    expected=inner_checkpoint,
                )
            pseudo_scores = pseudo_dir / "scores"
            if "export" in stages:
                runner.run(
                    _export_command(
                        python,
                        detector_cfg,
                        _dataset(config, pseudo),
                        inner_checkpoint,
                        pseudo_scores,
                    ),
                    expected=pseudo_scores / "manifest.json",
                )
            pseudo_episode = pseudo_dir / "episodes.npz"
            pseudo_episode_files.append(pseudo_episode)
            if "episodes" in stages:
                runner.run(
                    _episode_command(
                        python,
                        pseudo_episode_cfg,
                        pseudo_scores,
                        pseudo_episode,
                    ),
                    expected=pseudo_episode,
                )

        if "baselines" in stages:
            final_source_episodes: list[Path] = []
            for source in sources:
                source_root = ensure_dir(outer_dir / "final_source" / source)
                source_scores = source_root / "scores"
                runner.run(
                    _export_command(
                        python,
                        detector_cfg,
                        _dataset(config, source),
                        final_checkpoint,
                        source_scores,
                    ),
                    expected=source_scores / "manifest.json",
                )
                source_episode = source_root / "episodes.npz"
                runner.run(
                    _episode_command(
                        python,
                        target_episode_cfg,
                        source_scores,
                        source_episode,
                    ),
                    expected=source_episode,
                )
                final_source_episodes.append(source_episode)

            baseline_command = [
                python,
                "-m",
                "rc_irstd.pipelines.evaluate_baselines",
                "--target-episode",
                str(target_episode),
            ]
            for source_episode in final_source_episodes:
                baseline_command.extend(["--source-episode", str(source_episode)])
            baseline_command.extend(
                [
                    "--pixel-budget",
                    str(budget_cfg.get("pixel", 1e-6)),
                    "--peak-budget",
                    str(budget_cfg.get("peak_per_mp", 1.0)),
                    "--output-dir",
                    str(outer_dir / "baselines"),
                ]
            )
            runner.run(
                baseline_command,
                expected=outer_dir / "baselines" / "summary.json",
            )

        curve_dir = outer_dir / "risk_curve"
        curve_checkpoint = curve_dir / "best.pt"
        if "curve" in stages:
            runner.run(
                _curve_command(
                    python,
                    curve_cfg,
                    budget_cfg,
                    pseudo_episode_files,
                    curve_dir,
                ),
                expected=curve_checkpoint,
            )

        if "zero" in stages:
            runner.run(
                [
                    python,
                    "-m",
                    "rc_irstd.pipelines.evaluate_zero_label",
                    "--episode",
                    str(target_episode),
                    "--curve-checkpoint",
                    str(curve_checkpoint),
                    "--pixel-budget",
                    str(budget_cfg.get("pixel", 1e-6)),
                    "--peak-budget",
                    str(budget_cfg.get("peak_per_mp", 1.0)),
                    "--device",
                    str(curve_cfg.get("device", "auto")),
                    "--output-dir",
                    str(outer_dir / "zero_label"),
                ],
                expected=outer_dir / "zero_label" / "summary.json",
            )

        if "calibrate" in stages:
            command = [
                python,
                "-m",
                "rc_irstd.pipelines.calibrate_and_evaluate",
                "--episode",
                str(target_episode),
                "--curve-checkpoint",
                str(curve_checkpoint),
                "--pixel-budget",
                str(budget_cfg.get("pixel", 1e-6)),
                "--peak-budget",
                str(budget_cfg.get("peak_per_mp", 1.0)),
                "--alpha",
                str(calibration_cfg.get("alpha", 0.1)),
                "--calibration-sizes",
                *[
                    str(value)
                    for value in calibration_cfg.get("sizes", [10, 20, 50])
                ],
                "--seeds",
                *[
                    str(value)
                    for value in calibration_cfg.get("seeds", [0, 1, 2, 3, 4])
                ],
                "--calibration-unit",
                str(calibration_cfg.get("unit", "image")),
                "--offset-step",
                str(calibration_cfg.get("offset_step", 1)),
                "--device",
                str(curve_cfg.get("device", "auto")),
                "--output-dir",
                str(outer_dir / "few_shot_crc"),
            ]
            runner.run(
                command,
                expected=outer_dir / "few_shot_crc" / "summary.json",
            )

    print(json.dumps(protocol_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

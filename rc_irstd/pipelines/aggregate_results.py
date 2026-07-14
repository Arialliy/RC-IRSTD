from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rc_irstd.utils.io import atomic_json_dump, ensure_dir


METRICS = [
    "joint_bsr",
    "pixel_bsr",
    "peak_bsr",
    "pixel_excess",
    "peak_excess",
    "mean_pd_selected",
    "effective_pd_with_rejects",
    "conditional_pd_non_rejected",
    "worst_domain_pd_selected",
    "rejection_rate",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate nested-LODO outputs into paper-ready CSV and Markdown tables."
    )
    parser.add_argument("--lodo-root", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _outer_name(path: Path) -> str:
    return path.name.removeprefix("outer_")


def _zero_rows(outer_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outer_dir in outer_dirs:
        path = outer_dir / "zero_label" / "summary.json"
        if not path.is_file():
            continue
        payload = _read_json(path)
        selected = payload.get("metrics", {}).get("selected", {})
        rows.append(
            {
                "outer_target": _outer_name(outer_dir),
                "method": "risk_curve_zero_label",
                **selected,
                "pixel_log_mae": payload.get("metrics", {}).get("pixel_log_mae"),
                "peak_log_mae": payload.get("metrics", {}).get("peak_log_mae"),
                "joint_pointwise_coverage": payload.get("metrics", {}).get(
                    "joint_pointwise_coverage"
                ),
            }
        )
    return rows


def _baseline_rows(outer_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outer_dir in outer_dirs:
        path = outer_dir / "baselines" / "summary.json"
        if not path.is_file():
            continue
        payload = _read_json(path)
        for method, metrics in payload.get("methods", {}).items():
            rows.append(
                {
                    "outer_target": _outer_name(outer_dir),
                    "method": method,
                    **metrics,
                }
            )
    return rows


def _crc_rows(outer_dirs: list[Path]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for outer_dir in outer_dirs:
        path = outer_dir / "few_shot_crc" / "results.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "outer_target", _outer_name(outer_dir))
        frames.append(frame)
    return frames


def _write_markdown(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.lodo_root).expanduser().resolve()
    output = ensure_dir(
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "paper_tables"
    )
    outer_dirs = sorted(path for path in root.glob("outer_*") if path.is_dir())
    if not outer_dirs:
        raise FileNotFoundError(f"No outer_* directories found under {root}")

    manifest: dict[str, Any] = {
        "lodo_root": str(root),
        "outer_targets": [_outer_name(path) for path in outer_dirs],
        "files": {},
    }

    zero = pd.DataFrame(_zero_rows(outer_dirs))
    if not zero.empty:
        zero_path = output / "zero_label_by_domain.csv"
        zero.to_csv(zero_path, index=False)
        summary = zero.groupby("method", dropna=False)[METRICS].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        summary_path = output / "zero_label_summary.csv"
        summary.to_csv(summary_path, index=False)
        _write_markdown(summary, output / "zero_label_summary.md")
        manifest["files"]["zero_label"] = [str(zero_path), str(summary_path)]

    baselines = pd.DataFrame(_baseline_rows(outer_dirs))
    if not baselines.empty:
        baseline_path = output / "baselines_by_domain.csv"
        baselines.to_csv(baseline_path, index=False)
        summary = baselines.groupby("method", dropna=False)[METRICS].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        summary_path = output / "baselines_summary.csv"
        summary.to_csv(summary_path, index=False)
        _write_markdown(summary, output / "baselines_summary.md")
        manifest["files"]["baselines"] = [str(baseline_path), str(summary_path)]

    crc_frames = _crc_rows(outer_dirs)
    if crc_frames:
        crc = pd.concat(crc_frames, ignore_index=True)
        crc_path = output / "few_shot_crc_all_runs.csv"
        crc.to_csv(crc_path, index=False)
        group_keys = ["method", "calibration_size"]
        available_metrics = [metric for metric in METRICS if metric in crc.columns]
        summary = crc.groupby(group_keys, dropna=False)[available_metrics].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        feasibility = (
            crc.groupby(group_keys, dropna=False)["formal_crc_feasible"]
            .mean()
            .reset_index(name="formal_feasible_fraction")
        )
        summary = summary.merge(feasibility, on=group_keys, how="left")
        summary_path = output / "few_shot_crc_summary.csv"
        summary.to_csv(summary_path, index=False)
        _write_markdown(summary, output / "few_shot_crc_summary.md")
        manifest["files"]["few_shot_crc"] = [str(crc_path), str(summary_path)]

    atomic_json_dump(manifest, output / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

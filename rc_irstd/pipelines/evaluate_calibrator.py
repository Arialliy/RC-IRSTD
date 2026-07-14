from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from rc_irstd.episodes.meta_dataset import concatenate_meta_episode_files
from rc_irstd.evaluation.calibrator_replay import HardReplayEvaluator
from rc_irstd.models.calibrator_io import load_monotone_calibrator, predict_threshold_curve
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact hard replay of the no-reject calibrator.")
    parser.add_argument("--meta", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    arrays = concatenate_meta_episode_files(args.meta)
    loaded = load_monotone_calibrator(args.checkpoint, device)
    if arrays.feature_names != loaded.feature_names:
        raise ValueError("Meta feature schema differs from checkpoint")
    if not np.array_equal(arrays.budgets, loaded.budgets):
        raise ValueError("Meta budget grid differs from checkpoint")
    eta = predict_threshold_curve(
        loaded,
        arrays.features,
        device,
        source_distances=arrays.source_distances,
    )
    summary = HardReplayEvaluator(arrays).evaluate(eta)
    output_dir = ensure_dir(args.output_dir)
    atomic_json_dump(summary.to_dict(), output_dir / "summary.json")
    with (output_dir / "episode_budget_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "domain", "budget", "threshold_logit", "threshold", "pixel_risk", "pd", "satisfied"])
        for episode in range(len(arrays.features)):
            for budget_index, budget in enumerate(arrays.budgets):
                writer.writerow([
                    episode,
                    arrays.domains[episode],
                    float(budget),
                    float(eta[episode, budget_index]),
                    float(1.0 / (1.0 + np.exp(-eta[episode, budget_index]))),
                    float(summary.pixel_risk[episode, budget_index]),
                    float(summary.pd[episode, budget_index]),
                    bool(summary.satisfied[episode, budget_index]),
                ])
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

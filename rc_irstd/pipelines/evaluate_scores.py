from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rc_irstd.data.score_records import load_score_record
from rc_irstd.episodes.builder import default_threshold_grid
from rc_irstd.evaluation.component_curves import compute_component_curve
from rc_irstd.evaluation.curves import (
    aggregate_curve_counts,
    compute_image_curves,
    rates_from_counts,
)
from rc_irstd.evaluation.irstd_metrics import evaluate_irstd_at_threshold
from rc_irstd.utils.io import atomic_json_dump, ensure_dir, list_npz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate formal monotone risks and standard component metrics."
    )
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--threshold-grid", default=None)
    parser.add_argument("--component-grid-points", type=int, default=101)
    parser.add_argument("--pixel-budget", type=float, action="append", default=None)
    parser.add_argument("--peak-budget", type=float, action="append", default=None)
    parser.add_argument("--component-budget", type=float, action="append", default=None)
    parser.add_argument("--peak-min-distance", type=int, default=2)
    parser.add_argument("--peak-min-score", type=float, default=0.0)
    parser.add_argument("--peak-tolerance", type=float, default=2.0)
    parser.add_argument("--object-tolerance", type=float, default=2.0)
    parser.add_argument(
        "--max-candidates", type=int, default=0,
        help="Fixed peaks/image cap; 0 means no truncation."
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def _best_feasible(pd: np.ndarray, risk: np.ndarray, budget: float) -> int | None:
    feasible = np.flatnonzero(risk <= budget)
    if len(feasible) == 0:
        return None
    best_pd = np.max(pd[feasible])
    # Earliest threshold among equal-Pd feasible points preserves recall.
    return int(feasible[np.flatnonzero(pd[feasible] == best_pd)[0]])


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_candidates < 0:
        raise ValueError("max-candidates must be non-negative")
    pixel_budgets = args.pixel_budget or [1e-6, 1e-5]
    peak_budgets = args.peak_budget or [1.0, 5.0]
    component_budgets = args.component_budget or [1.0, 5.0]
    thresholds = (
        np.load(args.threshold_grid).astype(np.float32)
        if args.threshold_grid
        else default_threshold_grid()
    )
    records = [load_score_record(path, require_mask=True) for path in list_npz(args.score_dir)]
    probabilities: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    formal_curves = []
    for record in records:
        assert record.mask is not None
        probabilities.append(record.probability)
        masks.append(record.mask)
        formal_curves.append(
            compute_image_curves(
                record.probability,
                record.mask,
                thresholds,
                peak_min_distance=args.peak_min_distance,
                peak_min_score=args.peak_min_score,
                peak_tolerance=args.peak_tolerance,
                max_candidates=(None if args.max_candidates <= 0 else args.max_candidates),
            )
        )
    counts = aggregate_curve_counts(formal_curves)
    rates = rates_from_counts(counts)
    output_dir = ensure_dir(args.output_dir)

    formal_rows: list[dict[str, object]] = []
    for index, threshold in enumerate(thresholds):
        formal_rows.append(
            {
                "threshold": float(threshold),
                "pd": float(rates["pd"][index]),
                "fa_pixel": float(rates["pixel_false_rate"][index]),
                "false_peak_per_mp": float(rates["peak_false_per_mp"][index]),
                "false_pixels": int(counts.pixel_false[index]),
                "false_peaks": int(counts.peak_false[index]),
                "matched_gt": int(counts.matched_gt[index]),
                "total_gt": int(counts.total_gt),
                "total_pixels": int(counts.total_pixels),
            }
        )
    formal_path = output_dir / "formal_curve.csv"
    _write_dict_rows(formal_path, formal_rows)
    # Backward-compatible filename.
    _write_dict_rows(output_dir / "curve.csv", formal_rows)

    component_grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.95, max(args.component_grid_points // 2, 2), endpoint=False),
                np.linspace(0.95, 1.0, max(args.component_grid_points // 2, 2)),
                np.asarray([np.nextafter(np.float32(1.0), np.float32(2.0))]),
            ]
        )
    ).astype(np.float32)
    component_rows_obj = compute_component_curve(
        probabilities, masks, component_grid, args.object_tolerance
    )
    component_rows = [row.to_dict() for row in component_rows_obj]
    component_path = output_dir / "component_curve.csv"
    _write_dict_rows(component_path, component_rows)

    operating_points: dict[str, object] = {}
    for budget in pixel_budgets:
        index = _best_feasible(rates["pd"], rates["pixel_false_rate"], budget)
        operating_points[f"formal_pixel_{budget:g}"] = None if index is None else {
            "index": index,
            "threshold": float(thresholds[index]),
            "pd": float(rates["pd"][index]),
            "risk": float(rates["pixel_false_rate"][index]),
            "rejected": bool(thresholds[index] > 1.0),
            "standard_metrics": evaluate_irstd_at_threshold(
                probabilities, masks, float(thresholds[index]), args.object_tolerance
            ).to_dict(),
        }
    for budget in peak_budgets:
        index = _best_feasible(rates["pd"], rates["peak_false_per_mp"], budget)
        operating_points[f"formal_peak_{budget:g}"] = None if index is None else {
            "index": index,
            "threshold": float(thresholds[index]),
            "pd": float(rates["pd"][index]),
            "risk": float(rates["peak_false_per_mp"][index]),
            "rejected": bool(thresholds[index] > 1.0),
        }
    component_pd = np.asarray([row.pd for row in component_rows_obj])
    component_fa = np.asarray([row.false_components_per_mp for row in component_rows_obj])
    for budget in component_budgets:
        index = _best_feasible(component_pd, component_fa, budget)
        operating_points[f"component_{budget:g}"] = None if index is None else component_rows_obj[index].to_dict()

    fixed_metrics = evaluate_irstd_at_threshold(
        probabilities, masks, 0.5, args.object_tolerance
    )
    summary = {
        "score_dir": str(Path(args.score_dir).resolve()),
        "num_images": len(records),
        "num_formal_thresholds": len(thresholds),
        "num_component_thresholds": len(component_grid),
        "total_pixels": counts.total_pixels,
        "total_gt": counts.total_gt,
        "fixed_0p5_metrics": fixed_metrics.to_dict(),
        "operating_points": operating_points,
        "formal_curve_csv": str(formal_path.resolve()),
        "component_curve_csv": str(component_path.resolve()),
        "metric_note": {
            "formal": "pixel false rate + threshold-independent fixed false local peaks/MP",
            "standard": "thresholded connected components with overlap/centroid-tolerance matching",
            "niou": "mean image IoU over images with non-empty prediction/GT union",
            "hiou": "harmonic mean of foreground and background IoU",
        },
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

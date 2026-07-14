from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from rc_irstd.candidates.peaks import extract_fixed_peaks
from rc_irstd.data.score_records import load_score_record
from rc_irstd.utils.io import atomic_json_dump, ensure_dir, list_npz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a fixed operating point to score maps.")
    parser.add_argument("--score-dir", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--threshold", type=float)
    group.add_argument("--threshold-json")
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--peak-min-distance", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    return parser


def _threshold(args: argparse.Namespace) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    payload = json.loads(Path(args.threshold_json).read_text(encoding="utf-8"))
    if "threshold" not in payload:
        raise KeyError("threshold-json does not contain a threshold field")
    return float(payload["threshold"])


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    threshold = _threshold(args)
    output_dir = ensure_dir(args.output_dir)
    mask_root = ensure_dir(output_dir / "masks")
    candidate_path = output_dir / "candidates.csv"
    records = [load_score_record(path, require_mask=False) for path in list_npz(args.score_dir)]
    records.sort(key=lambda item: (item.sequence_id, item.frame_index, item.image_id))
    selected = records[max(args.skip_first, 0) :]
    candidate_rows: list[dict[str, object]] = []
    for record in selected:
        binary = (record.probability >= threshold).astype(np.uint8) * 255
        mask_path = mask_root / f"{record.image_id}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary).save(mask_path)
        scores, ys, xs = extract_fixed_peaks(
            record.probability,
            min_distance=args.peak_min_distance,
            min_score=max(0.0, threshold),
        )
        for score, y, x in zip(scores, ys, xs, strict=True):
            if score < threshold:
                continue
            candidate_rows.append(
                {
                    "image_id": record.image_id,
                    "sequence_id": record.sequence_id,
                    "frame_index": record.frame_index,
                    "y": int(y),
                    "x": int(x),
                    "score": float(score),
                    "threshold": threshold,
                }
            )
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["image_id", "sequence_id", "frame_index", "y", "x", "score", "threshold"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_rows)
    summary = {
        "threshold": threshold,
        "skip_first": args.skip_first,
        "num_processed_images": len(selected),
        "num_candidates": len(candidate_rows),
        "mask_directory": str(mask_root.resolve()),
        "candidate_csv": str(candidate_path.resolve()),
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

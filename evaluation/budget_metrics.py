"""Budget satisfaction, relative excess and rejection-aware coverage metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .operating_point import validate_budgets


def relative_budget_excess(value: float, budget: float) -> float:
    """Return ``max(value - budget, 0) / budget``."""

    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Observed false-alarm value must be finite and non-negative: {value}")
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError(f"Budget must be finite and positive: {budget}")
    return max(value - budget, 0.0) / budget


def compute_budget_metrics(
    records: Iterable[Mapping[str, object]],
    *,
    pixel_budget: float | None = None,
    component_budget: float | None = None,
    rejected_key: str = "rejected",
) -> dict[str, float | int | None]:
    """Compute joint dual-budget metrics without rewarding blanket rejection.

    ``bsr`` and ``excess`` are measured among non-rejected (covered) records.
    ``unconditional_bsr`` divides safe covered records by all records, while
    ``coverage`` exposes the rejection rate.  With two active budgets, joint
    satisfaction requires both and joint relative excess is the larger of the
    two relative violations.
    """

    validate_budgets(pixel_budget, component_budget)
    materialised = list(records)
    if not materialised:
        raise ValueError("At least one evaluation record is required")

    covered: list[Mapping[str, object]] = []
    for record in materialised:
        rejected = bool(record.get(rejected_key, False))
        if not rejected:
            _validate_observation(record, pixel_budget, component_budget)
            covered.append(record)

    joint_satisfied = 0
    joint_excesses: list[float] = []
    pixel_satisfied = 0
    pixel_excesses: list[float] = []
    component_satisfied = 0
    component_excesses: list[float] = []

    for record in covered:
        active_satisfied: list[bool] = []
        active_excesses: list[float] = []
        if pixel_budget is not None:
            value = float(record["fa_pixel"])
            satisfied = value <= pixel_budget
            excess = relative_budget_excess(value, pixel_budget)
            pixel_satisfied += int(satisfied)
            pixel_excesses.append(excess)
            active_satisfied.append(satisfied)
            active_excesses.append(excess)
        if component_budget is not None:
            value = float(record["fa_component_mp"])
            satisfied = value <= component_budget
            excess = relative_budget_excess(value, component_budget)
            component_satisfied += int(satisfied)
            component_excesses.append(excess)
            active_satisfied.append(satisfied)
            active_excesses.append(excess)

        joint_satisfied += int(all(active_satisfied))
        joint_excesses.append(max(active_excesses))

    num_total = len(materialised)
    num_covered = len(covered)
    result: dict[str, float | int | None] = {
        "num_records": num_total,
        "num_covered": num_covered,
        "num_rejected": num_total - num_covered,
        "coverage": num_covered / num_total,
        "pixel_budget": pixel_budget,
        "component_budget": component_budget,
        "bsr": _safe_ratio(joint_satisfied, num_covered),
        "joint_bsr": _safe_ratio(joint_satisfied, num_covered),
        "unconditional_bsr": joint_satisfied / num_total,
        "excess": _safe_mean(joint_excesses),
        "joint_excess": _safe_mean(joint_excesses),
        "pixel_bsr": (
            _safe_ratio(pixel_satisfied, num_covered)
            if pixel_budget is not None
            else None
        ),
        "pixel_excess": (
            _safe_mean(pixel_excesses) if pixel_budget is not None else None
        ),
        "component_bsr": (
            _safe_ratio(component_satisfied, num_covered)
            if component_budget is not None
            else None
        ),
        "component_excess": (
            _safe_mean(component_excesses)
            if component_budget is not None
            else None
        ),
    }
    return result


def _validate_observation(
    record: Mapping[str, object],
    pixel_budget: float | None,
    component_budget: float | None,
) -> None:
    required = []
    if pixel_budget is not None:
        required.append("fa_pixel")
    if component_budget is not None:
        required.append("fa_component_mp")
    for field in required:
        if field not in record:
            raise KeyError(f"Evaluation record is missing {field!r}")
        value = float(record[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and non-negative, got {value}")


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _safe_mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _read_json_records(path: str | Path) -> list[Mapping[str, object]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input JSON is empty: {source}")
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list of records")
        return payload
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON list or JSONL records")
    parser.add_argument("--pixel-budget", type=float)
    parser.add_argument("--component-budget", type=float)
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metrics = compute_budget_metrics(
        _read_json_records(args.input),
        pixel_budget=args.pixel_budget,
        component_budget=args.component_budget,
    )
    rendered = json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

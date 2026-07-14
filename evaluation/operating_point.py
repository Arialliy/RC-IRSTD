"""Select maximum-Pd operating points under pixel and/or component budgets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .threshold_sweep import read_curve_csv


def validate_budgets(
    pixel_budget: float | None,
    component_budget: float | None,
) -> tuple[float | None, float | None]:
    """Validate that at least one finite, positive budget is active."""

    if pixel_budget is None and component_budget is None:
        raise ValueError("At least one of pixel_budget or component_budget is required")
    for name, value in (
        ("pixel_budget", pixel_budget),
        ("component_budget", component_budget),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive, got {value}")
    return pixel_budget, component_budget


def satisfies_budgets(
    row: Mapping[str, object],
    *,
    pixel_budget: float | None = None,
    component_budget: float | None = None,
) -> bool:
    """Return whether a curve/result row satisfies every active budget."""

    validate_budgets(pixel_budget, component_budget)
    if pixel_budget is not None and float(row["fa_pixel"]) > pixel_budget:
        return False
    if (
        component_budget is not None
        and float(row["fa_component_mp"]) > component_budget
    ):
        return False
    return True


def select_operating_point(
    rows: Iterable[Mapping[str, object]],
    *,
    pixel_budget: float | None = None,
    component_budget: float | None = None,
) -> dict[str, object] | None:
    """Select highest Pd, breaking ties toward the lowest safe threshold.

    When both budgets are provided they are conjunctive: a row must satisfy
    both the pixel and component constraints.
    """

    validate_budgets(pixel_budget, component_budget)
    feasible: list[Mapping[str, object]] = []
    for row in rows:
        _validate_curve_row(row)
        if satisfies_budgets(
            row,
            pixel_budget=pixel_budget,
            component_budget=component_budget,
        ):
            feasible.append(row)
    if not feasible:
        return None

    best = min(
        feasible,
        key=lambda row: (-float(row["pd"]), float(row["threshold"])),
    )
    result = dict(best)
    result["pixel_budget"] = pixel_budget
    result["component_budget"] = component_budget
    return result


def select_budget_grid(
    rows: Iterable[Mapping[str, object]],
    *,
    pixel_budgets: Sequence[float | None],
    component_budgets: Sequence[float | None],
) -> list[dict[str, object]]:
    """Select operating points for a Cartesian grid of dual budgets."""

    materialised_rows = list(rows)
    selected: list[dict[str, object]] = []
    for pixel_budget in pixel_budgets:
        for component_budget in component_budgets:
            if pixel_budget is None and component_budget is None:
                continue
            point = select_operating_point(
                materialised_rows,
                pixel_budget=pixel_budget,
                component_budget=component_budget,
            )
            selected.append(
                {
                    "pixel_budget": pixel_budget,
                    "component_budget": component_budget,
                    "feasible": point is not None,
                    "operating_point": point,
                }
            )
    return selected


def _validate_curve_row(row: Mapping[str, object]) -> None:
    for field in ("threshold", "pd", "fa_pixel", "fa_component_mp"):
        if field not in row:
            raise KeyError(f"Operating-point row is missing {field!r}")
        value = float(row[field])
        if not math.isfinite(value):
            raise ValueError(f"Operating-point field {field!r} is not finite")
    threshold = float(row["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold is outside [0, 1]: {threshold}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", required=True)
    parser.add_argument("--pixel-budget", type=float)
    parser.add_argument("--component-budget", type=float)
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    point = select_operating_point(
        read_curve_csv(args.curve),
        pixel_budget=args.pixel_budget,
        component_budget=args.component_budget,
    )
    rendered = json.dumps(point, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if point is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())

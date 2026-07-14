"""Select maximum-Pd operating points under pixel and/or component budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .threshold_sweep import (
    ORACLE_DEPLOYMENT_STATUS,
    ORACLE_ONLY,
    ORACLE_SELECTION_SCOPE,
    read_curve_csv,
)


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
    curve_sha256: str | None = None,
    curve_manifest_sha256: str | None = None,
) -> dict[str, object] | None:
    """Select a label-oracle point, breaking Pd ties toward lower thresholds.

    When both budgets are provided they are conjunctive: a row must satisfy
    both the pixel and component constraints.  Because ``pd`` and both false
    alarm risks in a threshold curve require query ground-truth labels, the
    returned point is always marked non-deployable.  File-backed callers
    should pass the curve and manifest SHA-256 values to bind provenance.
    """

    validate_budgets(pixel_budget, component_budget)
    provenance = _oracle_selection_contract(
        curve_sha256=curve_sha256,
        curve_manifest_sha256=curve_manifest_sha256,
    )
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
    # Apply after copying the row so a caller cannot smuggle deployable=True
    # or a different selection scope through extra CSV columns.
    result.update(provenance)
    return result


def select_budget_grid(
    rows: Iterable[Mapping[str, object]],
    *,
    pixel_budgets: Sequence[float | None],
    component_budgets: Sequence[float | None],
    curve_sha256: str | None = None,
    curve_manifest_sha256: str | None = None,
) -> list[dict[str, object]]:
    """Select explicitly non-deployable oracle points over a budget grid."""

    materialised_rows = list(rows)
    provenance = _oracle_selection_contract(
        curve_sha256=curve_sha256,
        curve_manifest_sha256=curve_manifest_sha256,
    )
    selected: list[dict[str, object]] = []
    for pixel_budget in pixel_budgets:
        for component_budget in component_budgets:
            if pixel_budget is None and component_budget is None:
                continue
            point = select_operating_point(
                materialised_rows,
                pixel_budget=pixel_budget,
                component_budget=component_budget,
                curve_sha256=curve_sha256,
                curve_manifest_sha256=curve_manifest_sha256,
            )
            selected.append(
                {
                    "pixel_budget": pixel_budget,
                    "component_budget": component_budget,
                    "feasible": point is not None,
                    "operating_point": point,
                    **provenance,
                }
            )
    return selected


def _oracle_selection_contract(
    *,
    curve_sha256: str | None,
    curve_manifest_sha256: str | None,
) -> dict[str, object]:
    """Return the immutable machine-readable label-oracle contract."""

    curve_sha256 = _normalise_optional_sha256(curve_sha256, "curve_sha256")
    curve_manifest_sha256 = _normalise_optional_sha256(
        curve_manifest_sha256,
        "curve_manifest_sha256",
    )
    if curve_manifest_sha256 is not None and curve_sha256 is None:
        raise ValueError("curve_manifest_sha256 requires curve_sha256")
    return {
        "oracle_only": ORACLE_ONLY,
        "selection_scope": ORACLE_SELECTION_SCOPE,
        "deployable": False,
        "deployment_status": ORACLE_DEPLOYMENT_STATUS,
        "selection_uses_ground_truth_labels": True,
        "selection_provenance_bound": curve_sha256 is not None,
        "curve_sha256": curve_sha256,
        "curve_manifest_sha256": curve_manifest_sha256,
    }


def _normalise_optional_sha256(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    normalised = str(value).strip().lower()
    if len(normalised) != 64 or any(
        character not in "0123456789abcdef" for character in normalised
    ):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return normalised


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
    parser.add_argument(
        "--curve-manifest",
        help=(
            "Curve manifest to verify and bind. By default, use "
            "<curve>.manifest.json when it exists."
        ),
    )
    parser.add_argument("--pixel-budget", type=float)
    parser.add_argument("--component-budget", type=float)
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    curve_path = Path(args.curve).expanduser().resolve()
    curve_sha256, curve_manifest_sha256 = _verify_curve_provenance(
        curve_path,
        args.curve_manifest,
    )
    point = select_operating_point(
        read_curve_csv(curve_path),
        pixel_budget=args.pixel_budget,
        component_budget=args.component_budget,
        curve_sha256=curve_sha256,
        curve_manifest_sha256=curve_manifest_sha256,
    )
    rendered = json.dumps(point, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if point is not None else 2


def _verify_curve_provenance(
    curve_path: Path,
    explicit_manifest: str | Path | None,
) -> tuple[str, str | None]:
    """Hash a curve and fail closed when an associated manifest disagrees."""

    if not curve_path.is_file():
        raise FileNotFoundError(f"Curve CSV does not exist: {curve_path}")
    curve_sha256 = _sha256(curve_path)
    if explicit_manifest is None:
        candidate = curve_path.with_suffix(curve_path.suffix + ".manifest.json")
        manifest_path = candidate if candidate.is_file() else None
    else:
        manifest_path = Path(explicit_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Curve manifest does not exist: {manifest_path}")
    if manifest_path is None:
        return curve_sha256, None

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Curve manifest must be a JSON object")
    manifest_curve_sha256 = _normalise_optional_sha256(
        payload.get("curve_sha256"),
        "curve manifest curve_sha256",
    )
    if manifest_curve_sha256 is None:
        raise KeyError("Curve manifest is missing 'curve_sha256'")
    if manifest_curve_sha256 != curve_sha256:
        raise ValueError("Curve CSV SHA-256 does not match curve manifest")

    curve_file = payload.get("curve_file")
    if not isinstance(curve_file, str) or not curve_file.strip():
        raise KeyError("Curve manifest is missing non-empty 'curve_file'")
    declared_curve = (manifest_path.parent / curve_file).resolve()
    if declared_curve != curve_path:
        raise ValueError("Curve manifest curve_file does not resolve to --curve")
    return curve_sha256, _sha256(manifest_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

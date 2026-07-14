"""Replay an online-adapter decision on its cryptographically bound query set.

The adapter consumes only an unlabeled context and emits one threshold (or a
rejection) for a disjoint query suffix.  This module is the label-using,
offline half of that protocol: it verifies the adapter/manifest binding before
loading any query labels, applies exactly ``probability > threshold``, and
reports native-resolution object and false-alarm counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .budget_metrics import relative_budget_excess
from .component_matching import aggregate_match_results, match_components
from .threshold_sweep import THRESHOLD_SEMANTICS, load_score_map


ADAPTER_EVALUATION_SCHEMA_VERSION = "rc-irstd.adapter-evaluation.v1"
ADAPTER_SUMMARY_SCHEMA_VERSION = "rc-irstd.adapter-evaluation-summary.v1"
_RAW_METRIC_FIELDS = (
    "pd",
    "fa_pixel",
    "fa_component_mp",
    "tp_objects",
    "gt_objects",
    "pred_components",
    "fp_components",
    "fp_pixels",
    "total_pixels",
    "num_images",
)


def evaluate_adapter_output(
    adapter_output: str | Path | Mapping[str, Any],
    score_manifest: str | Path,
    *,
    matching_rule: str = "overlap",
    centroid_distance: float = 3.0,
) -> dict[str, Any]:
    """Verify and replay one online adapter result on its declared query IDs.

    Rejected outputs deliberately contain no Pd/FA values or raw label-derived
    counts.  They remain useful records for coverage-aware aggregation.
    """

    if matching_rule not in {"overlap", "centroid"}:
        raise ValueError("matching_rule must be 'overlap' or 'centroid'")
    if not math.isfinite(float(centroid_distance)) or centroid_distance <= 0.0:
        raise ValueError("centroid_distance must be finite and positive")

    adapter, adapter_name = _load_adapter_output(adapter_output)
    manifest_path = Path(score_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Score manifest does not exist: {manifest_path}")
    if (manifest_path.parent / ".export_incomplete").exists():
        raise RuntimeError(
            f"Score export under {manifest_path.parent} is incomplete; replay is unsafe"
        )
    manifest = _read_json_mapping(manifest_path, "score manifest")
    manifest_sha256 = _sha256(manifest_path)

    target_domain, context_ids, query_ids = _verify_binding(
        adapter,
        manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    budgets = _normalise_budgets(adapter.get("budgets"))
    rejected = _require_bool(adapter, "reject")
    threshold = _finite_probability(adapter.get("threshold"), "threshold")
    query_paths = _resolve_bound_query_paths(
        manifest,
        manifest_path=manifest_path,
        context_ids=context_ids,
        query_ids=query_ids,
    )

    result: dict[str, Any] = {
        "schema_version": ADAPTER_EVALUATION_SCHEMA_VERSION,
        "adapter_output_file": adapter_name,
        "outer_fold_id": _nonempty_string(adapter["outer_fold_id"], "outer_fold_id"),
        "target_domain": target_domain,
        "score_manifest_file": manifest_path.name,
        "score_manifest_sha256": manifest_sha256,
        "query_image_ids": list(query_ids),
        "num_query_images": len(query_ids),
        "budgets": budgets,
        "threshold": threshold,
        "threshold_semantics": THRESHOLD_SEMANTICS,
        "matching_rule": matching_rule,
        "centroid_distance": float(centroid_distance),
        "rejected": rejected,
    }
    if rejected:
        return result

    matches = []
    for expected_id, path in zip(query_ids, query_paths):
        record = load_score_map(path)
        if record.image_id != expected_id:
            raise ValueError(
                "Score-map image_id disagrees with its bound manifest item: "
                f"expected {expected_id!r}, found {record.image_id!r} in {path.name!r}"
            )
        matches.append(
            match_components(
                record.probability > threshold,
                record.mask,
                rule=matching_rule,
                centroid_distance=centroid_distance,
            )
        )
    result.update(aggregate_match_results(matches))
    return result


def summarise_adapter_evaluations(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate coverage, per-record budget safety, excess, and covered Pd.

    Budgets may differ between adapter records because they are model inputs.
    BSR and excess therefore use each record's own active-budget contract.
    Covered Pd is a micro-average reconstructed from raw TP/GT counts, never
    an average of rounded per-record Pd values.
    """

    materialised = list(records)
    if not materialised:
        raise ValueError("At least one adapter evaluation is required")

    covered: list[Mapping[str, Any]] = []
    joint_satisfied = 0
    joint_excesses: list[float] = []
    pixel_satisfied = 0
    pixel_excesses: list[float] = []
    component_satisfied = 0
    component_excesses: list[float] = []
    covered_tp = 0
    covered_gt = 0
    covered_images = 0

    for record in materialised:
        if "rejected" not in record or not isinstance(record["rejected"], bool):
            raise TypeError("Each evaluation record requires a boolean 'rejected' field")
        budgets = _normalise_budgets(record.get("budgets"))
        if record["rejected"]:
            leaked = [field for field in _RAW_METRIC_FIELDS if field in record]
            if leaked:
                raise ValueError(
                    "Rejected records must not contain label-derived metrics: "
                    f"{leaked}"
                )
            continue

        _validate_covered_record(record)
        covered.append(record)
        values = (float(record["fa_pixel"]), float(record["fa_component_mp"]))
        active_satisfied: list[bool] = []
        active_excesses: list[float] = []
        for index, (value, budget, active) in enumerate(
            zip(values, budgets["values"], budgets["active"])
        ):
            if not active:
                continue
            satisfied = value <= budget
            excess = relative_budget_excess(value, budget)
            active_satisfied.append(satisfied)
            active_excesses.append(excess)
            if index == 0:
                pixel_satisfied += int(satisfied)
                pixel_excesses.append(excess)
            else:
                component_satisfied += int(satisfied)
                component_excesses.append(excess)
        joint_satisfied += int(all(active_satisfied))
        joint_excesses.append(max(active_excesses))
        covered_tp += int(record["tp_objects"])
        covered_gt += int(record["gt_objects"])
        covered_images += int(record["num_images"])

    num_total = len(materialised)
    num_covered = len(covered)
    return {
        "schema_version": ADAPTER_SUMMARY_SCHEMA_VERSION,
        "num_records": num_total,
        "num_covered": num_covered,
        "num_rejected": num_total - num_covered,
        "coverage": num_covered / num_total,
        "bsr": _safe_ratio(joint_satisfied, num_covered),
        "joint_bsr": _safe_ratio(joint_satisfied, num_covered),
        "unconditional_bsr": joint_satisfied / num_total,
        "excess": _safe_mean(joint_excesses),
        "joint_excess": _safe_mean(joint_excesses),
        "pixel_bsr": _safe_ratio(pixel_satisfied, len(pixel_excesses)),
        "pixel_excess": _safe_mean(pixel_excesses),
        "component_bsr": _safe_ratio(
            component_satisfied, len(component_excesses)
        ),
        "component_excess": _safe_mean(component_excesses),
        "covered_pd": (covered_tp / covered_gt) if covered_gt else None,
        "covered_tp_objects": covered_tp,
        "covered_gt_objects": covered_gt,
        "covered_num_images": covered_images,
        "budget_scope": "per_record_active_budgets",
    }


def _verify_binding(
    adapter: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    required = (
        "outer_fold_id",
        "outer_target",
        "target_domain",
        "detector_source_domains",
        "detector_checkpoint_sha",
        "score_manifest_sha256",
        "score_manifest_target_dataset",
        "score_manifest_detector_checkpoint_sha",
        "context_image_ids",
        "query_image_ids",
    )
    missing = [field for field in required if field not in adapter]
    if missing:
        raise KeyError(f"Adapter output is missing binding fields: {missing}")

    expected_manifest_sha = _sha256_value(
        adapter["score_manifest_sha256"], "score_manifest_sha256"
    )
    if expected_manifest_sha != manifest_sha256:
        raise ValueError(
            "Score manifest SHA-256 mismatch: adapter output is not bound to "
            f"{manifest_path.name!r}"
        )

    target_domain = _nonempty_string(adapter["target_domain"], "target_domain")
    target_values = {
        "target_domain": target_domain,
        "outer_target": _nonempty_string(adapter["outer_target"], "outer_target"),
        "score_manifest_target_dataset": _nonempty_string(
            adapter["score_manifest_target_dataset"],
            "score_manifest_target_dataset",
        ),
        "manifest.target_dataset": _nonempty_string(
            manifest.get("target_dataset"), "manifest.target_dataset"
        ),
    }
    if len(set(target_values.values())) != 1:
        raise ValueError(f"Target-domain binding mismatch: {target_values}")

    outer_fold_id = _nonempty_string(adapter["outer_fold_id"], "outer_fold_id")
    if _nonempty_string(manifest.get("outer_fold_id"), "manifest.outer_fold_id") != outer_fold_id:
        raise ValueError("Outer-fold binding mismatch")
    if _nonempty_string(manifest.get("outer_target"), "manifest.outer_target") != target_domain:
        raise ValueError("Manifest outer_target binding mismatch")
    raw_adapter_sources = adapter["detector_source_domains"]
    raw_manifest_sources = manifest.get("detector_source_domains")
    if not isinstance(raw_adapter_sources, (list, tuple)) or not isinstance(
        raw_manifest_sources, (list, tuple)
    ):
        raise TypeError("detector_source_domains must be ordered lists")
    adapter_sources = tuple(
        _nonempty_string(value, "detector_source_domains")
        for value in raw_adapter_sources
    )
    manifest_sources = tuple(
        _nonempty_string(value, "manifest.detector_source_domains")
        for value in raw_manifest_sources
    )
    if (
        not adapter_sources
        or len(set(adapter_sources)) != len(adapter_sources)
        or adapter_sources != manifest_sources
    ):
        raise ValueError("Detector-source-domain binding mismatch")
    if target_domain in manifest_sources:
        raise ValueError("Target domain occurs in detector source domains")
    if manifest.get("protocol_scope") != "multi_source_protocol_candidate":
        raise ValueError("Adapter replay requires a multi-source detector artifact")
    if manifest.get("target_exclusion_verified") is not True:
        raise ValueError("Manifest does not verify target-domain exclusion")

    adapter_detector_sha = _sha256_value(
        adapter["detector_checkpoint_sha"], "detector_checkpoint_sha"
    )
    score_detector_sha = _sha256_value(
        adapter["score_manifest_detector_checkpoint_sha"],
        "score_manifest_detector_checkpoint_sha",
    )
    manifest_detector_sha = _sha256_value(
        manifest.get("weight_sha256"), "manifest.weight_sha256"
    )
    if len({adapter_detector_sha, score_detector_sha, manifest_detector_sha}) != 1:
        raise ValueError("Detector-checkpoint SHA-256 binding mismatch")

    if manifest.get("threshold_semantics") not in {None, THRESHOLD_SEMANTICS}:
        raise ValueError(
            "Score manifest threshold semantics disagree with replay semantics"
        )
    if manifest.get("score_type") not in {None, "sigmoid_probability"}:
        raise ValueError("Score manifest must contain sigmoid probabilities")

    context_ids = _image_ids(adapter["context_image_ids"], "context_image_ids")
    query_ids = _image_ids(adapter["query_image_ids"], "query_image_ids")
    overlap = set(context_ids).intersection(query_ids)
    if overlap:
        raise ValueError(f"Context/query image IDs overlap: {sorted(overlap)}")
    return target_domain, context_ids, query_ids


def _resolve_bound_query_paths(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    context_ids: Sequence[str],
    query_ids: Sequence[str],
) -> tuple[Path, ...]:
    raw_items = manifest.get("items", manifest.get("records"))
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Score manifest requires a non-empty items/records list")
    if "num_images" in manifest and int(manifest["num_images"]) != len(raw_items):
        raise ValueError("Score manifest num_images disagrees with its item count")

    ordered_ids: list[str] = []
    by_id: dict[str, Path] = {}
    for item in raw_items:
        if not isinstance(item, Mapping) or "image_id" not in item:
            raise ValueError("Every score manifest item must contain image_id")
        image_id = _nonempty_string(item["image_id"], "manifest item image_id")
        if image_id in by_id:
            raise ValueError(f"Duplicate image_id in score manifest: {image_id!r}")
        file_value = item.get("file", item.get("prob_path", item.get("score_path")))
        if file_value is None:
            raise KeyError(f"Score manifest item {image_id!r} has no score-map file")
        path = Path(str(file_value)).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        by_id[image_id] = path.resolve()
        ordered_ids.append(image_id)

    bound_prefix = list(context_ids) + list(query_ids)
    if ordered_ids[: len(bound_prefix)] != bound_prefix:
        raise ValueError(
            "Adapter context/query IDs do not exactly match the score-manifest prefix"
        )
    missing = [image_id for image_id in query_ids if image_id not in by_id]
    if missing:
        raise KeyError(f"Query image IDs are absent from score manifest: {missing}")
    paths = tuple(by_id[image_id] for image_id in query_ids)
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Query score maps are missing: {absent}")
    return paths


def _normalise_budgets(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("budgets must be a mapping")
    names = tuple(value.get("names", ("pixel", "component")))
    if names != ("pixel", "component"):
        raise ValueError("budget names must be ['pixel', 'component']")
    raw_values = value.get("values")
    raw_active = value.get("active")
    if not isinstance(raw_values, (list, tuple)) or len(raw_values) != 2:
        raise ValueError("budgets.values must contain pixel and component values")
    if not isinstance(raw_active, (list, tuple)) or len(raw_active) != 2:
        raise ValueError("budgets.active must contain two booleans")
    if not all(isinstance(item, bool) for item in raw_active):
        raise TypeError("budgets.active values must be booleans")
    values = tuple(float(item) for item in raw_values)
    if not all(math.isfinite(item) and item >= 0.0 for item in values):
        raise ValueError("budget values must be finite and non-negative")
    active = tuple(bool(item) for item in raw_active)
    if not any(active):
        raise ValueError("at least one budget must be active")
    if any(enabled and budget <= 0.0 for budget, enabled in zip(values, active)):
        raise ValueError("active budgets must be positive")
    return {"names": list(names), "values": list(values), "active": list(active)}


def _validate_covered_record(record: Mapping[str, Any]) -> None:
    missing = [field for field in _RAW_METRIC_FIELDS if field not in record]
    if missing:
        raise KeyError(f"Covered evaluation is missing metrics: {missing}")
    for field in ("pd", "fa_pixel", "fa_component_mp"):
        value = float(record[field])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
    counts: dict[str, int] = {}
    for field in _RAW_METRIC_FIELDS[3:]:
        value = record[field]
        integer = int(value)
        if isinstance(value, bool) or float(value) != integer or integer < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        counts[field] = integer
    if counts["tp_objects"] > counts["gt_objects"]:
        raise ValueError("tp_objects cannot exceed gt_objects")
    if counts["tp_objects"] > counts["pred_components"]:
        raise ValueError("tp_objects cannot exceed pred_components")
    if counts["fp_components"] != (
        counts["pred_components"] - counts["tp_objects"]
    ):
        raise ValueError("fp_components disagrees with pred_components - tp_objects")
    if counts["total_pixels"] <= 0 or counts["num_images"] <= 0:
        raise ValueError("covered records require positive total_pixels and num_images")
    if counts["fp_pixels"] > counts["total_pixels"]:
        raise ValueError("fp_pixels cannot exceed total_pixels")
    expected_pd = (
        counts["tp_objects"] / counts["gt_objects"]
        if counts["gt_objects"]
        else 0.0
    )
    if not math.isclose(float(record["pd"]), expected_pd, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("pd disagrees with raw tp_objects/gt_objects counts")
    expected_pixel_fa = counts["fp_pixels"] / counts["total_pixels"]
    if not math.isclose(
        float(record["fa_pixel"]), expected_pixel_fa, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("fa_pixel disagrees with raw fp_pixels/total_pixels counts")
    expected_component_fa = counts["fp_components"] / (
        counts["total_pixels"] / 1_000_000.0
    )
    if not math.isclose(
        float(record["fa_component_mp"]),
        expected_component_fa,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "fa_component_mp disagrees with raw fp_components/total_pixels counts"
        )


def _load_adapter_output(
    value: str | Path | Mapping[str, Any],
) -> tuple[Mapping[str, Any], str | None]:
    if isinstance(value, Mapping):
        return value, None
    path = Path(value).expanduser().resolve()
    return _read_json_mapping(path, "adapter output"), path.name


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label.title()} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label.title()} JSON must be an object")
    return payload


def _image_ids(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(_nonempty_string(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate values")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_bool(payload: Mapping[str, Any], name: str) -> bool:
    if name not in payload or not isinstance(payload[name], bool):
        raise TypeError(f"{name} must be a boolean")
    return bool(payload[name])


def _finite_probability(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite probability") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _sha256_value(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 string")
    result = value.lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _safe_mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-output",
        action="append",
        required=True,
        help="Online-adapter JSON; repeat together with --score-manifest to aggregate",
    )
    parser.add_argument(
        "--score-manifest",
        action="append",
        required=True,
        help="Cryptographically bound score manifest; repeat in matching order",
    )
    parser.add_argument("--matching-rule", choices=("overlap", "centroid"), default="overlap")
    parser.add_argument("--centroid-distance", type=float, default=3.0)
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if len(args.adapter_output) != len(args.score_manifest):
        raise ValueError(
            "--adapter-output and --score-manifest must be repeated the same number of times"
        )
    evaluations = [
        evaluate_adapter_output(
            adapter_path,
            manifest_path,
            matching_rule=args.matching_rule,
            centroid_distance=args.centroid_distance,
        )
        for adapter_path, manifest_path in zip(args.adapter_output, args.score_manifest)
    ]
    payload: Mapping[str, Any]
    if len(evaluations) == 1:
        payload = evaluations[0]
    else:
        payload = {
            "schema_version": ADAPTER_SUMMARY_SCHEMA_VERSION,
            "evaluations": evaluations,
            "summary": summarise_adapter_evaluations(evaluations),
        }
    rendered = json.dumps(
        payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False
    )
    if args.output:
        _write_json_atomic(Path(args.output), payload)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

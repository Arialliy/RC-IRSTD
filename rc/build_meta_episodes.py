"""Build leakage-safe RC meta episodes from exported context scores and curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .domain_statistics import (
    DomainStatistics,
    extract_unlabeled_statistics,
    load_probability_and_grayscale,
    load_source_reference,
)
from .oracle_threshold import OracleResult, select_oracle_operating_point
from .schema import (
    BudgetSpec,
    EpisodeProvenance,
    FoldContract,
    RCEpisode,
    SourceReference,
    StatisticsConfig,
    VALID_THRESHOLD_TRANSFORMS,
)


def build_meta_episode(
    *,
    episode_id: str,
    pseudo_target: str,
    context_image_ids: Sequence[str],
    query_image_ids: Sequence[str],
    statistics: DomainStatistics | Sequence[float],
    statistics_config: StatisticsConfig,
    source_reference: SourceReference,
    fold: FoldContract,
    provenance: EpisodeProvenance,
    curve: Any,
    budgets: BudgetSpec,
    p_min: float,
    feature_names: Sequence[str] | None = None,
    threshold_transform: str = "identity",
    metadata: Mapping[str, Any] | None = None,
) -> RCEpisode:
    """Create one episode; query labels influence only ``curve`` and oracle."""

    if isinstance(statistics, DomainStatistics):
        values = statistics.vector
        names = statistics.feature_names
        statistics_metadata = dict(statistics.metadata or {})
        if statistics.statistics_config != statistics_config:
            raise ValueError("DomainStatistics config differs from episode statistics_config")
    else:
        values = np.asarray(statistics, dtype=np.float32).reshape(-1)
        if feature_names is None:
            raise ValueError("feature_names are required for a raw statistics vector")
        names = tuple(feature_names)
        statistics_metadata = {}
    if feature_names is not None and tuple(feature_names) != tuple(names):
        raise ValueError("provided feature_names do not match DomainStatistics.feature_names")
    oracle = select_oracle_operating_point(curve, budgets, p_min=p_min)
    episode_metadata = dict(metadata or {})
    episode_metadata.update(
        {
            "statistics": statistics_metadata,
            "oracle_selected_index": oracle.selected_index,
            "oracle_feasible_count": oracle.feasible_count,
        }
    )
    return RCEpisode.create(
        episode_id=episode_id,
        pseudo_target=pseudo_target,
        context_image_ids=context_image_ids,
        query_image_ids=query_image_ids,
        statistics=values,
        feature_names=names,
        statistics_config=statistics_config,
        source_reference=source_reference,
        fold=fold,
        provenance=provenance,
        budgets=budgets,
        oracle_threshold=oracle.threshold,
        oracle_pd=oracle.pd,
        oracle_pixel_risk=oracle.pixel_risk,
        oracle_component_risk=oracle.component_risk,
        p_min=p_min,
        threshold_transform=threshold_transform,
        metadata=episode_metadata,
    )


def causal_windows(
    ordered_image_ids: Sequence[str],
    *,
    context_size: int,
    query_size: int,
    stride: int | None = None,
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Return context windows followed immediately by disjoint query windows."""

    if context_size <= 0 or query_size <= 0:
        raise ValueError("context_size and query_size must be positive")
    if stride is None:
        stride = query_size
    if stride <= 0:
        raise ValueError("stride must be positive")
    image_ids = tuple(str(value) for value in ordered_image_ids)
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("ordered_image_ids must be unique")
    windows = []
    final_start = len(image_ids) - context_size - query_size
    for start in range(0, final_start + 1, stride):
        boundary = start + context_size
        context = image_ids[start:boundary]
        query = image_ids[boundary : boundary + query_size]
        windows.append((context, query))
    return windows


def _load_curve_csv(path: Path) -> dict[str, np.ndarray]:
    columns: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"curve CSV has no header: {path}")
        for row in reader:
            for key, value in row.items():
                if key is not None and value not in (None, ""):
                    try:
                        columns.setdefault(key, []).append(float(value))
                    except ValueError:
                        # Non-numeric audit columns are irrelevant to oracle selection.
                        pass
    return {key: np.asarray(values, dtype=np.float64) for key, values in columns.items()}


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_image_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    if "image_ids" in payload:
        return tuple(str(value) for value in payload["image_ids"])
    items = payload.get("items", payload.get("records"))
    if isinstance(items, list) and all(isinstance(item, Mapping) and "image_id" in item for item in items):
        return tuple(str(item["image_id"]) for item in items)
    raise ValueError("curve manifest must provide image_ids or items with image_id")


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return payload


def _checkpoint_sha(payload: Mapping[str, Any]) -> str:
    for key in (
        "weight_sha256",
        "detector_checkpoint_sha",
        "detector_weight_sha256",
        "checkpoint_sha256",
    ):
        if key in payload:
            return str(payload[key]).lower()
    raise KeyError("manifest is missing detector checkpoint SHA-256")


def _manifest_contract_value(payload: Mapping[str, Any], key: str) -> Any:
    """Read a protocol field while rejecting conflicting nested provenance."""

    nested = payload.get("detector_provenance")
    nested_present = isinstance(nested, Mapping) and key in nested
    top_present = key in payload
    if top_present and nested_present and payload[key] != nested[key]:
        raise ValueError(f"score manifest has conflicting top-level/nested {key}")
    if top_present:
        return payload[key]
    return nested[key] if nested_present else None


def _verify_score_manifest(
    path: Path,
    *,
    expected_target: str,
    expected_checkpoint_sha: str,
    expected_outer_fold_id: str,
    expected_outer_target: str,
    expected_detector_sources: Sequence[str],
    expected_held_out_domains: Sequence[str],
    expected_protocol_scope: str,
    expected_image_ids: Sequence[str] | None = None,
    exact_image_ids: bool,
) -> tuple[Mapping[str, Any], str]:
    payload = _read_json(path)
    target = str(payload.get("target_dataset", ""))
    if target != expected_target:
        raise ValueError(
            f"score manifest target_dataset mismatch: {target!r} != {expected_target!r}"
        )
    if _checkpoint_sha(payload) != expected_checkpoint_sha:
        raise ValueError("score manifest detector checkpoint SHA does not match episode spec")
    if str(_manifest_contract_value(payload, "outer_fold_id") or "") != expected_outer_fold_id:
        raise ValueError("score manifest outer_fold_id does not match episode spec")
    if str(_manifest_contract_value(payload, "outer_target") or "") != expected_outer_target:
        raise ValueError("score manifest outer_target does not match episode spec")
    manifest_sources = tuple(
        str(value)
        for value in (_manifest_contract_value(payload, "detector_source_domains") or ())
    )
    if manifest_sources != tuple(expected_detector_sources):
        raise ValueError("score manifest detector_source_domains do not match episode spec")
    manifest_held_out = tuple(
        str(value)
        for value in (_manifest_contract_value(payload, "held_out_domains") or ())
    )
    if manifest_held_out != tuple(expected_held_out_domains):
        raise ValueError("score manifest held_out_domains do not match episode spec")
    manifest_protocol_scope = _manifest_contract_value(payload, "protocol_scope")
    if manifest_protocol_scope != expected_protocol_scope:
        raise ValueError("score manifest protocol_scope does not match episode spec")
    if manifest_protocol_scope != "multi_source_protocol_candidate":
        raise ValueError(
            "verified main-protocol episodes require a multi-source detector; "
            "single-source inner folds are smoke-only"
        )
    if _manifest_contract_value(payload, "target_exclusion_verified") is not True:
        raise ValueError("score manifest does not verify target exclusion from detector sources")
    manifest_ids = _manifest_image_ids(payload)
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("score manifest image IDs must be unique")
    if int(payload.get("num_images", len(manifest_ids))) != len(manifest_ids):
        raise ValueError("score manifest num_images disagrees with its image IDs")
    if expected_image_ids is not None:
        expected = tuple(str(value) for value in expected_image_ids)
        if exact_image_ids:
            if manifest_ids != expected:
                raise ValueError("query score manifest image IDs must exactly match query_image_ids")
        else:
            selected = tuple(image_id for image_id in manifest_ids if image_id in set(expected))
            if selected != expected:
                raise ValueError("context IDs must occur in score manifest order")
    return payload, _sha256(path)


def _causal_window_status(
    *,
    context_manifest_sha: str,
    query_manifest_sha: str,
    manifest: Mapping[str, Any],
    context_ids: Sequence[str],
    query_ids: Sequence[str],
) -> tuple[bool, str | None]:
    """Verify one contiguous context-then-query window in one ordered export."""

    if context_manifest_sha != query_manifest_sha:
        return False, "context/query use different score manifests"
    manifest_ids = _manifest_image_ids(manifest)
    expected = tuple(str(value) for value in context_ids) + tuple(
        str(value) for value in query_ids
    )
    if not expected:
        return False, "context/query window is empty"
    try:
        start = manifest_ids.index(expected[0])
    except ValueError:
        return False, "context start is absent from score manifest"
    if manifest_ids[start : start + len(expected)] != expected:
        return False, "context/query IDs are not one contiguous context-first window"
    return True, None


def _audit_integer(payload: Mapping[str, Any], name: str) -> int:
    if name not in payload:
        raise KeyError(f"curve manifest is missing {name}")
    value = payload[name]
    if isinstance(value, bool):
        raise TypeError(f"curve manifest {name} must be an integer")
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"curve manifest {name} must be a non-negative integer"
        ) from error
    if not np.isfinite(numeric) or numeric != float(result) or result < 0:
        raise ValueError(f"curve manifest {name} must be a non-negative integer")
    return result


def _audit_probability(payload: Mapping[str, Any], name: str) -> float:
    value = float(payload[name])
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"curve manifest {name} must lie in [0, 1]")
    return value


def _verify_oracle_event_coverage(
    curve_manifest: Mapping[str, Any],
    *,
    oracle_threshold: float,
) -> None:
    """Require the selected oracle to lie inside an event-exact suffix."""

    mode = curve_manifest.get("threshold_mode_requested")
    if mode not in {"adaptive", "exact"}:
        raise ValueError(
            "verified episodes require threshold_mode_requested in {adaptive, exact}"
        )
    global_exact = curve_manifest.get("global_exact")
    if not isinstance(global_exact, bool):
        raise TypeError("curve manifest global_exact must be boolean")

    candidate_count = _audit_integer(curve_manifest, "event_candidate_count")
    covered_count = _audit_integer(curve_manifest, "event_threshold_count")
    added_count = _audit_integer(curve_manifest, "event_thresholds_added")
    if covered_count > candidate_count:
        raise ValueError("curve event_threshold_count exceeds event_candidate_count")
    candidate_lower_bound = _audit_probability(
        curve_manifest, "event_candidate_score_lower_bound"
    )
    if mode == "exact" and candidate_lower_bound != 0.0:
        raise ValueError("exact threshold mode must use candidate lower bound 0")

    capped = curve_manifest.get("event_thresholds_capped")
    if not isinstance(capped, bool):
        raise TypeError("curve manifest event_thresholds_capped must be boolean")
    cap_value = curve_manifest.get("event_threshold_cap")
    if cap_value is None:
        cap = None
    else:
        cap = _audit_integer(curve_manifest, "event_threshold_cap")
        if cap <= 0:
            raise ValueError("curve manifest event_threshold_cap must be positive or null")
    if capped and cap is None:
        raise ValueError("capped curve must record a positive event_threshold_cap")
    if cap is not None and added_count > cap:
        raise ValueError("event_thresholds_added exceeds event_threshold_cap")

    coverage_fraction = _audit_probability(
        curve_manifest, "event_coverage_fraction_lower_bound"
    )
    expected_fraction = covered_count / candidate_count if candidate_count else 1.0
    if not np.isclose(coverage_fraction, expected_fraction, rtol=0.0, atol=1e-12):
        raise ValueError("curve event coverage fraction disagrees with event counts")
    expected_global_exact = mode == "exact" and covered_count == candidate_count and not capped
    if global_exact != expected_global_exact:
        raise ValueError("curve global_exact flag disagrees with its event coverage audit")
    if global_exact:
        return

    threshold = float(oracle_threshold)
    if np.isclose(threshold, 1.0, rtol=0.0, atol=1e-12):
        return
    coverage_raw = curve_manifest.get("event_coverage_score_lower_bound")
    if candidate_count == 0:
        if coverage_raw is not None:
            raise ValueError(
                "zero-event curve must use null event_coverage_score_lower_bound"
            )
        if threshold + 1e-12 < candidate_lower_bound:
            raise ValueError(
                "oracle threshold lies below the event-candidate lower bound of a "
                "zero-event curve"
            )
        return
    if coverage_raw is None:
        raise ValueError("curve has events but records no complete event-exact suffix")
    coverage_lower_bound = _audit_probability(
        curve_manifest, "event_coverage_score_lower_bound"
    )
    if coverage_lower_bound + 1e-12 < candidate_lower_bound:
        raise ValueError("event coverage suffix starts below the candidate score range")
    if threshold + 1e-12 < coverage_lower_bound:
        raise ValueError(
            "oracle threshold lies below the curve's complete event-exact suffix"
        )


def _budget_from_spec(payload: Mapping[str, Any]) -> BudgetSpec:
    if "budgets" in payload:
        return BudgetSpec.from_dict(payload["budgets"])
    pixel = payload.get("pixel_budget")
    component = payload.get("component_budget")
    active_payload = payload.get("budget_active")
    if active_payload is None:
        return BudgetSpec.from_optional(pixel, component)
    active = tuple(bool(value) for value in active_payload)
    values = (
        0.0 if pixel is None else float(pixel),
        0.0 if component is None else float(component),
    )
    return BudgetSpec(values=values, active=active)  # type: ignore[arg-type]


def _episode_from_spec(
    payload: Mapping[str, Any],
    *,
    root: Path,
    index: int,
    default_transform: str,
) -> RCEpisode:
    pseudo_target = str(payload["pseudo_target"])
    statistics_config = StatisticsConfig.from_dict(payload["statistics_config"])
    fold = FoldContract(
        outer_fold_id=str(payload["outer_fold_id"]),
        outer_target=str(payload["outer_target"]),
        detector_source_domains=tuple(str(value) for value in payload["detector_source_domains"]),
        detector_checkpoint_sha=str(payload["detector_checkpoint_sha"]),
        held_out_domains=tuple(str(value) for value in payload["held_out_domains"]),
        protocol_scope=str(payload["protocol_scope"]),
    )
    source_reference_path = _resolve_path(root, payload["source_reference"])
    source_reference = load_source_reference(
        source_reference_path, statistics_config=statistics_config
    )
    fold.assert_matches_source_reference(source_reference)
    if pseudo_target not in fold.held_out_domains:
        raise ValueError("pseudo_target must occur in the detector held_out_domains contract")
    context_ids_requested = tuple(str(value) for value in payload["context_image_ids"])
    query_ids = tuple(str(value) for value in payload["query_image_ids"])
    if "p_min" not in payload:
        raise ValueError("p_min must be explicit in every episode spec")

    if "context_manifest" not in payload:
        raise ValueError("verified episode construction requires context_manifest")
    context_manifest_path = _resolve_path(root, payload["context_manifest"])
    context_manifest, context_manifest_sha = _verify_score_manifest(
        context_manifest_path,
        expected_target=pseudo_target,
        expected_checkpoint_sha=fold.detector_checkpoint_sha,
        expected_outer_fold_id=fold.outer_fold_id,
        expected_outer_target=fold.outer_target,
        expected_detector_sources=fold.detector_source_domains,
        expected_held_out_domains=fold.held_out_domains,
        expected_protocol_scope=fold.protocol_scope,
        expected_image_ids=context_ids_requested,
        exact_image_ids=False,
    )
    if str(payload.get("context_score_manifest_sha256", "")).lower() != context_manifest_sha:
        raise ValueError("spec context_score_manifest_sha256 does not match context manifest")
    manifest_items = context_manifest.get("items", context_manifest.get("records"))
    if not isinstance(manifest_items, list):
        raise ValueError("context manifest requires items/records")
    by_id = {str(item["image_id"]): item for item in manifest_items}
    records = [by_id[image_id] for image_id in context_ids_requested]
    record_root = context_manifest_path.parent
    manifest_dataset_dir = context_manifest.get("dataset_dir")
    probabilities = []
    grays = []
    context_ids = []
    context_paths = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("context records must be mappings")
        probability_value = record.get(
            "prob_path", record.get("score_path", record.get("file"))
        )
        if probability_value is None:
            raise KeyError("context record requires file, prob_path, or score_path")
        probability_path = _resolve_path(record_root, probability_value)
        grayscale_value = record.get("gray_path", record.get("image_path"))
        if grayscale_value is None and manifest_dataset_dir is not None:
            image_root = Path(manifest_dataset_dir)
            if not image_root.is_absolute():
                image_root = context_manifest_path.parent / image_root
            image_root = image_root / "images"
            matches = sorted(image_root.glob(f"{record['image_id']}.*"))
            grayscale_path = matches[0] if matches else None
        else:
            grayscale_path = (
                None if grayscale_value is None else _resolve_path(record_root, grayscale_value)
            )
        probability, grayscale = load_probability_and_grayscale(
            probability_path, grayscale_path
        )
        probabilities.append(probability)
        grays.append(grayscale)
        context_paths.append(_portable_path(probability_path, root))
        context_ids.append(str(record.get("image_id", probability_path.stem)))
    if any(value is None for value in grays) and not all(value is None for value in grays):
        raise ValueError("either every context record or no context record must provide grayscale")
    grayscale_images = None if all(value is None for value in grays) else grays
    statistics = extract_unlabeled_statistics(
        probabilities,
        grayscale_images,
        source_reference=source_reference,
        statistics_config=statistics_config,
    )
    if tuple(context_ids) != context_ids_requested:
        raise ValueError("context_image_ids disagree with context record order")

    if "curve_manifest" in payload:
        curve_manifest_path = _resolve_path(root, payload["curve_manifest"])
        curve_manifest = _read_json(curve_manifest_path)
        curve_manifest_sha = _sha256(curve_manifest_path)
        if str(payload.get("curve_manifest_sha256", "")).lower() != curve_manifest_sha:
            raise ValueError("spec curve_manifest_sha256 does not match curve manifest")
        curve_image_ids = _manifest_image_ids(curve_manifest)
        if int(curve_manifest.get("num_images", -1)) != len(curve_image_ids):
            raise ValueError("curve manifest num_images disagrees with image_ids")
        if str(curve_manifest.get("target_dataset", "")) != pseudo_target:
            raise ValueError("curve manifest target_dataset must equal pseudo_target")
        if _checkpoint_sha(curve_manifest) != fold.detector_checkpoint_sha:
            raise ValueError("curve manifest checkpoint SHA differs from detector contract")
        curve_path = _resolve_path(curve_manifest_path.parent, curve_manifest["curve_file"])
        if "curve_path" in payload and _resolve_path(root, payload["curve_path"]).resolve() != curve_path.resolve():
            raise ValueError("spec curve_path disagrees with curve_manifest curve_file")
        curve_sha = _sha256(curve_path)
        if str(curve_manifest.get("curve_sha256", "")).lower() != curve_sha:
            raise ValueError("curve_file SHA-256 does not match curve manifest")
        query_manifest_path = _resolve_path(
            curve_manifest_path.parent, curve_manifest["score_manifest_file"]
        )
        query_manifest, query_manifest_sha = _verify_score_manifest(
            query_manifest_path,
            expected_target=pseudo_target,
            expected_checkpoint_sha=fold.detector_checkpoint_sha,
            expected_outer_fold_id=fold.outer_fold_id,
            expected_outer_target=fold.outer_target,
            expected_detector_sources=fold.detector_source_domains,
            expected_held_out_domains=fold.held_out_domains,
            expected_protocol_scope=fold.protocol_scope,
            expected_image_ids=query_ids,
            exact_image_ids=False,
        )
        if str(curve_manifest.get("score_manifest_sha256", "")).lower() != query_manifest_sha:
            raise ValueError("query score manifest SHA-256 does not match curve manifest")
        if str(payload.get("query_score_manifest_sha256", "")).lower() != query_manifest_sha:
            raise ValueError("spec query_score_manifest_sha256 does not match query manifest")
        causal_window_verified, causal_window_issue = _causal_window_status(
            context_manifest_sha=context_manifest_sha,
            query_manifest_sha=query_manifest_sha,
            manifest=query_manifest,
            context_ids=context_ids_requested,
            query_ids=query_ids,
        )
        provenance_status = (
            "verified" if causal_window_verified else "asserted_unverified"
        )
    else:
        # Hand assertions are retained only for diagnostics.  The training CLI
        # rejects asserted_unverified episodes from main results.
        curve_manifest_path = None
        curve_manifest_sha = ""
        curve_path = _resolve_path(root, payload["curve_path"])
        curve_sha = _sha256(curve_path)
        curve_image_ids = tuple(str(value) for value in payload["curve_image_ids"])
        query_manifest_path = _resolve_path(root, payload["query_score_manifest"])
        query_manifest, query_manifest_sha = _verify_score_manifest(
            query_manifest_path,
            expected_target=pseudo_target,
            expected_checkpoint_sha=fold.detector_checkpoint_sha,
            expected_outer_fold_id=fold.outer_fold_id,
            expected_outer_target=fold.outer_target,
            expected_detector_sources=fold.detector_source_domains,
            expected_held_out_domains=fold.held_out_domains,
            expected_protocol_scope=fold.protocol_scope,
            expected_image_ids=query_ids,
            exact_image_ids=True,
        )
        if str(payload.get("query_score_manifest_sha256", "")).lower() != query_manifest_sha:
            raise ValueError("spec query_score_manifest_sha256 does not match query manifest")
        provenance_status = "asserted_unverified"
        causal_window_verified = False
        causal_window_issue = "curve provenance is hand-asserted without a curve manifest"
    if curve_image_ids != query_ids:
        raise ValueError(
            "curve image IDs must exactly equal query_image_ids in the same order; "
            f"curve={curve_image_ids}, query={query_ids}"
        )
    curve = _load_curve_csv(curve_path)
    if provenance_status == "verified":
        for field in ("num_images", "gt_objects", "total_pixels"):
            if field not in curve_manifest:
                raise ValueError(f"curve manifest is missing audited count {field!r}")
            if field not in curve:
                raise ValueError(f"curve CSV is missing audited count {field!r}")
            values = {int(value) for value in curve[field]}
            if values != {int(curve_manifest[field])}:
                raise ValueError(f"curve CSV {field} disagrees with curve manifest")
        if int(curve_manifest["num_images"]) != len(query_ids):
            raise ValueError("curve manifest num_images must equal query_image_ids length")
        score_manifest_count = int(query_manifest.get("num_images", -1))
        if int(curve_manifest.get("score_manifest_num_images", -2)) != score_manifest_count:
            raise ValueError("curve manifest score_manifest_num_images is inconsistent")
    provenance = EpisodeProvenance(
        status=provenance_status,
        curve_file_sha256=curve_sha,
        curve_manifest_sha256=curve_manifest_sha,
        context_score_manifest_sha256=context_manifest_sha,
        query_score_manifest_sha256=query_manifest_sha,
        query_score_target_dataset=str(query_manifest["target_dataset"]),
    )
    episode = build_meta_episode(
        episode_id=str(payload.get("episode_id", f"{pseudo_target}:{index:06d}")),
        pseudo_target=pseudo_target,
        context_image_ids=context_ids,
        query_image_ids=query_ids,
        statistics=statistics,
        statistics_config=statistics_config,
        source_reference=source_reference,
        fold=fold,
        provenance=provenance,
        curve=curve,
        budgets=_budget_from_spec(payload),
        p_min=float(payload["p_min"]),
        threshold_transform=str(payload.get("threshold_transform", default_transform)),
        metadata={
            "curve_file": _portable_path(curve_path, root),
            "curve_sha256": _sha256(curve_path),
            "curve_manifest_file": (
                None if curve_manifest_path is None else _portable_path(curve_manifest_path, root)
            ),
            "curve_manifest_sha256": (
                None if curve_manifest_path is None else curve_manifest_sha
            ),
            "curve_provenance_status": provenance_status,
            "causal_window_verified": causal_window_verified,
            "causal_window_issue": causal_window_issue,
            "query_score_manifest_file": _portable_path(query_manifest_path, root),
            "context_score_manifest_file": _portable_path(context_manifest_path, root),
            "source_reference_file": _portable_path(source_reference_path, root),
            "context_score_paths": context_paths,
        },
    )
    if provenance_status == "verified":
        _verify_oracle_event_coverage(
            curve_manifest,
            oracle_threshold=episode.oracle_threshold,
        )
    return episode


def write_episodes(path: str | Path, episodes: Sequence[RCEpisode]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-file", required=True, help="JSON list or {'episodes': [...]} spec")
    parser.add_argument("--output", required=True, help="Output JSONL episode file")
    parser.add_argument(
        "--threshold-transform",
        choices=VALID_THRESHOLD_TRANSFORMS,
        default="identity",
        help="Schema-level loss transform; oracle thresholds remain raw",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    spec_path = Path(args.spec_file).resolve()
    with spec_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    specs = payload.get("episodes", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(specs, list) or not specs:
        raise ValueError("spec file must contain a non-empty episode list")
    episodes = [
        _episode_from_spec(
            spec,
            root=spec_path.parent,
            index=index,
            default_transform=args.threshold_transform,
        )
        for index, spec in enumerate(specs)
    ]
    write_episodes(args.output, episodes)
    summary = {
        "num_episodes": len(episodes),
        "pseudo_targets": sorted({episode.pseudo_target for episode in episodes}),
        "feature_dim": len(episodes[0].feature_names),
        "input_dim": len(episodes[0].input_feature_names),
        "reject_rate": sum(episode.reject for episode in episodes) / len(episodes),
        "output": Path(args.output).name,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

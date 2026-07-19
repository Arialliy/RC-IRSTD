"""Crossed seed x window/query bootstrap core for the RC5 primary gate.

Training seed and evaluation sample uncertainty are distinct factors.  RC5
therefore resamples three seed slots and, independently, draws the domain's
window/query hierarchy exactly once.  The same window/query draw is crossed
with every selected seed slot and is shared by T8 and T4.  No selected seed or
method identifier enters the window/query draw preimage.

This file is a pure in-memory/stateless core.  Artifact I/O and publication
are intentionally left to the later run-contract layer; unit tests can audit
the statistical geometry without opening datasets or observed result files.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


CROSSED_BOOTSTRAP_SCHEMA = "rc-irstd.stage2-crossed-paired-bootstrap.v1"
PROTOCOL_ID = "outer_fixed_seed_x_window_query_crossed_paired_bootstrap_v1"
SEED_FACTOR_TAG = "rc-irstd.stage2.bootstrap.seed-factor.v3"
WINDOW_FACTOR_TAG = "rc-irstd.stage2.bootstrap.window-factor.v3"
QUERY_FACTOR_TAG = "rc-irstd.stage2.bootstrap.query-factor.v3"
FACTOR_DRAW_STREAM_DIGEST_ALGORITHM = (
    "sha256-u64be-length-prefixed-canonical-json-factor-draw-stream-v1"
)
DOMAIN_ORDER = (
    "outer_leave_nuaa_sirst",
    "outer_leave_nudt_sirst",
    "outer_leave_irstd_1k",
)
WINDOW_COUNT_BY_DOMAIN = {
    "outer_leave_nuaa_sirst": 1,
    "outer_leave_nudt_sirst": 3,
    "outer_leave_irstd_1k": 3,
}
BASE_SEED_ORDER = (42, 123, 3407)
METHOD_ORDER = ("T8", "T4")
PRIMARY_BUDGET_NUMERATOR = 1
PRIMARY_BUDGET_DENOMINATOR = 100_000
PRIMARY_RESAMPLES = 10_000
CI_QUANTILES = (0.025, 0.975)
MINIMUM_QUERY_SIZE = 28


class Stage2CrossedBootstrapError(ValueError):
    """An RC5 pair, factor draw or sufficient-count row is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage2CrossedBootstrapError(f"{name} must be int >= {minimum}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Stage2CrossedBootstrapError(f"{name} must be nonempty text")
    return value


def _exact_keys(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise Stage2CrossedBootstrapError(
            f"{name} fields differ; missing={sorted(fields-observed)}, "
            f"extra={sorted(observed-fields)}"
        )
    return value


def _count_row(value: Any, name: str) -> dict[str, Any]:
    row = _exact_keys(
        value,
        {
            "image_id",
            "original_image_sha256",
            "false_positive_pixels",
            "total_pixels",
            "background_pixels",
            "matched_targets",
            "ground_truth_targets",
        },
        name,
    )
    result = {
        "image_id": _text(row["image_id"], f"{name}.image_id"),
        "original_image_sha256": _text(
            row["original_image_sha256"], f"{name}.original_image_sha256"
        ),
    }
    if len(result["original_image_sha256"]) != 64 or any(
        char not in "0123456789abcdef" for char in result["original_image_sha256"]
    ):
        raise Stage2CrossedBootstrapError(f"{name}.original_image_sha256 is invalid")
    for field in (
        "false_positive_pixels",
        "total_pixels",
        "background_pixels",
        "matched_targets",
        "ground_truth_targets",
    ):
        result[field] = _strict_int(row[field], f"{name}.{field}")
    if result["total_pixels"] <= 0:
        raise Stage2CrossedBootstrapError(f"{name}.total_pixels must be positive")
    if result["background_pixels"] > result["total_pixels"]:
        raise Stage2CrossedBootstrapError(f"{name}.background_pixels exceeds total")
    if result["false_positive_pixels"] > result["background_pixels"]:
        raise Stage2CrossedBootstrapError(
            f"{name}.false_positive_pixels exceeds background"
        )
    if result["matched_targets"] > result["ground_truth_targets"]:
        raise Stage2CrossedBootstrapError(f"{name}.matched_targets exceeds GT")
    return result


def validate_crossed_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    """Validate variable-Q pairing and all cross-seed/method identities."""

    top = _exact_keys(pair, {"schema_version", "domains"}, "pair")
    if top["schema_version"] != CROSSED_BOOTSTRAP_SCHEMA:
        raise Stage2CrossedBootstrapError("pair schema mismatch")
    domains = top["domains"]
    if not isinstance(domains, list) or len(domains) != len(DOMAIN_ORDER):
        raise Stage2CrossedBootstrapError("pair must contain three fixed domains")
    canonical_domains: list[dict[str, Any]] = []
    for domain_index, (raw_domain, expected_domain) in enumerate(
        zip(domains, DOMAIN_ORDER, strict=True)
    ):
        domain = _exact_keys(
            raw_domain, {"outer_fold_id", "window_count", "cells"},
            f"domains[{domain_index}]"
        )
        if domain["outer_fold_id"] != expected_domain:
            raise Stage2CrossedBootstrapError("domain order changed")
        expected_windows = WINDOW_COUNT_BY_DOMAIN[expected_domain]
        if domain["window_count"] != expected_windows:
            raise Stage2CrossedBootstrapError("domain window count changed")
        cells = domain["cells"]
        if not isinstance(cells, list) or len(cells) != 3:
            raise Stage2CrossedBootstrapError("every domain needs three seed cells")
        canonical_cells: list[dict[str, Any]] = []
        reference_identity: list[dict[str, Any]] | None = None
        for seed_index, (raw_cell, expected_seed) in enumerate(
            zip(cells, BASE_SEED_ORDER, strict=True)
        ):
            cell = _exact_keys(
                raw_cell, {"base_seed", "methods"},
                f"domains[{domain_index}].cells[{seed_index}]"
            )
            if cell["base_seed"] != expected_seed:
                raise Stage2CrossedBootstrapError("base-seed order changed")
            methods = cell["methods"]
            if not isinstance(methods, Mapping) or set(methods) != set(METHOD_ORDER):
                raise Stage2CrossedBootstrapError(
                    "methods must contain exactly T8 and T4"
                )
            canonical_methods: dict[str, Any] = {}
            seed_identity: list[dict[str, Any]] | None = None
            for method in METHOD_ORDER:
                method_payload = _exact_keys(
                    methods[method], {"windows"},
                    f"domains[{domain_index}].cells[{seed_index}].methods.{method}"
                )
                windows = method_payload["windows"]
                if not isinstance(windows, list) or len(windows) != expected_windows:
                    raise Stage2CrossedBootstrapError("method window count mismatch")
                canonical_windows: list[dict[str, Any]] = []
                identities: list[dict[str, Any]] = []
                for window_index, raw_window in enumerate(windows):
                    window = _exact_keys(
                        raw_window,
                        {"window_id", "window_identity_sha256", "query_counts"},
                        f"{method}.windows[{window_index}]",
                    )
                    window_id = _text(window["window_id"], "window_id")
                    window_sha = _text(
                        window["window_identity_sha256"], "window_identity_sha256"
                    )
                    if len(window_sha) != 64 or any(
                        char not in "0123456789abcdef" for char in window_sha
                    ):
                        raise Stage2CrossedBootstrapError(
                            "window identity SHA is invalid"
                        )
                    rows = window["query_counts"]
                    if not isinstance(rows, list) or len(rows) < MINIMUM_QUERY_SIZE:
                        raise Stage2CrossedBootstrapError(
                            "every RC5 window needs at least 28 query rows"
                        )
                    counts = [
                        _count_row(row, f"{method}.windows[{window_index}].query[{q}]")
                        for q, row in enumerate(rows)
                    ]
                    identity_rows = [
                        {
                            "image_id": row["image_id"],
                            "original_image_sha256": row["original_image_sha256"],
                            "total_pixels": row["total_pixels"],
                            "background_pixels": row["background_pixels"],
                            "ground_truth_targets": row["ground_truth_targets"],
                        }
                        for row in counts
                    ]
                    identities.append(
                        {
                            "window_id": window_id,
                            "window_identity_sha256": window_sha,
                            "query_size": len(counts),
                            "query_identity": identity_rows,
                        }
                    )
                    canonical_windows.append(
                        {
                            "window_id": window_id,
                            "window_identity_sha256": window_sha,
                            "query_counts": counts,
                        }
                    )
                if seed_identity is None:
                    seed_identity = identities
                elif identities != seed_identity:
                    raise Stage2CrossedBootstrapError(
                        "T8/T4 window-query identities are not byte-identical"
                    )
                canonical_methods[method] = {"windows": canonical_windows}
            assert seed_identity is not None
            if reference_identity is None:
                reference_identity = seed_identity
            elif seed_identity != reference_identity:
                raise Stage2CrossedBootstrapError(
                    "window-query identities differ across training seeds"
                )
            canonical_cells.append(
                {"base_seed": expected_seed, "methods": canonical_methods}
            )
        canonical_domains.append(
            {
                "outer_fold_id": expected_domain,
                "window_count": expected_windows,
                "cells": canonical_cells,
            }
        )
    return {"schema_version": CROSSED_BOOTSTRAP_SCHEMA, "domains": canonical_domains}


def _draw_index(tag: str, root: int, parts: Sequence[Any], population: int) -> int:
    _strict_int(root, "factor root", minimum=1)
    _strict_int(population, "population", minimum=1)
    return _draw_index_canonical(tag, root, parts, population)


def _draw_index_canonical(
    tag: str, root: int, parts: Sequence[Any], population: int
) -> int:
    """Fast stateless draw after roots and geometry have been validated once."""

    preimage = canonical_json_bytes(
        {"tag": tag, "root": root, "parts": list(parts)}
    )
    return int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big") % population


def _factor_mode(value: Any) -> str:
    if not isinstance(value, str) or value not in {
        "crossed",
        "seed_only",
        "window_query_only",
    }:
        raise Stage2CrossedBootstrapError("unknown factor_mode")
    return value


def _canonical_factor_roots(
    factor_roots: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    if not isinstance(factor_roots, Mapping) or set(factor_roots) != set(DOMAIN_ORDER):
        raise Stage2CrossedBootstrapError(
            "factor roots must cover the three domains"
        )
    canonical: dict[str, dict[str, int]] = {}
    for outer in DOMAIN_ORDER:
        roots = _exact_keys(
            factor_roots[outer],
            {"seed_factor_root", "window_query_factor_root"},
            f"factor_roots.{outer}",
        )
        canonical[outer] = {
            "seed_factor_root": _strict_int(
                roots["seed_factor_root"], "seed_factor_root", minimum=1
            ),
            "window_query_factor_root": _strict_int(
                roots["window_query_factor_root"],
                "window_query_factor_root",
                minimum=1,
            ),
        }
    return canonical


def _generate_crossed_factor_draw_canonical(
    canonical: Mapping[str, Any],
    *,
    factor_roots: Mapping[str, Mapping[str, int]],
    replicate_index: int,
    factor_mode: str,
) -> dict[str, Any]:
    """Private draw path for an already-canonical pair, roots, index, and mode."""

    draw_domains: list[dict[str, Any]] = []
    for domain in canonical["domains"]:
        outer = domain["outer_fold_id"]
        roots = factor_roots[outer]
        seed_root = roots["seed_factor_root"]
        sample_root = roots["window_query_factor_root"]
        if factor_mode == "window_query_only":
            seed_indices = [0, 1, 2]
        else:
            seed_indices = [
                _draw_index_canonical(
                    SEED_FACTOR_TAG,
                    seed_root,
                    [outer, replicate_index, slot],
                    3,
                )
                for slot in range(3)
            ]
        reference_windows = domain["cells"][0]["methods"]["T8"]["windows"]
        sample_draws: list[dict[str, Any]] = []
        for slot in range(domain["window_count"]):
            if factor_mode == "seed_only":
                selected_window = slot
            else:
                selected_window = _draw_index_canonical(
                    WINDOW_FACTOR_TAG,
                    sample_root,
                    [outer, replicate_index, slot],
                    domain["window_count"],
                )
            window = reference_windows[selected_window]
            query_size = len(window["query_counts"])
            if factor_mode == "seed_only":
                query_indices = list(range(query_size))
            else:
                query_indices = [
                    _draw_index_canonical(
                        QUERY_FACTOR_TAG,
                        sample_root,
                        [
                            outer,
                            replicate_index,
                            slot,
                            window["window_id"],
                            qslot,
                        ],
                        query_size,
                    )
                    for qslot in range(query_size)
                ]
            sample_draws.append(
                {
                    "window_draw_slot": slot,
                    "selected_window_index": selected_window,
                    "selected_window_id": window["window_id"],
                    "query_size": query_size,
                    "query_indices": query_indices,
                }
            )
        draw_domains.append(
            {
                "outer_fold_id": outer,
                "selected_seed_indices": seed_indices,
                "shared_window_query_draws": sample_draws,
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "replicate_index": replicate_index,
        "factor_mode": factor_mode,
        "domains": draw_domains,
    }


def generate_crossed_factor_draw(
    pair: Mapping[str, Any],
    *,
    factor_roots: Mapping[str, Mapping[str, int]],
    replicate_index: int,
    factor_mode: str = "crossed",
) -> dict[str, Any]:
    """Generate one factor draw with no method/selected-seed in sample preimages."""

    canonical = validate_crossed_pair(pair)
    replicate = _strict_int(replicate_index, "replicate_index")
    mode = _factor_mode(factor_mode)
    roots = _canonical_factor_roots(factor_roots)
    return _generate_crossed_factor_draw_canonical(
        canonical,
        factor_roots=roots,
        replicate_index=replicate,
        factor_mode=mode,
    )


def _window_metrics(
    counts: Sequence[Mapping[str, Any]], query_indices: Sequence[int]
) -> tuple[float, float, int, int]:
    if len(query_indices) != len(counts):
        raise Stage2CrossedBootstrapError(
            "a query bootstrap must draw the selected window's actual query size"
        )
    fp = pixels = matched = gt = 0
    for raw in query_indices:
        index = _strict_int(raw, "query index")
        if index >= len(counts):
            raise Stage2CrossedBootstrapError("query index exceeds selected window")
        row = counts[index]
        fp += int(row["false_positive_pixels"])
        pixels += int(row["total_pixels"])
        matched += int(row["matched_targets"])
        gt += int(row["ground_truth_targets"])
    if pixels <= 0:
        raise Stage2CrossedBootstrapError("zero pixel denominator")
    satisfied = 1.0 if fp * PRIMARY_BUDGET_DENOMINATOR <= (
        PRIMARY_BUDGET_NUMERATOR * pixels
    ) else 0.0
    fa = fp / pixels
    log_excess = math.log(
        max(fa * PRIMARY_BUDGET_DENOMINATOR / PRIMARY_BUDGET_NUMERATOR, 1.0)
    )
    return satisfied, log_excess, matched, gt


def method_macro_metrics(
    pair: Mapping[str, Any],
    *,
    method_id: str,
    factor_draw: Mapping[str, Any] | None,
) -> tuple[float, float, float]:
    canonical = validate_crossed_pair(pair)
    if method_id not in METHOD_ORDER:
        raise Stage2CrossedBootstrapError("method_id must be T8 or T4")
    if factor_draw is not None:
        if not isinstance(factor_draw, Mapping):
            raise Stage2CrossedBootstrapError("factor draw must be a mapping")
        if factor_draw.get("protocol_id") != PROTOCOL_ID:
            raise Stage2CrossedBootstrapError("factor draw protocol mismatch")
        draw_domains = factor_draw.get("domains")
        if not isinstance(draw_domains, list) or len(draw_domains) != 3:
            raise Stage2CrossedBootstrapError("factor draw domain geometry mismatch")
    return _method_macro_metrics_canonical(
        canonical, method_id=method_id, factor_draw=factor_draw
    )


def _method_macro_metrics_canonical(
    canonical: Mapping[str, Any],
    *,
    method_id: str,
    factor_draw: Mapping[str, Any] | None,
) -> tuple[float, float, float]:
    """Private metric path for a pair that has already passed full validation."""

    domain_bsr: list[float] = []
    domain_log: list[float] = []
    domain_pd: list[float] = []
    for domain_index, domain in enumerate(canonical["domains"]):
        if factor_draw is None:
            seed_indices = [0, 1, 2]
            window_draws = [
                {
                    "selected_window_index": window_index,
                    "query_indices": list(
                        range(
                            len(
                                domain["cells"][0]["methods"]["T8"]["windows"][
                                    window_index
                                ]["query_counts"]
                            )
                        )
                    ),
                }
                for window_index in range(domain["window_count"])
            ]
        else:
            draw_domain = factor_draw["domains"][domain_index]
            if draw_domain.get("outer_fold_id") != domain["outer_fold_id"]:
                raise Stage2CrossedBootstrapError("factor draw domain order changed")
            seed_indices = draw_domain.get("selected_seed_indices")
            window_draws = draw_domain.get("shared_window_query_draws")
            if not isinstance(seed_indices, list) or len(seed_indices) != 3:
                raise Stage2CrossedBootstrapError("seed factor draw is incomplete")
            if (
                not isinstance(window_draws, list)
                or len(window_draws) != domain["window_count"]
            ):
                raise Stage2CrossedBootstrapError(
                    "window/query factor draw is incomplete"
                )
        seed_bsr: list[float] = []
        seed_log: list[float] = []
        seed_pd: list[float] = []
        for raw_seed in seed_indices:
            seed_index = _strict_int(raw_seed, "selected seed index")
            if seed_index >= 3:
                raise Stage2CrossedBootstrapError(
                    "selected seed index exceeds three cells"
                )
            windows = domain["cells"][seed_index]["methods"][method_id]["windows"]
            bsr_values: list[float] = []
            log_values: list[float] = []
            matched_total = gt_total = 0
            for raw_draw in window_draws:
                window_index = _strict_int(
                    raw_draw["selected_window_index"], "selected window index"
                )
                if window_index >= len(windows):
                    raise Stage2CrossedBootstrapError(
                        "selected window index is invalid"
                    )
                counts = windows[window_index]["query_counts"]
                bsr, log_excess, matched, gt = _window_metrics(
                    counts, raw_draw["query_indices"]
                )
                bsr_values.append(bsr)
                log_values.append(log_excess)
                matched_total += matched
                gt_total += gt
            if gt_total <= 0:
                raise Stage2CrossedBootstrapError("zero GT makes Pd undefined")
            seed_bsr.append(sum(bsr_values) / len(bsr_values))
            seed_log.append(sum(log_values) / len(log_values))
            seed_pd.append(matched_total / gt_total)
        domain_bsr.append(sum(seed_bsr) / 3.0)
        domain_log.append(sum(seed_log) / 3.0)
        domain_pd.append(sum(seed_pd) / 3.0)
    return (
        sum(domain_bsr) / 3.0,
        sum(domain_log) / 3.0,
        sum(domain_pd) / 3.0,
    )


def type7_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise Stage2CrossedBootstrapError("type-7 quantile needs values")
    q = float(probability)
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise Stage2CrossedBootstrapError("quantile probability is invalid")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise Stage2CrossedBootstrapError("quantile values must be finite")
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def evaluate_crossed_bootstrap(
    pair: Mapping[str, Any],
    *,
    factor_roots: Mapping[str, Mapping[str, int]],
    resamples: int = PRIMARY_RESAMPLES,
    factor_mode: str = "crossed",
) -> dict[str, Any]:
    """Compute point estimates and percentile intervals from fixed domain strata."""

    canonical = validate_crossed_pair(pair)
    roots = _canonical_factor_roots(factor_roots)
    count = _strict_int(resamples, "resamples", minimum=1)
    mode = _factor_mode(factor_mode)
    point_t8 = _method_macro_metrics_canonical(
        canonical, method_id="T8", factor_draw=None
    )
    point_t4 = _method_macro_metrics_canonical(
        canonical, method_id="T4", factor_draw=None
    )
    bsr_delta: list[float] = []
    pd_delta: list[float] = []
    factor_digest = hashlib.sha256()
    for replicate in range(count):
        draw = _generate_crossed_factor_draw_canonical(
            canonical,
            factor_roots=roots,
            replicate_index=replicate,
            factor_mode=mode,
        )
        draw_bytes = canonical_json_bytes(draw)
        factor_digest.update(len(draw_bytes).to_bytes(8, "big"))
        factor_digest.update(draw_bytes)
        t8 = _method_macro_metrics_canonical(
            canonical, method_id="T8", factor_draw=draw
        )
        t4 = _method_macro_metrics_canonical(
            canonical, method_id="T4", factor_draw=draw
        )
        bsr_delta.append(t8[0] - t4[0])
        pd_delta.append(t8[2] - t4[2])
    return {
        "schema_version": CROSSED_BOOTSTRAP_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "factor_mode": mode,
        "resamples": count,
        "domains_fixed_equal_weight": True,
        "seed_factor_crossed_with_window_query_factor": mode == "crossed",
        "shared_window_query_draw_across_seed_slots_and_methods": True,
        "selected_seed_in_window_query_preimage": False,
        "method_id_in_any_factor_preimage": False,
        "variable_query_size_replayed": True,
        "point_estimate": {
            "T8": {
                "macro_bsr": point_t8[0],
                "macro_log_excess": point_t8[1],
                "macro_pd": point_t8[2],
            },
            "T4": {
                "macro_bsr": point_t4[0],
                "macro_log_excess": point_t4[1],
                "macro_pd": point_t4[2],
            },
            "delta_macro_bsr": point_t8[0] - point_t4[0],
            "delta_macro_pd": point_t8[2] - point_t4[2],
        },
        "confidence_interval": {
            "method": "two_sided_percentile_hyndman_fan_type_7",
            "quantiles": list(CI_QUANTILES),
            "delta_macro_bsr": [
                type7_quantile(bsr_delta, CI_QUANTILES[0]),
                type7_quantile(bsr_delta, CI_QUANTILES[1]),
            ],
            "delta_macro_pd": [
                type7_quantile(pd_delta, CI_QUANTILES[0]),
                type7_quantile(pd_delta, CI_QUANTILES[1]),
            ],
        },
        "factor_draw_stream_sha256_algorithm": (
            FACTOR_DRAW_STREAM_DIGEST_ALGORITHM
        ),
        "factor_draw_stream_sha256": factor_digest.hexdigest(),
    }


__all__ = [
    "BASE_SEED_ORDER",
    "CI_QUANTILES",
    "CROSSED_BOOTSTRAP_SCHEMA",
    "DOMAIN_ORDER",
    "FACTOR_DRAW_STREAM_DIGEST_ALGORITHM",
    "METHOD_ORDER",
    "PRIMARY_RESAMPLES",
    "PROTOCOL_ID",
    "Stage2CrossedBootstrapError",
    "canonical_json_bytes",
    "evaluate_crossed_bootstrap",
    "generate_crossed_factor_draw",
    "method_macro_metrics",
    "type7_quantile",
    "validate_crossed_pair",
]

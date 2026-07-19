from __future__ import annotations

import hashlib
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

import evaluation.stage2_crossed_bootstrap_v3 as bootstrap
from evaluation.stage2_crossed_bootstrap_v3 import (
    BASE_SEED_ORDER,
    CROSSED_BOOTSTRAP_SCHEMA,
    DOMAIN_ORDER,
    FACTOR_DRAW_STREAM_DIGEST_ALGORITHM,
    METHOD_ORDER,
    PRIMARY_BUDGET_DENOMINATOR,
    PRIMARY_BUDGET_NUMERATOR,
    PROTOCOL_ID,
    QUERY_FACTOR_TAG,
    SEED_FACTOR_TAG,
    WINDOW_COUNT_BY_DOMAIN,
    WINDOW_FACTOR_TAG,
    Stage2CrossedBootstrapError,
    canonical_json_bytes,
    evaluate_crossed_bootstrap,
    generate_crossed_factor_draw,
    method_macro_metrics,
    type7_quantile,
    validate_crossed_pair,
)


QUERY_SIZES = {
    DOMAIN_ORDER[0]: (28,),
    DOMAIN_ORDER[1]: (28, 31, 29),
    DOMAIN_ORDER[2]: (30, 28, 33),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _query_counts(
    outer_fold: str,
    seed_index: int,
    window_index: int,
    method_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_size = QUERY_SIZES[outer_fold][window_index]
    for query_index in range(query_size):
        if method_id == "T8":
            fp = 2 if (query_index + seed_index + window_index) % 2 == 0 else 0
            matched = int((query_index + seed_index + window_index) % 4 != 0)
        else:
            fp = 2 if (query_index + 2 * seed_index + window_index) % 3 == 0 else 0
            matched = int((query_index + seed_index + 2 * window_index) % 5 != 0)
        identity = f"{outer_fold}:window-{window_index}:query-{query_index:02d}"
        rows.append(
            {
                "image_id": identity,
                "original_image_sha256": _sha(f"{identity}:image"),
                "false_positive_pixels": fp,
                "total_pixels": 100_000,
                "background_pixels": 99_999,
                "matched_targets": matched,
                "ground_truth_targets": 1,
            }
        )
    return rows


def _pair() -> dict[str, Any]:
    domains: list[dict[str, Any]] = []
    for outer_fold in DOMAIN_ORDER:
        cells: list[dict[str, Any]] = []
        for seed_index, base_seed in enumerate(BASE_SEED_ORDER):
            methods: dict[str, Any] = {}
            for method_id in METHOD_ORDER:
                windows: list[dict[str, Any]] = []
                for window_index in range(WINDOW_COUNT_BY_DOMAIN[outer_fold]):
                    window_id = f"{outer_fold}:window-{window_index}"
                    windows.append(
                        {
                            "window_id": window_id,
                            "window_identity_sha256": _sha(
                                f"{window_id}:window-identity"
                            ),
                            "query_counts": _query_counts(
                                outer_fold,
                                seed_index,
                                window_index,
                                method_id,
                            ),
                        }
                    )
                methods[method_id] = {"windows": windows}
            cells.append({"base_seed": base_seed, "methods": methods})
        domains.append(
            {
                "outer_fold_id": outer_fold,
                "window_count": WINDOW_COUNT_BY_DOMAIN[outer_fold],
                "cells": cells,
            }
        )
    return {"schema_version": CROSSED_BOOTSTRAP_SCHEMA, "domains": domains}


def _roots(
    *, seed_offset: int = 0, window_query_offset: int = 0
) -> dict[str, dict[str, int]]:
    return {
        outer_fold: {
            "seed_factor_root": 10_001 + 101 * index + seed_offset,
            "window_query_factor_root": 20_003
            + 103 * index
            + window_query_offset,
        }
        for index, outer_fold in enumerate(DOMAIN_ORDER)
    }


def _seed_projection(draw: Mapping[str, Any]) -> list[list[int]]:
    return [
        list(domain["selected_seed_indices"])
        for domain in draw["domains"]
    ]


def _sample_projection(draw: Mapping[str, Any]) -> list[Any]:
    return [
        domain["shared_window_query_draws"]
        for domain in draw["domains"]
    ]


def test_variable_q_is_valid_and_point_estimate_visits_every_unit_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = validate_crossed_pair(_pair())
    observed: list[tuple[int, tuple[int, ...]]] = []
    real_window_metrics = bootstrap._window_metrics

    def recording_metrics(
        counts: Sequence[Mapping[str, Any]], query_indices: Sequence[int]
    ) -> tuple[float, float, int, int]:
        observed.append((len(counts), tuple(query_indices)))
        return real_window_metrics(counts, query_indices)

    monkeypatch.setattr(bootstrap, "_window_metrics", recording_metrics)
    method_macro_metrics(canonical, method_id="T8", factor_draw=None)

    expected_sizes = [
        size
        for outer_fold in DOMAIN_ORDER
        for _seed in BASE_SEED_ORDER
        for size in QUERY_SIZES[outer_fold]
    ]
    assert [size for size, _indices in observed] == expected_sizes
    assert all(indices == tuple(range(size)) for size, indices in observed)
    assert sum(len(indices) for _size, indices in observed) == sum(expected_sizes)


def test_factor_roots_change_seed_and_window_query_streams_independently() -> None:
    pair = _pair()
    draws = [
        generate_crossed_factor_draw(
            pair,
            factor_roots=_roots(),
            replicate_index=replicate,
        )
        for replicate in range(6)
    ]
    seed_changed = [
        generate_crossed_factor_draw(
            pair,
            factor_roots=_roots(seed_offset=97_531),
            replicate_index=replicate,
        )
        for replicate in range(6)
    ]
    sample_changed = [
        generate_crossed_factor_draw(
            pair,
            factor_roots=_roots(window_query_offset=81_793),
            replicate_index=replicate,
        )
        for replicate in range(6)
    ]

    assert [_sample_projection(draw) for draw in draws] == [
        _sample_projection(draw) for draw in seed_changed
    ]
    assert [_seed_projection(draw) for draw in draws] != [
        _seed_projection(draw) for draw in seed_changed
    ]
    assert [_seed_projection(draw) for draw in draws] == [
        _seed_projection(draw) for draw in sample_changed
    ]
    assert [_sample_projection(draw) for draw in draws] != [
        _sample_projection(draw) for draw in sample_changed
    ]


def test_window_query_draw_is_generated_once_and_has_seed_method_free_preimages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, tuple[Any, ...], int]] = []
    real_draw = bootstrap._draw_index_canonical

    def recording_draw(
        tag: str, root: int, parts: Sequence[Any], population: int
    ) -> int:
        calls.append((tag, root, tuple(parts), population))
        return real_draw(tag, root, parts, population)

    monkeypatch.setattr(bootstrap, "_draw_index_canonical", recording_draw)
    draw = generate_crossed_factor_draw(
        _pair(),
        factor_roots=_roots(),
        replicate_index=17,
    )

    seed_calls = [call for call in calls if call[0] == SEED_FACTOR_TAG]
    window_calls = [call for call in calls if call[0] == WINDOW_FACTOR_TAG]
    query_calls = [call for call in calls if call[0] == QUERY_FACTOR_TAG]
    assert len(seed_calls) == 3 * len(DOMAIN_ORDER)
    assert len(window_calls) == sum(WINDOW_COUNT_BY_DOMAIN.values())
    assert len(query_calls) == sum(
        sample["query_size"]
        for domain in draw["domains"]
        for sample in domain["shared_window_query_draws"]
    )

    for _tag, _root, parts, _population in window_calls:
        assert len(parts) == 3
        assert parts[0] in DOMAIN_ORDER
        assert parts[1] == 17
    for _tag, _root, parts, _population in query_calls:
        assert len(parts) == 5
        assert parts[0] in DOMAIN_ORDER
        assert parts[1] == 17
        assert isinstance(parts[3], str) and ":window-" in parts[3]
    factor_preimages = canonical_json_bytes(
        [
            {"tag": tag, "root": root, "parts": list(parts)}
            for tag, root, parts, _population in calls
        ]
    )
    assert b'"T8"' not in factor_preimages
    assert b'"T4"' not in factor_preimages

    before = canonical_json_bytes(draw)
    calls.clear()
    method_macro_metrics(_pair(), method_id="T8", factor_draw=draw)
    method_macro_metrics(_pair(), method_id="T4", factor_draw=draw)
    assert calls == []
    assert canonical_json_bytes(draw) == before


@pytest.mark.parametrize("tamper_axis", ["method", "seed"])
def test_cross_method_and_cross_seed_identity_tampering_is_rejected(
    tamper_axis: str,
) -> None:
    pair = _pair()
    if tamper_axis == "method":
        row = pair["domains"][1]["cells"][0]["methods"]["T4"]["windows"][1][
            "query_counts"
        ][0]
        row["image_id"] = "method-identity-tamper"
        message = "T8/T4 window-query identities"
    else:
        for method_id in METHOD_ORDER:
            row = pair["domains"][2]["cells"][1]["methods"][method_id]["windows"][2][
                "query_counts"
            ][0]
            row["image_id"] = "cross-seed-identity-tamper"
            row["original_image_sha256"] = _sha("cross-seed-identity-tamper")
        message = "differ across training seeds"
    with pytest.raises(Stage2CrossedBootstrapError, match=message):
        validate_crossed_pair(pair)


def test_method_mapping_order_is_irrelevant_but_field_closure_is_exact() -> None:
    pair = _pair()
    methods = pair["domains"][1]["cells"][0]["methods"]
    pair["domains"][1]["cells"][0]["methods"] = {
        "T4": methods["T4"],
        "T8": methods["T8"],
    }
    validate_crossed_pair(pair)

    extra = deepcopy(pair)
    extra["domains"][1]["cells"][0]["methods"]["T9"] = deepcopy(
        extra["domains"][1]["cells"][0]["methods"]["T8"]
    )
    with pytest.raises(
        Stage2CrossedBootstrapError, match="exactly T8 and T4"
    ):
        validate_crossed_pair(extra)


def test_primary_budget_comparison_uses_exact_integer_arithmetic() -> None:
    huge_pixels = 10**18
    exact_fp = (
        PRIMARY_BUDGET_NUMERATOR * huge_pixels // PRIMARY_BUDGET_DENOMINATOR
    )

    def row(false_positive_pixels: int) -> dict[str, int]:
        return {
            "false_positive_pixels": false_positive_pixels,
            "total_pixels": huge_pixels,
            "matched_targets": 1,
            "ground_truth_targets": 1,
        }

    exact = bootstrap._window_metrics([row(exact_fp)], [0])
    above = bootstrap._window_metrics([row(exact_fp + 1)], [0])
    assert exact[0] == 1.0
    assert above[0] == 0.0
    assert above[1] > 0.0


def test_type7_quantile_and_reported_interval_contract() -> None:
    values = [0.0, 10.0, 20.0, 30.0]
    assert type7_quantile(values, 0.025) == pytest.approx(0.75)
    assert type7_quantile(values, 0.5) == 15.0
    assert type7_quantile(values, 0.975) == pytest.approx(29.25)
    report = evaluate_crossed_bootstrap(
        _pair(), factor_roots=_roots(), resamples=40
    )
    interval = report["confidence_interval"]
    assert interval["method"] == "two_sided_percentile_hyndman_fan_type_7"
    assert interval["quantiles"] == [0.025, 0.975]
    assert interval["delta_macro_bsr"][0] <= interval["delta_macro_bsr"][1]
    assert interval["delta_macro_pd"][0] <= interval["delta_macro_pd"][1]


def test_seed_only_and_window_query_only_sensitivity_freeze_the_other_factor() -> None:
    pair = _pair()
    seed_only = [
        generate_crossed_factor_draw(
            pair,
            factor_roots=_roots(),
            replicate_index=replicate,
            factor_mode="seed_only",
        )
        for replicate in range(8)
    ]
    assert len(
        {
            canonical_json_bytes(_seed_projection(draw))
            for draw in seed_only
        }
    ) > 1
    assert all(
        sample["selected_window_index"] == sample["window_draw_slot"]
        and sample["query_indices"] == list(range(sample["query_size"]))
        for draw in seed_only
        for domain in draw["domains"]
        for sample in domain["shared_window_query_draws"]
    )

    window_query_only = [
        generate_crossed_factor_draw(
            pair,
            factor_roots=_roots(),
            replicate_index=replicate,
            factor_mode="window_query_only",
        )
        for replicate in range(8)
    ]
    assert all(
        _seed_projection(draw) == [[0, 1, 2]] * len(DOMAIN_ORDER)
        for draw in window_query_only
    )
    assert len(
        {
            canonical_json_bytes(_sample_projection(draw))
            for draw in window_query_only
        }
    ) > 1

    seed_report = evaluate_crossed_bootstrap(
        pair,
        factor_roots=_roots(),
        resamples=20,
        factor_mode="seed_only",
    )
    sample_report = evaluate_crossed_bootstrap(
        pair,
        factor_roots=_roots(),
        resamples=20,
        factor_mode="window_query_only",
    )
    assert seed_report["factor_mode"] == "seed_only"
    assert sample_report["factor_mode"] == "window_query_only"
    assert seed_report["seed_factor_crossed_with_window_query_factor"] is False
    assert sample_report["seed_factor_crossed_with_window_query_factor"] is False
    assert (
        seed_report["factor_draw_stream_sha256"]
        != sample_report["factor_draw_stream_sha256"]
    )


def test_nuaa_single_window_factor_degenerates_but_query_factor_does_not() -> None:
    draws = [
        generate_crossed_factor_draw(
            _pair(),
            factor_roots=_roots(),
            replicate_index=replicate,
        )
        for replicate in range(8)
    ]
    nuaa_samples = [
        draw["domains"][0]["shared_window_query_draws"][0]
        for draw in draws
    ]
    assert all(sample["selected_window_index"] == 0 for sample in nuaa_samples)
    assert all(
        sample["selected_window_id"].endswith("window-0")
        for sample in nuaa_samples
    )
    assert len(
        {
            tuple(sample["query_indices"])
            for sample in nuaa_samples
        }
    ) > 1


def test_evaluator_validates_pair_once_then_uses_canonical_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_validate = bootstrap.validate_crossed_pair

    def recording_validate(pair: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return real_validate(pair)

    monkeypatch.setattr(bootstrap, "validate_crossed_pair", recording_validate)
    report = bootstrap.evaluate_crossed_bootstrap(
        _pair(), factor_roots=_roots(), resamples=11
    )
    assert calls == 1
    assert report["resamples"] == 11
    assert report["protocol_id"] == PROTOCOL_ID


def test_factor_draw_stream_digest_uses_u64be_length_framing() -> None:
    pair = _pair()
    roots = _roots()
    digest = hashlib.sha256()
    for replicate in range(4):
        draw_bytes = canonical_json_bytes(
            generate_crossed_factor_draw(
                pair,
                factor_roots=roots,
                replicate_index=replicate,
            )
        )
        digest.update(len(draw_bytes).to_bytes(8, "big"))
        digest.update(draw_bytes)
    report = evaluate_crossed_bootstrap(
        pair, factor_roots=roots, resamples=4
    )
    assert report["factor_draw_stream_sha256_algorithm"] == (
        FACTOR_DRAW_STREAM_DIGEST_ALGORITHM
    )
    assert report["factor_draw_stream_sha256"] == digest.hexdigest()


def test_10000_synthetic_resamples_complete_with_variable_q() -> None:
    report = evaluate_crossed_bootstrap(
        _pair(), factor_roots=_roots(), resamples=10_000
    )
    assert report["resamples"] == 10_000
    assert len(report["factor_draw_stream_sha256"]) == 64
    assert report["shared_window_query_draw_across_seed_slots_and_methods"] is True
    assert report["variable_query_size_replayed"] is True

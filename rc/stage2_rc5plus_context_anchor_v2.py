"""Derive anchor-v2 only from a freshly replayed label-blind producer bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rc.build_stage2_rc5_context import (
    VerifiedStage2RC5ContextBundle,
    _prepare_inputs,
    _produce_context_and_anchor,
    canonical_json_bytes,
    replay_verified_stage2_rc5_context_bundle,
)
from rc.stage2_context_tail_anchor_v2 import (
    VerifiedContextTailAnchorV2,
    build_context_tail_anchor_v2,
    verify_context_tail_anchor_v2,
)


class Stage2RC5PlusContextAnchorV2Error(ValueError):
    """The v2 anchor did not replay from the producer's exact context maps."""


def build_context_tail_anchor_v2_from_producer_bundle(
    *,
    producer_bundle: VerifiedStage2RC5ContextBundle,
    requested_budget_rationals: Sequence[tuple[int, int]] = (),
) -> VerifiedContextTailAnchorV2:
    """Reopen only the bundle-bound 14 context maps and issue anchor-v2."""

    try:
        bundle = replay_verified_stage2_rc5_context_bundle(producer_bundle)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Stage2RC5PlusContextAnchorV2Error(
            "producer bundle failed current-state replay"
        ) from error
    statistics = bundle.producer_manifest.get("inputs", {}).get(
        "statistics_config"
    )
    if (
        not isinstance(statistics, Mapping)
        or set(statistics) != {"path", "sha256"}
    ):
        raise Stage2RC5PlusContextAnchorV2Error(
            "producer statistics-config binding is invalid"
        )
    root = bundle.score_manifest_metadata.repository_root
    window_index = bundle.context.payload.get("window_index")
    if type(window_index) is not int or window_index < 0:
        raise Stage2RC5PlusContextAnchorV2Error(
            "producer context window index is invalid"
        )
    try:
        prepared = _prepare_inputs(
            variable_query_window=bundle.variable_query_window,
            score_bundle=bundle.score_bundle,
            source_reference=bundle.source_reference,
            statistics_config=bundle.statistics_config,
            statistics_config_path=root / str(statistics["path"]),
            statistics_config_sha256=str(statistics["sha256"]),
            window_index=window_index,
            repository_root=root,
        )
        replayed_context, replayed_v1_anchor, probability_maps = (
            _produce_context_and_anchor(prepared)
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Stage2RC5PlusContextAnchorV2Error(
            "producer context maps failed direct label-blind replay"
        ) from error
    if (
        replayed_context.canonical_payload != bundle.context.canonical_payload
        or replayed_context.payload_sha256 != bundle.context.payload_sha256
        or canonical_json_bytes(replayed_v1_anchor.payload)
        != canonical_json_bytes(bundle.anchor.payload)
    ):
        raise Stage2RC5PlusContextAnchorV2Error(
            "direct context-map replay differs from the producer capability"
        )
    identity = str(bundle.context.payload["context_full_identity_sha256"])
    payload = build_context_tail_anchor_v2(
        context_probability_maps=probability_maps,
        context_identity_sha256=identity,
        requested_budget_rationals=requested_budget_rationals,
    )
    return verify_context_tail_anchor_v2(
        payload,
        context_probability_maps=probability_maps,
        expected_context_identity_sha256=identity,
        expected_requested_budget_rationals=requested_budget_rationals,
    )


__all__ = [
    "Stage2RC5PlusContextAnchorV2Error",
    "build_context_tail_anchor_v2_from_producer_bundle",
]

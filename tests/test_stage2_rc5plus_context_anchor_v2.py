from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from rc.stage2_context_tail_anchor import (
    build_context_tail_anchor,
    canonical_json_bytes,
    verify_context_tail_anchor,
)
import rc.stage2_rc5plus_context_anchor_v2 as producer_v2
from rc.stage2_rc5plus_context_anchor_v2 import (
    Stage2RC5PlusContextAnchorV2Error,
    build_context_tail_anchor_v2_from_producer_bundle,
)


REQUESTED = ((1, 20_000), (1, 100_000), (1, 250_000))


def _maps():
    values = np.linspace(0.0, 1.0, 14_000, dtype=np.float64)
    return tuple(
        np.ascontiguousarray(chunk.reshape(20, 50), dtype=np.float64)
        for chunk in np.array_split(values, 14)
    )


def _fixture(tmp_path):
    identity = hashlib.sha256(b"producer-v2-context").hexdigest()
    maps = _maps()
    payload = build_context_tail_anchor(
        context_probability_maps=maps,
        context_identity_sha256=identity,
    )
    v1 = verify_context_tail_anchor(
        payload,
        context_probability_maps=maps,
        expected_context_identity_sha256=identity,
    )
    context_bytes = b'{"synthetic":"context"}'
    context = SimpleNamespace(
        payload={
            "window_index": 3,
            "context_full_identity_sha256": identity,
        },
        canonical_payload=context_bytes,
        payload_sha256=hashlib.sha256(context_bytes).hexdigest(),
    )
    bundle = SimpleNamespace(
        producer_manifest={
            "inputs": {
                "statistics_config": {
                    "path": "config/statistics.json",
                    "sha256": hashlib.sha256(b"statistics").hexdigest(),
                }
            }
        },
        score_manifest_metadata=SimpleNamespace(repository_root=tmp_path),
        variable_query_window=object(),
        score_bundle=object(),
        source_reference=object(),
        statistics_config=object(),
        context=context,
        anchor=v1,
    )
    return bundle, context, v1, maps


def test_producer_bundle_builder_replays_same_maps_and_issues_requested_v2(
    tmp_path, monkeypatch
) -> None:
    bundle, context, v1, maps = _fixture(tmp_path)
    observed = {}
    monkeypatch.setattr(
        producer_v2, "replay_verified_stage2_rc5_context_bundle", lambda value: value
    )

    def prepare(**kwargs):
        observed.update(kwargs)
        return "prepared"

    monkeypatch.setattr(producer_v2, "_prepare_inputs", prepare)
    monkeypatch.setattr(
        producer_v2,
        "_produce_context_and_anchor",
        lambda prepared: (context, v1, maps),
    )

    result = build_context_tail_anchor_v2_from_producer_bundle(
        producer_bundle=bundle,
        requested_budget_rationals=REQUESTED,
    )

    assert result.requested_budget_rationals == REQUESTED
    assert len(result.grid_coordinates) == 9
    assert len(result.requested_coordinates) == len(REQUESTED)
    assert result.payload["context_identity_sha256"] == (
        context.payload["context_full_identity_sha256"]
    )
    assert observed["window_index"] == 3
    assert observed["repository_root"] == tmp_path
    assert observed["statistics_config_path"] == (
        tmp_path / "config/statistics.json"
    )


def test_producer_bundle_builder_rejects_context_or_v1_anchor_drift(
    tmp_path, monkeypatch
) -> None:
    bundle, context, v1, maps = _fixture(tmp_path)
    monkeypatch.setattr(
        producer_v2, "replay_verified_stage2_rc5_context_bundle", lambda value: value
    )
    monkeypatch.setattr(producer_v2, "_prepare_inputs", lambda **kwargs: "prepared")
    changed_context = SimpleNamespace(
        payload=context.payload,
        canonical_payload=b'{"changed":"context"}',
        payload_sha256=context.payload_sha256,
    )
    monkeypatch.setattr(
        producer_v2,
        "_produce_context_and_anchor",
        lambda prepared: (changed_context, v1, maps),
    )
    with pytest.raises(Stage2RC5PlusContextAnchorV2Error, match="differs"):
        build_context_tail_anchor_v2_from_producer_bundle(
            producer_bundle=bundle
        )


def test_producer_bundle_builder_rejects_invalid_statistics_binding(
    tmp_path, monkeypatch
) -> None:
    bundle, _, _, _ = _fixture(tmp_path)
    bundle.producer_manifest["inputs"]["statistics_config"]["extra"] = True
    monkeypatch.setattr(
        producer_v2, "replay_verified_stage2_rc5_context_bundle", lambda value: value
    )
    with pytest.raises(Stage2RC5PlusContextAnchorV2Error, match="binding"):
        build_context_tail_anchor_v2_from_producer_bundle(
            producer_bundle=bundle
        )

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from model.budget_conditioned_endpoint_calibrator import (
    BUDGET_KNOT_RATIONALS,
    PRIMARY_BUDGET_KNOT_INDICES,
)
from rc.stage2_context_tail_anchor import (
    build_context_tail_anchor,
    verify_context_tail_anchor,
)
from rc.stage2_context_tail_anchor_v2 import (
    build_context_tail_anchor_v2,
    verify_context_tail_anchor_v2,
)
from rc.stage2_cyclic_training_geometry import build_stage2_cyclic_training_geometry
import rc.stage2_rc5plus_cyclic_anchor_overlay as overlay


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _fake_base(count: int = 43):
    geometry = build_stage2_cyclic_training_geometry(count)
    episodes = tuple(
        {
            "context_indices": row["context_indices"],
            "context_full_identity_sha256": _sha(f"context-{index}"),
            "episode_identity_sha256": _sha(f"episode-{index}"),
        }
        for index, row in enumerate(geometry["episodes"])
    )
    metadata = SimpleNamespace(
        items=tuple(range(count)),
        repository_root=Path("/synthetic/not-opened"),
    )
    return SimpleNamespace(
        score_bundle=SimpleNamespace(score_manifest_metadata=metadata),
        episodes=episodes,
    )


def _map(index: int) -> np.ndarray:
    base = np.linspace(0.01, 0.99, 64, dtype=np.float64).reshape(8, 8)
    return np.mod(base + index * 0.013, 1.0).astype(np.float64)


def test_overlay_recomputes_every_start_from_each_unlabelled_map_once(
    monkeypatch,
) -> None:
    base = _fake_base()
    opened: list[int] = []

    def load(item, root):
        del root
        opened.append(int(item))
        return _map(int(item))

    monkeypatch.setattr(overlay, "_read_context_score_probability", load)

    anchors, rows = overlay._materialize(base)

    assert anchors.shape == (43, len(BUDGET_KNOT_RATIONALS))
    assert anchors.dtype == np.float64
    assert anchors.flags.writeable is False
    assert np.all(anchors[:, 1:] >= anchors[:, :-1])
    assert opened == list(range(43))
    assert len(rows) == 43
    assert all(row["context_labels_accessed"] is False for row in rows)
    assert all(row["query_scores_accessed"] is False for row in rows)
    assert all(row["query_labels_accessed"] is False for row in rows)
    assert all(row["anchor_interpolation_used"] is False for row in rows)
    assert all(
        row["grid_budget_rationals"] == [list(item) for item in BUDGET_KNOT_RATIONALS]
        for row in rows
    )


def test_overlay_primary_knots_replay_frozen_three_budget_anchor_exactly(
    monkeypatch,
) -> None:
    base = _fake_base()
    monkeypatch.setattr(
        overlay,
        "_read_context_score_probability",
        lambda item, root: _map(int(item)),
    )

    anchors, _ = overlay._materialize(base)
    first_indices = tuple(base.episodes[0]["context_indices"])
    probabilities = tuple(_map(index) for index in first_indices)
    identity = base.episodes[0]["context_full_identity_sha256"]
    v1 = verify_context_tail_anchor(
        build_context_tail_anchor(
            context_probability_maps=probabilities,
            context_identity_sha256=identity,
        ),
        context_probability_maps=probabilities,
        expected_context_identity_sha256=identity,
    )
    v2 = verify_context_tail_anchor_v2(
        build_context_tail_anchor_v2(
            context_probability_maps=probabilities,
            context_identity_sha256=identity,
        ),
        context_probability_maps=probabilities,
        expected_context_identity_sha256=identity,
    )

    np.testing.assert_array_equal(
        anchors[0, list(PRIMARY_BUDGET_KNOT_INDICES)],
        np.asarray(v1.coordinates, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        anchors[0], np.asarray(v2.grid_coordinates, dtype=np.float64)
    )

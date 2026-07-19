from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rc.stage2_cyclic_training_collection_v1 import REQUIRED_ROLE
import rc.stage2_rc5plus_cyclic_training_view as view_module
from rc.stage2_rc5plus_cyclic_training_view import (
    Stage2RC5PlusCyclicTrainingViewError,
    build_stage2_rc5plus_cyclic_training_view,
)


class _FakeBase:
    def __init__(self) -> None:
        roles = []
        self.episodes = []
        self.domain_episode_indices = {"A": [], "B": []}
        start = 0
        for domain in ("A", "B"):
            for fold in (0, 1):
                stop = start + 2
                roles.append(
                    {
                        "source_domain": domain,
                        "oof_fold": fold,
                        "episode_start": start,
                        "episode_stop": stop,
                        "upstream_bindings": {
                            "cyclic_context_collection_sha256": f"base-{domain}-{fold}"
                        },
                    }
                )
                for local in range(2):
                    global_index = start + local
                    self.episodes.append(
                        {
                            "global_episode_index": global_index,
                            "source_domain": domain,
                            "domain_episode_index": len(
                                self.domain_episode_indices[domain]
                            ),
                            "ordered_context_image_identity_sha256": [
                                f"context-{domain}-{fold}-{local}"
                            ],
                            "ordered_query_image_identity_sha256": [
                                f"query-{domain}-{fold}-{local}"
                            ],
                        }
                    )
                    self.domain_episode_indices[domain].append(global_index)
                start = stop
        self.episodes = tuple(self.episodes)
        self.domain_episode_indices = {
            key: tuple(value) for key, value in self.domain_episode_indices.items()
        }
        self.manifest = {
            "outer_fold_id": "outer_x",
            "outer_target": "X",
            "episode_count": 8,
            "role_inventory": roles,
        }
        self.arrays = {
            "context_features": np.arange(8 * 93, dtype=np.float32).reshape(8, 93),
            "anchor_coordinates": np.tile(
                np.asarray([0.2, 0.5, 0.8], dtype=np.float64), (8, 1)
            ),
        }
        self.commit_sha256 = "training-commit"
        self.artifact_scope = "synthetic_cpu_contract_test"
        self.boundary_values = {"canonical_id": frozenset({"train"})}
        self.curve_bank_id = "curve-bank"

    def episode_for_domain(self, domain: str, index: int):
        return self.episodes[self.domain_episode_indices[domain][index]]

    def provider_for_episode(self, domain: str, index: int):
        return (domain, index)

    def fit_training_standardizer(self):
        return np.zeros(93), np.ones(93)


def _overlays(base: _FakeBase):
    result = []
    for role_index, role in enumerate(base.manifest["role_inventory"]):
        start = role["episode_start"]
        stop = role["episode_stop"]
        base_context = SimpleNamespace(
            commit_sha256=role["upstream_bindings"][
                "cyclic_context_collection_sha256"
            ],
            context_features=base.arrays["context_features"][start:stop].copy(),
            anchor_coordinates=base.arrays["anchor_coordinates"][start:stop].copy(),
            episodes=tuple(
                {
                    "ordered_context_original_image_sha256": row[
                        "ordered_context_image_identity_sha256"
                    ],
                    "ordered_query_original_image_sha256": row[
                        "ordered_query_image_identity_sha256"
                    ],
                }
                for row in base.episodes[start:stop]
            ),
        )
        anchors = np.tile(
            np.linspace(0.1 + role_index * 0.01, 0.9 + role_index * 0.01, 9),
            (2, 1),
        ).astype(np.float64)
        result.append(
            SimpleNamespace(
                manifest={
                    "outer_fold_id": "outer_x",
                    "outer_target": "X",
                    "source_domain": role["source_domain"],
                    "oof_fold_index": role["oof_fold"],
                    "score_role": REQUIRED_ROLE,
                    "base_cyclic_context": {
                        "sha256": base_context.commit_sha256
                    },
                },
                base_collection=base_context,
                rows=({}, {}),
                anchor_coordinates=anchors,
                commit_sha256=f"overlay-{role_index}",
            )
        )
    return result


@pytest.fixture(autouse=True)
def _capability_stubs(monkeypatch):
    monkeypatch.setattr(
        view_module, "assert_verified_cyclic_training_collection", lambda value: value
    )
    monkeypatch.setattr(
        view_module,
        "assert_verified_stage2_rc5plus_cyclic_anchor_overlay",
        lambda value: value,
    )
    monkeypatch.setattr(
        view_module,
        "replay_verified_stage2_rc5plus_cyclic_anchor_overlay",
        lambda value: value,
    )


def test_four_role_view_assembles_exact_overlay_rows_and_delegates_provider() -> None:
    base = _FakeBase()
    overlays = _overlays(base)

    view = build_stage2_rc5plus_cyclic_training_view(
        base_collection=base,
        anchor_overlays=list(reversed(overlays)),
    )

    assert view.anchor_coordinates.shape == (8, 9)
    assert view.anchor_coordinates.flags.writeable is False
    for role_index, role in enumerate(base.manifest["role_inventory"]):
        start, stop = role["episode_start"], role["episode_stop"]
        np.testing.assert_array_equal(
            view.anchor_coordinates[start:stop], overlays[role_index].anchor_coordinates
        )
    feature, anchor = view.feature_anchor_for_episode("B", 2)
    row = base.episode_for_domain("B", 2)
    global_index = row["global_episode_index"]
    np.testing.assert_array_equal(feature, base.arrays["context_features"][global_index])
    np.testing.assert_array_equal(anchor, view.anchor_coordinates[global_index])
    assert view.provider_for_episode("A", 1) == ("A", 1)
    assert view.curve_bank_id == "curve-bank"
    assert len(view.overlay_commit_by_role) == 4


def test_view_identity_is_independent_of_overlay_argument_order() -> None:
    base = _FakeBase()
    overlays = _overlays(base)

    forward = build_stage2_rc5plus_cyclic_training_view(
        base_collection=base, anchor_overlays=overlays
    )
    reverse = build_stage2_rc5plus_cyclic_training_view(
        base_collection=base, anchor_overlays=list(reversed(overlays))
    )

    assert forward.view_identity_sha256 == reverse.view_identity_sha256


def test_view_rejects_overlay_bound_to_a_different_base_commit() -> None:
    base = _FakeBase()
    overlays = _overlays(base)
    overlays[2].base_collection.commit_sha256 = "wrong-base"

    with pytest.raises(Stage2RC5PlusCyclicTrainingViewError, match="does not bind"):
        build_stage2_rc5plus_cyclic_training_view(
            base_collection=base, anchor_overlays=overlays
        )


def test_view_rejects_feature_or_episode_order_drift() -> None:
    base = _FakeBase()
    feature_drift = _overlays(base)
    feature_drift[0].base_collection.context_features[0, 0] += 1.0
    with pytest.raises(Stage2RC5PlusCyclicTrainingViewError, match="features differ"):
        build_stage2_rc5plus_cyclic_training_view(
            base_collection=base, anchor_overlays=feature_drift
        )

    order_drift = _overlays(base)
    order_drift[0].base_collection.episodes[0][
        "ordered_query_original_image_sha256"
    ] = ["wrong-query"]
    with pytest.raises(Stage2RC5PlusCyclicTrainingViewError, match="episode 0 order"):
        build_stage2_rc5plus_cyclic_training_view(
            base_collection=base, anchor_overlays=order_drift
        )

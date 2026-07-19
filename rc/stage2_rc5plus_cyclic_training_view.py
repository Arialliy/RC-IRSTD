"""Verifier-issued RC5+ view over exact curves and nine-budget anchors.

The v1 cyclic training collection remains immutable and keeps its three-point
anchors.  This view binds exactly four commit-last RC5+ anchor overlays to the
matching source-domain/OOF role ranges, while delegating Q28 exact-event curve
providers and 93D features to the already verified v1 collection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Any

import numpy as np

from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from rc.stage2_cyclic_training_collection_v1 import (
    REQUIRED_ROLE,
    VerifiedCyclicTrainingCollection,
    assert_verified_cyclic_training_collection,
    canonical_json_bytes,
)
from rc.stage2_rc5plus_cyclic_anchor_overlay import (
    VerifiedStage2RC5PlusCyclicAnchorOverlay,
    assert_verified_stage2_rc5plus_cyclic_anchor_overlay,
    replay_verified_stage2_rc5plus_cyclic_anchor_overlay,
)


RC5PLUS_TRAINING_VIEW_SCHEMA = "rc-irstd.stage2-rc5plus-cyclic-training-view.v1"
_CAPABILITY = object()


class Stage2RC5PlusCyclicTrainingViewError(ValueError):
    """Training collection and anchor overlays do not form one causal view."""


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5PlusCyclicTrainingView:
    base_collection: VerifiedCyclicTrainingCollection
    anchor_overlays: tuple[VerifiedStage2RC5PlusCyclicAnchorOverlay, ...]
    anchor_coordinates: np.ndarray
    overlay_commit_by_role: Mapping[tuple[str, int], str]
    view_identity_sha256: str
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("RC5+ cyclic training views are verifier-issued only")

    @property
    def manifest(self) -> Mapping[str, Any]:
        assert_verified_stage2_rc5plus_cyclic_training_view(self)
        return self.base_collection.manifest

    @property
    def artifact_scope(self) -> str:
        assert_verified_stage2_rc5plus_cyclic_training_view(self)
        return self.base_collection.artifact_scope

    @property
    def boundary_values(self) -> Mapping[str, frozenset[str]]:
        assert_verified_stage2_rc5plus_cyclic_training_view(self)
        return self.base_collection.boundary_values

    @property
    def domain_episode_indices(self) -> Mapping[str, tuple[int, ...]]:
        assert_verified_stage2_rc5plus_cyclic_training_view(self)
        return self.base_collection.domain_episode_indices

    @property
    def curve_bank_id(self) -> str:
        assert_verified_stage2_rc5plus_cyclic_training_view(self)
        return self.base_collection.curve_bank_id

    def episode_for_domain(
        self, source_domain: str, domain_episode_index: int
    ) -> Mapping[str, Any]:
        assert_verified_stage2_rc5plus_cyclic_training_view(self)
        return self.base_collection.episode_for_domain(
            source_domain, domain_episode_index
        )

    def feature_anchor_for_episode(
        self, source_domain: str, domain_episode_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        row = self.episode_for_domain(source_domain, domain_episode_index)
        index = int(row["global_episode_index"])
        return (
            self.base_collection.arrays["context_features"][index],
            self.anchor_coordinates[index],
        )

    def feature_for_episode(
        self, source_domain: str, domain_episode_index: int
    ) -> np.ndarray:
        """Resolve context features without touching the anchor array."""

        row = self.episode_for_domain(source_domain, domain_episode_index)
        index = int(row["global_episode_index"])
        return self.base_collection.arrays["context_features"][index]

    def provider_for_episode(self, source_domain: str, domain_episode_index: int):
        assert_verified_stage2_rc5plus_cyclic_training_view(self)
        return self.base_collection.provider_for_episode(
            source_domain, domain_episode_index
        )

    def fit_training_standardizer(self) -> tuple[np.ndarray, np.ndarray]:
        assert_verified_stage2_rc5plus_cyclic_training_view(self)
        return self.base_collection.fit_training_standardizer()


def assert_verified_stage2_rc5plus_cyclic_training_view(
    value: object,
) -> VerifiedStage2RC5PlusCyclicTrainingView:
    if (
        type(value) is not VerifiedStage2RC5PlusCyclicTrainingView
        or getattr(value, "_capability", None) is not _CAPABILITY
    ):
        raise TypeError("a verifier-issued RC5+ cyclic training view is required")
    assert_verified_cyclic_training_collection(value.base_collection)
    if len(value.anchor_overlays) != 4:
        raise TypeError("RC5+ cyclic training view must retain four overlays")
    for item in value.anchor_overlays:
        assert_verified_stage2_rc5plus_cyclic_anchor_overlay(item)
    expected = (
        int(value.base_collection.manifest["episode_count"]),
        len(BUDGET_KNOT_RATIONALS),
    )
    if value.anchor_coordinates.dtype != np.float64 or value.anchor_coordinates.shape != expected:
        raise TypeError("RC5+ cyclic training view anchor matrix is invalid")
    return value


def build_stage2_rc5plus_cyclic_training_view(
    *,
    base_collection: VerifiedCyclicTrainingCollection,
    anchor_overlays: Sequence[VerifiedStage2RC5PlusCyclicAnchorOverlay],
) -> VerifiedStage2RC5PlusCyclicTrainingView:
    """Bind four overlay commits to their exact role ranges and episode order."""

    base = assert_verified_cyclic_training_collection(base_collection)
    if (
        isinstance(anchor_overlays, (str, bytes))
        or not isinstance(anchor_overlays, Sequence)
        or len(anchor_overlays) != 4
    ):
        raise Stage2RC5PlusCyclicTrainingViewError(
            "exactly four RC5+ anchor overlays are required"
        )
    overlays = tuple(
        replay_verified_stage2_rc5plus_cyclic_anchor_overlay(
            assert_verified_stage2_rc5plus_cyclic_anchor_overlay(item)
        )
        for item in anchor_overlays
    )
    role_overlays: dict[tuple[str, int], Any] = {}
    for item in overlays:
        manifest = item.manifest
        if manifest["outer_fold_id"] != base.manifest["outer_fold_id"] or manifest[
            "outer_target"
        ] != base.manifest["outer_target"]:
            raise Stage2RC5PlusCyclicTrainingViewError(
                "overlay outer fold/target differs from training collection"
            )
        if manifest["score_role"] != REQUIRED_ROLE:
            raise Stage2RC5PlusCyclicTrainingViewError(
                "RC5+ training overlay is not an OOF holdout fit role"
            )
        key = (str(manifest["source_domain"]), int(manifest["oof_fold_index"]))
        if key in role_overlays:
            raise Stage2RC5PlusCyclicTrainingViewError(
                "duplicate source-domain/OOF anchor overlay"
            )
        role_overlays[key] = item

    inventory = base.manifest["role_inventory"]
    expected_roles = {
        (str(row["source_domain"]), int(row["oof_fold"])) for row in inventory
    }
    if set(role_overlays) != expected_roles:
        raise Stage2RC5PlusCyclicTrainingViewError(
            "anchor overlay source-domain/OOF coverage is incomplete"
        )
    count = int(base.manifest["episode_count"])
    anchors = np.empty((count, len(BUDGET_KNOT_RATIONALS)), dtype=np.float64)
    overlay_rows: list[dict[str, Any]] = []
    for role in inventory:
        key = (str(role["source_domain"]), int(role["oof_fold"]))
        item = role_overlays[key]
        start = int(role["episode_start"])
        stop = int(role["episode_stop"])
        if stop - start != len(item.rows) or item.anchor_coordinates.shape != (
            stop - start,
            len(BUDGET_KNOT_RATIONALS),
        ):
            raise Stage2RC5PlusCyclicTrainingViewError(
                "overlay row count differs from its role range"
            )
        expected_base_commit = str(
            role["upstream_bindings"]["cyclic_context_collection_sha256"]
        )
        if item.base_collection.commit_sha256 != expected_base_commit or item.manifest[
            "base_cyclic_context"
        ]["sha256"] != expected_base_commit:
            raise Stage2RC5PlusCyclicTrainingViewError(
                "overlay does not bind the cyclic context used by its role"
            )
        if not np.array_equal(
            np.asarray(base.arrays["context_features"][start:stop]),
            np.asarray(item.base_collection.context_features),
        ):
            raise Stage2RC5PlusCyclicTrainingViewError(
                "training features differ from overlay base cyclic context"
            )
        if not np.array_equal(
            np.asarray(base.arrays["anchor_coordinates"][start:stop]),
            np.asarray(item.base_collection.anchor_coordinates),
        ):
            raise Stage2RC5PlusCyclicTrainingViewError(
                "frozen three-point anchors differ from overlay base binding"
            )
        for local, (training_episode, base_episode) in enumerate(
            zip(base.episodes[start:stop], item.base_collection.episodes, strict=True)
        ):
            if tuple(training_episode["ordered_context_image_identity_sha256"]) != tuple(
                base_episode["ordered_context_original_image_sha256"]
            ) or tuple(training_episode["ordered_query_image_identity_sha256"]) != tuple(
                base_episode["ordered_query_original_image_sha256"]
            ):
                raise Stage2RC5PlusCyclicTrainingViewError(
                    f"role {key} episode {local} order differs from overlay base"
                )
        anchors[start:stop] = item.anchor_coordinates
        overlay_rows.append(
            {
                "source_domain": key[0],
                "oof_fold": key[1],
                "episode_start": start,
                "episode_stop": stop,
                "overlay_commit_sha256": item.commit_sha256,
                "base_cyclic_context_commit_sha256": expected_base_commit,
            }
        )
    if not np.isfinite(anchors).all() or np.any(anchors[:, 1:] < anchors[:, :-1]):
        raise Stage2RC5PlusCyclicTrainingViewError(
            "assembled RC5+ anchor matrix is non-finite or decreasing"
        )
    anchors.setflags(write=False)
    overlay_rows.sort(key=lambda row: (row["episode_start"], row["source_domain"]))
    view_identity = _identity(
        {
            "schema_version": RC5PLUS_TRAINING_VIEW_SCHEMA,
            "base_training_collection_commit_sha256": base.commit_sha256,
            "grid_budget_rationals": [list(row) for row in BUDGET_KNOT_RATIONALS],
            "role_overlays": overlay_rows,
            "anchor_interpolation_used": False,
        }
    )
    value = object.__new__(VerifiedStage2RC5PlusCyclicTrainingView)
    for name, item in {
        "base_collection": base,
        "anchor_overlays": overlays,
        "anchor_coordinates": anchors,
        "overlay_commit_by_role": MappingProxyType(
            {key: item.commit_sha256 for key, item in role_overlays.items()}
        ),
        "view_identity_sha256": view_identity,
        "_capability": _CAPABILITY,
    }.items():
        object.__setattr__(value, name, item)
    return assert_verified_stage2_rc5plus_cyclic_training_view(value)


__all__ = [
    "RC5PLUS_TRAINING_VIEW_SCHEMA",
    "Stage2RC5PlusCyclicTrainingViewError",
    "VerifiedStage2RC5PlusCyclicTrainingView",
    "assert_verified_stage2_rc5plus_cyclic_training_view",
    "build_stage2_rc5plus_cyclic_training_view",
]

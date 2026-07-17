"""Exact, fail-closed Stage-2 replay over verified v5 episodes.

Public replay accepts one verifier-created T0--T8 decision member.  Arbitrary
threshold vectors are intentionally confined to the trainer-private function
at the bottom of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from data_ext.stage2_label_attachment import (
    VerifiedStage2LabelAttachment,
    canonical_json_sha256 as w05_canonical_json_sha256,
    load_stage2_label_mask,
    stage2_ordered_query_identity,
)
from data_ext.stage2_threshold_decision import (
    COMPLETE_OUTCOME,
    PIXEL_BUDGET_GRID,
    PRELABEL_METHOD_ORDER,
    STRICT_THRESHOLD_SEMANTICS,
    VerifiedStage2ThresholdDecision,
)
from evaluation.component_matching import match_components, prepare_target
from rc.stage2_crossfit_dataset import (
    Stage2TrainerReplayCapability,
    assert_stage2_trainer_replay_capability,
)
from rc.stage2_crossfit_schema import (
    COLLECTION_OUTER,
    COLLECTION_VALIDATION,
    Stage2CrossfitContractError,
    Stage2CrossfitEpisode,
    VerifiedEpisodeArtifacts,
    VerifiedStage2EpisodeCollection,
    assert_verified_episode_collection,
    bootstrap_query_identity_projection,
    bootstrap_query_identity_sha256,
    bootstrap_window_identity_sha256,
    canonical_json_bytes,
    canonical_json_sha256,
    full_identity_sha256,
    parse_json_bytes,
    stable_read,
)


IMAGE_COUNTS_SCHEMA = "rc-irstd.stage2-image-sufficient-counts.v1"
PRIMARY_BUDGET = 1e-5
THRESHOLD_SEMANTICS = "prediction = probability > threshold"
IDENTITY_BRIDGE_SCHEMA = "rc-irstd.stage2-four-identity-bridge.v1"


@dataclass(frozen=True)
class Stage2ExactReplayResult:
    method_id: str
    decision_sha256: str
    episode_id: str
    identity_bridge: Mapping[str, Any]
    budget_summaries: tuple[Mapping[str, Any], ...]
    primary_sufficient_counts: Mapping[str, Any]


def _decision_member(value: object) -> VerifiedStage2ThresholdDecision:
    # VerifiedStage2ThresholdDecision has a token-gated constructor and is only
    # returned by the public complete-set verifier.
    if not isinstance(value, VerifiedStage2ThresholdDecision):
        raise TypeError("public replay requires a verified T0--T8 decision member")
    payload = value.payload
    if payload["method_id"] not in PRELABEL_METHOD_ORDER:
        raise Stage2CrossfitContractError("T9/non-prelabel decisions are forbidden")
    if payload["outcome"] != COMPLETE_OUTCOME or payload["thresholds"] is None:
        raise Stage2CrossfitContractError("a complete three-threshold decision is required")
    if payload["prediction_semantics"] != STRICT_THRESHOLD_SEMANTICS:
        raise Stage2CrossfitContractError("decision threshold semantics mismatch")
    # Re-read the immutable member under its externally verified digest.
    data = stable_read(value.path, value.manifest_sha256, "threshold decision member")
    if parse_json_bytes(data, "threshold decision member") != dict(payload):
        raise Stage2CrossfitContractError("verified decision object was mutated")
    return value


def _source_window_identity(artifact: VerifiedEpisodeArtifacts) -> str:
    selected = artifact.context.window.window
    if not isinstance(selected, Mapping):
        raise Stage2CrossfitContractError("verified context lacks its selected window")
    return w05_canonical_json_sha256(selected)


def _source_query_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    projection = [
        {
            key: row[key]
            for key in (
                "canonical_id",
                "image_id",
                "original_image_sha256",
                "exclusion_group_id",
                "near_duplicate_cluster_id_or_unique_sentinel",
                "source_role_record_index",
            )
        }
        for row in rows
    ]
    return w05_canonical_json_sha256(stage2_ordered_query_identity(projection))


def recompute_stage2_identity_bridge(
    episode: Stage2CrossfitEpisode,
    artifact: VerifiedEpisodeArtifacts,
) -> Mapping[str, Any]:
    """Recompute all four W05/W11 hashes from original verified rows."""

    payload = episode.payload
    context_rows = payload["context_records"]
    query_rows = payload["query_records"]
    source_window = _source_window_identity(artifact)
    source_query = _source_query_identity(query_rows)
    context_identity = full_identity_sha256(context_rows)
    bootstrap_query = bootstrap_query_identity_sha256(query_rows)
    bootstrap_window = bootstrap_window_identity_sha256(
        payload["window_binding"]["window_id"],
        context_identity,
        bootstrap_query,
    )
    if source_window != payload["window_binding"]["window_identity_sha256"]:
        raise Stage2CrossfitContractError("source window identity bridge mismatch")
    if source_query != payload["source_ordered_query_identity_sha256"]:
        raise Stage2CrossfitContractError("source query identity bridge mismatch")
    if context_identity != payload["context_full_identity_sha256"]:
        raise Stage2CrossfitContractError("context identity bridge mismatch")
    result = {
        "schema_version": IDENTITY_BRIDGE_SCHEMA,
        "source_window_identity_sha256": source_window,
        "source_ordered_query_identity_sha256": source_query,
        "bootstrap_ordered_query_identity_sha256": bootstrap_query,
        "bootstrap_window_identity_sha256": bootstrap_window,
        "bootstrap_context_identity_sha256": context_identity,
        "bootstrap_query_identity_projection": bootstrap_query_identity_projection(
            query_rows
        ),
    }
    return MappingProxyType(result)


def _match_episode(
    collection: VerifiedStage2EpisodeCollection,
    decision: VerifiedStage2ThresholdDecision,
) -> int:
    decision_payload = decision.payload
    shared = decision_payload["shared_bindings"]
    matches: list[int] = []
    for index, episode in enumerate(collection.episodes):
        payload = episode.payload
        if (
            payload["outer_fold_id"] == decision_payload["outer_fold_id"]
            and payload["outer_target"] == decision_payload["outer_target_domain"]
            and payload["base_seed"] == decision_payload["base_seed"]
            and payload["derived_seed"] == decision_payload["derived_seed"]
            and payload["window_binding"]["window_id"] == shared["window_id"]
        ):
            matches.append(index)
    if len(matches) != 1:
        raise Stage2CrossfitContractError(
            "decision must bind exactly one verified outer episode"
        )
    return matches[0]


def _validate_decision_binding(
    episode: Stage2CrossfitEpisode,
    artifact: VerifiedEpisodeArtifacts,
    decision: VerifiedStage2ThresholdDecision,
    bridge: Mapping[str, Any],
) -> None:
    payload = episode.payload
    shared = decision.payload["shared_bindings"]
    expected = {
        "window_id": payload["window_binding"]["window_id"],
        "window_identity_sha256": bridge["source_window_identity_sha256"],
        "ordered_query_identity_sha256": bridge[
            "source_ordered_query_identity_sha256"
        ],
        "score_manifest_sha256": payload["score_manifest_binding"]["sha256"],
        "score_records_content_sha256": payload["score_manifest_binding"][
            "records_content_sha256"
        ],
        "detector_checkpoint_sha256": payload["detector_identity"][
            "checkpoint_sha256"
        ],
    }
    for field, value in expected.items():
        if shared[field] != value:
            raise Stage2CrossfitContractError(
                f"decision/episode shared binding mismatch: {field}"
            )
    context_binding = payload["context_package_binding"]
    for shared_field, episode_field in (
        ("context_package", "path"),
        ("context_package_commit", "commit_path"),
    ):
        binding = shared[shared_field]
        digest_field = "sha256" if episode_field == "path" else "commit_sha256"
        if (
            binding["path"] != context_binding[episode_field]
            or binding["sha256"] != context_binding[digest_field]
        ):
            raise Stage2CrossfitContractError(
                f"decision/episode {shared_field} mismatch"
            )
    if artifact.attachment.ordered_query_identity_sha256 != bridge[
        "source_ordered_query_identity_sha256"
    ]:
        raise Stage2CrossfitContractError("label attachment query identity mismatch")


def _load_bound_probability(item: Any) -> np.ndarray:
    digest = str(item.score_item.record["score_file_sha256"])
    data = stable_read(item.score_item.score_path, digest, "query probability map")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        probability = np.asarray(archive["prob"])
    if (
        probability.dtype != np.float64
        or probability.shape != item.original_hw
        or not np.isfinite(probability).all()
        or np.any((probability < 0.0) | (probability > 1.0))
    ):
        raise Stage2CrossfitContractError(
            "query probability violates exact native float64 contract"
        )
    return probability


def _replay_thresholds(
    episode: Stage2CrossfitEpisode,
    attachment: VerifiedStage2LabelAttachment,
    thresholds: Sequence[float],
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    values = tuple(float(value) for value in thresholds)
    if len(values) != 3 or any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values
    ):
        raise Stage2CrossfitContractError("exact replay requires three thresholds")
    if len(attachment.items) != 28:
        raise Stage2CrossfitContractError("exact replay requires Q28")
    query_rows = episode.payload["query_records"]
    prepared: list[tuple[Mapping[str, Any], np.ndarray, Any, int]] = []
    for index, (row, item) in enumerate(
        zip(query_rows, attachment.items, strict=True)
    ):
        if (
            row["image_id"] != item.image_id
            or row["canonical_id"] != item.canonical_id
            or row["original_image_sha256"]
            != item.score_item.record["original_image_sha256"]
        ):
            raise Stage2CrossfitContractError(
                f"query artifact identity mismatch at index {index}"
            )
        probability = _load_bound_probability(item)
        target = load_stage2_label_mask(item)
        if target.shape != item.original_hw:
            raise Stage2CrossfitContractError("query mask geometry mismatch")
        prepared.append(
            (
                row,
                probability,
                prepare_target(target),
                int(target.size - np.count_nonzero(target)),
            )
        )
    outputs: list[tuple[Mapping[str, Any], ...]] = []
    for threshold in values:
        rows: list[Mapping[str, Any]] = []
        for row, probability, target, background in prepared:
            result = match_components(
                probability > threshold, target, rule="overlap"
            )
            rows.append(
                MappingProxyType(
                    {
                        "image_id": row["image_id"],
                        "original_image_sha256": row[
                            "original_image_sha256"
                        ],
                        "false_positive_pixels": result.num_fp_pixels,
                        "total_pixels": result.total_pixels,
                        "background_pixels": background,
                        "matched_targets": result.num_tp_objects,
                        "ground_truth_targets": result.num_gt,
                    }
                )
            )
        outputs.append(tuple(rows))
    return tuple(outputs)


def _summary(
    budget: float,
    threshold: float,
    counts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    fp = sum(int(row["false_positive_pixels"]) for row in counts)
    pixels = sum(int(row["total_pixels"]) for row in counts)
    tp = sum(int(row["matched_targets"]) for row in counts)
    gt = sum(int(row["ground_truth_targets"]) for row in counts)
    return MappingProxyType(
        {
            "pixel_budget": float(budget),
            "threshold": float(threshold),
            "fp_pixels": fp,
            "total_pixels": pixels,
            "fa_pixel": float(fp / pixels),
            "tp_objects": tp,
            "gt_objects": gt,
            "pd": float(tp / gt) if gt else 0.0,
        }
    )


class Stage2ExactGroupedPixelRiskReplay:
    """Public exact replay for one complete outer-target v5 collection."""

    def __init__(self, collection: VerifiedStage2EpisodeCollection) -> None:
        collection = assert_verified_episode_collection(collection)
        if collection.manifest["collection_role"] != COLLECTION_OUTER:
            raise Stage2CrossfitContractError(
                "public decision replay requires an outer-target collection"
            )
        self._collection = collection

    def evaluate(
        self, decision: VerifiedStage2ThresholdDecision
    ) -> Stage2ExactReplayResult:
        verified_decision = _decision_member(decision)
        index = _match_episode(self._collection, verified_decision)
        episode = self._collection.episodes[index]
        artifact = self._collection.artifacts[index]
        bridge = recompute_stage2_identity_bridge(episode, artifact)
        _validate_decision_binding(
            episode, artifact, verified_decision, bridge
        )
        thresholds = tuple(float(value) for value in verified_decision.payload["thresholds"])
        count_sets = _replay_thresholds(
            episode, artifact.attachment, thresholds
        )
        summaries = tuple(
            _summary(budget, threshold, counts)
            for budget, threshold, counts in zip(
                PIXEL_BUDGET_GRID, thresholds, count_sets, strict=True
            )
        )
        primary_index = tuple(PIXEL_BUDGET_GRID).index(PRIMARY_BUDGET)
        payload = verified_decision.payload
        method_binding = payload["method_binding"]
        method_checkpoint_sha256 = (
            method_binding["sha256"]
            if method_binding is not None
            else payload["method_contract_sha256"]
        )
        primary = MappingProxyType(
            {
                "schema_version": IMAGE_COUNTS_SCHEMA,
                "method_id": payload["method_id"],
                "detector_checkpoint_sha256": payload["shared_bindings"][
                    "detector_checkpoint_sha256"
                ],
                "method_checkpoint_sha256": method_checkpoint_sha256,
                "windows": [
                    {
                        "window_id": episode.payload["window_binding"][
                            "window_id"
                        ],
                        "window_identity_sha256": bridge[
                            "bootstrap_window_identity_sha256"
                        ],
                        "context_identity_sha256": bridge[
                            "bootstrap_context_identity_sha256"
                        ],
                        "ordered_query_identity_sha256": bridge[
                            "bootstrap_ordered_query_identity_sha256"
                        ],
                        "decision_sha256": verified_decision.manifest_sha256,
                        "decision_sealed": True,
                        "threshold": thresholds[primary_index],
                        "threshold_semantics": THRESHOLD_SEMANTICS,
                        "online_update_count": 0,
                        "threshold_reselected": False,
                        "query_counts": [
                            dict(row) for row in count_sets[primary_index]
                        ],
                    }
                ],
            }
        )
        return Stage2ExactReplayResult(
            method_id=payload["method_id"],
            decision_sha256=verified_decision.manifest_sha256,
            episode_id=episode.episode_id,
            identity_bridge=bridge,
            budget_summaries=summaries,
            primary_sufficient_counts=primary,
        )


def _evaluate_private_thresholds(
    validation: VerifiedStage2EpisodeCollection,
    episode_index: int,
    thresholds: Sequence[float],
    capability: Stage2TrainerReplayCapability,
) -> tuple[Mapping[str, Any], ...]:
    """Trainer-private exact validation replay; deliberately not exported."""

    validation = assert_verified_episode_collection(validation)
    assert_stage2_trainer_replay_capability(capability, validation)
    if validation.manifest["collection_role"] != COLLECTION_VALIDATION:
        raise Stage2CrossfitContractError(
            "private threshold replay is validation-only"
        )
    if isinstance(episode_index, bool) or not 0 <= episode_index < len(validation):
        raise IndexError("validation episode index out of range")
    episode = validation.episodes[episode_index]
    artifact = validation.artifacts[episode_index]
    count_sets = _replay_thresholds(episode, artifact.attachment, thresholds)
    return tuple(
        _summary(budget, threshold, counts)
        for budget, threshold, counts in zip(
            PIXEL_BUDGET_GRID, thresholds, count_sets, strict=True
        )
    )


__all__ = [
    "IDENTITY_BRIDGE_SCHEMA",
    "IMAGE_COUNTS_SCHEMA",
    "PRIMARY_BUDGET",
    "Stage2ExactGroupedPixelRiskReplay",
    "Stage2ExactReplayResult",
    "recompute_stage2_identity_bridge",
]

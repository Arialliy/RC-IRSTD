"""Deterministic source-only cyclic episode geometry for RC5 training.

The mandatory variable-query geometry remains the sole geometry for source
validation and outer-target evaluation.  It intentionally yields only a few
independent windows per role, however, which is too sparse to fit the frozen
3.1k-parameter Stage-2 calibrator reliably.  This module defines a separate
*training-only* augmentation over each verified source OOF role.

For an ordered role containing ``N >= 42`` records, every cyclic start
``s in [0, N)`` yields one C14/Q28 episode.  Indices wrap modulo ``N``.  Across
the complete set, every record appears exactly 14 times as context and 28
times as query; within an episode the two partitions are disjoint.  No target
role, validation role, label value, score value, result, or Python hash enters
the construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "rc-irstd.stage2-source-cyclic-training-geometry.v1"
CONTEXT_SIZE = 14
QUERY_SIZE = 28
MINIMUM_ROLE_RECORDS = CONTEXT_SIZE + QUERY_SIZE
EPISODE_COUNT_RULE = "ordered_role_record_count"
INDEX_RULE = "(cyclic_start + local_offset) % ordered_role_record_count"
ROLE_SCOPE = "source_oof_training_only"


class Stage2CyclicTrainingGeometryError(ValueError):
    """A source-only cyclic training geometry failed its exact contract."""


def _exact_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Stage2CyclicTrainingGeometryError(
            f"{name} must be an exact int >= {minimum}"
        )
    return value


def build_stage2_cyclic_training_geometry(
    ordered_role_record_count: int,
) -> dict[str, Any]:
    """Build the complete balanced C14/Q28 cyclic training geometry."""

    count = _exact_int(
        ordered_role_record_count,
        "ordered_role_record_count",
        minimum=MINIMUM_ROLE_RECORDS,
    )
    episodes: list[dict[str, Any]] = []
    context_frequency = [0] * count
    query_frequency = [0] * count
    for start in range(count):
        context = tuple((start + offset) % count for offset in range(CONTEXT_SIZE))
        query = tuple(
            (start + CONTEXT_SIZE + offset) % count
            for offset in range(QUERY_SIZE)
        )
        if len(set(context)) != CONTEXT_SIZE or len(set(query)) != QUERY_SIZE:
            raise RuntimeError("cyclic episode contains a duplicate partition index")
        if set(context) & set(query):
            raise RuntimeError("cyclic context/query partitions overlap")
        for index in context:
            context_frequency[index] += 1
        for index in query:
            query_frequency[index] += 1
        episodes.append(
            {
                "episode_index": start,
                "cyclic_start": start,
                "context_indices": list(context),
                "query_indices": list(query),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "role_scope": ROLE_SCOPE,
        "ordered_role_record_count": count,
        "context_size": CONTEXT_SIZE,
        "query_size": QUERY_SIZE,
        "episode_count": count,
        "episode_count_rule": EPISODE_COUNT_RULE,
        "index_rule": INDEX_RULE,
        "within_episode_context_query_disjoint": True,
        "context_frequency_per_record": CONTEXT_SIZE,
        "query_frequency_per_record": QUERY_SIZE,
        "episodes": episodes,
    }
    return validate_stage2_cyclic_training_geometry(payload)


def validate_stage2_cyclic_training_geometry(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the construction exactly and return a canonical plain payload."""

    if not isinstance(payload, Mapping):
        raise TypeError("cyclic training geometry must be a mapping")
    expected_fields = {
        "schema_version",
        "role_scope",
        "ordered_role_record_count",
        "context_size",
        "query_size",
        "episode_count",
        "episode_count_rule",
        "index_rule",
        "within_episode_context_query_disjoint",
        "context_frequency_per_record",
        "query_frequency_per_record",
        "episodes",
    }
    if set(payload) != expected_fields:
        raise Stage2CyclicTrainingGeometryError(
            "cyclic training geometry field closure mismatch"
        )
    count = _exact_int(
        payload["ordered_role_record_count"],
        "ordered_role_record_count",
        minimum=MINIMUM_ROLE_RECORDS,
    )
    exact = {
        "schema_version": SCHEMA_VERSION,
        "role_scope": ROLE_SCOPE,
        "context_size": CONTEXT_SIZE,
        "query_size": QUERY_SIZE,
        "episode_count": count,
        "episode_count_rule": EPISODE_COUNT_RULE,
        "index_rule": INDEX_RULE,
        "within_episode_context_query_disjoint": True,
        "context_frequency_per_record": CONTEXT_SIZE,
        "query_frequency_per_record": QUERY_SIZE,
    }
    for field, value in exact.items():
        if type(payload[field]) is not type(value) or payload[field] != value:
            raise Stage2CyclicTrainingGeometryError(
                f"cyclic training geometry {field} mismatch"
            )
    episodes = payload["episodes"]
    if not isinstance(episodes, list) or len(episodes) != count:
        raise Stage2CyclicTrainingGeometryError(
            "cyclic training geometry episode cardinality mismatch"
        )
    canonical_episodes: list[dict[str, Any]] = []
    context_frequency = [0] * count
    query_frequency = [0] * count
    for start, raw in enumerate(episodes):
        if not isinstance(raw, Mapping) or set(raw) != {
            "episode_index",
            "cyclic_start",
            "context_indices",
            "query_indices",
        }:
            raise Stage2CyclicTrainingGeometryError(
                f"episodes[{start}] field closure mismatch"
            )
        if raw["episode_index"] != start or type(raw["episode_index"]) is not int:
            raise Stage2CyclicTrainingGeometryError(
                f"episodes[{start}].episode_index mismatch"
            )
        if raw["cyclic_start"] != start or type(raw["cyclic_start"]) is not int:
            raise Stage2CyclicTrainingGeometryError(
                f"episodes[{start}].cyclic_start mismatch"
            )
        expected_context = [
            (start + offset) % count for offset in range(CONTEXT_SIZE)
        ]
        expected_query = [
            (start + CONTEXT_SIZE + offset) % count
            for offset in range(QUERY_SIZE)
        ]
        if raw["context_indices"] != expected_context:
            raise Stage2CyclicTrainingGeometryError(
                f"episodes[{start}].context_indices mismatch"
            )
        if raw["query_indices"] != expected_query:
            raise Stage2CyclicTrainingGeometryError(
                f"episodes[{start}].query_indices mismatch"
            )
        if set(expected_context) & set(expected_query):
            raise Stage2CyclicTrainingGeometryError(
                f"episodes[{start}] context/query overlap"
            )
        for index in expected_context:
            context_frequency[index] += 1
        for index in expected_query:
            query_frequency[index] += 1
        canonical_episodes.append(
            {
                "episode_index": start,
                "cyclic_start": start,
                "context_indices": expected_context,
                "query_indices": expected_query,
            }
        )
    if context_frequency != [CONTEXT_SIZE] * count:
        raise Stage2CyclicTrainingGeometryError(
            "cyclic training context frequency is not exactly balanced"
        )
    if query_frequency != [QUERY_SIZE] * count:
        raise Stage2CyclicTrainingGeometryError(
            "cyclic training query frequency is not exactly balanced"
        )
    return {
        **exact,
        "ordered_role_record_count": count,
        "episodes": canonical_episodes,
    }


__all__ = [
    "CONTEXT_SIZE",
    "EPISODE_COUNT_RULE",
    "INDEX_RULE",
    "MINIMUM_ROLE_RECORDS",
    "QUERY_SIZE",
    "ROLE_SCOPE",
    "SCHEMA_VERSION",
    "Stage2CyclicTrainingGeometryError",
    "build_stage2_cyclic_training_geometry",
    "validate_stage2_cyclic_training_geometry",
]

from __future__ import annotations

import copy

import pytest

from rc.stage2_cyclic_training_geometry import (
    CONTEXT_SIZE,
    QUERY_SIZE,
    ROLE_SCOPE,
    SCHEMA_VERSION,
    Stage2CyclicTrainingGeometryError,
    build_stage2_cyclic_training_geometry,
    validate_stage2_cyclic_training_geometry,
)


@pytest.mark.parametrize("count", [42, 43, 85, 127, 159, 254, 255, 319])
def test_complete_cyclic_geometry_is_exactly_balanced(count: int) -> None:
    payload = build_stage2_cyclic_training_geometry(count)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["role_scope"] == ROLE_SCOPE
    assert payload["episode_count"] == count
    assert len(payload["episodes"]) == count

    context_frequency = [0] * count
    query_frequency = [0] * count
    for start, episode in enumerate(payload["episodes"]):
        context = episode["context_indices"]
        query = episode["query_indices"]
        assert episode["episode_index"] == start
        assert episode["cyclic_start"] == start
        assert len(context) == CONTEXT_SIZE
        assert len(query) == QUERY_SIZE
        assert len(set(context)) == CONTEXT_SIZE
        assert len(set(query)) == QUERY_SIZE
        assert set(context).isdisjoint(query)
        for index in context:
            context_frequency[index] += 1
        for index in query:
            query_frequency[index] += 1

    assert context_frequency == [CONTEXT_SIZE] * count
    assert query_frequency == [QUERY_SIZE] * count
    assert validate_stage2_cyclic_training_geometry(payload) == payload


def test_wraparound_episode_is_frozen_and_ordered() -> None:
    payload = build_stage2_cyclic_training_geometry(43)
    episode = payload["episodes"][-1]

    assert episode["context_indices"] == [42, *range(13)]
    assert episode["query_indices"] == list(range(13, 41))


@pytest.mark.parametrize("value", [True, 0, 41])
def test_invalid_record_counts_fail_closed(value: object) -> None:
    with pytest.raises(Stage2CyclicTrainingGeometryError):
        build_stage2_cyclic_training_geometry(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(extra=True), "field closure"),
        (
            lambda value: value["episodes"][0]["query_indices"].__setitem__(0, 0),
            "query_indices mismatch",
        ),
        (
            lambda value: value.__setitem__("context_frequency_per_record", 13),
            "context_frequency_per_record mismatch",
        ),
        (
            lambda value: value.__setitem__("within_episode_context_query_disjoint", 1),
            "within_episode_context_query_disjoint mismatch",
        ),
    ],
)
def test_geometry_tampering_is_rejected(mutation, message: str) -> None:
    payload = copy.deepcopy(build_stage2_cyclic_training_geometry(43))
    mutation(payload)
    with pytest.raises(Stage2CyclicTrainingGeometryError, match=message):
        validate_stage2_cyclic_training_geometry(payload)

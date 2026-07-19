from __future__ import annotations

import pytest

from rc.stage2_domain_balanced_cyclic_sampler import (
    ALGORITHM_ID,
    SCHEMA_VERSION,
    Stage2DomainBalancedSamplerError,
    build_domain_balanced_cyclic_epoch,
)


def _build(epoch: int = 0):
    return build_domain_balanced_cyclic_epoch(
        outer_fold_id="outer_leave_irstd_1k",
        derived_seed=123456,
        epoch=epoch,
        episode_counts={"NUAA-SIRST": 170, "NUDT-SIRST": 509},
    )


def test_epoch_is_equal_domain_without_replacement_and_pair_balanced() -> None:
    payload = _build()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["algorithm_id"] == ALGORITHM_ID
    assert payload["draws_per_domain"] == 170
    assert payload["epoch_size"] == 340
    assert payload["replacement_within_domain_epoch"] is False

    selection = payload["ordered_selection"]
    assert len(selection) == 340
    by_domain = {"NUAA-SIRST": [], "NUDT-SIRST": []}
    for row in selection:
        by_domain[row["source_domain"]].append(row["domain_episode_index"])
    assert {key: len(value) for key, value in by_domain.items()} == {
        "NUAA-SIRST": 170,
        "NUDT-SIRST": 170,
    }
    assert len(set(by_domain["NUAA-SIRST"])) == 170
    assert len(set(by_domain["NUDT-SIRST"])) == 170
    for start in range(0, len(selection), 2):
        assert {
            selection[start]["source_domain"],
            selection[start + 1]["source_domain"],
        } == {"NUAA-SIRST", "NUDT-SIRST"}


def test_replay_is_byte_identity_stable_and_epoch_specific() -> None:
    first = _build(epoch=3)
    replay = _build(epoch=3)
    other = _build(epoch=4)
    assert first == replay
    assert first["ordered_selection_sha256"] == replay["ordered_selection_sha256"]
    assert first["ordered_selection"] != other["ordered_selection"]
    assert first["ordered_selection_sha256"] != other["ordered_selection_sha256"]


def test_rotating_slice_covers_the_larger_domain() -> None:
    observed: set[int] = set()
    for epoch in range(3):
        payload = _build(epoch)
        observed.update(
            row["domain_episode_index"]
            for row in payload["ordered_selection"]
            if row["source_domain"] == "NUDT-SIRST"
        )
    assert observed == set(range(509))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outer_fold_id": "unknown"},
        {"derived_seed": True},
        {"epoch": -1},
        {"episode_counts": {"NUAA-SIRST": 170}},
        {
            "episode_counts": {
                "NUAA-SIRST": 83,
                "NUDT-SIRST": 509,
            }
        },
        {
            "episode_counts": {
                "NUAA-SIRST": 170,
                "NUDT-SIRST": 509,
                "IRSTD-1K": 638,
            }
        },
    ],
)
def test_invalid_sampler_inputs_fail_closed(kwargs) -> None:
    arguments = {
        "outer_fold_id": "outer_leave_irstd_1k",
        "derived_seed": 123456,
        "epoch": 0,
        "episode_counts": {"NUAA-SIRST": 170, "NUDT-SIRST": 509},
    }
    arguments.update(kwargs)
    with pytest.raises(Stage2DomainBalancedSamplerError):
        build_domain_balanced_cyclic_epoch(**arguments)

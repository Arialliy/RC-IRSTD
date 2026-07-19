from __future__ import annotations

import copy

import pytest

from rc.stage2_variable_query_geometry import (
    CONSTRUCTION,
    CONTEXT_SIZE,
    INDEX_SEMANTICS,
    MINIMUM_QUERY_SIZE,
    MINIMUM_WINDOW_SIZE,
    QUERY_SIZE_POLICY,
    SCHEMA_VERSION,
    WINDOW_COUNT_RULE,
    Stage2VariableQueryGeometryContractError,
    build_stage2_variable_query_geometry,
    derive_stage2_query_sizes,
    validate_stage2_variable_query_geometry,
)


GOLDEN_QUERY_SIZES = {
    43: (29,),
    85: (29, 28),
    127: (29, 28, 28),
    159: (39, 39, 39),
    254: (29, 29, 28, 28, 28, 28),
    255: (29, 29, 29, 28, 28, 28),
    319: (32, 32, 32, 32, 31, 31, 31),
}


@pytest.mark.parametrize(
    ("record_count", "expected_query_sizes"), GOLDEN_QUERY_SIZES.items()
)
def test_golden_geometry_is_balanced_contiguous_and_exhaustive(
    record_count: int, expected_query_sizes: tuple[int, ...]
) -> None:
    assert derive_stage2_query_sizes(record_count) == expected_query_sizes
    geometry = build_stage2_variable_query_geometry(record_count)
    assert validate_stage2_variable_query_geometry(geometry) == geometry
    assert geometry["window_count"] == record_count // MINIMUM_WINDOW_SIZE

    windows = geometry["windows"]
    assert tuple(window["query_size"] for window in windows) == expected_query_sizes
    assert all(window["context_size"] == CONTEXT_SIZE for window in windows)
    assert min(expected_query_sizes) >= MINIMUM_QUERY_SIZE
    assert max(expected_query_sizes) - min(expected_query_sizes) <= 1
    assert sum(expected_query_sizes) + CONTEXT_SIZE * len(windows) == record_count

    covered: list[int] = []
    for previous, window in zip([None, *windows[:-1]], windows, strict=True):
        if previous is None:
            assert window["context_start"] == 0
        else:
            assert window["context_start"] == previous["query_stop"]
        assert window["context_stop"] == window["context_start"] + CONTEXT_SIZE
        assert window["query_start"] == window["context_stop"]
        assert window["query_stop"] == window["query_start"] + window["query_size"]
        covered.extend(range(window["context_start"], window["context_stop"]))
        covered.extend(range(window["query_start"], window["query_stop"]))
    assert covered == list(range(record_count))


def test_n43_payload_is_schema_golden() -> None:
    assert build_stage2_variable_query_geometry(43) == {
        "schema_version": SCHEMA_VERSION,
        "context_size": 14,
        "minimum_query_size": 28,
        "minimum_window_size": 42,
        "ordered_record_count": 43,
        "window_count": 1,
        "window_count_rule": WINDOW_COUNT_RULE,
        "query_size_policy": QUERY_SIZE_POLICY,
        "construction": CONSTRUCTION,
        "index_semantics": INDEX_SEMANTICS,
        "all_indices_consumed_once": True,
        "windows": [
            {
                "window_index": 0,
                "context_start": 0,
                "context_stop": 14,
                "query_start": 14,
                "query_stop": 43,
                "context_size": 14,
                "query_size": 29,
            }
        ],
    }


def test_minimum_complete_window_is_valid() -> None:
    geometry = build_stage2_variable_query_geometry(MINIMUM_WINDOW_SIZE)
    assert geometry["window_count"] == 1
    assert geometry["windows"][0]["query_size"] == MINIMUM_QUERY_SIZE
    assert geometry["windows"][0]["query_stop"] == MINIMUM_WINDOW_SIZE


@pytest.mark.parametrize(
    "record_count", [None, True, False, "43", 43.0, -1, 0, 1, 41]
)
def test_invalid_record_counts_fail_closed(record_count: object) -> None:
    with pytest.raises(Stage2VariableQueryGeometryContractError):
        derive_stage2_query_sizes(record_count)  # type: ignore[arg-type]
    with pytest.raises(Stage2VariableQueryGeometryContractError):
        build_stage2_variable_query_geometry(record_count)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("context_size", "minimum_query_size"),
    [
        (13, 28),
        (15, 28),
        (14, 27),
        (14, 29),
        (True, 28),
        (14, False),
        (14.0, 28),
        (14, "28"),
    ],
)
def test_nonfrozen_or_noninteger_parameters_fail_closed(
    context_size: object, minimum_query_size: object
) -> None:
    for operation in (
        derive_stage2_query_sizes,
        build_stage2_variable_query_geometry,
    ):
        with pytest.raises(Stage2VariableQueryGeometryContractError):
            operation(
                43,
                context_size=context_size,  # type: ignore[arg-type]
                minimum_query_size=minimum_query_size,  # type: ignore[arg-type]
            )


def test_validator_rejects_nonmapping_and_field_closure_changes() -> None:
    with pytest.raises(TypeError):
        validate_stage2_variable_query_geometry([])  # type: ignore[arg-type]

    geometry = build_stage2_variable_query_geometry(85)
    missing = copy.deepcopy(geometry)
    missing.pop("query_size_policy")
    with pytest.raises(Stage2VariableQueryGeometryContractError, match="missing"):
        validate_stage2_variable_query_geometry(missing)

    extra = copy.deepcopy(geometry)
    extra["query_size"] = 28
    with pytest.raises(Stage2VariableQueryGeometryContractError, match="extra"):
        validate_stage2_variable_query_geometry(extra)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", "rc-irstd.stage2-variable-query-geometry.v0"),
        ("context_size", 13),
        ("minimum_query_size", 29),
        ("minimum_window_size", 43),
        ("window_count", 1),
        ("window_count_rule", "ceil(N/42)"),
        ("query_size_policy", "all_suffix_to_last_window"),
        ("construction", "noncontiguous"),
        ("index_semantics", "one_based_closed"),
        ("all_indices_consumed_once", False),
    ],
)
def test_validator_rejects_top_level_contract_mutations(
    field: str, replacement: object
) -> None:
    geometry = build_stage2_variable_query_geometry(85)
    geometry[field] = replacement
    with pytest.raises(Stage2VariableQueryGeometryContractError):
        validate_stage2_variable_query_geometry(geometry)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("window_index", True),
        ("context_start", 1),
        ("context_stop", 15),
        ("query_start", 15),
        ("query_stop", 42),
        ("context_size", 13),
        ("query_size", 28),
    ],
)
def test_validator_rejects_window_type_overlap_gap_or_size_mutation(
    field: str, replacement: object
) -> None:
    geometry = build_stage2_variable_query_geometry(85)
    geometry["windows"][0][field] = replacement
    with pytest.raises(Stage2VariableQueryGeometryContractError):
        validate_stage2_variable_query_geometry(geometry)


def test_validator_rejects_window_reordering_and_nested_field_changes() -> None:
    geometry = build_stage2_variable_query_geometry(85)
    geometry["windows"].reverse()
    with pytest.raises(Stage2VariableQueryGeometryContractError):
        validate_stage2_variable_query_geometry(geometry)

    missing = build_stage2_variable_query_geometry(43)
    missing["windows"][0].pop("query_stop")
    with pytest.raises(Stage2VariableQueryGeometryContractError, match="missing"):
        validate_stage2_variable_query_geometry(missing)

    extra = build_stage2_variable_query_geometry(43)
    extra["windows"][0]["record_stop"] = 43
    with pytest.raises(Stage2VariableQueryGeometryContractError, match="extra"):
        validate_stage2_variable_query_geometry(extra)


def test_validator_rejects_bool_for_integer_and_integer_for_bool() -> None:
    record_count_bool = build_stage2_variable_query_geometry(43)
    record_count_bool["ordered_record_count"] = True
    with pytest.raises(Stage2VariableQueryGeometryContractError, match="exact integer"):
        validate_stage2_variable_query_geometry(record_count_bool)

    consumed_integer = build_stage2_variable_query_geometry(43)
    consumed_integer["all_indices_consumed_once"] = 1
    with pytest.raises(Stage2VariableQueryGeometryContractError, match="exactly true"):
        validate_stage2_variable_query_geometry(consumed_integer)

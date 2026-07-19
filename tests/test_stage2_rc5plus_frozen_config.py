from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from rc.stage2_rc5plus_frozen_config import (
    Stage2RC5PlusFrozenConfigError,
    VerifiedStage2RC5PlusFrozenConfig,
    assert_verified_stage2_rc5plus_frozen_config,
    verify_stage2_rc5plus_frozen_config_file,
    verify_stage2_rc5plus_frozen_config_payload,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/aaai27_stage2_crossfit_rc5plus_v1.json"


def _payload() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_unique_rc5plus_configuration_replays_against_live_contracts() -> None:
    verified = verify_stage2_rc5plus_frozen_config_file(CONFIG)
    assert len(verified.canonical_sha256) == 64
    assert len(verified.source_bytes_sha256 or "") == 64
    assert_verified_stage2_rc5plus_frozen_config(verified)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("budget_contract", "grid_exact_rationals"), [[1, 10_000], [1, 100_000], [1, 1_000_000]]),
        (("checkpoint_contract", "schema_version"), "rc-irstd.calibrator.v7"),
        (("model", "methods", "T6_PLUS", "expected_trainable_parameters"), 3_306),
        (("source_validation_contract", "selection_grid_index"), 0),
        (("performance_success_gate", "macro_domain_delta_BSR_point_min"), 0.0),
        (("performance_success_gate", "secondary_metric_rescue"), True),
        (("deployment_contract", "caller_threshold_injection"), True),
    ],
)
def test_contract_drift_fails_closed(
    path: tuple[str, ...], replacement: object
) -> None:
    payload = deepcopy(_payload())
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index,assignment]
    cursor[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(Stage2RC5PlusFrozenConfigError):
        verify_stage2_rc5plus_frozen_config_payload(payload)


def test_unknown_top_level_authority_fails_closed() -> None:
    payload = _payload()
    payload["runtime_override"] = {"threshold": 0.5}
    with pytest.raises(Stage2RC5PlusFrozenConfigError, match="field closure"):
        verify_stage2_rc5plus_frozen_config_payload(payload)


def test_capability_cannot_be_constructed_or_forged() -> None:
    with pytest.raises(TypeError, match="verifier-issued"):
        VerifiedStage2RC5PlusFrozenConfig()
    forged = object.__new__(VerifiedStage2RC5PlusFrozenConfig)
    with pytest.raises(TypeError, match="verifier-issued"):
        assert_verified_stage2_rc5plus_frozen_config(forged)

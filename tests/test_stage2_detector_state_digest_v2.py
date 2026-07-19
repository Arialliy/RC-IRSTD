from __future__ import annotations

import torch

from data_ext.stage2_detector_run_complete_v2 import _state_dict_digest


def test_state_digest_supports_zero_dimensional_batchnorm_counters() -> None:
    state = {
        "layer.weight": torch.tensor([1.0, 2.0]),
        "layer.num_batches_tracked": torch.tensor(7, dtype=torch.int64),
    }
    first = _state_dict_digest(state)
    second = _state_dict_digest({key: state[key] for key in reversed(state)})
    assert first == second
    assert len(first) == 64

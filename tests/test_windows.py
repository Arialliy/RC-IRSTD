from rc_irstd.data.windows import build_causal_windows


def test_causal_windows_are_disjoint_and_sequence_local():
    sequences = ["a"] * 10 + ["b"] * 10
    frames = list(range(10)) + list(range(10))
    windows = build_causal_windows(sequences, frames, context_size=4, horizon=2, stride=2)
    assert windows
    for window in windows:
        assert set(window.context_indices).isdisjoint(window.future_indices)
        expected_range = range(0, 10) if window.sequence_id == "a" else range(10, 20)
        assert all(index in expected_range for index in window.context_indices + window.future_indices)

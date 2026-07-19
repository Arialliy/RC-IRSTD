from __future__ import annotations

import pytest

from evaluation import export_stage2_development_scores_run_complete_v2 as gated


def test_gated_export_requires_and_checks_capability_before_legacy_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    capability = object()

    def verify(value, **kwargs):
        events.append(("verify", value))
        assert kwargs["run_contract_sha256"] == "a" * 64
        assert kwargs["checkpoint_sha256"] == "b" * 64

    def export(*args, **kwargs):
        events.append(("export", args))
        return {"status": "synthetic"}

    monkeypatch.setattr(gated, "assert_stage2_run_complete_for_score_export_v2", verify)
    monkeypatch.setattr(gated, "export_stage2_development_scores", export)
    result = gated.export_stage2_development_scores_run_complete_v2(
        "selection.json",
        "run.json",
        "checkpoint.pt",
        "scores",
        selection_contract_sha256="c" * 64,
        run_contract_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        role="source_oof_train",
        run_complete_capability=capability,
    )
    assert result == {"status": "synthetic"}
    assert events == [
        ("verify", capability),
        (
            "export",
            ("selection.json", "run.json", "checkpoint.pt", "scores"),
        ),
    ]


def test_gated_export_has_no_default_completion_capability() -> None:
    with pytest.raises(TypeError, match="run_complete_capability"):
        gated.export_stage2_development_scores_run_complete_v2(
            "selection.json",
            "run.json",
            "checkpoint.pt",
            "scores",
            selection_contract_sha256="c" * 64,
            run_contract_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            role="source_oof_train",
        )

"""RUN_COMPLETE-v2-gated wrapper for Stage-2 score export.

This additive entry point leaves the RC4 exporter untouched while making the
RC5 completion capability mandatory for callers that opt into the v2 path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from data_ext.stage2_detector_run_complete_v2 import (
    assert_stage2_run_complete_for_score_export_v2,
)
from evaluation.export_stage2_development_scores import (
    export_stage2_development_scores,
)


def export_stage2_development_scores_run_complete_v2(
    selection_contract: str | Path,
    run_contract: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    selection_contract_sha256: str,
    run_contract_sha256: str,
    checkpoint_sha256: str,
    role: str,
    run_complete_capability: Any,
    device: str = "cuda",
    repository_root: str | Path | None = None,
    model_factory: Callable[[Mapping[str, Any], torch.device], torch.nn.Module]
    | None = None,
) -> dict[str, Any]:
    """Export only after replaying an exact RUN_COMPLETE-v2 capability."""

    assert_stage2_run_complete_for_score_export_v2(
        run_complete_capability,
        run_contract_path=run_contract,
        run_contract_sha256=run_contract_sha256,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    return export_stage2_development_scores(
        selection_contract,
        run_contract,
        checkpoint,
        output_dir,
        selection_contract_sha256=selection_contract_sha256,
        run_contract_sha256=run_contract_sha256,
        checkpoint_sha256=checkpoint_sha256,
        role=role,
        device=device,
        repository_root=repository_root,
        model_factory=model_factory,
    )


__all__ = ["export_stage2_development_scores_run_complete_v2"]

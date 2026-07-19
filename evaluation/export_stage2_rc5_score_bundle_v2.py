"""Authoritative RC5 detector-score export with persistent completion edge."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from data_ext import stage2_score_manifest as _score_v4
from data_ext.stage2_detector_run_complete_v2 import (
    VerifiedStage2DetectorRunCompleteV2,
    assert_stage2_run_complete_for_score_export_v2,
)
from data_ext.stage2_rc5_score_bundle_v2 import (
    VerifiedStage2RC5ScoreBundleV2,
    publish_stage2_rc5_score_attestation_v2,
)
from data_ext.stage2_score_manifest_metadata_v5 import (
    verify_stage2_score_manifest_metadata_v5,
)
from evaluation.export_stage2_development_scores import (
    export_stage2_development_scores,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_stage2_rc5_score_bundle_v2(
    selection_contract: str | Path,
    run_contract: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    selection_contract_sha256: str,
    run_contract_sha256: str,
    checkpoint_sha256: str,
    role: str,
    run_complete_capability: VerifiedStage2DetectorRunCompleteV2,
    device: str = "cuda",
    repository_root: str | Path | None = None,
    model_factory: Callable[[Mapping[str, Any], torch.device], torch.nn.Module]
    | None = None,
) -> VerifiedStage2RC5ScoreBundleV2:
    """Export v4 scores, then commit an RC5 RUN_COMPLETE attestation.

    A successfully published v4 directory is still non-authoritative for RC5
    until ``RC5_SCORE_ATTESTATION.json.sha256`` exists and the returned bundle
    is issued.  Any failure before that commit leaves no authoritative RC5
    score capability.
    """

    root = _score_v4._repository_root(repository_root)
    run_path = _score_v4._existing_direct_path(
        run_contract, root, "score-export run contract"
    )
    checkpoint_path = _score_v4._existing_direct_path(
        checkpoint, root, "score-export checkpoint"
    )
    complete = assert_stage2_run_complete_for_score_export_v2(
        run_complete_capability,
        run_contract_path=run_path,
        run_contract_sha256=run_contract_sha256,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )
    export_stage2_development_scores(
        selection_contract,
        run_path,
        checkpoint_path,
        output_dir,
        selection_contract_sha256=selection_contract_sha256,
        run_contract_sha256=run_contract_sha256,
        checkpoint_sha256=checkpoint_sha256,
        role=role,
        device=device,
        repository_root=root,
        model_factory=model_factory,
    )
    raw_output = Path(output_dir).expanduser()
    final_output = raw_output if raw_output.is_absolute() else root / raw_output
    manifest_path = final_output / "manifest.json"
    manifest_sha256 = _sha256(manifest_path)
    metadata = verify_stage2_score_manifest_metadata_v5(
        manifest_path,
        manifest_sha256,
        role,
        repository_root=root,
    )
    return publish_stage2_rc5_score_attestation_v2(metadata, complete)


__all__ = ["export_stage2_rc5_score_bundle_v2"]

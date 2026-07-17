"""Reusable synthetic Lane-A fixtures; not collected as tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from data_ext.stage2_label_attachment import verify_stage2_window_contract
from data_ext.stage2_score_manifest import verify_stage2_score_manifest
from rc import stage2_crossfit_schema as crossfit_schema
from rc.build_stage2_crossfit_episodes import build_stage2_context_package
from rc.schema import StatisticsConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_file(path: Path, payload: Mapping[str, Any]) -> str:
    data = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def publish_synthetic_verified_context(
    workspace: Mapping[str, Any],
    monkeypatch: Any,
    *,
    output_name: str = "context-package.json",
) -> dict[str, Any]:
    """Publish a real Lane-A context bundle over a W05 synthetic workspace.

    W03 window/score verification, the 93D extractor, publisher and public
    context verifier are real.  Only the independent W04 reference verifier is
    replaced by a deterministic verifier-shaped source reference.
    """

    root = Path(workspace["root"])
    source_root = Path(crossfit_schema.__file__).resolve().parents[1]
    synthetic_rc = root / "rc"
    synthetic_rc.mkdir(parents=True, exist_ok=True)
    (synthetic_rc / "domain_statistics.py").write_bytes(
        (source_root / "rc" / "domain_statistics.py").read_bytes()
    )
    for binding in crossfit_schema.GOVERNANCE_BINDINGS.values():
        source = source_root / binding["path"]
        target = root / binding["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    monkeypatch.setattr(
        crossfit_schema, "__file__", str(synthetic_rc / "stage2_crossfit_schema.py")
    )
    config = StatisticsConfig(
        peak_kernel_size=3,
        peak_min_score=0.05,
        quantile_sample_limit=128,
    )
    support = Path(workspace["output"]).parent / "lane-a-synthetic-support"
    support.mkdir(parents=True, exist_ok=True)
    config_path = support / "statistics-config.json"
    config_sha = _canonical_file(config_path, config.to_dict())
    reference_path = support / "source-reference.npz"
    audit_path = support / "source-reference.audit.json"
    if not reference_path.exists():
        reference_path.write_bytes(b"synthetic-w04-reference\n")
    if not audit_path.exists():
        audit_path.write_bytes(b'{"synthetic":true}\n')
    reference_sha = _sha256(reference_path)
    audit_sha = _sha256(audit_path)

    window = verify_stage2_window_contract(
        workspace["window_path"],
        workspace["window_sha"],
        workspace["window_id"],
        workspace["role"],
        repository_root=root,
    )
    score = verify_stage2_score_manifest(
        workspace["score_path"],
        workspace["score_sha"],
        workspace["role"],
        repository_root=root,
    )
    identity = {
        "run_id": str(score.payload.get("run_id", "synthetic-run")),
        "outer_fold_id": score.payload["outer_fold_id"],
        "outer_target": score.payload["outer_target"],
        "base_seed": score.payload["base_seed"],
        "derived_seed": score.payload["derived_seed"],
        "detector_role": score.payload["detector_role"],
        "oof_fold_index": score.payload["oof_fold_index"],
        "checkpoint_sha256": score.bindings["checkpoint"]["sha256"],
    }
    consumer = {
        "path": window.path.relative_to(root).as_posix(),
        "sha256": window.manifest_sha256,
        "domain": score.payload["source_domain"],
        "episode_role": crossfit_schema.ROLE_TO_EPISODE[window.role],
    }
    stage2_contract = {
        "reference_role": workspace["role"],
        "detector_identity": identity,
        "bindings": {
            "statistics_config": {
                "path": config_path.relative_to(root).as_posix(),
                "sha256": config_sha,
            },
            "consumer_window_manifests": [consumer],
        },
    }
    fake_reference = SimpleNamespace(
        path=reference_path,
        npz_sha256=reference_sha,
        audit_path=audit_path,
        audit_sha256=audit_sha,
        statistics_config=config,
        stage2_contract=stage2_contract,
        source_reference=SimpleNamespace(
            centers=(tuple(np.zeros(87, dtype=np.float64)),),
            scale=tuple(np.ones(87, dtype=np.float64)),
        ),
    )

    def fake_verify(
        path: str | Path,
        expected_sha256: str,
        expected_audit_sha256: str,
        *,
        statistics_config: StatisticsConfig,
        expected_consumer_window_path: str | Path | None = None,
        expected_consumer_window_sha256: str | None = None,
        expected_consumer_window_id: str | None = None,
        repository_root: str | Path | None = None,
    ) -> Any:
        assert Path(path) == reference_path
        assert expected_sha256 == reference_sha
        assert expected_audit_sha256 == audit_sha
        assert statistics_config == config
        assert Path(expected_consumer_window_path) == window.path
        assert expected_consumer_window_sha256 == window.manifest_sha256
        assert expected_consumer_window_id == window.window_id
        assert Path(repository_root) == root
        return fake_reference

    monkeypatch.setattr(
        crossfit_schema, "verify_stage2_source_reference", fake_verify
    )
    output = support / output_name
    result = build_stage2_context_package(
        window_manifest=workspace["window_path"],
        window_manifest_sha256=workspace["window_sha"],
        window_id=workspace["window_id"],
        expected_role=workspace["role"],
        score_manifest=workspace["score_path"],
        score_manifest_sha256=workspace["score_sha"],
        source_reference=reference_path,
        source_reference_sha256=reference_sha,
        source_reference_audit_sha256=audit_sha,
        statistics_config=config,
        output=output,
        repository_root_value=root,
    )
    verified = crossfit_schema.verify_stage2_context_package(
        output,
        result["context_sha256"],
        result["commit_sha256"],
        statistics_config=config,
        repository_root=root,
    )
    return {
        "path": output,
        "sha256": result["context_sha256"],
        "commit_path": verified.commit_path,
        "commit_sha256": result["commit_sha256"],
        "statistics_config_path": config_path,
        "statistics_config_sha256": config_sha,
        "statistics_config": config,
        "verified": verified,
    }

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from evaluation import stage2_threshold_family as threshold_module
from data_ext.stage2_threshold_decision import (
    PRELABEL_METHOD_ORDER,
    Stage2ThresholdDecisionContractError,
    T5_MISSING_OUTCOME,
    VerifiedStage2ThresholdDecisionSet,
    canonical_json_sha256,
    verify_stage2_threshold_decision_set,
)
from evaluation.stage2_threshold_family import (
    build_prelabel_decision,
    build_source_threshold_reference,
    build_t9_postlabel_diagnostic,
    calibrator_logits_to_thresholds,
    make_shared_input_bindings,
    publish_prelabel_decision_set,
    select_source_safe_threshold,
    t0_fixed_thresholds,
    t1_pooled_source_thresholds,
    t2_safer_source_thresholds,
    t3_nearest_source_thresholds,
    t4_context_order_statistic,
    t5_evt_gpd_thresholds,
)


def _sha(tag: str) -> str:
    return canonical_json_sha256({"tag": tag})


def _shared() -> dict[str, object]:
    return make_shared_input_bindings(
        context_package_path="synthetic/context.json",
        context_package_sha256=_sha("context"),
        context_package_commit_path="synthetic/context.COMMIT.json",
        context_package_commit_sha256=_sha("context-commit"),
        window_id="outer_leave_nuaa_sirst::outer_target::window_0000",
        window_identity_sha256=_sha("window"),
        ordered_query_identity_sha256=_sha("query"),
        score_manifest_sha256=_sha("score-manifest"),
        score_records_content_sha256=_sha("scores"),
        detector_checkpoint_sha256=_sha("detector"),
    )


def _decision(method_id: str) -> dict[str, object]:
    values: tuple[float, float, float] | None
    if method_id == "T0":
        values = t0_fixed_thresholds()
    elif method_id == "T5":
        values = None
    elif method_id in {"T1", "T2", "T3", "T4", "T7", "T8"}:
        values = (0.6, 0.7, 0.8)
    else:
        values = (0.8, 0.7, 0.6)
    return build_prelabel_decision(
        method_id=method_id,
        thresholds=values,
        shared_bindings=_shared(),
        outer_fold_id="outer_leave_nuaa_sirst",
        outer_target_domain="nuaa-sirst",
        base_seed=42,
        derived_seed=123456,
        method_contract={"method_id": method_id, "frozen": True},
        method_binding=(
            {"path": f"synthetic/{method_id}.checkpoint.pt", "sha256": _sha(method_id)}
            if method_id in {"T1", "T2", "T3", "T5", "T6", "T7", "T8"}
            else None
        ),
    )


def _curve(thresholds: list[float], fps: list[int], tps: list[int]) -> list[dict[str, object]]:
    return [
        {
            "threshold": threshold,
            "tp_objects": tp,
            "gt_objects": 10,
            "fp_pixels": fp,
            "total_pixels": 1_000_000,
        }
        for threshold, fp, tp in zip(thresholds, fps, tps, strict=True)
    ]


def test_source_safe_rank_is_exact_pd_then_fp_then_larger_threshold() -> None:
    rows = _curve(
        [0.7, 0.8, 0.9, 1.0],
        [90, 80, 80, 0],
        [8, 8, 8, 0],
    )
    selected = select_source_safe_threshold(rows, 1e-4)
    assert selected["threshold"] == 0.9
    assert selected["tp_objects"] == 8


def test_source_reference_drives_t1_t2_and_nearest_t3() -> None:
    pooled = _curve([0.7, 0.8, 0.9, 1.0], [90, 9, 0, 0], [9, 8, 7, 0])
    left = _curve([0.6, 0.75, 0.95, 1.0], [90, 9, 0, 0], [9, 8, 7, 0])
    right = _curve([0.65, 0.85, 0.99, 1.0], [90, 9, 0, 0], [9, 8, 7, 0])
    reference = build_source_threshold_reference(
        pooled_curve=pooled,
        domain_curves={"irstd-1k": left, "nudt-sirst": right},
        standardized_source_centers_0_86={
            "irstd-1k": np.zeros(87),
            "nudt-sirst": np.ones(87),
        },
        outer_fold_id="outer_leave_nuaa_sirst",
        outer_target_domain="nuaa-sirst",
        base_seed=42,
        derived_seed=123456,
        detector_checkpoint_sha256=_sha("detector"),
        collection_path="synthetic/source.jsonl",
        collection_sha256=_sha("collection"),
        collection_commit_path="synthetic/source.COMMIT.json",
        collection_commit_sha256=_sha("collection-commit"),
        collection_identity_sha256=_sha("collection-identity"),
        standardizer_fit_manifest_sha256=_sha("standardizer-fit"),
        standardizer_train_collection_sha256=_sha("training-collection"),
    )
    assert reference["standardizer_binding"] == {
        "fit_manifest_sha256": _sha("standardizer-fit"),
        "train_collection_sha256": _sha("training-collection"),
    }
    assert t1_pooled_source_thresholds(reference) == (0.7, 0.8, 0.9)
    assert t2_safer_source_thresholds(reference) == (0.65, 0.85, 0.99)
    context = np.zeros(93)
    assert t3_nearest_source_thresholds(reference, context) == (0.6, 0.75, 0.95)
    context[:87] = 1.0
    assert t3_nearest_source_thresholds(reference, context) == (0.65, 0.85, 0.99)


def test_t4_uses_frozen_strict_greater_order_statistic() -> None:
    maps = [np.arange(100, dtype=np.float64).reshape(10, 10) / 1400.0 + i / 14.0 for i in range(14)]
    values = np.sort(np.concatenate([array.reshape(-1) for array in maps]))
    got = t4_context_order_statistic(maps)
    expected = tuple(
        float(values[max(0, values.size - int(np.floor(b * values.size)) - 1)])
        for b in (1e-4, 1e-5, 1e-6)
    )
    assert got == expected


def test_t5_insufficient_tail_is_missing_without_fallback() -> None:
    maps = [np.zeros((2, 2), dtype=np.float64) for _ in range(14)]
    assert t5_evt_gpd_thresholds(maps) is None
    decision = _decision("T5")
    assert decision["outcome"] == T5_MISSING_OUTCOME
    assert decision["thresholds"] is None
    assert decision["fallback_used"] is False


def test_calibrator_probability_projection_and_monotone_guard() -> None:
    direct = calibrator_logits_to_thresholds([-1.0, 1.0, 0.0], method_id="T6")
    assert direct[0] < direct[2] < direct[1]
    monotone = calibrator_logits_to_thresholds([-1.0, 0.0, 1.0], method_id="T8")
    assert monotone[0] < monotone[1] < monotone[2]
    with pytest.raises(ValueError, match="monotonicity"):
        calibrator_logits_to_thresholds([0.0, -1.0, 1.0], method_id="T7")


def test_atomic_t0_t8_bundle_is_publicly_verified_and_unforgeable(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    decisions = [_decision(method) for method in PRELABEL_METHOD_ORDER]
    path, digest = publish_prelabel_decision_set(
        decisions, root / "sealed", repository_root=root
    )
    verified = verify_stage2_threshold_decision_set(
        path,
        digest,
        expected_context_package_sha256=_sha("context"),
        expected_context_commit_sha256=_sha("context-commit"),
        expected_window_id="outer_leave_nuaa_sirst::outer_target::window_0000",
        expected_outer_fold_id="outer_leave_nuaa_sirst",
        expected_base_seed=42,
        expected_derived_seed=123456,
        expected_detector_checkpoint_sha256=_sha("detector"),
        repository_root=root,
    )
    assert isinstance(verified, VerifiedStage2ThresholdDecisionSet)
    assert tuple(verified.decision_by_method) == PRELABEL_METHOD_ORDER
    assert verified.decision_by_method["T5"].payload["outcome"] == T5_MISSING_OUTCOME
    with pytest.raises(TypeError, match="verifier-only"):
        VerifiedStage2ThresholdDecisionSet(
            path=path,
            payload=verified.payload,
            manifest_sha256=digest,
            decisions=verified.decisions,
            _token=object(),
        )


def test_bundle_rejects_member_mutation_and_t9_injection(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    path, digest = publish_prelabel_decision_set(
        [_decision(method) for method in PRELABEL_METHOD_ORDER],
        root / "sealed",
        repository_root=root,
    )
    member = path.parent / "T8.decision.json"
    payload = json.loads(member.read_text(encoding="utf-8"))
    payload["method_id"] = "T9"
    member.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Stage2ThresholdDecisionContractError):
        verify_stage2_threshold_decision_set(path, digest, repository_root=root)


def test_active_publication_lock_fails_closed(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    path, digest = publish_prelabel_decision_set(
        [_decision(method) for method in PRELABEL_METHOD_ORDER],
        root / "sealed",
        repository_root=root,
    )
    (root / ".sealed.lock").write_text("foreign\n", encoding="ascii")
    with pytest.raises(Stage2ThresholdDecisionContractError, match="lock"):
        verify_stage2_threshold_decision_set(path, digest, repository_root=root)


def _publication_residue(root: Path, name: str) -> list[str]:
    return sorted(path.name for path in root.glob(f".{name}.staging-*"))


def test_publication_keeps_owned_lock_through_final_public_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    original = threshold_module.verify_stage2_threshold_decision_set
    final_seen = False

    def observing_verify(path: str | Path, digest: str, **kwargs: object):
        nonlocal final_seen
        candidate = Path(path)
        if candidate.parent == root / "sealed":
            final_seen = True
            lock = root / ".sealed.lock"
            assert lock.is_file()
            with pytest.raises(Stage2ThresholdDecisionContractError, match="lock"):
                original(candidate, digest, repository_root=root)
        return original(path, digest, **kwargs)

    monkeypatch.setattr(
        threshold_module,
        "verify_stage2_threshold_decision_set",
        observing_verify,
    )
    path, digest = publish_prelabel_decision_set(
        [_decision(method) for method in PRELABEL_METHOD_ORDER],
        root / "sealed",
        repository_root=root,
    )
    assert final_seen
    assert not (root / ".sealed.lock").exists()
    assert not _publication_residue(root, "sealed")
    assert original(path, digest, repository_root=root).manifest_sha256 == digest


def test_no_replace_target_race_preserves_foreign_directory_and_cleans_owned_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    original = threshold_module._rename_directory_no_replace

    def inject_target(source: Path, target: Path, identity: tuple[int, int]):
        target.mkdir()
        (target / "foreign.txt").write_text("foreign\n", encoding="ascii")
        return original(source, target, identity)

    monkeypatch.setattr(
        threshold_module,
        "_rename_directory_no_replace",
        inject_target,
    )
    with pytest.raises(BaseException):
        publish_prelabel_decision_set(
            [_decision(method) for method in PRELABEL_METHOD_ORDER],
            root / "sealed",
            repository_root=root,
        )
    assert (root / "sealed" / "foreign.txt").read_text(encoding="ascii") == "foreign\n"
    assert not (root / ".sealed.lock").exists()
    assert not _publication_residue(root, "sealed")


def test_foreign_replacement_lock_is_never_unlinked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()

    def replace_lock_then_fail(path: Path, payload: object) -> None:
        lock = root / ".sealed.lock"
        lock.unlink()
        lock.write_text("foreign-lock\n", encoding="ascii")
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(
        threshold_module,
        "_write_json_exclusive",
        replace_lock_then_fail,
    )
    with pytest.raises(BaseException):
        publish_prelabel_decision_set(
            [_decision(method) for method in PRELABEL_METHOD_ORDER],
            root / "sealed",
            repository_root=root,
        )
    assert (root / ".sealed.lock").read_text(encoding="ascii") == "foreign-lock\n"
    assert not (root / "sealed").exists()
    assert not _publication_residue(root, "sealed")


def test_staging_open_failure_removes_new_directory_and_owned_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    original_open = threshold_module.os.open
    injected = False

    def fail_staging_open(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal injected
        candidate = Path(path)  # type: ignore[arg-type]
        if (
            not injected
            and candidate.name.startswith(".sealed.staging-")
            and flags & getattr(threshold_module.os, "O_DIRECTORY", 0)
        ):
            injected = True
            raise OSError("injected staging directory open failure")
        return original_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(threshold_module.os, "open", fail_staging_open)
    with pytest.raises(OSError, match="staging directory open"):
        publish_prelabel_decision_set(
            [_decision(method) for method in PRELABEL_METHOD_ORDER],
            root / "sealed",
            repository_root=root,
        )
    assert not (root / ".sealed.lock").exists()
    assert not _publication_residue(root, "sealed")


def test_foreign_replacement_staging_directory_is_never_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()

    def replace_staging_then_fail(path: Path, payload: object) -> None:
        staging = path.parent
        shutil.rmtree(staging)
        staging.mkdir()
        (staging / "foreign.txt").write_text("foreign-staging\n", encoding="ascii")
        raise RuntimeError("injected staging replacement")

    monkeypatch.setattr(
        threshold_module,
        "_write_json_exclusive",
        replace_staging_then_fail,
    )
    with pytest.raises(BaseException):
        publish_prelabel_decision_set(
            [_decision(method) for method in PRELABEL_METHOD_ORDER],
            root / "sealed",
            repository_root=root,
        )
    residues = _publication_residue(root, "sealed")
    assert len(residues) == 1
    assert (root / residues[0] / "foreign.txt").read_text(encoding="ascii") == (
        "foreign-staging\n"
    )
    assert not (root / ".sealed.lock").exists()
    assert not (root / "sealed").exists()


def test_post_publication_verifier_failure_rolls_back_owned_bundle_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    original = threshold_module.verify_stage2_threshold_decision_set

    def fail_final(path: str | Path, digest: str, **kwargs: object):
        verified = original(path, digest, **kwargs)
        if Path(path).parent == root / "sealed":
            assert (root / ".sealed.lock").is_file()
            raise RuntimeError("injected post-publication verification failure")
        return verified

    monkeypatch.setattr(
        threshold_module,
        "verify_stage2_threshold_decision_set",
        fail_final,
    )
    with pytest.raises(RuntimeError, match="post-publication"):
        publish_prelabel_decision_set(
            [_decision(method) for method in PRELABEL_METHOD_ORDER],
            root / "sealed",
            repository_root=root,
        )
    assert not (root / "sealed").exists()
    assert not (root / ".sealed.lock").exists()
    assert not _publication_residue(root, "sealed")


def test_commit_and_commit_sidecar_are_the_last_staged_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    events: list[tuple[str, str]] = []
    original_json = threshold_module._write_json_exclusive
    original_sidecar = threshold_module._write_sidecar

    def observe_json(path: Path, payload: object) -> None:
        events.append(("json", path.name))
        original_json(path, payload)

    def observe_sidecar(path: Path, digest: str) -> None:
        events.append(("sidecar", path.name))
        original_sidecar(path, digest)

    monkeypatch.setattr(threshold_module, "_write_json_exclusive", observe_json)
    monkeypatch.setattr(threshold_module, "_write_sidecar", observe_sidecar)
    publish_prelabel_decision_set(
        [_decision(method) for method in PRELABEL_METHOD_ORDER],
        root / "sealed",
        repository_root=root,
    )
    assert events[-2:] == [("json", "COMMIT.json"), ("sidecar", "COMMIT.json")]


def test_atomic_publisher_cli_requires_external_hash_and_emits_verified_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path.resolve()
    decision_list = root / "decisions.json"
    data = json.dumps(
        [_decision(method) for method in PRELABEL_METHOD_ORDER],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    decision_list.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    assert threshold_module.main(
        [
            "--decisions",
            str(decision_list),
            "--decisions-sha256",
            digest,
            "--output",
            "sealed",
            "--repository-root",
            str(root),
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    verified = verify_stage2_threshold_decision_set(
        root / result["decision_set_path"],
        result["decision_set_sha256"],
        repository_root=root,
    )
    assert verified.manifest_sha256 == result["decision_set_sha256"]

    other = root / "other.json"
    other.write_bytes(data)
    with pytest.raises(ValueError, match="external SHA"):
        threshold_module.main(
            [
                "--decisions",
                str(other),
                "--decisions-sha256",
                "0" * 64,
                "--output",
                "must-not-exist",
                "--repository-root",
                str(root),
            ]
        )
    assert not (root / "must-not-exist").exists()


def test_t9_is_postlabel_only_and_not_a_prelabel_decision() -> None:
    diagnostic = build_t9_postlabel_diagnostic(
        query_curve_rows=_curve([0.7, 0.8, 1.0], [90, 9, 0], [9, 8, 0]),
        query_curve_sha256=_sha("curve"),
        outer_fold_id="outer_leave_nuaa_sirst",
        outer_target_domain="nuaa-sirst",
        base_seed=42,
        derived_seed=123456,
    )
    assert diagnostic["schema_version"].endswith("postlabel-oracle-diagnostic.v1")
    assert diagnostic["prelabel_eligible"] is False
    assert diagnostic["may_enter_selection_or_gate"] is False
    assert diagnostic["method_id"] not in PRELABEL_METHOD_ORDER

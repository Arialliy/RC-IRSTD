from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from data_ext.dataset_identity import (
    SCORE_MANIFEST_CONTENT_ALGORITHM,
    score_manifest_content_sha256,
    sha256_file,
)
from data_ext.label_manifest_artifacts import (
    LABEL_MANIFEST_ARTIFACT_TYPE,
    LABEL_MANIFEST_CONTENT_ALGORITHM,
    LABEL_MANIFEST_SCHEMA_VERSION,
    label_manifest_content_sha256,
)
from evaluation.threshold_sweep import (
    CURVE_FIELDS,
    ScoreMapRecord,
    build_threshold_plan,
    default_threshold_grid,
    sweep_thresholds,
    threshold_grid_metadata,
    write_curve_csv,
)
from model.threshold_calibrator import ThresholdCalibrator, asymmetric_threshold_loss
from rc.build_meta_episodes import main as build_episodes_main
from rc.build_meta_episodes import _causal_window_status, _verify_oracle_event_coverage
from rc.build_source_reference import main as build_source_reference_main
from rc.domain_statistics import (
    BASE_FEATURE_DIM,
    FEATURE_DIM,
    extract_unlabeled_statistics,
    load_source_reference,
)
from rc.meta_dataset import (
    FeatureStandardizer,
    assert_verified_provenance,
    load_episodes,
    split_by_pseudo_target,
    validate_episode_collection,
)
from rc.online_adapter import main as online_main
from rc.oracle_threshold import select_oracle_operating_point
from rc.schema import (
    BudgetSpec,
    EpisodeProvenance,
    FoldContract,
    RCEpisode,
    SourceReference,
    StatisticsConfig,
)
from rc.train_calibrator import main as train_calibrator_main


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _statistics_config(min_score: float = 0.05) -> StatisticsConfig:
    return StatisticsConfig(peak_kernel_size=3, peak_min_score=min_score)


def _write_source_reference(
    path: Path,
    config: StatisticsConfig,
    domains: tuple[str, ...] = ("S1", "S2"),
    *,
    checkpoint_sha: str = SHA_A,
    outer_fold_id: str = "fold-out",
    outer_target: str = "OUT",
    held_out_domains: tuple[str, ...] = ("A", "B", "OUT"),
    protocol_scope: str = "multi_source_protocol_candidate",
) -> SourceReference:
    centers = np.zeros((len(domains), BASE_FEATURE_DIM), dtype=np.float32)
    scale = np.ones(BASE_FEATURE_DIM, dtype=np.float32)
    np.savez_compressed(
        path,
        domains=np.asarray(domains),
        centers=centers,
        scale=scale,
        statistics_config_json=np.asarray(json.dumps(config.to_dict(), sort_keys=True)),
        source_contract_json=np.asarray(
            json.dumps(
                {
                    "detector_checkpoint_sha": checkpoint_sha,
                    "detector_source_domains": list(domains),
                    "outer_fold_id": outer_fold_id,
                    "outer_target": outer_target,
                    "held_out_domains": list(held_out_domains),
                    "protocol_scope": protocol_scope,
                },
                sort_keys=True,
            )
        ),
    )
    return load_source_reference(path, statistics_config=config)


def _provenance(target: str, status: str = "verified") -> EpisodeProvenance:
    return EpisodeProvenance(
        status=status,
        curve_file_sha256=SHA_A,
        curve_manifest_sha256=SHA_B if status == "verified" else "",
        context_score_manifest_sha256=SHA_C,
        query_score_manifest_sha256=SHA_C if status == "verified" else SHA_D,
        query_score_target_dataset=target,
        label_manifest_sha256=SHA_D if status == "verified" else "",
        label_manifest_content_sha256=SHA_A if status == "verified" else "",
    )


def _episode(
    episode_id: str,
    pseudo_target: str,
    value: float,
    source_reference: SourceReference,
    *,
    config: StatisticsConfig | None = None,
    pd: float = 0.8,
    p_min: float = 0.5,
    provenance_status: str = "verified",
) -> RCEpisode:
    config = config or _statistics_config()
    stats = extract_unlabeled_statistics(
        np.full((2, 8, 8), value, dtype=np.float32),
        source_reference=source_reference,
        statistics_config=config,
    )
    return RCEpisode.create(
        episode_id=episode_id,
        pseudo_target=pseudo_target,
        context_image_ids=[f"{episode_id}-context"],
        query_image_ids=[f"{episode_id}-query"],
        statistics=stats.vector,
        feature_names=stats.feature_names,
        statistics_config=config,
        source_reference=source_reference,
        fold=FoldContract(
            outer_fold_id=str(source_reference.contract.outer_fold_id),
            outer_target=str(source_reference.contract.outer_target),
            detector_source_domains=source_reference.contract.detector_source_domains,
            detector_checkpoint_sha=source_reference.contract.detector_checkpoint_sha,
            held_out_domains=source_reference.contract.held_out_domains,
            protocol_scope=str(source_reference.contract.protocol_scope),
        ),
        provenance=_provenance(pseudo_target, provenance_status),
        budgets=BudgetSpec(values=(1e-5, 0.0), active=(True, False)),
        oracle_threshold=0.7,
        oracle_pd=pd,
        oracle_pixel_risk=5e-6,
        oracle_component_risk=2.0,
        p_min=p_min,
    )


def _write_score_manifest(
    directory: Path,
    target: str,
    checkpoint_sha: str,
    image_ids: tuple[str, ...],
    *,
    held_out_domains: tuple[str, ...] = ("A", "B", "OUT"),
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    image_directory = directory / "images"
    image_directory.mkdir(exist_ok=True)
    items = []
    for image_id in image_ids:
        image_path = image_directory / f"{image_id}.png"
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(image_path)
        score_path = directory / f"{image_id}.npz"
        np.savez_compressed(
            score_path,
            prob=np.full((8, 8), 0.25, dtype=np.float32),
            image_id=np.asarray(image_id),
            dataset_name=np.asarray(target),
            original_hw=np.asarray((8, 8), dtype=np.int32),
        )
        items.append(
            {
                "image_id": image_id,
                "file": score_path.name,
                "score_file_sha256": sha256_file(score_path),
                "image_path": f"images/{image_path.name}",
                "gray_file_sha256": sha256_file(image_path),
                "original_hw": [8, 8],
            }
        )
    manifest = {
        "schema_version": 2,
        "artifact_type": "label_free_score_export",
        "path_anchor": "manifest_directory",
        "target_dataset": target,
        "weight_sha256": checkpoint_sha,
        "detector_source_domains": ["S1", "S2"],
        "held_out_domains": list(held_out_domains),
        "outer_fold_id": "fold-out",
        "outer_target": "OUT",
        "protocol_scope": "multi_source_protocol_candidate",
        "target_exclusion_verified": True,
        "score_type": "sigmoid_probability",
        "restored_to_original_hw": True,
        "threshold_semantics": "prediction = probability > threshold",
        "labels_embedded": False,
        "label_contract": "external_label_attachment_manifest_required_offline",
        "num_images": len(items),
        "content_sha256_algorithm": SCORE_MANIFEST_CONTENT_ALGORITHM,
        "content_sha256": score_manifest_content_sha256(items),
        "items": items,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def _write_label_manifest(score_manifest: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    score_payload = json.loads(score_manifest.read_text(encoding="utf-8"))
    items = []
    for score_item in score_payload["items"]:
        image_id = str(score_item["image_id"])
        mask = np.zeros(tuple(score_item["original_hw"]), dtype=np.uint8)
        mask[0, 0] = 1
        label_path = directory / f"{image_id}.label.npz"
        np.savez_compressed(
            label_path,
            mask=mask,
            image_id=np.asarray(image_id),
            original_hw=np.asarray(score_item["original_hw"], dtype=np.int32),
        )
        items.append(
            {
                "image_id": image_id,
                "file": label_path.name,
                "label_file_sha256": sha256_file(label_path),
                "source_image_file_sha256": score_item["gray_file_sha256"],
                "original_hw": list(score_item["original_hw"]),
            }
        )
    payload = {
        "schema_version": LABEL_MANIFEST_SCHEMA_VERSION,
        "artifact_type": LABEL_MANIFEST_ARTIFACT_TYPE,
        "path_anchor": "manifest_directory",
        "score_manifest_file": Path(
            os.path.relpath(score_manifest.resolve(), start=directory.resolve())
        ).as_posix(),
        "score_manifest_sha256": _sha256(score_manifest),
        "score_manifest_content_sha256": score_payload["content_sha256"],
        "target_dataset": score_payload["target_dataset"],
        "labels_embedded_in_scores": False,
        "num_images": len(items),
        "content_sha256_algorithm": LABEL_MANIFEST_CONTENT_ALGORITHM,
        "content_sha256": label_manifest_content_sha256(items),
        "items": items,
    }
    path = directory / "label-manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_source_domain_manifest(
    root: Path,
    target: str,
    probability: np.ndarray,
    gray: np.ndarray,
) -> Path:
    score_directory = root / f"scores-{target}"
    image_directory = root / f"dataset-{target}" / "images"
    score_directory.mkdir(parents=True)
    image_directory.mkdir(parents=True)
    image_id = f"{target}-0"
    Image.fromarray(gray.astype(np.uint8), mode="L").save(
        image_directory / f"{image_id}.png"
    )
    score_path = score_directory / f"{image_id}.npz"
    image_path = image_directory / f"{image_id}.png"
    np.savez_compressed(
        score_path,
        prob=probability.astype(np.float32),
        image_id=np.asarray(image_id),
        dataset_name=np.asarray(target),
        original_hw=np.asarray(probability.shape, dtype=np.int32),
    )
    items = [
        {
            "image_id": image_id,
            "file": score_path.name,
            "score_file_sha256": sha256_file(score_path),
            "image_path": f"../dataset-{target}/images/{image_path.name}",
            "gray_file_sha256": sha256_file(image_path),
            "original_hw": list(probability.shape),
        }
    ]
    manifest = {
        "schema_version": 2,
        "artifact_type": "label_free_score_export",
        "path_anchor": "manifest_directory",
        "target_dataset": target,
        "weight_sha256": SHA_A,
        "detector_source_domains": ["A", "B"],
        "held_out_domains": ["OUT"],
        "outer_fold_id": "fold-out",
        "outer_target": "OUT",
        "protocol_scope": "multi_source_protocol_candidate",
        "target_exclusion_verified": False,
        "score_type": "sigmoid_probability",
        "restored_to_original_hw": True,
        "threshold_semantics": "prediction = probability > threshold",
        "labels_embedded": False,
        "label_contract": "external_label_attachment_manifest_required_offline",
        "dataset_dir": f"../dataset-{target}",
        "num_images": 1,
        "content_sha256_algorithm": SCORE_MANIFEST_CONTENT_ALGORITHM,
        "content_sha256": score_manifest_content_sha256(items),
        "items": items,
    }
    path = score_directory / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


class RCContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="rc-tests-")
        self.root = Path(self.temporary.name)
        self.config = _statistics_config()
        self.reference_path = self.root / "source-reference.npz"
        self.reference = _write_source_reference(
            self.reference_path, self.config
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_oracle_contract_and_budget_mask(self) -> None:
        curve = {
            "threshold": [0.2, 0.4, 1.0],
            "pd": [0.8, 0.8, 0.9],
            "fa_pixel": [1e-4, 1e-6, 1.0],
            "fa_component_mp": [100.0, 2.0, 100.0],
        }
        result = select_oracle_operating_point(
            curve,
            BudgetSpec(values=(2e-4, 3.0), active=(True, True)),
            p_min=0.5,
        )
        self.assertEqual(result.threshold, 0.4)
        sentinel = select_oracle_operating_point(
            curve,
            BudgetSpec(values=(1e-12, 0.1), active=(True, True)),
            p_min=0.1,
        )
        self.assertEqual((sentinel.threshold, sentinel.pd), (1.0, 0.0))
        self.assertTrue(sentinel.reject)

    def test_statistics_config_and_local_rank_nms_are_stable(self) -> None:
        stats = extract_unlabeled_statistics(
            np.full((2, 8, 8), 0.5, dtype=np.float32),
            source_reference=self.reference,
            statistics_config=self.config,
        )
        self.assertEqual(stats.vector.shape, (FEATURE_DIM,))
        self.assertEqual(stats.metadata["num_peaks"], 2)
        self.assertEqual(stats.statistics_config, self.config)
        self.assertEqual(stats.vector[-6], 1.0)

    def test_source_reference_rejects_config_mismatch(self) -> None:
        self.assertEqual(self.reference.contract.detector_checkpoint_sha, SHA_A)
        with self.assertRaisesRegex(ValueError, "at least two detector sources"):
            FoldContract(
                "fold-out",
                "OUT",
                ("S1",),
                SHA_A,
                ("A", "OUT"),
                "multi_source_protocol_candidate",
            )
        with self.assertRaisesRegex(ValueError, "exactly one detector source"):
            FoldContract(
                "fold-out",
                "OUT",
                ("S1", "S2"),
                SHA_A,
                ("A", "OUT"),
                "single_source_inner_smoke_not_main_result",
            )
        smoke_fold = FoldContract(
            "fold-out",
            "OUT",
            ("S1",),
            SHA_A,
            ("A", "OUT"),
            "single_source_inner_smoke_not_main_result",
        )
        self.assertEqual(smoke_fold.detector_source_domains, ("S1",))
        mismatched_fold = FoldContract(
            "fold-out",
            "OUT",
            ("S1", "S2"),
            SHA_B,
            ("A", "B", "OUT"),
            "multi_source_protocol_candidate",
        )
        with self.assertRaisesRegex(ValueError, "detector/fold contract"):
            mismatched_fold.assert_matches_source_reference(self.reference)
        with self.assertRaisesRegex(ValueError, "statistics_config"):
            load_source_reference(
                self.reference_path,
                statistics_config=_statistics_config(min_score=0.2),
            )
        legacy_path = self.root / "legacy-reference.npz"
        np.savez_compressed(
            legacy_path,
            domains=np.asarray(["S1"]),
            centers=np.zeros((1, BASE_FEATURE_DIM), dtype=np.float32),
            scale=np.ones(BASE_FEATURE_DIM, dtype=np.float32),
            statistics_config_json=np.asarray(json.dumps(self.config.to_dict())),
        )
        with self.assertRaisesRegex(KeyError, "source_contract_json"):
            load_source_reference(legacy_path, statistics_config=self.config)

    def test_source_reference_cli_uses_unlabeled_scores_and_recovers_gray(self) -> None:
        probability_a = np.linspace(0.0, 0.7, 64, dtype=np.float32).reshape(8, 8)
        probability_b = np.linspace(0.2, 0.9, 64, dtype=np.float32).reshape(8, 8)
        gray_a = np.arange(64, dtype=np.uint8).reshape(8, 8)
        gray_b = np.flipud(gray_a)
        manifest_a = _write_source_domain_manifest(
            self.root, "A", probability_a, gray_a
        )
        manifest_b = _write_source_domain_manifest(
            self.root, "B", probability_b, gray_b
        )
        output = self.root / "built-source-reference.npz"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = build_source_reference_main(
                [
                    "--score-manifest",
                    str(manifest_b),
                    "--score-manifest",
                    str(manifest_a),
                    "--domain",
                    "B",
                    "--domain",
                    "A",
                    "--peak-kernel-size",
                    "3",
                    "--peak-min-score",
                    "0.05",
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(result, 0)
        reference = load_source_reference(output, statistics_config=self.config)
        self.assertEqual(reference.domains, ("A", "B"))
        expected_centers = np.stack(
            [
                extract_unlabeled_statistics(
                    probability_a,
                    gray_a,
                    statistics_config=self.config,
                ).vector[:BASE_FEATURE_DIM],
                extract_unlabeled_statistics(
                    probability_b,
                    gray_b,
                    statistics_config=self.config,
                ).vector[:BASE_FEATURE_DIM],
            ]
        )
        expected_scale = expected_centers.astype(np.float64).std(axis=0)
        expected_scale = np.where(expected_scale < 1e-8, 1.0, expected_scale)
        np.testing.assert_allclose(np.asarray(reference.centers), expected_centers)
        np.testing.assert_allclose(np.asarray(reference.scale), expected_scale, rtol=1e-6)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["source_reference_sha256"], _sha256(output))
        self.assertEqual(summary["detector_checkpoint_sha"], SHA_A)
        with np.load(output, allow_pickle=False) as artifact:
            source_contract = json.loads(
                str(np.asarray(artifact["source_contract_json"]).item())
            )
        self.assertEqual(
            source_contract,
            {
                "detector_checkpoint_sha": SHA_A,
                "detector_source_domains": ["A", "B"],
                "held_out_domains": ["OUT"],
                "outer_fold_id": "fold-out",
                "outer_target": "OUT",
                "protocol_scope": "multi_source_protocol_candidate",
            },
        )

        broken = json.loads(manifest_b.read_text(encoding="utf-8"))
        broken["weight_sha256"] = SHA_B
        manifest_b.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "same detector checkpoint"):
            build_source_reference_main(
                [
                    "--score-manifest",
                    str(manifest_a),
                    "--score-manifest",
                    str(manifest_b),
                    "--peak-kernel-size",
                    "3",
                    "--peak-min-score",
                    "0.05",
                    "--output",
                    str(self.root / "invalid-source-reference.npz"),
                ]
            )

        broken["weight_sha256"] = SHA_A
        broken["detector_provenance"] = {
            "detector_source_domains": ["A", "tampered-domain"]
        }
        manifest_b.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "conflicting top-level/nested"):
            build_source_reference_main(
                [
                    "--score-manifest",
                    str(manifest_a),
                    "--score-manifest",
                    str(manifest_b),
                    "--peak-kernel-size",
                    "3",
                    "--peak-min-score",
                    "0.05",
                    "--output",
                    str(self.root / "tampered-source-reference.npz"),
                ]
            )

    def test_source_reference_rejects_incomplete_and_tampered_exports(self) -> None:
        probability = np.full((8, 8), 0.25, dtype=np.float32)
        gray = np.zeros((8, 8), dtype=np.uint8)
        manifest_a = _write_source_domain_manifest(
            self.root, "A", probability, gray
        )
        manifest_b = _write_source_domain_manifest(
            self.root, "B", probability, gray
        )
        marker = manifest_a.parent / ".export_incomplete"
        marker.write_text("incomplete", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            build_source_reference_main(
                [
                    "--score-manifest",
                    str(manifest_a),
                    "--score-manifest",
                    str(manifest_b),
                    "--peak-kernel-size",
                    "3",
                    "--peak-min-score",
                    "0.05",
                    "--output",
                    str(self.root / "incomplete-reference.npz"),
                ]
            )
        marker.unlink()

        item = json.loads(manifest_a.read_text(encoding="utf-8"))["items"][0]
        score_path = manifest_a.parent / item["file"]
        np.savez_compressed(
            score_path,
            prob=np.full((8, 8), 0.75, dtype=np.float32),
            image_id=np.asarray("A-0"),
            dataset_name=np.asarray("A"),
            original_hw=np.asarray((8, 8), dtype=np.int32),
        )
        with self.assertRaisesRegex(ValueError, "score-file SHA-256 mismatch"):
            build_source_reference_main(
                [
                    "--score-manifest",
                    str(manifest_a),
                    "--score-manifest",
                    str(manifest_b),
                    "--peak-kernel-size",
                    "3",
                    "--peak-min-score",
                    "0.05",
                    "--output",
                    str(self.root / "tampered-reference.npz"),
                ]
            )

    def test_episode_rejects_source_and_outer_leakage(self) -> None:
        with self.assertRaisesRegex(ValueError, "outer_target"):
            FoldContract(
                "f",
                "OUT",
                ("OUT", "S2"),
                SHA_A,
                ("OUT",),
                "multi_source_protocol_candidate",
            )
        leaking_path = self.root / "leaking.npz"
        leaking = _write_source_reference(
            leaking_path,
            self.config,
            domains=("A", "D"),
            held_out_domains=("B", "OUT"),
        )
        stats = extract_unlabeled_statistics(
            np.ones((1, 4, 4), dtype=np.float32),
            source_reference=leaking,
            statistics_config=self.config,
        )
        with self.assertRaisesRegex(ValueError, "detector/fold contract"):
            RCEpisode.create(
                episode_id="leak",
                pseudo_target="B",
                context_image_ids=["c"],
                query_image_ids=["q"],
                statistics=stats.vector,
                feature_names=stats.feature_names,
                statistics_config=self.config,
                source_reference=leaking,
                fold=FoldContract(
                    "fold-out",
                    "OUT",
                    ("C", "D"),
                    SHA_A,
                    ("B", "OUT"),
                    "multi_source_protocol_candidate",
                ),
                provenance=_provenance("B"),
                budgets=BudgetSpec((1e-5, 0.0), (True, False)),
                oracle_threshold=0.5,
                oracle_pd=0.8,
                oracle_pixel_risk=1e-6,
                oracle_component_risk=1.0,
                p_min=0.5,
            )

    def test_collection_rejects_config_pmin_and_unverified_provenance(self) -> None:
        first = _episode("a", "A", 0.1, self.reference)
        different_pmin = _episode(
            "b", "B", 0.2, self.reference, p_min=0.6
        )
        with self.assertRaisesRegex(ValueError, "p_min"):
            validate_episode_collection([first, different_pmin])
        different_config = _statistics_config(min_score=0.2)
        different_ref_path = self.root / "different-ref.npz"
        different_ref = _write_source_reference(
            different_ref_path, different_config
        )
        changed = _episode(
            "b2", "B", 0.2, different_ref, config=different_config
        )
        with self.assertRaisesRegex(ValueError, "statistics_config"):
            validate_episode_collection([first, changed])
        unverified = _episode(
            "u", "B", 0.2, self.reference, provenance_status="asserted_unverified"
        )
        with self.assertRaisesRegex(ValueError, "asserted_unverified"):
            assert_verified_provenance([first, unverified])

    def _verified_builder_fixture(self) -> dict[str, object]:
        shared_manifest = _write_score_manifest(
            self.root / "shared", "A", SHA_A, ("c", "q")
        )
        label_manifest_path = _write_label_manifest(
            shared_manifest,
            self.root / "labels",
        )
        label_manifest = json.loads(
            label_manifest_path.read_text(encoding="utf-8")
        )
        curve_path = self.root / "curve.csv"
        query_mask = np.zeros((8, 8), dtype=np.uint8)
        query_mask[0, 0] = 1
        query_record = ScoreMapRecord(
            probability=np.full((8, 8), 0.25, dtype=np.float32),
            mask=query_mask,
            image_id="q",
        )
        plan = build_threshold_plan(
            [query_record],
            default_threshold_grid(),
            mode="adaptive",
            high_tail_lower_bound=0.2,
            event_threshold_cap=4096,
        )
        rows = sweep_thresholds(
            [query_record],
            plan.thresholds,
            matching_rule="overlap",
            centroid_distance=3.0,
            threshold_mode="fixed",
        )
        write_curve_csv(rows, curve_path, write_manifest=False)
        curve_manifest = {
            **threshold_grid_metadata(
                plan.thresholds,
                threshold_audit=plan.audit,
            ),
            "curve_file": curve_path.name,
            "curve_sha256": _sha256(curve_path),
            "image_ids": ["q"],
            "num_images": 1,
            "score_manifest_file": "shared/manifest.json",
            "score_manifest_sha256": _sha256(shared_manifest),
            "score_manifest_num_images": 2,
            "target_dataset": "A",
            "detector_weight_sha256": SHA_A,
            "gt_objects": 1,
            "total_pixels": 64,
            "matching_rule": "overlap",
            "centroid_distance": 3.0,
            "evaluation_scope": "score_bound_label_attachment_verified",
            "label_manifest_file": str(
                label_manifest_path.relative_to(self.root)
            ),
            "label_manifest_sha256": _sha256(label_manifest_path),
            "label_manifest_content_sha256": label_manifest["content_sha256"],
            "label_manifest_num_images": label_manifest["num_images"],
            "label_manifest_target_dataset": label_manifest["target_dataset"],
        }
        curve_manifest_path = self.root / "curve-manifest.json"
        curve_manifest_path.write_text(
            json.dumps(curve_manifest, sort_keys=True), encoding="utf-8"
        )
        spec = {
            "episodes": [
                {
                    "episode_id": "verified",
                    "pseudo_target": "A",
                    "outer_fold_id": "fold-out",
                    "outer_target": "OUT",
                    "detector_source_domains": ["S1", "S2"],
                    "detector_checkpoint_sha": SHA_A,
                    "held_out_domains": ["A", "B", "OUT"],
                    "protocol_scope": "multi_source_protocol_candidate",
                    "statistics_config": self.config.to_dict(),
                    "source_reference": self.reference_path.name,
                    "context_manifest": str(shared_manifest.relative_to(self.root)),
                    "context_score_manifest_sha256": _sha256(shared_manifest),
                    "context_image_ids": ["c"],
                    "query_image_ids": ["q"],
                    "curve_manifest": curve_manifest_path.name,
                    "curve_manifest_sha256": _sha256(curve_manifest_path),
                    "query_score_manifest_sha256": _sha256(shared_manifest),
                    "budgets": {
                        "names": ["pixel", "component"],
                        "values": [2e-6, 0.0],
                        "active": [True, False],
                    },
                    "p_min": 0.5,
                }
            ]
        }
        spec_path = self.root / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        output = self.root / "episodes.jsonl"
        return {
            "shared_manifest": shared_manifest,
            "curve_path": curve_path,
            "plan": plan,
            "curve_manifest": curve_manifest,
            "curve_manifest_path": curve_manifest_path,
            "label_manifest": label_manifest,
            "label_manifest_path": label_manifest_path,
            "spec": spec,
            "spec_path": spec_path,
            "output": output,
        }

    def test_builder_requires_verified_three_way_curve_contract(self) -> None:
        fixture = self._verified_builder_fixture()
        shared_manifest = fixture["shared_manifest"]
        curve_manifest = fixture["curve_manifest"]
        curve_manifest_path = fixture["curve_manifest_path"]
        label_manifest = fixture["label_manifest"]
        label_manifest_path = fixture["label_manifest_path"]
        spec = fixture["spec"]
        spec_path = fixture["spec_path"]
        output = fixture["output"]
        assert isinstance(shared_manifest, Path)
        assert isinstance(curve_manifest, dict)
        assert isinstance(curve_manifest_path, Path)
        assert isinstance(label_manifest, dict)
        assert isinstance(label_manifest_path, Path)
        assert isinstance(spec, dict)
        assert isinstance(spec_path, Path)
        assert isinstance(output, Path)
        build_episodes_main(
            ["--spec-file", str(spec_path), "--output", str(output)]
        )
        episode = load_episodes(output)[0]
        self.assertEqual(episode.provenance.status, "verified")
        self.assertEqual(episode.provenance.query_score_manifest_sha256, _sha256(shared_manifest))
        self.assertEqual(
            episode.provenance.label_manifest_sha256,
            _sha256(label_manifest_path),
        )
        self.assertEqual(
            episode.provenance.label_manifest_content_sha256,
            label_manifest["content_sha256"],
        )

        score_manifest_payload = json.loads(
            shared_manifest.read_text(encoding="utf-8")
        )
        score_manifest_payload["restored_to_original_hw"] = False
        shared_manifest.write_text(
            json.dumps(score_manifest_payload, sort_keys=True), encoding="utf-8"
        )
        changed_score_manifest_sha = _sha256(shared_manifest)
        curve_manifest["score_manifest_sha256"] = changed_score_manifest_sha
        curve_manifest_path.write_text(
            json.dumps(curve_manifest, sort_keys=True), encoding="utf-8"
        )
        spec["episodes"][0][
            "context_score_manifest_sha256"
        ] = changed_score_manifest_sha
        spec["episodes"][0][
            "query_score_manifest_sha256"
        ] = changed_score_manifest_sha
        spec["episodes"][0]["curve_manifest_sha256"] = _sha256(
            curve_manifest_path
        )
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "restored_to_original_hw"):
            build_episodes_main(
                ["--spec-file", str(spec_path), "--output", str(output)]
            )

        score_manifest_payload["restored_to_original_hw"] = True
        shared_manifest.write_text(
            json.dumps(score_manifest_payload, sort_keys=True), encoding="utf-8"
        )
        restored_score_manifest_sha = _sha256(shared_manifest)
        curve_manifest["score_manifest_sha256"] = restored_score_manifest_sha
        spec["episodes"][0][
            "context_score_manifest_sha256"
        ] = restored_score_manifest_sha
        spec["episodes"][0][
            "query_score_manifest_sha256"
        ] = restored_score_manifest_sha

        curve_manifest["score_manifest_sha256"] = SHA_D
        curve_manifest_path.write_text(json.dumps(curve_manifest), encoding="utf-8")
        spec["episodes"][0]["curve_manifest_sha256"] = _sha256(curve_manifest_path)
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "score manifest SHA"):
            build_episodes_main(
                ["--spec-file", str(spec_path), "--output", str(output)]
            )

    def test_verified_builder_rejects_forged_event_counts(self) -> None:
        fixture = self._verified_builder_fixture()
        curve_manifest = fixture["curve_manifest"]
        curve_manifest_path = fixture["curve_manifest_path"]
        spec = fixture["spec"]
        spec_path = fixture["spec_path"]
        output = fixture["output"]
        assert isinstance(curve_manifest, dict)
        assert isinstance(curve_manifest_path, Path)
        assert isinstance(spec, dict)
        assert isinstance(spec_path, Path)
        assert isinstance(output, Path)

        # Keep the hand-authored audit internally self-consistent so only a
        # score-derived recount can expose the forgery.
        curve_manifest["event_candidate_count"] = 2
        curve_manifest["event_threshold_count"] = 2
        curve_manifest["event_coverage_fraction_lower_bound"] = 1.0
        curve_manifest_path.write_text(
            json.dumps(curve_manifest, sort_keys=True), encoding="utf-8"
        )
        spec["episodes"][0]["curve_manifest_sha256"] = _sha256(
            curve_manifest_path
        )
        spec_path.write_text(json.dumps(spec), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "event_candidate_count"):
            build_episodes_main(
                ["--spec-file", str(spec_path), "--output", str(output)]
            )

    def test_verified_builder_rejects_csv_manifest_threshold_mismatch(self) -> None:
        fixture = self._verified_builder_fixture()
        curve_path = fixture["curve_path"]
        curve_manifest = fixture["curve_manifest"]
        curve_manifest_path = fixture["curve_manifest_path"]
        spec = fixture["spec"]
        spec_path = fixture["spec_path"]
        output = fixture["output"]
        assert isinstance(curve_path, Path)
        assert isinstance(curve_manifest, dict)
        assert isinstance(curve_manifest_path, Path)
        assert isinstance(spec, dict)
        assert isinstance(spec_path, Path)
        assert isinstance(output, Path)

        with curve_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames
        assert rows and fieldnames is not None
        rows[0]["threshold"] = "0.001"
        with curve_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        curve_manifest["curve_sha256"] = _sha256(curve_path)
        curve_manifest_path.write_text(
            json.dumps(curve_manifest, sort_keys=True), encoding="utf-8"
        )
        spec["episodes"][0]["curve_manifest_sha256"] = _sha256(
            curve_manifest_path
        )
        spec_path.write_text(json.dumps(spec), encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError, "CSV threshold column differs from curve manifest"
        ):
            build_episodes_main(
                ["--spec-file", str(spec_path), "--output", str(output)]
            )

    def test_verified_builder_requires_and_replays_matching_contract(self) -> None:
        mutations = (
            ("missing-rule", lambda manifest: manifest.pop("matching_rule"), "matching_rule"),
            (
                "missing-distance",
                lambda manifest: manifest.pop("centroid_distance"),
                "centroid_distance",
            ),
            (
                "tampered-rule",
                lambda manifest: manifest.__setitem__("matching_rule", "centroid"),
                "independently recomputed query sweep",
            ),
        )
        for name, mutate, expected_error in mutations:
            with self.subTest(name=name):
                fixture = self._verified_builder_fixture()
                curve_manifest = fixture["curve_manifest"]
                curve_manifest_path = fixture["curve_manifest_path"]
                spec = fixture["spec"]
                spec_path = fixture["spec_path"]
                output = fixture["output"]
                assert isinstance(curve_manifest, dict)
                assert isinstance(curve_manifest_path, Path)
                assert isinstance(spec, dict)
                assert isinstance(spec_path, Path)
                assert isinstance(output, Path)

                mutate(curve_manifest)
                curve_manifest_path.write_text(
                    json.dumps(curve_manifest, sort_keys=True), encoding="utf-8"
                )
                spec["episodes"][0]["curve_manifest_sha256"] = _sha256(
                    curve_manifest_path
                )
                spec_path.write_text(json.dumps(spec), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected_error):
                    build_episodes_main(
                        ["--spec-file", str(spec_path), "--output", str(output)]
                    )

    def test_verified_builder_requires_independent_label_binding(self) -> None:
        mutations = (
            (
                "missing-label-file",
                lambda manifest: manifest.pop("label_manifest_file"),
                "label_manifest_file",
            ),
            (
                "wrong-label-sha",
                lambda manifest: manifest.__setitem__(
                    "label_manifest_sha256", SHA_D
                ),
                "label manifest SHA-256",
            ),
            (
                "diagnostic-scope",
                lambda manifest: manifest.__setitem__(
                    "evaluation_scope", "legacy_combined_npz_diagnostic"
                ),
                "score-bound label attachment",
            ),
        )
        for name, mutate, expected_error in mutations:
            with self.subTest(name=name):
                fixture = self._verified_builder_fixture()
                curve_manifest = fixture["curve_manifest"]
                curve_manifest_path = fixture["curve_manifest_path"]
                spec = fixture["spec"]
                spec_path = fixture["spec_path"]
                output = fixture["output"]
                assert isinstance(curve_manifest, dict)
                assert isinstance(curve_manifest_path, Path)
                assert isinstance(spec, dict)
                assert isinstance(spec_path, Path)
                assert isinstance(output, Path)

                mutate(curve_manifest)
                curve_manifest_path.write_text(
                    json.dumps(curve_manifest, sort_keys=True), encoding="utf-8"
                )
                spec["episodes"][0]["curve_manifest_sha256"] = _sha256(
                    curve_manifest_path
                )
                spec_path.write_text(json.dumps(spec), encoding="utf-8")
                with self.assertRaisesRegex((KeyError, ValueError), expected_error):
                    build_episodes_main(
                        ["--spec-file", str(spec_path), "--output", str(output)]
                    )

    def test_verified_builder_rejects_tampered_risk_column(self) -> None:
        fixture = self._verified_builder_fixture()
        curve_path = fixture["curve_path"]
        curve_manifest = fixture["curve_manifest"]
        curve_manifest_path = fixture["curve_manifest_path"]
        spec = fixture["spec"]
        spec_path = fixture["spec_path"]
        output = fixture["output"]
        assert isinstance(curve_path, Path)
        assert isinstance(curve_manifest, dict)
        assert isinstance(curve_manifest_path, Path)
        assert isinstance(spec, dict)
        assert isinstance(spec_path, Path)
        assert isinstance(output, Path)

        with curve_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames
        assert rows and fieldnames == list(CURVE_FIELDS)
        rows[0]["fa_pixel"] = "0.123456789"
        with curve_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        curve_manifest["curve_sha256"] = _sha256(curve_path)
        curve_manifest_path.write_text(
            json.dumps(curve_manifest, sort_keys=True), encoding="utf-8"
        )
        spec["episodes"][0]["curve_manifest_sha256"] = _sha256(
            curve_manifest_path
        )
        spec_path.write_text(json.dumps(spec), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "field 'fa_pixel'.*recomputed"):
            build_episodes_main(
                ["--spec-file", str(spec_path), "--output", str(output)]
            )

    def test_verified_builder_rejects_thresholds_stale_for_query_scores(self) -> None:
        fixture = self._verified_builder_fixture()
        shared_manifest = fixture["shared_manifest"]
        curve_manifest = fixture["curve_manifest"]
        curve_manifest_path = fixture["curve_manifest_path"]
        label_manifest = fixture["label_manifest"]
        label_manifest_path = fixture["label_manifest_path"]
        spec = fixture["spec"]
        spec_path = fixture["spec_path"]
        output = fixture["output"]
        assert isinstance(shared_manifest, Path)
        assert isinstance(curve_manifest, dict)
        assert isinstance(curve_manifest_path, Path)
        assert isinstance(label_manifest, dict)
        assert isinstance(label_manifest_path, Path)
        assert isinstance(spec, dict)
        assert isinstance(spec_path, Path)
        assert isinstance(output, Path)

        score_manifest = json.loads(shared_manifest.read_text(encoding="utf-8"))
        query_item = next(
            item for item in score_manifest["items"] if item["image_id"] == "q"
        )
        query_score_path = shared_manifest.parent / query_item["file"]
        np.savez_compressed(
            query_score_path,
            # 0.253 is absent from the default grid, unlike the old 0.25
            # event.  A stale curve therefore omits a required event point.
            prob=np.full((8, 8), 0.253, dtype=np.float32),
            image_id=np.asarray("q"),
            dataset_name=np.asarray("A"),
            original_hw=np.asarray((8, 8), dtype=np.int32),
        )
        query_item["score_file_sha256"] = sha256_file(query_score_path)
        score_manifest["content_sha256"] = score_manifest_content_sha256(
            score_manifest["items"]
        )
        shared_manifest.write_text(
            json.dumps(score_manifest, sort_keys=True), encoding="utf-8"
        )
        score_manifest_sha = _sha256(shared_manifest)
        label_manifest["score_manifest_sha256"] = score_manifest_sha
        label_manifest["score_manifest_content_sha256"] = score_manifest[
            "content_sha256"
        ]
        label_manifest_path.write_text(
            json.dumps(label_manifest, sort_keys=True), encoding="utf-8"
        )
        curve_manifest["score_manifest_sha256"] = score_manifest_sha
        curve_manifest["label_manifest_sha256"] = _sha256(label_manifest_path)
        curve_manifest_path.write_text(
            json.dumps(curve_manifest, sort_keys=True), encoding="utf-8"
        )
        spec["episodes"][0]["context_score_manifest_sha256"] = score_manifest_sha
        spec["episodes"][0]["query_score_manifest_sha256"] = score_manifest_sha
        spec["episodes"][0]["curve_manifest_sha256"] = _sha256(
            curve_manifest_path
        )
        spec_path.write_text(json.dumps(spec), encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError, "thresholds differ from the threshold plan rederived"
        ):
            build_episodes_main(
                ["--spec-file", str(spec_path), "--output", str(output)]
            )

    def test_verified_curve_requires_an_event_exact_oracle_suffix(self) -> None:
        manifest = {
            "threshold_mode_requested": "adaptive",
            "event_candidate_count": 4,
            "event_threshold_count": 2,
            "event_thresholds_added": 2,
            "event_threshold_cap": 2,
            "event_thresholds_capped": True,
            "event_candidate_score_lower_bound": 0.25,
            "event_coverage_score_lower_bound": 0.75,
            "event_coverage_fraction_lower_bound": 0.5,
            "global_exact": False,
        }
        _verify_oracle_event_coverage(manifest, oracle_threshold=0.8)
        with self.assertRaisesRegex(ValueError, "complete event-exact suffix"):
            _verify_oracle_event_coverage(manifest, oracle_threshold=0.5)
        manifest["threshold_mode_requested"] = "fixed"
        with self.assertRaisesRegex(ValueError, "threshold_mode_requested"):
            _verify_oracle_event_coverage(manifest, oracle_threshold=0.8)

        no_events = {
            "threshold_mode_requested": "adaptive",
            "event_candidate_count": 0,
            "event_threshold_count": 0,
            "event_thresholds_added": 0,
            "event_threshold_cap": 4096,
            "event_thresholds_capped": False,
            "event_candidate_score_lower_bound": 0.99,
            "event_coverage_score_lower_bound": None,
            "event_coverage_fraction_lower_bound": 1.0,
            "global_exact": False,
        }
        _verify_oracle_event_coverage(no_events, oracle_threshold=0.99)
        _verify_oracle_event_coverage(no_events, oracle_threshold=1.0)
        with self.assertRaisesRegex(ValueError, "zero-event curve"):
            _verify_oracle_event_coverage(no_events, oracle_threshold=0.9)

    def test_verified_causal_window_requires_one_contiguous_manifest_window(self) -> None:
        verified, issue = _causal_window_status(
            context_manifest_sha=SHA_A,
            query_manifest_sha=SHA_A,
            context_manifest_path=self.root / "manifest.json",
            query_manifest_path=self.root / "manifest.json",
            manifest={"image_ids": ["prefix", "c0", "c1", "q0", "suffix"]},
            context_ids=("c0", "c1"),
            query_ids=("q0",),
        )
        self.assertTrue(verified)
        self.assertIsNone(issue)
        cross_manifest, _ = _causal_window_status(
            context_manifest_sha=SHA_A,
            query_manifest_sha=SHA_B,
            context_manifest_path=self.root / "manifest.json",
            query_manifest_path=self.root / "manifest.json",
            manifest={"image_ids": ["c0", "q0"]},
            context_ids=("c0",),
            query_ids=("q0",),
        )
        self.assertFalse(cross_manifest)
        non_contiguous, _ = _causal_window_status(
            context_manifest_sha=SHA_A,
            query_manifest_sha=SHA_A,
            context_manifest_path=self.root / "manifest.json",
            query_manifest_path=self.root / "manifest.json",
            manifest={"image_ids": ["c0", "interposed", "q0"]},
            context_ids=("c0",),
            query_ids=("q0",),
        )
        self.assertFalse(non_contiguous)
        copied_manifest, _ = _causal_window_status(
            context_manifest_sha=SHA_A,
            query_manifest_sha=SHA_A,
            context_manifest_path=self.root / "context" / "manifest.json",
            query_manifest_path=self.root / "query" / "manifest.json",
            manifest={"image_ids": ["c0", "q0"]},
            context_ids=("c0",),
            query_ids=("q0",),
        )
        self.assertFalse(copied_manifest)

    def test_asymmetric_loss_and_model(self) -> None:
        target = torch.tensor([0.8])
        under = asymmetric_threshold_loss(
            torch.tensor([0.7]), target, under_weight=4.0
        )
        over = asymmetric_threshold_loss(
            torch.tensor([0.9]), target, under_weight=4.0
        )
        self.assertAlmostEqual(under.item(), 4.0 * over.item(), places=5)
        model = ThresholdCalibrator(6, hidden_dim=8, dropout=0.0)
        threshold, reject = model(torch.zeros(3, 6))
        self.assertEqual(tuple(threshold.shape), (3,))
        self.assertEqual(tuple(reject.shape), (3,))

    def test_train_checkpoint_and_online_audit_contract(self) -> None:
        episodes = [
            _episode("a0", "A", 0.1, self.reference),
            _episode("a1", "A", 0.2, self.reference, pd=0.2),
            _episode("b0", "B", 0.3, self.reference),
            _episode("b1", "B", 0.4, self.reference, pd=0.2),
        ]
        episode_path = self.root / "episodes.jsonl"
        episode_path.write_text(
            "".join(json.dumps(episode.to_dict()) + "\n" for episode in episodes),
            encoding="utf-8",
        )
        deployment_reference_path = self.root / "deployment-source-reference.npz"
        deployment_reference = _write_source_reference(
            deployment_reference_path,
            self.config,
            checkpoint_sha=SHA_D,
            held_out_domains=("OUT",),
        )
        output_dir = self.root / "trained"
        result = train_calibrator_main(
            [
                "--episodes",
                str(episode_path),
                "--val-pseudo-target",
                "B",
                "--output-dir",
                str(output_dir),
                "--deployment-detector-checkpoint-sha",
                SHA_D,
                "--deployment-detector-source-domain",
                "S1",
                "--deployment-detector-source-domain",
                "S2",
                "--deployment-source-reference",
                str(deployment_reference_path),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--hidden-dim",
                "8",
                "--dropout",
                "0",
                "--device",
                "cpu",
            ]
        )
        self.assertEqual(result, 0)
        checkpoint = torch.load(
            output_dir / "calibrator.pt", map_location="cpu", weights_only=False
        )
        self.assertEqual(checkpoint["statistics_config"], self.config.to_dict())
        self.assertEqual(checkpoint["p_min"], 0.5)
        self.assertEqual(checkpoint["outer_target"], "OUT")
        self.assertEqual(checkpoint["episode_collection_sha256"], _sha256(episode_path))
        self.assertEqual(
            checkpoint["episode_collection_provenance"]["combined"]["sha256"],
            _sha256(episode_path),
        )
        self.assertEqual(checkpoint["training_config"]["lr"], 1e-3)
        self.assertEqual(checkpoint["training_config"]["seed"], 42)
        self.assertEqual(
            checkpoint["deployment_source_reference"]["sha256"],
            deployment_reference.sha256,
        )

        online_manifest = _write_score_manifest(
            self.root / "online",
            "OUT",
            SHA_D,
            ("0", "1", "2"),
            held_out_domains=("OUT",),
        )
        online_output = self.root / "online-result.json"
        online_main(
            [
                "--manifest",
                str(online_manifest),
                "--calibrator-checkpoint",
                str(output_dir / "calibrator.pt"),
                "--target-domain",
                "OUT",
                "--context-size",
                "2",
                "--pixel-budget",
                "1e-5",
                "--device",
                "cpu",
                "--output",
                str(online_output),
            ]
        )
        online = json.loads(online_output.read_text(encoding="utf-8"))
        self.assertEqual(online["mode"], "prefix_holdout")
        self.assertEqual(online["outer_target"], "OUT")
        self.assertEqual(online["calibrator_format_version"], "rc-irstd.calibrator.v3")
        self.assertEqual(
            online["episode_collection_sha256"], _sha256(episode_path)
        )
        self.assertEqual(
            online["calibrator_checkpoint_sha256"],
            _sha256(output_dir / "calibrator.pt"),
        )
        self.assertEqual(online["score_manifest_sha256"], _sha256(online_manifest))
        self.assertEqual(online["context_image_ids"], ["0", "1"])
        self.assertEqual(online["query_image_ids"], ["2"])

        broken = json.loads(online_manifest.read_text(encoding="utf-8"))
        broken["weight_sha256"] = SHA_A
        online_manifest.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "detector checkpoint"):
            online_main(
                [
                    "--manifest",
                    str(online_manifest),
                    "--calibrator-checkpoint",
                    str(output_dir / "calibrator.pt"),
                    "--target-domain",
                    "OUT",
                    "--context-size",
                    "2",
                    "--pixel-budget",
                    "1e-5",
                    "--device",
                    "cpu",
                    "--output",
                    str(online_output),
                ]
            )


if __name__ == "__main__":
    unittest.main()

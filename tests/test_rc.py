from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from model.threshold_calibrator import ThresholdCalibrator, asymmetric_threshold_loss
from rc.build_meta_episodes import main as build_episodes_main
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
    domains: tuple[str, ...] = ("S1",),
) -> SourceReference:
    centers = np.zeros((len(domains), BASE_FEATURE_DIM), dtype=np.float32)
    scale = np.ones(BASE_FEATURE_DIM, dtype=np.float32)
    np.savez_compressed(
        path,
        domains=np.asarray(domains),
        centers=centers,
        scale=scale,
        statistics_config_json=np.asarray(json.dumps(config.to_dict(), sort_keys=True)),
    )
    return load_source_reference(path, statistics_config=config)


def _provenance(target: str, status: str = "verified") -> EpisodeProvenance:
    return EpisodeProvenance(
        status=status,
        curve_file_sha256=SHA_A,
        curve_manifest_sha256=SHA_B if status == "verified" else "",
        context_score_manifest_sha256=SHA_C,
        query_score_manifest_sha256=SHA_D,
        query_score_target_dataset=target,
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
            outer_fold_id="fold-out",
            outer_target="OUT",
            detector_source_domains=("S1",),
            detector_checkpoint_sha=SHA_A,
        ),
        provenance=_provenance(pseudo_target, provenance_status),
        budgets=BudgetSpec(values=(1e-5, 0.0), active=(True, False)),
        oracle_threshold=0.7,
        oracle_pd=pd,
        oracle_pixel_risk=5e-6,
        oracle_component_risk=2.0,
        p_min=p_min,
    )


def _write_curve(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "threshold",
                "pd",
                "fa_pixel",
                "fa_component_mp",
                "num_images",
                "gt_objects",
                "total_pixels",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "threshold": 0.5,
                "pd": 0.8,
                "fa_pixel": 1e-6,
                "fa_component_mp": 1.0,
                "num_images": 1,
                "gt_objects": 1,
                "total_pixels": 64,
            }
        )


def _write_score_manifest(
    directory: Path,
    target: str,
    checkpoint_sha: str,
    image_ids: tuple[str, ...],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    items = []
    for image_id in image_ids:
        np.savez_compressed(
            directory / f"{image_id}.npz",
            prob=np.full((8, 8), 0.25, dtype=np.float32),
            mask=np.ones((8, 8), dtype=np.uint8),
            image_id=np.asarray(image_id),
        )
        items.append({"image_id": image_id, "file": f"{image_id}.npz"})
    manifest = {
        "target_dataset": target,
        "weight_sha256": checkpoint_sha,
        "detector_source_domains": ["S1"],
        "outer_fold_id": "fold-out",
        "outer_target": "OUT",
        "protocol_scope": "multi_source_protocol_candidate",
        "target_exclusion_verified": True,
        "num_images": len(items),
        "items": items,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
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
    np.savez_compressed(
        score_directory / f"{image_id}.npz",
        prob=probability.astype(np.float32),
        # A deliberately unrelated shape makes accidental mask consumption fail.
        mask=np.ones((3, 5), dtype=np.uint8),
        image_id=np.asarray(image_id),
    )
    manifest = {
        "target_dataset": target,
        "weight_sha256": SHA_A,
        "detector_source_domains": ["A", "B"],
        "held_out_domains": ["OUT"],
        "outer_fold_id": "fold-out",
        "outer_target": "OUT",
        "protocol_scope": "multi_source_protocol_candidate",
        "target_exclusion_verified": False,
        "score_type": "sigmoid_probability",
        "dataset_dir": f"../dataset-{target}",
        "num_images": 1,
        "items": [{"image_id": image_id, "file": f"{image_id}.npz"}],
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
        with self.assertRaisesRegex(ValueError, "statistics_config"):
            load_source_reference(
                self.reference_path,
                statistics_config=_statistics_config(min_score=0.2),
            )

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

    def test_episode_rejects_source_and_outer_leakage(self) -> None:
        with self.assertRaisesRegex(ValueError, "outer_target"):
            FoldContract("f", "OUT", ("OUT",), SHA_A)
        leaking_path = self.root / "leaking.npz"
        leaking = _write_source_reference(
            leaking_path, self.config, domains=("A",)
        )
        stats = extract_unlabeled_statistics(
            np.ones((1, 4, 4), dtype=np.float32),
            source_reference=leaking,
            statistics_config=self.config,
        )
        with self.assertRaisesRegex(ValueError, "source domain"):
            RCEpisode.create(
                episode_id="leak",
                pseudo_target="A",
                context_image_ids=["c"],
                query_image_ids=["q"],
                statistics=stats.vector,
                feature_names=stats.feature_names,
                statistics_config=self.config,
                source_reference=leaking,
                fold=FoldContract("f", "OUT", ("A",), SHA_A),
                provenance=_provenance("A"),
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

    def test_builder_requires_verified_three_way_curve_contract(self) -> None:
        context_manifest = _write_score_manifest(
            self.root / "context", "A", SHA_A, ("c",)
        )
        query_manifest = _write_score_manifest(
            self.root / "query", "A", SHA_A, ("q",)
        )
        curve_path = self.root / "curve.csv"
        _write_curve(curve_path)
        curve_manifest = {
            "curve_file": curve_path.name,
            "curve_sha256": _sha256(curve_path),
            "image_ids": ["q"],
            "num_images": 1,
            "score_manifest_file": "query/manifest.json",
            "score_manifest_sha256": _sha256(query_manifest),
            "score_manifest_num_images": 1,
            "target_dataset": "A",
            "detector_weight_sha256": SHA_A,
            "gt_objects": 1,
            "total_pixels": 64,
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
                    "detector_source_domains": ["S1"],
                    "detector_checkpoint_sha": SHA_A,
                    "statistics_config": self.config.to_dict(),
                    "source_reference": self.reference_path.name,
                    "context_manifest": str(context_manifest.relative_to(self.root)),
                    "context_score_manifest_sha256": _sha256(context_manifest),
                    "context_image_ids": ["c"],
                    "query_image_ids": ["q"],
                    "curve_manifest": curve_manifest_path.name,
                    "curve_manifest_sha256": _sha256(curve_manifest_path),
                    "query_score_manifest_sha256": _sha256(query_manifest),
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
        build_episodes_main(
            ["--spec-file", str(spec_path), "--output", str(output)]
        )
        episode = load_episodes(output)[0]
        self.assertEqual(episode.provenance.status, "verified")
        self.assertEqual(episode.provenance.query_score_manifest_sha256, _sha256(query_manifest))

        curve_manifest["score_manifest_sha256"] = SHA_D
        curve_manifest_path.write_text(json.dumps(curve_manifest), encoding="utf-8")
        spec["episodes"][0]["curve_manifest_sha256"] = _sha256(curve_manifest_path)
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "score manifest SHA"):
            build_episodes_main(
                ["--spec-file", str(spec_path), "--output", str(output)]
            )

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
                "--deployment-source-reference",
                str(self.reference_path),
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
        self.assertEqual(checkpoint["deployment_source_reference"]["sha256"], self.reference.sha256)

        online_manifest = _write_score_manifest(
            self.root / "online", "OUT", SHA_D, ("0", "1", "2")
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

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


class Stage1PilotArtifactTests(unittest.TestCase):
    @staticmethod
    def _git(root: Path, *arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_checkpoint_sha256_sidecar_is_written_and_detects_tampering(self):
        from scripts.train_multisource_tail import (
            save_checkpoint,
            verify_checkpoint_sha256,
        )

        args = SimpleNamespace(
            seed=42,
            outer_fold_id="fold-A",
            outer_target="TARGET",
            held_out_domains=["TARGET"],
            risk_objective="margin",
            tail_q=0.05,
            miss_q=0.25,
            object_pixel_q=0.25,
            target_background_margin=1.0,
            lambda_margin=0.2,
            tail_mode="local-peak",
            lambda_tail=0.1,
            lambda_miss=0.1,
            engineering_smoke=False,
            aaai27_pilot=False,
            epochs=1,
            resume=None,
        )
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            save_checkpoint(
                run_dir,
                model,
                optimizer,
                epoch=0,
                args=args,
                names=["A", "B"],
                detector_source_records=[],
                epoch_metrics={"epoch": 0},
                run_config_sha256="a" * 64,
                execution_fingerprint_payload={"schema_version": 1},
                run_contract_sha256="c" * 64,
            )
            checkpoint = run_dir / "checkpoint_last.pt"
            digest = verify_checkpoint_sha256(checkpoint, required=True)
            self.assertEqual(digest, self._sha256(checkpoint))
            self.assertTrue((run_dir / "checkpoint_sha256.txt").is_file())
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["run_contract_sha256"], "c" * 64)

            with checkpoint.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint_last.pt differs"):
                verify_checkpoint_sha256(checkpoint, required=True)

    def test_frozen_matrix_materializes_eight_trainer_compatible_commands(self):
        from scripts.train_multisource_tail import (
            _select_pilot_matrix_run,
            _validate_args,
            _validate_pilot_matrix_run_contract,
            parse_args,
        )
        from scripts.validate_stage1_pilot_matrix import validate_matrix

        root = Path(__file__).resolve().parents[1]
        matrix_path = root / "configs/aaai27_stage1_pilot_matrix.json"
        plan_path = root / "configs/aaai27_analysis_plan.json"
        report = validate_matrix(
            matrix_path,
            root,
            plan_path=plan_path,
            require_release_artifacts=False,
        )
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(report["run_count"], 8)
        for invocation in report["normalized_invocations"]:
            python_argv = invocation["python_argv"]
            self.assertEqual(python_argv[:2], ["-m", "scripts.train_multisource_tail"])
            with patch.object(
                sys,
                "argv",
                ["train_multisource_tail", *python_argv[2:]],
            ):
                args = parse_args()
            _validate_args(args)
            selected = _select_pilot_matrix_run(matrix, args.pilot_run_id)
            with patch.dict(os.environ, invocation["environment"]):
                _validate_pilot_matrix_run_contract(
                    args,
                    selected,
                    matrix,
                    plan,
                    root,
                )

    def test_pilot_run_contract_binds_release_config_command_and_environment(self):
        from data_ext.dataset_identity import sha256_file
        from scripts.train_multisource_tail import (
            write_aaai27_pilot_run_artifacts,
            write_json,
        )

        args = SimpleNamespace(
            aaai27_pilot=True,
            pilot_run_id="D0_leave-X_s42",
            seed=42,
            epochs=30,
            risk_objective="segmentation-only",
            lambda_margin=0.0,
            outer_fold_id="leave-X",
            outer_target="X",
            held_out_domains=["X"],
        )
        release_binding = {
            "schema_version": "rc-irstd.aaai27-pilot-release-binding.v1",
            "git": {
                "revision": "b" * 40,
                "dirty": False,
                "tracked_diff_sha256": "0" * 64,
                "untracked_manifest_sha256": "1" * 64,
                "untracked_file_count": 0,
            },
            "release_tag": "release-v1",
            "release_commit": "b" * 40,
            "analysis_plan": {"path": "plan.json", "sha256": "2" * 64},
            "pilot_matrix": {
                "path": "matrix.json",
                "sha256": "3" * 64,
                "run_id": args.pilot_run_id,
                "selected_run": {"run_id": args.pilot_run_id},
            },
            "source_archive": {
                "path": "release.zip",
                "sha256": "4" * 64,
                "checksum_file": "release.zip.sha256",
                "checksum_file_sha256": "5" * 64,
            },
        }
        execution = {"schema_version": 1, "git": release_binding["git"]}
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            sys,
            "argv",
            ["train_multisource_tail", "--aaai27-pilot", "--seed", "42"],
        ):
            run_dir = Path(temporary)
            write_json(run_dir / "config.json", {"frozen": True})
            config_sha = sha256_file(run_dir / "config.json")
            contract_sha = write_aaai27_pilot_run_artifacts(
                run_dir,
                args,
                ["A", "B"],
                [
                    {
                        "source_name": "A",
                        "dataset_identity_sha256": "6" * 64,
                        "split_sha256": "7" * 64,
                        "ordered_sample_ids_sha256": "8" * 64,
                        "split_image_artifact_sha256": "9" * 64,
                        "training_artifact_sha256": "a" * 64,
                        "num_samples": 10,
                    }
                ],
                config_sha,
                release_binding,
                execution,
            )
            contract_path = run_dir / "run_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))

            self.assertEqual(contract_sha, sha256_file(contract_path))
            self.assertEqual(contract["run_config"]["sha256"], config_sha)
            self.assertEqual(contract["release"], release_binding)
            self.assertFalse(contract["claim_bearing"])
            self.assertEqual(contract["environment"], execution)
            self.assertEqual(
                (run_dir / "command.txt").read_text(encoding="utf-8").strip(),
                contract["command"],
            )
            self.assertIn("clean=true", (run_dir / "git_status.txt").read_text())
            for name in (
                "git_commit.txt",
                "git_tag.txt",
                "source_archive_sha256.txt",
                "analysis_plan_sha256.txt",
                "pilot_matrix_sha256.txt",
                "environment.json",
            ):
                self.assertTrue((run_dir / name).is_file(), name)

    def test_release_validation_rehashes_and_reconstructs_git_archive(self):
        from scripts.train_multisource_tail import validate_aaai27_pilot_release

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            self._git(root, "init", "-q")
            self._git(root, "config", "user.email", "test@example.invalid")
            self._git(root, "config", "user.name", "Artifact Test")

            for relative in (
                "datasets/A",
                "datasets/B",
                "splits/A",
                "splits/B",
                "outputs/stage1",
                "configs",
                "outputs/release",
            ):
                (root / relative).mkdir(parents=True, exist_ok=True)
            for relative in (
                "splits/A/detector_fit.txt",
                "splits/B/detector_fit.txt",
                "splits/A/detector_diagnostic.txt",
                "splits/B/detector_diagnostic.txt",
            ):
                (root / relative).write_text("sample\n", encoding="utf-8")
            stage1_config_path = root / "configs/stage1.json"
            stage1_config_path.write_text('{"frozen":true}\n', encoding="utf-8")
            manifest_path = root / "splits/manifest.json"
            manifest_path.write_text('{"frozen":true}\n', encoding="utf-8")

            matrix = {
                "schema_version": "rc-irstd.aaai27-stage1-pilot-matrix.v1",
                "contains_observed_results": False,
                "analysis_plan_binding": {
                    "path": "configs/plan.json",
                    "schema_version": "rc-irstd.aaai27-analysis-plan.v1",
                    "allowed_plan_statuses": ["frozen_stage1_pilot_authorized"],
                    "stage1_config": {
                        "path": "configs/stage1.json",
                        "sha256": self._sha256(stage1_config_path),
                    },
                    "split_manifest": {
                        "path": "splits/manifest.json",
                        "sha256": self._sha256(manifest_path),
                    },
                },
                "release_contract": {
                    "tag": "release-v1",
                    "source_archive": "outputs/release/release.zip",
                    "source_archive_sha256_file": (
                        "outputs/release/release.zip.sha256"
                    ),
                },
                "protocol": {
                    "seed": 42,
                    "epochs": 30,
                    "checkpoint_selection": (
                        "fixed_last_no_test_or_target_validation"
                    ),
                    "diagnostics_select_checkpoint": False,
                    "deterministic": True,
                    "optimizer": "Adagrad",
                    "learning_rate": 0.05,
                    "warm_epoch": 5,
                    "risk_warmup_epochs": 5,
                    "risk_ramp_epochs": 10,
                    "base_size": 256,
                    "crop_size": 256,
                    "batch_per_domain": 3,
                    "num_workers": 4,
                    "epoch_steps_mode": "full_longest_domain",
                    "tail_mode": "local-peak",
                    "lambda_tail": 0.1,
                    "lambda_miss": 0.1,
                    "target_background_margin": 1.0,
                    "tail_q": 0.05,
                    "miss_q": 0.25,
                    "object_pixel_q": 0.25,
                    "tail_gamma": 10.0,
                    "peak_kernel_size": 5,
                    "peak_min_score": 0.05,
                    "plateau_atol": 0.0,
                    "grad_clip_norm": 0.0,
                    "variants": {
                        "D0": {
                            "risk_objective": "segmentation-only",
                            "lambda_margin": 0.0,
                            "exclusion_radius": 2,
                        }
                    },
                },
                "scheduling": {
                    "phases": [
                        {
                            "phase_id": "P0",
                            "concurrent_run_ids": ["D0_leave-X_s42"],
                        }
                    ]
                },
                "runs": [
                    {
                        "run_id": "D0_leave-X_s42",
                        "phase": "P0",
                        "experiment_scope": "single_seed_stage1_gate",
                        "variant": "D0",
                        "seed": 42,
                        "epochs": 30,
                        "fixed_last": True,
                        "sources": ["A", "B"],
                        "source_dirs": ["datasets/A", "datasets/B"],
                        "source_split_files": [
                            "splits/A/detector_fit.txt",
                            "splits/B/detector_fit.txt",
                        ],
                        "outer_fold_id": "leave-X",
                        "outer_target": "X",
                        "held_out": ["X"],
                        "primary_diagnostic_domains": ["A", "B"],
                        "evaluation_diagnostic_domains": ["A", "B"],
                        "evaluation_diagnostic_files": [
                            "splits/A/detector_diagnostic.txt",
                            "splits/B/detector_diagnostic.txt",
                        ],
                        "gpu_visible_devices": [0],
                        "data_parallel": False,
                        "output_dir": "outputs/stage1/D0_leave-X_s42",
                    }
                ],
            }
            matrix_path = root / "configs/matrix.json"
            matrix_path.write_text(
                json.dumps(matrix, sort_keys=True) + "\n", encoding="utf-8"
            )
            plan = {
                "schema_version": "rc-irstd.aaai27-analysis-plan.v1",
                "plan_status": "frozen_stage1_pilot_authorized",
                "contains_observed_results": False,
                "authorization": {
                    "gate_minus_1": True,
                    "stage1_development_comparisons": True,
                    "official_test_model_evaluation": False,
                    "paper_performance_claims": False,
                },
                "hash_contracts": {
                    "stage1_config": matrix["analysis_plan_binding"]["stage1_config"],
                    "official_train_split_manifest": matrix[
                        "analysis_plan_binding"
                    ]["split_manifest"],
                    "stage1_pilot_matrix": {
                        "path": "configs/matrix.json",
                        "sha256": self._sha256(matrix_path),
                    },
                },
                "stage1_contract": {
                    "single_seed_pilot_epochs": 30,
                    "single_seed_pilot_seed": 42,
                    "common_training": {
                        "learning_rate": 0.05,
                        "warm_epoch": 5,
                        "risk_warmup_epochs": 5,
                        "risk_ramp_epochs": 10,
                        "base_size": 256,
                        "crop_size": 256,
                        "deterministic": True,
                    },
                    "gpu_protocol": {"batch_per_domain": 3},
                    "variants": {
                        "D0": {
                            "risk_objective": "segmentation-only",
                            "lambda_margin": 0.0,
                        }
                    },
                },
            }
            plan_path = root / "configs/plan.json"
            plan_path.write_text(
                json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8"
            )
            (root / ".gitignore").write_text("outputs/\n", encoding="utf-8")
            (root / "README.md").write_text("release\n", encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "-qm", "frozen test release")
            self._git(root, "tag", "release-v1")

            archive = root / "outputs/release/release.zip"
            self._git(
                root,
                "archive",
                "--format=zip",
                f"--output={archive}",
                "release-v1",
            )
            checksum = root / "outputs/release/release.zip.sha256"
            checksum.write_text(
                f"{self._sha256(archive)}  {archive.name}\n", encoding="utf-8"
            )
            args = SimpleNamespace(
                aaai27_pilot=True,
                release_tag="release-v1",
                source_archive=str(archive),
                source_archive_sha256_file=str(checksum),
                analysis_plan="configs/plan.json",
                pilot_matrix="configs/matrix.json",
                pilot_run_id="D0_leave-X_s42",
                run_name="D0_leave-X_s42",
                save_dir=str(root / "outputs/stage1"),
                resume=None,
                risk_objective="segmentation-only",
                lambda_margin=0.0,
                seed=42,
                epochs=30,
                lr=0.05,
                warm_epoch=5,
                risk_warmup_epochs=5,
                risk_ramp_epochs=10,
                base_size=256,
                crop_size=256,
                batch_per_domain=3,
                deterministic=True,
                data_parallel=False,
                device="cuda",
                epoch_steps=None,
                allow_single_source_inner_smoke=False,
                num_workers=4,
                tail_mode="local-peak",
                lambda_tail=0.1,
                lambda_miss=0.1,
                target_background_margin=1.0,
                tail_q=0.05,
                miss_q=0.25,
                object_pixel_q=0.25,
                tail_gamma=10.0,
                peak_kernel_size=5,
                exclusion_radius=2,
                peak_min_score=0.05,
                plateau_atol=0.0,
                grad_clip_norm=0.0,
                source_names=["A", "B"],
                source_dirs=[str(root / "datasets/A"), str(root / "datasets/B")],
                source_split_files=[
                    str(root / "splits/A/detector_fit.txt"),
                    str(root / "splits/B/detector_fit.txt"),
                ],
                outer_fold_id="leave-X",
                outer_target="X",
                held_out_domains=["X"],
            )
            plan_sha = self._sha256(plan_path)
            audit = {
                "status": "PASS",
                "gate_minus_1": True,
                "plan_sha256": plan_sha,
            }
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), patch(
                "scripts.validate_aaai27_analysis_plan.validate_plan",
                return_value=audit,
            ):
                binding = validate_aaai27_pilot_release(args, root)
                self.assertEqual(binding["source_archive"]["sha256"], self._sha256(archive))
                self.assertEqual(binding["pilot_matrix"]["run_id"], args.pilot_run_id)

                args.tail_q = 0.1
                with self.assertRaisesRegex(ValueError, "protocol.tail_q"):
                    validate_aaai27_pilot_release(args, root)
                args.tail_q = 0.05

                self._git(root, "tag", "release-alias")
                args.release_tag = "release-alias"
                with self.assertRaisesRegex(ValueError, "differ from the pilot matrix"):
                    validate_aaai27_pilot_release(args, root)
                args.release_tag = "release-v1"

                with archive.open("ab") as handle:
                    handle.write(b"not-the-tagged-archive")
                checksum.write_text(
                    f"{self._sha256(archive)}  {archive.name}\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "exact git archive"):
                    validate_aaai27_pilot_release(args, root)


if __name__ == "__main__":
    unittest.main()

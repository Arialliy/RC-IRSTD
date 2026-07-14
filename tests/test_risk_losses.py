import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from data_ext.balanced_domain_loader import BalancedDomainLoader
from losses.hard_target_loss import hard_target_miss_loss, object_top_fraction_scores
from losses.local_peak_cvar import (
    aggregate_image_risks_by_domain,
    image_tail_risks,
    local_background_peak_scores,
    top_fraction_mean,
)
from losses.smooth_worst_domain import smooth_max
from losses.schedules import linear_risk_weight
from losses.target_background_margin import (
    background_local_peak_mask,
    domain_tail_separation_loss,
    domain_target_background_margin_risks,
    image_target_background_margin_risks,
    legacy_image_margin_loss,
)


class RiskLossTests(unittest.TestCase):
    def test_proposed_cli_defaults_match_reference_aligned_strict_config(self):
        from scripts.train_multisource_tail import parse_args

        with patch.object(
            sys,
            "argv",
            ["train_multisource_tail", "--source-dirs", "source-a", "source-b"],
        ):
            args = parse_args()
        config = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs"
                / "aaai27_detector_tail_sep.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(args.risk_objective, "margin")
        self.assertEqual(args.lambda_margin, config["lambda_margin"])
        self.assertEqual(args.tail_q, config["background_tail_fraction"])
        self.assertEqual(args.miss_q, config["hard_object_fraction"])
        self.assertEqual(args.object_pixel_q, config["object_top_pixel_fraction"])
        self.assertEqual(args.peak_kernel_size, config["peak_kernel_size"])
        self.assertEqual(args.exclusion_radius, config["gt_exclusion_radius"])
        self.assertEqual(args.risk_warmup_epochs, 5)

    def test_engineering_smoke_scope_cannot_be_claim_candidate(self):
        from scripts.train_multisource_tail import protocol_scope

        args = SimpleNamespace(engineering_smoke=True)
        self.assertEqual(
            protocol_scope(args, ["source-a", "source-b"]),
            "engineering_smoke_not_paper_evidence",
        )

    def test_non_smoke_training_requires_outer_identity(self):
        from scripts.train_multisource_tail import validate_fold_identity

        with self.assertRaisesRegex(ValueError, "non-smoke detector training"):
            validate_fold_identity(
                SimpleNamespace(
                    engineering_smoke=False,
                    outer_fold_id=None,
                    outer_target=None,
                )
            )
        validate_fold_identity(
            SimpleNamespace(
                engineering_smoke=True,
                outer_fold_id=None,
                outer_target=None,
            )
        )
        with self.assertRaisesRegex(ValueError, "supplied together"):
            validate_fold_identity(
                SimpleNamespace(
                    engineering_smoke=True,
                    outer_fold_id="outer-a",
                    outer_target=None,
                )
            )

    def test_git_fingerprint_is_independent_of_calling_directory(self):
        from scripts.train_multisource_tail import _git_state

        expected = _git_state()
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                actual = _git_state()
            finally:
                os.chdir(previous)
        self.assertEqual(actual, expected)

    def test_installed_source_fingerprint_is_content_addressed(self):
        from scripts.train_multisource_tail import _installed_source_state

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "scripts"
            source.mkdir()
            module = source / "train_multisource_tail.py"
            module.write_text("value = 1\n", encoding="utf-8")
            first = _installed_source_state(root)
            module.write_text("value = 2\n", encoding="utf-8")
            second = _installed_source_state(root)
        self.assertEqual(first["mode"], "installed_source_tree_sha256")
        self.assertEqual(first["source_file_count"], 1)
        self.assertNotEqual(first["source_tree_sha256"], second["source_tree_sha256"])

    def test_nested_package_does_not_bind_host_git_repository(self):
        from scripts.train_multisource_tail import _git_state

        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            dir=repository_root,
            prefix="fingerprint-installed-package-",
        ) as temporary:
            package_root = Path(temporary)
            scripts_root = package_root / "scripts"
            scripts_root.mkdir()
            (scripts_root / "train_multisource_tail.py").write_text(
                "value = 1\n",
                encoding="utf-8",
            )
            state = _git_state(package_root)
        self.assertEqual(state["mode"], "installed_source_tree_sha256")
        self.assertIsNone(state["revision"])

    def test_engineering_smoke_requires_explicit_small_active_risk_run(self):
        from scripts.train_multisource_tail import _validate_args, parse_args

        with patch.object(
            sys,
            "argv",
            ["train_multisource_tail", "--source-dirs", "source-a", "source-b"],
        ):
            args = parse_args()
        args.engineering_smoke = True
        with self.assertRaisesRegex(ValueError, "explicit --epoch-steps"):
            _validate_args(args)
        args.epoch_steps = 1
        args.epochs = 1
        args.risk_warmup_epochs = 0
        args.risk_ramp_epochs = 0
        _validate_args(args)

    def test_risk_terms_have_zero_warmup_and_linear_ramp(self):
        self.assertEqual(linear_risk_weight(0, 5, 10), 0.0)
        self.assertEqual(linear_risk_weight(4, 5, 10), 0.0)
        self.assertAlmostEqual(linear_risk_weight(5, 5, 10), 0.1)
        self.assertAlmostEqual(linear_risk_weight(9, 5, 10), 0.5)
        self.assertEqual(linear_risk_weight(14, 5, 10), 1.0)
        self.assertEqual(linear_risk_weight(20, 5, 10), 1.0)
        self.assertEqual(linear_risk_weight(5, 5, 0), 1.0)

    def test_warmup_supervises_fused_and_all_auxiliary_heads(self):
        from losses.sls import SLSIoULoss
        from scripts.train_multisource_tail import multiscale_sls_loss

        masks = torch.zeros((2, 1, 8, 8))
        masks[:, :, 3:5, 3:5] = 1.0
        final = torch.zeros((2, 1, 8, 8), requires_grad=True)
        auxiliaries = [
            torch.zeros((2, 1, 8, 8), requires_grad=True),
            torch.zeros((2, 1, 4, 4), requires_grad=True),
        ]
        loss = multiscale_sls_loss(
            SLSIoULoss(),
            final,
            auxiliaries,
            masks,
            warm_epoch=5,
            epoch=0,
        )
        loss.backward()
        self.assertGreater(torch.count_nonzero(final.grad).item(), 0)
        for prediction in auxiliaries:
            self.assertGreater(torch.count_nonzero(prediction.grad).item(), 0)

    def test_top_fraction_mean_uses_ceiling_and_largest_values(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(top_fraction_mean(values, 0.26).item(), 3.5)

    def test_local_peak_nms_reduces_constant_plateau_to_one_peak(self):
        logits = torch.full((1, 1, 7, 7), 2.0, requires_grad=True)
        masks = torch.zeros_like(logits)
        peaks = local_background_peak_scores(
            logits,
            masks,
            kernel_size=3,
            min_score=0.05,
        )
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0].numel(), 1)
        self.assertAlmostEqual(
            peaks[0].item(),
            torch.sigmoid(torch.tensor(2.0)).item(),
        )

    def test_min_score_can_produce_graph_connected_zero(self):
        logits = torch.full((2, 1, 5, 5), -10.0, requires_grad=True)
        masks = torch.zeros_like(logits)
        risks = image_tail_risks(logits, masks, q=0.1, min_score=0.05)
        self.assertTrue(torch.equal(risks, torch.zeros_like(risks)))
        risks.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_local_peak_nms_keeps_separated_peaks(self):
        logits = torch.full((1, 1, 9, 9), -10.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 2, 2] = 4.0
            logits[0, 0, 6, 6] = 3.0
        peaks = local_background_peak_scores(
            logits,
            torch.zeros_like(logits),
            min_score=0.05,
        )
        self.assertEqual(peaks[0].numel(), 2)
        peaks[0].sum().backward()
        self.assertGreater(logits.grad[0, 0, 2, 2].item(), 0.0)
        self.assertGreater(logits.grad[0, 0, 6, 6].item(), 0.0)

    def test_domain_aggregation_is_image_first_mean(self):
        image_risks = torch.tensor([0.2, 0.8, 0.4], requires_grad=True)
        domain_ids = torch.tensor([5, 5, 9])
        risks, ids = aggregate_image_risks_by_domain(
            image_risks,
            domain_ids,
            return_domain_ids=True,
        )
        self.assertEqual(ids.tolist(), [5, 9])
        self.assertTrue(torch.allclose(risks, torch.tensor([0.5, 0.4])))

    def test_smooth_max_is_normalized_by_number_of_domains(self):
        equal = torch.tensor([0.7, 0.7, 0.7], requires_grad=True)
        result = smooth_max(equal, gamma=10.0)
        self.assertAlmostEqual(result.item(), 0.7, places=6)
        result.backward()
        self.assertTrue(torch.allclose(equal.grad, torch.full_like(equal, 1.0 / 3.0)))

    def test_diagonal_gt_pixels_form_one_eight_connected_object(self):
        logits = torch.zeros((1, 1, 5, 5), requires_grad=True)
        masks = torch.zeros_like(logits)
        masks[0, 0, 1, 1] = 1.0
        masks[0, 0, 2, 2] = 1.0
        scores = object_top_fraction_scores(
            logits,
            masks,
            object_pixel_fraction=1.0,
        )
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0].numel(), 1)
        self.assertAlmostEqual(scores[0].item(), 0.5)

    def test_hard_miss_cvar_selects_the_worst_object(self):
        logits = torch.full((1, 1, 6, 6), -8.0, requires_grad=True)
        masks = torch.zeros_like(logits)
        masks[0, 0, 1, 1] = 1.0
        masks[0, 0, 4, 4] = 1.0
        with torch.no_grad():
            logits[0, 0, 1, 1] = 4.0
            logits[0, 0, 4, 4] = -2.0

        loss = hard_target_miss_loss(
            logits,
            masks,
            q=0.5,
            object_pixel_fraction=1.0,
        )
        expected = 1.0 - torch.sigmoid(torch.tensor(-2.0)).item()
        self.assertAlmostEqual(loss.item(), expected)
        loss.backward()
        self.assertLess(logits.grad[0, 0, 4, 4].item(), 0.0)
        self.assertEqual(logits.grad[0, 0, 1, 1].item(), 0.0)

    def test_target_free_hard_miss_is_graph_connected_zero(self):
        logits = torch.randn((2, 1, 5, 5), requires_grad=True)
        masks = torch.zeros_like(logits)
        loss = hard_target_miss_loss(logits, masks)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    @staticmethod
    def _margin_example(
        background_logits,
        target_logits,
    ):
        count = len(background_logits)
        logits = torch.full((count, 1, 7, 7), -12.0)
        masks = torch.zeros_like(logits)
        for index, (background, target) in enumerate(
            zip(background_logits, target_logits)
        ):
            masks[index, 0, 1, 1] = 1.0
            logits[index, 0, 1, 1] = target
            logits[index, 0, 5, 5] = background
        return logits, masks

    def test_logit_margin_is_invariant_to_common_global_shift(self):
        logits, masks = self._margin_example([1.0, -1.0], [0.0, 2.0])
        baseline = image_target_background_margin_risks(
            logits,
            masks,
            background_q=0.01,
            target_q=1.0,
            object_pixel_fraction=1.0,
            margin=1.0,
        )
        shifted = image_target_background_margin_risks(
            logits + 7.0,
            masks,
            background_q=0.01,
            target_q=1.0,
            object_pixel_fraction=1.0,
            margin=1.0,
        )
        self.assertTrue(torch.equal(baseline, shifted))

    def test_margin_violation_pushes_target_up_and_background_peak_down(self):
        logits, masks = self._margin_example([2.0], [-2.0])
        logits.requires_grad_()
        risk = image_target_background_margin_risks(
            logits,
            masks,
            background_q=0.01,
            target_q=1.0,
            object_pixel_fraction=1.0,
            margin=1.0,
        ).sum()
        self.assertAlmostEqual(risk.item(), 5.0)
        risk.backward()
        self.assertGreater(logits.grad[0, 0, 5, 5].item(), 0.0)
        self.assertLess(logits.grad[0, 0, 1, 1].item(), 0.0)
        self.assertAlmostEqual(logits.grad.sum().item(), 0.0)

    def test_margin_reduces_image_first_then_balances_domains(self):
        logits, masks = self._margin_example([1.0, 3.0, 7.0], [0.0, 0.0, 0.0])
        domain_risks, represented_ids = domain_target_background_margin_risks(
            logits,
            masks,
            torch.tensor([5, 5, 9]),
            background_q=0.01,
            target_q=1.0,
            object_pixel_fraction=1.0,
            margin=0.0,
            return_domain_ids=True,
        )
        self.assertEqual(represented_ids.tolist(), [5, 9])
        self.assertTrue(torch.equal(domain_risks, torch.tensor([2.0, 7.0])))

    def test_margin_empty_target_or_background_is_graph_connected_zero(self):
        logits = torch.randn((2, 1, 5, 5), requires_grad=True)
        masks = torch.zeros_like(logits)
        # Image 0 has background but no target. Image 1 has one all-image
        # target component but no background candidate.
        masks[1] = 1.0
        risks = image_target_background_margin_risks(logits, masks)
        self.assertTrue(torch.equal(risks, torch.zeros_like(risks)))
        risks.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_final_domain_margin_uses_target_free_images_and_is_shift_invariant(self):
        logits = torch.full((4, 1, 9, 9), -12.0, requires_grad=True)
        masks = torch.zeros_like(logits)
        masks[0, 0, 4, 4] = 1.0
        masks[2, 0, 2, 2] = 1.0
        with torch.no_grad():
            # Difficult targets keep both domain hinges active so gradients
            # expose the no-target images' background contribution.
            logits[0, 0, 4, 4] = -8.0
            logits[2, 0, 2, 2] = -8.0
            # Images 1 and 3 contain no target, but their peaks must contribute
            # to the corresponding domain background summaries.
            logits[1, 0, 7, 7] = 2.0
            logits[3, 0, 6, 6] = 1.5
        domain_ids = torch.tensor([0, 0, 1, 1])
        first = domain_tail_separation_loss(
            logits,
            masks,
            domain_ids,
            margin=1.0,
            background_tail_fraction=0.5,
            object_top_fraction=1.0,
            hard_object_fraction=1.0,
            peak_kernel_size=3,
            exclusion_radius=1,
        )
        shifted = domain_tail_separation_loss(
            logits + 7.0,
            masks,
            domain_ids,
            margin=1.0,
            background_tail_fraction=0.5,
            object_top_fraction=1.0,
            hard_object_fraction=1.0,
            peak_kernel_size=3,
            exclusion_radius=1,
        )
        self.assertTrue(torch.allclose(first.loss, shifted.loss, atol=1e-6))
        self.assertGreaterEqual(first.image_background_tail[1].item(), 1.9)
        self.assertGreaterEqual(first.image_background_tail[3].item(), 1.4)
        self.assertEqual(first.valid_domain_mask.tolist(), [True, True])
        first.loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[1, 0, 7, 7].item(), 0.0)

    def test_D1_D2_D3_share_forward_hinge_but_route_single_tail_gradients(self):
        base_logits, masks = self._margin_example([2.0], [-2.0])
        losses = {}
        gradients = {}
        for mode in ("background", "target", "both"):
            logits = base_logits.clone().requires_grad_()
            output = domain_tail_separation_loss(
                logits,
                masks,
                torch.tensor([0]),
                margin=1.0,
                background_tail_fraction=0.01,
                object_top_fraction=1.0,
                hard_object_fraction=1.0,
                peak_kernel_size=3,
                exclusion_radius=0,
                trainable_tail=mode,
            )
            losses[mode] = output.loss.item()
            output.loss.backward()
            gradients[mode] = (
                logits.grad[0, 0, 5, 5].item(),
                logits.grad[0, 0, 1, 1].item(),
            )

        self.assertAlmostEqual(losses["background"], losses["target"])
        self.assertAlmostEqual(losses["target"], losses["both"])
        self.assertGreater(gradients["background"][0], 0.0)
        self.assertEqual(gradients["background"][1], 0.0)
        self.assertEqual(gradients["target"][0], 0.0)
        self.assertLess(gradients["target"][1], 0.0)
        self.assertGreater(gradients["both"][0], 0.0)
        self.assertLess(gradients["both"][1], 0.0)

    def test_invalid_trainable_tail_is_rejected(self):
        logits, masks = self._margin_example([2.0], [-2.0])
        with self.assertRaisesRegex(ValueError, "trainable_tail"):
            domain_tail_separation_loss(
                logits,
                masks,
                torch.tensor([0]),
                trainable_tail="unknown",
            )

    def test_final_margin_collapses_constant_plateau_and_dilates_gt(self):
        logits = torch.zeros((1, 1, 11, 11))
        masks = torch.zeros_like(logits)
        peaks, valid_background = background_local_peak_mask(
            logits,
            masks,
            kernel_size=3,
            exclusion_radius=0,
        )
        self.assertEqual(int(peaks.sum()), 1)
        self.assertTrue(valid_background.all())

        masks[0, 0, 5, 5] = 1.0
        with torch.no_grad():
            logits.fill_(-4.0)
            logits[0, 0, 5, 6] = 9.0
            logits[0, 0, 1, 1] = 3.0
        peaks, valid_background = background_local_peak_mask(
            logits,
            masks,
            kernel_size=3,
            exclusion_radius=1,
        )
        self.assertFalse(valid_background[0, 0, 5, 6])
        self.assertFalse(peaks[0, 0, 5, 6])
        self.assertTrue(peaks[0, 0, 1, 1])

    def test_final_margin_forms_domain_tails_before_hinge(self):
        logits, masks = self._margin_example([3.0, -3.0], [2.0, 0.0])
        domain_ids = torch.tensor([4, 4])
        legacy = domain_target_background_margin_risks(
            logits,
            masks,
            domain_ids,
            background_q=0.01,
            target_q=1.0,
            object_pixel_fraction=1.0,
            margin=0.0,
        )
        final = domain_tail_separation_loss(
            logits,
            masks,
            domain_ids,
            margin=0.0,
            background_tail_fraction=0.01,
            object_top_fraction=1.0,
            hard_object_fraction=1.0,
            peak_kernel_size=3,
            exclusion_radius=0,
        )
        self.assertAlmostEqual(legacy.item(), 0.5)
        self.assertAlmostEqual(final.domain_background_tail.item(), 0.0)
        self.assertAlmostEqual(final.domain_target_tail.item(), 1.0)
        self.assertAlmostEqual(final.domain_raw_gap.item(), 1.0)
        self.assertAlmostEqual(final.domain_violation.item(), 0.0)
        self.assertAlmostEqual(final.domain_gap.item(), 0.0)
        self.assertAlmostEqual(final.loss.item(), 0.0)

    def test_legacy_image_margin_is_trainable_and_matches_preserved_api(self):
        logits, masks = self._margin_example([3.0, -3.0], [2.0, 0.0])
        logits.requires_grad_()
        domain_ids = torch.tensor([4, 4])
        preserved = domain_target_background_margin_risks(
            logits,
            masks,
            domain_ids,
            background_q=0.01,
            target_q=1.0,
            object_pixel_fraction=1.0,
            margin=0.0,
        )
        output = legacy_image_margin_loss(
            logits,
            masks,
            domain_ids,
            margin=0.0,
            background_tail_fraction=0.01,
            object_top_fraction=1.0,
            hard_object_fraction=1.0,
            peak_kernel_size=3,
        )
        self.assertTrue(torch.allclose(output.domain_violation, preserved))
        self.assertAlmostEqual(output.loss.item(), 0.5)
        self.assertAlmostEqual(output.domain_background_tail.item(), 0.0)
        self.assertAlmostEqual(output.domain_target_tail.item(), 1.0)
        self.assertAlmostEqual(output.domain_raw_gap.item(), 1.0)
        self.assertGreaterEqual(output.domain_background_candidate_mean.item(), 1.0)
        self.assertEqual(output.domain_object_count.tolist(), [2])
        output.loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(torch.count_nonzero(logits.grad).item(), 0)

    def test_final_margin_excludes_target_free_domain_without_fake_positive(self):
        logits = torch.randn((2, 1, 7, 7), requires_grad=True)
        masks = torch.zeros_like(logits)
        output = domain_tail_separation_loss(
            logits,
            masks,
            torch.tensor([3, 7]),
        )
        self.assertEqual(output.valid_domain_mask.tolist(), [False, False])
        self.assertEqual(output.loss.item(), 0.0)
        output.loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_margin_capability_contract_is_persisted_in_detector_checkpoint(self):
        from scripts.train_multisource_tail import save_checkpoint

        args = SimpleNamespace(
            seed=42,
            outer_fold_id="fold-A",
            outer_target="TARGET",
            held_out_domains=["TARGET"],
            risk_objective="margin",
            tail_q=0.01,
            miss_q=0.2,
            object_pixel_q=0.25,
            target_background_margin=1.5,
            lambda_margin=0.3,
            tail_mode="local-peak",
            lambda_tail=0.1,
            lambda_miss=0.1,
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
                names=["SOURCE-A", "SOURCE-B"],
                detector_source_records=[],
                epoch_metrics={"loss_margin": 1.0},
                run_config_sha256="a" * 64,
            )
            checkpoint = torch.load(
                run_dir / "checkpoint_last.pt",
                map_location="cpu",
                weights_only=False,
            )
        self.assertEqual(checkpoint["risk_objective"], "margin")
        capability = checkpoint["detector_capability_contract"]["risk_objective"]
        self.assertEqual(
            capability["name"],
            "domain_target_background_tail_separation",
        )
        self.assertEqual(
            capability["hinge_level"],
            "domain_after_two_tail_aggregation",
        )
        self.assertEqual(capability["score_space"], "logit_difference")
        self.assertTrue(capability["common_logit_shift_invariant"])
        self.assertEqual(capability["margin_logit"], 1.5)
        self.assertEqual(checkpoint["format_version"], "rc-irstd.detector.v2")
        self.assertTrue(
            checkpoint["detector_capability_contract"]["resume"]["supported"]
        )
        self.assertIn(
            "raw_gap_logit",
            checkpoint["detector_capability_contract"]["training_diagnostics"][
                "domain_tail_fields"
            ],
        )

    def test_legacy_objective_contract_is_explicit_and_not_separate(self):
        from scripts.train_multisource_tail import risk_objective_contract

        common = dict(
            tail_q=0.01,
            miss_q=0.2,
            object_pixel_q=0.25,
            target_background_margin=1.0,
            lambda_margin=0.1,
            tail_mode="local-peak",
            lambda_tail=0.1,
            lambda_miss=0.1,
            peak_kernel_size=3,
            plateau_atol=0.0,
            tail_gamma=10.0,
        )
        legacy = risk_objective_contract(
            SimpleNamespace(risk_objective="legacy-image-margin", **common)
        )
        separate = risk_objective_contract(
            SimpleNamespace(risk_objective="separate", **common)
        )
        self.assertEqual(
            legacy["name"], "legacy_image_paired_target_background_margin"
        )
        self.assertEqual(legacy["hinge_level"], "image_before_domain_aggregation")
        self.assertEqual(separate["name"], "separate_background_tail_plus_hard_miss")
        self.assertNotEqual(legacy["name"], separate["name"])

    def test_frozen_D0_D3_contract_identities_are_distinct(self):
        from scripts.train_multisource_tail import risk_objective_contract

        common = dict(
            tail_q=0.05,
            miss_q=0.25,
            object_pixel_q=0.25,
            target_background_margin=1.0,
            lambda_margin=0.2,
            tail_mode="local-peak",
            lambda_tail=0.1,
            lambda_miss=0.1,
            peak_kernel_size=5,
            plateau_atol=0.0,
            tail_gamma=10.0,
            exclusion_radius=2,
        )
        objectives = {
            "D0": ("segmentation-only", 0.0),
            "D1": ("margin-background-only", 0.2),
            "D2": ("margin-target-only", 0.2),
            "D3": ("margin", 0.2),
        }
        contracts = {}
        for variant, (objective, weight) in objectives.items():
            args = SimpleNamespace(
                risk_objective=objective,
                **{**common, "lambda_margin": weight},
            )
            contracts[variant] = risk_objective_contract(args)
            self.assertEqual(contracts[variant]["stage1_variant"], variant)
        self.assertEqual(contracts["D1"]["trainable_tail"], "background")
        self.assertEqual(contracts["D1"]["stop_gradient_tail"], "target")
        self.assertEqual(contracts["D2"]["trainable_tail"], "target")
        self.assertEqual(contracts["D2"]["stop_gradient_tail"], "background")
        self.assertEqual(contracts["D3"]["trainable_tail"], "both")
        self.assertFalse(contracts["D0"]["auxiliary_risk_gradient"])

    @staticmethod
    def _detector_training_args(**overrides):
        values = dict(
            seed=42,
            outer_fold_id="fold-A",
            outer_target="TARGET",
            held_out_domains=["TARGET"],
            risk_objective="legacy-image-margin",
            tail_q=0.01,
            miss_q=0.2,
            object_pixel_q=0.25,
            target_background_margin=1.0,
            lambda_margin=0.1,
            tail_mode="local-peak",
            lambda_tail=0.1,
            lambda_miss=0.1,
            peak_kernel_size=3,
            plateau_atol=0.0,
            tail_gamma=10.0,
            epochs=1,
            resume=None,
            source_dirs=["source-a", "source-b"],
            source_names=["A", "B"],
            source_split_files=None,
            batch_per_domain=1,
            epoch_steps=1,
            risk_warmup_epochs=0,
            risk_ramp_epochs=0,
            base_size=16,
            crop_size=16,
            num_workers=0,
            peak_min_score=0.05,
            exclusion_radius=2,
            grad_clip_norm=0.0,
            lr=0.1,
            warm_epoch=0,
            device="cpu",
            data_parallel=False,
            deterministic=True,
            save_dir="runs",
            run_name="resume-test",
            allow_single_source_inner_smoke=False,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_fixed_last_resume_restores_state_and_rejects_contract_drift(self):
        from scripts.train_multisource_tail import (
            append_jsonl,
            load_resume_checkpoint,
            save_checkpoint,
            stage1_segmentation_loss_implementation,
            write_json,
        )
        from data_ext.dataset_identity import sha256_file

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            frozen_config = {
                "frozen": True,
                "segmentation_loss_implementation": (
                    stage1_segmentation_loss_implementation()
                ),
            }
            write_json(run_dir / "config.json", frozen_config)
            config_sha = sha256_file(run_dir / "config.json")
            args = self._detector_training_args()
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
            loss = model(torch.ones((1, 2))).sum()
            loss.backward()
            optimizer.step()
            append_jsonl(run_dir / "metrics.jsonl", {"epoch": 0, "loss": 1.0})
            save_checkpoint(
                run_dir,
                model,
                optimizer,
                epoch=0,
                args=args,
                names=["A", "B"],
                detector_source_records=[],
                epoch_metrics={"epoch": 0, "loss": 1.0},
                run_config_sha256=config_sha,
            )

            resumed_args = self._detector_training_args(
                epochs=3,
                resume=str(run_dir / "checkpoint_last.pt"),
            )
            resumed_model = torch.nn.Linear(2, 1)
            resumed_optimizer = torch.optim.Adagrad(
                resumed_model.parameters(), lr=resumed_args.lr
            )
            loaded_dir, start_epoch, loaded_sha, loaded_config = (
                load_resume_checkpoint(
                    resumed_args,
                    resumed_model,
                    resumed_optimizer,
                    ["A", "B"],
                    [],
                    torch.device("cpu"),
                )
            )
            self.assertEqual(loaded_dir, run_dir.resolve())
            self.assertEqual(start_epoch, 1)
            self.assertEqual(loaded_sha, config_sha)
            self.assertEqual(loaded_config, frozen_config)
            for expected, actual in zip(model.parameters(), resumed_model.parameters()):
                self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(resumed_optimizer.state_dict()["state"])

            drifted_args = self._detector_training_args(
                epochs=3,
                resume=str(run_dir / "checkpoint_last.pt"),
                target_background_margin=2.0,
            )
            drifted_model = torch.nn.Linear(2, 1)
            drifted_optimizer = torch.optim.Adagrad(
                drifted_model.parameters(), lr=0.1
            )
            with self.assertRaisesRegex(ValueError, "immutable training/objective"):
                load_resume_checkpoint(
                    drifted_args,
                    drifted_model,
                    drifted_optimizer,
                    ["A", "B"],
                    [],
                    torch.device("cpu"),
                )

    def test_legacy_training_logs_tail_logit_and_gradient_diagnostics(self):
        from losses.sls import SLSIoULoss
        from scripts.train_multisource_tail import train_one_epoch

        class TinyDetector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Conv2d(3, 1, kernel_size=1)

            def forward(self, images, multiscale_forward):
                logits = self.head(images)
                return [logits], logits

        class OneBatchLoader:
            domain_ids = {"A": 0, "B": 1}
            domain_names = ["A", "B"]
            last_cycle_counts = {"A": 0, "B": 0}

            def __init__(self):
                self.epoch = None
                self.batch = {
                    "image": torch.zeros((2, 3, 8, 8)),
                    "mask": torch.zeros((2, 1, 8, 8)),
                    "domain_id": torch.tensor([0, 1]),
                }
                self.batch["mask"][0, 0, 2, 2] = 1.0
                self.batch["mask"][1, 0, 5, 5] = 1.0

            def set_epoch(self, epoch):
                self.epoch = epoch

            def __len__(self):
                return 1

            def __iter__(self):
                yield self.batch

        model = TinyDetector()
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
        loader = OneBatchLoader()
        args = self._detector_training_args()
        metrics = train_one_epoch(
            model,
            optimizer,
            SLSIoULoss(),
            loader,
            torch.device("cpu"),
            args,
            epoch=0,
        )
        self.assertEqual(loader.epoch, 0)
        for name in (
            "background_tail_logit_by_domain",
            "target_tail_logit_by_domain",
            "raw_gap_logit_by_domain",
            "margin_violation_by_domain",
            "background_candidate_count_per_image_by_domain",
            "object_count_per_batch_by_domain",
        ):
            self.assertEqual(set(metrics[name]), {"A", "B"})
        for name in (
            "logit_mean",
            "logit_std",
            "logit_min",
            "logit_max",
            "logit_q001",
            "logit_q50",
            "logit_q99",
            "logit_q999",
            "max_abs_logit",
            "parameter_norm",
            "learning_rate",
        ):
            self.assertTrue(math.isfinite(metrics[name]))
        self.assertEqual(metrics["nonfinite_count"], 0)
        self.assertEqual(metrics["elapsed_steps"], 1)
        self.assertEqual(
            set(metrics["num_empty_target_images_by_domain"]), {"A", "B"}
        )
        self.assertTrue(metrics["logits_finite"])
        self.assertTrue(metrics["gradients_finite"])
        self.assertTrue(metrics["parameters_finite_after_epoch"])
        self.assertTrue(metrics["risk_gradient_checked"])
        self.assertTrue(metrics["risk_gradients_finite"])
        self.assertTrue(
            math.isfinite(metrics["risk_gradient_norm_first_active_step"])
        )
        self.assertGreater(metrics["gradient_norm_mean"], 0.0)
        self.assertGreaterEqual(
            metrics["gradient_norm_max"], metrics["gradient_norm_mean"]
        )
        self.assertEqual(
            metrics["margin_diagnostics_contract"]["aggregation"],
            "image_hinge_before_domain_mean",
        )

    def test_D0_training_is_segmentation_only_with_zero_effective_risk(self):
        from losses.sls import SLSIoULoss
        from scripts.train_multisource_tail import train_one_epoch

        class TinyDetector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Conv2d(3, 1, kernel_size=1)

            def forward(self, images, multiscale_forward):
                logits = self.head(images)
                return [logits], logits

        class OneBatchLoader:
            domain_ids = {"A": 0, "B": 1}
            domain_names = ["A", "B"]
            last_cycle_counts = {"A": 0, "B": 0}

            def set_epoch(self, epoch):
                self.epoch = epoch

            def __len__(self):
                return 1

            def __iter__(self):
                masks = torch.zeros((2, 1, 8, 8))
                masks[0, 0, 2, 2] = 1.0
                masks[1, 0, 5, 5] = 1.0
                yield {
                    "image": torch.zeros((2, 3, 8, 8)),
                    "mask": masks,
                    "domain_id": torch.tensor([0, 1]),
                }

        model = TinyDetector()
        args = self._detector_training_args(
            risk_objective="segmentation-only",
            lambda_margin=0.0,
        )
        metrics = train_one_epoch(
            model,
            torch.optim.Adagrad(model.parameters(), lr=0.01),
            SLSIoULoss(),
            OneBatchLoader(),
            torch.device("cpu"),
            args,
            epoch=0,
        )
        self.assertEqual(metrics["stage1_variant"], "D0")
        self.assertEqual(
            metrics["segmentation_loss_implementation"]["qualified_name"],
            "losses.sls.SLSIoULoss",
        )
        self.assertAlmostEqual(metrics["loss_total"], metrics["loss_seg"])
        self.assertEqual(metrics["loss_tail_sep"], 0.0)
        self.assertEqual(metrics["effective_lambda_margin"], 0.0)
        self.assertFalse(metrics["risk_gradient_checked"])

    @staticmethod
    def _toy_dataset(length: int, offset: int) -> TensorDataset:
        values = torch.arange(
            offset,
            offset + length,
            dtype=torch.float32,
        ).reshape(-1, 1, 1, 1)
        return TensorDataset(values, torch.zeros_like(values))

    def test_balanced_loader_cycles_short_domain_and_is_seeded(self):
        datasets = {
            "long": self._toy_dataset(6, 0),
            "short": self._toy_dataset(2, 100),
        }
        loader = BalancedDomainLoader(
            datasets,
            batch_size_per_domain=2,
            seed=17,
            num_workers=0,
        )
        loader.set_epoch(3)
        first = list(loader)

        repeat = BalancedDomainLoader(
            datasets,
            batch_size_per_domain=2,
            seed=17,
            num_workers=0,
        )
        repeat.set_epoch(3)
        second = list(repeat)

        self.assertEqual(len(loader), 3)
        self.assertEqual(loader.last_cycle_counts, {"long": 0, "short": 2})
        for left, right in zip(first, second):
            self.assertEqual(left["image"].shape[0], 4)
            self.assertTrue(torch.equal(left["image"], right["image"]))
            self.assertEqual(left["domain_id"].tolist(), [0, 1, 0, 1])
            self.assertEqual(left["domain_name"], ["long", "short", "long", "short"])
            self.assertEqual(
                torch.bincount(left["domain_id"], minlength=2).tolist(),
                [2, 2],
            )

            # Simulate DataParallel's two contiguous replica chunks.  Both
            # replicas must see both domains so retained BatchNorm state is not
            # tied to whichever domain happened to be concatenated first.
            first_replica, second_replica = left["domain_id"].chunk(2)
            self.assertEqual(first_replica.tolist(), [0, 1])
            self.assertEqual(second_replica.tolist(), [0, 1])

    def test_invalid_tail_fraction_is_rejected(self):
        for fraction in (0.0, -0.1, 1.1, math.inf):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    top_fraction_mean(torch.ones(3), fraction)


if __name__ == "__main__":
    unittest.main()

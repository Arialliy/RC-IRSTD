import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
)


class RiskLossTests(unittest.TestCase):
    def test_risk_terms_have_zero_warmup_and_linear_ramp(self):
        self.assertEqual(linear_risk_weight(0, 5, 10), 0.0)
        self.assertEqual(linear_risk_weight(4, 5, 10), 0.0)
        self.assertAlmostEqual(linear_risk_weight(5, 5, 10), 0.1)
        self.assertAlmostEqual(linear_risk_weight(9, 5, 10), 0.5)
        self.assertEqual(linear_risk_weight(14, 5, 10), 1.0)
        self.assertEqual(linear_risk_weight(20, 5, 10), 1.0)
        self.assertEqual(linear_risk_weight(5, 5, 0), 1.0)

    def test_warmup_supervises_fused_and_all_auxiliary_heads(self):
        from model.loss import SLSIoULoss
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
        self.assertAlmostEqual(final.domain_gap.item(), 0.0)
        self.assertAlmostEqual(final.loss.item(), 0.0)

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

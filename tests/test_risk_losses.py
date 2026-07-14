import math
import unittest

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


class RiskLossTests(unittest.TestCase):
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

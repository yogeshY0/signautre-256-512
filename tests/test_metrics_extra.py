"""Tests for AUC, FRR threshold, bootstrap CI and report helpers."""

from __future__ import annotations

import unittest

import numpy as np

from scoring.metrics import auc, bootstrap_ci, threshold_at_frr
from training import evaluate
from training.stage3_metric_learning import SignatureRecord, source_group
from pathlib import Path


class AucTests(unittest.TestCase):
    def test_perfect_and_empty_auc(self):
        self.assertEqual(auc([0.9, 0.8], [0.2, 0.1]), 1.0)
        self.assertIsNone(auc([], [0.2]))
        self.assertEqual(auc([0.5], [0.5]), 0.5)


class FrrThresholdTests(unittest.TestCase):
    def test_meets_the_frr_budget(self):
        threshold, achieved = threshold_at_frr(
            [0.9, 0.8, 0.7, 0.6], [0.2, 0.3], 0.25
        )
        self.assertLessEqual(achieved, 0.25)
        self.assertAlmostEqual(threshold, 0.7)


class BootstrapTests(unittest.TestCase):
    def test_returns_bounds_for_multiple_signers(self):
        bounds = bootstrap_ci(
            [0.9, 0.8, 0.3],
            [0.2, 0.1, 0.7],
            ["a", "a", "b"],
            ["a", "a", "b"],
            auc,
            resamples=100,
        )
        self.assertIsNotNone(bounds)
        self.assertLessEqual(bounds[0], bounds[1])

    def test_single_signer_has_no_ci(self):
        self.assertIsNone(
            bootstrap_ci([0.9], [0.2], ["a"], ["a"], auc, resamples=100)
        )


class SourceGroupTests(unittest.TestCase):
    def test_bhsig_is_split_by_script(self):
        hindi = SignatureRecord(Path("x"), "BHSig260_hi_001", False, "BHSig260")
        bengali = SignatureRecord(Path("x"), "BHSig260_bn_001", False, "BHSig260")
        cedar = SignatureRecord(Path("x"), "CEDAR_1", False, "CEDAR")
        self.assertEqual(source_group(hindi), "BHSig260-Hindi")
        self.assertEqual(source_group(bengali), "BHSig260-Bengali")
        self.assertEqual(source_group(cedar), "CEDAR")


class PerSignerTests(unittest.TestCase):
    def test_ranks_the_writer_with_the_highest_frr(self):
        scores = {
            "deep_genuine": np.array([0.9, 0.8, 0.6, 0.55]),
            "deep_skilled": np.array([0.2, 0.9, 0.1, 0.8]),
            "deep_random": np.array([]),
        }
        signers = {
            "genuine": ["a", "a", "b", "b"],
            "skilled": ["a", "a", "b", "b"],
            "random": [],
        }
        datasets = {
            "genuine": ["X", "X", "Y", "Y"],
            "skilled": ["X", "X", "Y", "Y"],
            "random": [],
        }
        block = evaluate._per_signer_block(
            scores, signers, datasets, "deep", 0.7, 10, {"a": "X", "b": "Y"}
        )
        self.assertEqual(block["signers_evaluated"], 2)
        self.assertEqual(block["worst_frr"][0]["signer"], "b")
        self.assertEqual(block["worst_frr"][0]["group"], "Y")


class OperatingPointFrrTests(unittest.TestCase):
    def test_threshold_is_set_on_validation(self):
        val = {
            "deep_genuine": np.array([0.9, 0.8, 0.7, 0.6]),
            "deep_skilled": np.array([0.2, 0.3, 0.4, 0.5]),
            "deep_random": np.array([0.25, 0.35]),
        }
        test = {
            "deep_genuine": np.array([0.9, 0.6]),
            "deep_skilled": np.array([0.2, 0.7]),
            "deep_random": np.array([0.2, 0.7]),
        }
        block = evaluate._operating_point_frr(val, test, "deep", 0.25)
        self.assertLessEqual(block["validation_frr_achieved"], 0.25)
        self.assertIn("far_skilled_at_threshold", block["test"])
        self.assertIn("far_random_at_threshold", block["test"])


class NoRandomNegativesTests(unittest.TestCase):
    """``--max-random-queries 0`` leaves the random-forgery list empty."""

    def setUp(self):
        self.scores = {
            "deep_genuine": np.array([0.9, 0.8, 0.7, 0.6]),
            "deep_skilled": np.array([0.2, 0.3, 0.75, 0.5]),
            "deep_random": np.array([]),
        }

    def test_metrics_skip_the_random_numbers(self):
        block = evaluate._metrics(self.scores, "deep", 0.7)
        self.assertIsNone(block["eer_random"])
        self.assertIsNone(block["auc_random"])
        self.assertIsNone(block["far_random_at_threshold"])
        self.assertEqual(block["random_pairs"], 0)
        self.assertIsNotNone(block["eer_skilled"])
        self.assertIsNotNone(block["frr_at_threshold"])

    def test_far_frr_skips_the_random_rate(self):
        block = evaluate._far_frr(self.scores, "deep", 0.7)
        self.assertIsNone(block["far_random_at_threshold"])
        self.assertIsNotNone(block["far_skilled_at_threshold"])

    def test_confidence_block_skips_the_random_intervals(self):
        signers = {"genuine": ["a", "a", "b", "b"], "skilled": ["a", "a", "b", "b"], "random": []}
        datasets = {"genuine": ["X"] * 4, "skilled": ["X"] * 4, "random": []}
        block = evaluate._confidence_block(
            self.scores, signers, datasets, "deep", 20
        )
        self.assertIsNone(block["pooled"]["auc_random"])
        self.assertIsNone(block["pooled"]["eer_random"])

    def test_the_two_printed_tables_accept_missing_random_numbers(self):
        metrics = evaluate._metrics(self.scores, "deep", 0.7)
        evaluate._print_reference_sweep(
            {"3": {"validation_eer_skilled": 0.1, "validation_auc_skilled": 0.9, "test": metrics}}
        )
        evaluate._print_operating_frr(
            "deep",
            {
                "target_frr": 0.1,
                "threshold_from_validation": 0.7,
                "validation_frr_achieved": 0.1,
                "test": evaluate._far_frr(self.scores, "deep", 0.7),
            },
        )


class MetricsBlockTests(unittest.TestCase):
    def test_metrics_include_auc(self):
        scores = {
            "deep_genuine": np.array([0.9, 0.8]),
            "deep_skilled": np.array([0.2, 0.1]),
            "deep_random": np.array([0.3, 0.4]),
        }
        block = evaluate._metrics(scores, "deep", 0.7)
        self.assertEqual(block["auc_skilled"], 1.0)
        self.assertIn("auc_random", block)


if __name__ == "__main__":
    unittest.main()

"""Focused tests for the evaluation report additions.

Two behaviours decide whether fusion ships: the per-dataset verdict (fusion must improve
*every* corpus) and the calibration table (a global threshold must be compared with a
threshold fitted on each corpus's own validation split). Both are pure functions, so
they are tested without loading a checkpoint.
"""

from __future__ import annotations

import unittest

import numpy as np

from training import evaluate


def _scores(genuine: list[float], skilled: list[float], prefix: str = "deep"):
    return {
        f"{prefix}_genuine": np.asarray(genuine, dtype=np.float64),
        f"{prefix}_skilled": np.asarray(skilled, dtype=np.float64),
    }


class CalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        # DS1 is easy, DS2 is much harder: one global threshold is deliberately a poor
        # compromise, which is exactly what the table has to reveal.
        self.val = _scores([0.95, 0.90, 0.85, 0.55, 0.50, 0.45], [0.10, 0.20, 0.30, 0.50, 0.45, 0.55])
        self.test = _scores([0.92, 0.88, 0.52, 0.48], [0.15, 0.25, 0.49, 0.51])
        self.val_labels = {
            "genuine": ["DS1", "DS1", "DS1", "DS2", "DS2", "DS2"],
            "skilled": ["DS1", "DS1", "DS1", "DS2", "DS2", "DS2"],
        }
        self.test_labels = {
            "genuine": ["DS1", "DS1", "DS2", "DS2"],
            "skilled": ["DS1", "DS1", "DS2", "DS2"],
        }

    def test_uses_validation_for_both_thresholds_and_test_for_the_rates(self) -> None:
        block = evaluate._calibration_block(
            self.val, self.test, self.val_labels, self.test_labels, "deep", None
        )
        self.assertIn("global_threshold", block)
        self.assertEqual(sorted(block["datasets"]), ["DS1", "DS2"])
        for dataset in ("DS1", "DS2"):
            entry = block["datasets"][dataset]
            self.assertIn("threshold", entry)
            self.assertIn("test_at_global_threshold", entry)
            self.assertIn("test_at_dataset_threshold", entry)
            self.assertIsNotNone(entry["test_at_global_threshold"])
        self.assertNotEqual(
            block["datasets"]["DS1"]["threshold"],
            block["datasets"]["DS2"]["threshold"],
        )

    def test_adds_a_far_budget_pair_when_requested(self) -> None:
        block = evaluate._calibration_block(
            self.val, self.test, self.val_labels, self.test_labels, "deep", 0.01
        )
        self.assertIn("global_threshold_at_far", block)
        for dataset in ("DS1", "DS2"):
            entry = block["datasets"][dataset]
            self.assertIn("threshold_at_far", entry)
            self.assertIn("test_at_global_far_threshold", entry)
            self.assertIn("test_at_dataset_far_threshold", entry)

    def test_returns_empty_when_validation_has_no_scores(self) -> None:
        empty = _scores([], [])
        self.assertEqual(
            evaluate._calibration_block(empty, empty, self.val_labels, self.test_labels, "deep", None),
            {},
        )


class FusionComparisonTests(unittest.TestCase):
    @staticmethod
    def _per_dataset(deep: dict[str, float], fused: dict[str, float]):
        per_dataset = {"val": {}, "test": {}}
        for name, value in deep.items():
            per_dataset["val"][name] = {"eer_skilled": value}
            per_dataset["test"][name] = {"eer_skilled": value}
        fused_block = {"val": {}, "test": {}}
        for name, value in fused.items():
            fused_block["val"][name] = {"eer_skilled": value}
            fused_block["test"][name] = {"eer_skilled": value}
        return per_dataset, fused_block

    def test_recommends_fusion_only_when_every_dataset_improves(self) -> None:
        deep, fused = self._per_dataset(
            {"BHSig260": 0.20, "CEDAR": 0.30}, {"BHSig260": 0.15, "CEDAR": 0.25}
        )
        verdict = evaluate._fusion_comparison(deep, fused)
        self.assertTrue(verdict["helps_every_dataset"])
        self.assertEqual(verdict["recommendation"], "keep_fusion")
        self.assertAlmostEqual(verdict["datasets"]["CEDAR"]["delta_val"], -0.05)

    def test_one_worse_dataset_keeps_the_deep_model(self) -> None:
        deep, fused = self._per_dataset(
            {"BHSig260": 0.20, "CEDAR": 0.30}, {"BHSig260": 0.15, "CEDAR": 0.35}
        )
        verdict = evaluate._fusion_comparison(deep, fused)
        self.assertFalse(verdict["helps_every_dataset"])
        self.assertEqual(verdict["recommendation"], "deep_only")

    def test_empty_comparison_does_not_recommend_fusion(self) -> None:
        verdict = evaluate._fusion_comparison({"val": {}, "test": {}}, {"val": {}, "test": {}})
        self.assertFalse(verdict["helps_every_dataset"])
        self.assertEqual(verdict["recommendation"], "deep_only")


if __name__ == "__main__":
    unittest.main()

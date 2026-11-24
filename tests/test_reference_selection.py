"""Tests for medoid and quality reference selection."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scoring.reference_selection import medoid_order, select_reference_indices
from training.stage3_metric_learning import SignatureRecord


def _records(count: int = 4):
    return [
        SignatureRecord(Path(f"a_{index}.png"), "a", False, "X")
        for index in range(count)
    ]


class MedoidTests(unittest.TestCase):
    def test_one_reference_is_the_medoid(self):
        emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        self.assertEqual(medoid_order(emb, [0, 1, 2], 1), [1])

    def test_two_references_cover_a_second_cluster(self):
        emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        self.assertEqual(medoid_order(emb, [0, 1, 2], 2), [1, 2])

    def test_count_at_least_size_returns_everything(self):
        self.assertEqual(medoid_order(np.eye(3), [0, 1, 2], 5), [0, 1, 2])

    def test_medoid_requires_embeddings(self):
        with self.assertRaisesRegex(ValueError, "requires embeddings"):
            select_reference_indices(_records(), [0, 1], 1, mode="medoid")

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown reference"):
            select_reference_indices(_records(), [0, 1], 1, mode="random")


class QualityTests(unittest.TestCase):
    def test_quality_keeps_the_highest_scoring_images(self):
        records = _records(4)
        scores = {str(r.image_path): i for i, r in enumerate(records)}
        with mock.patch(
            "scoring.reference_selection._quality_score",
            side_effect=lambda path, *args: scores[path],
        ):
            selected = select_reference_indices(
                records, [0, 1, 2, 3], 2, mode="quality"
            )
        self.assertEqual(selected, [3, 2])

    def test_first_mode_preserves_manifest_order(self):
        selected = select_reference_indices(
            _records(), [3, 1, 2], 2, mode="first"
        )
        self.assertEqual(selected, [3, 1])


if __name__ == "__main__":
    unittest.main()

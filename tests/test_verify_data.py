"""Tests for manifest validation used by the notebook's pretraining gate."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from training.verify_data import verify_manifest


class VerifyManifestTests(unittest.TestCase):
    def test_accepts_case_insensitive_boolean_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "image_path",
                        "signer_id",
                        "is_forgery",
                        "split",
                        "dataset",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "image_path": "genuine.png",
                            "signer_id": "signer_a",
                            "is_forgery": "False",
                            "split": "train",
                            "dataset": "SigResize",
                        },
                        {
                            "image_path": "forgery.png",
                            "signer_id": "signer_a",
                            "is_forgery": "True",
                            "split": "train",
                            "dataset": "SigResize",
                        },
                    ]
                )

            section, rows = verify_manifest(manifest)

            self.assertEqual(len(rows), 2)
            self.assertFalse(
                any("is_forgery" in failure for failure in section.failures)
            )
            self.assertTrue(
                any("genuine 1 / forgery 1" in note for note in section.notes)
            )

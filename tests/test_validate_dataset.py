"""Focused tests for manifest-wide image validation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from verification.validate_dataset import validate_manifest


class ValidateManifestTests(unittest.TestCase):
    def _write_manifest(self, root: Path, images: list[tuple[str, np.ndarray]]) -> Path:
        rows = []
        for name, array in images:
            Image.fromarray(array).save(root / name)
            rows.append(
                {
                    "image_path": name,
                    "dataset": "SigResize",
                    "split": "train",
                }
            )
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["image_path", "dataset", "split"])
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_reports_valid_grayscale_pixels_and_group_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((2, 4), 255, dtype=np.uint8)
            image[0, 0] = 0
            manifest = self._write_manifest(root, [("valid.png", image)])

            report = validate_manifest(
                manifest, image_height=2, image_width=4, show_progress=False
            )

            self.assertEqual(report["failure_count"], 0)
            result = report["files"][0]
            self.assertEqual(result["shape"], [2, 4])
            self.assertEqual(result["dtype"], "uint8")
            self.assertEqual(result["pixel_min"], 0)
            self.assertEqual(result["pixel_max"], 255)
            self.assertEqual(result["unique_pixel_values"], 2)
            self.assertAlmostEqual(result["pure_white_percent"], 87.5)
            self.assertEqual(
                report["dataset_summary"]["SigResize"]["median_unique_pixel_values"],
                2,
            )

    def test_flags_wrong_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_manifest(
                root, [("wrong.png", np.full((3, 5), 255, dtype=np.uint8))]
            )

            report = validate_manifest(
                manifest, image_height=2, image_width=4, show_progress=False
            )

            self.assertEqual(report["failure_count"], 1)
            self.assertIn("shape is (3, 5)", report["failing_files"][0]["failures"][0])

    def test_flags_non_uint8_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((2, 4), 1024, dtype=np.uint16)
            manifest = self._write_manifest(root, [("uint16.png", image)])

            report = validate_manifest(
                manifest, image_height=2, image_width=4, show_progress=False
            )

            self.assertEqual(report["failure_count"], 1)
            self.assertIn("dtype is uint16", report["failing_files"][0]["failures"][0])

    def test_flags_multichannel_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_manifest(
                root, [("rgb.png", np.full((2, 4, 3), 255, dtype=np.uint8))]
            )

            report = validate_manifest(
                manifest, image_height=2, image_width=4, show_progress=False
            )

            self.assertEqual(report["failure_count"], 1)
            self.assertTrue(
                any(
                    "not single-channel grayscale" in failure
                    for failure in report["failing_files"][0]["failures"]
                )
            )


if __name__ == "__main__":
    unittest.main()

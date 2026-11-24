"""Focused tests for the read-only SigResize audit.

The audit decides whether SigResize is safe to train on, so these tests pin the
measurements that could quietly lie: stroke width on a known line, identical-file
hashing, cross-signer duplicate detection, leave-one-out signer accuracy, and the
direction of the thin/thick ink change.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from training import audit_sigresize as audit


def _blank() -> np.ndarray:
    return np.full((256, 512), 255, np.uint8)


def _row(
    path: str,
    signer: str,
    *,
    forgery: bool = False,
    dataset: str = "SigResize",
    split: str = "train",
) -> audit.Row:
    return audit.Row(Path(path), path, signer, forgery, split, dataset)


def _stats(image: np.ndarray, sha: str = "sha") -> audit.ImageStats:
    ink = image < audit.INK_LEVEL
    return audit.ImageStats(
        sha256=sha,
        height=int(image.shape[0]),
        width=int(image.shape[1]),
        ink_pixels=int(ink.sum()),
        ink_ratio=float(ink.mean()),
        stroke_width=audit.stroke_width(image),
        thumb=cv2.resize(image, audit.THUMB_SIZE, interpolation=cv2.INTER_AREA),
        mask=audit._binary_mask(image),
        bits=audit._hash_bits(image),
    )


class StrokeWidthTests(unittest.TestCase):
    def test_estimates_a_known_line_width(self) -> None:
        image = _blank()
        # A filled 6-pixel band has a constant distance-transform ridge, unlike
        # cv2.line whose round caps make the ridge thicker than the stroke.
        cv2.rectangle(image, (50, 100), (450, 105), 0, thickness=-1)
        width = audit.stroke_width(image)
        self.assertIsNotNone(width)
        self.assertAlmostEqual(float(width), 6.0, delta=0.5)

    def test_blank_image_has_no_stroke_width(self) -> None:
        self.assertIsNone(audit.stroke_width(_blank()))


class DuplicateTests(unittest.TestCase):
    def test_identical_files_hash_the_same(self) -> None:
        image = _blank()
        cv2.line(image, (40, 40), (460, 200), 0, thickness=3)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"
            self.assertTrue(cv2.imwrite(str(first), image))
            self.assertTrue(cv2.imwrite(str(second), image))
            first_stats = audit.load_image_stats(first)
            second_stats = audit.load_image_stats(second)
        self.assertEqual(first_stats.sha256, second_stats.sha256)
        self.assertEqual(first_stats.bits, second_stats.bits)
        similarity = float(
            audit._unit_thumb(first_stats.thumb) @ audit._unit_thumb(second_stats.thumb)
        )
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_exact_pairs_ignore_within_signer_copies(self) -> None:
        image = _blank()
        cv2.line(image, (40, 40), (460, 200), 0, thickness=3)
        shared = _stats(image, "same-bytes")
        rows = [_row("a1.png", "A"), _row("a2.png", "A"), _row("b1.png", "B")]
        pairs, skipped = audit.exact_cross_signer_pairs(rows, [shared] * 3)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(pairs), 2)
        for first, second in pairs:
            self.assertNotEqual(rows[first].signer_id, rows[second].signer_id)

    def test_near_pairs_find_identical_images_across_signers(self) -> None:
        image = _blank()
        cv2.line(image, (60, 80), (450, 180), 0, thickness=4)
        different = _blank()
        cv2.line(different, (60, 240), (450, 60), 0, thickness=4)
        rows = [_row("a.png", "A"), _row("b.png", "B"), _row("c.png", "C")]
        stats = [_stats(image, "1"), _stats(image, "2"), _stats(different, "3")]
        pairs, skipped_bands, truncated = audit.near_duplicate_pairs(rows, stats)
        self.assertIn((0, 1), pairs)
        self.assertNotIn((0, 2), pairs)
        self.assertEqual(skipped_bands, 0)
        self.assertEqual(truncated, 0)
        self.assertAlmostEqual(pairs[(0, 1)][0], 1.0, places=5)
        self.assertGreaterEqual(pairs[(0, 1)][1], 0.99)

    def test_ink_iou_is_one_for_identical_masks(self) -> None:
        image = _blank()
        cv2.line(image, (40, 100), (460, 100), 0, thickness=4)
        bits = audit._hash_bits(image)
        self.assertEqual(audit._ink_iou(bits, bits), 1.0)


class SeparabilityTests(unittest.TestCase):
    @staticmethod
    def _pattern(kind: int) -> np.ndarray:
        image = _blank()
        if kind == 0:
            cv2.line(image, (40, 128), (470, 128), 0, thickness=5)
        elif kind == 1:
            cv2.line(image, (40, 220), (470, 40), 0, thickness=5)
        else:
            cv2.line(image, (256, 20), (256, 236), 0, thickness=5)
            cv2.line(image, (256, 128), (400, 128), 0, thickness=5)
        return image

    def test_distinct_shapes_identify_their_signer(self) -> None:
        rows: list[audit.Row] = []
        stats: list[audit.ImageStats] = []
        for signer in range(3):
            for variant in range(3):
                image = np.roll(self._pattern(signer), variant * 2, axis=1)
                rows.append(_row(f"{signer}_{variant}.png", f"s{signer}"))
                stats.append(_stats(image, f"{signer}-{variant}"))
        for signer in range(3):
            rows.append(_row(f"{signer}_forgery.png", f"s{signer}", forgery=True))
            stats.append(_stats(self._pattern(signer), f"{signer}-forgery"))
        result = audit.signer_separability(rows, stats, "SigResize")
        self.assertEqual(result.signers, 3)
        self.assertEqual(result.evaluated_signers, 3)
        self.assertIsNotNone(result.accuracy)
        self.assertGreaterEqual(float(result.accuracy), 0.99)
        self.assertIsNotNone(result.forgery_similarity)


class ManifestTests(unittest.TestCase):
    def test_rejects_a_non_boolean_forgery_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text(
                "image_path,signer_id,is_forgery,split,dataset\n"
                "a.png,s1,maybe,train,SigResize\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "is_forgery"):
                audit.read_manifest(manifest, Path(directory))


class AugmentationOpTests(unittest.TestCase):
    def test_thin_removes_ink_and_thick_adds_it(self) -> None:
        image = _blank()
        cv2.line(image, (60, 128), (450, 128), 0, thickness=5)
        augment = audit.SignatureAugment(seed=0)
        before = audit._ink_pixels(image)
        thin = audit._ink_pixels(audit.apply_op(augment, "thin", image))
        thick = audit._ink_pixels(audit.apply_op(augment, "thick", image))
        self.assertLess(thin, before)
        self.assertGreater(thick, before)

    def test_unknown_op_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown augmentation op"):
            audit.apply_op(audit.SignatureAugment(seed=0), "cutout", _blank())


if __name__ == "__main__":
    unittest.main()

"""Focused tests for the group-1 signature augmentation.

The contract these pin down:

* output is always ``(256, 512)`` ``uint8`` whatever the input size;
* a call is reproducible from ``(seed, epoch, index)`` and changes across epochs and
  samples, which is what makes the DataLoader worker count irrelevant;
* group 1 fires only group-1 transforms, and group 2 stays off unless asked for;
* ink is never wiped out on a signature with real stroke width.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from training.augment import H, W, SignatureAugment

GROUP1_OPERATIONS = {
    "rotation",
    "scale_shift",
    "thick",
    "thin",
    "lighter_ink",
    "blur",
    "noise_jpeg",
}
GROUP2_OPERATIONS = {"junk", "brightness_contrast"}


def _stroke_image(thickness: int = 6, size: tuple[int, int] = (H, W)) -> np.ndarray:
    """Return a white canvas with one dark signature-like stroke."""
    image = np.full(size, 255, dtype=np.uint8)
    height, width = size
    points = np.array(
        [
            [int(width * 0.10), int(height * 0.65)],
            [int(width * 0.35), int(height * 0.30)],
            [int(width * 0.60), int(height * 0.70)],
            [int(width * 0.90), int(height * 0.35)],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [points], False, 0, thickness=thickness)
    return image


def _augmenter(**overrides: object) -> SignatureAugment:
    """Build an augmenter with every transform disabled unless overridden."""
    settings: dict[str, object] = {
        "p_rot": 0.0,
        "p_scaleshift": 0.0,
        "p_thick": 0.0,
        "p_thin": 0.0,
        "p_light": 0.0,
        "p_blur": 0.0,
        "p_noise_jpeg": 0.0,
        "p_junk": 0.0,
        "p_bc": 0.0,
    }
    settings.update(overrides)
    return SignatureAugment(**settings)  # type: ignore[arg-type]


class ShapeTests(unittest.TestCase):
    def test_output_is_always_256x512_uint8(self) -> None:
        image = _stroke_image()
        augmenter = SignatureAugment(seed=0)
        for index in range(40):
            output = augmenter(image, index=index)
            self.assertEqual(output.shape, (H, W))
            self.assertEqual(output.dtype, np.uint8)

    def test_nonstandard_input_is_resized_to_the_canvas(self) -> None:
        output = _augmenter()(_stroke_image(size=(128, 256)))
        self.assertEqual(output.shape, (H, W))
        self.assertEqual(output.dtype, np.uint8)


class ReproducibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = _stroke_image()

    def test_same_seed_epoch_index_is_identical(self) -> None:
        augmenter = SignatureAugment(seed=42)
        augmenter.set_epoch(3)
        first = augmenter(self.image, index=7)
        second = augmenter(self.image, index=7)
        np.testing.assert_array_equal(first, second)

    def test_a_fresh_object_reproduces_the_same_result(self) -> None:
        first = SignatureAugment(seed=42)
        first.set_epoch(3)
        second = SignatureAugment(seed=42)
        second.set_epoch(3)
        np.testing.assert_array_equal(
            first(self.image, index=7), second(self.image, index=7)
        )

    def test_a_new_epoch_changes_the_result(self) -> None:
        augmenter = SignatureAugment(seed=42)
        augmenter.set_epoch(3)
        before = augmenter(self.image, index=7)
        augmenter.set_epoch(4)
        after = augmenter(self.image, index=7)
        self.assertFalse(np.array_equal(before, after))

    def test_a_new_index_changes_the_result(self) -> None:
        augmenter = SignatureAugment(seed=42)
        augmenter.set_epoch(3)
        self.assertFalse(
            np.array_equal(
                augmenter(self.image, index=7), augmenter(self.image, index=8)
            )
        )

    def test_without_an_index_the_stream_keeps_advancing(self) -> None:
        augmenter = SignatureAugment(seed=42)
        augmenter.set_epoch(0)
        first = augmenter(self.image)
        second = augmenter(self.image)
        self.assertFalse(np.array_equal(first, second))


class OperationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = _stroke_image()

    def test_group1_never_fires_group2_transforms(self) -> None:
        augmenter = SignatureAugment(seed=1, group2=False)
        seen: set[str] = set()
        for index in range(200):
            _, used = augmenter(self.image, return_ops=True, index=index)
            seen.update(used)
        self.assertTrue(
            seen <= GROUP1_OPERATIONS,
            f"unexpected operations: {seen - GROUP1_OPERATIONS}",
        )
        self.assertFalse(seen & GROUP2_OPERATIONS)

    def test_group2_fires_when_explicitly_enabled(self) -> None:
        augmenter = SignatureAugment(seed=1, group2=True, p_bc=1.0)
        for index in range(20):
            _, used = augmenter(self.image, return_ops=True, index=index)
            self.assertIn("brightness_contrast", used)

    def test_every_group1_transform_is_reachable(self) -> None:
        augmenter = SignatureAugment(
            seed=2,
            p_rot=1.0,
            p_scaleshift=1.0,
            p_thick=0.4,
            p_thin=0.4,
            p_light=1.0,
            p_blur=1.0,
            p_noise_jpeg=1.0,
        )
        seen: set[str] = set()
        for index in range(40):
            _, used = augmenter(self.image, return_ops=True, index=index)
            seen.update(used)
        self.assertEqual(seen, GROUP1_OPERATIONS)


class InkSurvivalTests(unittest.TestCase):
    def test_a_real_stroke_is_never_erased(self) -> None:
        """A 6 px stroke is the realistic worst case; ink must survive 300 draws."""
        image = _stroke_image(thickness=6)
        before = int((image < 200).sum())
        augmenter = SignatureAugment(seed=11)
        worst = 1.0
        for index in range(300):
            output = augmenter(image, index=index)
            after = int((output < 200).sum())
            self.assertGreater(after, 0, f"ink erased for index {index}")
            worst = min(worst, after / before)
        self.assertGreater(worst, 0.20, f"worst ink retention {worst:.3f} is too low")


if __name__ == "__main__":
    unittest.main()

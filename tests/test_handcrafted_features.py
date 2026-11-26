"""Focused tests for the handcrafted signature features.

These nine features are the handcrafted half of the Stage 4 score-level fusion, so the
tests pin the contract the fusion code depends on: a fixed name order, values bounded to
``[0, 1]``, symmetry under argument swap, a maximal self-pair, and the single distance
feature being inverted by :func:`hand_similarity_vector`.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from preprocessing.signature_preprocessing import (
    DEFAULT_TARGET_SIZE,
    load_preprocessed_image,
)
from scoring.handcrafted_features import (
    DISTANCE_FEATURES,
    FEATURE_NAMES,
    PROFILE_BINS,
    SIMILARITY_FEATURES,
    SignatureDescriptors,
    binarize_ink,
    describe_image,
    describe_records,
    hand_similarity_vector,
    pair_features,
)
from training.stage3_metric_learning import SignatureRecord

WIDTH, HEIGHT = DEFAULT_TARGET_SIZE


def _signature_image(seed: int = 0, stroke: int = 2, shift: int = 0) -> np.ndarray:
    """Draw a deterministic dark-on-white signature-like image."""
    rng = np.random.default_rng(seed)
    image = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    wave = np.column_stack(
        [
            np.linspace(40 + shift, WIDTH - 40 + shift, 24),
            HEIGHT / 2.0 + 40.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, 24)),
        ]
    ).astype(np.int32)
    cv2.polylines(image, [wave.reshape(-1, 1, 2)], False, 0, stroke)
    # A second, jittered stroke keeps the descriptor from being a single smooth curve.
    second = np.column_stack(
        [
            np.linspace(WIDTH * 0.3, WIDTH * 0.8, 12),
            np.linspace(HEIGHT * 0.3, HEIGHT * 0.7, 12),
        ]
    ).astype(np.int32) + rng.integers(-6, 7, size=(12, 2))
    cv2.polylines(image, [second.reshape(-1, 1, 2)], False, 0, 1)
    return image


class FeatureNameTests(unittest.TestCase):
    def test_declares_nine_unique_features(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 9)
        self.assertEqual(len(set(FEATURE_NAMES)), 9)
        self.assertEqual(
            set(FEATURE_NAMES), set(SIMILARITY_FEATURES) | set(DISTANCE_FEATURES)
        )
        self.assertEqual(len(SIMILARITY_FEATURES), 8)
        self.assertEqual(DISTANCE_FEATURES, ("x_slant_diff",))

    def test_places_the_distance_feature_after_four_similarities(self) -> None:
        self.assertEqual(FEATURE_NAMES[:4], SIMILARITY_FEATURES[:4])
        self.assertEqual(FEATURE_NAMES[4], "x_slant_diff")
        self.assertEqual(FEATURE_NAMES[5:], SIMILARITY_FEATURES[4:])


class BinarizeInkTests(unittest.TestCase):
    def test_rejects_a_blank_image(self) -> None:
        with self.assertRaisesRegex(ValueError, "no ink"):
            binarize_ink(np.full((HEIGHT, WIDTH), 255, dtype=np.uint8))

    def test_binary_image_uses_the_fixed_threshold(self) -> None:
        gray = np.full((8, 8), 255, dtype=np.uint8)
        gray[2:4, 2:4] = 0
        ink = binarize_ink(gray)
        self.assertEqual(ink.dtype, bool)
        self.assertEqual(int(ink.sum()), 4)
        np.testing.assert_array_equal(ink, gray < 200)

    def test_rejects_non_uint8_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "uint8"):
            binarize_ink(np.zeros((4, 4), dtype=np.float32))

class DescribeImageTests(unittest.TestCase):
    def test_rejects_unvalidated_image_size(self) -> None:
        image = np.full((256, 512), 255, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "support only HxW 192x384"):
            describe_image(image)

    def test_returns_bounded_scalars_and_profiles(self) -> None:
        descriptors = describe_image(_signature_image())
        self.assertIsInstance(descriptors, SignatureDescriptors)
        self.assertIsNone(descriptors.ink_mask)
        self.assertIsNone(descriptors.skeleton)
        self.assertGreater(descriptors.ink_area, 0)
        self.assertGreater(descriptors.skeleton_length, 0)
        self.assertGreaterEqual(descriptors.density, 0.0)
        self.assertLessEqual(descriptors.density, 1.0)
        self.assertEqual(descriptors.h_profile.shape, (PROFILE_BINS,))
        self.assertEqual(descriptors.v_profile.shape, (PROFILE_BINS,))
        self.assertEqual(descriptors.hu.shape, (7,))
        for value in (descriptors.aspect, descriptors.stroke_width, descriptors.slant):
            self.assertTrue(np.isfinite(value))

    def test_keeps_masks_only_on_request(self) -> None:
        descriptors = describe_image(_signature_image(), keep_masks=True)
        self.assertIsNotNone(descriptors.ink_mask)
        self.assertIsNotNone(descriptors.skeleton)
        assert descriptors.ink_mask is not None and descriptors.skeleton is not None
        self.assertEqual(descriptors.ink_mask.dtype, bool)
        self.assertEqual(descriptors.skeleton.dtype, bool)
        # Thinning never adds pixels, so the skeleton cannot exceed the ink mask.
        self.assertLessEqual(
            int(descriptors.skeleton.sum()), int(descriptors.ink_mask.sum())
        )

    def test_thinning_shrinks_a_thick_stroke(self) -> None:
        descriptors = describe_image(_signature_image(stroke=9), keep_masks=True)
        assert descriptors.ink_mask is not None and descriptors.skeleton is not None
        self.assertLess(
            int(descriptors.skeleton.sum()), int(descriptors.ink_mask.sum()) // 2
        )


class PairFeatureTests(unittest.TestCase):
    def test_returns_all_declared_features_in_order(self) -> None:
        first = describe_image(_signature_image(seed=1))
        second = describe_image(_signature_image(seed=2, shift=6))
        features = pair_features(first, second)
        self.assertEqual(tuple(features), FEATURE_NAMES)
        for name, value in features.items():
            self.assertIsInstance(value, float, name)
            self.assertGreaterEqual(value, 0.0, name)
            self.assertLessEqual(value, 1.0, name)

    def test_self_pair_is_maximally_similar(self) -> None:
        descriptors = describe_image(_signature_image())
        features = pair_features(descriptors, descriptors)
        for name in SIMILARITY_FEATURES:
            self.assertAlmostEqual(features[name], 1.0, places=9)
        for name in DISTANCE_FEATURES:
            self.assertAlmostEqual(features[name], 0.0, places=9)

    def test_is_symmetric(self) -> None:
        first = describe_image(_signature_image(seed=3))
        second = describe_image(_signature_image(seed=4, shift=-8, stroke=3))
        forward = pair_features(first, second)
        backward = pair_features(second, first)
        for name in FEATURE_NAMES:
            self.assertAlmostEqual(forward[name], backward[name], places=12)



class HandSimilarityVectorTests(unittest.TestCase):
    def test_inverts_only_the_distance_feature(self) -> None:
        features = {name: 0.25 for name in FEATURE_NAMES}
        vector = hand_similarity_vector(features)
        self.assertEqual(vector.shape, (len(FEATURE_NAMES),))
        for index, name in enumerate(FEATURE_NAMES):
            expected = 0.75 if name in DISTANCE_FEATURES else 0.25
            self.assertAlmostEqual(float(vector[index]), expected, places=12)

    def test_requires_every_declared_feature(self) -> None:
        features = {name: 0.5 for name in FEATURE_NAMES[:-1]}
        with self.assertRaisesRegex(KeyError, FEATURE_NAMES[-1]):
            hand_similarity_vector(features)


class DescribeRecordsTests(unittest.TestCase):
    def test_rejects_unvalidated_image_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "support only HxW 192x384"):
            describe_records([], image_height=256, image_width=512)

    def test_reads_through_the_shared_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(2):
                path = root / f"sig_{index}.png"
                self.assertTrue(cv2.imwrite(str(path), _signature_image(seed=index)))
                paths.append(path)
            records = [
                SignatureRecord(image_path=path, signer_id="signer_a", is_forgery=False)
                for path in paths
            ]
            descriptors = describe_records(
                records, already_preprocessed=True, progress_every=0
            )
            self.assertEqual(len(descriptors), 2)
            expected = describe_image(
                load_preprocessed_image(paths[0], already_preprocessed=True)
            )
            self.assertEqual(descriptors[0].ink_area, expected.ink_area)
            np.testing.assert_allclose(descriptors[0].h_profile, expected.h_profile)

    def test_propagates_loader_shape_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong_shape.png"
            self.assertTrue(
                cv2.imwrite(str(path), np.full((10, 10), 255, dtype=np.uint8))
            )
            records = [
                SignatureRecord(image_path=path, signer_id="signer_a", is_forgery=False)
            ]
            with self.assertRaisesRegex(ValueError, "expected"):
                describe_records(records, already_preprocessed=True, progress_every=0)


if __name__ == "__main__":
    unittest.main()


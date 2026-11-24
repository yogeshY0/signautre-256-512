"""Tests for the production-v2 package introduced in Phase 1."""

from __future__ import annotations

import unittest
from pathlib import Path
import json

import cv2
import numpy as np

from preprocessing.signature_preprocessing import preprocess_signature
from preprocessing.preprocess_dataset import parse_size
from preprocessing.v2.artifacts import clean_artifacts, _near_body, _is_hollow_box, _is_stamp, _is_edge_sliver, _is_line_component
from preprocessing.v2.geometry import deskew_small_angle, detect_crop_clipping
from preprocessing.v2.mask import select_ink_mask
from preprocessing.v2.params import Params
from preprocessing.v2.pipeline import (
    PRODUCTION_V2_DEFAULT_SIZE,
    apply_preprocessing,
    preprocess,
    preprocess_complete,
    whiten_pure_background,
)
from preprocessing.v2.optional import build_distance_map, build_skeleton
from preprocessing.signature_preprocess import load_gray, preprocess_image
from preprocessing.v2.polarity import estimate_polarity, flatten_illumination_preserving_faint
from preprocessing.v2.profile import profile_image
from preprocessing.v2.validate import validate_image


def signature_image() -> np.ndarray:
    image = np.full((120, 300), 255, dtype=np.uint8)
    cv2.ellipse(image, (150, 60), (90, 25), 8, 0, 320, 20, 2)
    cv2.line(image, (70, 85), (220, 70), 20, 2)
    return image


class PhaseOneValidationTests(unittest.TestCase):
    def test_blank_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "blank"):
            validate_image(np.full((100, 200), 255, dtype=np.uint8))

    def test_tiny_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tiny"):
            validate_image(np.full((20, 20), 255, dtype=np.uint8))

    def test_low_contrast_rejected(self) -> None:
        image = np.full((100, 200), 248, dtype=np.uint8)
        image[:, 100:] = 250
        with self.assertRaisesRegex(ValueError, "low-contrast"):
            validate_image(image)

    def test_saturated_rejected(self) -> None:
        image = np.full((100, 200), 255, dtype=np.uint8)
        image[40:50, 80:90] = 241
        with self.assertRaisesRegex(ValueError, "saturated"):
            validate_image(image)

    def test_mixed_binary_scan_is_valid(self) -> None:
        image = np.full((100, 200), 255, dtype=np.uint8)
        cv2.ellipse(image, (100, 50), (60, 20), 0, 0, 360, 0, 2)
        gray = validate_image(image)
        np.testing.assert_array_equal(gray, image)

    def test_dark_low_value_palette_scan_is_valid(self) -> None:
        image = np.zeros((100, 200), dtype=np.uint8)
        cv2.line(image, (20, 50), (180, 50), 255, 2)
        gray = validate_image(image)
        self.assertEqual(gray.shape, (100, 200))

    def test_bgr_is_converted_to_grayscale(self) -> None:
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        cv2.line(image, (20, 50), (180, 50), (20, 20, 20), 2)
        cv2.rectangle(image, (40, 70), (160, 90), (200, 200, 200), -1)
        gray = validate_image(image)
        self.assertEqual(gray.ndim, 2)
        self.assertEqual(gray.dtype, np.uint8)


class PhaseOneProfileTests(unittest.TestCase):
    def test_profile_contains_required_measurements(self) -> None:
        profile = profile_image(signature_image())
        for key in (
            "ink_ratio",
            "contrast_score",
            "blur_score",
            "border_contact",
            "component_count",
            "bbox_width",
            "bbox_height",
            "aspect_ratio",
        ):
            self.assertIn(key, profile)

    def test_near_blank_production_rejected(self) -> None:
        image = np.full((120, 300), 255, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "blank|saturated|usable"):
            preprocess(image, mode="production-v2")


class LegacyPassThroughTests(unittest.TestCase):
    def test_legacy_matches_existing_preprocessing_byte_for_byte(self) -> None:
        image = signature_image()
        old = preprocess_signature(image, target_size=(384, 192), binarize=False)
        gray, binary, artifacts, metadata = preprocess(
            image,
            mode="legacy",
            size=(384, 192),
            binarize=False,
        )
        np.testing.assert_array_equal(gray, old)
        self.assertEqual(set(np.unique(binary).tolist()), {0, 255})
        self.assertEqual(set(np.unique(artifacts).tolist()), {0})
        self.assertEqual(metadata["preprocessing_version"], "legacy")

    def test_unsupported_mode_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported preprocessing mode"):
            preprocess(signature_image(), mode="other")  # type: ignore[arg-type]


class PhaseTwoPolarityAndMaskTests(unittest.TestCase):
    def test_light_on_dark_polarity_is_detected_and_inverted(self) -> None:
        dark = signature_image()
        reversed_image = cv2.bitwise_not(dark)
        polarity, confidence = estimate_polarity(reversed_image)
        self.assertEqual(polarity, "light_on_dark")
        self.assertGreater(confidence, 0.0)

    def test_high_raw_dark_ratio_is_not_rejected_before_polarity(self) -> None:
        dark_background = np.full((120, 300), 0, dtype=np.uint8)
        cv2.ellipse(dark_background, (150, 60), (90, 25), 8, 0, 320, 255, 2)
        canonical, binary, artifacts, metadata = preprocess(
            dark_background,
            mode="production-v2",
            size=(384, 192),
        )
        self.assertEqual(canonical.shape, (192, 384))
        self.assertIn(metadata["selected_polarity"], {"dark_on_light", "light_on_dark"})

    def test_binary_image_passes_through_illumination_without_changes(self) -> None:
        binary = np.full((120, 300), 255, dtype=np.uint8)
        cv2.ellipse(binary, (150, 60), (90, 25), 8, 0, 320, 0, 2)
        unchanged = flatten_illumination_preserving_faint(binary)
        np.testing.assert_array_equal(unchanged, binary)

    def test_faint_ink_is_not_collapsed_to_white(self) -> None:
        image = np.full((120, 300), 255, dtype=np.uint8)
        cv2.line(image, (30, 60), (270, 60), 248, 1)
        cv2.line(image, (30, 80), (270, 80), 20, 2)
        normalized = flatten_illumination_preserving_faint(image)
        faint_before = int(image[60, 150])
        faint_after = int(normalized[60, 150])
        self.assertLess(faint_after, faint_before)
        self.assertGreater(faint_after, 0)
        self.assertLess(faint_after, 255)

    def test_mask_candidates_are_deterministic_and_binary(self) -> None:
        image = signature_image()
        first, name, details = select_ink_mask(image)
        second, second_name, second_details = select_ink_mask(image)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(name, second_name)
        self.assertEqual(set(np.unique(first).tolist()), {0, 255})
        self.assertIn("score", details)
        self.assertIn("details", second_details)

    def test_production_v2_returns_metadata_and_exact_dimensions(self) -> None:
        _, _, _, metadata = preprocess(
            signature_image(),
            mode="production-v2",
            size=(384, 192),
        )
        canonical, binary, artifacts, metadata = preprocess(
            signature_image(),
            mode="production-v2",
            size=(384, 192),
        )
        self.assertEqual(canonical.shape, (192, 384))
        self.assertEqual(binary.shape, (192, 384))
        self.assertEqual(artifacts.shape, (192, 384))
        self.assertEqual(canonical.dtype, np.uint8)
        self.assertEqual(set(np.unique(binary).tolist()), {0, 255})
        self.assertTrue(
            {"preprocessing_version", "selected_polarity", "quality_status", "warnings"}
            <= set(metadata)
        )


class PhaseThreeArtifactTests(unittest.TestCase):
    def test_clean_signature_artifact_step_removes_nearly_no_ink(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.ellipse(image, (200, 110), (130, 45), 0, 0, 360, 0, 3)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)
        removed = int(np.count_nonzero((artifact == 255)))
        original = int(np.count_nonzero(mask))
        self.assertLessEqual(removed / max(original, 1), 0.01)
        self.assertFalse(details["artifact_detected"])

    def test_border_line_is_removed(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.ellipse(image, (200, 110), (100, 40), 0, 0, 360, 0, 3)
        cv2.line(image, (0, 20), (399, 20), 0, 2)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)
        self.assertTrue(details["artifact_detected"])
        self.assertGreater(int(np.count_nonzero(artifact == 255)), 0)

    def test_real_vertical_stroke_about_40_percent_height_is_kept(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.line(image, (200, 66), (200, 154), 0, 3)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)
        self.assertEqual(int(np.count_nonzero(artifact == 255)), 0)
        self.assertGreater(int(np.count_nonzero(cleaned)), 0)

    def test_i_dot_and_dash_near_body_are_kept(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.ellipse(image, (200, 110), (100, 35), 0, 0, 360, 0, 2)
        cv2.circle(image, (175, 55), 3, 0, -1)
        cv2.line(image, (220, 145), (245, 145), 0, 2)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)
        self.assertEqual(int(np.count_nonzero(artifact == 255)), 0)

    def test_line_crossing_signature_is_kept_and_flagged(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.ellipse(image, (200, 110), (100, 40), 0, 0, 360, 0, 3)
        cv2.line(image, (0, 110), (399, 110), 0, 2)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)
        self.assertTrue(details["artifact_uncertain"])
        self.assertGreater(int(np.count_nonzero(artifact == 128)), 0)

    def test_stamp_is_removed_only_if_separate_and_dense(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.ellipse(image, (180, 110), (90, 35), 0, 0, 360, 0, 2)
        cv2.rectangle(image, (320, 25), (380, 105), 0, -1)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)
        self.assertTrue(details["artifact_detected"])


    def test_separate_vertical_line_is_removed(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.ellipse(image, (100, 110), (60, 35), 0, 0, 360, 0, 2)
        cv2.line(image, (350, 10), (350, 209), 0, 2)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)
        self.assertTrue(details["artifact_detected"])

    def test_separate_horizontal_line_is_removed(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.ellipse(image, (150, 110), (80, 30), 0, 0, 360, 0, 2)
        cv2.line(image, (50, 210), (350, 210), 0, 2)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)
        self.assertTrue(details["artifact_detected"])

    def test_artifact_uses_255_removed_mask_values(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.ellipse(image, (150, 110), (80, 30), 0, 0, 360, 0, 2)
        cv2.line(image, (10, 210), (390, 210), 0, 2)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)
        self.assertTrue(set(np.unique(artifact).tolist()).issubset({0, 128, 255}))

    def test_max_removal_cap_prevents_over_cleaning(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.ellipse(image, (150, 110), (80, 30), 0, 0, 360, 0, 2)
        for x in range(0, 400, 10):
            cv2.line(image, (x, 0), (x, 219), 0, 1)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)
        self.assertLessEqual(details["ink_removed_fraction"], 0.70)

    def test_rotation_limit_does_not_create_unbounded_deskew(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.line(image, (50, 110), (350, 110), 0, 3)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        gray, rotated_mask, artifacts, angle, applied = deskew_small_angle(
            image,
            mask,
            np.zeros_like(mask),
            Params(max_deskew_degrees=8.0),
        )
        self.assertLessEqual(abs(angle), 8.0)
        self.assertEqual(rotated_mask.shape, mask.shape)

    def test_256x512_output_contract(self) -> None:
        canonical, binary, artifacts, metadata = preprocess(
            signature_image(),
            mode="production-v2",
            size=(512, 256),
        )
        self.assertEqual(canonical.shape, (256, 512))
        self.assertEqual(binary.shape, (256, 512))
        self.assertEqual(artifacts.shape, (256, 512))
        self.assertGreaterEqual(int(canonical.min()), 0)
        self.assertLessEqual(int(canonical.max()), 255)

    def test_384x192_output_contract(self) -> None:
        canonical, binary, artifacts, metadata = preprocess(
            signature_image(),
            mode="production-v2",
            size=(384, 192),
        )
        self.assertEqual(canonical.shape, (192, 384))
        self.assertEqual(binary.shape, (192, 384))

    def test_deskew_bounded_on_single_blob(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.circle(image, (100, 110), 10, 0, -1)
        cv2.circle(image, (300, 110), 10, 0, -1)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        _, _, _, angle, applied = deskew_small_angle(
            image, mask, np.zeros_like(mask)
        )
        self.assertLessEqual(abs(angle), 8.0)

class PhaseTwoQualityMetadataTests(unittest.TestCase):
    def test_metadata_is_serializable(self) -> None:
        import json
        canonical, binary, artifacts, metadata = preprocess(
            signature_image(), mode="production-v2"
        )
        dumped = json.dumps(metadata, sort_keys=True)
        loaded = json.loads(dumped)
        self.assertEqual(loaded["preprocessing_version"], "production-v2")

    def test_mask_details_in_metadata(self) -> None:
        canonical, binary, artifacts, metadata = preprocess(
            signature_image(), mode="production-v2"
        )
        self.assertIn("mask_name", metadata)
        self.assertIn("mask_details", metadata)
        self.assertIsInstance(metadata.get("mask_details", {}), dict)

    def test_deskew_fields_in_metadata(self) -> None:
        canonical, binary, artifacts, metadata = preprocess(
            signature_image(), mode="production-v2"
        )
        self.assertIn("deskew_angle", metadata)
        self.assertIn("deskew_applied", metadata)
        self.assertIsInstance(metadata["deskew_angle"], float)
        self.assertIsInstance(metadata["deskew_applied"], bool)

    def test_ink_removed_fraction_in_metadata(self) -> None:
        canonical, binary, artifacts, metadata = preprocess(
            signature_image(), mode="production-v2"
        )
        self.assertIn("ink_removed_fraction", metadata)
        self.assertGreaterEqual(metadata["ink_removed_fraction"], 0.0)
        self.assertLessEqual(metadata["ink_removed_fraction"], 1.0)


class CompletePipelineTests(unittest.TestCase):
    def test_external_text_mask_marks_overlap_but_keeps_ink_by_default(self) -> None:
        image = np.full((120, 300), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "Photo/S",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            0,
            2,
            cv2.LINE_AA,
        )
        text_mask = np.zeros_like(image)
        text_mask[0:50, 0:180] = 255
        result = apply_preprocessing(
            image,
            size=(384, 192),
            text_mask=text_mask,
        )
        self.assertGreater(result.binary_mask.size, 0)
        self.assertEqual(result.metadata["ink_removed_fraction"], 0.0)
        self.assertGreaterEqual(result.metadata["text_mask_components"], 0)

    def test_complete_result_keeps_source_frame_artifact_mask(self) -> None:
        image = signature_image()
        result = preprocess_complete(image)
        self.assertIsNotNone(result.source_artifact_mask)
        assert result.source_artifact_mask is not None
        self.assertEqual(result.source_artifact_mask.shape, image.shape)
        self.assertTrue(
            set(np.unique(result.source_artifact_mask).tolist()).issubset({0, 128, 255})
        )
        json.dumps(result.metadata)

    def test_single_call_accepts_numpy_array(self) -> None:
        result = preprocess_image(signature_image())
        self.assertEqual(result.gray.shape, (256, 512))
        self.assertEqual(result.mask.shape, (256, 512))
        self.assertEqual(result.artifact_mask.shape, (120, 300))
        tensor = result.model_input()
        self.assertEqual(tensor.shape, (3, 256, 512))

    def test_load_gray_supports_16_bit_scaling(self) -> None:
        gray16 = np.full((100, 200), 65535, dtype=np.uint16)
        gray16[40:60, 80:120] = 0
        gray = load_gray(gray16)
        self.assertEqual(gray.dtype, np.uint8)
        self.assertEqual(int(gray.min()), 0)
        self.assertEqual(int(gray.max()), 255)

    def test_load_gray_composites_alpha_on_white(self) -> None:
        rgba = np.zeros((100, 200, 4), dtype=np.uint8)
        rgba[..., :3] = 20
        rgba[..., 3] = 128
        gray = load_gray(rgba)
        self.assertEqual(gray.dtype, np.uint8)
        self.assertGreater(int(gray.min()), 0)
        self.assertLessEqual(int(gray.max()), 255)

    def test_production_v2_default_size_is_512x256(self) -> None:
        self.assertEqual(PRODUCTION_V2_DEFAULT_SIZE, (512, 256))
        result = preprocess_complete(signature_image())
        self.assertEqual(result.canonical_grayscale.shape, (256, 512))
        self.assertEqual(result.binary_mask.shape, (256, 512))
        self.assertEqual(result.artifact_mask.shape, (256, 512))

    def test_complete_result_contains_optional_representations(self) -> None:
        result = apply_preprocessing(signature_image())
        self.assertIsNotNone(result.canonical_grayscale)
        self.assertIsNotNone(result.binary_mask)
        self.assertIsNotNone(result.artifact_mask)
        self.assertIsNotNone(result.skeleton_mask)
        self.assertIsNotNone(result.distance_map)
        self.assertIsNotNone(result.model_input)
        assert result.model_input is not None
        self.assertEqual(result.model_input.shape, (3, 256, 512))

    def test_optional_representations_can_be_disabled(self) -> None:
        result = preprocess_complete(
            signature_image(),
            include_skeleton=False,
            include_distance_map=False,
            include_model_input=False,
        )
        self.assertIsNone(result.skeleton_mask)
        self.assertIsNone(result.distance_map)
        self.assertIsNone(result.model_input)

    def test_skeleton_and_distance_outputs_are_valid(self) -> None:
        binary = np.full((100, 200), 255, dtype=np.uint8)
        cv2.line(binary, (20, 50), (180, 50), 0, 3)
        skeleton = build_skeleton(binary)
        distance = build_distance_map(binary)
        self.assertEqual(set(np.unique(skeleton).tolist()).issubset({0, 255}), True)
        self.assertEqual(distance.dtype, np.uint8)
        self.assertGreater(int(distance.max()), 0)

    def test_192x384_remains_available_as_explicit_option(self) -> None:
        result = preprocess_complete(signature_image(), size=(384, 192))
        self.assertEqual(result.canonical_grayscale.shape, (192, 384))
        assert result.model_input is not None
        self.assertEqual(result.model_input.shape, (3, 192, 384))


class PriorityFixTests(unittest.TestCase):
    def test_keep_by_default_never_removes_real_ink(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.ellipse(image, (200, 110), (100, 35), 0, 0, 360, 0, 3)
        cv2.rectangle(image, (5, 5), (394, 214), 0, 2)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)

        params = Params(destructive_cleaning_enabled=False)
        canonical, binary, output_artifact, metadata = preprocess(
            image,
            mode='production-v2',
            size=(384, 192),
            params=params,
        )
        self.assertEqual(metadata['ink_removed_fraction'], 0.0)
        self.assertEqual(metadata['border_removed_fraction'], 0.0)
        ink_after = int(np.count_nonzero(binary == 0))
        self.assertGreater(ink_after, 0)

    def test_full_border_removed_but_real_vertical_stroke_kept(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.rectangle(image, (5, 5), (394, 214), 0, 2)
        cv2.line(image, (200, 66), (200, 154), 0, 3)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)
        vertical_kept = int(np.count_nonzero(cleaned[66:155, 198:202]))
        self.assertGreater(vertical_kept, 0)
        self.assertLessEqual(details["ink_removed_fraction"], 0.70)

    def test_top_edge_text_is_removed_not_rescued_as_dot(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.ellipse(image, (200, 110), (100, 35), 0, 0, 360, 0, 3)
        cv2.putText(
            image,
            "ATURE",
            (80, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            0,
            1,
            cv2.LINE_AA,
        )
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)
        self.assertTrue(details["artifact_detected"])
        self.assertGreater(int(np.count_nonzero(artifact == 255)), 0)
        self.assertLessEqual(details["ink_removed_fraction"], 0.10)

    def test_shaded_paper_background_is_whitened(self) -> None:
        image = np.arange(400, dtype=np.uint16)[None, :]
        image = np.repeat(image, 220, axis=0)
        gray = np.clip(220 - (image / 400.0) * 30, 0, 255).astype(np.uint8)
        mask = np.zeros((220, 400), dtype=np.uint8)
        cv2.ellipse(mask, (200, 110), (100, 35), 0, 0, 360, 255, 3)
        result = whiten_pure_background(gray, mask)
        stroke = mask > 0
        keep = cv2.dilate(stroke.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
        non_keep = ~keep
        white_fraction = float(np.mean(result[non_keep] == 255))
        self.assertGreaterEqual(white_fraction, 0.99)
        np.testing.assert_array_equal(result[stroke], gray[stroke])

    def test_crop_clipping_left_only(self) -> None:
        mask = np.zeros((220, 400), dtype=np.uint8)
        mask[100:140, 0] = 255
        self.assertTrue(detect_crop_clipping(mask))

    def test_crop_clipping_all_four_sides_is_tight_crop_not_clipped(self) -> None:
        mask = np.zeros((220, 400), dtype=np.uint8)
        mask[:, 0] = 255
        mask[:, -1] = 255
        mask[0, :] = 255
        mask[-1, :] = 255
        self.assertFalse(detect_crop_clipping(mask))

    def test_top_text_outside_body_is_removed_but_verticals_and_dots_keep(self) -> None:
        image = np.full((220, 400), 255, dtype=np.uint8)
        cv2.ellipse(image, (200, 110), (100, 35), 0, 0, 360, 0, 3)
        cv2.line(image, (180, 75), (180, 155), 0, 3)
        cv2.line(image, (220, 75), (220, 155), 0, 3)
        cv2.circle(image, (175, 63), 3, 0, -1)
        cv2.line(image, (235, 145), (260, 145), 0, 2)
        cv2.line(image, (90, 15), (150, 15), 0, 2)
        cv2.line(image, (250, 12), (290, 12), 0, 2)
        mask = np.where(image < 240, 255, 0).astype(np.uint8)
        cleaned, artifact, details = clean_artifacts(mask)
        self.assertGreater(int(np.count_nonzero(artifact == 255)), 0)
        self.assertLessEqual(details["ink_removed_fraction"], 0.10)
        self.assertGreater(int(np.count_nonzero(cleaned)), 0)

    def test_s05_real_crop_fixture_if_available(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "S05_01.png"
        if not fixture.exists():
            self.skipTest("S05_01.png fixture has not been provided")
        image = cv2.imread(str(fixture), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        canonical, binary, artifact, metadata = preprocess(
            image,
            mode="production-v2",
            size=(384, 192),
        )
        self.assertEqual(canonical.shape, (192, 384))
        self.assertTrue(set(np.unique(binary).tolist()).issubset({0, 255}))
        self.assertTrue(set(np.unique(artifact).tolist()).issubset({0, 128, 255}))
        self.assertLessEqual(metadata["ink_removed_fraction"], 0.10)


class PhaseOneHelperTests(unittest.TestCase):
    def test_near_body_within_distance(self) -> None:
        self.assertTrue(_near_body((10, 10, 20, 20), (15, 15, 25, 25), 2))

    def test_near_body_far_apart(self) -> None:
        self.assertFalse(_near_body((0, 0, 10, 10), (20, 20, 30, 30), 2))

    def test_hollow_box_detection(self) -> None:
        stats = np.zeros((2, cv2.CC_STAT_MAX), dtype=np.int32)
        stats[1, cv2.CC_STAT_WIDTH] = 100
        stats[1, cv2.CC_STAT_HEIGHT] = 80
        stats[1, cv2.CC_STAT_AREA] = 200
        self.assertTrue(_is_hollow_box(stats, 1, 400, 220, Params()))

    def test_edge_sliver_detection(self) -> None:
        stats = np.zeros((2, cv2.CC_STAT_MAX), dtype=np.int32)
        stats[1, cv2.CC_STAT_LEFT] = 0
        stats[1, cv2.CC_STAT_TOP] = 50
        stats[1, cv2.CC_STAT_WIDTH] = 2
        stats[1, cv2.CC_STAT_HEIGHT] = 100
        stats[1, cv2.CC_STAT_AREA] = 200
        self.assertTrue(_is_edge_sliver(stats, 1, 400, 220, 1000, Params()))

    def test_line_component_identification(self) -> None:
        stats = np.zeros((2, cv2.CC_STAT_MAX), dtype=np.int32)
        stats[1, cv2.CC_STAT_WIDTH] = 250
        stats[1, cv2.CC_STAT_HEIGHT] = 3
        h_line, v_line = _is_line_component(stats, 1, 400, 220, Params())
        self.assertTrue(h_line)
        self.assertFalse(v_line)

    def test_stamp_detection_filled(self) -> None:
        stats = np.zeros((2, cv2.CC_STAT_MAX), dtype=np.int32)
        stats[1, cv2.CC_STAT_WIDTH] = 100
        stats[1, cv2.CC_STAT_HEIGHT] = 80
        stats[1, cv2.CC_STAT_AREA] = 7500
        self.assertTrue(_is_stamp(stats, 1, 400, 220, 10000, Params()))


if __name__ == "__main__":
    unittest.main()


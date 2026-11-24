"""Production-v2 dispatcher and Phase 1 pipeline implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import cv2

from preprocessing.signature_preprocessing import (
    DEFAULT_TARGET_SIZE,
    preprocess_signature,
    to_model_input,
)
from .params import Params
from .profile import profile_image
from .validate import validate_image
from .polarity import estimate_polarity, flatten_illumination_preserving_faint
from .mask import select_ink_mask
from .artifacts import clean_artifacts
from .geometry import deskew_small_angle, tight_crop_and_resize
from .optional import build_distance_map, build_skeleton
from .result import PreprocessingResult

PreprocessingMode = Literal["legacy", "production-v2"]
PRODUCTION_V2_DEFAULT_SIZE = (512, 256)


def whiten_pure_background(
    gray: np.ndarray,
    ink_mask: np.ndarray,
    *,
    params: Params | None = None,
) -> np.ndarray:
    """Whiten paper outside ink while preserving selected ink and a one-pixel fringe."""

    params = params or Params()
    if gray.shape != ink_mask.shape:
        raise ValueError("gray and ink mask must have identical dimensions")

    keep_radius = max(1, int(params.pure_white_fringe_px))
    stroke = (ink_mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * keep_radius + 1, 2 * keep_radius + 1),
    )
    keep_area = cv2.dilate(stroke, kernel)

    border = max(
        1,
        min(
            params.border_band,
            gray.shape[0] // 20,
            gray.shape[1] // 20,
        ),
    )
    border_pixels = np.concatenate(
        (
            gray[:border, :].ravel(),
            gray[-border:, :].ravel(),
            gray[:, :border].ravel(),
            gray[:, -border:].ravel(),
        )
    )
    paper_tone = float(np.median(border_pixels))

    result = gray.copy()
    result[keep_area == 0] = 255
    fringe = (keep_area > 0) & (stroke == 0)
    fringe_near_paper = fringe & (
        np.abs(result.astype(np.int16) - int(round(paper_tone)))
        <= params.pure_white_bg_tol
    )
    result[fringe_near_paper] = 255
    return result


def preprocess(
    image: np.ndarray,
    mode: PreprocessingMode = "legacy",
    size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    *,
    params: Params | None = None,
    binarize: bool = False,
    text_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Preprocess one image and return canonical gray, binary mask, artifact mask, metadata.

    Legacy mode delegates directly to ``preprocess_signature`` for byte-identical
    behaviour. Production-v2 is introduced in phases; geometry and artifact stages
    are added in later phases.
    """

    if mode == "legacy":
        canonical = preprocess_signature(
            image,
            target_size=size,
            binarize=binarize,
        )
        artifact_mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        legacy_profile = profile_image(canonical, params or Params())
        metadata = {
            "preprocessing_version": "legacy",
            "selected_polarity": "dark_on_light",
            "polarity_confidence": 1.0,
            "ink_ratio": legacy_profile["ink_ratio"],
            "bbox_width": legacy_profile["bbox_width"],
            "bbox_height": legacy_profile["bbox_height"],
            "aspect_ratio": legacy_profile["aspect_ratio"],
            "component_count": legacy_profile["component_count"],
            "border_contact": legacy_profile["border_contact"],
            "crop_clipped": False,
            "contrast_score": legacy_profile["contrast_score"],
            "blur_score": legacy_profile["blur_score"],
            "artifact_detected": False,
            "artifact_uncertain": False,
            "quality_status": "ok",
            "warnings": [],
        }
        binary = np.where(canonical < 200, 0, 255).astype(np.uint8)
        return canonical, binary, artifact_mask, metadata

    if mode != "production-v2":
        raise ValueError(f"unsupported preprocessing mode: {mode}")

    params = params or Params()
    gray = validate_image(image, params)
    profile = profile_image(gray, params)

    warnings: list[str] = []
    if float(profile["ink_ratio"]) < params.min_ink_ratio:
        raise ValueError("no usable signature ink found")

    if float(profile["contrast_score"]) < params.low_contrast_flag:
        warnings.append("low contrast")

    polarity_result = estimate_polarity(gray, params)
    if isinstance(polarity_result, tuple) and len(polarity_result) == 3:
        polarity, polarity_confidence, polarity_details = polarity_result
    else:
        polarity, polarity_confidence = polarity_result
        polarity_details = {}
    if polarity == "light_on_dark":
        gray = cv2.bitwise_not(gray)

    # Already-binary images return unchanged and are never blurred or morphologically altered.
    unique = set(np.unique(gray).tolist())
    already_binary = bool(unique and unique.issubset({0, 128, 255}))
    if already_binary:
        normalized = gray.copy()
    else:
        normalized = flatten_illumination_preserving_faint(gray, params)

    ink_mask, mask_name, mask_details = select_ink_mask(normalized, params)
    if int(np.count_nonzero(ink_mask)) < params.min_ink_pixels:
        raise ValueError("no usable signature ink found in selected mask")

    if params.destructive_cleaning_enabled:
        cleaned_mask, artifact_mask, artifact_details = clean_artifacts(ink_mask, params)
    else:
        cleaned_mask = ink_mask.copy()
        candidate_mask, artifact_candidate_mask, candidate_details = clean_artifacts(
            ink_mask,
            params,
        )
        artifact_mask = np.where(
            artifact_candidate_mask == 128,
            128,
            0,
        ).astype(np.uint8)
        artifact_details = {
            **candidate_details,
            "artifact_detected": bool(candidate_details["artifact_detected"]),
            "artifact_uncertain": bool(candidate_details["artifact_uncertain"]),
            "ink_removed": 0,
            "ink_removed_fraction": 0.0,
            "border_removed_fraction": 0.0,
        }

    if text_mask is not None:
        if text_mask.shape != ink_mask.shape:
            raise ValueError("text_mask must have the same dimensions as the source image")
        if text_mask.dtype != np.uint8:
            raise ValueError("text_mask must be uint8")

        high_confidence_text = (text_mask > 0).astype(np.uint8)
        if params.external_text_dilate_px > 0:
            radius = 2 * params.external_text_dilate_px + 1
            high_confidence_text = cv2.dilate(
                high_confidence_text,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)),
            )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (cleaned_mask > 0).astype(np.uint8),
            connectivity=8,
        )
        text_removed_pixels = 0
        text_uncertain_pixels = 0
        text_component_count = 0
        text_artifact_mask = np.zeros_like(artifact_mask)

        for label in range(1, count):
            component = labels == label
            overlap = int(np.count_nonzero(component & (high_confidence_text > 0)))
            component_pixels = int(stats[label, cv2.CC_STAT_AREA])
            overlap_fraction = overlap / max(1, component_pixels)
            if overlap_fraction < params.external_text_overlap_fraction:
                continue

            text_component_count += 1
            if (
                params.destructive_cleaning_enabled
                and params.external_text_mask_enabled
            ):
                text_removed_pixels += component_pixels
                text_artifact_mask[component] = 255
                cleaned_mask[component] = 0
            else:
                text_uncertain_pixels += component_pixels
                text_artifact_mask[component] = 128

        if params.destructive_cleaning_enabled:
            if text_removed_pixels > 0:
                artifact_detected = True
                artifact_mask = np.maximum(artifact_mask, text_artifact_mask)
                artifact_details["artifact_detected"] = True
                artifact_details["ink_removed"] = int(
                    artifact_details.get("ink_removed", 0)
                ) + text_removed_pixels
                artifact_details["ink_removed_fraction"] = (
                    (int(artifact_details.get("ink_removed", 0)))
                    / max(1, int(np.count_nonzero(ink_mask)))
                )
        elif text_uncertain_pixels > 0:
            artifact_mask = np.maximum(artifact_mask, text_artifact_mask)
            artifact_details["artifact_uncertain"] = True

        artifact_details["text_mask_components"] = int(text_component_count)
        artifact_details["text_mask_removed_pixels"] = int(text_removed_pixels)
        artifact_details["text_mask_uncertain_pixels"] = int(text_uncertain_pixels)
    source_artifact_mask = artifact_mask.copy()
    canonical = normalized.copy()
    canonical[artifact_mask == 255] = 255

    # After mask selection, whiten paper outside the ink and its 1 px fringe.
    canonical = whiten_pure_background(canonical, cleaned_mask, params=params)

    canonical, cleaned_mask, artifact_mask, deskew_angle, deskew_applied = deskew_small_angle(
        canonical,
        cleaned_mask,
        artifact_mask,
        params,
    )
    canonical, cleaned_mask, artifact_mask, geometry = tight_crop_and_resize(
        canonical,
        cleaned_mask,
        artifact_mask,
        size,
        params,
    )
    canonical = whiten_pure_background(canonical, cleaned_mask, params=params)
    final_profile = profile_image(canonical, params)
    binary = np.where(cleaned_mask > 0, 0, 255).astype(np.uint8)

    if float(final_profile["ink_ratio"]) < params.min_ink_ratio:
        raise ValueError("no usable signature ink remains after geometry processing")
    if float(final_profile["contrast_score"]) < params.low_contrast_flag:
        warnings.append("low contrast")
    if float(final_profile["blur_score"]) < params.low_blur_flag:
        warnings.append("low laplacian variance")

    quality_status: str = "ok"
    rejection_reason: str | None = None
    if float(final_profile["ink_ratio"]) < params.min_ink_ratio:
        quality_status = "rejected"
        rejection_reason = "no usable ink remains after processing"
    elif warnings or geometry.get("crop_clipped", False) or artifact_details["artifact_uncertain"]:
        quality_status = "flagged"

    metadata = {
        "preprocessing_version": "production-v2",
        "selected_polarity": polarity,
        "polarity_confidence": float(polarity_confidence),
        "ink_ratio": final_profile["ink_ratio"],
        "bbox_width": geometry["bbox_width"],
        "bbox_height": geometry["bbox_height"],
        "aspect_ratio": geometry["aspect_ratio"],
        "component_count": final_profile["component_count"],
        "border_contact": final_profile["border_contact"],
        "crop_clipped": geometry["crop_clipped"],
        "crop_clipped_legacy": geometry["crop_clipped_legacy"],
        "contrast_score": final_profile["contrast_score"],
        "blur_score": final_profile["blur_score"],
        "artifact_detected": bool(artifact_details["artifact_detected"]),
        "artifact_uncertain": bool(artifact_details["artifact_uncertain"]),
        "quality_status": quality_status,
        "warnings": warnings,
        "rejection_reason": rejection_reason,
        "mask_name": mask_name,
        "mask_details": mask_details,
        "ink_removed_fraction": artifact_details["ink_removed_fraction"],
        "border_removed_fraction": artifact_details.get("border_removed_fraction", 0.0),
        "deskew_angle": float(deskew_angle),
        "deskew_applied": bool(deskew_applied),
        "text_mask_supplied": bool(text_mask is not None),
        "text_mask_components": int(artifact_details.get("text_mask_components", 0)),
        "text_mask_removed_pixels": int(artifact_details.get("text_mask_removed_pixels", 0)),
        "text_mask_uncertain_pixels": int(artifact_details.get("text_mask_uncertain_pixels", 0)),
    }
    return canonical, binary, artifact_mask, metadata


def preprocess_complete(
    image: np.ndarray,
    *,
    size: tuple[int, int] = PRODUCTION_V2_DEFAULT_SIZE,
    params: Params | None = None,
    include_skeleton: bool = True,
    include_distance_map: bool = True,
    include_model_input: bool = True,
    text_mask: np.ndarray | None = None,
) -> PreprocessingResult:
    """Run the complete production-v2 pipeline and return every ready representation."""

    canonical, binary, artifact, metadata = preprocess(
        image,
        mode="production-v2",
        size=size,
        params=params,
        text_mask=text_mask,
    )
    skeleton = build_skeleton(binary) if include_skeleton else None
    distance = build_distance_map(binary) if include_distance_map else None
    model_input = to_model_input(canonical) if include_model_input else None
    exact_params = params or Params()
    source_gray = validate_image(image, exact_params)
    source_polarity, _ = estimate_polarity(source_gray, exact_params)
    if source_polarity == "light_on_dark":
        source_gray = cv2.bitwise_not(source_gray)
    source_unique = set(np.unique(source_gray).tolist())
    source_binary = bool(source_unique and source_unique.issubset({0, 128, 255}))
    if not source_binary:
        source_gray = flatten_illumination_preserving_faint(source_gray, exact_params)
    source_ink, _, _ = select_ink_mask(source_gray, exact_params)
    _, source_artifact_mask, _ = clean_artifacts(source_ink, exact_params)
    if text_mask is not None:
        if text_mask.shape != source_ink.shape:
            raise ValueError("text_mask must have the same dimensions as the source image")
        text = (text_mask > 0).astype(np.uint8)
        count, labels, _, _ = cv2.connectedComponentsWithStats(
            (source_ink > 0).astype(np.uint8),
            connectivity=8,
        )
        for label in range(1, count):
            component = labels == label
            overlap = int(np.count_nonzero(component & (text > 0)))
            if overlap / max(1, int(np.count_nonzero(component))) >= exact_params.external_text_overlap_fraction:
                source_artifact_mask[component] = 128
    return PreprocessingResult(
        canonical_grayscale=canonical,
        binary_mask=binary,
        artifact_mask=artifact,
        metadata=metadata,
        skeleton_mask=skeleton,
        distance_map=distance,
        model_input=model_input,
        source_artifact_mask=source_artifact_mask,
    )


def apply_preprocessing(
    image: np.ndarray,
    *,
    size: tuple[int, int] = PRODUCTION_V2_DEFAULT_SIZE,
    params: Params | None = None,
    include_skeleton: bool = True,
    include_distance_map: bool = True,
    include_model_input: bool = True,
    text_mask: np.ndarray | None = None,
) -> PreprocessingResult:
    """Public one-call production preprocessing API."""

    return preprocess_complete(
        image,
        size=size,
        params=params,
        include_skeleton=include_skeleton,
        include_distance_map=include_distance_map,
        include_model_input=include_model_input,
        text_mask=text_mask,
    )

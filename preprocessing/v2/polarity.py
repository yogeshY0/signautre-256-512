"""Polarity estimation and conservative illumination normalization."""

from __future__ import annotations

import cv2
import numpy as np

from .params import Params
from .profile import profile_image


def _polarity_score(gray: np.ndarray, params: Params) -> tuple[float, dict[str, float]]:
    profile = profile_image(gray, params)
    ink_ratio = float(profile["ink_ratio"])
    border_dark = float(profile["border_contact"])
    component_count = float(profile["component_count"])
    contrast = float(profile["contrast_score"])
    aspect = float(profile["aspect_ratio"])

    border_score = 1.0 - border_dark
    h, w = gray.shape[:2]
    border = max(1, min(params.border_band, h // 20, w // 20))
    border_pixels_dark = float(np.mean(gray[:border, :] < 128)) + float(np.mean(gray[-border:, :] < 128)) + float(np.mean(gray[:, :border] < 128)) + float(np.mean(gray[:, -border:] < 128))
    border_score2 = 1.0 - min(border_pixels_dark / 4.0, 1.0)
    border_score_final = 0.5 * border_score + 0.5 * border_score2
    ratio_score = 1.0 - abs(ink_ratio - params.max_ink_ratio * 0.12) / max(
        params.max_ink_ratio, 1e-6
    )
    ratio_score = float(np.clip(ratio_score, 0.0, 1.0))
    aspect_score = 1.0 if 1.0 <= aspect <= 12.0 else max(0.0, 1.0 - abs(aspect - 4.0) / 10.0)
    component_score = 1.0 if component_count > 0 else 0.0
    contrast_score = float(np.clip(contrast / 80.0, 0.0, 1.0))

    # Stroke continuity score
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))
    stroke_score = float(np.clip(edge_density * 50.0, 0.0, 1.0))

    score = (
        params.polarity_border_weight * border_score_final
        + params.polarity_ratio_weight * ratio_score
        + params.polarity_component_weight * component_score
        + params.polarity_contrast_weight * contrast_score
        + params.polarity_aspect_weight * aspect_score
        + params.polarity_stroke_weight * stroke_score
        + params.polarity_bbox_plausibility_weight * 0.5
    )
    details = {
        "border_score": border_score,
        "ratio_score": ratio_score,
        "component_score": component_score,
        "contrast_score": contrast_score,
        "aspect_score": aspect_score,
    }
    return float(score), details


def estimate_polarity(gray: np.ndarray, params: Params | None = None) -> tuple[str, float]:
    params = params or Params()
    dark_score, _ = _polarity_score(gray, params)
    inverted = cv2.bitwise_not(gray)
    light_score, _ = _polarity_score(inverted, params)
    selected = "dark_on_light" if dark_score >= light_score else "light_on_dark"
    best = dark_score if dark_score >= light_score else light_score
    diff = abs(dark_score - light_score)
    confidence = float(np.clip(best * (0.5 + 0.5 * diff / max(best, 1e-6)), 0.0, 1.0))
    return selected, confidence


def flatten_illumination_preserving_faint(
    gray: np.ndarray,
    params: Params | None = None,
) -> np.ndarray:
    """Remove slowly varying background without erasing faint ink or blurring strokes."""

    params = params or Params()
    height, width = gray.shape[:2]

    unique = set(np.unique(gray).tolist())
    if unique and unique.issubset(set(params.illumination_binary_check_values)):
        return gray.copy()

    kernel = max(params.sauvola_window, min(height, width) // params.illumination_kernel_divisor)
    if kernel % 2 == 0:
        kernel += 1

    background = cv2.GaussianBlur(gray, (kernel, kernel), 0)

    values = gray.astype(np.float32)
    bg = background.astype(np.float32)
    reference = float(np.percentile(bg, 50))
    illumination_delta = params.illumination_delta_factor * (bg - reference)
    corrected = values - illumination_delta
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    # Preserve original ink pressure values, gently stretch very faint strokes
    # (keep stroke pressure information)
    ink_mask = values < params.faint_ink_threshold
    if np.any(ink_mask):
        corrected[ink_mask] = gray[ink_mask]

    # Gentle stretch for near-white faint strokes (improves visibility)
    faint_mask = (values >= params.faint_ink_threshold) & (values < 252) & (corrected >= 200)
    if np.any(faint_mask):
        stretched = 255.0 - (255.0 - corrected.astype(np.float32)) * 1.08
        corrected[faint_mask] = np.clip(stretched, 0, 255).astype(np.uint8)[faint_mask]

    return corrected

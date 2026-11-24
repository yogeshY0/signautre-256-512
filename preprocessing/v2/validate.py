"""Stage 0 input validation for production-v2."""

from __future__ import annotations

import cv2
import numpy as np

from .params import Params


def validate_image(image: np.ndarray, params: Params | None = None) -> np.ndarray:
    """Return a grayscale uint8 image or raise ``ValueError`` with a clear reason."""

    params = params or Params()
    if image is None:
        raise ValueError("unreadable or empty input image")
    if not isinstance(image, np.ndarray):
        raise ValueError("input image must be a numpy array")
    if image.dtype != np.uint8:
        raise ValueError("input image must have dtype uint8")
    if image.ndim not in (2, 3):
        raise ValueError("input image must be 2D or 3-channel")

    if image.ndim == 3:
        if image.shape[2] == 1:
            gray = image[..., 0].copy()
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("input image must have 1, 3, or 4 channels")
    else:
        gray = image.copy()

    height, width = gray.shape[:2]
    if width < params.min_image_width or height < params.min_image_height:
        raise ValueError(
            f"image too tiny: expected at least "
            f"{params.min_image_width}x{params.min_image_height}, got {width}x{height}"
        )
    if height * width < params.tiny_image_area:
        raise ValueError("image too tiny: area below minimum")

    unique = np.unique(gray)
    if unique.size == 1:
        raise ValueError("blank or saturated image: no intensity variation")

    is_mixed_binary = bool(
        unique.size > 1
        and set(unique.tolist()).issubset({0, 128, 255})
    )
    saturated = float(np.mean((gray == 0) | (gray == 255)))
    if saturated >= params.saturated_fraction_limit and not is_mixed_binary:
        raise ValueError("saturated image: nearly all pixels are 0 or 255")

    paper_tone = float(np.median(gray))
    robust_deviation = np.abs(gray.astype(np.float32) - paper_tone)
    contrast = float(np.percentile(robust_deviation, 99.9))
    ink_like_pixels = int(np.count_nonzero(robust_deviation >= params.min_contrast_score))
    if not is_mixed_binary:
        if contrast < params.min_contrast_score:
            raise ValueError(f"very low-contrast image: contrast_score={contrast:.2f}")
        if ink_like_pixels < params.min_ink_pixels:
            raise ValueError(
                f"blank or very low-contrast image: ink_like_pixels={ink_like_pixels}"
            )

    return gray

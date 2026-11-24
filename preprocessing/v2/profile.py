"""Stage 1 image profiling for production-v2."""

from __future__ import annotations

import cv2
import numpy as np

from .params import Params


def profile_image(gray: np.ndarray, params: Params | None = None) -> dict[str, object]:
    """Measure simple, deterministic quality and geometry properties."""

    params = params or Params()
    height, width = gray.shape[:2]

    contrast = float(np.percentile(gray, 95) - float(gray.min()))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    dark_ratio = float(np.mean(gray < 250))
    border = max(1, min(params.border_band, height // 20, width // 20))
    border_pixels = np.concatenate(
        (
            gray[:border].ravel(),
            gray[-border:].ravel(),
            gray[:, :border].ravel(),
            gray[:, -border:].ravel(),
        )
    )
    border_contact = bool(np.mean(border_pixels < 250) > 0.02)

    ink_mask = gray < params.faint_ink_threshold
    ink_pixels = int(np.count_nonzero(ink_mask))
    ink_ratio = float(ink_pixels / max(1, ink_mask.size))

    ys, xs = np.where(ink_mask)
    if ink_pixels:
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)
        aspect_ratio = float(bbox_width / max(1, bbox_height))
    else:
        bbox_width = 0
        bbox_height = 0
        aspect_ratio = 0.0

    count, _ = cv2.connectedComponents(
        ink_mask.astype(np.uint8),
        connectivity=8,
    )
    component_count = max(0, int(count - 1))

    return {
        "ink_ratio": ink_ratio,
        "contrast_score": contrast,
        "blur_score": blur,
        "border_contact": border_contact,
        "component_count": component_count,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "aspect_ratio": aspect_ratio,
        "dark_ratio": dark_ratio,
        "input_width": width,
        "input_height": height,
    }

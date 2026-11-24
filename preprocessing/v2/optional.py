"""Optional representations derived from the final binary signature mask."""

from __future__ import annotations

import cv2
import numpy as np

from scoring.handcrafted_features import _zhang_suen_thinning


def build_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """Return a single-channel 0/255 Zhang-Suen skeleton mask."""

    if binary_mask.dtype != np.uint8:
        raise ValueError("binary mask must be uint8")
    ink = binary_mask == 0
    if not ink.any():
        raise ValueError("binary mask contains no ink")
    skeleton = _zhang_suen_thinning(ink)
    return (skeleton.astype(np.uint8) * 255)


def build_distance_map(binary_mask: np.ndarray) -> np.ndarray:
    """Return a normalized Euclidean distance map of the ink."""

    if binary_mask.dtype != np.uint8:
        raise ValueError("binary mask must be uint8")
    ink = (binary_mask == 0).astype(np.uint8)
    if not ink.any():
        raise ValueError("binary mask contains no ink")
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    maximum = float(distance.max())
    if maximum <= 0:
        return np.zeros(binary_mask.shape, dtype=np.uint8)
    normalized = np.rint(distance * 255.0 / maximum).clip(0, 255)
    return normalized.astype(np.uint8)

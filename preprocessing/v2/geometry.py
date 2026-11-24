"""Geometry normalization and crop-clipping detection."""

from __future__ import annotations

import cv2
import numpy as np

from .params import Params
from .mask import _is_binary


def detect_crop_clipping(mask: np.ndarray, params: Params | None = None) -> bool:
    params = params or Params()
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return False
    height, width = mask.shape[:2]
    run = max(1, int(params.clip_contact_run_px))

    def edge_runs(values: np.ndarray) -> tuple[bool, int]:
        longest = 0
        current = 0
        for value in values:
            if value:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest >= run, longest

    left = edge_runs(mask[:, 0] > 0)[0]
    right = edge_runs(mask[:, -1] > 0)[0]
    top = edge_runs(mask[0, :] > 0)[0]
    bottom = edge_runs(mask[-1, :] > 0)[0]

    # A tight crop can legitimately touch all four borders.
    touched = sum((left, right, top, bottom))
    if touched >= 4:
        return False

    if touched > 2:
        return False

    return bool(
        (left and not right)
        or (right and not left)
        or (top and not bottom)
        or (bottom and not top)
    )


def deskew_small_angle(
    canonical: np.ndarray,
    ink_mask: np.ndarray,
    artifact_mask: np.ndarray,
    params: Params | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    params = params or Params()
    ys, xs = np.where(ink_mask > 0)
    if xs.size < params.min_ink_pixels:
        return canonical, ink_mask, artifact_mask, 0.0, False

    points = np.column_stack((xs, ys)).astype(np.float32)
    mean = points.mean(axis=0)
    centered = points - mean
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.degrees(np.arctan2(principal[1], principal[0])))
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0

    eigen_sum = float(np.sum(eigenvalues))
    confidence = 0.0 if eigen_sum <= 0 else float(
        np.clip((eigenvalues.max() - eigenvalues.min()) / eigen_sum, 0.0, 1.0)
    )
    if abs(angle) > params.max_deskew_degrees:
        angle = 0.0
    if confidence < params.min_deskew_confidence:
        return canonical, ink_mask, artifact_mask, 0.0, False

    height, width = canonical.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_gray = cv2.warpAffine(
        canonical,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    rotated_mask = cv2.warpAffine(
        ink_mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_artifacts = cv2.warpAffine(
        artifact_mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated_gray, rotated_mask, rotated_artifacts, angle, True


def tight_crop_and_resize(
    canonical: np.ndarray,
    ink_mask: np.ndarray,
    artifact_mask: np.ndarray,
    target_size: tuple[int, int],
    params: Params | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    params = params or Params()
    ys, xs = np.where(ink_mask > 0)
    if xs.size < params.min_ink_pixels:
        raise ValueError("no usable signature ink found after geometry processing")

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    ink_width = x1 - x0
    ink_height = y1 - y0
    pad = int(round(params.crop_margin_fraction * max(ink_width, ink_height)))

    source_height, source_width = canonical.shape[:2]
    clipped = detect_crop_clipping(ink_mask, params)
    legacy_clipped = bool(
        xs.min() <= 1
        or xs.max() >= source_width - 2
        or ys.min() <= 1
        or ys.max() >= source_height - 2
    )
    cx0 = max(0, x0 - pad)
    cx1 = min(source_width, x1 + pad)
    cy0 = max(0, y0 - pad)
    cy1 = min(source_height, y1 + pad)

    crop_gray = canonical[cy0:cy1, cx0:cx1]
    crop_mask = ink_mask[cy0:cy1, cx0:cx1]
    crop_artifacts = artifact_mask[cy0:cy1, cx0:cx1]

    target_w, target_h = target_size
    ch, cw = crop_gray.shape[:2]
    scale = min(target_w / max(cw, 1), target_h / max(ch, 1))
    new_w = max(1, min(target_w, round(cw * scale)))
    new_h = max(1, min(target_h, round(ch * scale)))

    gray_resized = cv2.resize(
        crop_gray,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    mask_resized = cv2.resize(
        crop_mask,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )
    artifact_resized = cv2.resize(
        crop_artifacts,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )

    gray_canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    mask_canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    artifact_canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    gray_canvas[top:top + new_h, left:left + new_w] = gray_resized
    mask_canvas[top:top + new_h, left:left + new_w] = mask_resized
    artifact_canvas[top:top + new_h, left:left + new_w] = artifact_resized

    return gray_canvas, mask_canvas, artifact_canvas, {
        "crop_clipped": clipped,
        "crop_clipped_legacy": legacy_clipped,
        "bbox_width": ink_width,
        "bbox_height": ink_height,
        "aspect_ratio": float(ink_width / max(1, ink_height)),
    }

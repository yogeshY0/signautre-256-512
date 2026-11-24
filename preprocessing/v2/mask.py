"""Ink-mask candidates and deterministic mask selection."""

from __future__ import annotations

import cv2
import numpy as np

from .params import Params


def _is_binary(gray: np.ndarray) -> bool:
    unique = set(np.unique(gray).tolist())
    return bool(unique and unique.issubset({0, 128, 255}))


def _candidates(gray: np.ndarray, params: Params) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []
    if _is_binary(gray):
        candidates.append(("existing_binary", np.where(gray < 128, 255, 0).astype(np.uint8)))
        return candidates

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    candidates.append(("otsu", otsu))

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        params.adaptive_window,
        params.adaptive_offset,
    )
    candidates.append(("adaptive", adaptive))

    window = params.sauvola_window
    mean = cv2.blur(gray, (window, window)).astype(np.float32)
    mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (window, window))
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(variance)
    threshold = mean * (1.0 - params.sauvola_k * (std / 128.0 - 1.0))
    sauvola = np.where(gray < threshold, 255, 0).astype(np.uint8)
    candidates.append(("sauvola", sauvola))

    confidence = np.clip((250.0 - gray.astype(np.float32)) * 255.0 / 250.0, 0, 255)
    _, confidence_mask = cv2.threshold(
        confidence.astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    candidates.append(("grayscale_confidence", confidence_mask))
    return candidates


def _candidate_score(mask: np.ndarray, gray: np.ndarray, params: Params) -> tuple[float, dict[str, float]]:
    ink_area = int(np.count_nonzero(mask))
    ink_ratio = ink_area / max(mask.size, 1)
    height, width = mask.shape[:2]

    plausible = (
        params.min_ink_ratio <= ink_ratio <= params.max_ink_ratio
    )
    plausible_score = 1.0 if plausible else max(0.0, 1.0 - abs(ink_ratio - params.max_ink_ratio * 0.12) * 8.0)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0)
    speck_count = int(np.count_nonzero(areas < params.isolated_speck_area))
    speck_score = 1.0 if speck_count == 0 else float(
        np.clip(1.0 - speck_count / max(1.0, count - 1), 0.0, 1.0)
    )

    if ink_area:
        ys, xs = np.where(mask > 0)
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)
        aspect = bbox_width / max(1, bbox_height)
        aspect_score = 1.0 if 1.0 <= aspect <= 12.0 else 0.25
    else:
        bbox_width = bbox_height = 0
        aspect_score = 0.0

    ys_mask = np.mean(mask > 0, axis=1)
    xs_mask = np.mean(mask > 0, axis=0)
    horizontal_continuity = 1.0 - float(np.mean(np.abs(np.diff(ys_mask))))
    vertical_continuity = 1.0 - float(np.mean(np.abs(np.diff(xs_mask))))
    continuity = float(np.clip((horizontal_continuity + vertical_continuity) / 2.0, 0.0, 1.0))

    selected = mask > 0
    if np.count_nonzero(selected):
        original_faint = float(np.mean(gray[selected]))
        faint_score = float(np.clip((250.0 - original_faint) / 200.0, 0.0, 1.0))
    else:
        faint_score = 0.0

    details = {
        "ink_ratio": float(ink_ratio),
        "plausible": plausible_score,
        "speck": speck_score,
        "aspect": aspect_score,
        "continuity": continuity,
        "faint": faint_score,
    }
    score = (
        params.mask_plausible_weight * plausible_score
        + params.mask_speck_weight * speck_score
        + params.mask_aspect_weight * aspect_score
        + params.mask_continuity_weight * continuity
        + params.mask_faint_weight * faint_score
    )
    return float(score), details


def select_ink_mask(
    gray: np.ndarray,
    params: Params | None = None,
) -> tuple[np.ndarray, str, dict[str, object]]:
    params = params or Params()
    candidates = _candidates(gray, params)
    scored: list[tuple[float, str, np.ndarray, dict[str, float]]] = []
    for name, mask in candidates:
        score, details = _candidate_score(mask, gray, params)
        scored.append((score, name, mask, details))
    scored.sort(key=lambda item: item[0], reverse=True)
    score, name, mask, details = scored[0]
    return mask, name, {"score": score, "details": details}

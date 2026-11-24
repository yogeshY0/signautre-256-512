"""Reference selection: first, medoid, or image quality."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import cv2
import numpy as np

from preprocessing.signature_preprocessing import load_preprocessed_image

MODES = ("first", "medoid", "quality")


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    out = np.zeros_like(matrix)
    return np.divide(matrix, norms, out=out, where=norms > 0)


def medoid_order(embeddings, indices, count):
    """Return count indices: medoid then greedy coverage."""
    indices = [int(i) for i in indices]
    if count >= len(indices):
        return indices
    if count <= 0:
        return []
    vectors = _unit_rows(np.asarray(embeddings)[indices])
    sim = vectors @ vectors.T
    distance = 1.0 - sim
    chosen = [int(np.argmax(sim.mean(axis=1)))]
    while len(chosen) < count:
        best_index = None
        best_objective = None
        for candidate in range(len(indices)):
            if candidate in chosen:
                continue
            nearest = distance[:, chosen].min(axis=1)
            nearest = np.minimum(nearest, distance[:, candidate])
            objective = float(nearest.sum())
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_index = candidate
        chosen.append(int(best_index))
    return [indices[i] for i in chosen]


@lru_cache(maxsize=4096)
def _quality_score(path, already_preprocessed, ink, image_height, image_width):
    """Composite quality in [0, 1]; higher is better."""
    try:
        image = load_preprocessed_image(
            path,
            already_preprocessed=already_preprocessed,
            ink=ink,
            image_height=image_height,
            image_width=image_width,
        )
    except (FileNotFoundError, ValueError):
        return 0.0
    gray = image.astype(np.float32)
    mask = gray < 200
    if int(mask.sum()) < 5:
        return 0.0
    ratio = float(mask.mean())
    log_gap = np.log(max(ratio, 1e-6)) - np.log(0.04)
    ink_term = float(np.exp(-(log_gap ** 2)))
    contrast = min(float(gray.std()) / 80.0, 1.0)
    sharp = float(cv2.Laplacian(image, cv2.CV_64F).var())
    sharp = min(sharp / 500.0, 1.0)
    return 0.4 * ink_term + 0.3 * contrast + 0.3 * sharp


def quality_order(
    records,
    indices,
    count,
    *,
    already_preprocessed,
    ink,
    image_height,
    image_width,
):
    """Return count indices ranked by image quality."""
    indices = [int(i) for i in indices]
    if count >= len(indices):
        return indices
    scored = []
    for index in indices:
        score = _quality_score(
            str(getattr(records[index], "image_path")),
            already_preprocessed,
            ink,
            image_height,
            image_width,
        )
        scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [index for _, index in scored[:count]]


def select_reference_indices(
    records,
    indices,
    count,
    *,
    mode="first",
    embeddings=None,
    already_preprocessed=True,
    ink="auto",
    image_height=256,
    image_width=512,
):
    """Select reference indices for one signer."""
    if mode not in MODES:
        raise ValueError(f"unknown reference selection mode: {mode!r}")
    if mode == "first":
        return [int(i) for i in indices[:count]]
    if mode == "medoid":
        if embeddings is None:
            raise ValueError("medoid selection requires embeddings")
        return medoid_order(embeddings, indices, count)
    return quality_order(
        records,
        indices,
        count,
        already_preprocessed=already_preprocessed,
        ink=ink,
        image_height=image_height,
        image_width=image_width,
    )

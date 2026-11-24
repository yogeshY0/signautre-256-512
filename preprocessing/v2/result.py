"""Structured return type for a complete production-v2 preprocessing pass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessingResult:
    """Canonical images and optional representations for one signature."""

    canonical_grayscale: np.ndarray
    binary_mask: np.ndarray
    artifact_mask: np.ndarray
    metadata: dict[str, object]
    skeleton_mask: np.ndarray | None = None
    distance_map: np.ndarray | None = None
    model_input: np.ndarray | None = None
    source_artifact_mask: np.ndarray | None = None

    @property
    def gray(self) -> np.ndarray:
        """One-call API alias for the canonical grayscale image."""
        return self.canonical_grayscale

    @property
    def mask(self) -> np.ndarray:
        """One-call API alias for the binary ink mask."""
        return self.binary_mask

    def make_model_input(self) -> np.ndarray:
        """Return a float32 3-channel ImageNet-normalized model tensor."""
        if self.model_input is not None:
            return self.model_input
        gray = self.canonical_grayscale.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return (np.repeat(gray[None, ...], 3, axis=0) - mean) / std

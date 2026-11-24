"""Shared FAR / FRR / EER helpers for score evaluation.

Both the deep-only baseline in :mod:`training.evaluate` and the fusion fitter in
:mod:`training.fit_fusion` interpret scores the same way: a higher score means "more
likely the same signer", and a pair is accepted when ``score >= threshold``. Keeping
the metric code in one module means the two scripts cannot drift apart, and the deep
baseline numbers stay identical to the pre-fusion implementation.
"""

from __future__ import annotations

import numpy as np


def rates_at_threshold(
    genuine: np.ndarray | list[float],
    negatives: np.ndarray | list[float],
    threshold: float,
) -> tuple[float, float]:
    """Return ``(FAR, FRR)`` at ``threshold``.

    A genuine pair is rejected when its score falls below the threshold, and a
    negative pair is falsely accepted when its score reaches it. This is the
    convention used throughout the repository.
    """
    genuine_array = np.asarray(genuine, dtype=np.float64)
    negative_array = np.asarray(negatives, dtype=np.float64)
    if genuine_array.size == 0 or negative_array.size == 0:
        raise ValueError("genuine and negative score arrays must both be non-empty")
    false_rejection_rate = float((genuine_array < threshold).mean())
    false_acceptance_rate = float((negative_array >= threshold).mean())
    return false_acceptance_rate, false_rejection_rate


def eer_threshold(
    genuine: np.ndarray | list[float],
    negatives: np.ndarray | list[float],
) -> tuple[float, float]:
    """Return the threshold where FAR and FRR are closest, and their mean (the EER).

    The candidate set is every observed score. The first candidate with the smallest
    absolute FAR/FRR gap wins, matching the original single-pass implementation so
    baseline results are reproducible.

    The sweep is evaluated with sorted arrays and ``searchsorted`` rather than a Python
    loop over candidates. At a threshold ``t`` the rates are exactly
    ``count(negative >= t) / n_negative`` and ``count(genuine < t) / n_genuine``, which
    ``searchsorted`` reproduces in ``O(n log n)`` total instead of ``O(n^2)``. The
    returned threshold and EER are identical to the naive loop; the speed-up matters
    because a full validation split contributes ~40k random negative pairs and the
    fusion fitter scores several hundred thousand pairs.
    """
    genuine_array = np.asarray(genuine, dtype=np.float64)
    negative_array = np.asarray(negatives, dtype=np.float64)
    if genuine_array.size == 0 or negative_array.size == 0:
        raise ValueError("genuine and negative score arrays must both be non-empty")

    candidates = np.unique(np.concatenate([genuine_array, negative_array]))
    genuine_sorted = np.sort(genuine_array)
    negative_sorted = np.sort(negative_array)

    # count(genuine < t) and count(negative < t) for every candidate in one pass.
    genuine_below = np.searchsorted(genuine_sorted, candidates, side="left")
    negative_below = np.searchsorted(negative_sorted, candidates, side="left")
    false_rejection = genuine_below / genuine_array.size
    false_acceptance = (negative_array.size - negative_below) / negative_array.size
    gaps = np.abs(false_acceptance - false_rejection)

    # ``argmin`` returns the first minimum, matching the loop's strict ``<`` comparison.
    best_index = int(np.argmin(gaps))
    best_threshold = float(candidates[best_index])
    equal_error_rate = float(
        (false_acceptance[best_index] + false_rejection[best_index]) / 2.0
    )
    return best_threshold, equal_error_rate


def eer(
    genuine: np.ndarray | list[float],
    negatives: np.ndarray | list[float],
) -> float:
    """Return only the equal error rate at the best threshold."""
    return eer_threshold(genuine, negatives)[1]


def threshold_at_far(
    genuine: np.ndarray | list[float],
    negatives: np.ndarray | list[float],
    target_far: float,
) -> tuple[float, float]:
    """Return the lowest threshold whose false acceptance rate is at most ``target_far``.

    EER is a summary statistic; a verification system usually has to respect a fixed FAR
    budget instead. Candidates are the observed scores plus one step above the maximum,
    so even ``target_far`` close to 0 is reachable. Among the admissible thresholds the
    *lowest* is returned because it is the one that still accepts the most genuine
    pairs, i.e. the lowest false rejection rate for that budget.

    Raises:
        ValueError: If both arrays are not non-empty, or ``target_far`` is not in (0, 1).
    """
    genuine_array = np.asarray(genuine, dtype=np.float64)
    negative_array = np.asarray(negatives, dtype=np.float64)
    if genuine_array.size == 0 or negative_array.size == 0:
        raise ValueError("genuine and negative score arrays must both be non-empty")
    if not 0.0 < target_far < 1.0:
        raise ValueError("target_far must be strictly between 0 and 1")

    candidates = np.unique(np.concatenate([genuine_array, negative_array]))
    # One step past the largest observed score makes FAR = 0 admissible at any target.
    candidates = np.append(candidates, np.nextafter(candidates[-1], np.inf))
    negative_sorted = np.sort(negative_array)
    # count(negative >= t) for every candidate in one pass.
    negative_below = np.searchsorted(negative_sorted, candidates, side="left")
    false_acceptance = (negative_array.size - negative_below) / negative_array.size
    # The last candidate always admits everything, so ``admissible`` is never empty.
    best_index = int(np.flatnonzero(false_acceptance <= target_far)[0])
    return float(candidates[best_index]), float(false_acceptance[best_index])

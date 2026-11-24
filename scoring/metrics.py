"""Shared FAR / FRR / EER helpers for score evaluation.

Both the deep-only baseline in :mod:`training.evaluate` and the fusion fitter in
:mod:`training.fit_fusion` interpret scores the same way: a higher score means "more
likely the same signer", and a pair is accepted when ``score >= threshold``. Keeping
the metric code in one module means the two scripts cannot drift apart, and the deep
baseline numbers stay identical to the pre-fusion implementation.
"""

from __future__ import annotations

from typing import Sequence

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


def auc(
    genuine: np.ndarray | list[float],
    negatives: np.ndarray | list[float],
) -> float | None:
    """Return ROC AUC with genuine=1, or None if undefined."""
    from sklearn.metrics import roc_auc_score
    g = np.asarray(genuine, dtype=np.float64)
    n = np.asarray(negatives, dtype=np.float64)
    if g.size == 0 or n.size == 0:
        return None
    labels = np.concatenate([np.ones(g.size), np.zeros(n.size)])
    scores = np.concatenate([g, n])
    if np.unique(labels).size < 2:
        return None
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return None


def threshold_at_frr(
    genuine: np.ndarray | list[float],
    negatives: np.ndarray | list[float],
    target_frr: float,
) -> tuple[float, float]:
    """Return the highest threshold whose FRR is at most target_frr."""
    g = np.asarray(genuine, dtype=np.float64)
    n = np.asarray(negatives, dtype=np.float64)
    if g.size == 0 or n.size == 0:
        raise ValueError("genuine and negative score arrays must both be non-empty")
    if not 0.0 < target_frr < 1.0:
        raise ValueError("target_frr must be strictly between 0 and 1")
    candidates = np.unique(np.concatenate([g, n]))
    candidates = np.append(candidates, np.nextafter(candidates[-1], np.inf))
    g_sorted = np.sort(g)
    below = np.searchsorted(g_sorted, candidates, side="left")
    frr = below / g.size
    admissible = np.flatnonzero(frr <= target_frr)
    index = int(admissible[-1])
    return float(candidates[index]), float(frr[index])


def bootstrap_ci(
    genuine: np.ndarray | list[float],
    negatives: np.ndarray | list[float],
    genuine_signers: Sequence[object],
    negative_signers: Sequence[object],
    metric,
    *,
    resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
):
    """Signer-level bootstrap CI for a score metric."""
    g = np.asarray(genuine, dtype=np.float64)
    n = np.asarray(negatives, dtype=np.float64)
    if g.size == 0 or n.size == 0:
        return None
    if len(g) != len(genuine_signers) or len(n) != len(negative_signers):
        raise ValueError("signer labels must align with the score arrays")
    if resamples < 1 or not 0.0 < confidence < 1.0:
        return None
    signers = sorted({str(s) for s in genuine_signers} | {str(s) for s in negative_signers})
    if len(signers) < 2:
        return None
    index = {signer: position for position, signer in enumerate(signers)}
    g_groups = [[] for _ in signers]
    n_groups = [[] for _ in signers]
    for value, signer in zip(g, genuine_signers):
        g_groups[index[str(signer)]].append(float(value))
    for value, signer in zip(n, negative_signers):
        n_groups[index[str(signer)]].append(float(value))
    g_arrays = [np.asarray(group) for group in g_groups]
    n_arrays = [np.asarray(group) for group in n_groups]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(signers), size=len(signers))
        g_parts = [g_arrays[i] for i in chosen if g_arrays[i].size]
        n_parts = [n_arrays[i] for i in chosen if n_arrays[i].size]
        if not g_parts or not n_parts:
            continue
        gg = np.concatenate(g_parts)
        nn = np.concatenate(n_parts)
        try:
            value = metric(gg, nn)
        except (ValueError, ZeroDivisionError):
            continue
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if len(values) < max(10, resamples // 2):
        return None
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    )

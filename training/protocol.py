"""The one verification protocol shared by training selection and evaluation.

Training picks the checkpoint by the skilled-forgery EER; evaluation reports it. If those
two code paths built their own enrollment, a change in one could silently make the
selection metric different from the reported metric. Both therefore call
:func:`skilled_eer` here, and :func:`training.evaluate._evaluate_split` is tested against
it on identical embeddings.

The protocol: the first ``references`` genuine images of each signer are enrolled; a
query scores as its mean cosine similarity to those references; genuine queries are the
signer's remaining genuine images; skilled negatives are that signer's forgeries. Random
negatives are deliberately not built here, because checkpoint selection must not depend
on the (large) random-pair sample.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from scoring.fusion import enroll_and_build_queries, score_queries
from scoring.metrics import eer_threshold


def skilled_eer(
    records: Sequence[object],
    embeddings: np.ndarray,
    references: int,
    *,
    reference_selection: str = "first",
    already_preprocessed: bool = True,
    ink: str = "auto",
    image_height: int = 256,
    image_width: int = 512,
) -> float:
    """Return the skilled-forgery EER for one embedding matrix.

    Args:
        records: Manifest rows in the same order as ``embeddings``.
        embeddings: One unit-norm embedding per row.
        references: Genuine references enrolled per signer.

    Raises:
        ValueError: Fewer than ``references + 1`` genuine images for a signer, or one of
            the genuine/skilled score sets is empty.
    """
    enrollment = enroll_and_build_queries(
        records,
        references,
        max_random_queries=0,
        reference_selection=reference_selection,
        embeddings=embeddings,
        already_preprocessed=already_preprocessed,
        ink=ink,
        image_height=image_height,
        image_width=image_width,
    )
    scores = score_queries(embeddings, [], enrollment)
    genuine = scores["deep_genuine"]
    skilled = scores["deep_skilled"]
    if genuine.size == 0 or skilled.size == 0:
        raise ValueError("skilled EER needs at least one genuine and one skilled query")
    _, value = eer_threshold(genuine, skilled)
    return float(value)

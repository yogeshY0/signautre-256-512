"""Fit a score-level fusion config on the validation split.

The fusion config combines the deep cosine similarity of the ConvNeXt-Tiny Siamese
embedding with the nine handcrafted geometric/structural similarity features from
:mod:`scoring.handcrafted_features`. Two rules are fitted and compared:

``weighted``
    ``Final = w1 * S_deep + w2 * mean(H_hand)``, with ``w1`` grid-searched on the
    validation split.
``logistic``
    A logistic-regression classifier over ``[S_deep, *H_hand]``, stored as plain JSON
    coefficients so the config stays portable.

The fit runs on the **validation** split only; the test split is never touched here.
``--cv-folds`` (default 5) additionally measures the fused score out-of-fold with
signer-grouped folds, so the stored threshold is not the optimistic in-sample one.

Usage:
    python -m training.fit_fusion --checkpoint checkpoints/cedar_v1/best_model.pt \\
        --manifest data/processed/manifest.csv --already-preprocessed \\
        --output checkpoints/fusion/fusion.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from scoring.fusion import (
    FusionConfig,
    FusionScorer,
    PairMatrix,
    build_pair_matrix,
    enroll_and_build_queries,
    fit_logistic_fusion,
    fit_weighted_fusion,
    save_fusion_config,
)
from scoring.handcrafted_features import FEATURE_NAMES, describe_records
from scoring.metrics import eer_threshold
from training.stage3_metric_learning import (
    SiameseSignatureModel,
    embed_records,
    read_split,
)


def _fit_config(
    fusion_type: str, deep: np.ndarray, hand: np.ndarray, labels: np.ndarray
) -> tuple[FusionConfig, dict[str, object]]:
    """Fit one fusion rule on the supplied rows."""
    if fusion_type == "weighted":
        return fit_weighted_fusion(deep, hand, labels)
    if fusion_type == "logistic":
        return fit_logistic_fusion(deep, hand, labels)
    raise ValueError(f"unsupported fusion type: {fusion_type}")


def _query_scores_from_rows(
    row_scores: np.ndarray, query_ids: np.ndarray, kinds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average contiguous per-pair rows back to per-query scores.

    :func:`build_pair_matrix` emits all reference rows of one query consecutively, so the
    block boundaries can be recovered from the first occurrence of each query id. The
    per-query average is the unit the evaluation protocol thresholds, so the fitted
    threshold must live in that space.
    """
    _, first_index = np.unique(query_ids, return_index=True)
    boundaries = np.append(first_index[1:], len(query_ids))
    counts = boundaries - first_index
    totals = np.add.reduceat(row_scores, first_index)
    return totals / counts, kinds[first_index]


def _cross_validated_threshold(
    matrix: PairMatrix, fusion_type: str, folds: int
) -> tuple[float, float]:
    """Return ``(threshold, out_of_fold_eer)`` from signer-grouped folds.

    The split is grouped by signer, so a held-out fold contains identities the fold's
    rule never saw. That makes the measured threshold a fair estimate instead of the
    optimistic in-sample one.
    """
    signers = np.unique(matrix.signers)
    if folds < 2:
        raise ValueError("cross-validation needs at least two folds")
    if len(signers) < folds:
        raise ValueError(
            f"cannot build {folds} signer-grouped folds from {len(signers)} signers"
        )

    oof_rows = np.full(len(matrix), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for train_index, test_index in splitter.split(
        matrix.deep, matrix.labels, groups=matrix.signers
    ):
        train = matrix.subset(train_index)
        if len(np.unique(train.labels)) < 2:
            raise ValueError("a fitting fold contains only genuine or only negative pairs")
        config, _ = _fit_config(fusion_type, train.deep, train.hand, train.labels)
        scorer = FusionScorer(config)
        oof_rows[test_index] = scorer.score_rows(
            matrix.deep[test_index], matrix.hand[test_index], FEATURE_NAMES
        )

    if np.isnan(oof_rows).any():
        raise RuntimeError("out-of-fold scoring left rows unscored")

    query_scores, query_kinds = _query_scores_from_rows(
        oof_rows, matrix.query_ids, matrix.kinds
    )
    genuine = query_scores[query_kinds == "genuine"]
    skilled = query_scores[query_kinds == "skilled"]
    if genuine.size == 0 or skilled.size == 0:
        raise ValueError("cross-validation needs both genuine and skilled queries")
    threshold, eer = eer_threshold(genuine, skilled)
    return float(threshold), float(eer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--already-preprocessed", action="store_true")
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--references", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fusion-type",
        choices=("weighted", "logistic", "both"),
        default="both",
        help="Which fusion rule(s) to fit; 'both' keeps the lower out-of-fold EER.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Signer-grouped folds used to measure the out-of-fold threshold.",
    )
    parser.add_argument(
        "--max-random-queries",
        type=int,
        default=0,
        help=(
            "Deterministic cap on random query pairs. The default 0 fits fusion on "
            "genuine and skilled-forgery pairs only; random impostors are left to "
            "training.evaluate, which reports them."
        ),
    )
    args = parser.parse_args()

    root = (args.image_root or args.manifest.parent).resolve()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SiameseSignatureModel(
        pretrained=False, embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    records = read_split(args.manifest, root, {"val", "validation"})
    print(f"Read {len(records)} validation records")

    embeddings = embed_records(
        model,
        records,
        already_preprocessed=args.already_preprocessed,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    print(f"Embedded {len(embeddings)} records")

    descriptors = describe_records(
        records,
        already_preprocessed=args.already_preprocessed,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    print(f"Described {len(descriptors)} records")

    enrollment = enroll_and_build_queries(
        records,
        args.references,
        max_random_queries=args.max_random_queries,
    )
    print(f"Built {len(enrollment.queries)} queries")

    matrix = build_pair_matrix(embeddings, descriptors, enrollment)
    print(
        f"Built {len(matrix)} fitting pairs across "
        f"{len(enrollment.references)} signers "
        f"({int(matrix.labels.sum())} genuine)"
    )

    types = ["weighted", "logistic"] if args.fusion_type == "both" else [args.fusion_type]
    results: dict[str, tuple[FusionConfig, dict[str, object], float | None]] = {}
    for fusion_type in types:
        config, metadata = _fit_config(fusion_type, matrix.deep, matrix.hand, matrix.labels)
        print(f"Fitted {fusion_type} fusion on all pairs")
        try:
            threshold, oof_eer = _cross_validated_threshold(matrix, fusion_type, args.cv_folds)
            config = replace(config, threshold=threshold)
            print(f"{fusion_type}: out-of-fold EER {oof_eer:.4f} at threshold {threshold:.4f}")
        except ValueError as error:
            print(f"{fusion_type}: cross-validation skipped ({error})")
            oof_eer = None
        results[fusion_type] = (config, metadata, oof_eer)

    if len(results) > 1:
        best_type = min(
            results,
            key=lambda name: results[name][2] if results[name][2] is not None else 1.0,
        )
        print(f"Selected {best_type} fusion (lowest out-of-fold EER)")
    else:
        best_type = types[0]

    config, _, out_of_fold_eer = results[best_type]
    # Record how the config was produced so ``training.evaluate`` can detect a
    # preprocessing mismatch (``--already-preprocessed``) instead of silently
    # computing handcrafted features on different pixels than during fitting.
    config = replace(
        config,
        metadata={
            **config.metadata,
            "already_preprocessed": bool(args.already_preprocessed),
            "image_height": args.image_height,
            "image_width": args.image_width,
            "references": args.references,
            "fusion_type": best_type,
            "cv_folds": args.cv_folds,
            "out_of_fold_eer": out_of_fold_eer,
            "signers": len(enrollment.references),
            "fit_pairs": len(matrix),
            "checkpoint": str(args.checkpoint.resolve()),
            "manifest": str(args.manifest.resolve()),
        },
    )
    saved = save_fusion_config(config, args.output)
    print(f"Saved {best_type} fusion config to {saved}")
    print(json.dumps(config.to_dict(), indent=2))


if __name__ == "__main__":
    main()


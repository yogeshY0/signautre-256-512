"""Evaluate a trained checkpoint with FAR / FRR / EER on signer-disjoint splits.

Protocol (per signer): the first ``--references`` genuine signatures are enrolled as
references. A query's score is its mean cosine similarity to those references.
  * positives:        the signer's remaining genuine signatures
  * skilled negatives: forgeries of that signer
  * random negatives:  genuine signatures of other signers
The decision threshold is chosen at the EER on the validation signers and then
applied unchanged to the test signers, so the test numbers are not tuned on test data.

When ``--fusion-config`` is supplied, the same protocol is repeated with the fused
score from :mod:`scoring.fusion`. The deep-only baseline block is still reported
unchanged, so one run shows both results side by side. Fused decisions use the
threshold stored in the fusion config when the fitter recorded one, falling back to
the validation EER threshold otherwise.

Two extras beyond the headline numbers:

* ``per_dataset`` repeats the skilled EER and threshold behaviour for every source
  corpus (``BHSig260``, ``CEDAR``, ``SigResize``) on both validation and test, for the
  deep score and for the fused score. One weak corpus can otherwise hide behind the
  average.
* ``--target-far`` picks the threshold that meets a fixed FAR budget on the validation
  signers and reports the resulting FRR on validation and test, deep and fused. That is
  the operating point a checklist review usually cares about, unlike the EER.

Usage:
    python -m training.evaluate --checkpoint checkpoints/cedar_v1/best_model.pt \
        --manifest data/processed/manifest.csv --already-preprocessed
    python -m training.evaluate ... --fusion-config checkpoints/fusion/fusion.json \
        --target-far 0.001 --fusion-report checkpoints/fusion/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scoring.fusion import (
    Enrollment,
    FusionScorer,
    enroll_and_build_queries,
    load_fusion_config,
    score_queries,
)
from scoring.handcrafted_features import describe_records
from scoring.metrics import eer_threshold, rates_at_threshold, threshold_at_far
from training.stage3_metric_learning import (
    SiameseSignatureModel,
    embed_records,
    read_split,
)


def _evaluate_split(
    records,
    embeddings: np.ndarray,
    descriptors,
    references: int,
    scorer: FusionScorer | None,
    max_random_queries: int | None,
) -> tuple[dict[str, np.ndarray], Enrollment]:
    """Run the reference protocol and return scores by query kind plus the enrollment.

    The enrollment is returned because the per-dataset report needs the query order, and
    that is exactly the order ``score_queries`` appends its per-kind scores in.
    """
    enrollment: Enrollment = enroll_and_build_queries(
        records, references, max_random_queries=max_random_queries
    )
    return score_queries(embeddings, descriptors or [], enrollment, scorer), enrollment


def _query_datasets(enrollment: Enrollment, records) -> dict[str, list[str]]:
    """Return each query's source dataset, grouped by kind and in score order."""
    datasets: dict[str, list[str]] = {"genuine": [], "skilled": [], "random": []}
    for query in enrollment.queries:
        # Older manifests have no dataset column; label them instead of reporting "".
        datasets[query.kind].append(records[query.index].dataset or "unknown")
    return datasets


def _per_dataset_metrics(
    scores: dict[str, np.ndarray],
    datasets: dict[str, list[str]],
    prefix: str,
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Return EER and threshold behaviour for each source dataset.

    The threshold is the global one so the numbers show how a single decision point
    behaves per corpus rather than silently re-tuning per corpus. A genuine query is a
    signature of the enrolled signer, so its dataset is that signer's corpus.

    Random queries are left out: their query image comes from another signer's corpus,
    so grouping them by either dataset would be arbitrary.
    """
    genuine = scores[f"{prefix}_genuine"]
    skilled = scores[f"{prefix}_skilled"]
    if len(genuine) != len(datasets["genuine"]) or len(skilled) != len(datasets["skilled"]):
        raise RuntimeError(
            "score and dataset labels are not aligned; the per-dataset report is unreliable"
        )
    if genuine.size == 0 or skilled.size == 0:
        return {}

    results: dict[str, dict[str, float]] = {}
    for dataset in sorted(set(datasets["genuine"]) | set(datasets["skilled"])):
        genuine_mask = np.asarray([label == dataset for label in datasets["genuine"]])
        skilled_mask = np.asarray([label == dataset for label in datasets["skilled"]])
        if not genuine_mask.any() or not skilled_mask.any():
            continue
        dataset_genuine = genuine[genuine_mask]
        dataset_skilled = skilled[skilled_mask]
        far_skilled, frr = rates_at_threshold(dataset_genuine, dataset_skilled, threshold)
        _, eer_skilled = eer_threshold(dataset_genuine, dataset_skilled)
        results[dataset] = {
            "eer_skilled": eer_skilled,
            "far_skilled_at_threshold": far_skilled,
            "frr_at_threshold": frr,
            "genuine_queries": int(dataset_genuine.size),
            "skilled_forgeries": int(dataset_skilled.size),
        }
    return results


def _far_frr(
    scores: dict[str, np.ndarray], prefix: str, threshold: float
) -> dict[str, float]:
    """Return FAR/FRR on skilled forgeries at one fixed threshold."""
    far_skilled, frr = rates_at_threshold(
        scores[f"{prefix}_genuine"], scores[f"{prefix}_skilled"], threshold
    )
    return {"far_skilled_at_threshold": far_skilled, "frr_at_threshold": frr}


def _metrics(scores: dict[str, np.ndarray], prefix: str, threshold: float) -> dict[str, object]:
    """Return the FAR / FRR / EER block for one score family and split."""
    genuine = scores[f"{prefix}_genuine"]
    skilled = scores[f"{prefix}_skilled"]
    random = scores[f"{prefix}_random"]
    far_skilled, frr = rates_at_threshold(genuine, skilled, threshold)
    far_random, _ = rates_at_threshold(genuine, random, threshold)
    _, eer_skilled = eer_threshold(genuine, skilled)
    _, eer_random = eer_threshold(genuine, random)
    return {
        "far_skilled_at_threshold": far_skilled,
        "far_random_at_threshold": far_random,
        "frr_at_threshold": frr,
        "eer_skilled": eer_skilled,
        "eer_random": eer_random,
        "genuine_queries": int(len(genuine)),
        "skilled_forgeries": int(len(skilled)),
        "random_pairs": int(len(random)),
    }


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
    parser.add_argument(
        "--fusion-config",
        type=Path,
        help="Fusion config written by training.fit_fusion; enables fused scoring.",
    )
    parser.add_argument(
        "--fusion-report",
        type=Path,
        help="Optional path to also write the printed JSON report to disk.",
    )
    parser.add_argument(
        "--max-random-queries",
        type=int,
        default=None,
        help="Deterministic cap on random query pairs (default: use every pair).",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=None,
        help=(
            "Also report FRR at this FAR budget on skilled forgeries, e.g. 0.001. "
            "The threshold is chosen on the validation signers and applied unchanged."
        ),
    )
    args = parser.parse_args()
    if args.target_far is not None and not 0.0 < args.target_far < 1.0:
        parser.error("--target-far must be strictly between 0 and 1")

    root = (args.image_root or args.manifest.parent).resolve()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SiameseSignatureModel(
        pretrained=False, embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    scorer: FusionScorer | None = None
    config = None
    if args.fusion_config is not None:
        config = load_fusion_config(args.fusion_config)
        scorer = FusionScorer(config)
        fitted_preprocessed = bool(config.metadata.get("already_preprocessed", True))
        if fitted_preprocessed != bool(args.already_preprocessed):
            print(
                "warning: --already-preprocessed differs from the fusion config metadata; "
                "handcrafted features may be computed on different pixels than during fitting"
            )
        fitted_size = (
            config.metadata.get("image_height"),
            config.metadata.get("image_width"),
        )
        if fitted_size != (None, None) and fitted_size != (
            args.image_height,
            args.image_width,
        ):
            print(
                f"warning: fusion config was fitted at HxW {fitted_size}, "
                f"evaluating at ({args.image_height}, {args.image_width})"
            )

    deep: dict[str, dict[str, np.ndarray]] = {}
    fused: dict[str, dict[str, np.ndarray]] = {}
    datasets: dict[str, dict[str, list[str]]] = {}
    for name, splits in (("val", {"val", "validation"}), ("test", {"test"})):
        records = read_split(args.manifest, root, splits)
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
        # Descriptors are only needed when fusion is active; skip the cost otherwise.
        descriptors = (
            describe_records(
                records,
                already_preprocessed=args.already_preprocessed,
                image_height=args.image_height,
                image_width=args.image_width,
            )
            if scorer is not None
            else []
        )
        scores, enrollment = _evaluate_split(
            records,
            embeddings,
            descriptors,
            args.references,
            scorer,
            args.max_random_queries,
        )
        datasets[name] = _query_datasets(enrollment, records)
        deep[name] = {key: value for key, value in scores.items() if key.startswith("deep_")}
        if scorer is not None:
            fused[name] = {
                key: value for key, value in scores.items() if key.startswith("fused_")
            }

    # Threshold from validation signers; "skilled" is the harder and more relevant case.
    threshold, val_eer = eer_threshold(
        deep["val"]["deep_genuine"], deep["val"]["deep_skilled"]
    )
    report: dict[str, object] = {
        "threshold_from_validation": threshold,
        "validation_eer_skilled": val_eer,
        "test": _metrics(deep["test"], "deep", threshold),
        # Both splits: the validation row shows what the checkpoint was selected on,
        # the test row is the number to report. CEDAR is the usual weak spot.
        "per_dataset": {
            "val": _per_dataset_metrics(deep["val"], datasets["val"], "deep", threshold),
            "test": _per_dataset_metrics(deep["test"], datasets["test"], "deep", threshold),
        },
    }

    if scorer is not None and config is not None:
        fused_threshold, fused_val_eer = eer_threshold(
            fused["val"]["fused_genuine"], fused["val"]["fused_skilled"]
        )
        # Prefer the threshold the fitter measured out-of-fold; fall back to the
        # validation EER when the config carries no calibrated threshold.
        if config.threshold is not None:
            applied = float(config.threshold)
            threshold_source = "fusion_config_out_of_fold"
        else:
            applied = fused_threshold
            threshold_source = "validation_eer"
        report["fusion"] = {
            "config": str(Path(args.fusion_config).expanduser().resolve()),  # type: ignore[union-attr]
            "type": config.type,
            "features": list(config.features),
            "dropped_features": list(config.dropped_features),
            "threshold_from_validation": fused_threshold,
            "threshold_used": applied,
            "threshold_source": threshold_source,
            "validation_eer_skilled": fused_val_eer,
            "test": _metrics(fused["test"], "fused", applied),
            "per_dataset": {
                "val": _per_dataset_metrics(fused["val"], datasets["val"], "fused", applied),
                "test": _per_dataset_metrics(
                    fused["test"], datasets["test"], "fused", applied
                ),
            },
        }

    # A fixed FAR budget is what an operations team actually signs off on; EER is only
    # a summary. The threshold is always picked on validation signers and then applied
    # unchanged to test, so the test FRR is not tuned on test data.
    if args.target_far is not None:
        deep_far_threshold, deep_far = threshold_at_far(
            deep["val"]["deep_genuine"], deep["val"]["deep_skilled"], args.target_far
        )
        operating: dict[str, object] = {
            "target_far": args.target_far,
            "deep": {
                "threshold_from_validation": deep_far_threshold,
                "validation_far_skilled_achieved": deep_far,
                "validation": _far_frr(deep["val"], "deep", deep_far_threshold),
                "test": _far_frr(deep["test"], "deep", deep_far_threshold),
            },
        }
        if scorer is not None:
            fused_far_threshold, fused_far = threshold_at_far(
                fused["val"]["fused_genuine"],
                fused["val"]["fused_skilled"],
                args.target_far,
            )
            operating["fused"] = {
                "threshold_from_validation": fused_far_threshold,
                "validation_far_skilled_achieved": fused_far,
                "validation": _far_frr(fused["val"], "fused", fused_far_threshold),
                "test": _far_frr(fused["test"], "fused", fused_far_threshold),
            }
        report["operating_point"] = operating

    text = json.dumps(report, indent=2)
    print(text)
    if args.fusion_report is not None:
        destination = Path(args.fusion_report).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

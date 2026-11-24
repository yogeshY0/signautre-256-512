"""Evaluate a trained checkpoint with FAR / FRR / EER on signer-disjoint splits.

Protocol (per signer): the selected ``--references`` genuine signatures are enrolled as
references (``--reference-selection`` defaults to medoid). A query's score is its mean cosine similarity to those references.
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

* ``per_dataset`` repeats the skilled EER and threshold behaviour for every reporting
  group on both validation and test, for the deep score and for the fused score.
  ``BHSig260`` is split into ``BHSig260-Hindi`` and ``BHSig260-Bengali`` by signer id,
  and ``CEDAR`` / ``SigResize`` stay separate. One weak corpus can otherwise hide
  behind the average.
* ``--target-far`` picks the threshold that meets a fixed FAR budget on the validation
  signers and reports the resulting FRR on validation and test, deep and fused. That is
  the operating point a checklist review usually cares about, unlike the EER.
* ``--reference-counts`` sweeps the primary 3-reference protocol against 1 and 2
  references; the top-level block remains the primary ``--references`` count.
* ``test.auc_skilled`` / ``auc_random`` and ``confidence_intervals`` add AUC and 95%
  signer-level bootstrap CIs (default 1,000 resamples).
* ``operating_point_frr`` fixes the threshold at 10% validation FRR and reports skilled
  and random forgery acceptance on test.
* ``per_signer`` lists the writers with the highest FRR and FAR; ``per_dataset`` splits
  BHSig260 into Hindi and Bengali and keeps CEDAR separate.
* ``calibration`` compares one global threshold with a per-dataset threshold chosen on
  that dataset's validation signers, then reports FAR/FRR on test under both. It shows
  whether a single decision point is miscalibrated for a corpus.
* ``deep_vs_fused`` prints the skilled EER of the deep and fused scores per dataset and
  recommends ``deep_only`` unless fusion improves *every* corpus on validation. Fusion
  is opt-in, so the deep baseline is the default run.

Usage:
    python -m training.evaluate --checkpoint checkpoints/cedar_v1/best_model.pt \
        --manifest data/processed/manifest.csv --already-preprocessed
    python -m training.evaluate ... --fusion-config checkpoints/fusion/fusion.json \
        --target-far 0.001 --fusion-report checkpoints/fusion/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
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
from scoring.metrics import (
    auc,
    bootstrap_ci,
    eer_threshold,
    rates_at_threshold,
    threshold_at_far,
    threshold_at_frr,
)
from scoring.reference_selection import MODES
from training.stage3_metric_learning import (
    SiameseSignatureModel,
    embed_records,
    read_split,
    source_group,
)


def _evaluate_split(
    records,
    embeddings: np.ndarray,
    descriptors,
    references: int,
    scorer: FusionScorer | None,
    max_random_queries: int | None,
    *,
    reference_selection: str = "first",
    already_preprocessed: bool = True,
    ink: str = "auto",
    image_height: int = 256,
    image_width: int = 512,
) -> tuple[dict[str, np.ndarray], Enrollment]:
    """Run the reference protocol and return scores by query kind plus the enrollment.

    The enrollment is returned because the per-dataset report needs the query order, and
    that is exactly the order ``score_queries`` appends its per-kind scores in.
    """
    enrollment: Enrollment = enroll_and_build_queries(
        records,
        references,
        max_random_queries=max_random_queries,
        reference_selection=reference_selection,
        embeddings=embeddings,
        already_preprocessed=already_preprocessed,
        ink=ink,
        image_height=image_height,
        image_width=image_width,
    )
    return score_queries(embeddings, descriptors or [], enrollment, scorer), enrollment


def _query_datasets(enrollment: Enrollment, records) -> dict[str, list[str]]:
    """Return each query's source dataset, grouped by kind and in score order."""
    datasets: dict[str, list[str]] = {"genuine": [], "skilled": [], "random": []}
    for query in enrollment.queries:
        # Older manifests have no dataset column; label them instead of reporting "".
        datasets[query.kind].append(source_group(records[query.index]))
    return datasets


def _query_signers(enrollment: Enrollment) -> dict[str, list[str]]:
    """Return the target signer for each query, grouped by kind."""
    kinds = ("genuine", "skilled", "random")
    return {
        kind: [q.target_signer for q in enrollment.queries if q.kind == kind]
        for kind in kinds
    }


def _per_dataset_metrics(
    scores: dict[str, np.ndarray],
    datasets: dict[str, list[str]],
    prefix: str,
    threshold: float,
) -> dict[str, dict[str, object]]:
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
        far_skilled, frr = rates_at_threshold(
            dataset_genuine, dataset_skilled, threshold
        )
        random_scores = scores[f"{prefix}_random"]
        random_labels = datasets["random"]
        random_mask = np.asarray([label == dataset for label in random_labels])
        dataset_random = (
            random_scores[random_mask]
            if random_mask.size == random_scores.size
            else np.asarray([])
        )
        far_random = None
        if dataset_random.size:
            far_random, _ = rates_at_threshold(
                dataset_genuine, dataset_random, threshold
            )
        _, eer_skilled = eer_threshold(dataset_genuine, dataset_skilled)
        results[dataset] = {
            "eer_skilled": eer_skilled,
            "far_skilled_at_threshold": far_skilled,
            "frr_at_threshold": frr,
            "far_random_at_threshold": far_random,
            "auc_skilled": auc(dataset_genuine, dataset_skilled),
            "auc_random": (
                auc(dataset_genuine, dataset_random)
                if dataset_random.size
                else None
            ),
            "genuine_queries": int(dataset_genuine.size),
            "skilled_forgeries": int(dataset_skilled.size),
        }
    return results


def _far_frr(
    scores: dict[str, np.ndarray], prefix: str, threshold: float
) -> dict[str, float]:
    """Return FAR/FRR on skilled forgeries at one fixed threshold."""
    genuine = scores[f"{prefix}_genuine"]
    skilled = scores[f"{prefix}_skilled"]
    random_scores = scores[f"{prefix}_random"]
    far_skilled, frr = rates_at_threshold(genuine, skilled, threshold)
    far_random, _ = rates_at_threshold(genuine, random_scores, threshold)
    return {
        "far_skilled_at_threshold": far_skilled,
        "far_random_at_threshold": far_random,
        "frr_at_threshold": frr,
    }


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
        "auc_skilled": auc(genuine, skilled),
        "auc_random": auc(genuine, random),
        "genuine_queries": int(len(genuine)),
        "skilled_forgeries": int(len(skilled)),
        "random_pairs": int(len(random)),
    }


def _dataset_scores(
    scores: dict[str, np.ndarray],
    datasets: dict[str, list[str]],
    prefix: str,
    kind: str,
    dataset: str,
) -> np.ndarray:
    """Return one kind of score for one dataset, in score order."""
    mask = np.asarray([label == dataset for label in datasets[kind]], dtype=bool)
    return scores[f"{prefix}_{kind}"][mask]


def _rates_for_dataset(
    scores: dict[str, np.ndarray],
    datasets: dict[str, list[str]],
    prefix: str,
    dataset: str,
    threshold: float,
) -> dict[str, object] | None:
    """Return test FAR/FRR for one dataset at one fixed threshold."""
    genuine = _dataset_scores(scores, datasets, prefix, "genuine", dataset)
    skilled = _dataset_scores(scores, datasets, prefix, "skilled", dataset)
    if genuine.size == 0 or skilled.size == 0:
        return None
    far, frr = rates_at_threshold(genuine, skilled, threshold)
    return {
        "far_skilled_at_threshold": float(far),
        "frr_at_threshold": float(frr),
        "genuine_queries": int(genuine.size),
        "skilled_forgeries": int(skilled.size),
    }


def _calibration_block(
    val_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    val_datasets: dict[str, list[str]],
    test_datasets: dict[str, list[str]],
    prefix: str,
    target_far: float | None,
) -> dict[str, object]:
    """Compare one global threshold with per-dataset thresholds chosen on validation.

    Every threshold is fitted on the validation signers only. The test table then shows
    whether a single decision point is miscalibrated for a corpus: the same global
    threshold can be fine on BHSig260 and too loose on CEDAR. The per-dataset threshold
    is fitted on that dataset's validation split and applied unchanged to test, so it is
    a legitimate comparison rather than test-set tuning.

    Args:
        val_scores/test_scores: Score dictionaries with ``f"{prefix}_genuine"`` and
            ``f"{prefix}_skilled"`` arrays.
        val_datasets/test_datasets: Per-query dataset labels aligned with the scores.
        prefix: ``"deep"`` or ``"fused"``.
        target_far: Optional fixed FAR budget; when set, a second threshold pair is
            computed at that budget.

    Returns:
        A JSON-serialisable block; empty when there are no validation scores.
    """
    genuine_val = val_scores[f"{prefix}_genuine"]
    skilled_val = val_scores[f"{prefix}_skilled"]
    if genuine_val.size == 0 or skilled_val.size == 0:
        return {}
    global_threshold, global_eer = eer_threshold(genuine_val, skilled_val)
    block: dict[str, object] = {
        "global_threshold": float(global_threshold),
        "global_validation_eer_skilled": float(global_eer),
        "target_far": target_far,
        "datasets": {},
    }
    global_far_threshold: float | None = None
    if target_far is not None:
        global_far_threshold, achieved = threshold_at_far(
            genuine_val, skilled_val, target_far
        )
        block["global_threshold_at_far"] = float(global_far_threshold)
        block["global_validation_far_achieved"] = float(achieved)

    datasets: dict[str, dict[str, object]] = {}
    for dataset in sorted(set(val_datasets["genuine"]) | set(val_datasets["skilled"])):
        dataset_genuine = _dataset_scores(
            val_scores, val_datasets, prefix, "genuine", dataset
        )
        dataset_skilled = _dataset_scores(
            val_scores, val_datasets, prefix, "skilled", dataset
        )
        if dataset_genuine.size == 0 or dataset_skilled.size == 0:
            continue
        dataset_threshold, dataset_eer = eer_threshold(dataset_genuine, dataset_skilled)
        entry: dict[str, object] = {
            "validation_eer_skilled": float(dataset_eer),
            "threshold": float(dataset_threshold),
            "test_at_global_threshold": _rates_for_dataset(
                test_scores, test_datasets, prefix, dataset, float(global_threshold)
            ),
            "test_at_dataset_threshold": _rates_for_dataset(
                test_scores, test_datasets, prefix, dataset, float(dataset_threshold)
            ),
        }
        if global_far_threshold is not None and target_far is not None:
            dataset_far_threshold, dataset_far = threshold_at_far(
                dataset_genuine, dataset_skilled, target_far
            )
            entry["threshold_at_far"] = float(dataset_far_threshold)
            entry["validation_far_achieved"] = float(dataset_far)
            entry["test_at_global_far_threshold"] = _rates_for_dataset(
                test_scores, test_datasets, prefix, dataset, global_far_threshold
            )
            entry["test_at_dataset_far_threshold"] = _rates_for_dataset(
                test_scores, test_datasets, prefix, dataset, dataset_far_threshold
            )
        datasets[dataset] = entry
    block["datasets"] = datasets
    return block


def _fusion_comparison(
    deep_per_dataset: dict[str, dict[str, dict[str, object]]],
    fused_per_dataset: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object]:
    """Summarise whether fusion improves every corpus on validation.

    Fusion is report-only unless it lowers the skilled EER for *every* dataset, not just
    the pooled average. A corpus that gets worse is a miscalibrated handcrafted branch,
    not a rounding error, so the recommendation stays ``deep_only``.
    """
    rows: dict[str, dict[str, object]] = {}
    for dataset in sorted(set(deep_per_dataset["val"]) & set(fused_per_dataset["val"])):
        deep_val = deep_per_dataset["val"][dataset].get("eer_skilled")
        fused_val = fused_per_dataset["val"][dataset].get("eer_skilled")
        if deep_val is None or fused_val is None:
            continue
        deep_test = deep_per_dataset["test"].get(dataset, {}).get("eer_skilled")
        fused_test = fused_per_dataset["test"].get(dataset, {}).get("eer_skilled")
        rows[dataset] = {
            "deep_val": float(deep_val),
            "fused_val": float(fused_val),
            "delta_val": float(fused_val) - float(deep_val),
            "deep_test": None if deep_test is None else float(deep_test),
            "fused_test": None if fused_test is None else float(fused_test),
            "delta_test": (
                None
                if deep_test is None or fused_test is None
                else float(fused_test) - float(deep_test)
            ),
        }
    helps = bool(rows) and all(
        row["fused_val"] < row["deep_val"] for row in rows.values()
    )
    return {
        "datasets": rows,
        "helps_every_dataset": helps,
        "recommendation": "keep_fusion" if helps else "deep_only",
    }


def _number(value: object, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(float(value), spec)


def _print_dataset_comparison(
    deep_per_dataset: dict[str, dict[str, dict[str, object]]],
    fused_per_dataset: dict[str, dict[str, dict[str, object]]] | None = None,
) -> None:
    """Print deep (and, when available, fused) skilled EER per dataset."""
    print("--- deep vs fused per dataset (skilled EER) ---", file=sys.stderr)
    print(
        f"{'dataset':<12}{'deep_val':>10}{'deep_test':>11}"
        f"{'fused_val':>11}{'fused_test':>12}{'delta_test':>12}",
        file=sys.stderr,
    )
    datasets = set(deep_per_dataset["val"]) | set(deep_per_dataset["test"])
    if fused_per_dataset is not None:
        datasets |= set(fused_per_dataset["val"]) | set(fused_per_dataset["test"])
    for dataset in sorted(datasets):
        deep_val = deep_per_dataset["val"].get(dataset, {}).get("eer_skilled")
        deep_test = deep_per_dataset["test"].get(dataset, {}).get("eer_skilled")
        fused_val = (
            fused_per_dataset["val"].get(dataset, {}).get("eer_skilled")
            if fused_per_dataset is not None
            else None
        )
        fused_test = (
            fused_per_dataset["test"].get(dataset, {}).get("eer_skilled")
            if fused_per_dataset is not None
            else None
        )
        delta = (
            None
            if deep_test is None or fused_test is None
            else float(fused_test) - float(deep_test)
        )
        print(
            f"{dataset:<12}{_number(deep_val):>10}{_number(deep_test):>11}"
            f"{_number(fused_val):>11}{_number(fused_test):>12}{_number(delta):>12}",
            file=sys.stderr,
        )
    print(file=sys.stderr)


def _print_calibration(name: str, block: dict[str, object]) -> None:
    """Print the global-threshold versus per-dataset-threshold comparison."""
    if not block or not block.get("datasets"):
        return
    print(
        f"--- calibration ({name}): global threshold vs per-dataset threshold on test ---",
        file=sys.stderr,
    )
    print(
        f"  global threshold from validation: {float(block['global_threshold']):.4f}",
        file=sys.stderr,
    )
    header = (
        f"{'dataset':<12}{'thr_ds':>9}{'FAR_global':>12}{'FRR_global':>12}"
        f"{'FAR_ds':>9}{'FRR_ds':>9}"
    )
    print(header, file=sys.stderr)
    for dataset, entry in block["datasets"].items():  # type: ignore[union-attr]
        at_global = entry.get("test_at_global_threshold") or {}
        at_dataset = entry.get("test_at_dataset_threshold") or {}
        print(
            f"{dataset:<12}{_number(entry.get('threshold')):>9}"
            f"{_number(at_global.get('far_skilled_at_threshold')):>12}"
            f"{_number(at_global.get('frr_at_threshold')):>12}"
            f"{_number(at_dataset.get('far_skilled_at_threshold')):>9}"
            f"{_number(at_dataset.get('frr_at_threshold')):>9}",
            file=sys.stderr,
        )
    if block.get("target_far") is not None:
        print(
            f"  at FAR budget {float(block['target_far']):g}: "
            f"global threshold {float(block['global_threshold_at_far']):.4f}",
            file=sys.stderr,
        )
        print(header, file=sys.stderr)
        for dataset, entry in block["datasets"].items():  # type: ignore[union-attr]
            at_global = entry.get("test_at_global_far_threshold") or {}
            at_dataset = entry.get("test_at_dataset_far_threshold") or {}
            print(
                f"{dataset:<12}{_number(entry.get('threshold_at_far')):>9}"
                f"{_number(at_global.get('far_skilled_at_threshold')):>12}"
                f"{_number(at_global.get('frr_at_threshold')):>12}"
                f"{_number(at_dataset.get('far_skilled_at_threshold')):>9}"
                f"{_number(at_dataset.get('frr_at_threshold')):>9}",
                file=sys.stderr,
            )
    print(file=sys.stderr)


def _fmt_ci(value):
    if value is None:
        return "n/a"
    return f"[{value[0]:.3f},{value[1]:.3f}]"


def _print_confidence(name, block):
    if not block:
        return
    resamples = block["resamples"]
    print(f"--- 95% bootstrap CI ({name}, {resamples} resamples) ---", file=sys.stderr)
    header = "group".ljust(20) + "AUC_skilled".rjust(18) + "EER_skilled".rjust(18)
    print(header, file=sys.stderr)
    rows = [("pooled", block["pooled"])] + list(block["per_dataset"].items())
    for group, values in rows:
        auc_ci = _fmt_ci(values.get("auc_skilled"))
        eer_ci = _fmt_ci(values.get("eer_skilled"))
        print(f"{group:<20}{auc_ci:>18}{eer_ci:>18}", file=sys.stderr)
    print(file=sys.stderr)


def _print_operating_frr(name, block):
    if not block:
        return
    target = block["target_frr"]
    threshold = block["threshold_from_validation"]
    achieved = block["validation_frr_achieved"]
    print(f"--- operating point at {target:.1%} validation FRR ({name}) ---", file=sys.stderr)
    print(f"  threshold {threshold:.4f} (validation FRR {achieved:.4f})", file=sys.stderr)
    test = block["test"]
    skilled = test["far_skilled_at_threshold"]
    random_score = test["far_random_at_threshold"]
    rejected = test["frr_at_threshold"]
    print(f"  test: skilled accepted {skilled:.4f} random accepted {random_score:.4f} genuine rejected {rejected:.4f}", file=sys.stderr)
    for dataset, row in block.get("per_dataset", {}).items():
        thr = row["threshold_from_validation"]
        sk = row["test_far_skilled"]
        fr = row["test_frr"]
        print(f"    {dataset:<12} thr {thr:.4f} skilled {sk:.4f} FRR {fr:.4f}", file=sys.stderr)
    print(file=sys.stderr)


def _print_per_signer(name, block):
    if not block:
        return
    print(f"--- worst writers ({name}) ---", file=sys.stderr)
    print("signer / group / FRR / FAR_skilled / n_genuine / n_skilled", file=sys.stderr)
    for row in block["worst_frr"]:
        signer = row["signer"]
        group = row["group"]
        frr = row["frr"]
        far = row["far_skilled"]
        n_gen = row["genuine_queries"]
        n_skl = row["skilled_forgeries"]
        print(f"  {signer:<24} {group:<18} FRR {frr:.3f} FAR {far:.3f} n={n_gen}/{n_skl}", file=sys.stderr)
    print(file=sys.stderr)


def _print_reference_sweep(sweep):
    if not sweep:
        return
    print("--- reference-count sweep (deep) ---", file=sys.stderr)
    print("refs   val_EER   val_AUC   test_EER   test_AUC   test_EER_random", file=sys.stderr)
    for refs, block in sweep.items():
        val_eer = block["validation_eer_skilled"]
        val_auc = block.get("validation_auc_skilled")
        test = block["test"]
        test_eer = test["eer_skilled"]
        test_auc = test.get("auc_skilled")
        test_random = test["eer_random"]
        auc_text = "n/a" if test_auc is None else f"{test_auc:.3f}"
        val_auc_text = "n/a" if val_auc is None else f"{val_auc:.3f}"
        print(f"{refs:>4} {val_eer:>8.4f} {val_auc_text:>9} {test_eer:>9.4f} {auc_text:>10} {test_random:>16.4f}", file=sys.stderr)
    print(file=sys.stderr)


def _confidence_block(
    scores: dict[str, np.ndarray],
    signers: dict[str, list[str]],
    datasets: dict[str, list[str]],
    prefix: str,
    resamples: int,
    seed: int = 0,
) -> dict[str, object]:
    """95% signer-level bootstrap CIs for AUC and EER on the test split."""
    if resamples < 1:
        return {}
    def ci(kind: str, metric) -> tuple[float, float] | None:
        return bootstrap_ci(
            scores[f"{prefix}_genuine"],
            scores[f"{prefix}_{kind}"],
            signers["genuine"],
            signers[kind],
            metric,
            resamples=resamples,
            seed=seed,
        )
    eer_metric = lambda g, n: eer_threshold(g, n)[1]
    pooled = {
        "auc_skilled": ci("skilled", auc),
        "eer_skilled": ci("skilled", eer_metric),
        "auc_random": ci("random", auc),
        "eer_random": ci("random", eer_metric),
    }
    per_dataset = {}
    for dataset in sorted(set(datasets["genuine"]) | set(datasets["skilled"])):
        masks = {
            kind: np.asarray([label == dataset for label in datasets[kind]])
            for kind in ("genuine", "skilled", "random")
        }
        if not masks["genuine"].any() or not masks["skilled"].any():
            continue
        def dataset_ci(kind: str, metric):
            return bootstrap_ci(
                scores[f"{prefix}_genuine"][masks["genuine"]],
                scores[f"{prefix}_{kind}"][masks[kind]],
                [s for s, keep in zip(signers["genuine"], masks["genuine"]) if keep],
                [s for s, keep in zip(signers[kind], masks[kind]) if keep],
                metric,
                resamples=resamples,
                seed=seed,
            )
        per_dataset[dataset] = {
            "auc_skilled": dataset_ci("skilled", auc),
            "eer_skilled": dataset_ci("skilled", eer_metric),
        }
    return {"resamples": resamples, "pooled": pooled, "per_dataset": per_dataset}


def _operating_point_frr(
    val_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    prefix: str,
    target_frr: float,
    val_datasets: dict[str, list[str]] | None = None,
    test_datasets: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    """Threshold at a fixed validation FRR, then test skilled/random acceptance."""
    threshold, achieved = threshold_at_frr(
        val_scores[f"{prefix}_genuine"],
        val_scores[f"{prefix}_skilled"],
        target_frr,
    )
    block = {
        "target_frr": target_frr,
        "threshold_from_validation": threshold,
        "validation_frr_achieved": achieved,
        "validation": _far_frr(val_scores, prefix, threshold),
        "test": _far_frr(test_scores, prefix, threshold),
    }
    if val_datasets is not None and test_datasets is not None:
        per_dataset = {}
        for dataset in sorted(set(val_datasets["genuine"]) | set(val_datasets["skilled"])):
            vg = _dataset_scores(val_scores, val_datasets, prefix, "genuine", dataset)
            vs = _dataset_scores(val_scores, val_datasets, prefix, "skilled", dataset)
            if vg.size == 0 or vs.size == 0:
                continue
            ds_threshold, ds_achieved = threshold_at_frr(vg, vs, target_frr)
            tg = _dataset_scores(test_scores, test_datasets, prefix, "genuine", dataset)
            ts = _dataset_scores(test_scores, test_datasets, prefix, "skilled", dataset)
            if tg.size == 0 or ts.size == 0:
                continue
            far, frr = rates_at_threshold(tg, ts, ds_threshold)
            per_dataset[dataset] = {
                "threshold_from_validation": ds_threshold,
                "validation_frr_achieved": ds_achieved,
                "test_far_skilled": far,
                "test_frr": frr,
            }
        block["per_dataset"] = per_dataset
    return block


def _per_signer_block(
    scores: dict[str, np.ndarray],
    signers: dict[str, list[str]],
    datasets: dict[str, list[str]],
    prefix: str,
    threshold: float,
    top: int,
    group_by_signer: dict[str, str],
) -> dict[str, object]:
    """Per-signer FAR/FRR at one threshold, with the worst writers ranked."""
    genuine = scores[f"{prefix}_genuine"]
    skilled = scores[f"{prefix}_skilled"]
    random_scores = scores[f"{prefix}_random"]
    rows = []
    for signer in sorted(set(signers["genuine"]) | set(signers["skilled"])):
        gmask = np.asarray([s == signer for s in signers["genuine"]])
        smask = np.asarray([s == signer for s in signers["skilled"]])
        if not gmask.any() and not smask.any():
            continue
        g = genuine[gmask] if gmask.any() else np.asarray([])
        s = skilled[smask] if smask.any() else np.asarray([])
        frr = None
        far = None
        if g.size and s.size:
            far, frr = rates_at_threshold(g, s, threshold)
        rmask = np.asarray([x == signer for x in signers["random"]])
        r = random_scores[rmask] if rmask.any() else np.asarray([])
        far_random = None
        if g.size and r.size:
            far_random, _ = rates_at_threshold(g, r, threshold)
        rows.append({
            "signer": signer,
            "group": group_by_signer.get(signer, "unknown"),
            "frr": frr,
            "far_skilled": far,
            "far_random": far_random,
            "genuine_queries": int(g.size),
            "skilled_forgeries": int(s.size),
            "random_queries": int(r.size),
        })
    worst_frr = sorted(
        [row for row in rows if row["frr"] is not None],
        key=lambda row: (-float(row["frr"]), row["signer"]),
    )[:top]
    worst_far = sorted(
        [row for row in rows if row["far_skilled"] is not None],
        key=lambda row: (-float(row["far_skilled"]), row["signer"]),
    )[:top]
    return {
        "signers_evaluated": len(rows),
        "worst_frr": worst_frr,
        "worst_far": worst_far,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--already-preprocessed", action="store_true")
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--references", type=int, default=3)
    parser.add_argument("--ink", choices=("auto", "light", "dark"), default="auto")
    parser.add_argument(
        "--reference-counts",
        default="1,2,3",
        help="Comma-separated reference counts to sweep (primary comes from --references).",
    )
    parser.add_argument(
        "--reference-selection",
        choices=MODES,
        default="medoid",
        help="How genuine references are chosen (default: medoid).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--fusion-config",
        type=Path,
        help="Fusion config; fusion is opt-in and report-only.",
    )
    parser.add_argument(
        "--fusion-report",
        type=Path,
        help="Also write the JSON report here.",
    )
    parser.add_argument(
        "--max-random-queries",
        type=int,
        default=None,
        help="Deterministic cap on random pairs (default: every pair).",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=None,
        help="Also report at this FAR budget, e.g. 0.001.",
    )
    parser.add_argument(
        "--target-frr",
        type=float,
        default=0.10,
        help="Validation FRR for the operating point (default: 0.10).",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--top-signers", type=int, default=10)
    args = parser.parse_args()
    counts = sorted(
        {int(item) for item in args.reference_counts.split(",") if item.strip()}
    )
    if not counts:
        counts = [args.references]
    if args.references not in counts:
        counts = sorted(counts + [args.references])
    if any(count < 1 for count in counts):
        parser.error("reference counts must be at least 1")
    if not 0.0 < args.target_frr < 1.0:
        parser.error("--target-frr must be strictly between 0 and 1")
    if args.target_far is not None and not 0.0 < args.target_far < 1.0:
        parser.error("--target-far must be strictly between 0 and 1")
    if args.bootstrap < 0:
        parser.error("--bootstrap cannot be negative")

    root = (args.image_root or args.manifest.parent).resolve()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SiameseSignatureModel(
        pretrained=False, embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    scorer = None
    config = None
    if args.fusion_config is not None:
        config = load_fusion_config(args.fusion_config)
        scorer = FusionScorer(config)
        fitted_preprocessed = bool(config.metadata.get("already_preprocessed", True))
        if fitted_preprocessed != bool(args.already_preprocessed):
            print("warning: fusion preprocessing metadata differs", file=sys.stderr)
        fitted_references = config.metadata.get("references")
        if fitted_references is not None and int(fitted_references) != args.references:
            print("warning: fusion was fitted with a different reference count", file=sys.stderr)
        fitted_selection = config.metadata.get("reference_selection")
        if fitted_selection is not None and fitted_selection != args.reference_selection:
            print("warning: fusion was fitted with a different reference selection", file=sys.stderr)
    print(f"references primary={args.references} sweep={counts} selection={args.reference_selection}", file=sys.stderr)

    deep_by_count = {count: {} for count in counts}
    fused_by_count = {count: {} for count in counts}
    datasets_by_count = {count: {} for count in counts}
    signers_by_count = {count: {} for count in counts}
    skipped_by_count = {count: [] for count in counts}
    groups_by_split = {}
    for name, splits in (("val", {"val", "validation"}), ("test", {"test"})):
        records = read_split(args.manifest, root, splits)
        groups = {}
        for record in records:
            groups.setdefault(record.signer_id, source_group(record))
        groups_by_split[name] = groups
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
        for count in counts:
            scores, enrollment = _evaluate_split(
                records,
                embeddings,
                descriptors,
                count,
                scorer,
                args.max_random_queries,
                reference_selection=args.reference_selection,
                already_preprocessed=args.already_preprocessed,
                ink=args.ink,
                image_height=args.image_height,
                image_width=args.image_width,
            )
            deep_by_count[count][name] = {
                key: value for key, value in scores.items()
                if key.startswith("deep_")
            }
            if scorer is not None:
                fused_by_count[count][name] = {
                    key: value for key, value in scores.items()
                    if key.startswith("fused_")
                }
            datasets_by_count[count][name] = _query_datasets(enrollment, records)
            signers_by_count[count][name] = _query_signers(enrollment)
            if name == "test":
                skipped_by_count[count] = list(enrollment.skipped_signers)
            split_signers = signers_by_count[count][name]
            genuine_count = len(split_signers["genuine"])
            skilled_count = len(split_signers["skilled"])
            random_count = len(split_signers["random"])
            print(
                f"{name} refs={count}: genuine={genuine_count} "
                f"skilled={skilled_count} random={random_count} "
                f"skipped={len(skipped_by_count[count])}",
                file=sys.stderr,
            )

    primary = args.references
    deep = deep_by_count[primary]
    fused = fused_by_count[primary]
    datasets = datasets_by_count[primary]
    signers = signers_by_count[primary]
    groups_test = groups_by_split["test"]
    threshold, val_eer = eer_threshold(
        deep["val"]["deep_genuine"], deep["val"]["deep_skilled"]
    )
    report = {
        "references": primary,
        "reference_selection": args.reference_selection,
        "skipped_signers": skipped_by_count[primary],
        "threshold_from_validation": threshold,
        "validation_eer_skilled": val_eer,
        "validation_auc_skilled": auc(
            deep["val"]["deep_genuine"], deep["val"]["deep_skilled"]
        ),
        "test": _metrics(deep["test"], "deep", threshold),
        "per_dataset": {
            "val": _per_dataset_metrics(
                deep["val"], datasets["val"], "deep", threshold
            ),
            "test": _per_dataset_metrics(
                deep["test"], datasets["test"], "deep", threshold
            ),
        },
        "confidence_intervals": {
            "deep": _confidence_block(
                deep["test"],
                signers["test"],
                datasets["test"],
                "deep",
                args.bootstrap,
                args.bootstrap_seed,
            )
        },
        "operating_point_frr": {
            "deep": _operating_point_frr(
                deep["val"],
                deep["test"],
                "deep",
                args.target_frr,
                datasets["val"],
                datasets["test"],
            )
        },
        "per_signer": {
            "deep": _per_signer_block(
                deep["test"],
                signers["test"],
                datasets["test"],
                "deep",
                threshold,
                args.top_signers,
                groups_test,
            )
        },
    }

    fused_per_dataset = None
    if scorer is not None and config is not None:
        fused_threshold, fused_val_eer = eer_threshold(
            fused["val"]["fused_genuine"], fused["val"]["fused_skilled"]
        )
        if config.threshold is not None:
            applied = float(config.threshold)
            threshold_source = "fusion_config_out_of_fold"
        else:
            applied = fused_threshold
            threshold_source = "validation_eer"
        fused_per_dataset = {
            "val": _per_dataset_metrics(
                fused["val"], datasets["val"], "fused", applied
            ),
            "test": _per_dataset_metrics(
                fused["test"], datasets["test"], "fused", applied
            ),
        }
        report["fusion"] = {
            "config": str(Path(args.fusion_config).expanduser().resolve()),
            "type": config.type,
            "features": list(config.features),
            "dropped_features": list(config.dropped_features),
            "threshold_from_validation": fused_threshold,
            "threshold_used": applied,
            "threshold_source": threshold_source,
            "validation_eer_skilled": fused_val_eer,
            "test": _metrics(fused["test"], "fused", applied),
            "per_dataset": fused_per_dataset,
            "confidence_intervals": _confidence_block(
                fused["test"],
                signers["test"],
                datasets["test"],
                "fused",
                args.bootstrap,
                args.bootstrap_seed,
            ),
            "operating_point_frr": _operating_point_frr(
                fused["val"],
                fused["test"],
                "fused",
                args.target_frr,
                datasets["val"],
                datasets["test"],
            ),
            "per_signer": _per_signer_block(
                fused["test"],
                signers["test"],
                datasets["test"],
                "fused",
                applied,
                args.top_signers,
                groups_test,
            ),
        }
        report["confidence_intervals"]["fused"] = report["fusion"]["confidence_intervals"]
        report["operating_point_frr"]["fused"] = report["fusion"]["operating_point_frr"]
        report["per_signer"]["fused"] = report["fusion"]["per_signer"]

    if args.target_far is not None:
        deep_far_threshold, deep_far = threshold_at_far(
            deep["val"]["deep_genuine"],
            deep["val"]["deep_skilled"],
            args.target_far,
        )
        operating = {
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

    report["deep_vs_fused"] = (
        _fusion_comparison(report["per_dataset"], fused_per_dataset)
        if fused_per_dataset is not None
        else None
    )
    calibration = {
        "deep": _calibration_block(
            deep["val"],
            deep["test"],
            datasets["val"],
            datasets["test"],
            "deep",
            args.target_far,
        )
    }
    if fused_per_dataset is not None:
        calibration["fused"] = _calibration_block(
            fused["val"],
            fused["test"],
            datasets["val"],
            datasets["test"],
            "fused",
            args.target_far,
        )
    report["calibration"] = calibration

    sweep = {}
    for count in counts:
        count_val = deep_by_count[count]["val"]
        count_test = deep_by_count[count]["test"]
        count_datasets = datasets_by_count[count]
        count_signers = signers_by_count[count]
        count_threshold, count_eer = eer_threshold(
            count_val["deep_genuine"], count_val["deep_skilled"]
        )
        sweep[str(count)] = {
            "references": count,
            "validation_eer_skilled": count_eer,
            "validation_auc_skilled": auc(
                count_val["deep_genuine"], count_val["deep_skilled"]
            ),
            "test": _metrics(count_test, "deep", count_threshold),
            "per_dataset": {
                "val": _per_dataset_metrics(
                    count_val, count_datasets["val"], "deep", count_threshold
                ),
                "test": _per_dataset_metrics(
                    count_test, count_datasets["test"], "deep", count_threshold
                ),
            },
            "confidence_intervals": _confidence_block(
                count_test,
                count_signers["test"],
                count_datasets["test"],
                "deep",
                args.bootstrap,
                args.bootstrap_seed,
            ),
            "operating_point_frr": _operating_point_frr(
                count_val,
                count_test,
                "deep",
                args.target_frr,
                count_datasets["val"],
                count_datasets["test"],
            ),
            "skipped_signers": skipped_by_count[count],
        }
    report["reference_sweep"] = sweep

    _print_dataset_comparison(report["per_dataset"], fused_per_dataset)
    if report["deep_vs_fused"] is not None:
        verdict = report["deep_vs_fused"]
        recommendation = verdict["recommendation"]
        helps = verdict["helps_every_dataset"]
        print(
            f"FUSION VERDICT: {recommendation} "
            f"(helps every dataset: {helps})",
            file=sys.stderr,
        )
    _print_calibration("deep", calibration["deep"])
    if "fused" in calibration:
        _print_calibration("fused", calibration["fused"])
    _print_confidence("deep", report["confidence_intervals"]["deep"])
    if "fused" in report["confidence_intervals"]:
        _print_confidence("fused", report["confidence_intervals"]["fused"])
    _print_operating_frr("deep", report["operating_point_frr"]["deep"])
    if "fused" in report["operating_point_frr"]:
        _print_operating_frr("fused", report["operating_point_frr"]["fused"])
    _print_per_signer("deep", report["per_signer"]["deep"])
    if "fused" in report["per_signer"]:
        _print_per_signer("fused", report["per_signer"]["fused"])
    _print_reference_sweep(sweep)
    print(file=sys.stderr)

    text = json.dumps(report, indent=2)
    print(text)
    if args.fusion_report is not None:
        destination = Path(args.fusion_report).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()

"""Per-dataset audit of an existing checkpoint. No training, no pixels modified.

Answers one question: is one source corpus (``BHSig260``, ``CEDAR``, ``SigResize``)
carrying the error rate, and does one corpus separate genuine signatures from impostors
less well than the others?

The protocol is exactly the one in :mod:`training.evaluate`: the first ``--references``
genuine images per signer are enrolled, a query's score is its mean cosine similarity to
those references, genuine queries are the signer's remaining genuine images, skilled
queries are that signer's forgeries, and random queries are other signers' genuine
queries. Splits are read with :func:`read_split` and embedded with :func:`embed_records`
at the same ``--image_height``/``--image_width``, so these numbers are comparable with the
evaluation report for the same checkpoint.

Per dataset, for validation and test:

* skilled-forgery EER and random-forgery EER;
* FRR at each ``--target-fars`` budget. The threshold is chosen on the *validation*
  scores of that dataset and applied unchanged to test, so test is never tuned on;
* mean cosine similarity for genuine, skilled and random pairs, plus the separation gap
  ``mean(genuine) - mean(random)``;
* median and p10/p90 of each score distribution, i.e. how much the genuine and impostor
  distributions overlap;
* a histogram of the three distributions, saved per dataset and split.

A random pair is attributed to the *target signer's* corpus. The query image belongs to
another signer's corpus, so labelling by the probe would mix corpora inside one cell;
labelling by the claimed identity keeps "dataset" meaning "corpus of the enrolled
signer" for all three query kinds.

Usage:
    python -m training.audit_embeddings --checkpoint checkpoints/best_model.pt \
        --manifest training_clean/manifest_aug.csv --already-preprocessed
    python -m training.audit_embeddings ... --target-fars 0.001,0.01 \
        --hist-dir audit_histograms --top-random 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: the script only writes PNG files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from scoring.reference_selection import MODES
from scoring.fusion import Enrollment, enroll_and_build_queries, score_queries  # noqa: E402
from scoring.metrics import eer_threshold, rates_at_threshold, threshold_at_far  # noqa: E402
from training.stage3_metric_learning import (  # noqa: E402
    SignatureRecord,
    SiameseSignatureModel,
    embed_records,
    read_split,
)

KINDS = ("genuine", "skilled", "random")


@dataclass
class SplitAudit:
    """Everything one split produced, kept so the tables can be built afterwards."""

    name: str
    records: list[SignatureRecord]
    enrollment: Enrollment
    queries: dict[str, list]
    scores: dict[str, np.ndarray]
    labels: dict[str, list[str]]


def _collect(
    records: list[SignatureRecord],
    embeddings: np.ndarray,
    references: int,
    max_random_queries: int | None,
    name: str,
    reference_selection: str = "first",
    already_preprocessed: bool = True,
    image_height: int = 256,
    image_width: int = 512,
) -> SplitAudit:
    """Enrol, score and label one split."""
    enrollment = enroll_and_build_queries(
        records,
        references,
        max_random_queries=max_random_queries,
        reference_selection=reference_selection,
        embeddings=embeddings,
        already_preprocessed=already_preprocessed,
        ink="auto",
        image_height=image_height,
        image_width=image_width,
    )
    scores = score_queries(embeddings, [], enrollment, None)
    # ``score_queries`` appends one score per query in enrollment order, so filtering the
    # query list by kind gives exactly the score order for that kind.
    queries = {kind: [q for q in enrollment.queries if q.kind == kind] for kind in KINDS}
    signer_dataset: dict[str, str] = {}
    for record in records:
        from training.stage3_metric_learning import source_group
        signer_dataset.setdefault(record.signer_id, source_group(record))
    labels = {
        kind: [signer_dataset[q.target_signer] for q in queries[kind]] for kind in KINDS
    }
    return SplitAudit(name, records, enrollment, queries, scores, labels)


def _datasets(audit: SplitAudit) -> list[str]:
    return sorted({label for kind in KINDS for label in audit.labels[kind]})


def _dataset_scores(audit: SplitAudit, dataset: str) -> dict[str, np.ndarray]:
    """Return the deep scores of one dataset, split by query kind."""
    out: dict[str, np.ndarray] = {}
    for kind in KINDS:
        mask = np.asarray([label == dataset for label in audit.labels[kind]], dtype=bool)
        out[kind] = audit.scores[f"deep_{kind}"][mask]
    return out


def _val_thresholds(
    audit: SplitAudit, target_fars: list[float]
) -> dict[str, dict[float, float | None]]:
    """Per-dataset FAR thresholds chosen on validation only."""
    thresholds: dict[str, dict[float, float | None]] = {}
    for dataset in _datasets(audit):
        scores = _dataset_scores(audit, dataset)
        per_far: dict[float, float | None] = {}
        for far in target_fars:
            if scores["genuine"].size and scores["skilled"].size:
                per_far[far] = threshold_at_far(scores["genuine"], scores["skilled"], far)[0]
            else:
                per_far[far] = None
        thresholds[dataset] = per_far
    return thresholds


def _rows(
    audit: SplitAudit,
    thresholds: dict[str, dict[float, float | None]],
    target_fars: list[float],
) -> list[dict[str, object]]:
    """Build one table row per dataset."""
    rows: list[dict[str, object]] = []
    for dataset in _datasets(audit):
        scores = _dataset_scores(audit, dataset)
        genuine, skilled, random_ = scores["genuine"], scores["skilled"], scores["random"]
        row: dict[str, object] = {
            "dataset": dataset,
            "n_genuine": int(genuine.size),
            "n_skilled": int(skilled.size),
            "n_random": int(random_.size),
        }
        row["eer_skilled"] = (
            float(eer_threshold(genuine, skilled)[1]) if genuine.size and skilled.size else None
        )
        row["eer_random"] = (
            float(eer_threshold(genuine, random_)[1]) if genuine.size and random_.size else None
        )
        for far in target_fars:
            threshold = thresholds.get(dataset, {}).get(far)
            if threshold is not None and genuine.size and skilled.size:
                row[f"frr@{far}"] = float(rates_at_threshold(genuine, skilled, threshold)[1])
            else:
                row[f"frr@{far}"] = None
        for kind, values in (("genuine", genuine), ("skilled", skilled), ("random", random_)):
            row[f"mean_{kind}"] = float(values.mean()) if values.size else None
            row[f"median_{kind}"] = float(np.median(values)) if values.size else None
            row[f"p10_{kind}"] = float(np.percentile(values, 10)) if values.size else None
            row[f"p90_{kind}"] = float(np.percentile(values, 90)) if values.size else None
        mean_genuine, mean_random = row["mean_genuine"], row["mean_random"]
        row["gap"] = (
            float(mean_genuine) - float(mean_random)
            if mean_genuine is not None and mean_random is not None
            else None
        )
        rows.append(row)
    return rows


def _fmt(value: object, spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return format(float(value), spec)


def _print_table(rows: list[dict[str, object]], target_fars: list[float], title: str) -> None:
    print(f"--- {title} ---")
    header = (
        f"{'dataset':<12}{'n_gen':>7}{'n_skl':>7}{'n_rnd':>8}"
        f"{'EER_skl':>9}{'EER_rnd':>9}"
        + "".join(f"{'FRR@' + format(far, 'g'):>12}" for far in target_fars)
        + f"{'mean_gen':>10}{'mean_skl':>10}{'mean_rnd':>10}{'gap':>9}"
    )
    print(header)
    for row in rows:
        line = (
            f"{row['dataset']:<12}{_fmt(row['n_genuine']):>7}{_fmt(row['n_skilled']):>7}"
            f"{_fmt(row['n_random']):>8}"
            f"{_fmt(row['eer_skilled']):>9}{_fmt(row['eer_random']):>9}"
            + "".join(f"{_fmt(row[f'frr@{far}']):>12}" for far in target_fars)
            + f"{_fmt(row['mean_genuine']):>10}{_fmt(row['mean_skilled']):>10}"
            f"{_fmt(row['mean_random']):>10}{_fmt(row['gap']):>9}"
        )
        print(line)
    print()


def _print_distributions(rows: list[dict[str, object]], title: str) -> None:
    print(f"--- {title}: score distributions (median [p10, p90]) ---")
    for row in rows:
        print(
            f"  {row['dataset']:<12}"
            f"genuine {_fmt(row['median_genuine'])} "
            f"[{_fmt(row['p10_genuine'])}, {_fmt(row['p90_genuine'])}]"
            f" | skilled {_fmt(row['median_skilled'])} "
            f"[{_fmt(row['p10_skilled'])}, {_fmt(row['p90_skilled'])}]"
            f" | random {_fmt(row['median_random'])} "
            f"[{_fmt(row['p10_random'])}, {_fmt(row['p90_random'])}]"
        )
    print()


def _print_top_random(audit: SplitAudit, top: int) -> None:
    print(f"--- {audit.name}: {top} highest-similarity random pairs (different signers) ---")
    random_queries = audit.queries["random"]
    random_scores = audit.scores["deep_random"]
    order = np.argsort(-random_scores)[:top]
    for position in order:
        query = random_queries[int(position)]
        score = float(random_scores[int(position)])
        reference_indices = audit.enrollment.references[query.target_signer]
        references = [str(audit.records[index].image_path) for index in reference_indices]
        shown = ", ".join(references[:2])
        more = f" (+{len(references) - 2} more)" if len(references) > 2 else ""
        print(
            f"  {score:.4f}  target={query.target_signer}  "
            f"query={audit.records[query.index].image_path}"
        )
        print(f"          refs: {shown}{more}")
    print()


def _save_histograms(audit: SplitAudit, hist_dir: Path, bins: int) -> list[Path]:
    """Write one genuine/skilled/random histogram per dataset."""
    written: list[Path] = []
    hist_dir.mkdir(parents=True, exist_ok=True)
    colours = {"genuine": "tab:green", "skilled": "tab:red", "random": "tab:blue"}
    for dataset in _datasets(audit):
        scores = _dataset_scores(audit, dataset)
        if not scores["genuine"].size:
            continue
        figure, axis = plt.subplots(figsize=(7, 4))
        for kind in KINDS:
            values = scores[kind]
            if values.size:
                axis.hist(
                    values,
                    bins=bins,
                    range=(-1.0, 1.0),
                    alpha=0.45,
                    density=True,
                    color=colours[kind],
                    label=f"{kind} (n={values.size})",
                )
        axis.set_xlabel("mean cosine similarity to enrolled references")
        axis.set_ylabel("density")
        axis.set_title(f"{dataset} - {audit.name}")
        axis.legend(loc="upper left", fontsize=8)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        destination = hist_dir / f"{dataset}_{audit.name}.png"
        figure.savefig(destination, dpi=120)
        plt.close(figure)
        written.append(destination)
    return written


def _verdict(rows: list[dict[str, object]], split: str) -> None:
    """State the worst dataset and how the SigResize gap compares."""
    scored = [row for row in rows if row["eer_skilled"] is not None]
    if not scored:
        print(f"{split}: no dataset had both genuine and skilled queries; no verdict")
        return
    worst = max(scored, key=lambda row: float(row["eer_skilled"]))
    print(
        f"{split}: worst skilled EER is {worst['dataset']} at "
        f"{_fmt(worst['eer_skilled'])} (n_skilled={worst['n_skilled']})"
    )
    gaps = {
        str(row["dataset"]): float(row["gap"]) for row in scored if row["gap"] is not None
    }
    if "SigResize" in gaps and len(gaps) > 1:
        others = {name: value for name, value in gaps.items() if name != "SigResize"}
        smaller = all(gaps["SigResize"] < value for value in others.values())
        comparison = ", ".join(f"{name} {value:+.4f}" for name, value in sorted(others.items()))
        print(
            f"{split}: SigResize gap {gaps['SigResize']:+.4f} vs {comparison} -> "
            f"{'SMALLER than every other dataset' if smaller else 'NOT the smallest'}"
        )
    elif "SigResize" in gaps:
        print(f"{split}: SigResize gap {gaps['SigResize']:+.4f} (no other dataset to compare)")
    else:
        print(
            f"{split}: SigResize not present in this split; gaps: "
            + ", ".join(f"{name} {value:+.4f}" for name, value in sorted(gaps.items()))
        )
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--already-preprocessed", action="store_true")
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--references", type=int, default=3)
    parser.add_argument(
        "--reference-selection",
        choices=MODES,
        default="medoid",
        help="How the genuine references are chosen (default: medoid).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--max-random-queries",
        type=int,
        default=None,
        help="Deterministic cap on random pairs per split (default: use every pair).",
    )
    parser.add_argument(
        "--target-fars",
        default="0.001,0.01",
        help="Comma-separated FAR budgets for the FRR columns.",
    )
    parser.add_argument("--top-random", type=int, default=20)
    parser.add_argument("--hist-dir", type=Path, default=Path("audit_histograms"))
    parser.add_argument("--hist-bins", type=int, default=40)
    args = parser.parse_args()
    args.target_fars = [float(value) for value in args.target_fars.split(",") if value.strip()]
    for far in args.target_fars:
        if not 0.0 < far < 1.0:
            parser.error(f"--target-fars values must be between 0 and 1, got {far}")
    if args.references < 1:
        parser.error("--references must be at least 1")
    return args


def main() -> None:
    args = _parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    for path in (checkpoint_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"not found: {path}")
    root = (args.image_root or manifest_path.parent).resolve()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SiameseSignatureModel(
        pretrained=False, embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    preprocessing = checkpoint.get("preprocessing", {})
    fitted_size = (preprocessing.get("image_height"), preprocessing.get("image_width"))
    if fitted_size != (None, None) and fitted_size != (args.image_height, args.image_width):
        print(
            f"warning: checkpoint was trained at HxW {fitted_size}, "
            f"auditing at ({args.image_height}, {args.image_width})"
        )
    print(f"checkpoint:   {checkpoint_path}")
    print(
        "checkpoint metadata: "
        f"selection_metric={checkpoint.get('selection_metric')} "
        f"selection_value={checkpoint.get('selection_value')} "
        f"best_validation_eer_skilled={checkpoint.get('best_validation_eer_skilled')} "
        f"augmentation={checkpoint.get('augmentation')} "
        f"epoch={checkpoint.get('epoch')} phase={checkpoint.get('phase')}"
    )
    print(f"manifest:     {manifest_path}")
    print(f"image root:   {root}")
    print(
        f"device:       {args.device}  batch_size={args.batch_size}  "
        f"workers={args.workers}  references={args.references}\n"
    )

    audits: dict[str, SplitAudit] = {}
    for name, splits in (("val", {"val", "validation"}), ("test", {"test"})):
        records = read_split(manifest_path, root, splits)
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
        audits[name] = _collect(
            records,
            embeddings,
            args.references,
            args.max_random_queries,
            name,
            reference_selection=args.reference_selection,
            already_preprocessed=args.already_preprocessed,
            image_height=args.image_height,
            image_width=args.image_width,
        )
        print(
            f"{name}: {len(records):,} images | "
            f"{len(audits[name].queries['genuine']):,} genuine, "
            f"{len(audits[name].queries['skilled']):,} skilled, "
            f"{len(audits[name].queries['random']):,} random queries"
        )
    print()

    thresholds = _val_thresholds(audits["val"], args.target_fars)
    for split in ("val", "test"):
        rows = _rows(audits[split], thresholds, args.target_fars)
        _print_table(rows, args.target_fars, f"{split} per dataset")
        _print_distributions(rows, split)
        _print_top_random(audits[split], args.top_random)
        _verdict(rows, split)

    print("thresholds chosen on validation (per dataset, per FAR):")
    for dataset, per_far in sorted(thresholds.items()):
        print(
            f"  {dataset:<12}"
            + "  ".join(f"FAR {far:g} -> {_fmt(per_far[far])}" for far in args.target_fars)
        )
    print()

    written: list[Path] = []
    for split in ("val", "test"):
        written.extend(_save_histograms(audits[split], args.hist_dir, args.hist_bins))
    print(f"histograms written to {args.hist_dir.resolve()}:")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()

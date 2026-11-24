"""Train the Stage 3 ConvNeXt-Tiny signature embedding model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from scoring.fusion import Enrollment, enroll_and_build_queries, score_queries
from scoring.metrics import eer_threshold
from training.augment import SignatureAugment
from training.stage3_metric_learning import (
    PKBatchSampler,
    SignatureDataset,
    SignatureRecord,
    SiameseSignatureModel,
    embed_records,
    load_manifest,
    triplet_loss,
)


def _device_from_argument(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_epoch(
    model: SiameseSignatureModel,
    loader: DataLoader,
    device: torch.device,
    *,
    mining: Literal["semi-hard", "batch-hard"],
    margin: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "mean_positive": 0.0, "mean_negative": 0.0, "active_fraction": 0.0}
    batch_count = 0

    for images, signer_ids, is_forgery in loader:
        images = images.to(device)
        signer_ids = signer_ids.to(device)
        is_forgery = is_forgery.to(device=device, dtype=torch.bool)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            embeddings = model(images)
            loss, metrics = triplet_loss(
                embeddings,
                signer_ids,
                is_forgery,
                margin=margin,
                mining=mining,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=5.0,
                )
                optimizer.step()

        totals["loss"] += loss.detach().item()
        for name, value in metrics.items():
            totals[name] += value.item()
        batch_count += 1

    if batch_count == 0:
        raise RuntimeError("data loader produced no batches")
    return {name: value / batch_count for name, value in totals.items()}


def _set_seed(seed: int, device: torch.device) -> None:
    # This must be set before CUDA work starts. It makes cuBLAS reductions
    # deterministic on supported CUDA versions.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    elif device.type == "mps":
        warnings.warn(
            "MPS does not guarantee deterministic training; use --device cpu "
            "or a deterministic CUDA setup for repeatable results.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        torch.use_deterministic_algorithms(True)


def _improves(candidate: float, best: float) -> bool:
    """Return True when ``candidate`` beats ``best``.

    Both supported selection metrics are error rates, so lower is better. A tie keeps the
    earlier checkpoint, which is the one with fewer epochs behind it, and NaN never wins,
    so a degenerate metric cannot silently replace the best model.
    """
    return candidate < best


def _validation_skilled_eer(
    model: SiameseSignatureModel,
    records: Sequence[SignatureRecord],
    enrollment: Enrollment,
    device: torch.device,
    *,
    already_preprocessed: bool,
    image_height: int,
    image_width: int,
    batch_size: int,
) -> float:
    """Return the skilled-forgery EER on the validation signers.

    Uses the same protocol as ``training.evaluate``: the first references of each signer
    are enrolled and a query scores as its mean cosine similarity to them. Selecting on
    that number means the checkpoint is chosen by the metric it is reported on. Random
    pairs play no part in the choice, so the enrollment carries none and the per-epoch
    cost is limited to the genuine and skilled queries.

    The call leaves the model in eval mode; ``_run_epoch`` restores train mode on the
    next training epoch.
    """
    embeddings = embed_records(
        model,
        records,
        already_preprocessed=already_preprocessed,
        device=device,
        batch_size=batch_size,
        image_height=image_height,
        image_width=image_width,
    )
    scores = score_queries(embeddings, [], enrollment)
    _, skilled_eer = eer_threshold(scores["deep_genuine"], scores["deep_skilled"])
    return float(skilled_eer)


def _require_existing_images(*groups: Sequence[SignatureRecord]) -> None:
    """Fail at startup with the offending paths instead of mid-epoch inside a worker.

    ``load_manifest`` resolves relative paths from ``--image-root`` or the manifest's
    own folder, so a wrong root shows up here as a list of resolved paths to compare.
    """
    missing = [
        record.image_path
        for group in groups
        for record in group
        if not record.image_path.is_file()
    ]
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} manifest image(s) do not exist, e.g.:\n{preview}\n"
            "Relative paths are resolved from --image-root, or from the manifest's "
            "folder when --image-root is omitted."
        )


def _filter_train_datasets(
    records: Sequence[SignatureRecord], requested: Sequence[str] | None
) -> list[SignatureRecord]:
    """Keep only training rows from the requested source corpora.

    Used for the source-corpus ablation: ``--train-datasets BHSig260,CEDAR`` trains without
    SigResize. Validation and test are never passed through here, because an ablation that
    also changed the evaluation set would not be comparable to the baseline run.

    Filtering is whole-signer: each signer belongs to exactly one corpus, so a signer's
    images are never split across the filtered and unfiltered sets. A signer spanning two
    corpora would break that assumption, so it is reported rather than silently kept.

    Args:
        records: Training rows only.
        requested: Corpus names to keep, or ``None``/empty for "keep everything".

    Raises:
        ValueError: An unknown corpus name or a signer spanning two corpora. An empty
            ``records`` list also fails, because no corpus name can be available then.
    """
    wanted = [name.strip() for name in (requested or []) if name.strip()]
    if not wanted:
        return list(records)

    available = sorted({record.dataset for record in records if record.dataset})
    unknown = [name for name in wanted if name not in available]
    if unknown:
        raise ValueError(
            f"--train-datasets: unknown dataset(s) {unknown}; available: {available}"
        )

    keep = set(wanted)
    kept = [record for record in records if record.dataset in keep]

    signer_datasets: dict[str, set[str]] = {}
    for record in kept:
        signer_datasets.setdefault(record.signer_id, set()).add(record.dataset)
    mixed = {signer: sorted(names) for signer, names in signer_datasets.items() if len(names) > 1}
    if mixed:
        raise ValueError(f"--train-datasets: signers span multiple datasets: {mixed}")
    return kept


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument(
        "--train-datasets",
        default=None,
        help=(
            "Comma-separated source corpora to train on, e.g. BHSig260,CEDAR. "
            "Filters TRAIN rows only; validation and test always keep every corpus."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/stage3"))
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--ink", choices=("auto", "light", "dark"), default="auto")
    parser.add_argument(
        "--already-preprocessed",
        action="store_true",
        help="Input files are already fixed-size Stage 2 grayscale images",
    )
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    parser.add_argument("--signers-per-batch", type=int, default=12)
    parser.add_argument("--genuine-per-signer", type=int, default=4)
    parser.add_argument("--forgeries-per-signer", type=int, default=2)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group-1 augmentation on the train split only (val/test are never augmented)",
    )
    parser.add_argument(
        "--augment-group2",
        action="store_true",
        help="Also enable the group-2 transforms (junk, brightness/contrast)",
    )
    parser.add_argument(
        "--select-by",
        choices=("eer", "loss"),
        default="eer",
        help="Validation metric that picks the best checkpoint (default: skilled EER)",
    )
    parser.add_argument(
        "--references",
        type=int,
        default=5,
        help="Genuine references per signer used by the validation EER",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Batch size for the per-epoch validation embedding pass",
    )
    parser.add_argument(
        "--cosine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cosine LR decay across the fine-tune phase (warmup stays flat)",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.train_datasets = (
        [name.strip() for name in args.train_datasets.split(",") if name.strip()]
        if args.train_datasets
        else None
    )
    return args


def main() -> None:
    args = _parse_args()
    if args.phase1_epochs < 0 or args.phase2_epochs < 0:
        raise ValueError("epoch counts cannot be negative")
    if args.phase1_epochs + args.phase2_epochs == 0:
        raise ValueError("at least one training phase must have a positive epoch count")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.margin <= 0:
        raise ValueError("margin must be positive")
    if args.references < 1:
        raise ValueError("references must be at least 1")
    if args.eval_batch_size < 1:
        raise ValueError("eval-batch-size must be positive")

    device = _device_from_argument(args.device)
    _set_seed(args.seed, device)
    train_records, validation_records = load_manifest(args.manifest, args.image_root)
    # Validation stays untouched: the ablation changes what the model trains on, never
    # what it is measured on.
    train_records = _filter_train_datasets(train_records, args.train_datasets)
    _require_existing_images(train_records, validation_records)
    if args.train_datasets:
        print(
            f"train datasets: {sorted(set(args.train_datasets))} -> "
            f"{len(train_records):,} train rows, "
            f"{len({record.signer_id for record in train_records})} signers "
            f"(validation unchanged: {len(validation_records):,} rows)"
        )
    all_signers = sorted(
        {record.signer_id for record in train_records + validation_records}
    )
    signer_to_index = {signer: index for index, signer in enumerate(all_signers)}

    # The augmenter goes to the train dataset only. Validation images are the real
    # pixels, otherwise the validation EER would be measured on distorted signatures.
    train_augment = (
        SignatureAugment(group2=args.augment_group2, seed=args.seed)
        if args.augment
        else None
    )
    train_dataset = SignatureDataset(
        train_records,
        signer_to_index,
        already_preprocessed=args.already_preprocessed,
        ink=args.ink,
        image_height=args.image_height,
        image_width=args.image_width,
        augment=train_augment,
    )
    validation_dataset = SignatureDataset(
        validation_records,
        signer_to_index,
        already_preprocessed=args.already_preprocessed,
        ink=args.ink,
        image_height=args.image_height,
        image_width=args.image_width,
        augment=None,
    )

    train_sampler = PKBatchSampler(
        train_records,
        signers_per_batch=args.signers_per_batch,
        genuine_per_signer=args.genuine_per_signer,
        forgeries_per_signer=args.forgeries_per_signer,
        seed=args.seed,
    )
    validation_eligible_signers = {
        record.signer_id
        for record in validation_records
        if not record.is_forgery
    }
    genuine_counts = {
        signer: sum(
            record.signer_id == signer and not record.is_forgery
            for record in validation_records
        )
        for signer in validation_eligible_signers
    }
    eligible_validation_signers = sum(count >= 2 for count in genuine_counts.values())
    if eligible_validation_signers < 2:
        raise ValueError(
            "validation needs at least two signers with two genuine images each"
        )
    validation_batch_signers = min(args.signers_per_batch, eligible_validation_signers)
    validation_sampler = PKBatchSampler(
        validation_records,
        signers_per_batch=validation_batch_signers,
        genuine_per_signer=2,
        forgeries_per_signer=args.forgeries_per_signer,
        seed=args.seed,
    )

    # persistent_workers stays off so every epoch re-forks the workers and picks up the
    # epoch passed to the dataset; with it on, a worker would keep its first epoch.
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    # Built once: the enrollment depends only on the manifest, so rebuilding it every
    # epoch would cost time without changing a single query. No random pairs are built
    # because checkpoint selection never looks at them.
    validation_enrollment = (
        enroll_and_build_queries(
            validation_records, args.references, max_random_queries=0
        )
        if args.select_by == "eer"
        else None
    )

    print(f"Using device: {device}")
    model = SiameseSignatureModel(pretrained=not args.no_pretrained).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_selection = math.inf
    best_validation_loss = math.inf
    best_validation_eer: float | None = None

    phases: list[tuple[str, int, float, Literal["semi-hard", "batch-hard"]]] = [
        ("warmup", args.phase1_epochs, 1e-3, "semi-hard"),
        ("fine_tune", args.phase2_epochs, 1e-4, "batch-hard"),
    ]
    global_epoch = 0
    for phase_name, epoch_count, learning_rate, mining in phases:
        if epoch_count == 0:
            continue
        if phase_name == "warmup":
            model.freeze_backbone()
        else:
            model.unfreeze_last_stages(count=2)

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        # Cosine decay on the fine-tune phase only; the warmup keeps a flat LR because
        # the backbone is frozen and the projection head is still far from converged.
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epoch_count, eta_min=1e-6
            )
            if phase_name == "fine_tune" and args.cosine
            else None
        )
        for phase_epoch in range(epoch_count):
            train_sampler.set_epoch(global_epoch)
            train_dataset.set_epoch(global_epoch)
            train_metrics = _run_epoch(
                model,
                train_loader,
                device,
                mining=mining,
                margin=args.margin,
                optimizer=optimizer,
            )
            # Validate with batch-hard in every phase: semi-hard and batch-hard
            # losses are not comparable, which would skew best-checkpoint selection.
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device,
                mining="batch-hard",
                margin=args.margin,
            )
            # Skilled-forgery EER is the metric that matters, so it can pick the
            # checkpoint instead of the triplet loss, whose scale also changes between
            # the semi-hard and batch-hard phases.
            validation_eer: float | None = None
            selection_value = validation_metrics["loss"]
            if validation_enrollment is not None:
                validation_eer = _validation_skilled_eer(
                    model,
                    validation_records,
                    validation_enrollment,
                    device,
                    already_preprocessed=args.already_preprocessed,
                    image_height=args.image_height,
                    image_width=args.image_width,
                    batch_size=args.eval_batch_size,
                )
                selection_value = validation_eer

            global_epoch += 1
            print(
                json.dumps(
                    {
                        "phase": phase_name,
                        "epoch": phase_epoch + 1,
                        "lr": optimizer.param_groups[0]["lr"],
                        "train": train_metrics,
                        "validation": validation_metrics,
                        "validation_eer_skilled": validation_eer,
                        "selected_by": args.select_by,
                        "selection_value": selection_value,
                    }
                )
            )

            # Lower is better for both metrics, so one comparison covers either choice.
            if _improves(selection_value, best_selection):
                best_selection = selection_value
                best_validation_loss = validation_metrics["loss"]
                best_validation_eer = validation_eer
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "embedding_dim": model.embedding_dim,
                        "selection_metric": args.select_by,
                        "selection_value": best_selection,
                        "best_validation_loss": best_validation_loss,
                        "best_validation_eer_skilled": best_validation_eer,
                        "epoch": global_epoch,
                        "phase": phase_name,
                        "seed": args.seed,
                        "preprocessing": {
                            "already_preprocessed": args.already_preprocessed,
                            "ink": args.ink,
                            # (width, height), the same order preprocessing uses.
                            "target_size": [args.image_width, args.image_height],
                            "image_height": args.image_height,
                            "image_width": args.image_width,
                        },
                        "augmentation": {
                            "enabled": args.augment,
                            "group2": args.augment_group2,
                        },
                        # Which corpora the train split was limited to, so each ablation
                        # checkpoint says what it saw without reading the command line.
                        "train_datasets": (
                            sorted(set(args.train_datasets)) if args.train_datasets else "all"
                        ),
                    },
                    checkpoint_path,
                )
            if scheduler is not None:
                scheduler.step()

    best_eer_text = (
        f", val skilled EER {best_validation_eer:.4f}"
        if best_validation_eer is not None
        else ""
    )
    print(
        f"Best checkpoint: {checkpoint_path} "
        f"({args.select_by} {best_selection:.4f}{best_eer_text}, "
        f"val loss {best_validation_loss:.4f})"
    )


if __name__ == "__main__":
    main()

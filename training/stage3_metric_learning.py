"""Stage 3 ConvNeXt-Tiny embeddings and signer-balanced triplet learning."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Mapping, Sequence

import numpy as np
import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from preprocessing.signature_preprocessing import (
    load_preprocessed_image,
    to_model_input,
)
from training.augment import SignatureAugment

MiningMode = Literal["semi-hard", "batch-hard"]


@dataclass(frozen=True)
class SignatureRecord:
    """One manifest example and the signer whose identity it represents.

    ``dataset`` is the source corpus (``BHSig260``, ``CEDAR``, ``SigResize``). It is
    carried through to evaluation so FAR/FRR can be reported per source; it is never an
    input to the model. Manifests without the column yield an empty string.
    """

    image_path: Path
    signer_id: str
    is_forgery: bool
    dataset: str = ""


def source_group(record: SignatureRecord | object) -> str:
    """Return the reporting group: split BHSig260 into Hindi and Bengali."""
    dataset = str(getattr(record, "dataset", "") or "unknown")
    signer = str(getattr(record, "signer_id", ""))
    if dataset == "BHSig260":
        if "_hi_" in signer:
            return "BHSig260-Hindi"
        if "_bn_" in signer:
            return "BHSig260-Bengali"
    return dataset


def _parse_bool(value: str, *, row_number: int) -> bool:
    """Parse common CSV boolean spellings and report invalid values with row context."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(
        f"manifest row {row_number}: is_forgery must be true/false or 1/0"
    )


def load_manifest(
    manifest_path: str | Path,
    image_root: str | Path | None = None,
) -> tuple[list[SignatureRecord], list[SignatureRecord]]:
    """Load train/validation rows from a signer-disjoint CSV manifest.

    Required columns: ``image_path,signer_id,is_forgery,split``. Relative image
    paths are resolved from ``image_root`` or the manifest's directory. Forgeries
    must use the ID of the signer they imitate.
    """
    manifest = Path(manifest_path).expanduser().resolve()
    # Relative paths are rooted at image_root when provided, otherwise beside the CSV.
    root = Path(image_root).expanduser().resolve() if image_root else manifest.parent
    required = {"image_path", "signer_id", "is_forgery", "split"}
    train: list[SignatureRecord] = []
    validation: list[SignatureRecord] = []
    test_signers: set[str] = set()

    with manifest.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing required columns: {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            image_path = row["image_path"].strip()
            signer_id = row["signer_id"].strip()
            split = row["split"].strip().lower()
            if not image_path or not signer_id:
                raise ValueError(
                    f"manifest row {row_number}: image_path and signer_id cannot be empty"
                )
            if split not in {"train", "val", "validation", "test"}:
                raise ValueError(
                    f"manifest row {row_number}: split must be train, val, or test"
                )

            path = Path(image_path).expanduser()
            if not path.is_absolute():
                path = root / path
            record = SignatureRecord(
                image_path=path.resolve(),
                signer_id=signer_id,
                is_forgery=_parse_bool(row["is_forgery"], row_number=row_number),
                # Optional column: evaluation groups its report by it, training ignores it.
                dataset=(row.get("dataset") or "").strip(),
            )
            if split == "train":
                train.append(record)
            elif split in {"val", "validation"}:
                validation.append(record)
            else:
                test_signers.add(signer_id)

    if not train or not validation:
        raise ValueError("manifest must contain at least one train row and one val row")

    train_signers = {record.signer_id for record in train}
    validation_signers = {record.signer_id for record in validation}
    # Keep every signer confined to one split so validation does not leak identities.
    overlap = (train_signers & validation_signers) | (train_signers & test_signers) | (
        validation_signers & test_signers
    )
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"signers must be disjoint across train, val, and test: {examples}")
    return train, validation


def validate_checkpoint_image_size(
    checkpoint: Mapping[str, object],
    *,
    image_height: int,
    image_width: int,
    checkpoint_path: str | Path | None = None,
) -> None:
    """Reject evaluation when a checkpoint's trained image size differs from the request."""
    label = str(checkpoint_path) if checkpoint_path is not None else "checkpoint"
    preprocessing = checkpoint.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ValueError(f"{label} is missing preprocessing image-size metadata")

    saved_height = preprocessing.get("image_height")
    saved_width = preprocessing.get("image_width")
    if (
        isinstance(saved_height, int)
        and not isinstance(saved_height, bool)
        and isinstance(saved_width, int)
        and not isinstance(saved_width, bool)
    ):
        saved_size = (saved_height, saved_width)
    else:
        # Older checkpoints stored the preprocessing tuple as (width, height).
        target_size = preprocessing.get("target_size")
        if (
            isinstance(target_size, (list, tuple))
            and len(target_size) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in target_size
            )
        ):
            saved_size = (target_size[1], target_size[0])
        else:
            raise ValueError(f"{label} is missing valid image_height/image_width metadata")

    requested_size = (image_height, image_width)
    if saved_size != requested_size:
        raise ValueError(
            f"{label} was trained at HxW {saved_size[0]}x{saved_size[1]}, "
            f"but evaluation requested HxW {image_height}x{image_width}"
        )


def read_split(
    manifest: str | Path,
    root: str | Path,
    splits: set[str],
) -> list[SignatureRecord]:
    """Read manifest rows whose ``split`` column is in ``splits``.

    :func:`load_manifest` returns only the train and validation rows it needs for
    training. Evaluation and fusion fitting also need the test rows, so this helper
    reads an arbitrary split selection. It is the single split reader used by
    ``training.evaluate`` and ``training.fit_fusion``, which keeps both scripts
    scoring exactly the same records.

    Args:
        manifest: CSV manifest with ``image_path,signer_id,is_forgery,split``. The
            optional ``dataset`` column is carried into ``SignatureRecord.dataset`` and
            drives the per-dataset report in :mod:`training.evaluate`.
        root: Directory used to resolve relative image paths.
        splits: Lower-case split names to keep, e.g. ``{"val", "validation"}``.

    Raises:
        ValueError: If no row matches ``splits``.
    """
    manifest_path = Path(manifest).expanduser().resolve()
    image_root = Path(root).expanduser().resolve()
    wanted = {name.strip().lower() for name in splits}
    records: list[SignatureRecord] = []
    with manifest_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"].strip().lower() not in wanted:
                continue
            path = Path(row["image_path"].strip()).expanduser()
            records.append(
                SignatureRecord(
                    image_path=(path if path.is_absolute() else image_root / path).resolve(),
                    signer_id=row["signer_id"].strip(),
                    is_forgery=row["is_forgery"].strip().lower() in {"1", "true", "yes"},
                    dataset=(row.get("dataset") or "").strip(),
                )
            )
    if not records:
        raise ValueError(f"no manifest rows for splits {sorted(wanted)}")
    return records


class SignatureDataset(Dataset[tuple[Tensor, int, bool]]):
    """Load signatures, preprocess them, and return model-ready tensors and labels."""

    def __init__(
        self,
        records: Sequence[SignatureRecord],
        signer_to_index: dict[str, int],
        *,
        already_preprocessed: bool = False,
        ink: Literal["auto", "light", "dark"] = "auto",
        image_height: int = 256,
        image_width: int = 512,
        augment: SignatureAugment | None = None,
    ) -> None:
        self.records = list(records)
        self.signer_to_index = signer_to_index
        self.already_preprocessed = already_preprocessed
        self.ink = ink
        self.image_height = image_height
        self.image_width = image_width
        # Only the train dataset receives an augmenter; validation and test pass None so
        # their pixels stay untouched and the validation EER measures the real images.
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Forward the epoch so augmentation is reproducible per (seed, epoch, index)."""
        if self.augment is not None:
            self.augment.set_epoch(epoch)

    def __getitem__(self, index: int) -> tuple[Tensor, int, bool]:
        record = self.records[index]
        # One shared loading path: the handcrafted feature extractor reads the same
        # pixels through ``load_preprocessed_image``, so the two modalities cannot drift.
        preprocessed = load_preprocessed_image(
            record.image_path,
            already_preprocessed=self.already_preprocessed,
            ink=self.ink,
            image_height=self.image_height,
            image_width=self.image_width,
        )
        # Augment the uint8 image before normalization: every transform works on 0-255
        # pixels, and the augmenter never sees the label, so genuine and forged samples
        # get identical treatment.
        if self.augment is not None:
            preprocessed = self.augment(preprocessed, index=index)
        # The model accepts three channels; preprocessing repeats grayscale for RGB.
        tensor = torch.from_numpy(to_model_input(preprocessed))
        return tensor, self.signer_to_index[record.signer_id], record.is_forgery


class PKBatchSampler(Sampler[list[int]]):
    """Build batches with P signers, K genuine samples, and up to F forgeries each.

    P and K provide genuine positive pairs across distinct identities. Forgeries
    are sampled only as optional negatives associated with their imitated signer.
    """

    def __init__(
        self,
        records: Sequence[SignatureRecord],
        *,
        signers_per_batch: int,
        genuine_per_signer: int,
        forgeries_per_signer: int = 0,
        balance_datasets: bool = False,
        seed: int = 42,
    ) -> None:
        if signers_per_batch < 2:
            raise ValueError("signers_per_batch must be at least 2")
        if genuine_per_signer < 2:
            raise ValueError("genuine_per_signer must be at least 2")
        if forgeries_per_signer < 0:
            raise ValueError("forgeries_per_signer cannot be negative")

        self.signers_per_batch = signers_per_batch
        self.genuine_per_signer = genuine_per_signer
        self.forgeries_per_signer = forgeries_per_signer
        self.balance_datasets = balance_datasets
        self.seed = seed
        self.epoch = 0
        # Keep genuine and forged record indices separate for each signer.
        self.by_signer: dict[str, dict[bool, list[int]]] = {}
        self.signer_dataset: dict[str, str] = {}
        for index, record in enumerate(records):
            groups = self.by_signer.setdefault(record.signer_id, {False: [], True: []})
            groups[record.is_forgery].append(index)
            self.signer_dataset.setdefault(record.signer_id, record.dataset or "unknown")

        self.eligible_signers = [
            signer
            for signer, groups in self.by_signer.items()
            if len(groups[False]) >= genuine_per_signer
        ]
        # Each batch needs enough genuine samples per signer to form positives.
        if len(self.eligible_signers) < signers_per_batch:
            raise ValueError(
                f"need at least {signers_per_batch} signers with "
                f"{genuine_per_signer} genuine images; found {len(self.eligible_signers)}"
            )

        # Optional equal-share sampling: partition the eligible signers by corpus once,
        # so every batch can draw a quota from each dataset without touching the index.
        by_dataset: dict[str, list[str]] = {}
        for signer in self.eligible_signers:
            by_dataset.setdefault(self.signer_dataset[signer], []).append(signer)
        self.signers_by_dataset = {
            name: sorted(signers) for name, signers in sorted(by_dataset.items())
        }

    def __len__(self) -> int:
        # Number of batches per epoch; signers are sampled anew for each batch.
        return max(1, (len(self.eligible_signers) + self.signers_per_batch - 1) // self.signers_per_batch)

    def set_epoch(self, epoch: int) -> None:
        """Change the deterministic sampling sequence for the next epoch."""
        self.epoch = epoch

    def _balanced_signer_batch(self, rng: random.Random) -> list[str]:
        """Pick signers so every dataset fills an equal share of the batch.

        Quotas are filled round-robin over the datasets, skipping one that has no
        eligible signer left. A small corpus therefore contributes all its signers and
        the remaining slots go to the larger corpora; a corpus is never over-sampled
        with replacement inside one batch.
        """
        datasets = list(self.signers_by_dataset)
        rng.shuffle(datasets)
        quotas = {name: 0 for name in datasets}
        remaining = self.signers_per_batch
        while remaining:
            progressed = False
            for name in datasets:
                if remaining and quotas[name] < len(self.signers_by_dataset[name]):
                    quotas[name] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                raise RuntimeError(
                    "balanced sampling could not fill a batch; the eligible signer "
                    "counts changed after construction"
                )
        batch: list[str] = []
        for name in datasets:
            if quotas[name]:
                batch.extend(rng.sample(self.signers_by_dataset[name], quotas[name]))
        return batch

    def __iter__(self) -> Iterator[list[int]]:
        # A local seeded RNG makes each epoch reproducible without changing global RNG.
        rng = random.Random(self.seed + self.epoch)
        for _ in range(len(self)):
            if self.balance_datasets:
                signer_batch = self._balanced_signer_batch(rng)
            else:
                signer_batch = rng.sample(self.eligible_signers, self.signers_per_batch)
            indices: list[int] = []
            for signer in signer_batch:
                groups = self.by_signer[signer]
                indices.extend(rng.sample(groups[False], self.genuine_per_signer))
                # Not every signer has forgeries; use all available up to the cap.
                forgery_count = min(self.forgeries_per_signer, len(groups[True]))
                if forgery_count:
                    indices.extend(rng.sample(groups[True], forgery_count))
            yield indices


class SiameseSignatureModel(nn.Module):
    """Map each signature to a normalized embedding using a shared encoder.

    ConvNeXt extracts visual features. The projection head maps them to the
    embedding space, and L2 normalization makes cosine and Euclidean distances
    useful for comparing signatures.
    """

    def __init__(self, *, pretrained: bool = True, embedding_dim: int = 512) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.projection = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            # LayerNorm acts per sample, independent of PK batch composition.
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, images: Tensor) -> Tensor:
        """Encode a batch and return one unit-length embedding per image."""
        return F.normalize(self.projection(self.backbone(images)), p=2, dim=1)

    def compare(self, reference: Tensor, query: Tensor) -> Tensor:
        """Return pairwise cosine similarities; these are not calibrated probabilities."""
        reference_embedding = self(reference)
        query_embedding = self(query)
        return F.cosine_similarity(reference_embedding, query_embedding, dim=1)

    def freeze_backbone(self) -> None:
        """Stop gradient updates to the pretrained feature extractor."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_last_stages(self, count: int = 2) -> None:
        """Freeze the backbone, then enable gradients only in its last stages."""
        stages = getattr(self.backbone, "stages", None)
        if stages is None or count < 1:
            raise ValueError("backbone must expose stages and count must be positive")
        self.freeze_backbone()
        for stage in stages[-count:]:
            for parameter in stage.parameters():
                parameter.requires_grad = True


def triplet_loss(
    embeddings: Tensor,
    signer_ids: Tensor,
    is_forgery: Tensor,
    *,
    margin: float = 0.3,
    mining: MiningMode = "batch-hard",
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute triplet loss for genuine anchors and return detached diagnostics.

    Positives are other genuine samples from the same signer. Negatives are
    genuine samples from different signers or forgeries labeled as imitating
    the anchor's signer. Forged samples never act as anchors or positives.
    """
    if embeddings.ndim != 2 or signer_ids.ndim != 1 or is_forgery.ndim != 1:
        raise ValueError("expected 2D embeddings and 1D signer/forgery labels")
    if embeddings.shape[0] != signer_ids.shape[0] or embeddings.shape[0] != is_forgery.shape[0]:
        raise ValueError("embeddings and labels must have the same batch size")
    if mining not in {"semi-hard", "batch-hard"}:
        raise ValueError(f"unsupported mining mode: {mining}")

    # Pairwise distances are used to select examples for each genuine anchor.
    distances = torch.cdist(embeddings.float(), embeddings.float(), p=2)
    same_signer = signer_ids[:, None].eq(signer_ids[None, :])
    genuine = ~is_forgery.bool()
    not_self = ~torch.eye(len(signer_ids), dtype=torch.bool, device=signer_ids.device)
    # Positives exclude the anchor itself and must be genuine.
    positives = same_signer & genuine[None, :] & not_self
    # A same-signer forgery is a valid negative because it imitates the anchor.
    negatives = ((~same_signer) & genuine[None, :]) | (
        same_signer & is_forgery.bool()[None, :]
    )
    anchors = torch.where(genuine)[0]

    positive_mask = positives[anchors]
    negative_mask = negatives[anchors]
    if torch.any(positive_mask.sum(dim=1) == 0):
        raise ValueError("each genuine anchor needs another genuine sample from its signer")
    if torch.any(negative_mask.sum(dim=1) == 0):
        raise ValueError("each genuine anchor needs at least one valid negative")

    anchor_distances = distances[anchors]
    # Batch-hard loss takes the farthest valid positive and closest valid negative.
    positive_distances = anchor_distances.masked_fill(~positive_mask, float("-inf"))
    hardest_positive = positive_distances.max(dim=1).values
    negative_distances = anchor_distances.masked_fill(~negative_mask, float("inf"))
    hardest_negative = negative_distances.min(dim=1).values

    if mining == "semi-hard":
        # Choose the nearest negative farther away than the hardest positive but
        # still within the margin. Fall back to the hardest negative if none qualify.
        semi_hard_mask = (
            negative_mask
            & (anchor_distances > hardest_positive[:, None])
            & (anchor_distances < (hardest_positive + margin)[:, None])
        )
        semi_hard_distances = anchor_distances.masked_fill(
            ~semi_hard_mask, float("inf")
        )
        selected_negative = torch.where(
            semi_hard_mask.any(dim=1),
            semi_hard_distances.min(dim=1).values,
            hardest_negative,
        )
    else:
        selected_negative = hardest_negative

    per_anchor_loss = F.relu(hardest_positive - selected_negative + margin)
    # Metrics are detached so logging them does not retain the autograd graph.
    return per_anchor_loss.mean(), {
        "mean_positive": hardest_positive.mean().detach(),
        "mean_negative": selected_negative.mean().detach(),
        "active_fraction": (per_anchor_loss > 0).float().mean().detach(),
    }


@torch.no_grad()
def embed_records(
    model: nn.Module,
    records: Sequence[SignatureRecord],
    *,
    already_preprocessed: bool = False,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
    workers: int = 0,
    image_height: int = 256,
    image_width: int = 512,
) -> np.ndarray:
    """Embed ``records`` in order and return an ``(N, embedding_dim)`` float array.

    The output row order matches ``records``, so callers can index embeddings by
    manifest position. This is the embedding routine shared by ``training.evaluate``
    and ``training.fit_fusion``; keeping one implementation guarantees the deep score
    in a fusion fit is computed from the same forward pass as the reported baseline.

    Args:
        model: A trained Siamese embedding model in evaluation mode.
        records: Manifest records to embed.
        already_preprocessed: Forwarded to :class:`SignatureDataset`.
        device: Device used for the forward pass.
        batch_size: DataLoader batch size.
        workers: DataLoader worker count.
    """
    dataset = SignatureDataset(
        records,
        {record.signer_id: 0 for record in records},
        already_preprocessed=already_preprocessed,
        image_height=image_height,
        image_width=image_width,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    model.eval()
    chunks = [model(images.to(device)).cpu().numpy() for images, _, _ in loader]
    return np.concatenate(chunks)

"""Read-only audit of the SigResize corpus against BHSig260 and CEDAR.

Answers four questions, in this order:

1. **Stroke width** - is SigResize drawn with a different pen than the scanned corpora?
   Width is measured per image from the distance-transform ridge (2 x radius), so it does
   not depend on total ink or image size.
2. **Signer separability** - can signers be told apart from the pixels alone? A
   leave-one-out nearest-centroid classifier over 32x64 ink thumbnails is a model-free
   proxy: if signers are not separable at pixel level, no embedding can recover them.
   Skilled-forgery similarity to the genuine centroid is reported next to it.
3. **Cross-signer duplicates** - the same image under two signer ids would leak identity
   across the signer-disjoint splits. Exact duplicates are found by SHA-256; near
   duplicates by a 256-bit ink hash (banded candidate lookup), then a cosine check on the
   thumbnails and an ink-mask IoU check.
4. **Ink lost per augmentation op** - each transform is applied in isolation to a seeded
   sample of train images, so one op's retention is not confounded by the others. A
   retention above 1.0 means the op spread ink across the 200 cut (blur, noise); it is
   printed as a negative loss and explained.

Nothing is written and the manifest is never modified. Recommendations are printed only.

Usage:
    python -m training.audit_sigresize --manifest training_clean/manifest_aug.csv
    python -m training.audit_sigresize --manifest training_clean/manifest_aug.csv \\
        --image-root training_clean --near-cosine 0.97 --augment-sample 100

Exit code is 0 when the manifest was readable and every section ran; the audit is
diagnostic, not a pass/fail gate (training.verify_data owns that verdict).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from training.augment import SignatureAugment

DATASETS = ("BHSig260", "CEDAR", "SigResize")
INK_LEVEL = 200
THUMB_SIZE = (64, 128)   # (width, height) of the identity thumbnail
MASK_SIZE = (64, 128)    # binary ink mask used to verify near-duplicates
MASK_COVERAGE = 0.02     # a 64x128 cell is ink at >=2% full-resolution coverage
HASH_SIZE = 32           # 32x32 binary ink hash = 1,024 bits, sixteen 64-bit bands
REQUIRED_COLUMNS = ("image_path", "signer_id", "is_forgery", "split", "dataset")


@dataclass(frozen=True)
class Row:
    """One manifest row with its resolved path."""

    path: Path
    rel_path: str
    signer_id: str
    is_forgery: bool
    split: str
    dataset: str


@dataclass
class ImageStats:
    """Everything the audit needs from one image, kept as compact arrays."""

    sha256: str
    height: int
    width: int
    ink_pixels: int
    ink_ratio: float
    stroke_width: float | None
    thumb: np.ndarray                # uint8 64x128, ink-bbox cropped
    mask: np.ndarray                 # bool 64x128, bbox-cropped ink mask
    bits: tuple[int, ...]            # 1024-bit ink mask, sixteen 64-bit bands


def read_manifest(manifest: Path, image_root: Path | None = None) -> list[Row]:
    """Read and validate the manifest; paths resolve against ``image_root``.

    Raises:
        ValueError: A required column is missing, a cell is empty, a boolean is not
            ``true``/``false``, or a path escapes the image root.
    """
    manifest = manifest.expanduser().resolve()
    root = image_root.expanduser().resolve() if image_root else manifest.parent
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{manifest} has no header")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{manifest} is missing column(s): {missing}")
        rows: list[Row] = []
        for line, record in enumerate(reader, start=2):
            for column in REQUIRED_COLUMNS:
                if not record[column] or not record[column].strip():
                    raise ValueError(f"{manifest}:{line}: empty {column!r}")
            forgery_text = record["is_forgery"].strip().lower()
            if forgery_text not in {"true", "false"}:
                raise ValueError(
                    f"{manifest}:{line}: is_forgery must be true/false, got "
                    f"{record['is_forgery']!r}"
                )
            rel_path = record["image_path"].strip()
            path = (root / rel_path).resolve()
            if root not in path.parents and path != root:
                raise ValueError(f"{manifest}:{line}: path escapes image root: {rel_path}")
            rows.append(
                Row(
                    path=path,
                    rel_path=rel_path,
                    signer_id=record["signer_id"].strip(),
                    is_forgery=forgery_text == "true",
                    split=record["split"].strip(),
                    dataset=record["dataset"].strip(),
                )
            )
    if not rows:
        raise ValueError(f"{manifest} has no data rows")
    return rows


def stroke_width(image: np.ndarray) -> float | None:
    """Return the 2 x median distance-transform radius on the ink ridge.

    The distance transform gives each ink pixel its distance to the nearest background
    pixel. On the centre line of a stroke of width ``w`` that value is ``w / 2``, so
    doubling the median over ridge pixels estimates the pen width without depending on
    stroke length or image resolution.
    """
    ink = (image < INK_LEVEL).astype(np.uint8)
    if int(ink.sum()) < 20:
        return None
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    # Ridge = pixel equal to its 3x3 maximum; a flat ridge keeps every tied pixel.
    ridge = (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)) - 1e-6) & (
        distance > 0.5
    )
    if not ridge.any():
        return None
    return float(2.0 * np.median(distance[ridge]))


def _unit_thumb(thumb: np.ndarray) -> np.ndarray:
    """Return a mean-subtracted, L2-normalised identity vector for one thumbnail."""
    vector = (255.0 - thumb.astype(np.float32)).ravel()
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _binary_mask(image: np.ndarray) -> np.ndarray:
    """Return the 64x128 ink mask from full-resolution pixel coverage.

    Binarising after an INTER_AREA resize would drop 1-pixel strokes; resizing the binary
    mask with area coverage keeps them, which is what makes the IoU discriminative.
    """
    ink = (image < INK_LEVEL).astype(np.float32)
    return cv2.resize(ink, MASK_SIZE, interpolation=cv2.INTER_AREA) > MASK_COVERAGE


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.logical_or(first, second).sum())
    return float(np.logical_and(first, second).sum()) / union if union else 0.0


def _hash_bits(image: np.ndarray) -> tuple[int, ...]:
    """Pack a 32x32 ink mask into sixteen 64-bit bands.

    The mask is computed on the ink bounding box, not the full white canvas: a blank
    band would otherwise collide with every other blank signature and make the candidate
    lookup useless.
    """
    small = cv2.resize(image, (HASH_SIZE, HASH_SIZE), interpolation=cv2.INTER_AREA)
    bits = (small < INK_LEVEL).reshape(-1).astype(np.uint8)
    packed = np.packbits(bits).tobytes()
    return tuple(
        int.from_bytes(packed[start:start + 8], "big")
        for start in range(0, len(packed), 8)
    )


def load_image_stats(path: Path) -> ImageStats:
    """Decode one image once and return its audit statistics."""
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read signature image: {path}")
    if image.ndim != 2:
        raise ValueError(f"expected a single-channel image: {path}")
    ink = image < INK_LEVEL
    ink_pixels = int(ink.sum())
    if ink_pixels:
        ys, xs = np.nonzero(ink)
        margin = 2
        top = max(0, int(ys.min()) - margin)
        bottom = min(image.shape[0], int(ys.max()) + margin + 1)
        left = max(0, int(xs.min()) - margin)
        right = min(image.shape[1], int(xs.max()) + margin + 1)
        crop = image[top:bottom, left:right]
    else:
        crop = image
    thumb = cv2.resize(crop, THUMB_SIZE, interpolation=cv2.INTER_AREA)
    return ImageStats(
        sha256=hashlib.sha256(data).hexdigest(),
        height=int(image.shape[0]),
        width=int(image.shape[1]),
        ink_pixels=ink_pixels,
        ink_ratio=ink_pixels / float(image.size),
        stroke_width=stroke_width(image),
        thumb=thumb,
        mask=_binary_mask(crop),
        bits=_hash_bits(crop),
    )


@dataclass
class Separability:
    """Model-free signer separability for one dataset."""

    dataset: str
    genuine_images: int
    signers: int
    evaluated_signers: int
    accuracy: float | None
    chance: float
    own_similarity: float | None
    best_other: float | None
    gap: float | None
    forgery_similarity: float | None


def _values(stats: Sequence[ImageStats], rows: Sequence[Row], dataset: str, kind: str):
    """Return row/stat pairs for ``dataset`` and ``genuine`` or ``forgery``."""
    wanted_forgery = kind == "forgery"
    return [
        (rows[index], stats[index])
        for index in range(len(rows))
        if rows[index].dataset == dataset and rows[index].is_forgery == wanted_forgery
    ]


def signer_separability(
    rows: Sequence[Row], stats: Sequence[ImageStats], dataset: str
) -> Separability:
    """Leave-one-out nearest-centroid signer identification on ink thumbnails.

    Every genuine image is a probe: its similarity to its own signer's centroid is
    compared with the best other signer's centroid, with the probe removed from its own
    centroid. Accuracy is therefore a pixel-level upper bound on identity separability
    and is directly comparable across datasets because the thumbnail size is fixed.
    """
    genuine = _values(stats, rows, dataset, "genuine")
    if not genuine:
        return Separability(dataset, 0, 0, 0, None, 0.0, None, None, None, None)
    index_of = {id(stat): index for index, stat in enumerate(stats)}
    genuine_indices = [index_of[id(stat)] for _, stat in genuine]
    matrix = np.stack([_unit_thumb(stats[index].thumb) for index in genuine_indices])
    signers = [row.signer_id for row, _ in genuine]
    unique = sorted(set(signers))
    labels = np.asarray([unique.index(signer) for signer in signers])
    counts = np.bincount(labels, minlength=len(unique))
    sums = np.zeros((len(unique), matrix.shape[1]), dtype=np.float32)
    np.add.at(sums, labels, matrix)
    centroids = sums / counts[:, None]
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    unit_centroids = np.divide(
        centroids, norms, out=np.zeros_like(centroids), where=norms > 0
    )
    similarities = matrix @ unit_centroids.T

    own = np.full(len(matrix), np.nan)
    best_other = np.full(len(matrix), np.nan)
    for signer_index, count in enumerate(counts):
        if count < 2:
            continue
        members = np.flatnonzero(labels == signer_index)
        block = matrix[members] @ matrix[members].T
        own[members] = (block.sum(axis=1) - 1.0) / (count - 1)
        other = similarities[members].copy()
        other[:, signer_index] = -np.inf
        best_other[members] = other.max(axis=1)
    evaluated = ~np.isnan(own)
    accuracy = float(np.mean(own[evaluated] > best_other[evaluated])) if evaluated.any() else None

    forgery_similarity: float | None = None
    forgeries = _values(stats, rows, dataset, "forgery")
    if forgeries:
        forgery_matrix = np.stack(
            [_unit_thumb(stats[index_of[id(stat)]].thumb) for _, stat in forgeries]
        )
        forgery_labels = np.asarray(
            [
                unique.index(row.signer_id) if row.signer_id in unique else -1
                for row, _ in forgeries
            ]
        )
        valid = forgery_labels >= 0
        if valid.any():
            forgery_scores = forgery_matrix[valid] @ unit_centroids[forgery_labels[valid]].T
            # One centroid per probe: take the matched signer's own column.
            rows_index = np.arange(int(valid.sum()))
            matched = forgery_scores[rows_index, forgery_labels[valid]]
            forgery_similarity = float(np.median(matched))

    return Separability(
        dataset=dataset,
        genuine_images=len(matrix),
        signers=len(unique),
        evaluated_signers=int(np.sum(counts >= 2)),
        accuracy=accuracy,
        chance=1.0 / len(unique),
        own_similarity=float(np.median(own[evaluated])) if evaluated.any() else None,
        best_other=float(np.median(best_other[evaluated])) if evaluated.any() else None,
        gap=(
            float(np.median(own[evaluated] - best_other[evaluated]))
            if evaluated.any()
            else None
        ),
        forgery_similarity=forgery_similarity,
    )


@dataclass(frozen=True)
class DuplicatePair:
    """One cross-signer duplicate candidate."""

    first: int
    second: int
    similarity: float
    exact: bool


def exact_cross_signer_pairs(
    rows: Sequence[Row], stats: Sequence[ImageStats], max_pairs: int = 500
) -> tuple[list[tuple[int, int]], int]:
    """Return SHA-256-identical pairs whose rows have different signer ids.

    Within-signer exact duplicates are not a split leak and are ignored here; the
    cleaner already removed them from the dataset. Returns the pairs and the number of
    groups too large to enumerate.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for index, stat in enumerate(stats):
        groups[stat.sha256].append(index)
    pairs: list[tuple[int, int]] = []
    skipped = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        if len({rows[index].signer_id for index in members}) < 2:
            continue
        if len(members) > 200 or len(pairs) >= max_pairs:
            skipped += 1
            continue
        for first in range(len(members)):
            for second in range(first + 1, len(members)):
                if rows[members[first]].signer_id != rows[members[second]].signer_id:
                    pairs.append((members[first], members[second]))
    return pairs, skipped


def _ink_iou(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    intersection = sum((a & b).bit_count() for a, b in zip(first, second))
    union = sum((a | b).bit_count() for a, b in zip(first, second))
    return intersection / union if union else 0.0


def near_duplicate_pairs(
    rows: Sequence[Row],
    stats: Sequence[ImageStats],
    *,
    near_cosine: float = 0.97,
    min_iou: float = 0.70,
    min_ink_ratio: float = 0.005,
    max_bucket: int = 2000,
    max_candidates: int = 4000,
    max_pairs: int = 500,
) -> tuple[dict[tuple[int, int], tuple[float, float]], int, int]:
    """Find near-identical cross-signer images.

    Candidates come from sixteen 64-bit bands of a 1024-bit ink hash: two copies of an
    image keep at least one band intact even if resize interpolation flips a few bits.
    All-zero bands are left out of the index because they are blank paper and would put
    most of the corpus in one bucket. Each candidate is then verified with a 64x128 ink
    mask IoU and a cosine on the identity thumbnails; the mask keeps 1-pixel strokes that
    a grayscale downsample would blur away.

    Returns:
        ``(pairs, skipped_bands, truncated_images)`` where each pair maps to
        ``(cosine, iou)``. Informative buckets larger than ``max_bucket`` and images with
        more than ``max_candidates`` candidates are skipped and counted rather than
        silently compared.
    """
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, stat in enumerate(stats):
        for band, value in enumerate(stat.bits):
            if value == 0:
                continue
            buckets[(band, value)].append(index)
    cache: dict[int, np.ndarray] = {}

    def unit(index: int) -> np.ndarray:
        if index not in cache:
            cache[index] = _unit_thumb(stats[index].thumb)
        return cache[index]

    pairs: dict[tuple[int, int], tuple[float, float]] = {}
    skipped_bands = 0
    truncated = 0
    for first, stat in enumerate(stats):
        if stat.ink_ratio < min_ink_ratio:
            continue
        candidates: set[int] = set()
        for band, value in enumerate(stat.bits):
            if value == 0:
                continue
            members = buckets[(band, value)]
            if len(members) > max_bucket:
                skipped_bands += 1
                continue
            candidates.update(members)
        if len(candidates) > max_candidates:
            truncated += 1
            candidates = set(sorted(candidates)[:max_candidates])
        for second in candidates:
            if second <= first:
                continue
            other = stats[second]
            if rows[first].signer_id == rows[second].signer_id:
                continue
            if other.ink_ratio < min_ink_ratio:
                continue
            iou = _mask_iou(stat.mask, other.mask)
            if iou < min_iou:
                continue
            similarity = float(unit(first) @ unit(second))
            if similarity < near_cosine:
                continue
            pairs[(first, second)] = (similarity, iou)
            if len(pairs) >= max_pairs:
                return pairs, skipped_bands, truncated
    return pairs, skipped_bands, truncated


OP_ORDER = (
    "rotation",
    "scale_shift",
    "thick",
    "thin",
    "lighter_ink",
    "blur",
    "noise_jpeg",
    "junk",
    "brightness_contrast",
)
GROUP1_OPS = frozenset(
    {"rotation", "scale_shift", "thick", "thin", "lighter_ink", "blur", "noise_jpeg"}
)
GROUP2_OPS = frozenset({"junk", "brightness_contrast"})


@dataclass
class OpRetention:
    """Ink retention of one isolated augmentation op on one dataset."""

    op: str
    dataset: str
    samples: int
    median_loss: float
    p90_loss: float
    over_5pct: float


def apply_op(augment: SignatureAugment, op: str, image: np.ndarray) -> np.ndarray:
    """Apply exactly one transform from the augmenter.

    Calling the private transform keeps this audit honest: the random pipeline composes
    several ops, so measuring it would answer "what does the pipeline do" instead of
    "what does this op cost in ink".
    """
    if op == "rotation":
        return augment._geometry(image, True, False, [])
    if op == "scale_shift":
        return augment._geometry(image, False, True, [])
    if op == "thick":
        return augment._thickness(image, True)
    if op == "thin":
        return augment._thickness(image, False)
    if op == "lighter_ink":
        return augment._lighter_ink(image)
    if op == "blur":
        return augment._blur(image)
    if op == "noise_jpeg":
        return augment._noise_jpeg(image)
    if op == "junk":
        return augment._junk(image)[0]
    if op == "brightness_contrast":
        return augment._brightness_contrast(image)
    raise ValueError(f"unknown augmentation op: {op}")


def _ink_pixels(image: np.ndarray) -> int:
    return int((image < INK_LEVEL).sum())


def augmentation_retention(
    rows: Sequence[Row], *, sample_per_dataset: int = 100, seed: int = 42
) -> list[OpRetention]:
    """Measure ink retention per op on a seeded sample of train images.

    Only the train split is used because that is the only split the augmenter ever sees.
    A loss is ``1 - after / before`` at the 200 ink cut: positive means pixels were
    removed or pushed above the cut, negative means the op spread ink across it.
    """
    results: list[OpRetention] = []
    for dataset in sorted({row.dataset for row in rows}):
        candidates = [
            index
            for index, row in enumerate(rows)
            if row.dataset == dataset and row.split == "train"
        ]
        if not candidates:
            continue
        take = min(sample_per_dataset, len(candidates))
        rng = np.random.default_rng([seed, len(dataset), len(candidates)])
        chosen = rng.choice(np.asarray(candidates), size=take, replace=False)
        losses: dict[str, list[float]] = {op: [] for op in OP_ORDER}
        augment = SignatureAugment(seed=seed)
        # Deterministic stream even though the audit is a one-off run.
        augment._rng = np.random.default_rng(seed)
        for index in chosen:
            path = rows[int(index)].path
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"could not read signature image: {path}")
            before = _ink_pixels(image)
            if before == 0:
                continue
            for op in OP_ORDER:
                after = _ink_pixels(apply_op(augment, op, image))
                losses[op].append(1.0 - after / before)
        for op in OP_ORDER:
            values = np.asarray(losses[op], dtype=np.float64)
            if values.size == 0:
                continue
            results.append(
                OpRetention(
                    op=op,
                    dataset=dataset,
                    samples=int(values.size),
                    median_loss=float(np.median(values)),
                    p90_loss=float(np.percentile(values, 90)),
                    over_5pct=float(np.mean(values > 0.05)),
                )
            )
    return results


def _fmt(value: object, spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return format(float(value), spec)


def summarise_stroke(
    rows: Sequence[Row], stats: Sequence[ImageStats]
) -> tuple[dict[str, dict[str, object]], float | None]:
    """Per-dataset stroke width percentiles plus the pooled non-SigResize median."""
    summary: dict[str, dict[str, object]] = {}
    others: list[float] = []
    for dataset in sorted({row.dataset for row in rows}):
        values = np.asarray(
            [
                stat.stroke_width
                for row, stat in zip(rows, stats)
                if row.dataset == dataset and stat.stroke_width is not None
            ],
            dtype=np.float64,
        )
        summary[dataset] = {
            "n": int(values.size),
            "median": float(np.median(values)) if values.size else None,
            "p10": float(np.percentile(values, 10)) if values.size else None,
            "p90": float(np.percentile(values, 90)) if values.size else None,
        }
        if dataset != "SigResize":
            others.extend(values.tolist())
    other_array = np.asarray(others, dtype=np.float64)
    return summary, (float(np.median(other_array)) if other_array.size else None)


def print_stroke(summary: dict[str, dict[str, object]], others_median: float | None) -> None:
    print("--- stroke width, pixels (distance-transform ridge; 2 x radius) ---")
    print(f"{'dataset':<12}{'n':>7}{'median':>9}{'p10':>8}{'p90':>8}{'vs_others':>11}")
    for dataset, row in summary.items():
        median = row["median"]
        ratio = (
            float(median) / others_median
            if median is not None and others_median
            else None
        )
        print(
            f"{dataset:<12}{_fmt(row['n']):>7}{_fmt(median, '.2f'):>9}"
            f"{_fmt(row['p10'], '.2f'):>8}{_fmt(row['p90'], '.2f'):>8}"
            f"{_fmt(ratio, '.2f') + 'x' if ratio is not None else 'n/a':>11}"
        )
    print(f"  non-SigResize pooled median: {_fmt(others_median, '.2f')} px\n")


def print_separability(results: Sequence[Separability]) -> None:
    print("--- signer separability (genuine only, pixel nearest-centroid) ---")
    print(
        f"{'dataset':<12}{'signers':>8}{'evaluated':>10}{'LOO_acc':>9}{'chance':>8}"
        f"{'own_sim':>9}{'best_other':>11}{'gap':>8}{'forgery_sim':>12}"
    )
    for row in results:
        print(
            f"{row.dataset:<12}{_fmt(row.signers):>8}{_fmt(row.evaluated_signers):>10}"
            f"{_fmt(row.accuracy, '.3f'):>9}{_fmt(row.chance, '.3f'):>8}"
            f"{_fmt(row.own_similarity, '.3f'):>9}{_fmt(row.best_other, '.3f'):>11}"
            f"{_fmt(row.gap, '.3f'):>8}{_fmt(row.forgery_similarity, '.3f'):>12}"
        )
    print("  LOO_acc is a pixel-level upper bound, not a model score; forgery_sim is the")
    print("  median cosine of a skilled forgery to the genuine centroid of its signer.\n")


def _kind(row: Row) -> str:
    return "forgery" if row.is_forgery else "genuine"


def print_duplicates(
    rows: Sequence[Row],
    exact: Sequence[tuple[int, int]],
    near: dict[tuple[int, int], tuple[float, float]],
    skipped_bands: int,
    truncated_images: int,
    top: int,
) -> None:
    print("--- cross-signer duplicates (exact SHA-256 and near-identical) ---")
    print(f"  exact cross-signer pairs: {len(exact)}")
    print(
        f"  near-identical cross-signer pairs: {len(near)} "
        f"(skipped bands={skipped_bands}, truncated images={truncated_images})"
    )
    if near:
        by_pair: Counter[tuple[str, str]] = Counter()
        for first, second in near:
            by_pair[tuple(sorted((rows[first].dataset, rows[second].dataset)))] += 1
        print("  near pairs by dataset pair: " + ", ".join(
            f"{a}/{b}={count}" for (a, b), count in sorted(by_pair.items())
        ))
    if exact:
        print("  exact pairs:")
        for first, second in exact[:top]:
            print(
                f"    {rows[first].signer_id} ({_kind(rows[first])}) == "
                f"{rows[second].signer_id} ({_kind(rows[second])})"
            )
            print(f"      {rows[first].rel_path}")
            print(f"      {rows[second].rel_path}")
        if len(exact) > top:
            print(f"    ... {len(exact) - top} more exact pairs")
    if near:
        print(f"  top {min(top, len(near))} near pairs by cosine:")
        for (first, second), (similarity, iou) in sorted(
            near.items(), key=lambda item: -item[1][0]
        )[:top]:
            print(
                f"    cos={similarity:.4f}  iou={iou:.3f}  "
                f"{rows[first].dataset}/{rows[second].dataset}  "
                f"{rows[first].signer_id} ({_kind(rows[first])}) vs "
                f"{rows[second].signer_id} ({_kind(rows[second])})"
            )
            print(f"      {rows[first].rel_path}")
            print(f"      {rows[second].rel_path}")
        if len(near) > top:
            print(f"    ... {len(near) - top} more near pairs")
    print()


def _effective_probabilities() -> dict[str, float]:
    """Read the probabilities the current augmenter actually uses by default."""
    defaults = SignatureAugment()
    return {
        "rotation": defaults.p_rot,
        "scale_shift": defaults.p_scaleshift,
        "thick": defaults.p_thick,
        "thin": defaults.p_thin,
        "lighter_ink": defaults.p_light,
        "blur": defaults.p_blur,
        "noise_jpeg": defaults.p_noise_jpeg,
        # Group 2 is off by default, so its effective probability is zero.
        "junk": 0.0,
        "brightness_contrast": 0.0,
    }


def print_ops(results: Sequence[OpRetention]) -> None:
    probabilities = _effective_probabilities()
    print("--- ink lost per augmentation op (train only, each op applied alone) ---")
    print("  loss = 1 - after/before at the <200 ink cut; negative means ink was spread")
    print(
        f"{'op':<20}{'p_on':>8}  {'dataset':<12}{'n':>6}{'median':>9}{'p90':>9}{'>5%':>8}"
    )
    for row in results:
        tag = "*" if row.op in GROUP2_OPS else " "
        print(
            f"{row.op + tag:<20}{probabilities[row.op]:>8.2f}  {row.dataset:<12}"
            f"{row.samples:>6}{row.median_loss:>9.3f}{row.p90_loss:>9.3f}"
            f"{row.over_5pct * 100:>7.1f}%"
        )
    print("  * group-2 op, disabled by default. >5% is the share of sampled images that")
    print("  lost more than 5% of their ink to that op.\n")


def _recommendations(
    stroke: dict[str, dict[str, object]],
    others_median: float | None,
    separability: Sequence[Separability],
    exact: Sequence[tuple[int, int]],
    near: dict[tuple[int, int], tuple[float, float]],
    ops: Sequence[OpRetention],
) -> list[str]:
    """Turn the measurements into concrete, unapplied proposals."""
    proposals: list[str] = []
    sigresize = stroke.get("SigResize", {})
    median = sigresize.get("median")
    if median is not None and others_median:
        ratio = float(median) / others_median
        if abs(ratio - 1.0) > 0.15:
            proposals.append(
                f"SigResize stroke width is {ratio:.2f}x the scanned corpora "
                f"({float(median):.2f} px vs {others_median:.2f} px). Either normalise "
                "stroke width per dataset before training (morphological open/close to "
                "the pooled median) or report per-dataset thresholds; do not retrain on "
                "one global threshold first."
            )
    accuracies = {
        row.dataset: row.accuracy for row in separability if row.accuracy is not None
    }
    if "SigResize" in accuracies and len(accuracies) > 1:
        others = [value for name, value in accuracies.items() if name != "SigResize"]
        if accuracies["SigResize"] < min(others) - 0.05:
            proposals.append(
                f"SigResize signers are less separable at pixel level "
                f"({accuracies['SigResize']:.3f} vs {min(others):.3f} for the weakest other "
                "corpus). Keep a per-dataset breakdown in every report and avoid "
                "dataset-balanced batches until the ablation shows otherwise."
            )
    if exact or near:
        proposals.append(
            f"{len(exact)} exact and {len(near)} near-identical cross-signer pairs found. "
            "Deduplicate by SHA-256 plus the 256-bit ink hash, keeping the higher-priority "
            "corpus (BHSig260 > CEDAR > SigResize) and dropping the other copy; if the two "
            "signers are genuinely the same person, merge their signer ids instead. Then "
            "rebuild the splits and re-run training.verify_data. No manifest change has "
            "been made."
        )
    lossy = [
        row for row in ops if row.dataset == "SigResize" and row.median_loss > 0.05
    ]
    if lossy:
        worst = max(lossy, key=lambda row: row.median_loss)
        proposals.append(
            f"On SigResize, {worst.op} loses a median {worst.median_loss * 100:.1f}% of "
            "ink in isolation. Lower its probability or strength before adding more "
            "transforms."
        )
    if not proposals:
        proposals.append("No finding crossed its threshold; no change proposed.")
    return proposals


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Directory for relative image_path values (default: the manifest's folder)",
    )
    parser.add_argument(
        "--near-cosine",
        type=float,
        default=0.97,
        help="Cosine on 32x64 thumbnails above which a candidate is near-identical.",
    )
    parser.add_argument(
        "--near-iou",
        type=float,
        default=0.70,
        help="Minimum 16x16 ink-mask IoU for a near-duplicate candidate.",
    )
    parser.add_argument(
        "--min-ink-ratio",
        type=float,
        default=0.005,
        help=(
            "Images below this ink ratio are skipped for near-duplicates: their hashes "
            "are dominated by blank paper and collide with everything."
        ),
    )
    parser.add_argument(
        "--augment-sample",
        type=int,
        default=100,
        help="Train images sampled per dataset for the isolated-op ink-loss section.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-duplicates", type=int, default=20)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=4000,
        help="Per-image candidate cap for near-duplicate lookup; over this is counted.",
    )
    args = parser.parse_args()
    if not 0.0 < args.near_cosine <= 1.0:
        parser.error("--near-cosine must be in (0, 1]")
    if not 0.0 <= args.near_iou <= 1.0:
        parser.error("--near-iou must be in [0, 1]")
    if not 0.0 <= args.min_ink_ratio < 1.0:
        parser.error("--min-ink-ratio must be in [0, 1)")
    if args.augment_sample < 1:
        parser.error("--augment-sample must be positive")
    if args.top_duplicates < 1:
        parser.error("--top-duplicates must be positive")
    if args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    return args


def main() -> int:
    args = _parse_args()
    rows = read_manifest(args.manifest, args.image_root)
    datasets = sorted({row.dataset for row in rows})
    print("=== manifest ===")
    print(f"manifest: {args.manifest.expanduser().resolve()}")
    print(f"rows: {len(rows)}  datasets: {datasets}")
    print(f"signers: {len({row.signer_id for row in rows})}")
    print(
        "splits: "
        + ", ".join(f"{name}={count}" for name, count in sorted(Counter(r.split for r in rows).items()))
    )
    print(
        "kinds: "
        + ", ".join(
            f"{name}={count}"
            for name, count in sorted(Counter(_kind(r) for r in rows).items())
        )
    )

    stats: list[ImageStats] = []
    print(f"\nloading {len(rows)} images ...", file=sys.stderr, flush=True)
    for index, row in enumerate(rows, start=1):
        try:
            stats.append(load_image_stats(row.path))
        except Exception as exc:  # noqa: BLE001 - report the offending row, then stop
            raise SystemExit(f"failed on {row.rel_path}: {exc}") from exc
        if index % 2000 == 0 or index == len(rows):
            print(f"  loaded {index}/{len(rows)}", file=sys.stderr, end="\r", flush=True)
    print(file=sys.stderr)

    shapes = Counter((stat.height, stat.width) for stat in stats)
    print(
        "shapes (H x W): "
        + ", ".join(f"{height}x{width}={count}" for (height, width), count in sorted(shapes.items()))
    )
    print()

    stroke, others_median = summarise_stroke(rows, stats)
    print_stroke(stroke, others_median)

    separability = [signer_separability(rows, stats, dataset) for dataset in datasets]
    print_separability(separability)

    exact, skipped_groups = exact_cross_signer_pairs(rows, stats)
    near, skipped_bands, truncated_images = near_duplicate_pairs(
        rows,
        stats,
        near_cosine=args.near_cosine,
        min_iou=args.near_iou,
        min_ink_ratio=args.min_ink_ratio,
        max_candidates=args.max_candidates,
    )
    print_duplicates(
        rows,
        exact,
        near,
        skipped_bands,
        truncated_images,
        args.top_duplicates,
    )
    if skipped_groups:
        print(f"  warning: {skipped_groups} exact-duplicate group(s) were too large to list\n")

    ops = augmentation_retention(
        rows, sample_per_dataset=args.augment_sample, seed=args.seed
    )
    print_ops(ops)

    print("=== proposed fixes (NOT applied; manifest untouched) ===")
    for number, proposal in enumerate(
        _recommendations(stroke, others_median, separability, exact, near, ops), start=1
    ):
        print(f"  {number}. {proposal}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

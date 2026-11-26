"""Read-only verification of the cleaned 256x512 signature manifest.

Three independent layers, nothing is modified:

1. manifest: required columns, no empty cells, no duplicate paths, case-insensitive
   true/false booleans, signer-disjoint splits, per-split row and signer counts,
   and a genuine plus a forgery for every signer
2. images: every row exists, decodes as uint8 grayscale 256x512, is not blank,
   has dark ink on a white background, and has an ink ratio inside a sane band
3. augmentation: group-1 SignatureAugment over the train split only, checking
   shape, dtype, and how much ink survives

The image pass reads through ``load_preprocessed_image``, the same loader the
training dataset uses, so the verifier cannot disagree with training about
decode behaviour, shape, or dtype. Relative paths resolve against the manifest's
own folder unless ``--image-root`` overrides it; a path that escapes that root is
reported as a failure rather than read.

Usage:
    python -m training.verify_data --manifest training_clean/manifest_clean.csv
    python -m training.verify_data --manifest training_clean/manifest_clean.csv --skip-augment

Exit code is 0 when every section passes and 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from preprocessing.signature_preprocessing import load_preprocessed_image
from training.augment import SignatureAugment

IMAGE_HEIGHT, IMAGE_WIDTH = 256, 512
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)
PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH

INK_LEVEL = 200  # same "ink" cut the augmenter and binarisation use
INK_CORE = 128   # stricter cut, reported only: lighter_ink is meant to leave this band
MIN_INK_PIXELS = 5
MIN_BACKGROUND = 200  # 90th percentile of a dark-on-white image
FAINT_MARGIN = 30     # darkest pixel within this of the cut is reported as faint

REQUIRED_COLUMNS = ("image_path", "signer_id", "is_forgery", "split", "dataset")
VALID_SPLITS = ("train", "val", "test")
VALID_BOOLEANS = ("true", "false")

EXPECTED_ROWS = {"train": 14_401, "val": 3_114, "test": 3_033}
EXPECTED_SIGNERS = {"train": 296, "val": 63, "test": 62}


@dataclass
class Section:
    """One report section: a PASS/FAIL verdict plus the evidence behind it."""

    name: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        """Record something worth seeing that is not a violation on its own."""
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    @property
    def passed(self) -> bool:
        return not self.failures


def _percentiles(values: list[float], label: str) -> str:
    """Format min/p1/p50/p99/max so an over-tight band is obvious, not hidden."""
    if not values:
        return f"{label}: n/a (no images measured)"
    array = np.asarray(values, dtype=np.float64)
    return (
        f"{label}: min {array.min():.4f} p1 {np.percentile(array, 1):.4f} "
        f"p50 {np.percentile(array, 50):.4f} p99 {np.percentile(array, 99):.4f} "
        f"max {array.max():.4f}"
    )


def _print_section(section: Section, max_shown: int) -> None:
    status = "PASS" if section.passed else "FAIL"
    print(f"=== {section.name} === {status}")
    for note in section.notes:
        print(f"  {note}")
    for warning in section.warnings[:max_shown]:
        print(f"  ~ {warning}")
    hidden_warnings = len(section.warnings) - max_shown
    if hidden_warnings > 0:
        print(f"  ... and {hidden_warnings} more warning(s) not shown")
    for failure in section.failures[:max_shown]:
        print(f"  ! {failure}")
    hidden = len(section.failures) - max_shown
    if hidden > 0:
        print(f"  ... and {hidden} more failure(s) not shown")
    print()


def _resolve(root: Path, image_path: str) -> Path:
    path = Path(image_path).expanduser()
    return path if path.is_absolute() else root / path


def _load(path: Path) -> np.ndarray:
    """Read one preprocessed image through the production loading path."""
    return load_preprocessed_image(
        path,
        already_preprocessed=True,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
    )


def verify_manifest(manifest_path: Path) -> tuple[Section, list[dict[str, str]]]:
    """Check the CSV contract and the signer-disjoint split bookkeeping."""
    section = Section("manifest")
    if not manifest_path.is_file():
        section.fail(f"manifest not found: {manifest_path}")
        return section, []

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        if missing:
            section.fail(f"missing required column(s): {', '.join(missing)}")
            return section, []
        rows = list(reader)

    if not rows:
        section.fail("manifest has no data rows")
        return section, []

    signer_splits: dict[str, set[str]] = defaultdict(set)
    genuine: Counter[str] = Counter()
    forgeries: Counter[str] = Counter()
    split_rows: Counter[str] = Counter()
    split_signers: dict[str, set[str]] = defaultdict(set)
    datasets: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()

    for line_number, row in enumerate(rows, start=2):
        values = {name: (row.get(name) or "").strip() for name in REQUIRED_COLUMNS}
        empty = [name for name, value in values.items() if not value]
        if empty:
            section.fail(f"row {line_number}: empty cell(s) in {', '.join(empty)}")
            continue

        split = values["split"]
        is_forgery = values["is_forgery"].lower()
        if is_forgery not in VALID_BOOLEANS:
            section.fail(
                f"row {line_number}: is_forgery={values['is_forgery']!r}, "
                "expected true/false"
            )
            continue
        if split.lower() not in VALID_SPLITS:
            section.fail(f"row {line_number}: split={split!r}, expected train/val/test")
            continue

        split = split.lower()
        signer_id = values["signer_id"]
        path_counts[values["image_path"]] += 1
        signer_splits[signer_id].add(split)
        split_rows[split] += 1
        split_signers[split].add(signer_id)
        datasets[values["dataset"]] += 1
        if is_forgery == "true":
            forgeries[signer_id] += 1
        else:
            genuine[signer_id] += 1

    duplicates = [path for path, count in path_counts.items() if count > 1]
    for path in duplicates[:20]:
        section.fail(f"duplicate image_path ({path_counts[path]} rows): {path}")
    if len(duplicates) > 20:
        section.fail(f"... and {len(duplicates) - 20} more duplicated image_path values")

    for signer_id, splits in sorted(signer_splits.items()):
        if len(splits) > 1:
            section.fail(f"signer {signer_id} appears in multiple splits: {sorted(splits)}")

    for split, expected in EXPECTED_ROWS.items():
        if split_rows[split] != expected:
            section.fail(f"{split} rows: {split_rows[split]}, expected {expected}")
    for split, expected in EXPECTED_SIGNERS.items():
        if len(split_signers[split]) != expected:
            section.fail(f"{split} signers: {len(split_signers[split])}, expected {expected}")

    for signer_id in sorted(signer_splits):
        if genuine[signer_id] == 0:
            section.fail(f"signer {signer_id}: no genuine image")
        if forgeries[signer_id] == 0:
            section.fail(f"signer {signer_id}: no forgery image")

    section.note(f"{len(rows):,} rows, {len(signer_splits)} signers")
    section.note(
        "rows: "
        + ", ".join(f"{split} {split_rows[split]:,}" for split in VALID_SPLITS)
        + " | signers: "
        + ", ".join(f"{split} {len(split_signers[split])}" for split in VALID_SPLITS)
    )
    section.note(
        f"genuine {sum(genuine.values()):,} / forgery {sum(forgeries.values()):,}"
        f" | datasets: {dict(sorted(datasets.items()))}"
    )
    section.note(
        f"genuine per signer: min {min(genuine.values()) if genuine else 'n/a'}"
        f" | forgeries per signer: min {min(forgeries.values()) if forgeries else 'n/a'}"
    )
    return section, rows


def verify_images(
    rows: list[dict[str, str]], root: Path, args: argparse.Namespace
) -> Section:
    """Check every image can be read and looks like a dark-on-white signature.

    ``image.min()`` is the darkness statistic, deliberately not a low percentile.
    A signature covering under 1% of the canvas forces every percentile up to 255,
    so a percentile cannot tell "no ink" apart from "sparse ink".
    """
    section = Section("images")
    ink_ratios: list[float] = []
    backgrounds: list[float] = []
    darkest_by_dataset: dict[str, list[float]] = defaultdict(list)
    ink_by_dataset: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        image_path = row["image_path"].strip()
        dataset = row["dataset"].strip()
        path = _resolve(root, image_path)
        if not path.resolve().is_relative_to(root):
            section.fail(f"{image_path}: resolves outside the image root ({path})")
            continue
        try:
            image = _load(path)
        except (FileNotFoundError, ValueError) as error:
            section.fail(str(error))
            continue
        if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
            section.fail(f"{image_path}: got shape {image.shape} dtype {image.dtype}")
            continue

        ink_pixels = int((image < INK_LEVEL).sum())
        if ink_pixels < MIN_INK_PIXELS:
            section.fail(f"{image_path}: blank, no ink below {INK_LEVEL}")
            continue

        ratio = ink_pixels / PIXELS
        background = float(np.percentile(image, 90))
        darkest = float(image.min())
        ink_ratios.append(ratio)
        backgrounds.append(background)
        darkest_by_dataset[dataset].append(darkest)
        ink_by_dataset[dataset].append(ratio)

        if background < MIN_BACKGROUND:
            section.fail(f"{image_path}: background not white (p90={background:.0f})")
        if darkest >= args.min_dark_pixel:
            section.fail(f"{image_path}: no dark ink (darkest pixel {darkest:.0f})")
        elif darkest >= args.min_dark_pixel - FAINT_MARGIN:
            section.warn(f"{image_path}: faint ink (darkest pixel {darkest:.0f})")
        if not args.ink_ratio_min <= ratio <= args.ink_ratio_max:
            section.fail(
                f"{image_path}: ink ratio {ratio:.4f} outside "
                f"[{args.ink_ratio_min}, {args.ink_ratio_max}]"
            )

    section.note(f"checked {len(rows):,} images")
    section.note(_percentiles(ink_ratios, "ink ratio"))
    section.note(_percentiles(backgrounds, "background p90 (0-255)"))
    section.note(
        _percentiles(
            [value for values in darkest_by_dataset.values() for value in values],
            "darkest pixel (0-255)",
        )
    )
    for dataset in sorted(darkest_by_dataset):
        darkest = np.asarray(darkest_by_dataset[dataset])
        ink = np.asarray(ink_by_dataset[dataset])
        section.note(
            f"  {dataset or '(blank)'}: n={len(darkest):,}"
            f" darkest max {darkest.max():.0f}"
            f" | ink ratio {ink.min():.4f}-{ink.max():.4f} median {np.median(ink):.4f}"
        )
    return section


def verify_augmentation(
    rows: list[dict[str, str]], root: Path, args: argparse.Namespace
) -> Section:
    """Run group-1 augmentation over train rows only and measure ink survival.

    Ink is counted below ``INK_LEVEL`` because that is the cut the augmenter and the
    binarisation step use. The ink core (``INK_CORE``) is reported separately:
    ``lighter_ink`` is *designed* to move ink out of that band, so a missing core
    means a lighter signature, not a destroyed one. Only a zero count at
    ``INK_LEVEL`` fails the section; anything under ``--min-retention`` is a warning.
    """
    section = Section("augmentation (train only)")
    train_rows = [row for row in rows if row["split"].strip().lower() == "train"]
    if args.limit_augment:
        train_rows = train_rows[: args.limit_augment]

    augmenter = SignatureAugment(group2=False, seed=args.seed)
    retentions: list[float] = []
    core_retentions: list[float] = []
    retention_by_dataset: dict[str, list[float]] = defaultdict(list)
    operations: Counter[str] = Counter()
    worst_per_operation: dict[str, tuple[float, str]] = {}
    erased = 0
    near_erased = 0
    fed = 0

    for row in train_rows:
        image_path = row["image_path"].strip()
        dataset = row["dataset"].strip()
        assert row["split"].strip().lower() == "train", "only train rows may be augmented"
        try:
            image = _load(_resolve(root, image_path))
        except (FileNotFoundError, ValueError) as error:
            section.fail(str(error))
            continue

        before = int((image < INK_LEVEL).sum())
        before_core = int((image < INK_CORE).sum())
        if before == 0:
            section.fail(f"{image_path}: no ink before augmentation")
            continue

        augmented, used = augmenter(image, return_ops=True)
        fed += 1
        operations.update(used)

        if augmented.shape != IMAGE_SHAPE or augmented.dtype != np.uint8:
            section.fail(
                f"{image_path}: augmentation returned shape {augmented.shape} "
                f"dtype {augmented.dtype}"
            )
            continue

        after = int((augmented < INK_LEVEL).sum())
        after_core = int((augmented < INK_CORE).sum())
        retention = after / before
        retentions.append(retention)
        retention_by_dataset[dataset].append(retention)
        core_retentions.append(after_core / before_core if before_core else 1.0)
        for operation in used:
            if retention < worst_per_operation.get(operation, (2.0, ""))[0]:
                worst_per_operation[operation] = (retention, image_path)

        if after == 0:
            erased += 1
            section.fail(f"{image_path}: ink erased (no pixel below {INK_LEVEL})")
        elif retention < args.min_retention:
            near_erased += 1
            section.warn(
                f"{image_path}: kept {retention:.1%} of ink "
                f"({', '.join(used) or 'no ops'})"
            )

    section.note(
        f"fed to augmenter: train {fed:,}, val 0, test 0 "
        f"(val/test are never passed in)"
    )
    section.note(_percentiles(retentions, f"ink retention <{INK_LEVEL}"))
    section.note(
        _percentiles(core_retentions, f"ink retention <{INK_CORE} (lighter ink lowers this)")
    )
    for dataset in sorted(retention_by_dataset):
        values = np.asarray(retention_by_dataset[dataset])
        section.note(
            f"  {dataset or '(blank)'}: n={len(values):,} min {values.min():.4f}"
            f" mean {values.mean():.4f}"
        )
    section.note(
        f"erased: {erased} | below {args.min_retention:.0%} retention: {near_erased}"
    )
    section.note(
        "operations fired: "
        + ", ".join(f"{name} {count:,}" for name, count in sorted(operations.items()))
    )
    section.note(
        "worst retention per operation: "
        + ", ".join(
            f"{name} {value:.4f}"
            for name, (value, _) in sorted(
                worst_per_operation.items(), key=lambda item: item[1][0]
            )
        )
    )
    return section


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Directory for relative image_path values (default: the manifest's folder)",
    )
    parser.add_argument("--ink-ratio-min", type=float, default=0.0005)
    parser.add_argument("--ink-ratio-max", type=float, default=0.50)
    parser.add_argument(
        "--min-dark-pixel",
        type=int,
        default=180,
        help="Fail an image whose darkest pixel is at or above this 0-255 value",
    )
    parser.add_argument(
        "--min-retention",
        type=float,
        default=0.05,
        help="Warn when augmentation keeps less than this fraction of the ink",
    )
    parser.add_argument("--seed", type=int, default=42, help="SignatureAugment seed")
    parser.add_argument("--skip-augment", action="store_true")
    parser.add_argument(
        "--limit-augment",
        type=int,
        default=0,
        help="Augment only the first N train images (0 = all); for quick runs",
    )
    parser.add_argument("--max-failures-shown", type=int, default=50)
    args = parser.parse_args()
    if not 0.0 <= args.ink_ratio_min < args.ink_ratio_max <= 1.0:
        parser.error("ink ratio band must satisfy 0 <= min < max <= 1")
    if not 0.0 <= args.min_retention <= 1.0:
        parser.error("--min-retention must be between 0 and 1")
    if not 0 <= args.min_dark_pixel <= 255:
        parser.error("--min-dark-pixel must be between 0 and 255")
    return args


def main() -> int:
    args = _parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    root = (
        args.image_root.expanduser().resolve()
        if args.image_root
        else manifest_path.parent
    )
    print(f"manifest:   {manifest_path}")
    print(f"image root: {root}\n")

    manifest_section, rows = verify_manifest(manifest_path)
    sections = [manifest_section]
    if rows:
        sections.append(verify_images(rows, root, args))
        if not args.skip_augment:
            sections.append(verify_augmentation(rows, root, args))

    for section in sections:
        _print_section(section, args.max_failures_shown)

    failed = [section for section in sections if not section.passed]
    verdict = "PASS" if not failed else "FAIL"
    print(
        f"RESULT: {verdict} "
        f"({len(sections) - len(failed)}/{len(sections)} sections passed)"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

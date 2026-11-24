"""Clean the training-ready signature dataset without modifying source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def image_hash(path: Path) -> tuple[str, str | None]:
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            pixels = gray.tobytes()
        return (hashlib.md5(pixels).hexdigest(), None)
    except (OSError, UnidentifiedImageError) as error:
        return ("", str(error))


def count_rows(frame: pd.DataFrame) -> int:
    return int(len(frame))


def validate_manifest(frame: pd.DataFrame) -> None:
    required = {
        "image_path",
        "signer_id",
        "is_forgery",
        "split",
        "dataset",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")


def clean_dataset(dataset_root: Path, output_root: Path) -> dict[str, object]:
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    frame = pd.read_csv(manifest_path)
    validate_manifest(frame)
    before = count_rows(frame)
    report: dict[str, object] = {"rows": {"before": before}}

    open_failures: list[dict[str, str]] = []
    valid_mask: dict[str, bool] = {}
    hash_mask: dict[str, str] = {}

    def resolve(row: pd.Series) -> Path:
        source = Path(str(row["image_path"]))
        return source if source.is_absolute() else dataset_root / source

    paths = [resolve(row) for _, row in frame.iterrows()]
    with ThreadPoolExecutor(max_workers=min(32, max(1, len(paths)))) as executor:
        results = list(
            tqdm(
                executor.map(image_hash, paths),
                total=len(paths),
                desc="Verifying and hashing",
                unit="image",
            )
        )

    for row_index, (path, (digest, error)) in enumerate(zip(paths, results)):
        if error is not None:
            open_failures.append(
                {"image_path": str(frame.iloc[row_index]["image_path"]), "reason": error}
            )
            valid_mask[str(row_index)] = False
            continue
        valid_mask[str(row_index)] = True
        hash_mask[str(row_index)] = digest

    initial_failure_paths = {
        str(frame.iloc[int(index)]["image_path"])
        for index, valid in valid_mask.items()
        if not valid
    }
    if open_failures:
        frame = frame[
            ~frame["image_path"].astype(str).isin(initial_failure_paths)
        ].copy()

    report["open_failures"] = open_failures
    report["rows"]["after_open_validation"] = count_rows(frame)

    kept_indices: set[int] = set()
    seen_hashes: set[str] = set()
    for original_index, row in frame.iterrows():
        index_key = str(original_index)
        digest = hash_mask.get(index_key)
        if digest is None:
            continue
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        kept_indices.add(original_index)

    duplicate_count = count_rows(frame) - len(kept_indices)
    frame = frame.loc[
        sorted(kept_indices),
        list(frame.columns),
    ].copy()
    report["rows"]["after_duplicate_removal"] = count_rows(frame)
    report["exact_duplicates_removed"] = int(duplicate_count)

    thai_count = int((frame["dataset"] == "Thai").sum())
    frame = frame.loc[frame["dataset"] != "Thai"].copy()
    report["rows"]["after_thai_removal"] = count_rows(frame)
    report["thai_rows_removed"] = thai_count

    counts = (
        frame.groupby("signer_id")
        .agg(
            genuine_count=("is_forgery", lambda values: int((~parse_bool(values)).sum())),
            forgery_count=("is_forgery", lambda values: int(parse_bool(values).sum())),
        )
        .reset_index()
    )
    invalid_signers = counts.loc[
        (counts["genuine_count"] < 5) | (counts["forgery_count"] < 1),
        "signer_id",
    ].tolist()
    invalid_signer_set = set(invalid_signers)
    rows_removed = int(frame["signer_id"].isin(invalid_signer_set).sum())
    frame = frame.loc[~frame["signer_id"].isin(invalid_signer_set)].copy()
    report["rows"]["after_signer_filter"] = count_rows(frame)
    report["signers_removed"] = sorted(map(str, invalid_signers))
    report["signer_rows_removed"] = rows_removed

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for _, row in tqdm(frame.iterrows(), total=count_rows(frame), desc="Copying images", unit="image"):
        relative = Path(str(row["image_path"]))
        source = dataset_root / relative
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1

    frame.to_csv(output_root / "manifest_clean.csv", index=False)

    report["final"] = {
        "rows": count_rows(frame),
        "splits": split_counts(frame),
        "datasets": dataset_counts(frame),
        "classes": class_counts(frame),
        "signers_per_split": signers_per_split(frame),
        "copied_files": copied,
    }

    assert_signer_disjoint(frame)
    assert_no_duplicate_hashes(output_root, frame)
    assert_no_thai(frame)
    assert_signer_counts(frame)
    assert_manifest_files(output_root, frame)

    return report


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def split_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame["split"].value_counts().sort_index().items()}


def dataset_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame["dataset"].value_counts().sort_index().items()
    }


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    values = parse_bool(frame["is_forgery"])
    return {
        "genuine": int((~values).sum()),
        "forgery": int(values.sum()),
    }


def signers_per_split(frame: pd.DataFrame) -> dict[str, int]:
    result: dict[str, int] = {}
    for split in sorted(frame["split"].dropna().unique()):
        result[str(split)] = int(frame.loc[frame["split"] == split, "signer_id"].nunique())
    return result


def assert_signer_disjoint(frame: pd.DataFrame) -> None:
    split_map = (
        frame.groupby("signer_id")["split"].nunique()
    )
    offenders = split_map.loc[split_map > 1].index.tolist()
    assert not offenders, f"signers appear in multiple splits: {offenders}"


def assert_no_duplicate_hashes(root: Path, frame: pd.DataFrame) -> None:
    seen: set[str] = set()
    for relative in tqdm(frame["image_path"], desc="Checking duplicate hashes", unit="image"):
        path = root / str(relative)
        digest, error = image_hash(path)
        assert error is None, f"failed to read {path}: {error}"
        assert digest not in seen, f"duplicate pixel hash at {relative}"
        seen.add(digest)


def assert_no_thai(frame: pd.DataFrame) -> None:
    assert not frame["dataset"].eq("Thai").any(), "Thai rows remain"


def assert_signer_counts(frame: pd.DataFrame) -> None:
    forgery = parse_bool(frame["is_forgery"])
    for signer_id, group in frame.groupby("signer_id"):
        group_forgery = parse_bool(group["is_forgery"])
        genuine_count = int((~group_forgery).sum())
        forgery_count = int(group_forgery.sum())
        assert genuine_count >= 5, f"{signer_id} has only {genuine_count} genuine images"
        assert forgery_count >= 1, f"{signer_id} has no forgery images"


def assert_manifest_files(root: Path, frame: pd.DataFrame) -> None:
    missing = [
        str(relative)
        for relative in frame["image_path"]
        if not (root / str(relative)).exists()
    ]
    assert not missing, f"missing manifest files: {missing[:10]}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("training_ready_256x512"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("training_clean"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("cleaning_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    report = clean_dataset(dataset_root, output_root)
    args.report.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    final = report["final"]
    assert isinstance(final, dict)
    print(
        "Cleaning summary:",
        f"before={report['rows']['before']}",
        f"after={final['rows']}",
        f"signers_removed={len(report['signers_removed'])}",
        f"splits={final['splits']}",
        f"datasets={final['datasets']}",
        f"classes={final['classes']}",
    )


if __name__ == "__main__":
    main()

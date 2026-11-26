"""Validate manifest images against the training-ready grayscale image contract."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


def _file_result(
    row: dict[str, str],
    image_root: Path,
    image_height: int,
    image_width: int,
) -> dict[str, Any]:
    relative_path = (row.get("image_path") or "").strip()
    path = Path(relative_path).expanduser()
    if not path.is_absolute():
        path = image_root / path

    result: dict[str, Any] = {
        "image_path": relative_path,
        "dataset": (row.get("dataset") or "").strip(),
        "split": (row.get("split") or "").strip(),
        "exists": path.is_file(),
        "opened": False,
        "mode": None,
        "shape": None,
        "dtype": None,
        "pixel_min": None,
        "pixel_max": None,
        "pure_white_percent": None,
        "unique_pixel_values": None,
        "failures": [],
    }
    if not result["exists"]:
        result["failures"].append("file does not exist")
        return result

    try:
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image)
            result["opened"] = True
            result["mode"] = image.mode
    except (OSError, ValueError) as error:
        result["failures"].append(f"could not open image: {error}")
        return result

    result["shape"] = list(array.shape)
    result["dtype"] = str(array.dtype)
    if array.dtype != np.uint8:
        result["failures"].append(f"dtype is {array.dtype}, expected uint8")
    if array.shape != (image_height, image_width):
        result["failures"].append(
            f"shape is {array.shape}, expected {(image_height, image_width)}"
        )
    if result["mode"] != "L" or array.ndim != 2:
        result["failures"].append(
            f"image is not single-channel grayscale (mode={result['mode']}, "
            f"dimensions={array.ndim})"
        )
    if array.size:
        result["pixel_min"] = int(array.min())
        result["pixel_max"] = int(array.max())
        result["unique_pixel_values"] = int(np.unique(array).size)
        if array.ndim == 2:
            result["pure_white_percent"] = float(
                np.count_nonzero(array == 255) * 100.0 / array.size
            )
    return result


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    unique_counts = [
        item["unique_pixel_values"]
        for item in results
        if item["unique_pixel_values"] is not None
    ]
    white_percentages = [
        item["pure_white_percent"]
        for item in results
        if item["pure_white_percent"] is not None
    ]
    minima = [item["pixel_min"] for item in results if item["pixel_min"] is not None]
    maxima = [item["pixel_max"] for item in results if item["pixel_max"] is not None]
    return {
        "count": len(results),
        "failures": sum(bool(item["failures"]) for item in results),
        "mean_unique_pixel_values": (
            statistics.fmean(unique_counts) if unique_counts else None
        ),
        "median_unique_pixel_values": (
            statistics.median(unique_counts) if unique_counts else None
        ),
        "min_unique_pixel_values": min(unique_counts) if unique_counts else None,
        "max_unique_pixel_values": max(unique_counts) if unique_counts else None,
        "mean_pure_white_percent": (
            statistics.fmean(white_percentages) if white_percentages else None
        ),
        "pixel_min": min(minima) if minima else None,
        "pixel_max": max(maxima) if maxima else None,
    }


def validate_manifest(
    manifest_path: str | Path,
    image_root: str | Path | None = None,
    *,
    image_height: int = 256,
    image_width: int = 512,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Inspect every manifest row and return per-file results and grouped summaries."""
    manifest = Path(manifest_path).expanduser().resolve()
    root = Path(image_root).expanduser().resolve() if image_root else manifest.parent
    if image_height < 1 or image_width < 1:
        raise ValueError("image_height and image_width must be positive")

    with manifest.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"image_path", "dataset", "split"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"manifest is missing required columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)

    iterator = tqdm(rows, desc="Validating images", unit="image", disable=not show_progress)
    files = [_file_result(row, root, image_height, image_width) for row in iterator]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in files:
        by_dataset[item["dataset"] or "(missing)"].append(item)
        by_split[item["split"] or "(missing)"].append(item)

    failures = [item for item in files if item["failures"]]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "image_root": str(root),
        "expected_shape": [image_height, image_width],
        "total_rows": len(files),
        "failure_count": len(failures),
        "dataset_summary": {
            name: _summarize(items) for name, items in sorted(by_dataset.items())
        },
        "split_summary": {
            name: _summarize(items) for name, items in sorted(by_split.items())
        },
        "failing_files": [
            {
                "image_path": item["image_path"],
                "dataset": item["dataset"],
                "split": item["split"],
                "failures": item["failures"],
            }
            for item in failures
        ],
        "files": files,
    }


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _print_table(title: str, summary: dict[str, dict[str, Any]]) -> None:
    headers = (
        "group",
        "count",
        "failures",
        "unique mean/min/max",
        "unique median",
        "mean white %",
        "pixel range",
    )
    rows = []
    for group, values in summary.items():
        rows.append(
            [
                group,
                _format(values["count"]),
                _format(values["failures"]),
                "/".join(
                    _format(values[key])
                    for key in (
                        "mean_unique_pixel_values",
                        "min_unique_pixel_values",
                        "max_unique_pixel_values",
                    )
                ),
                _format(values["median_unique_pixel_values"]),
                _format(values["mean_pure_white_percent"]),
                f"{_format(values['pixel_min'])}-{_format(values['pixel_max'])}",
            ]
        )
    widths = [
        max([len(headers[index]), *(len(row[index]) for row in rows)])
        for index in range(len(headers))
    ]
    print(title)
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training_clean/manifest_clean.csv"),
    )
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=512)
    args = parser.parse_args()

    report = validate_manifest(
        args.manifest,
        args.image_root,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    report_path = args.report or args.manifest.parent / "validate_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Manifest rows: {report['total_rows']:,}; failures: {report['failure_count']:,}")
    _print_table("Per-dataset summary", report["dataset_summary"])
    _print_table("Per-split summary", report["split_summary"])
    if report["failure_count"]:
        print("Failing files (up to 20):")
        for item in report["failing_files"][:20]:
            print(f"  {item['image_path']}: {'; '.join(item['failures'])}")
    print(f"Report: {report_path.resolve()}")
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

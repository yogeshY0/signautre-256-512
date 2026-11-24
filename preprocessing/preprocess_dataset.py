"""One-time preprocessing: raw manifest -> 192x384 black-and-white PNGs + processed manifest
(pass --grayscale to keep grayscale strokes).

Usage:
    python -m preprocessing.preprocess_dataset --manifest data/manifest.csv --output-dir data/processed

Output layout mirrors the raw layout (``raw/cedar/...`` -> ``processed/cedar/...``) and
``<output-dir>/manifest.csv`` is written with paths relative to itself. Train with:
    python -m training.train_siamese --manifest data/processed/manifest.csv --already-preprocessed
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from preprocessing.signature_preprocessing import preprocess_signature
from preprocessing.v2.pipeline import (
    PRODUCTION_V2_DEFAULT_SIZE,
    preprocess_complete,
)


def parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must use WIDTHxHEIGHT, for example 384x192")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--image-root", type=Path, help="Defaults to the manifest's directory.")
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Keep grayscale strokes instead of the default black-and-white output.",
    )
    parser.add_argument(
        "--mode",
        choices=("legacy", "production-v2"),
        default="legacy",
        help="Legacy preserves current behavior; production-v2 uses the versioned pipeline.",
    )
    parser.add_argument(
        "--size",
        type=parse_size,
        default=None,
        help="Output WIDTHxHEIGHT. Legacy default is 384x192; production-v2 default is 512x256.",
    )
    parser.add_argument(
        "--skeleton",
        action="store_true",
        help="Production-v2: additionally write 0/255 skeleton masks.",
    )
    parser.add_argument(
        "--distance-map",
        action="store_true",
        help="Production-v2: additionally write normalized distance maps.",
    )
    parser.add_argument(
        "--model-input",
        action="store_true",
        help="Production-v2: additionally save ImageNet-ready float32 model tensors as .npy.",
    )
    args = parser.parse_args()
    if args.size is None:
        args.size = (
            (384, 192)
            if args.mode == "legacy"
            else PRODUCTION_V2_DEFAULT_SIZE
        )

    root = (args.image_root or args.manifest.parent).resolve()
    out_dir = args.output_dir.resolve()
    if out_dir.exists():
        protected = (
            (out_dir / "manifest.csv").exists()
            or (out_dir / "manifests" / "manifest_v2.csv").exists()
            or (out_dir / "gray").exists()
            or (out_dir / "mask").exists()
        )
        if protected:
            raise ValueError(
                f"refusing to overwrite existing processed output: {out_dir}"
            )

    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    out_rows: list[dict[str, str]] = []
    failures: list[str] = []
    metadata_records: list[dict[str, object]] = []
    quality_reports: dict[str, dict[str, float | int]] = {}
    quality_fields = [
        "preprocessing_version",
        "selected_polarity",
        "polarity_confidence",
        "ink_ratio",
        "bbox_width",
        "bbox_height",
        "aspect_ratio",
        "component_count",
        "border_contact",
        "crop_clipped",
        "crop_clipped_legacy",
        "contrast_score",
        "blur_score",
        "artifact_detected",
        "artifact_uncertain",
        "quality_status",
        "ink_removed_fraction",
        "border_removed_fraction",
        "deskew_applied",
        "deskew_angle",
        "mask_name",
    ]
    for i, row in enumerate(rows, 1):
        source = Path(row["image_path"])
        source = source if source.is_absolute() else root / source
        manifest_relative = Path(row["image_path"])
        if manifest_relative.is_absolute():
            try:
                relative = source.relative_to(root)
            except ValueError:
                relative = Path(source.name)
        else:
            # Drop the leading "raw/" so processed/cedar mirrors raw/cedar.
            parts = manifest_relative.parts
            relative = Path(*parts[1:]) if parts and parts[0] == "raw" else manifest_relative
        target = (out_dir / relative).with_suffix(".png")

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            failures.append(f"{source}: unreadable")
            continue
        try:
            if args.mode == "legacy":
                processed = preprocess_signature(
                    image,
                    target_size=args.size,
                    binarize=not args.grayscale,
                )
                metadata = None
                mask = None
                artifact = None
            else:
                result = preprocess_complete(
                    image,
                    size=args.size,
                    include_skeleton=args.skeleton,
                    include_distance_map=args.distance_map,
                    include_model_input=args.model_input,
                )
                processed = result.canonical_grayscale
                mask = result.binary_mask
                artifact = result.artifact_mask
                metadata = result.metadata
        except ValueError as error:
            failures.append(f"{source}: {error}")
            if args.mode == "production-v2":
                dataset = row.get("dataset", "unknown")
                report = quality_reports.setdefault(
                    dataset,
                    {
                        "ok": 0,
                        "flagged": 0,
                        "rejected": 0,
                        "artifact_fired": 0,
                        "ink_removed_fraction_sum": 0.0,
                        "processed": 0,
                        "crop_clipped_old": 0,
                        "crop_clipped_new": 0,
                        "crop_clipped_status_changed": 0,
                    },
                )
                report["rejected"] = int(report["rejected"]) + 1
            continue

        if args.mode == "legacy":
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), processed):
                failures.append(f"{source}: failed to write {target}")
                continue
            out_rows.append({**row, "image_path": target.relative_to(out_dir).as_posix()})
        else:
            assert metadata is not None and mask is not None and artifact is not None
            gray_target = (out_dir / "gray" / relative).with_suffix(".png")
            mask_target = (out_dir / "mask" / relative).with_suffix(".png")
            artifact_target = (out_dir / "artifacts" / relative).with_suffix(".png")
            gray_target.parent.mkdir(parents=True, exist_ok=True)
            mask_target.parent.mkdir(parents=True, exist_ok=True)
            artifact_target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(gray_target), processed):
                failures.append(f"{source}: failed to write {gray_target}")
                continue
            cv2.imwrite(str(mask_target), mask)
            cv2.imwrite(str(artifact_target), artifact)

            if args.skeleton and result.skeleton_mask is not None:
                skeleton_target = (out_dir / "skeleton" / relative).with_suffix(".png")
                skeleton_target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(skeleton_target), result.skeleton_mask)
            if args.distance_map and result.distance_map is not None:
                distance_target = (out_dir / "distance_map" / relative).with_suffix(".png")
                distance_target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(distance_target), result.distance_map)
            if args.model_input and result.model_input is not None:
                model_target = (out_dir / "model_input" / relative).with_suffix(".npy")
                model_target.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(model_target), result.model_input)

            out_row = {**row, "image_path": gray_target.relative_to(out_dir).as_posix()}
            for field in quality_fields:
                out_row[field] = str(metadata[field])
            out_rows.append(out_row)
            metadata_records.append(
                {
                    "image_path": gray_target.relative_to(out_dir).as_posix(),
                    "source_image_path": row["image_path"],
                    **metadata,
                }
            )

            dataset = row.get("dataset", "unknown")
            report = quality_reports.setdefault(
                dataset,
                {
                    "ok": 0,
                    "flagged": 0,
                    "rejected": 0,
                    "artifact_fired": 0,
                    "ink_removed_fraction_sum": 0.0,
                    "processed": 0,
                    "crop_clipped_old": 0,
                    "crop_clipped_new": 0,
                    "crop_clipped_status_changed": 0,
                },
            )
            status = str(metadata["quality_status"])
            report[status] = int(report.get(status, 0)) + 1
            report["processed"] = int(report["processed"]) + 1
            report["ink_removed_fraction_sum"] = float(
                report["ink_removed_fraction_sum"]
                + float(metadata["ink_removed_fraction"])
            )
            if metadata["artifact_detected"]:
                report["artifact_fired"] = int(report["artifact_fired"]) + 1
            legacy_clipped = bool(metadata.get("crop_clipped_legacy", False))
            new_clipped = bool(metadata["crop_clipped"])
            if legacy_clipped:
                report["crop_clipped_old"] = int(report["crop_clipped_old"]) + 1
            if new_clipped:
                report["crop_clipped_new"] = int(report["crop_clipped_new"]) + 1
            if legacy_clipped != new_clipped:
                report["crop_clipped_status_changed"] = int(
                    report["crop_clipped_status_changed"]
                ) + 1
        if i % 200 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} processed")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "production-v2":
        manifest_dir = out_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "manifest_v2.csv"
        fieldnames = list(rows[0].keys()) + [
            field for field in quality_fields if field not in rows[0]
        ]
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

        metadata_path = out_dir / "metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8") as handle:
            for record in metadata_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        report_output: dict[str, dict[str, object]] = {}
        for dataset, report in quality_reports.items():
            processed = int(report["processed"])
            mean_removed = (
                float(report["ink_removed_fraction_sum"]) / processed
                if processed > 0
                else 0.0
            )
            report_output[dataset] = {
                key: value
                for key, value in report.items()
                if key != "ink_removed_fraction_sum"
            }
            report_output[dataset]["mean_ink_removed_fraction"] = mean_removed

        quality_path = out_dir / "quality_report.json"
        quality_path.write_text(
            json.dumps(
                {
                    "mode": "production-v2",
                    "size": list(args.size),
                    "datasets": report_output,
                    "processed": len(out_rows),
                    "skipped": len(failures),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    else:
        manifest_path = out_dir / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)

    print(f"wrote {len(out_rows)} images")
    print(f"manifest: {manifest_path}")
    if failures:
        print(f"{len(failures)} skipped:")
        for message in failures:
            print(f"  {message}")


if __name__ == "__main__":
    main()

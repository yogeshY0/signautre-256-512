"""Single-call production-v2 signature preprocessing API and CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from preprocessing.v2.params import Params
    from preprocessing.v2.pipeline import apply_preprocessing
    from preprocessing.v2.result import PreprocessingResult
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocessing.v2.params import Params
    from preprocessing.v2.pipeline import apply_preprocessing
    from preprocessing.v2.result import PreprocessingResult

VERSION = "production-v2.1-single-file"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class SinglePreprocessResult:
    """Simple API matching the production preprocessing contract."""

    gray: np.ndarray
    mask: np.ndarray
    artifact_mask: np.ndarray
    metadata: dict[str, object]
    skeleton: np.ndarray | None = None
    distance_map: np.ndarray | None = None
    model_tensor: np.ndarray | None = None

    def model_input(self) -> np.ndarray:
        if self.model_tensor is not None:
            return self.model_tensor
        image = self.gray.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return (np.repeat(image[None, ...], 3, axis=0) - mean) / std


def load_gray(source: str | os.PathLike[str] | np.ndarray) -> np.ndarray:
    """Load a path or NumPy array and return a uint8 grayscale image."""

    if isinstance(source, (str, os.PathLike)):
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"unreadable image: {source}")
    else:
        image = np.asarray(source)

    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("invalid image dimensions")

    if image.dtype == np.uint16:
        image = np.rint(image.astype(np.float64) / 257.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"unsupported dtype {image.dtype}; expected uint8")

    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32) * alpha
        rgb += 255.0 * (1.0 - alpha)
        return cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("invalid image channel count")


def _source_artifact_mask(
    artifact_mask: np.ndarray,
    source_shape: tuple[int, int],
) -> np.ndarray:
    """Resize the processed artifact mask back to source dimensions for overlays."""

    source_h, source_w = source_shape
    return cv2.resize(
        artifact_mask,
        (source_w, source_h),
        interpolation=cv2.INTER_NEAREST,
    )


def preprocess_image(
    source: str | os.PathLike[str] | np.ndarray,
    size: tuple[int, int] = (512, 256),
    params: Params | None = None,
    text_mask: str | os.PathLike[str] | np.ndarray | None = None,
) -> SinglePreprocessResult:
    """One call: path or array -> complete canonical signature result."""

    gray = load_gray(source)
    text_mask_array: np.ndarray | None = None
    if text_mask is not None:
        text_mask_array = load_gray(text_mask)
        text_mask_array = np.where(text_mask_array < 128, 0, 255).astype(np.uint8)
        if text_mask_array.shape != gray.shape:
            raise ValueError("text_mask must match the source image dimensions")
    result: PreprocessingResult = apply_preprocessing(
        gray,
        size=size,
        params=params,
        include_skeleton=True,
        include_distance_map=True,
        include_model_input=True,
        text_mask=text_mask_array,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "preprocessing_version": VERSION,
            "input_width": int(gray.shape[1]),
            "input_height": int(gray.shape[0]),
            "output_width": int(size[0]),
            "output_height": int(size[1]),
        }
    )
    source_artifact = (
        result.source_artifact_mask
        if result.source_artifact_mask is not None
        else _source_artifact_mask(result.artifact_mask, gray.shape)
    )
    return SinglePreprocessResult(
        gray=result.canonical_grayscale,
        mask=result.binary_mask,
        artifact_mask=source_artifact,
        metadata=metadata,
        skeleton=result.skeleton_mask,
        distance_map=result.distance_map,
        model_tensor=result.model_input,
    )


def parse_size(text: str) -> tuple[int, int]:
    width, height = text.lower().split("x")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return width, height


def _debug_overlay(source: str | os.PathLike[str], result: SinglePreprocessResult) -> np.ndarray:
    source_gray = load_gray(source)
    overlay = cv2.cvtColor(source_gray, cv2.COLOR_GRAY2BGR)
    overlay[result.artifact_mask == 255] = (0, 0, 255)
    overlay[result.artifact_mask == 128] = (0, 200, 255)
    return overlay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single-step production-v2 signature preprocessing")
    parser.add_argument("input", help="Image file or folder")
    parser.add_argument("--out", required=True, help="New output folder")
    parser.add_argument("--size", default="512x256", help="WIDTHxHEIGHT")
    parser.add_argument("--debug", action="store_true", help="Write source-frame artifact overlays")
    parser.add_argument(
        "--text-mask",
        help="Optional source-frame 0/255 text-mask image; text pixels are marked/removed only with explicit evidence",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"Refusing to write into non-empty folder: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    size = parse_size(args.size)
    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        print(f"Input does not exist: {input_path}", file=sys.stderr)
        return 2

    for subdir in ("gray", "mask", "artifacts", "skeleton", "distance_map", "model_input"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)
    if args.debug:
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "version": VERSION,
        "size": list(size),
        "processed": 0,
        "ok": 0,
        "flagged": 0,
        "rejected": [],
        "warnings_count": {},
    }

    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as metadata_file:
        for source in files:
            stem = source.stem
            try:
                result = preprocess_image(
                    source,
                    size=size,
                    text_mask=args.text_mask,
                )
            except ValueError as error:
                rejected = report["rejected"]
                assert isinstance(rejected, list)
                rejected.append({"file": str(source), "reason": str(error)})
                print(f"REJECTED {stem}: {error}")
                continue

            cv2.imwrite(str(out_dir / "gray" / f"{stem}.png"), result.gray)
            cv2.imwrite(str(out_dir / "mask" / f"{stem}.png"), result.mask)
            cv2.imwrite(str(out_dir / "artifacts" / f"{stem}.png"), result.artifact_mask)
            if result.skeleton is not None:
                cv2.imwrite(str(out_dir / "skeleton" / f"{stem}.png"), result.skeleton)
            if result.distance_map is not None:
                cv2.imwrite(str(out_dir / "distance_map" / f"{stem}.png"), result.distance_map)
            if result.model_tensor is not None:
                np.save(str(out_dir / "model_input" / f"{stem}.npy"), result.model_tensor)
            if args.debug:
                cv2.imwrite(str(out_dir / "debug" / f"{stem}.png"), _debug_overlay(source, result))

            metadata_file.write(json.dumps({"file": str(source), **result.metadata}, sort_keys=True) + "\n")
            report["processed"] = int(report["processed"]) + 1
            status = str(result.metadata["quality_status"])
            report[status] = int(report.get(status, 0)) + 1
            warning_counts = report["warnings_count"]
            assert isinstance(warning_counts, dict)
            for warning in result.metadata["warnings"]:
                warning_counts[warning] = int(warning_counts.get(warning, 0)) + 1
            print(f"{status.upper():8s} {stem}")

    (out_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    rejected_count = len(report["rejected"]) if isinstance(report["rejected"], list) else 0
    print(f"done: {report['processed']} processed, {rejected_count} rejected -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

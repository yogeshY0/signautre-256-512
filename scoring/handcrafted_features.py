"""Handcrafted geometric and structural similarity features for signature pairs.

This module is deliberately narrow. It extracts the nine scalar metrics requested for
score-level fusion and nothing else: no deep embeddings, no unaligned DTW sequences,
and no raw high-dimensional histograms. The fused handcrafted vector therefore stays
low-dimensional and each value can be explained to a reviewer.

Two functions do the work:

* :func:`describe_image` summarises one signature once into a
  :class:`SignatureDescriptors`. Summaries are cheap to cache and are reused for every
  pair the image takes part in.
* :func:`pair_features` turns two summaries into nine scalars.

All nine values are bounded to ``[0, 1]``. Eight are similarities where a larger value
means "more alike"; ``x_slant_diff`` is a normalised *distance* where a smaller value
means "more alike". :func:`hand_similarity_vector` flips that one sign so the mean of
the vector is meaningful as a single handcrafted score.

Images must be the same kind of input the CNN consumes: the ``uint8`` grayscale output
of :func:`preprocessing.signature_preprocessing.preprocess_signature`, or an already
preprocessed ``192 x 384`` grayscale file. Use :func:`describe_records` to read
manifest rows through the same loader the training dataset uses, which is what keeps
handcrafted features and deep embeddings free of train/serve skew.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import cv2
import numpy as np

from preprocessing.signature_preprocessing import load_preprocessed_image

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime import cycle
    from training.stage3_metric_learning import SignatureRecord

# Number of bins used for the row and column projection profiles.
PROFILE_BINS = 128
SUPPORTED_IMAGE_HEIGHT = 192
SUPPORTED_IMAGE_WIDTH = 384

# A pixel is treated as ink when it is darker than this value. The Stage 2.2 output is
# dark ink on white, so a fixed cut is safe and avoids re-estimating Otsu per image.
INK_THRESHOLD = 200

# Features where a *larger* value means "more similar".
SIMILARITY_FEATURES = (
    "sim_ink_aspect_similarity",
    "sim_stroke_width_similarity",
    "x_endpoints_ratio",
    "x_junctions_ratio",
    "sim_density_similarity",
    "h_proj",
    "v_proj",
    "sim_hu_similarity",
)

# Features where a *smaller* value means "more similar".
DISTANCE_FEATURES = ("x_slant_diff",)

# Declared order of the fused handcrafted vector. Fixed so saved fusion coefficients
# always line up with the columns produced here.
FEATURE_NAMES = SIMILARITY_FEATURES[:4] + DISTANCE_FEATURES + SIMILARITY_FEATURES[4:]


def require_supported_image_size(
    image_height: int | None,
    image_width: int | None,
) -> None:
    """Reject fusion features for image dimensions they have not been validated on."""
    resolved_height = SUPPORTED_IMAGE_HEIGHT if image_height is None else image_height
    resolved_width = SUPPORTED_IMAGE_WIDTH if image_width is None else image_width
    if (resolved_height, resolved_width) != (
        SUPPORTED_IMAGE_HEIGHT,
        SUPPORTED_IMAGE_WIDTH,
    ):
        raise ValueError(
            "handcrafted fusion features support only HxW "
            f"{SUPPORTED_IMAGE_HEIGHT}x{SUPPORTED_IMAGE_WIDTH}; got "
            f"{resolved_height}x{resolved_width}"
        )


def _as_gray(image: np.ndarray) -> np.ndarray:
    """Return a 2D uint8 grayscale view, mirroring the preprocessing contract."""
    if image is None or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty 2D or 3D array")
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[..., 0]
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("image must have 1 or 3 channels")


def binarize_ink(gray: np.ndarray, threshold: int | None = None) -> np.ndarray:
    """Return a boolean ink mask where ``True`` marks a dark pixel.

    ``threshold`` is a fixed intensity cut. When omitted, an already-binary image (the
    Stage 2.2 output, whose only values are 0 and 255) uses :data:`INK_THRESHOLD` and
    anything else uses Otsu's method. An image without ink raises ``ValueError`` so a
    blank crop is routed to review instead of being scored.
    """
    grayscale = _as_gray(gray)
    if threshold is not None:
        ink = grayscale < threshold
    else:
        unique = np.unique(grayscale)
        if unique.size <= 2 and np.isin(unique, (0, 255)).all():
            ink = grayscale < INK_THRESHOLD
        else:
            _, binary = cv2.threshold(
                grayscale, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
            ink = binary > 0
    if not ink.any():
        raise ValueError("image contains no ink; nothing to describe")
    return ink


def _zhang_suen_thinning(mask: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """Thin a binary mask to a one-pixel skeleton (Zhang-Suen, vectorised).

    ``cv2.ximgproc.thinning`` is used when the OpenCV contrib build is installed;
    otherwise this NumPy implementation runs, so the module works with the
    ``opencv-python`` wheel that ``requirements.txt`` already pins. No new dependency
    is introduced.
    """
    thinning = getattr(getattr(cv2, "ximgproc", None), "thinning", None)
    if thinning is not None:  # pragma: no cover - only when opencv-contrib is installed
        thinned = thinning(
            mask.astype(np.uint8) * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
        return thinned > 0

    image = mask.astype(np.uint8)
    for _ in range(max_iterations):
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1, mode="constant")
            # P2..P9 in clockwise order around the centre pixel.
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            # A: number of 0 -> 1 transitions in the ring P2..P9, P2.
            ring = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            transitions = np.zeros_like(image, dtype=np.uint8)
            for index in range(8):
                transitions += ((ring[index] == 0) & (ring[index + 1] == 1)).astype(np.uint8)
            if step == 0:
                corners = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                corners = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (
                (image == 1)
                & (neighbours >= 2)
                & (neighbours <= 6)
                & (transitions == 1)
                & corners
            )
            if remove.any():
                image[remove] = 0
                changed = True
        if not changed:
            break
    return image > 0


def _skeleton_topology(skeleton: np.ndarray) -> tuple[int, int, int]:
    """Return ``(endpoints, junctions, skeleton_length)`` for a thinned mask.

    Endpoints have exactly one 8-neighbour and junctions have three or more. Counts are
    returned raw; :func:`describe_image` divides them by the skeleton length so the
    resulting densities do not depend on signature size.
    """
    if not skeleton.any():
        return 0, 0, 0
    binary = skeleton.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbours = cv2.filter2D(binary, -1, kernel, borderType=cv2.BORDER_CONSTANT) - binary
    endpoints = int(((neighbours == 1) & skeleton).sum())
    junctions = int(((neighbours >= 3) & skeleton).sum())
    return endpoints, junctions, int(binary.sum())


def _wrap_angle(angle: float) -> float:
    """Wrap an angle in radians into ``[-pi/2, pi/2)``."""
    return float((angle + np.pi / 2.0) % np.pi - np.pi / 2.0)


def _dominant_slant(gray: np.ndarray, ink: np.ndarray) -> float:
    """Return the dominant stroke orientation in radians, in ``[-pi/2, pi/2)``.

    A doubled-angle circular mean over Sobel gradients, restricted to a dilated ink
    region so paper texture and background gradients do not vote. The gradient is
    perpendicular to the stroke, so the mean is rotated by 90 degrees to describe the
    stroke itself. Only differences between images are used by the fusion features, so
    the absolute convention does not matter.
    """
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    region = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    magnitude = np.hypot(gx, gy)
    if not region.any() or float(magnitude[region].sum()) <= 0.0:
        return 0.0
    doubled = 2.0 * np.arctan2(gy, gx)
    sin_sum = float((magnitude[region] * np.sin(doubled[region])).sum())
    cos_sum = float((magnitude[region] * np.cos(doubled[region])).sum())
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return 0.0
    return _wrap_angle(0.5 * np.arctan2(sin_sum, cos_sum) + np.pi / 2.0)

def _resample_profile(profile: np.ndarray, bins: int) -> np.ndarray:
    """Interpolate a projection profile to ``bins`` values and L1-normalise it."""
    total = float(profile.sum())
    if total <= 0.0:
        return np.zeros(bins, dtype=np.float32)
    positions = np.linspace(0.0, profile.size - 1, bins)
    resampled = np.interp(positions, np.arange(profile.size), profile)
    resampled = np.clip(resampled, 0.0, None)
    return (resampled / resampled.sum()).astype(np.float32)


def _projection_profiles(
    ink: np.ndarray, bbox: tuple[int, int, int, int], bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return L1-normalised row and column ink profiles over the ink bounding box.

    ``horizontal`` counts ink per row (the horizontal projection of the strokes) and
    ``vertical`` counts ink per column. Both are length-normalised so two signatures of
    different size stay comparable.
    """
    x, y, width, height = bbox
    region = ink[y : y + height, x : x + width].astype(np.float32)
    horizontal = region.sum(axis=1)
    vertical = region.sum(axis=0)
    return _resample_profile(horizontal, bins), _resample_profile(vertical, bins)


def _hu_moments(ink: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Return the seven log-scaled Hu invariant moments of the ink bounding box."""
    x, y, width, height = bbox
    region = ink[y : y + height, x : x + width].astype(np.uint8)
    moments = cv2.moments(region, binaryImage=True)
    hu = cv2.HuMoments(moments).ravel()
    return np.sign(hu) * np.log10(np.abs(hu) + 1e-12)


@dataclass(frozen=True)
class SignatureDescriptors:
    """Compact, cacheable summary of a single signature image.

    ``ink_mask`` and ``skeleton`` are only populated when ``keep_masks=True`` because a
    full mask pair costs roughly 150 kB per image. The default keeps only the scalars
    and profiles, which is all the pair features need.
    """

    bbox: tuple[int, int, int, int]
    aspect: float
    stroke_width: float
    density: float
    endpoint_density: float
    junction_density: float
    slant: float
    ink_area: int
    skeleton_length: int
    h_profile: np.ndarray
    v_profile: np.ndarray
    hu: np.ndarray
    ink_mask: np.ndarray | None = None
    skeleton: np.ndarray | None = None


def describe_image(
    image: np.ndarray,
    *,
    threshold: int | None = None,
    keep_masks: bool = False,
    profile_bins: int = PROFILE_BINS,
) -> SignatureDescriptors:
    """Summarise one signature image into :class:`SignatureDescriptors`.

    Args:
        image: Grayscale or BGR ``uint8`` image, dark ink on a light background.
        threshold: Fixed ink cut. ``None`` selects the threshold automatically.
        keep_masks: Also retain the boolean ink mask and skeleton (memory heavy).
        profile_bins: Number of bins for the projection profiles.

    Raises:
        ValueError: If the image is invalid or contains no ink.
    """
    gray = _as_gray(image)
    require_supported_image_size(gray.shape[0], gray.shape[1])
    ink = binarize_ink(gray, threshold=threshold)
    skeleton = _zhang_suen_thinning(ink)

    x, y, width, height = cv2.boundingRect(ink.astype(np.uint8))
    ink_area = int(ink.sum())
    skeleton_length = int(skeleton.sum())
    endpoints, junctions, _ = _skeleton_topology(skeleton)
    h_profile, v_profile = _projection_profiles(ink, (x, y, width, height), profile_bins)

    # Densities are per skeleton pixel, so they do not grow with signature size.
    scale = max(skeleton_length, 1)
    return SignatureDescriptors(
        bbox=(x, y, width, height),
        aspect=width / max(height, 1),
        stroke_width=ink_area / scale,
        density=ink_area / max(width * height, 1),
        endpoint_density=endpoints / scale,
        junction_density=junctions / scale,
        slant=_dominant_slant(gray, ink),
        ink_area=ink_area,
        skeleton_length=skeleton_length,
        h_profile=h_profile,
        v_profile=v_profile,
        hu=_hu_moments(ink, (x, y, width, height)),
        ink_mask=ink if keep_masks else None,
        skeleton=skeleton if keep_masks else None,
    )


def _ratio(first: float, second: float) -> float:
    """Symmetric ratio in ``[0, 1]`` that equals 1 when both values match."""
    left, right = abs(float(first)), abs(float(second))
    largest = max(left, right)
    if largest <= 0.0:
        return 1.0
    return min(left, right) / largest


def _profile_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """``1 - 0.5 * L1`` between two L1-normalised profiles, bounded to ``[0, 1]``."""
    if first.shape != second.shape:
        raise ValueError(f"profile shapes differ: {first.shape} vs {second.shape}")
    return float(np.clip(1.0 - 0.5 * np.abs(first - second).sum(), 0.0, 1.0))


def _slant_difference(first: float, second: float) -> float:
    """Normalised angular difference in ``[0, 1]``; orientations are modulo pi."""
    delta = abs(_wrap_angle(first - second))
    delta = min(delta, np.pi - delta)
    return float(np.clip(delta / (np.pi / 2.0), 0.0, 1.0))


def _hu_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """``1 / (1 + mean |delta log-Hu|)``: 1 when the moments match, smaller otherwise."""
    return float(1.0 / (1.0 + float(np.abs(first - second).mean())))


def pair_features(
    reference: SignatureDescriptors, query: SignatureDescriptors
) -> dict[str, float]:
    """Return the nine handcrafted scalar features for one ``(reference, query)`` pair.

    Keys follow :data:`FEATURE_NAMES`. Eight values are similarities in ``[0, 1]`` and
    ``x_slant_diff`` is a normalised distance in ``[0, 1]``. The function is symmetric:
    swapping the arguments changes no value.
    """
    return {
        "sim_ink_aspect_similarity": _ratio(reference.aspect, query.aspect),
        "sim_stroke_width_similarity": _ratio(reference.stroke_width, query.stroke_width),
        "x_endpoints_ratio": _ratio(reference.endpoint_density, query.endpoint_density),
        "x_junctions_ratio": _ratio(reference.junction_density, query.junction_density),
        "x_slant_diff": _slant_difference(reference.slant, query.slant),
        "sim_density_similarity": _ratio(reference.density, query.density),
        "h_proj": _profile_similarity(reference.h_profile, query.h_profile),
        "v_proj": _profile_similarity(reference.v_profile, query.v_profile),
        "sim_hu_similarity": _hu_similarity(reference.hu, query.hu),
    }


def hand_similarity_vector(
    features: Mapping[str, float], names: Sequence[str] = FEATURE_NAMES
) -> np.ndarray:
    """Return hand features as a "higher is more similar" vector.

    ``x_slant_diff`` is a distance, so it is inverted (``1 - value``) before averaging.
    Without this step ``Mean(H_hand)`` would mix directions and the average would be
    meaningless.
    """
    values = np.empty(len(names), dtype=np.float64)
    for index, name in enumerate(names):
        if name not in features:
            raise KeyError(f"missing handcrafted feature: {name}")
        value = float(features[name])
        values[index] = 1.0 - value if name in DISTANCE_FEATURES else value
    return values


def describe_records(
    records: Sequence[SignatureRecord],
    *,
    already_preprocessed: bool = False,
    ink: Literal["auto", "light", "dark"] = "auto",
    threshold: int | None = None,
    keep_masks: bool = False,
    progress_every: int = 200,
    image_height: int | None = None,
    image_width: int | None = None,
) -> list[SignatureDescriptors]:
    """Describe every manifest record, reading images exactly as the model does.

    ``already_preprocessed``, ``ink``, ``image_height`` and ``image_width`` are forwarded
    to :func:`preprocessing.signature_preprocessing.load_preprocessed_image`, the same
    loader the training dataset uses. Keeping one loading path is what prevents the
    handcrafted features from being computed on different pixels than the CNN sees.

    Args:
        records: Manifest records exposing an ``image_path`` attribute.
        already_preprocessed: Read files as grayscale of the declared size instead of
            running Stage 2.2 preprocessing.
        ink: Retained for preprocessing compatibility; ignored by Stage 2.2.
        threshold: Fixed ink cut forwarded to :func:`describe_image`.
        keep_masks: Retain masks and skeletons in each descriptor.
        progress_every: Print progress every N images. ``0`` disables it.
        image_height: Expected/explicit preprocessed height; ``None`` uses the default.
        image_width: Expected/explicit preprocessed width; ``None`` uses the default.
    """
    require_supported_image_size(image_height, image_width)
    descriptors: list[SignatureDescriptors] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        image = load_preprocessed_image(
            record.image_path,
            already_preprocessed=already_preprocessed,
            ink=ink,
            image_height=image_height,
            image_width=image_width,
        )
        descriptors.append(describe_image(image, threshold=threshold, keep_masks=keep_masks))
        if progress_every and index % progress_every == 0:
            print(f"{index}/{total} signatures described")
    return descriptors

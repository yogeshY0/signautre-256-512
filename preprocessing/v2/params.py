"""Centralized, immutable configuration for production-v2 preprocessing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """All versioned thresholds and limits used by production-v2."""

    # Input validation
    min_image_width: int = 32
    min_image_height: int = 32
    tiny_image_area: int = 1024
    saturated_fraction_limit: float = 0.995
    min_contrast_score: float = 8.0
    min_ink_pixels: int = 5
    min_ink_ratio: float = 0.0002
    max_ink_ratio: float = 0.70

    # Polarity and profile
    border_band: int = 3
    faint_ink_threshold: int = 242
    substantial_component_ratio: float = 0.30

    # Polarity multi-signal weights
    polarity_border_weight: float = 0.25
    polarity_ratio_weight: float = 0.20
    polarity_component_weight: float = 0.15
    polarity_contrast_weight: float = 0.12
    polarity_aspect_weight: float = 0.08
    polarity_stroke_weight: float = 0.10
    polarity_bbox_plausibility_weight: float = 0.10

    # Illumination normalization
    illumination_kernel_divisor: int = 12
    illumination_delta_factor: float = 0.30
    illumination_contrast_boost: float = 1.05
    illumination_binary_check_values: tuple[int, ...] = (0, 128, 255)
    pure_white_fringe_px: int = 1
    pure_white_bg_tol: int = 12

    # Mask candidates
    sauvola_window: int = 25
    sauvola_k: float = 0.20
    adaptive_window: int = 31
    adaptive_offset: int = 10

    # Mask candidate scoring weights
    mask_plausible_weight: float = 0.25
    mask_speck_weight: float = 0.15
    mask_aspect_weight: float = 0.12
    mask_continuity_weight: float = 0.18
    mask_faint_weight: float = 0.15
    mask_boundary_weight: float = 0.10
    mask_stroke_width_weight: float = 0.05

    # Conservative artifact cleaning
    line_horizontal_fraction: float = 0.60
    line_vertical_fraction: float = 0.75
    line_thickness_fraction: float = 0.08
    rescue_distance_px: int = 2
    max_ink_removed_fraction: float = 0.70
    isolated_speck_area: int = 9

    # Artifact detection thresholds
    hollow_box_min_fraction: float = 0.20
    hollow_box_max_fill: float = 0.35
    stamp_min_area_fraction: float = 0.10
    stamp_max_area_fraction: float = 0.80
    stamp_min_width_fraction: float = 0.20
    stamp_min_height_fraction: float = 0.10
    stamp_min_fill: float = 0.50
    edge_sliver_thickness_fraction: float = 0.02
    crossing_density_run_fraction: float = 0.60
    body_substantial_ratio: float = 0.30
    edge_text_gap_px: int = 4

    # Geometry
    crop_margin_fraction: float = 0.05
    max_deskew_degrees: float = 8.0
    min_deskew_confidence: float = 0.70
    deskew_angle_resolution: float = 0.5
    clip_contact_run_px: int = 5

    # Quality
    low_contrast_flag: float = 18.0
    low_blur_flag: float = 20.0

    # KEEP-by-default conservative cleaning
    destructive_cleaning_enabled: bool = False
    artifact_metadata_only: bool = True
    external_text_mask_enabled: bool = False
    external_text_overlap_fraction: float = 0.80
    external_text_dilate_px: int = 2

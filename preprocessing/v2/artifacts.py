"""Conservative cheque-artifact detection for production-v2.
Artifact mask values:
    0   = untouched (signature ink)
    128 = suspected but kept (uncertain crossing lines, rescued fragments)
    255 = confidently removed (lines, stamps, boxes, edge slivers, specks)
"""
from __future__ import annotations
import cv2
import numpy as np
from .params import Params

def _near_body(candidate_box, body_box, distance):
    x0, y0, x1, y1 = candidate_box
    a0, b0, a1, b1 = body_box
    e = (a0 - distance, b0 - distance, a1 + distance, b1 + distance)
    return not (x1 < e[0] or x0 > e[2] or y1 < e[1] or y0 > e[3])

def _compute_body_box(stats, labels, count, binary, params):
    h, w = binary.shape[:2]
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    if areas.size == 0: return [], (0,0,0,0), 0
    largest = int(areas.max())
    substantial = []
    for lbl in range(1, count):
        a = int(stats[lbl, cv2.CC_STAT_AREA])
        cw, ch = int(stats[lbl, cv2.CC_STAT_WIDTH]), int(stats[lbl, cv2.CC_STAT_HEIGHT])
        x, y = int(stats[lbl, cv2.CC_STAT_LEFT]), int(stats[lbl, cv2.CC_STAT_TOP])
        is_line = ((cw >= params.line_horizontal_fraction*w and ch <= max(1, params.line_thickness_fraction*h)) or
                   (ch >= params.line_vertical_fraction*h and cw <= max(1, params.line_thickness_fraction*w)))
        edge_sides = int(x <= 1) + int(y <= 1) + int(x + cw >= w - 1) + int(y + ch >= h - 1)
        fill_ratio = a / max(1, cw * ch)
        is_edge_frame = (
            edge_sides >= 3
            and fill_ratio <= params.hollow_box_max_fill
            and cw * ch >= params.hollow_box_min_fraction * w * h
        )
        is_full_frame = _is_full_border_component(stats, lbl, w, h, params)
        if (
            a >= params.body_substantial_ratio*largest
            and not is_line
            and not is_edge_frame
            and not is_full_frame
        ):
            substantial.append(lbl)
    if not substantial:
        pl = int(np.argmax(areas)) + 1
        substantial = [pl]
    else:
        pl = max(substantial, key=lambda l: int(stats[l, cv2.CC_STAT_AREA]))
    x0,y0,x1,y1 = w,h,0,0
    for lbl in substantial:
        x,y = int(stats[lbl, cv2.CC_STAT_LEFT]), int(stats[lbl, cv2.CC_STAT_TOP])
        cw,ch = int(stats[lbl, cv2.CC_STAT_WIDTH]), int(stats[lbl, cv2.CC_STAT_HEIGHT])
        x0 = min(x0, x); y0 = min(y0, y)
        x1 = max(x1, x+cw); y1 = max(y1, y+ch)
    return substantial, (x0, y0, x1, y1), pl

def _is_line_component(stats, lbl, w, h, params):
    cw, ch = int(stats[lbl, cv2.CC_STAT_WIDTH]), int(stats[lbl, cv2.CC_STAT_HEIGHT])
    thin_h = ch <= max(1, params.line_thickness_fraction*h)
    thin_w = cw <= max(1, params.line_thickness_fraction*w)
    return (cw >= params.line_horizontal_fraction*w and thin_h,
            ch >= params.line_vertical_fraction*h and thin_w)

def _is_hollow_box(stats, lbl, w, h, params):
    cw, ch = int(stats[lbl, cv2.CC_STAT_WIDTH]), int(stats[lbl, cv2.CC_STAT_HEIGHT])
    if cw < params.hollow_box_min_fraction*w or ch < params.hollow_box_min_fraction*h: return False
    return int(stats[lbl, cv2.CC_STAT_AREA])/max(1,cw*ch) <= params.hollow_box_max_fill


def _is_full_border_component(stats, lbl, w, h, params):
    x = int(stats[lbl, cv2.CC_STAT_LEFT])
    y = int(stats[lbl, cv2.CC_STAT_TOP])
    cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
    ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
    band = max(10, int(params.rescue_distance_px * 5))
    touch_sides = sum((
        x <= band,
        y <= band,
        x + cw >= w - band,
        y + ch >= h - band,
    ))
    near_full_frame = False
    edge_strip = (
        touch_sides >= 2
        and cw <= 0.45 * w
        and ch >= 0.75 * h
        and int(stats[lbl, cv2.CC_STAT_AREA]) / max(1, cw * ch) <= params.hollow_box_max_fill
    )
    return near_full_frame or edge_strip

def _is_stamp(stats, lbl, w, h, la, params):
    a, cw, ch = int(stats[lbl, cv2.CC_STAT_AREA]), int(stats[lbl, cv2.CC_STAT_WIDTH]), int(stats[lbl, cv2.CC_STAT_HEIGHT])
    if a < params.stamp_min_area_fraction*la or a > params.stamp_max_area_fraction*la: return False
    if cw < params.stamp_min_width_fraction*w or ch < params.stamp_min_height_fraction*h: return False
    return a/max(1,cw*ch) >= params.stamp_min_fill

def _is_edge_sliver(stats, lbl, w, h, la, params):
    a = int(stats[lbl, cv2.CC_STAT_AREA])
    if a >= la: return False
    x,y = int(stats[lbl, cv2.CC_STAT_LEFT]), int(stats[lbl, cv2.CC_STAT_TOP])
    cw,ch = int(stats[lbl, cv2.CC_STAT_WIDTH]), int(stats[lbl, cv2.CC_STAT_HEIGHT])
    if not (x <=1 or y<=1 or x+cw >= w-1 or y+ch >= h-1): return False
    return min(cw,ch) <= max(1, params.edge_sliver_thickness_fraction*max(w,h)) and max(cw,ch) > min(cw,ch)*4

def _is_distant_speck(stats, lbl, labels_arr, body_box, unc_mask, params):
    a = int(stats[lbl, cv2.CC_STAT_AREA])
    if a > params.isolated_speck_area: return False
    x,y = int(stats[lbl, cv2.CC_STAT_LEFT]), int(stats[lbl, cv2.CC_STAT_TOP])
    cw,ch = int(stats[lbl, cv2.CC_STAT_WIDTH]), int(stats[lbl, cv2.CC_STAT_HEIGHT])
    if _near_body((x,y,x+cw,y+ch), body_box, params.rescue_distance_px): return False
    if unc_mask is not None and np.any(unc_mask):
        k = 2*params.rescue_distance_px+1
        e = cv2.dilate(unc_mask, np.ones((k,k), dtype=np.uint8))
        if bool(np.any(e[labels_arr == lbl])): return False
    return True

def _crossing_line_high_density(labels_arr, lbl, w, params):
    if labels_arr.shape[1] < params.line_horizontal_fraction*w: return False
    c = (labels_arr == lbl)
    return bool(np.max(np.count_nonzero(c, axis=1)) >= params.crossing_density_run_fraction*labels_arr.shape[1])


def _touches_image_edge(box, width, height):
    x0, y0, x1, y1 = box
    return x0 <= 1 or y0 <= 1 or x1 >= width - 1 or y1 >= height - 1


def _outside_expanded_body(box, body_box, expansion_fraction=0.10):
    x0, y0, x1, y1 = box
    bx0, by0, bx1, by1 = body_box
    body_width = max(1, bx1 - bx0)
    body_height = max(1, by1 - by0)
    ex = expansion_fraction * body_width
    ey = expansion_fraction * body_height
    expanded = (bx0 - ex, by0 - ey, bx1 + ex, by1 + ey)
    return not (
        x1 >= expanded[0]
        and x0 <= expanded[2]
        and y1 >= expanded[1]
        and y0 <= expanded[3]
    )


def _outside_body_by_pixels(box, body_box, pixels=4):
    x0, y0, x1, y1 = box
    bx0, by0, bx1, by1 = body_box
    horizontal_gap = max(bx0 - x1, x0 - bx1, 0)
    vertical_gap = max(by0 - y1, y0 - by1, 0)
    return horizontal_gap >= pixels or vertical_gap >= pixels


def _is_small_mark_near_body(box, body_box, area, largest_area, distance_px=25):
    x0, y0, x1, y1 = box
    bx0, by0, bx1, by1 = body_box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    if area >= 0.30 * largest_area:
        return False
    dot_like = width <= 12 and height <= 12
    dash_like = width >= 3 * height and width <= 40
    near = not (
        x1 < bx0 - distance_px
        or x0 > bx1 + distance_px
        or y1 < by0 - distance_px
        or y0 > by1 + distance_px
    )
    return near and (dot_like or dash_like)


def _detect_crossing_line_in_body(binary, body_box, primary_label, labels_arr, params):
    h, w = binary.shape[:2]
    body_mask = (labels_arr == primary_label)
    if not np.any(body_mask):
        return False
    row_ink = np.count_nonzero(body_mask, axis=1)
    thin_h = int(max(1, params.line_thickness_fraction * h))
    for y in range(int(body_box[1]), int(body_box[3])):
        if y >= len(row_ink):
            break
        if row_ink[y] >= params.line_horizontal_fraction * w:
            top = int(max(0, int(y) - thin_h))
            bottom = int(min(int(h) - 1, int(y) + thin_h))
            above_start = int(max(0, y - 6))
            above_end = int(max(0, y - 2))
            below_start = int(min(h, y + 2))
            below_end = int(min(h, y + 6))
            row_above = np.count_nonzero(body_mask[above_start:above_end, :]) if above_end > above_start else 0
            row_below = np.count_nonzero(body_mask[below_start:below_end, :]) if below_end > below_start else 0
            surrounding_rows = int(above_end - above_start + below_end - below_start)
            row_density = row_ink[y] / max(1, w)
            surrounding_density = (row_above + row_below) / max(1, surrounding_rows * w)
            if row_density >= 0.60 and surrounding_density < row_density * 0.15:
                return True
    return False


def _detect_crossing_line_in_body(binary, body_box, primary_label, labels_arr, params):
    h, w = binary.shape[:2]
    body_mask = (labels_arr == primary_label)
    if not np.any(body_mask):
        return False
    row_ink = np.count_nonzero(body_mask, axis=1)
    thin_h = int(max(1, params.line_thickness_fraction * h))
    for y in range(int(body_box[1]), int(body_box[3])):
        if y >= len(row_ink):
            break
        if row_ink[y] >= params.line_horizontal_fraction * w:
            above_start = int(max(0, y - 6))
            above_end = int(max(0, y - 1))
            below_start = int(min(h, y + 1))
            below_end = int(min(h, y + 6))
            surround_ink = 0
            surround_rows = 0
            if above_end > above_start:
                surround_ink += int(np.sum(row_ink[above_start:above_end]))
                surround_rows += above_end - above_start
            if below_end > below_start:
                surround_ink += int(np.sum(row_ink[below_start:below_end]))
                surround_rows += below_end - below_start
            surround_density = surround_ink / max(1.0, surround_rows * w)
            line_density = row_ink[y] / max(1.0, w)
            if line_density >= 0.60 and surround_density < line_density * 0.15:
                return True
    return False

def clean_artifacts(ink_mask, params=None):
    params = params or Params()
    if ink_mask.dtype != np.uint8: raise ValueError("ink mask must be uint8")
    binary = (ink_mask > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    empty = {"artifact_detected": False, "artifact_uncertain": False, "ink_removed": 0,
             "ink_removed_fraction": 0.0, "artifact_components_removed": 0,
             "artifact_components_uncertain": 0, "rescued_fragments": 0}
    if count <= 1: return (binary*255).astype(np.uint8), np.zeros_like(binary), empty
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    la = int(areas.max()) if areas.size else 1
    substantial, body_box, primary = _compute_body_box(stats, labels, count, binary, params)
    art_mask = np.zeros_like(binary)
    unc_mask = np.zeros_like(binary)
    detected = False; uncertain = False
    removed_px = 0; removed_n = 0; uncertain_n = 0; rescued_n = 0
    border_removed_px = 0
    decisions = {}
    has_crossing = _detect_crossing_line_in_body(binary, body_box, primary, labels, params)
    if has_crossing:
        uncertain = True
        body_mask = (labels == primary)
        h_b, w_b = binary.shape[:2]
        thin_h = int(max(1, params.line_thickness_fraction * int(h_b)))
        row_ink = np.count_nonzero(body_mask, axis=1)
        for y in range(int(body_box[1]), int(body_box[3])):
            if y >= len(row_ink):
                break
            if row_ink[y] >= params.line_horizontal_fraction * w_b:
                top = int(max(0, y - thin_h))
                bottom = int(min(h_b - 1, y + thin_h))
                art_mask[top:bottom + 1, :] = 128
                unc_mask[top:bottom + 1, :] = 255
                uncertain_n += 1

    for lbl in range(1, count):
        full_border = _is_full_border_component(stats, lbl, w, h, params)
        if full_border:
            border_removed_px += int(stats[lbl, cv2.CC_STAT_AREA])
            art_mask[labels == lbl] = 255
            detected = True
            removed_n += 1
            continue
        if lbl == primary or lbl in substantial:
            continue
        if int(stats[lbl, cv2.CC_STAT_AREA]) < params.min_ink_pixels: continue
        x,y = int(stats[lbl, cv2.CC_STAT_LEFT]), int(stats[lbl, cv2.CC_STAT_TOP])
        cw,ch = int(stats[lbl, cv2.CC_STAT_WIDTH]), int(stats[lbl, cv2.CC_STAT_HEIGHT])
        box = (x, y, x+cw, y+ch)
        hl, vl = _is_line_component(stats, lbl, w, h, params)
        thick_cw = cw > max(1, params.line_thickness_fraction * w) * 3
        thick_ch = ch > max(1, params.line_thickness_fraction * h) * 3
        touches = _near_body(box, body_box, params.rescue_distance_px)
        vs = vl and (thick_cw or touches)
        edge = _is_edge_sliver(stats, lbl, w, h, la, params)
        hbox = _is_hollow_box(stats, lbl, w, h, params)
        stamp = _is_stamp(stats, lbl, w, h, la, params)
        speck = _is_distant_speck(stats, lbl, labels, body_box, unc_mask, params)
        dense = _crossing_line_high_density(labels, lbl, w, params)
        atype = None
        small_mark_near_body = _is_small_mark_near_body(
            box,
            body_box,
            int(stats[lbl, cv2.CC_STAT_AREA]),
            la,
        )
        edge_text_piece = (
            int(stats[lbl, cv2.CC_STAT_AREA]) < params.body_substantial_ratio * la
            and _touches_image_edge(box, w, h)
            and _outside_body_by_pixels(box, body_box, params.edge_text_gap_px)
            and (
                y <= 1
                or x <= 1
                or x + cw >= w - 1
                or y + ch >= h - 1
            )
        )
        is_small_outside_component = (
            int(stats[lbl, cv2.CC_STAT_AREA]) < params.body_substantial_ratio * la
            and (
                _outside_expanded_body(box, body_box, 0.10)
                or (
                    _touches_image_edge(box, w, h)
                    and _outside_body_by_pixels(box, body_box, params.edge_text_gap_px)
                )
            )
        )
        if hbox:
            atype = "hollow_box"
        elif stamp:
            atype = "stamp"
        elif edge_text_piece:
            atype = "edge_text"
        elif small_mark_near_body:
            atype = None
        elif is_small_outside_component:
            atype = "edge_text_or_fragment"
        elif edge:
            atype = "edge_sliver"
        elif speck:
            atype = "speck"
        elif (hl or vl) and not (vs or (hl and thick_ch)):
            atype = "uncertain_crossing" if touches else "line"
        elif dense and touches:
            atype = "uncertain_crossing"
        if atype: decisions[lbl] = atype
    for lbl, atype in decisions.items():
        if atype == "uncertain_crossing":
            art_mask[labels == lbl] = 128; unc_mask[labels == lbl] = 255
            uncertain = True; uncertain_n += 1
        else:
            removed_px += int(stats[lbl, cv2.CC_STAT_AREA])
            art_mask[labels == lbl] = 255; detected = True; removed_n += 1
    if unc_mask.any():
        k = 2*params.rescue_distance_px+1
        e = cv2.dilate(unc_mask, np.ones((k,k), dtype=np.uint8))
        for lbl in range(1, count):
            if not bool(np.any(e[labels == lbl])): continue
            if _is_full_border_component(stats, lbl, w, h, params):
                continue
            if art_mask[labels == lbl].max() == 255:
                removed_px = max(0, removed_px - int(stats[lbl, cv2.CC_STAT_AREA]))
                removed_n = max(0, removed_n-1); rescued_n += 1
                art_mask[labels == lbl] = 128; unc_mask[labels == lbl] = 255
    total_ink = int(np.count_nonzero(binary))
    removed_frac = removed_px / max(1, total_ink)
    if removed_px / max(1, total_ink) > params.max_ink_removed_fraction:
        art_mask[:] = 0; unc_mask[:] = 0; removed_px = 0; removed_frac = 0.0
        detected = False; uncertain = True
    cleaned = (binary*255).astype(np.uint8)
    cleaned[art_mask == 255] = 0
    return cleaned, art_mask, {
        "artifact_detected": detected, "artifact_uncertain": uncertain,
        "ink_removed": int(removed_px), "ink_removed_fraction": float(removed_frac),
        "border_removed_fraction": float(border_removed_px / max(1, total_ink)),
        "border_removed_fraction": float(border_removed_px / max(1, total_ink)),
        "artifact_components_removed": removed_n, "artifact_components_uncertain": uncertain_n,
        "rescued_fragments": rescued_n,
    }

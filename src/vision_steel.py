import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

BIG_CONTAINER = {
    "name": "big_square",
    "width_mm": 340,
    "length_mm": 310,
    "height_mm": 70,
    "aspect_ratio": 340 / 310,
    "area_mm2": 340 * 310,
    "volume_ml": (340 * 310 * 70) / 1000,
}

SMALL_CONTAINER = {
    "name": "small_rectangle",
    "width_mm": 315,
    "length_mm": 165,
    "height_mm": 70,
    "aspect_ratio": 315 / 165,
    "area_mm2": 315 * 165,
    "volume_ml": (315 * 165 * 70) / 1000,
}

ASPECT_RATIO_THRESHOLD = 1.5


def detect_steel_utensil(
    frame: np.ndarray, utensil_hint: str = "auto", min_area: int = 10000, debug: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    info = {
        "method": None,
        "shape_type": None,
        "center": None,
        "bounding_rect": None,
        "area": None,
    }

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    contours_edge, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    steel_mask = cv2.inRange(hsv, (0, 0, 100), (180, 80, 255))

    steel_mask = cv2.morphologyEx(steel_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    steel_mask = cv2.morphologyEx(steel_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours_steel, _ = cv2.findContours(steel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_contours = list(contours_edge) + list(contours_steel)

    if not all_contours:
        return None, None, info

    best_contour = None
    best_score = 0

    for contour in all_contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        x, y, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = float(bw) / bh if bh > 0 else 0

        area_score = min(area / (h * w * 0.5), 1.0)

        cx = x + bw / 2
        cy = y + bh / 2
        center_dist = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
        center_score = 1.0 - min(center_dist / (min(h, w) * 0.5), 1.0)

        aspect_score = 1.0 if 0.3 < aspect_ratio < 3.0 else 0.5

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        solidity_score = solidity

        total_score = (
            area_score * 0.3 + center_score * 0.3 + aspect_score * 0.2 + solidity_score * 0.2
        )
        if utensil_hint == "round":
            circularity = 4 * np.pi * area / perimeter**2
            if circularity < 0.65 or min(bw, bh) / max(bw, bh) < 0.75:
                continue
            total_score *= circularity

        if total_score > best_score:
            best_score = total_score
            best_contour = contour

    if best_contour is None:
        return None, None, info

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [best_contour], -1, 255, -1)

    area = cv2.contourArea(best_contour)
    perimeter = cv2.arcLength(best_contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

    x, y, bw, bh = cv2.boundingRect(best_contour)
    aspect_ratio = float(bw) / bh if bh > 0 else 1.0

    rect_area = bw * bh
    extent = area / rect_area if rect_area > 0 else 0

    hull = cv2.convexHull(best_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(best_contour, epsilon, True)
    num_corners = len(approx)

    if circularity > 0.85:
        shape_type = "round"
    elif extent > 0.60 and solidity > 0.85 and 0.4 < circularity < 0.85:
        shape_type = "rounded_rectangle"
    elif num_corners == 4 and extent > 0.8 and 0.85 < aspect_ratio < 1.15:
        shape_type = "square"
    elif num_corners == 4 and extent > 0.8:
        shape_type = "rectangular"
    elif extent > 0.55 and solidity > 0.75:
        shape_type = "rounded_rectangle"
    else:
        shape_type = "irregular"

    M = cv2.moments(best_contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        info["center"] = (cx, cy)

    info["method"] = "combined_steel_edge"
    info["shape_type"] = shape_type
    info["bounding_rect"] = (x, y, bw, bh)
    info["area"] = area
    info["circularity"] = circularity
    info["extent"] = extent
    info["solidity"] = solidity

    return best_contour, mask, info


def segment_food_in_steel_utensil(
    frame: np.ndarray, utensil_contour: np.ndarray, utensil_mask: np.ndarray, debug: bool = False
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:

    info = {}

    if utensil_mask is None or np.count_nonzero(utensil_mask) == 0:
        return None, info

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    h_channel, s_channel, v_channel = cv2.split(hsv)
    l_channel, a_channel, b_channel = cv2.split(lab)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    interior_mask = cv2.erode(utensil_mask, kernel, iterations=2)
    rim_mask = cv2.subtract(utensil_mask, interior_mask)

    rim_saturation = s_channel[rim_mask > 0]
    if len(rim_saturation) > 100:
        steel_sat_threshold = np.percentile(rim_saturation, 85)
    else:
        steel_sat_threshold = 40

    sat_mask = np.zeros_like(utensil_mask)
    sat_mask[(s_channel > steel_sat_threshold) & (utensil_mask > 0)] = 255

    rim_a = a_channel[rim_mask > 0]
    rim_b = b_channel[rim_mask > 0]

    if len(rim_a) > 100:
        steel_a = np.median(rim_a)
        steel_b = np.median(rim_b)
    else:
        steel_a = 128
        steel_b = 128

    a_dev = np.abs(a_channel.astype(np.float32) - steel_a)
    b_dev = np.abs(b_channel.astype(np.float32) - steel_b)
    color_dev = np.sqrt(a_dev**2 + b_dev**2)

    color_thresh = 15
    color_mask = np.zeros_like(utensil_mask)
    color_mask[(color_dev > color_thresh) & (utensil_mask > 0)] = 255

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    blur_sq = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (5, 5), 0)
    variance = blur_sq - blur**2
    variance = np.clip(variance, 0, None)

    var_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    rim_var = var_norm[rim_mask > 0]
    if len(rim_var) > 100:
        texture_cutoff = np.percentile(rim_var, 75) + 10
    else:
        texture_cutoff = 30

    texture_mask = np.zeros_like(utensil_mask)
    texture_mask[(var_norm > texture_cutoff) & (utensil_mask > 0)] = 255

    # combine segmentation cues
    combined = (
        (sat_mask > 0).astype(np.uint8)
        + (color_mask > 0).astype(np.uint8)
        + (texture_mask > 0).astype(np.uint8)
    )
    food_mask = np.zeros_like(utensil_mask)
    food_mask[combined >= 2] = 255

    if debug:
        info["stages"] = {"raw": food_mask.copy(), "rim": rim_mask.copy()}

    # clean up the mask
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    if debug:
        info["stages"]["cleanup"] = food_mask.copy()

    contours, _ = cv2.findContours(food_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(food_mask)
    min_food_area = np.count_nonzero(utensil_mask) * 0.01

    for contour in contours:
        if cv2.contourArea(contour) >= min_food_area:
            cv2.drawContours(clean_mask, [contour], -1, 255, -1)

    food_pixels = np.count_nonzero(clean_mask)
    if debug:
        info["stages"]["filtered"] = clean_mask.copy()
    utensil_pixels = np.count_nonzero(utensil_mask)

    if food_pixels > utensil_pixels * 0.05:
        try:
            gc_mask = np.zeros_like(utensil_mask)
            gc_mask[utensil_mask == 0] = cv2.GC_BGD
            gc_mask[(utensil_mask > 0) & (clean_mask == 0)] = cv2.GC_PR_BGD
            gc_mask[(utensil_mask > 0) & (clean_mask > 0)] = cv2.GC_PR_FGD

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            cv2.grabCut(frame, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)

            food_mask_refined = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
            food_mask_refined[utensil_mask == 0] = 0

            refined_pixels = np.count_nonzero(food_mask_refined)
            if refined_pixels > food_pixels * 0.3:
                clean_mask = food_mask_refined
        except cv2.error:
            pass

    info["food_pixels"] = np.count_nonzero(clean_mask)
    info["utensil_pixels"] = utensil_pixels
    info["coverage_percent"] = (
        (info["food_pixels"] / utensil_pixels * 100) if utensil_pixels > 0 else 0
    )

    return clean_mask, info


def identify_container_type(aspect_ratio: float) -> Dict[str, Any]:

    ar = max(aspect_ratio, 1.0 / aspect_ratio) if aspect_ratio > 0 else 1.0

    if ar < ASPECT_RATIO_THRESHOLD:
        return BIG_CONTAINER.copy()
    else:
        return SMALL_CONTAINER.copy()


def estimate_volume_from_contour(
    utensil_contour: np.ndarray,
    utensil_mask: np.ndarray,
    food_mask: np.ndarray,
    utensil_info: Dict[str, Any],
    known_dimension_mm: Optional[float] = None,
    assumed_depth_mm: float = 70.0,
    utensil_hint: str = "auto",
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:

    info = {}

    utensil_pixels = np.count_nonzero(utensil_mask)
    food_pixels = np.count_nonzero(food_mask) if food_mask is not None else 0

    if utensil_pixels == 0:
        return None, None, info

    percent_fill = (food_pixels / utensil_pixels) * 100.0
    info["percent_fill"] = percent_fill
    info["food_pixels"] = food_pixels
    info["utensil_pixels"] = utensil_pixels

    if utensil_hint == "round":
        diameter_mm = known_dimension_mm if known_dimension_mm is not None else 280.0
        if not np.isfinite(diameter_mm) or diameter_mm <= 0:
            raise ValueError("Plate diameter must be positive and finite")
        if not np.isfinite(assumed_depth_mm) or assumed_depth_mm <= 0:
            raise ValueError("Food depth must be positive and finite")
        _, _, bw, bh = cv2.boundingRect(utensil_contour)
        diameter_px = max(bw, bh)
        mm_per_px = diameter_mm / diameter_px
        food_area_mm2 = food_pixels * mm_per_px**2
        volume_ml = food_area_mm2 * assumed_depth_mm / 1000.0
        info.update(
            container_type="plate",
            plate_diameter_mm=diameter_mm,
            mm_per_px=mm_per_px,
            food_area_mm2=food_area_mm2,
            effective_depth_mm=assumed_depth_mm,
            volume_ml=volume_ml,
        )
        return percent_fill, volume_ml, info

    x, y, bw, bh = utensil_info.get("bounding_rect", cv2.boundingRect(utensil_contour))
    aspect_ratio = float(bw) / bh if bh > 0 else 1.0

    if utensil_hint == "big_square":
        container = BIG_CONTAINER.copy()
    elif utensil_hint == "small_rectangle":
        container = SMALL_CONTAINER.copy()
    else:
        container = identify_container_type(aspect_ratio)

    info["container_type"] = container["name"]
    info["container_dims"] = (
        f"{container['width_mm'] / 10:.1f}x{container['length_mm'] / 10:.1f}x{container['height_mm'] / 10:.1f} cm"
    )

    long_px = max(bw, bh)
    short_px = min(bw, bh)

    long_mm = max(container["width_mm"], container["length_mm"])
    short_mm = min(container["width_mm"], container["length_mm"])

    if known_dimension_mm is not None and known_dimension_mm > 0:
        mm_per_px = known_dimension_mm / long_px if long_px > 0 else 0
    else:
        mm_per_px = long_mm / long_px if long_px > 0 else 0

    info["mm_per_px"] = mm_per_px
    info["detected_aspect_ratio"] = aspect_ratio

    volume_ml = None

    if mm_per_px > 0:
        food_area_mm2 = food_pixels * (mm_per_px**2)
        info["food_area_mm2"] = food_area_mm2

        container_area_mm2 = container["area_mm2"]
        info["container_area_mm2"] = container_area_mm2

        area_ratio = food_area_mm2 / container_area_mm2 if container_area_mm2 > 0 else 0
        info["area_fill_ratio"] = area_ratio

        height_mm = container["height_mm"]

        if percent_fill < 30:
            depth_factor = 0.2
        elif percent_fill < 70:
            depth_factor = 0.4
        else:
            depth_factor = 0.6

        depth_mm = height_mm * depth_factor
        info["effective_depth_mm"] = depth_mm

        volume_mm3 = food_area_mm2 * depth_mm
        info["volume_mm3"] = volume_mm3

        volume_ml = volume_mm3 / 1000.0
        info["volume_ml"] = volume_ml

        max_volume_ml = container["volume_ml"]
        info["max_container_volume_ml"] = max_volume_ml
        info["volume_percent_of_max"] = (
            (volume_ml / max_volume_ml * 100) if max_volume_ml > 0 else 0
        )

    return percent_fill, volume_ml, info


def detect_utensil_ellipse(frame, utensil_hint="auto", min_radius=60, max_radius=400, debug=False):

    contour, mask, info = detect_steel_utensil(frame, utensil_hint, debug=debug)

    if contour is None:
        return _detect_circle_fallback(frame, min_radius, max_radius, debug)

    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            return ellipse
        except cv2.error:
            pass

    x, y, w, h = cv2.boundingRect(contour)
    cx = x + w / 2
    cy = y + h / 2
    return ((cx, cy), (float(w), float(h)), 0.0)


def _detect_circle_fallback(frame, min_radius, max_radius, debug):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=120,
        param1=120,
        param2=40,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None and len(circles) > 0:
        circles = np.uint16(np.around(circles))
        h, w = gray.shape[:2]
        cx, cy = w / 2, h / 2

        def score(c):
            x, y, r = c
            return np.hypot(x - cx, y - cy) - 0.1 * r

        best = sorted(circles[0], key=score)[0]
        x, y, r = best
        return ((float(x), float(y)), (float(2 * r), float(2 * r)), 0.0)

    return None


def segment_food_in_utensil(img, ellipse, debug=False):

    if ellipse is None:
        return None, {}

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    (cx, cy), (MA, ma), angle = ellipse
    cv2.ellipse(mask, (int(cx), int(cy)), (int(MA / 2), int(ma / 2)), angle, 0, 360, 255, -1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0] if contours else None

    food_mask, info = segment_food_in_steel_utensil(img, contour, mask, debug)

    return food_mask, info


def _ellipse_mask(shape, ellipse):

    mask = np.zeros(shape[:2], dtype=np.uint8)
    (cx, cy), (MA, ma), angle = ellipse
    cv2.ellipse(mask, (int(cx), int(cy)), (int(MA / 2), int(ma / 2)), angle, 0, 360, 255, -1)
    return mask


def estimate_area_and_volume(ellipse, seg_mask, diameter_mm, utensil_hint, assumed_height_mm):

    if ellipse is None:
        return None, None

    (cx, cy), (MA, ma), angle = ellipse
    full_mask = _ellipse_mask(seg_mask.shape + (1,), ellipse)
    interior_px = int(np.count_nonzero(full_mask))
    food_px = int(np.count_nonzero(seg_mask))

    if interior_px == 0:
        return None, None

    percent_fill = 100.0 * food_px / interior_px

    est_volume_ml = None
    if diameter_mm is not None and assumed_height_mm is not None and assumed_height_mm > 0:
        major_axis_px = max(MA, ma)
        if major_axis_px > 0:
            mm_per_px = diameter_mm / major_axis_px
            food_area_mm2 = food_px * (mm_per_px**2)
            volume_mm3 = food_area_mm2 * assumed_height_mm
            est_volume_ml = volume_mm3 / 1000.0

    return percent_fill, est_volume_ml


def process_frame_steel_utensil(
    frame: np.ndarray,
    known_dimension_mm: Optional[float] = None,
    assumed_depth_mm: float = 70.0,
    utensil_hint: str = "auto",
    debug: bool = False,
) -> Dict[str, Any]:

    result = {
        "utensil_detected": False,
        "utensil_contour": None,
        "utensil_mask": None,
        "food_mask": None,
        "percent_fill": None,
        "volume_ml": None,
        "shape_type": None,
        "container_type": None,
    }

    if debug:
        result["debug_info"] = {}

    contour, mask, utensil_info = detect_steel_utensil(frame, utensil_hint, debug=debug)

    if contour is None:
        return result

    result["utensil_detected"] = True
    result["utensil_contour"] = contour
    result["utensil_mask"] = mask
    result["shape_type"] = utensil_info.get("shape_type")

    if debug:
        result["debug_info"]["utensil_detection"] = utensil_info

    food_mask, seg_info = segment_food_in_steel_utensil(frame, contour, mask, debug)
    result["food_mask"] = food_mask

    if debug:
        result["debug_info"]["food_segmentation"] = seg_info

    percent_fill, volume_ml, vol_info = estimate_volume_from_contour(
        contour, mask, food_mask, utensil_info, known_dimension_mm, assumed_depth_mm, utensil_hint
    )

    result["percent_fill"] = percent_fill
    result["volume_ml"] = volume_ml
    result["container_type"] = vol_info.get("container_type")
    result["volume_info"] = vol_info

    if debug:
        result["debug_info"]["volume_estimation"] = vol_info

    return result


def draw_detection_overlay(
    frame: np.ndarray,
    result: Dict[str, Any],
    show_contour: bool = True,
    show_food_mask: bool = True,
    show_info: bool = True,
) -> np.ndarray:

    display = frame.copy()

    if not result.get("utensil_detected"):
        if show_info:
            cv2.putText(
                display,
                "No utensil detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        return display

    if show_contour and result.get("utensil_contour") is not None:
        cv2.drawContours(display, [result["utensil_contour"]], -1, (0, 255, 255), 2)

    if show_food_mask and result.get("food_mask") is not None:
        overlay = display.copy()
        overlay[result["food_mask"] > 0] = (0, 0, 255)
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

    if show_info:
        y = 30
        texts = []

        if result.get("shape_type"):
            texts.append(f"Shape: {result['shape_type']}")

        if result.get("container_type"):
            container_name = {
                "big_square": "Big (34x31cm)",
                "small_rectangle": "Small (31.5x16.5cm)",
                "plate": "Plate",
            }.get(result["container_type"], result["container_type"])
            texts.append(f"Container: {container_name}")

        if result.get("percent_fill") is not None:
            texts.append(f"Fill: {result['percent_fill']:.1f}%")

        if result.get("volume_ml") is not None:
            texts.append(f"Volume: {result['volume_ml']:.0f} ml")

        for i, text in enumerate(texts):
            cv2.putText(
                display, text, (10, y + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

    return display

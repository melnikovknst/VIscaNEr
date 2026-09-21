"""Coordinate handling, mask geometry and bottle anatomy.

Everything in this module works in ONE coordinate frame, defined once and used
identically for the image, the label box and the mask:

    "source pixels" = the decoded image AFTER `ImageOps.exif_transpose`,
                      at full resolution, origin top-left, x right, y down.

Segmentation runs on a resized copy; its masks are mapped straight back to
source pixels. Label boxes that were measured against a different size are
rescaled explicitly by :func:`rescale_box`. Nothing downstream is allowed to
guess a frame.

Requires numpy, OpenCV and Pillow. The auditing stages do not import this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

BBOX_FORMATS = ("xyxy_abs", "xywh_abs", "cxcywh_norm", "xyxy_norm")


class GeometryError(ValueError):
    """Raised when a box or mask cannot be placed in the source frame."""


# ----------------------------------------------------------------- loading
@dataclass(frozen=True)
class LoadedImage:
    """An image in the source frame, plus the provenance needed to reproduce it."""

    rgb: np.ndarray                 # H x W x 3, uint8
    path: Path
    stored_size: tuple[int, int]    # (width, height) before EXIF transpose
    source_size: tuple[int, int]    # (width, height) after EXIF transpose
    exif_orientation: int           # 1 when absent or already upright

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    def record(self) -> dict[str, Any]:
        return {
            "stored_width": self.stored_size[0],
            "stored_height": self.stored_size[1],
            "source_width": self.source_size[0],
            "source_height": self.source_size[1],
            "exif_orientation": self.exif_orientation,
            "exif_applied": self.exif_orientation != 1,
        }


_EXIF_ORIENTATION_TAG = 274


def load_source_image(path: str | Path) -> LoadedImage:
    """Decode to the source frame. EXIF rotation is applied exactly once."""
    target = Path(path)
    with Image.open(target) as handle:
        stored_size = handle.size
        orientation = 1
        try:
            exif = handle.getexif()
            orientation = int(exif.get(_EXIF_ORIENTATION_TAG, 1) or 1)
        except Exception:  # noqa: BLE001 - malformed EXIF is common and harmless
            orientation = 1
        upright = ImageOps.exif_transpose(handle)
        rgb = np.asarray(upright.convert("RGB"))
        source_size = upright.size
    return LoadedImage(
        rgb=rgb,
        path=target,
        stored_size=(int(stored_size[0]), int(stored_size[1])),
        source_size=(int(source_size[0]), int(source_size[1])),
        exif_orientation=orientation,
    )


# -------------------------------------------------------------- box formats
def parse_box(
    values: Sequence[float],
    *,
    fmt: str,
    reference_size: tuple[int, int] | None = None,
) -> tuple[float, float, float, float]:
    """Return ``(x1, y1, x2, y2)`` in the frame the values were measured in."""
    if fmt not in BBOX_FORMATS:
        raise GeometryError(f"Unsupported bbox_format {fmt!r}; expected one of {BBOX_FORMATS}")
    a, b, c, d = (float(v) for v in values)
    if fmt == "xyxy_abs":
        x1, y1, x2, y2 = a, b, c, d
    elif fmt == "xywh_abs":
        x1, y1, x2, y2 = a, b, a + c, b + d
    else:
        if reference_size is None:
            raise GeometryError(f"bbox_format {fmt!r} needs a reference image size")
        width, height = reference_size
        if fmt == "cxcywh_norm":
            cx, cy, bw, bh = a * width, b * height, c * width, d * height
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        else:  # xyxy_norm
            x1, y1, x2, y2 = a * width, b * height, c * width, d * height
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def rescale_box(
    box: tuple[float, float, float, float],
    *,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Move a box between two image sizes of the same picture."""
    fw, fh = from_size
    tw, th = to_size
    if fw <= 0 or fh <= 0:
        raise GeometryError(f"Invalid source size {from_size}")
    sx, sy = tw / fw, th / fh
    x1, y1, x2, y2 = box
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def clip_box(
    box: tuple[float, float, float, float],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = size
    x1, y1, x2, y2 = box
    return (
        max(0, int(math.floor(x1))),
        max(0, int(math.floor(y1))),
        min(width, int(math.ceil(x2))),
        min(height, int(math.ceil(y2))),
    )


def box_centre(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_inside_image(box: tuple[float, float, float, float], size: tuple[int, int], *, tol: float = 2.0) -> bool:
    width, height = size
    x1, y1, x2, y2 = box
    return x1 >= -tol and y1 >= -tol and x2 <= width + tol and y2 <= height + tol


def infer_box_reference_size(
    box: tuple[float, float, float, float],
    candidates: Iterable[tuple[int, int]],
) -> tuple[int, int] | None:
    """Pick the first candidate size the box actually fits inside.

    Used only to *verify* a declared frame, never to silently repair a config:
    the caller flags the row when nothing fits.
    """
    for size in candidates:
        if box_inside_image(box, size):
            return size
    return None


# ------------------------------------------------------------ mask geometry
def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_box_coverage(mask: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """``|mask AND box| / |box|`` - how much of the label the mask owns."""
    x1, y1, x2, y2 = clip_box(box, (mask.shape[1], mask.shape[0]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    window = mask[y1:y2, x1:x2]
    return float(window.sum()) / float(window.size)


def mask_contains_point(mask: np.ndarray, point: tuple[float, float]) -> bool:
    x, y = int(round(point[0])), int(round(point[1]))
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return False
    return bool(mask[y, x])


def mask_solidity(mask: np.ndarray) -> float:
    """Mask area over its convex hull area. A sliver or a fence scores low."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    hull_area = cv2.contourArea(cv2.convexHull(largest))
    if hull_area <= 0:
        return 0.0
    return float(area / hull_area)


def mask_hole_fraction(mask: np.ndarray) -> float:
    """Area of enclosed holes over filled silhouette area.

    A neighbouring bottle or a price tag in front of the target leaves a bite in
    the mask; that is a quality flag, not something to inpaint.
    """
    binary = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not contours:
        return 0.0
    outer = sum(cv2.contourArea(c) for c, h in zip(contours, hierarchy[0]) if h[3] < 0)
    holes = sum(cv2.contourArea(c) for c, h in zip(contours, hierarchy[0]) if h[3] >= 0)
    if outer <= 0:
        return 0.0
    return float(holes / outer)


def mask_touches_border(mask: np.ndarray, *, margin: int = 2) -> dict[str, bool]:
    m = max(1, int(margin))
    return {
        "top": bool(mask[:m, :].any()),
        "bottom": bool(mask[-m:, :].any()),
        "left": bool(mask[:, :m].any()),
        "right": bool(mask[:, -m:].any()),
    }


# ------------------------------------------------------------ principal axis
@dataclass(frozen=True)
class AxisEstimate:
    """Mask principal axis, with the information needed to trust it.

    ``angle_deg`` is the tilt of the long axis away from vertical, positive
    clockwise. ``flip_required`` is True when the narrow (neck) end points down,
    i.e. the bottle is upside down in the source frame.
    """

    angle_deg: float
    elongation: float
    centroid: tuple[float, float]
    narrow_end: str                 # "first" or "second" end along the axis
    end_width_ratio: float          # wide end width / narrow end width
    body_half_area_share: float     # area share of the half holding the wide end
    orientation_confident: bool
    flip_required: bool
    reliable: bool
    reason: str

    def record(self) -> dict[str, Any]:
        return asdict(self)


def estimate_axis(
    mask: np.ndarray,
    *,
    min_elongation: float,
    max_abs_angle_deg: float,
    min_end_width_ratio: float,
    min_body_half_area_share: float = 0.56,
) -> AxisEstimate:
    """Principal axis of the silhouette, with the 180-degree ambiguity resolved.

    PCA gives a line, not a direction: the same axis describes an upright bottle
    and the same bottle upside down. Two independent signals have to agree
    before the crop is turned over:

    1. **End width.** A bottle is narrow at the neck and wide at the body, so
       the wide end is the base.
    2. **Area balance.** Most of a bottle's silhouette area sits in the body
       half, not the neck half.

    Requiring both matters. A catalog photograph taken on a reflective surface
    carries a mirrored bottle below the real one; that silhouette is narrow at
    *both* ends, and end width alone picks whichever reflection happens to fade
    faster - turning the bottle upside down. The area test is near-symmetric on
    such a mask, the two signals disagree, and the crop keeps its original
    orientation and is flagged instead.

    When the signals disagree or either is weak (a straight tumbler-shaped
    bottle, a bag-in-box, a heavily cropped mask) the estimate is marked
    unconfident and the caller does not flip.
    """
    ys, xs = np.nonzero(mask)
    if ys.size < 32:
        return AxisEstimate(0.0, 0.0, (0.0, 0.0), "first", 1.0, 0.5, False, False, False, "mask_too_small")

    points = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centroid = points.mean(axis=0)
    centred = points - centroid
    covariance = np.cov(centred, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    major = eigenvectors[:, order[0]]

    if eigenvalues[1] <= 1e-9:
        elongation = float("inf")
    else:
        elongation = float(math.sqrt(eigenvalues[0] / eigenvalues[1]))

    # Angle of the long axis away from vertical (the +y direction), clockwise.
    if major[1] < 0:
        major = -major
    angle_deg = float(math.degrees(math.atan2(major[0], major[1])))
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    # Width of the silhouette in the first and last decile along the axis.
    projection = centred @ major
    perpendicular = centred @ np.array([-major[1], major[0]])
    span = projection.max() - projection.min()
    if span <= 1e-6:
        return AxisEstimate(angle_deg, elongation, tuple(centroid), "first", 1.0, 0.5, False, False, False, "degenerate_axis")
    decile = 0.12 * span
    low = perpendicular[projection <= projection.min() + decile]
    high = perpendicular[projection >= projection.max() - decile]
    low_width = float(low.max() - low.min()) if low.size > 4 else 0.0
    high_width = float(high.max() - high.min()) if high.size > 4 else 0.0

    # Area balance along the axis: which half of the silhouette carries the mass.
    midpoint = (projection.max() + projection.min()) / 2.0
    total = float(projection.size)
    high_half_share = float((projection > midpoint).sum()) / total if total else 0.5

    if low_width <= 1e-6 or high_width <= 1e-6:
        narrow_end, ratio, width_says = "first", 1.0, False
    elif low_width <= high_width:
        narrow_end, ratio = "first", high_width / low_width
        width_says = ratio >= min_end_width_ratio
    else:
        narrow_end, ratio = "second", low_width / high_width
        width_says = ratio >= min_end_width_ratio

    # Share of the area in the half that the width test called the body.
    body_half_share = high_half_share if narrow_end == "first" else 1.0 - high_half_share
    area_says = body_half_share >= min_body_half_area_share
    confident = width_says and area_says

    # `major` points towards increasing y (downwards in image coordinates), so
    # its "second" end is the lower one. A narrow lower end means the neck is
    # pointing down and the crop has to be turned over.
    flip_required = confident and narrow_end == "second"

    reliable = True
    reason = "ok"
    if elongation < min_elongation:
        reliable, reason = False, f"elongation_{elongation:.2f}_below_{min_elongation}"
    elif abs(angle_deg) > max_abs_angle_deg:
        reliable, reason = False, f"angle_{angle_deg:.1f}_beyond_{max_abs_angle_deg}"
    elif not confident:
        reason = (
            "orientation_uncertain_signals_disagree"
            if width_says != area_says else "orientation_uncertain"
        )

    return AxisEstimate(
        angle_deg=round(angle_deg, 3),
        elongation=round(elongation, 3) if math.isfinite(elongation) else 999.0,
        centroid=(round(float(centroid[0]), 2), round(float(centroid[1]), 2)),
        narrow_end=narrow_end,
        end_width_ratio=round(ratio, 3),
        body_half_area_share=round(body_half_share, 3),
        orientation_confident=confident,
        flip_required=flip_required,
        reliable=reliable,
        reason=reason,
    )


def rotate_about_centre(
    image: np.ndarray,
    angle_deg: float,
    *,
    flags: int = cv2.INTER_LINEAR,
    border_value: Sequence[int] | int = 0,
) -> np.ndarray:
    """Rotate keeping the whole content: the canvas grows to fit.

    Rotating inside the original canvas would clip the neck of a tilted bottle,
    which is exactly the part this dataset exists to preserve.
    """
    height, width = image.shape[:2]
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += new_width / 2.0 - centre[0]
    matrix[1, 2] += new_height / 2.0 - centre[1]
    return cv2.warpAffine(
        image, matrix, (new_width, new_height),
        flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


# --------------------------------------------------------- bottle anatomy
@dataclass(frozen=True)
class ShoulderEstimate:
    """Where the shoulders are, measured on the upright mask width profile."""

    shoulder_row: int
    body_width: float
    neck_width: float
    method: str                     # "width_profile" or "fallback_fraction"
    confident: bool
    reason: str

    def record(self) -> dict[str, Any]:
        return asdict(self)


def find_shoulder(
    mask: np.ndarray,
    *,
    shoulder_width_ratio: float,
    fallback_top_fraction: float,
    min_height_fraction: float,
) -> ShoulderEstimate:
    """Locate the shoulder line of an upright bottle mask.

    The width profile of a wine bottle is flat and wide over the body, narrows
    sharply at the shoulders and stays flat and thin along the neck. The
    shoulder is taken as the topmost row whose width has already reached
    ``shoulder_width_ratio`` of the median body width, scanning downwards.

    When the profile does not look like a bottle - a mask cut in half, a bottle
    seen from far above, a sliver - the function falls back to a fixed fraction
    of the height and says so, so the crop can be flagged instead of trusted.
    """
    height, _ = mask.shape[:2]
    widths = mask.astype(bool).sum(axis=1).astype(np.float64)
    occupied = np.flatnonzero(widths > 0)
    fallback_row = max(1, int(round(height * fallback_top_fraction)))

    if occupied.size < 8:
        return ShoulderEstimate(fallback_row, 0.0, 0.0, "fallback_fraction", False, "mask_too_small")

    top, bottom = int(occupied[0]), int(occupied[-1])
    extent = bottom - top + 1
    if extent < 16:
        return ShoulderEstimate(fallback_row, 0.0, 0.0, "fallback_fraction", False, "extent_too_small")

    # The body is the lower 55% of the silhouette; its median width is the
    # reference the shoulder ratio is measured against.
    body_start = top + int(0.45 * extent)
    body = widths[body_start:bottom + 1]
    body = body[body > 0]
    if body.size < 4:
        return ShoulderEstimate(fallback_row, 0.0, 0.0, "fallback_fraction", False, "no_body_rows")
    body_width = float(np.median(body))

    # The neck is the upper 15%; used only to sanity-check that a neck exists.
    neck = widths[top:top + max(2, int(0.15 * extent))]
    neck = neck[neck > 0]
    neck_width = float(np.median(neck)) if neck.size else 0.0

    if body_width <= 0:
        return ShoulderEstimate(fallback_row, body_width, neck_width, "fallback_fraction", False, "zero_body_width")

    threshold = shoulder_width_ratio * body_width
    # Smooth the profile so JPEG fringing and a stray highlight do not create a
    # one-row spike that reads as the shoulder. The edges are replicated rather
    # than zero-padded: zero padding depresses the profile at the first and last
    # rows and drags the detected shoulder downwards whenever the silhouette
    # starts at row 0, which is the normal case for a tightly cropped mask.
    window = max(3, (extent // 40) | 1)
    kernel = np.ones(window, dtype=np.float64) / window
    half = window // 2
    padded = np.pad(widths, half, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")

    candidates = np.flatnonzero(smoothed[top:bottom + 1] >= threshold)
    if candidates.size == 0:
        return ShoulderEstimate(fallback_row, body_width, neck_width, "fallback_fraction", False, "width_never_reaches_body")
    shoulder_row = int(top + candidates[0])

    reason = "ok"
    confident = True
    if neck_width <= 0 or neck_width >= 0.9 * body_width:
        # No narrowing at all: either the neck is out of frame or the mask is
        # not a bottle. Either way the top crop is not trustworthy. Checked
        # first because it is the most diagnostic; a mask with no neck also
        # trips the two geometric checks below.
        confident, reason = False, "no_neck_narrowing"
    elif shoulder_row - top < min_height_fraction * extent:
        confident, reason = False, "shoulder_too_close_to_top"
    elif shoulder_row >= bottom - 0.2 * extent:
        confident, reason = False, "shoulder_too_low"

    if not confident:
        return ShoulderEstimate(fallback_row, body_width, neck_width, "fallback_fraction", False, reason)
    return ShoulderEstimate(shoulder_row, body_width, neck_width, "width_profile", True, reason)


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian - a cheap, comparable blur score."""
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pad_to_square(image: np.ndarray, *, value: Sequence[int] | int = 0) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(height, width)
    top = (side - height) // 2
    left = (side - width) // 2
    return cv2.copyMakeBorder(
        image, top, side - height - top, left, side - width - left,
        cv2.BORDER_CONSTANT, value=value,
    )


def limit_long_side(image: np.ndarray, max_long_side: int) -> np.ndarray:
    """Shrink only. Crops are never upscaled: fake resolution is not evidence."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_long_side or longest == 0:
        return image
    scale = max_long_side / longest
    return cv2.resize(
        image, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )

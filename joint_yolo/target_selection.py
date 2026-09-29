"""Shared confidence + vertical-axis scoring for YOLO target selection.

The product asks the user to aim at the target bottle near the vertical centre
line.  Detector confidence remains the primary signal; horizontal proximity
adds a bounded, smooth bonus.  The Y coordinate and mere intersection with a
crosshair do not affect the score.
"""

from __future__ import annotations

import math
from typing import Sequence


DEFAULT_AXIS_WEIGHT = 0.20
DEFAULT_AXIS_SIGMA = 0.20


def vertical_axis_features(
    box: Sequence[float],
    image_width: float,
    *,
    axis_x: float = 0.50,
    sigma: float = DEFAULT_AXIS_SIGMA,
) -> tuple[float, float]:
    """Return normalized horizontal distance and smooth axis proximity.

    ``axis_x`` and the returned distance are normalized to the image width.
    Proximity is one on the axis and decays as a Gaussian away from it.
    """
    if image_width <= 0:
        raise ValueError("image_width must be positive")
    if not 0.0 <= axis_x <= 1.0:
        raise ValueError("axis_x must be in the normalized range 0..1")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    center_x = (float(box[0]) + float(box[2])) * 0.5 / float(image_width)
    distance = abs(center_x - axis_x)
    proximity = math.exp(-0.5 * (distance / sigma) ** 2)
    return distance, proximity


def confidence_axis_score(
    confidence: float,
    box: Sequence[float],
    image_width: float,
    *,
    axis_x: float = 0.50,
    axis_weight: float = DEFAULT_AXIS_WEIGHT,
    sigma: float = DEFAULT_AXIS_SIGMA,
) -> tuple[float, float, float]:
    """Return selection score, axis distance and axis proximity.

    The geometric term is bounded by ``axis_weight``.  With the default 0.20
    it can correct a modest confidence advantage for a clearly lateral box,
    but cannot make a very weak central detection beat a strong detection.
    """
    if axis_weight < 0:
        raise ValueError("axis_weight must be non-negative")
    distance, proximity = vertical_axis_features(
        box,
        image_width,
        axis_x=axis_x,
        sigma=sigma,
    )
    return float(confidence) + axis_weight * proximity, distance, proximity

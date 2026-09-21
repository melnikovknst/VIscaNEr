"""Tests for the decision logic that the full pipeline cannot exercise here.

Steps 3, 6 and 7 need a segmentation checkpoint, a DINO checkpoint and a gallery
index that are not present in this checkout. Their *decisions* are pure
functions, though, and those are what go wrong silently: a coordinate frame
mismatch, a bottle picked because it was bigger, a crop turned upside down.

Run with:  python -m pytest bottle_reranker/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bottle_reranker import geometry as geo  # noqa: E402
from bottle_reranker.step3_segment import (  # noqa: E402
    STATUS_AMBIGUOUS,
    STATUS_NO_MASKS,
    STATUS_NO_OWNER,
    STATUS_SELECTED,
    Candidate,
    select_target,
)


# ------------------------------------------------------------- fixtures
def bottle_mask(
    height: int = 400,
    width: int = 160,
    *,
    neck_width: int = 24,
    body_width: int = 110,
    shoulder: int = 140,
    upside_down: bool = False,
    canvas: tuple[int, int] | None = None,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """A bottle-shaped silhouette: narrow neck on top, wide body below."""
    shape = np.zeros((height, width), np.uint8)
    centre = width // 2
    for row in range(height):
        if row < shoulder:
            half = neck_width // 2
        elif row < shoulder + 40:
            ratio = (row - shoulder) / 40.0
            half = int((neck_width + ratio * (body_width - neck_width)) // 2)
        else:
            half = body_width // 2
        shape[row, max(0, centre - half):centre + half] = 1
    if upside_down:
        shape = shape[::-1]
    if canvas is None:
        return shape
    full = np.zeros(canvas, np.uint8)
    y, x = offset
    full[y:y + height, x:x + width] = shape
    return full


def candidate(**kwargs) -> Candidate:
    defaults = dict(index=0, score=0.9, area_fraction=0.1, contains_label_centre=True,
                    label_box_coverage=0.9, solidity=0.9, bbox=(0, 0, 10, 10))
    defaults.update(kwargs)
    return Candidate(**defaults)


SELECTION = dict(require_centre=True, min_coverage=0.60, ambiguous_margin=0.15, min_solidity=0.55)


# ------------------------------------------------------------ box formats
def test_box_formats_agree_on_the_same_rectangle():
    size = (800, 600)
    expected = (100.0, 150.0, 300.0, 450.0)
    assert geo.parse_box([100, 150, 300, 450], fmt="xyxy_abs") == expected
    assert geo.parse_box([100, 150, 200, 300], fmt="xywh_abs") == expected
    assert geo.parse_box([0.25, 0.5, 0.25, 0.5], fmt="cxcywh_norm", reference_size=size) == expected
    assert geo.parse_box([0.125, 0.25, 0.375, 0.75], fmt="xyxy_norm", reference_size=size) == expected


def test_normalised_box_without_a_reference_size_is_an_error():
    # Guessing the frame here is exactly how boxes end up on the wrong bottle.
    with pytest.raises(geo.GeometryError):
        geo.parse_box([0.25, 0.5, 0.25, 0.5], fmt="cxcywh_norm")


def test_rescaling_moves_a_box_between_image_sizes():
    box = (100.0, 200.0, 300.0, 400.0)
    scaled = geo.rescale_box(box, from_size=(1000, 1000), to_size=(500, 250))
    assert scaled == (50.0, 50.0, 150.0, 100.0)


def test_a_box_from_the_wrong_frame_is_detected_not_clamped():
    # A box measured before EXIF rotation: it fits the landscape frame, not the
    # portrait one the image actually decodes to.
    box = (10.0, 10.0, 700.0, 400.0)
    assert not geo.box_inside_image(box, (600, 800))          # portrait: too wide
    assert geo.box_inside_image(box, (800, 600))              # landscape: fits
    assert geo.infer_box_reference_size(box, [(600, 800), (800, 600)]) == (800, 600)
    assert geo.infer_box_reference_size((0.0, 0.0, 9999.0, 9999.0), [(600, 800)]) is None


# ---------------------------------------------------------- mask measures
def test_coverage_is_measured_against_the_box_not_the_mask():
    mask = np.zeros((100, 100), np.uint8)
    mask[40:60, 40:60] = 1                                    # 20x20 mask
    assert geo.mask_box_coverage(mask, (40, 40, 60, 60)) == pytest.approx(1.0)
    assert geo.mask_box_coverage(mask, (40, 40, 80, 60)) == pytest.approx(0.5)
    assert geo.mask_box_coverage(mask, (0, 0, 20, 20)) == pytest.approx(0.0)


def test_holes_in_a_silhouette_are_measured():
    mask = np.zeros((200, 200), np.uint8)
    mask[20:180, 40:160] = 1
    assert geo.mask_hole_fraction(mask) == pytest.approx(0.0, abs=1e-6)
    mask[80:120, 80:120] = 0                                  # a neighbour in front
    assert geo.mask_hole_fraction(mask) > 0.05


def test_border_contact_is_reported_per_side():
    mask = np.zeros((100, 100), np.uint8)
    mask[0:50, 20:40] = 1
    touching = geo.mask_touches_border(mask, margin=2)
    assert touching["top"] and not touching["bottom"]
    assert not touching["left"] and not touching["right"]


# ------------------------------------------------------- axis orientation
def test_an_upright_bottle_is_not_rotated():
    axis = geo.estimate_axis(bottle_mask(), min_elongation=1.8, max_abs_angle_deg=35.0,
                             min_end_width_ratio=1.35)
    assert axis.reliable
    assert abs(axis.angle_deg) < 1.0
    assert axis.orientation_confident
    assert not axis.flip_required


def test_an_upside_down_bottle_is_flipped():
    axis = geo.estimate_axis(bottle_mask(upside_down=True), min_elongation=1.8,
                             max_abs_angle_deg=35.0, min_end_width_ratio=1.35)
    assert axis.orientation_confident
    assert axis.flip_required


def test_a_bottle_with_its_reflection_is_never_flipped():
    """The bug this guard exists for.

    A catalog photograph shot on a reflective surface has a mirrored bottle
    below the real one. That silhouette is narrow at BOTH ends, so an end-width
    test alone picks whichever end happens to be thinner and turns the bottle
    upside down. Two catalog references were rendered inverted before the area
    balance test was added.
    """
    upright = bottle_mask(height=300)
    # The mirrored copy fades towards its tip, so its neck reads THINNER than the
    # real one. End width alone therefore calls the bottom end the neck and asks
    # for a flip; the area balance is symmetric and vetoes it.
    reflection = bottle_mask(height=300, neck_width=16)[::-1]
    combined = np.vstack([upright, reflection])
    axis = geo.estimate_axis(combined, min_elongation=1.8, max_abs_angle_deg=35.0,
                             min_end_width_ratio=1.35)
    assert axis.end_width_ratio >= 1.35                       # end width alone would flip
    assert axis.body_half_area_share < 0.56                   # area balance says no
    assert not axis.orientation_confident
    assert not axis.flip_required
    assert axis.reason == "orientation_uncertain_signals_disagree"


def test_a_tilted_bottle_reports_its_angle():
    import cv2

    mask = bottle_mask(canvas=(600, 600), offset=(80, 200))
    rotated = geo.rotate_about_centre(mask, 12.0, flags=cv2.INTER_NEAREST, border_value=0)
    axis = geo.estimate_axis(rotated, min_elongation=1.8, max_abs_angle_deg=35.0,
                             min_end_width_ratio=1.35)
    assert axis.reliable
    assert abs(abs(axis.angle_deg) - 12.0) < 3.0


def test_a_squat_shape_is_not_deskewed():
    """A bag-in-box carton has no meaningful principal axis - do not rotate it."""
    mask = np.zeros((200, 180), np.uint8)
    mask[20:180, 20:160] = 1
    axis = geo.estimate_axis(mask, min_elongation=1.8, max_abs_angle_deg=35.0,
                             min_end_width_ratio=1.35)
    assert not axis.reliable
    assert "elongation" in axis.reason


# ---------------------------------------------------------------- shoulder
def test_the_shoulder_is_found_on_the_width_profile():
    mask = bottle_mask(shoulder=140)
    estimate = geo.find_shoulder(mask, shoulder_width_ratio=0.85,
                                 fallback_top_fraction=0.38, min_height_fraction=0.12)
    assert estimate.method == "width_profile"
    assert estimate.confident
    assert 140 <= estimate.shoulder_row <= 190          # at or just below the shoulder
    assert estimate.neck_width < estimate.body_width


def test_a_mask_with_no_neck_falls_back_and_says_so():
    mask = np.zeros((300, 120), np.uint8)
    mask[:, 10:110] = 1                                  # a plain rectangle
    estimate = geo.find_shoulder(mask, shoulder_width_ratio=0.85,
                                 fallback_top_fraction=0.38, min_height_fraction=0.12)
    assert estimate.method == "fallback_fraction"
    assert not estimate.confident
    assert estimate.reason == "no_neck_narrowing"


# -------------------------------------------------------- target selection
def test_the_mask_owning_the_label_wins():
    winner = candidate(index=1, label_box_coverage=0.95, area_fraction=0.05, score=0.4)
    loser = candidate(index=0, label_box_coverage=0.10, area_fraction=0.60, score=0.99,
                      contains_label_centre=False)
    chosen, status, _ = select_target([loser, winner], **SELECTION)
    assert status == STATUS_SELECTED
    assert chosen.index == 1


def test_a_bigger_more_confident_bottle_never_steals_the_label():
    """The rule the task is explicit about: size and score must not decide."""
    small_owner = candidate(index=0, label_box_coverage=0.80, area_fraction=0.02, score=0.30)
    big_rival = candidate(index=1, label_box_coverage=0.65, area_fraction=0.55, score=0.99)
    chosen, status, _ = select_target([small_owner, big_rival], **SELECTION)
    assert status == STATUS_SELECTED
    assert chosen.index == 0


def test_two_masks_sharing_the_label_are_flagged_not_guessed():
    a = candidate(index=0, label_box_coverage=0.72)
    b = candidate(index=1, label_box_coverage=0.66)        # within the 0.15 margin
    chosen, status, detail = select_target([a, b], **SELECTION)
    assert chosen is None
    assert status == STATUS_AMBIGUOUS
    assert detail["runner_up_coverage"] == pytest.approx(0.66)


def test_a_label_between_two_bottles_has_no_owner():
    a = candidate(index=0, contains_label_centre=False, label_box_coverage=0.40)
    b = candidate(index=1, contains_label_centre=False, label_box_coverage=0.35)
    chosen, status, detail = select_target([a, b], **SELECTION)
    assert chosen is None
    assert status == STATUS_NO_OWNER
    assert detail["best_coverage_seen"] == pytest.approx(0.40)


def test_weak_coverage_is_rejected_even_when_the_centre_is_inside():
    thin = candidate(index=0, label_box_coverage=0.30)
    chosen, status, _ = select_target([thin], **SELECTION)
    assert chosen is None
    assert status == STATUS_NO_OWNER


def test_a_sliver_mask_is_dropped_before_selection():
    sliver = candidate(index=0, solidity=0.20, label_box_coverage=0.99)
    chosen, status, detail = select_target([sliver], **SELECTION)
    assert chosen is None
    assert detail["candidates_after_solidity"] == 0


def test_no_masks_is_its_own_status():
    chosen, status, _ = select_target([], **SELECTION)
    assert chosen is None
    assert status == STATUS_NO_MASKS


# ------------------------------------------------------------ split hashing
def test_the_dino_identity_holdout_is_reproduced_exactly():
    """Must stay byte-identical to dinov3_retrieval._stable_unit_interval.

    If this drifts, identities reserved for generalisation start leaking into
    reranker training and every generalisation number becomes meaningless.
    """
    import hashlib

    from bottle_reranker.common import stable_unit_interval

    for slug in ("aligote-barrel-2024", "abrau-dyurso-abrau-estates-beloe-shardone-suhoe-12", ""):
        digest = hashlib.sha256(f"42:{slug}".encode("utf-8")).digest()
        assert stable_unit_interval(slug, 42) == int.from_bytes(digest[:8], "big") / float(2**64)


def test_provenance_groups_merge_transitively():
    from bottle_reranker.step5_splits import DisjointSet

    union = DisjointSet()
    union.union("a", "b")
    union.union("b", "c")
    union.union("x", "y")
    assert union.find("a") == union.find("c")
    assert union.find("a") != union.find("x")

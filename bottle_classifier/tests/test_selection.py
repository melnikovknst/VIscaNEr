from types import SimpleNamespace

import torch

from build_bottle_classifier_dataset import choose_output_policy, select_target_bottle


def result(boxes, confidences):
    payload = SimpleNamespace(
        xyxy=torch.tensor(boxes, dtype=torch.float32),
        conf=torch.tensor(confidences, dtype=torch.float32),
        cls=torch.zeros(len(boxes), dtype=torch.float32),
    )
    payload.__len__ = lambda self: len(boxes)
    return SimpleNamespace(boxes=BoxPayload(payload, len(boxes)))


class BoxPayload:
    def __init__(self, payload, length):
        self.xyxy = payload.xyxy
        self.conf = payload.conf
        self.cls = payload.cls
        self.length = length

    def __len__(self):
        return self.length


def test_label_ownership_beats_unrelated_high_confidence_box():
    prediction = result(
        boxes=[[0, 0, 40, 100], [50, 0, 100, 100]],
        confidences=[0.99, 0.70],
    )
    selected, context = select_target_bottle(
        prediction,
        label_box=(60, 40, 90, 70),
        minimum_confidence=0.05,
        minimum_label_coverage=0.01,
        ambiguity_margin=0.12,
        duplicate_iou=0.80,
    )
    assert selected is not None
    assert selected["box"] == (50.0, 0.0, 100.0, 100.0)
    assert context["eligible"] == 1
    assert not context["ambiguous"]


def test_distinct_close_owners_are_marked_ambiguous():
    prediction = result(
        boxes=[[10, 0, 70, 100], [40, 0, 100, 100]],
        confidences=[0.82, 0.81],
    )
    selected, context = select_target_bottle(
        prediction,
        label_box=(48, 45, 62, 62),
        minimum_confidence=0.05,
        minimum_label_coverage=0.01,
        ambiguity_margin=0.12,
        duplicate_iou=0.80,
    )
    assert selected is not None
    assert context["eligible"] == 2
    assert context["ambiguous"]


def test_duplicate_boxes_do_not_create_false_ambiguity():
    prediction = result(
        boxes=[[10, 0, 90, 100], [11, 1, 89, 99]],
        confidences=[0.90, 0.89],
    )
    selected, context = select_target_bottle(
        prediction,
        label_box=(40, 40, 60, 60),
        minimum_confidence=0.05,
        minimum_label_coverage=0.01,
        ambiguity_margin=0.12,
        duplicate_iou=0.80,
    )
    assert selected is not None
    assert context["eligible"] == 2
    assert not context["ambiguous"]


def test_low_confidence_uses_original_and_remains_trainable():
    status, image_mode = choose_output_policy(
        0.7499,
        ambiguous=False,
        vertically_truncated=True,
        crop_confidence_threshold=0.75,
    )
    assert status == "low_confidence"
    assert image_mode == "original_image"


def test_threshold_is_inclusive_for_yolo_crop():
    status, image_mode = choose_output_policy(
        0.75,
        ambiguous=False,
        vertically_truncated=False,
        crop_confidence_threshold=0.75,
    )
    assert status == "successful"
    assert image_mode == "yolo_crop"


def test_ambiguous_owner_is_never_admitted_to_training():
    status, image_mode = choose_output_policy(
        0.40,
        ambiguous=True,
        vertically_truncated=False,
        crop_confidence_threshold=0.75,
    )
    assert status == "ambiguous"
    assert image_mode == "original_image"

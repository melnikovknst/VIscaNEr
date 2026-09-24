from types import SimpleNamespace

import torch
from PIL import Image

from infer_wine import (
    crop_with_policy,
    fuse_gallery_similarities,
    select_target_detection,
    select_target_detections,
)


class FakeBoxes:
    def __init__(self, boxes, confidences):
        self.xyxy = torch.tensor(boxes, dtype=torch.float32)
        self.conf = torch.tensor(confidences, dtype=torch.float32)
        self.cls = torch.zeros(len(boxes), dtype=torch.float32)

    def __len__(self):
        return len(self.xyxy)


def test_crosshair_selects_containing_bottle_over_higher_confidence_neighbour():
    result = SimpleNamespace(
        boxes=FakeBoxes(
            boxes=[[0, 0, 40, 100], [45, 0, 100, 100]],
            confidences=[0.99, 0.80],
        )
    )
    selected, candidates = select_target_detection(
        result,
        image_width=100,
        image_height=100,
        candidate_confidence=0.05,
        crosshair_x=0.75,
        crosshair_y=0.50,
    )
    assert selected is not None
    assert selected["box"] == (45.0, 0.0, 100.0, 100.0)
    assert len(candidates) == 2


def test_two_equally_central_distinct_bottles_are_ambiguous():
    result = SimpleNamespace(
        boxes=FakeBoxes(
            boxes=[[0, 0, 45, 100], [55, 0, 100, 100]],
            confidences=[0.91, 0.90],
        )
    )
    selected, candidates, context = select_target_detections(
        result,
        image_width=100,
        image_height=100,
        candidate_confidence=0.05,
        crosshair_x=0.50,
        crosshair_y=0.50,
        ambiguity_confidence=0.75,
    )
    assert len(candidates) == 2
    assert len(selected) == 2
    assert context["ambiguous"]


def test_bottle_containing_crosshair_does_not_trigger_second_bottle():
    result = SimpleNamespace(
        boxes=FakeBoxes(
            boxes=[[25, 0, 75, 100], [76, 0, 100, 100]],
            confidences=[0.80, 0.99],
        )
    )
    selected, _, context = select_target_detections(
        result,
        image_width=100,
        image_height=100,
        candidate_confidence=0.05,
        crosshair_x=0.50,
        crosshair_y=0.50,
        ambiguity_confidence=0.75,
    )
    assert len(selected) == 1
    assert selected[0]["box"] == (25.0, 0.0, 75.0, 100.0)
    assert not context["ambiguous"]


def test_duplicate_yolo_boxes_do_not_trigger_ambiguity():
    result = SimpleNamespace(
        boxes=FakeBoxes(
            boxes=[[10, 0, 90, 100], [11, 1, 89, 99]],
            confidences=[0.90, 0.89],
        )
    )
    selected, _, context = select_target_detections(
        result,
        image_width=100,
        image_height=100,
        candidate_confidence=0.05,
        crosshair_x=0.50,
        crosshair_y=0.50,
        ambiguity_confidence=0.75,
    )
    assert len(selected) == 1
    assert not context["ambiguous"]
    assert context["pair_iou"] > 0.80


def test_low_confidence_second_box_does_not_trigger_ambiguity():
    result = SimpleNamespace(
        boxes=FakeBoxes(
            boxes=[[0, 0, 45, 100], [55, 0, 100, 100]],
            confidences=[0.91, 0.74],
        )
    )
    selected, _, context = select_target_detections(
        result,
        image_width=100,
        image_height=100,
        candidate_confidence=0.05,
        crosshair_x=0.50,
        crosshair_y=0.50,
        ambiguity_confidence=0.75,
    )
    assert len(selected) == 1
    assert not context["ambiguous"]


def test_two_bottle_scores_are_fused_by_best_similarity_per_wine():
    similarities = torch.tensor([[0.80, 0.20, 0.70], [0.30, 0.90, 0.60]])
    fused, winning_bottles = fuse_gallery_similarities(similarities)
    assert torch.allclose(fused, torch.tensor([0.80, 0.90, 0.70]))
    assert winning_bottles.tolist() == [0, 1, 0]


def test_confidence_below_threshold_preserves_original_image():
    image = Image.new("RGB", (100, 200))
    selected = {"confidence": 0.7499, "box": (20, 20, 80, 180)}
    output, mode, box, confidence = crop_with_policy(image, selected, 0.75, 0.0)
    assert output.size == image.size
    assert mode == "original_low_confidence"
    assert box == [20, 20, 80, 180]
    assert confidence == 0.7499


def test_confidence_threshold_is_inclusive_for_bottle_crop():
    image = Image.new("RGB", (100, 200))
    selected = {"confidence": 0.75, "box": (20, 20, 80, 180)}
    output, mode, box, confidence = crop_with_policy(image, selected, 0.75, 0.0)
    assert output.size == (60, 160)
    assert mode == "yolo_bottle_crop"
    assert box == [20, 20, 80, 180]
    assert confidence == 0.75


def test_missing_detection_falls_back_to_original_image():
    image = Image.new("RGB", (100, 200))
    output, mode, box, confidence = crop_with_policy(image, None, 0.75, 0.06)
    assert output.size == image.size
    assert mode == "original_no_detection"
    assert box is None
    assert confidence is None

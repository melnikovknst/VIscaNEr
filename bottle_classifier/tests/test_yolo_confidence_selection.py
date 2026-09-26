from types import SimpleNamespace

import pytest
import torch

from infer_wine_labels import select_central_label
from run_bulk_crop import extract_target_detection


class FakeBoxes:
    def __init__(self, boxes, confidences):
        self.xyxy = torch.tensor(boxes, dtype=torch.float32)
        self.conf = torch.tensor(confidences, dtype=torch.float32)
        self.cls = torch.zeros(len(boxes), dtype=torch.float32)

    def __len__(self):
        return len(self.xyxy)


def fake_result(boxes, confidences):
    return SimpleNamespace(boxes=FakeBoxes(boxes, confidences))


def test_bulk_crop_confidence_outranks_crosshair_intersection():
    selected, candidates = extract_target_detection(
        fake_result(
            boxes=[[45, 45, 55, 55], [45, 60, 65, 80]],
            confidences=[0.08, 0.79],
        ),
        image_width=100,
        image_height=100,
        candidate_confidence=0.05,
        crosshair_x=0.50,
        crosshair_y=0.50,
    )
    assert len(candidates) == 2
    assert selected is not None
    assert selected["confidence"] == pytest.approx(0.79)


def test_label_inference_confidence_outranks_image_centre():
    selected, count = select_central_label(
        fake_result(
            boxes=[[45, 45, 55, 55], [45, 60, 65, 80]],
            confidences=[0.08, 0.79],
        ),
        width=100,
        height=100,
    )
    assert count == 2
    assert selected is not None
    assert selected["confidence"] == pytest.approx(0.79)

from joint_yolo.infer import select_bottle_candidates, select_label_candidates


def detection(confidence, box):
    return {"confidence": confidence, "box": box, "class_id": 1}


def test_low_confidence_crosshair_hit_does_not_beat_confident_detection():
    labels = [
        detection(0.08, (45.0, 45.0, 55.0, 55.0)),
        detection(0.79, (45.0, 60.0, 65.0, 80.0)),
    ]
    selected, ambiguous = select_label_candidates(
        labels,
        crosshair=(50.0, 50.0),
        diagonal=141.42,
        ambiguity_margin=0.06,
        image_width=100,
    )
    assert selected[0]["confidence"] == 0.79
    assert not ambiguous


def test_close_confidence_and_geometry_can_return_two_candidates():
    labels = [
        detection(0.81, (20.0, 20.0, 45.0, 80.0)),
        detection(0.80, (55.0, 20.0, 80.0, 80.0)),
    ]
    selected, ambiguous = select_label_candidates(
        labels,
        crosshair=(50.0, 50.0),
        diagonal=141.42,
        ambiguity_margin=0.06,
        image_width=100,
    )
    assert len(selected) == 2
    assert ambiguous


def test_duplicate_boxes_do_not_create_two_candidates():
    labels = [
        detection(0.81, (20.0, 20.0, 80.0, 80.0)),
        detection(0.80, (21.0, 21.0, 79.0, 79.0)),
    ]
    selected, ambiguous = select_label_candidates(
        labels,
        crosshair=(50.0, 50.0),
        diagonal=141.42,
        ambiguity_margin=0.06,
        image_width=100,
    )
    assert len(selected) == 1
    assert not ambiguous


def test_vertical_axis_beats_modest_side_confidence_advantage():
    labels = [
        detection(0.70, (40.0, 5.0, 60.0, 30.0)),
        detection(0.80, (75.0, 70.0, 95.0, 95.0)),
    ]
    selected, ambiguous = select_label_candidates(
        labels,
        crosshair=(50.0, 50.0),
        diagonal=141.42,
        ambiguity_margin=0.06,
        image_width=100,
    )
    assert selected[0]["box"] == (40.0, 5.0, 60.0, 30.0)
    assert not ambiguous


def test_vertical_position_does_not_change_ranking():
    labels = [
        detection(0.70, (40.0, 0.0, 60.0, 20.0)),
        detection(0.71, (40.0, 80.0, 60.0, 100.0)),
    ]
    selected, _ = select_label_candidates(
        labels,
        crosshair=(50.0, 5.0),
        diagonal=141.42,
        ambiguity_margin=0.001,
        image_width=100,
    )
    assert selected[0]["confidence"] == 0.71


def test_bottle_selector_uses_the_same_bounded_axis_policy():
    bottles = [
        {"confidence": 0.70, "box": (40.0, 5.0, 60.0, 95.0), "class_id": 0},
        {"confidence": 0.80, "box": (75.0, 5.0, 95.0, 95.0), "class_id": 0},
    ]
    selected, ambiguous = select_bottle_candidates(bottles, image_width=100)
    assert selected[0]["box"] == (40.0, 5.0, 60.0, 95.0)
    assert not ambiguous

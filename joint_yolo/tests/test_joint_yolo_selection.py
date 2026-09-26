from joint_yolo.infer import select_label_candidates


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
    )
    assert len(selected) == 1
    assert not ambiguous

import pytest
from PIL import Image, ImageDraw

from backend.config import Settings
from scripts.build_store_shelves import PADDING_COLOR, frame
from scripts.evaluate_saved_run import evaluate, predictions_by_id, served_status
from scripts.visualize_relabeled import outcome, thumb


def settings():
    return Settings(_env_file=None, min_similarity=0.6, min_margin=0.04, min_suggest_similarity=0.5)


def test_top_shelf_keeps_label_below_point_and_pads_missing_area():
    source = Image.new("RGB", (1200, 1600), "white")
    # Marked centre near the top; the lower label used to fall outside the crop.
    ImageDraw.Draw(source).rectangle((285, 65, 435, 245), fill="red")
    shot = frame(source, 30, 5, 18)
    assert shot.size == (480, 640)
    assert shot.getpixel((240, 320)) == (255, 0, 0)
    assert shot.getpixel((240, 470)) == (255, 0, 0)
    assert shot.getpixel((240, 0)) == PADDING_COLOR


@pytest.mark.parametrize("x,y", [(0, 0), (100, 0), (0, 100), (100, 100), (50, 50)])
def test_frame_size_is_independent_of_edge_distance(x, y):
    shot = frame(Image.new("RGB", (1200, 1600)), x, y, 18)
    assert shot.size == (480, 640)


def truth(qid, known=True, scored=True):
    return {"query_id": qid, "scored": str(scored), "in_catalog": str(known),
            "accepted_slugs": "a" if known else "", "source": "one-shelf.jpg"}


def test_open_set_metrics_separate_wrong_cards_choices_and_rejections():
    rows = [truth("hit"), truth("wrong"), truth("decline"), truth("unknown", False),
            truth("choice", False), truth("reject", False), truth("excluded", False, False)]
    preds = {"hit": [("a", .8), ("b", .6)], "wrong": [("b", .8), ("a", .6)],
             "decline": [("a", .45), ("b", .4)], "unknown": [("b", .8), ("a", .6)],
             "choice": [("b", .55), ("a", .4)], "reject": []}
    report = evaluate(rows, preds, settings())
    assert report["samples"] == 6
    assert report["excluded"] == 1
    assert report["ranking"]["hits"] == {"top1": 2, "top2": 3, "top3": 3, "top5": 3}
    assert report["served"]["correct_cards"] == 1
    assert report["served"]["wrong_cards"] == 2
    assert report["served"]["wrong_cards_in_catalog"] == 1
    assert report["served"]["precision"] == .3333
    assert report["served"]["f1"] == .3333
    assert report["served"]["coverage"] == .5
    for status in ("matched", "uncertain", "not_found"):
        assert report["out_of_catalog_behavior"][status] == {"count": 1, "rate": .3333}


def test_missing_predictions_cannot_inflate_metrics():
    with pytest.raises(ValueError, match="Missing predictions"):
        evaluate([truth("missing")], {}, settings())
    with pytest.raises(ValueError, match="Duplicate query_id"):
        evaluate([truth("a"), truth("a")], {}, settings())
    with pytest.raises(ValueError, match="Duplicate prediction"):
        predictions_by_id({"results": [{"source": "a.jpg", "predictions": []}] * 2})


@pytest.mark.parametrize("ranked,expected", [
    ([], "not_found"), ([("a", .9)], "uncertain"),
    ([("a", .6), ("b", .5)], "matched"),
    ([("a", .8), ("b", .79)], "uncertain"),
    ([("a", .5), ("b", .3)], "uncertain"),
    ([("a", .49), ("b", .3)], "not_found"),
])
def test_saved_decision_matches_backend(tmp_path, ranked, expected):
    from backend.catalog import Catalog
    from backend.main import resolve_prediction
    from backend.schemas import Candidate, Prediction

    path = tmp_path / "catalog.csv"
    path.write_text("slug,name,category,color,region,grapes,description,winery\n"
                    "a,A,,,,,,\nb,B,,,,,,\n", encoding="utf-8")
    config = settings().model_copy(update={"catalog_path": path})
    catalog = Catalog(config)
    prediction = Prediction(candidates=[Candidate(slug=s, similarity=v) for s, v in ranked],
                            model_version="test", ranked=True)
    assert resolve_prediction(prediction, catalog, config, 0).status == expected
    assert served_status(ranked, config) == expected


def test_viewer_does_not_count_unknown_choices_as_rejections():
    row = truth("unknown", False)
    assert outcome(row, [], "uncertain") == "notcat_choice"
    assert outcome(row, [], "not_found") == "notcat_ok"
    assert outcome(row, [], "matched") == "notcat_wrong"


def test_viewer_replaces_stale_thumbnail(tmp_path):
    path = tmp_path / "thumb.jpg"
    thumb(Image.new("RGB", (40, 40), "red"), path, 40)
    thumb(Image.new("RGB", (80, 80), "blue"), path, 80)
    with Image.open(path) as image:
        assert image.size == (80, 80)
        assert image.getpixel((0, 0))[2] > 250

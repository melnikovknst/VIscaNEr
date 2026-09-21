import json

import pytest

from backend.metrics import read_metrics
from scripts.evaluate_api import compute_metrics


def row(truth, predicted, top5, error=False):
    result = {"true_slug": truth, "predicted_slug": predicted, "top5": top5,
              "model_version": "v1", "elapsed_ms": 10, "status": "matched"}
    if error:
        result["error"] = "Timeout"
    return result


def test_metrics_distinguish_top5_recall_f1_and_failures():
    report = compute_metrics([
        row("a", "a", ["a", "b"]),  # TP
        row("b", "c", ["c", "b"]),  # FP + FN, but top-5 hit
        row("a", None, [], error=True),  # failure still in denominator
        row(None, None, []),  # correct open-set refusal
    ])
    assert report["errors"] == 1
    assert report["accuracy"] == .5
    assert report["f1_top1"] == pytest.approx(2 / 5)
    assert report["f1_top5"] == pytest.approx(4 / 7)
    assert report["recall_at_5"] == pytest.approx(2 / 3)
    assert report["coverage"] == .5


def test_reports_are_null_until_actually_evaluated(tmp_path):
    path = tmp_path / "evaluation.json"
    assert read_metrics(path)["f1_top1"] is None
    path.write_text('{"f1_top1": 99}', encoding="utf-8")
    assert read_metrics(path)["source"] == "invalid_report"
    report = compute_metrics([row("a", "a", ["a", "b"])])
    path.write_text(json.dumps(report), encoding="utf-8")
    assert read_metrics(path)["f1_top1"] == 1
    assert read_metrics(path)["model_versions"] == ["v1"]

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from five_stream_transformer.ocr import fill_ocr_text


def test_easyocr_uses_attached_weights_and_disables_downloads(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in ("craft_mlt_25k.pth", "cyrillic_g2.pth"):
        (weights / name).write_bytes(b"present")

    captured = {}

    class FakeReader:
        def __init__(self, languages, **kwargs):
            captured["languages"] = languages
            captured.update(kwargs)

        def readtext(self, path, **kwargs):
            captured["path"] = path
            return ["Массандра", "2023"]

    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=FakeReader))
    frame = pd.DataFrame(
        [{
            "sample_id": "sample-1",
            "source_group": "main_45k",
            "label_path": str(tmp_path / "label.jpg"),
            "ocr_text": "",
        }]
    )
    result = fill_ocr_text(
        frame,
        tmp_path / "cache.csv",
        {"main_45k"},
        weights,
        gpu=False,
        cache_every=1,
    )

    assert result.loc[0, "ocr_text"] == "Массандра 2023"
    assert captured["languages"] == ["ru", "en"]
    assert captured["model_storage_directory"] == str(weights.resolve())
    assert captured["download_enabled"] is False
    assert (tmp_path / "cache.csv").is_file()


def test_easyocr_fails_before_reader_when_attached_weight_is_missing(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "craft_mlt_25k.pth").write_bytes(b"present")
    frame = pd.DataFrame(
        [{
            "sample_id": "sample-1",
            "source_group": "main_45k",
            "label_path": str(tmp_path / "label.jpg"),
            "ocr_text": "",
        }]
    )

    with pytest.raises(FileNotFoundError, match="cyrillic_g2.pth"):
        fill_ocr_text(frame, tmp_path / "cache.csv", {"main_45k"}, weights, gpu=False)

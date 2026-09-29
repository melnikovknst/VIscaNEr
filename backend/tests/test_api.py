import io
import zipfile
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import providers
from backend.config import Settings
from backend.main import create_app, decode_image
from backend.schemas import Candidate, Prediction

CATALOG = ("slug,name,category,color,region,grapes,description,winery\n"
           "red,Первое красное,Красное,Рубиновый,Кубань,Каберне,Вишня,Винодельня А\n"
           "white,Второе белое,Белое,Соломенный,Крым,Рислинг,Яблоко,Винодельня Б\n"
           "rose,Третье розовое,Розовое,Розовый,Дон,Пино,Ягоды,Винодельня В\n")


@pytest.fixture
def settings(tmp_path):
    archive = tmp_path / "catalog.zip"
    bottle = io.BytesIO()
    Image.new("RGBA", (60, 100), (0, 0, 0, 0)).save(bottle, format="WEBP")
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("catalog.csv", CATALOG)
        out.writestr("images/red.webp", bottle.getvalue())
    return Settings(_env_file=None, catalog_archive=archive, data_dir=tmp_path / "data", max_upload_mb=1)


def photo(fmt="JPEG", size=(120, 180)):
    out = io.BytesIO()
    Image.new("RGB", size, "green").save(out, format=fmt)
    return out.getvalue()


@dataclass
class StubProvider:
    response: Prediction
    received: Image.Image | None = None

    def predict(self, image):
        self.received = image
        return self.response


def prediction(first=0.91, second=0.05, slug="red", abstain=False):
    return Prediction(candidates=[Candidate(slug=slug, confidence=first), Candidate(slug="white", confidence=second)],
                      model_version="test-v1", abstain=abstain)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, StubProvider(prediction()))) as client:
        yield client


def test_catalog_search_pagination_and_details(client):
    assert client.get("/api/health").json()["catalog_count"] == 3
    assert client.get("/api/catalog", params={"q": "КУБАНЬ каберне"}).json()["items"][0]["slug"] == "red"
    assert client.get("/api/catalog", params={"category": "Белое"}).json()["total"] == 1
    assert len(client.get("/api/catalog?limit=1&offset=1").json()["items"]) == 1
    assert client.get("/api/catalog?offset=-1").status_code == 422
    assert client.get("/api/catalog/missing").status_code == 404


def test_catalog_images_keep_transparency(client):
    response = client.get("/api/catalog/red/image")
    assert response.status_code == 200 and response.headers["content-type"] == "image/webp"
    assert Image.open(io.BytesIO(response.content)).mode == "RGBA"
    assert client.get("/api/catalog/white/image").status_code == 404


def test_upload_roundtrip_and_history(client):
    result = client.post("/api/scan", files={"file": ("label.jpg", photo(), "image/jpeg")}).json()
    assert result["status"] == "matched" and result["wine"]["slug"] == "red"
    assert result["confidence"] == .91 and result["margin"] == pytest.approx(.86)
    assert len(client.get("/api/history").json()["items"]) == 1


def test_organisers_eval_endpoint_takes_the_image_field(client):
    # participant_test.sh: POST multipart "image" to /v1/eval/predict, top-1 {"slug": ...}.
    response = client.post("/v1/eval/predict", files={"image": ("q.jpg", photo(), "image/jpeg")})
    assert response.status_code == 200 and response.json() == {"slug": "red"}
    assert client.post("/predict", files={"file": ("q.jpg", photo())}).json() == {"slug": "red"}
    assert client.post("/v1/eval/predict", files={"other": ("q.jpg", photo())}).status_code == 422
    assert client.get("/api/history").json() == {"items": []}  # Evaluation doesn't pollute history.


@pytest.mark.parametrize("confidences,expected", [
    ((.50, .45), "matched"),     # no margin gate: softmax already weighs the runner-up
    ((.30, .10), "uncertain"),   # 0.20..0.45: offered as a choice, /predict says null
    ((.15, .10), "not_found"),
])
def test_confidence_thresholds_decide_between_answer_choice_and_null(settings, confidences, expected):
    with TestClient(create_app(settings, StubProvider(prediction(*confidences)))) as client:
        result = client.post("/api/scan", files={"file": ("a.jpg", photo())}).json()
        assert result["status"] == expected
        assert (result["wine"] is not None) == (expected == "matched")
        assert client.post("/v1/eval/predict", files={"image": ("a.jpg", photo())}).json() == {
            "slug": "red" if expected == "matched" else None}


def test_optional_margin_gate(settings):
    settings.min_margin = .1
    with TestClient(create_app(settings, StubProvider(prediction(.50, .45)))) as client:
        assert client.post("/api/scan", files={"file": ("a.jpg", photo())}).json()["status"] == "uncertain"


def test_repeated_slugs_keep_first_position(settings):
    response = Prediction(candidates=[Candidate(slug="white", confidence=.6), Candidate(slug="red", confidence=.3),
                                      Candidate(slug="white", confidence=.1)])
    with TestClient(create_app(settings, StubProvider(response))) as client:
        result = client.post("/api/scan", files={"file": ("a.jpg", photo())}).json()
        assert [c["wine"]["slug"] for c in result["candidates"]] == ["white", "red"]


def test_explicit_abstention_and_empty_candidates(settings):
    for result in [prediction(abstain=True), Prediction(), Prediction(candidates=[Candidate(slug="red", confidence=.1)])]:
        with TestClient(create_app(settings, StubProvider(result))) as client:
            assert client.post("/v1/eval/predict", files={"image": ("a.jpg", photo())}).json() == {"slug": None}


def test_unknown_top_result_not_silently_promoted(settings):
    with TestClient(create_app(settings, StubProvider(prediction(slug="not-in-catalog")))) as client:
        assert client.post("/v1/eval/predict", files={"image": ("a.jpg", photo())}).status_code == 503


def test_demo_never_claims_to_recognize_upload(settings):
    settings.model_provider = "demo"
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["model_ready"] is False
        result = client.post("/api/demo").json()
        assert result["status"] == "demo" and result["confidence"] is None and result["candidates"] == []
        assert client.post("/v1/eval/predict", files={"image": ("a.jpg", photo())}).status_code == 503


def test_missing_weights_keep_catalog_available(settings, monkeypatch):
    def missing(self, settings):
        raise FileNotFoundError("models/stage2c/manual_stage2c_best.pt")
    monkeypatch.setattr(providers.FiveStreamProvider, "__init__", missing)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["model_status"] == "error"
        assert client.get("/api/catalog").status_code == 200
        assert client.post("/v1/eval/predict", files={"image": ("a.jpg", photo())}).status_code == 503


def test_corrupted_unsupported_small_and_large_images(client):
    assert client.post("/api/scan", files={"file": ("a.jpg", b"not an image")}).status_code == 422
    assert client.post("/api/scan", files={"file": ("a.gif", photo("GIF"))}).status_code == 415
    assert client.post("/api/scan", files={"file": ("a.jpg", photo(size=(10, 10)))}).status_code == 422
    assert client.post("/api/scan", files={"file": ("a.jpg", b"x" * (1024 * 1024 + 1))}).status_code == 413
    assert client.post("/api/scan", content=b"x" * (2 * 1024 * 1024)).status_code == 413
    assert client.post("/api/scan").status_code == 422


def test_limits_apply_to_chunked_bodies(client):
    def chunks():
        for _ in range(20):
            yield b"x" * 65536
    assert client.post("/v1/eval/predict", content=chunks()).status_code == 413


def test_orientation_and_resolution_normalization(settings):
    image = Image.new("RGB", (3000, 1000), "red")
    exif = image.getexif()
    exif[274] = 6  # Rotate clockwise before resizing.
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    result = decode_image(buffer.getvalue(), settings)
    assert result.mode == "RGB" and result.height == 2048 and result.width < result.height


def test_history_cookie_isolation_and_clear(settings):
    app = create_app(settings, StubProvider(prediction()))
    with TestClient(app) as first:
        response = first.post("/api/demo")
        assert "HttpOnly" in response.headers["set-cookie"]
        assert "SameSite=lax" in response.headers["set-cookie"]
        with TestClient(app) as second:
            assert second.get("/api/history").json() == {"items": []}
        assert len(first.get("/api/history").json()["items"]) == 1
        assert first.delete("/api/history").status_code == 204
        assert first.get("/api/history").json() == {"items": []}


def test_history_limit_and_persistence(settings):
    settings.history_limit = 2
    with TestClient(create_app(settings, StubProvider(prediction()))) as client:
        for _ in range(3):
            client.post("/api/demo")
        cookie = client.cookies.get("viscaner_session")
        assert len(client.get("/api/history").json()["items"]) == 2
    with TestClient(create_app(settings, StubProvider(prediction()))) as other:
        other.cookies.set("viscaner_session", cookie)
        assert len(other.get("/api/history").json()["items"]) == 2


def test_pairing_rules_use_catalog_and_respect_preferences(client):
    result = client.post("/api/pairing", json={"dish": "fish"}).json()
    assert result["method"] == "editorial_rules"
    assert result["wines"][0]["category"] == "Белое"
    assert client.post("/api/pairing", json={"dish": "unknown"}).status_code == 422
    result = client.post("/api/pairing", json={"dish": "fish", "preference": "red"}).json()
    assert result["wines"][0]["category"] == "Красное"


class FakeSommelier:
    ready, remote, model_name = True, True, "fake"

    def __init__(self, wine=None, fail=False):
        self.wine, self.fail, self.questions = wine, fail, []

    def ask(self, question, wine_slug=None):
        self.questions.append(question)
        if self.fail:
            raise RuntimeError("LLM down")
        return {"answer": "Возьмите [2].", "wines": [self.wine], "cited": [2], "model": "fake"}


def test_pairing_goes_through_the_sommelier_and_falls_back_to_rules(settings):
    app = create_app(settings, StubProvider(prediction()))
    with TestClient(app) as client:
        app.state.sommelier = sommelier = FakeSommelier(app.state.catalog.wines["red"])
        result = client.post("/api/pairing", json={"dish": "meat", "preference": "any"}).json()
        assert result["method"] == "llm" and result["cited"] == [2]
        assert [w["slug"] for w in result["wines"]] == ["red"]
        assert sommelier.questions == ["Подбери вино к мясу и блюдам на гриле."]
        app.state.sommelier = FakeSommelier(fail=True)
        result = client.post("/api/pairing", json={"dish": "meat", "preference": "any"}).json()
        assert result["method"] == "editorial_rules" and result["wines"][0]["slug"] == "red"

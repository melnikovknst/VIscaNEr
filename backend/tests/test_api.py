import io
from dataclasses import dataclass

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.config import Settings
from backend.main import create_app, decode_image
from backend.providers import ModelUnavailable, RemoteProvider
from backend.schemas import Candidate, Prediction


@pytest.fixture
def settings(tmp_path):
    catalog = tmp_path / "catalog.csv"
    catalog.write_text("slug,name,category,color,region,grapes,description,winery\n"
        "red,Первое красное,Красное,Рубиновый,Кубань,Каберне,Вишня,Винодельня А\n"
        "white,Второе белое,Белое,Соломенный,Крым,Рислинг,Яблоко,Винодельня Б\n"
        "rose,Третье розовое,Розовое,Розовый,Дон,Пино,Ягоды,Винодельня В\n", encoding="utf-8")
    return Settings(_env_file=None, catalog_path=catalog, catalog_archive=tmp_path / "missing.zip",
        refs_root=tmp_path / "refs", data_dir=tmp_path / "data", max_upload_mb=1,
        model_provider="remote", remote_url="http://model.test/predict")


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


def prediction(first=0.91, second=0.60, slug="red", abstain=False):
    return Prediction(candidates=[Candidate(slug=slug, similarity=first), Candidate(slug="white", similarity=second)], model_version="test-v1", abstain=abstain)


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
    assert client.get("/api/catalog/red").json()["rosquality_rating"] is None


def test_upload_roundtrip_and_flat_evaluator_contract(client):
    response = client.post("/api/scan", files={"file": ("label.jpg", photo(), "image/jpeg")})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "matched"
    assert result["wine"]["slug"] == "red"
    assert result["similarity"] == .91
    assert result["metrics"]["f1_top1"] is None
    assert len(client.get("/api/history").json()["items"]) == 1
    assert client.post("/predict", files={"file": ("label.jpg", photo())}).json() == {"slug": "red"}
    assert len(client.get("/api/history").json()["items"]) == 1  # Evaluator doesn't pollute history.


@pytest.mark.parametrize("scores,expected", [((.8, .79), "uncertain"), ((.4, .3), "not_found"), ((.92, .60), "matched")])
def test_open_set_and_near_duplicate_thresholds(settings, scores, expected):
    with TestClient(create_app(settings, StubProvider(prediction(*scores)))) as client:
        result = client.post("/api/scan", files={"file": ("a.jpg", photo())}).json()
        assert result["status"] == expected
        assert (result["wine"] is not None) == (expected == "matched")
        flat = client.post("/predict", files={"file": ("a.jpg", photo())}).json()
        assert flat == {"slug": "red" if expected == "matched" else None}


@pytest.mark.parametrize("scores,expected", [((.6, .3), "uncertain"), ((.45, .3), "not_found"), ((.92, .6), "matched")])
def test_weak_but_plausible_answer_becomes_a_choice(settings, scores, expected):
    settings = settings.model_copy(update={"min_suggest_similarity": .5})
    with TestClient(create_app(settings, StubProvider(prediction(*scores)))) as client:
        result = client.post("/api/scan", files={"file": ("a.jpg", photo())}).json()
        assert result["status"] == expected
        assert (result["wine"] is not None) == (expected == "matched")
        assert client.post("/predict", files={"file": ("a.jpg", photo())}).json()["slug"] == (
            "red" if expected == "matched" else None)


def cascade(order, basis="label", margin=None, pipeline=None):
    """A cascade response: candidates in final order, scores are stage-1 scores."""
    return Prediction(candidates=[Candidate(slug=s, similarity=v) for s, v in order], model_version="cascade-test",
                      ranked=True, decision_basis=basis, decision_margin=margin, pipeline=pipeline)


def test_cascade_order_is_not_resorted_by_stage_one_score(settings):
    # The bottle model swapped the pair: "white" wins although its stage-1
    # score is lower. Re-sorting by score would silently undo that decision.
    swapped = cascade([("white", .80), ("red", .81)], basis="resolver", margin=.05,
                      pipeline={"resolver": {"invoked": True, "swapped": True}})
    with TestClient(create_app(settings, StubProvider(swapped))) as client:
        result = client.post("/api/scan", files={"file": ("a.jpg", photo())}).json()
        assert result["status"] == "matched" and result["wine"]["slug"] == "white"
        assert [c["wine"]["slug"] for c in result["candidates"]] == ["white", "red"]
        assert result["decision_basis"] == "resolver"
        assert result["pipeline"]["resolver"]["swapped"] is True


@pytest.mark.parametrize("basis,margin,expected", [
    # Stage 1 was a near tie by construction when the resolver ran; its own
    # separation is what decides, against its own threshold.
    ("resolver", .05, "matched"),
    ("resolver", .005, "uncertain"),
    # Without the resolver, the stage-1 margin must clear min_margin (0.04).
    ("label", .10, "matched"),
    ("label", .01, "uncertain"),
])
def test_each_stage_is_judged_by_its_own_margin(settings, basis, margin, expected):
    settings.min_resolver_margin = .02
    response = cascade([("red", .80), ("white", .795)], basis=basis, margin=margin)
    with TestClient(create_app(settings, StubProvider(response))) as client:
        assert client.post("/api/scan", files={"file": ("a.jpg", photo())}).json()["status"] == expected


@pytest.mark.parametrize("probs,expected", [
    # Softmax confidence, no margin gate: a clear-enough top-1 is answered even
    # when the runner-up is close.
    ((.50, .45), "matched"),
    ((.30, .10), "uncertain"),
    ((.15, .10), "not_found"),
])
def test_five_stream_uses_its_own_confidence_thresholds(settings, probs, expected):
    settings = settings.model_copy(update={"model_provider": "five_stream"})
    response = Prediction(candidates=[Candidate(slug="red", similarity=probs[0]),
                                      Candidate(slug="white", similarity=probs[1])],
                          model_version="five-stream-test", ranked=True, decision_margin=probs[0] - probs[1])
    with TestClient(create_app(settings, StubProvider(response))) as client:
        assert client.post("/api/scan", files={"file": ("a.jpg", photo())}).json()["status"] == expected
        assert client.post("/predict", files={"file": ("a.jpg", photo())}).json() == {
            "slug": "red" if expected == "matched" else None}


def test_cascade_duplicate_slugs_keep_first_position(settings):
    response = cascade([("white", .80), ("red", .81), ("white", .70)], basis="resolver", margin=.05)
    with TestClient(create_app(settings, StubProvider(response))) as client:
        result = client.post("/api/scan", files={"file": ("a.jpg", photo())}).json()
        assert [c["wine"]["slug"] for c in result["candidates"]] == ["white", "red"]


def test_explicit_abstention_and_empty_candidates(settings):
    for result in [prediction(abstain=True), Prediction(), Prediction(candidates=[Candidate(slug="red", similarity=.99)])]:
        with TestClient(create_app(settings, StubProvider(result))) as client:
            assert client.post("/predict", files={"file": ("a.jpg", photo())}).json() == {"slug": None}


def test_unknown_top_result_not_silently_promoted(settings):
    with TestClient(create_app(settings, StubProvider(prediction(slug="not-in-catalog")))) as client:
        assert client.post("/predict", files={"file": ("a.jpg", photo())}).status_code == 503


def test_demo_never_claims_to_recognize_upload(settings):
    settings.model_provider = "demo"
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["model_ready"] is False
        result = client.post("/api/demo").json()
        assert result["status"] == "demo" and result["similarity"] is None
        assert result["candidates"] == []
        assert client.post("/predict", files={"file": ("a.jpg", photo())}).status_code == 503


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
    assert client.post("/predict", content=chunks()).status_code == 413


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


def test_pairing_uses_catalog_and_respects_preferences(client):
    result = client.post("/api/pairing", json={"dish": "fish"}).json()
    assert result["method"] == "editorial_rules"
    assert result["wines"][0]["category"] == "Белое"
    assert client.post("/api/pairing", json={"dish": "unknown"}).status_code == 422
    result = client.post("/api/pairing", json={"dish": "fish", "preference": "red"}).json()
    assert result["wines"][0]["category"] == "Красное"


class FakeSommelier:
    ready, remote, model_name = True, True, "fake"

    def __init__(self, fail=False):
        self.fail, self.questions = fail, []

    def ask(self, question, wine_slug=None):
        self.questions.append(question)
        if self.fail:
            raise RuntimeError("LLM down")
        return {"answer": "Возьмите [2].", "wines": [self.wine], "cited": [2], "model": "fake"}


def test_pairing_goes_through_the_sommelier_and_falls_back_to_rules(settings):
    app = create_app(settings, StubProvider(prediction()))
    with TestClient(app) as client:
        sommelier = FakeSommelier()
        sommelier.wine = app.state.catalog.wines["red"]
        app.state.sommelier = sommelier
        result = client.post("/api/pairing", json={"dish": "meat", "preference": "any"}).json()
        assert result["method"] == "llm" and result["cited"] == [2]
        assert [w["slug"] for w in result["wines"]] == ["red"]
        assert sommelier.questions == ["Подбери вино к мясу и блюдам на гриле."]
        app.state.sommelier = FakeSommelier(fail=True)
        result = client.post("/api/pairing", json={"dish": "meat", "preference": "any"}).json()
        assert result["method"] == "editorial_rules" and result["wines"][0]["slug"] == "red"


def test_missing_local_weights_keep_catalog_available(settings):
    settings.model_provider = "local"
    settings.checkpoint_path = settings.data_dir / "missing.pt"
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["model_status"] == "error"
        assert client.get("/api/catalog").status_code == 200
        assert client.post("/predict", files={"file": ("a.jpg", photo())}).status_code == 503


def test_transparent_catalog_images(settings):
    folder = settings.refs_root / "rgba"
    folder.mkdir(parents=True)
    Image.new("RGBA", (60, 100), (0, 0, 0, 0)).save(folder / "red.webp")
    with TestClient(create_app(settings, StubProvider(prediction()))) as client:
        response = client.get("/api/catalog/red/image")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/webp"
        assert Image.open(io.BytesIO(response.content)).mode == "RGBA"


def test_remote_model_contract_and_invalid_scores(settings, monkeypatch):
    actual_client = httpx.Client
    def mock_response(request):
        assert request.url == httpx.URL(settings.remote_url)
        assert b'name="file"' in request.content and b'name="top_k"' in request.content
        return httpx.Response(200, json=prediction().model_dump())
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: actual_client(transport=httpx.MockTransport(mock_response), **kwargs))
    provider = RemoteProvider(settings)
    assert provider.predict(Image.new("RGB", (100, 100))).candidates[0].slug == "red"
    def invalid(request):
        return httpx.Response(200, json={"candidates": [{"slug": "red", "similarity": 5}]})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: actual_client(transport=httpx.MockTransport(invalid), **kwargs))
    with pytest.raises(ModelUnavailable):
        provider.predict(Image.new("RGB", (100, 100)))


def test_remote_errors_are_readable_and_do_not_leak_credentials(settings, monkeypatch):
    actual_client = httpx.Client
    def failed(request):
        raise httpx.ReadTimeout("private internal URL/token", request=request)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: actual_client(transport=httpx.MockTransport(failed), **kwargs))
    with pytest.raises(ModelUnavailable) as error:
        RemoteProvider(settings).predict(Image.new("RGB", (100, 100)))
    assert "token" not in str(error.value)

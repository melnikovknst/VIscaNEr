import httpx
import pytest

from backend.config import Settings
from backend.pairing import pairing_question
from backend.schemas import PairingRequest
from backend.sommelier import Sommelier, SommelierUnavailable, read_filters


def test_filters_read_colour_sweetness_and_sparkle():
    f = read_filters("Красное сухое к шашлыку")
    assert f.category == "Красное" and "сухое" in f.sweetness and f.sparkling is None
    assert read_filters("Игристое для праздника").sparkling is True
    assert read_filters("Брют к морепродуктам").sparkling is True


def test_negated_sparkle_is_not_read_as_sparkling():
    assert read_filters("Не игристое белое к курице").sparkling is False
    assert read_filters("тихое белое к рыбе").sparkling is False


def test_dessert_implies_sweet_and_semi_sweet_is_not_dry():
    assert read_filters("Что подать к медовику? Десерт").sweetness == {"сладкое", "полусладкое"}
    assert read_filters("Полусладкое белое").sweetness == {"полусладкое"}
    assert read_filters("полусухое розовое").sweetness == {"полусухое"}


def test_rose_is_not_mistaken_for_red_or_white():
    assert read_filters("розовое на пикник").category == "Розовое"
    assert read_filters("что-нибудь к пасте").category is None


def test_savoury_dish_leaves_sweet_wines_out_unless_asked():
    assert read_filters("Что взять к утке с вишнёвым соусом?").not_sweet is True
    assert read_filters("Полусладкое к утке").not_sweet is False
    assert read_filters("Сладкое вино к шоколадному десерту").not_sweet is False
    assert read_filters("К сырной тарелке").not_sweet is False


def test_pairing_buttons_become_a_sommelier_question():
    assert pairing_question(PairingRequest(dish="meat", preference="any")) == "Подбери вино к мясу и блюдам на гриле."
    question = pairing_question(PairingRequest(dish="fish", preference="white"))
    assert question == "Подбери белое вино к рыбе и морепродуктам."
    assert read_filters(question).category == "Белое" and read_filters(question).not_sweet


def openrouter(handler):
    settings = Settings(_env_file=None, sommelier_backend="openrouter", openrouter_api_key="sk-test",
                        openrouter_url="https://openrouter.test/chat")
    sommelier = Sommelier(settings, catalog=None)
    real = httpx.Client
    sommelier_client = lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    return sommelier, sommelier_client


def test_openrouter_sends_the_prompt_and_returns_the_answer(monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": "Возьмите [1]."}}]})

    sommelier, client = openrouter(handler)
    monkeypatch.setattr(httpx, "Client", client)
    assert sommelier.remote and "OpenRouter" in sommelier.model_name
    assert sommelier._generate("Список вин: [1] ...") == "Возьмите [1]."
    assert seen["auth"] == "Bearer sk-test"
    assert "anthropic/claude-sonnet-5.5" in seen["body"]


@pytest.mark.parametrize("response", [httpx.Response(500), httpx.Response(200, json={"choices": []}),
                                      httpx.Response(200, json={"choices": [{"message": {"content": " "}}]})])
def test_openrouter_failures_are_readable(monkeypatch, response):
    sommelier, client = openrouter(lambda request: response)
    monkeypatch.setattr(httpx, "Client", client)
    with pytest.raises(SommelierUnavailable):
        sommelier._generate("prompt")

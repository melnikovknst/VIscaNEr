from backend.sommelier import read_filters


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

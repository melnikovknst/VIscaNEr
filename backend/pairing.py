from backend.catalog import Catalog
from backend.schemas import PairingRequest, Wine

# Transparent editorial rules, not fabricated LLM or tasting/rating data.
RULES = {
    "meat": ("Красное", "К мясу", "Плотные красные вина обычно хорошо сочетаются с запечённым мясом и блюдами на гриле.", "16–18 °C"),
    "fish": ("Белое", "К рыбе", "Свежие белые вина дополняют рыбу, морепродукты и лёгкие соусы.", "8–12 °C"),
    "cheese": ("Белое", "К сыру", "Белое вино — универсальная отправная точка для сырной тарелки. К выдержанным сырам можно попробовать красное.", "10–12 °C"),
    "vegetables": ("Розовое", "К овощам", "Лёгкое розовое вино подойдёт к овощам на гриле, салатам и средиземноморским закускам.", "8–10 °C"),
    "dessert": ("Белое", "К десерту", "Ищите сладкое или полусладкое вино: оно должно быть не менее сладким, чем сам десерт.", "8–10 °C"),
}
PREFERENCES = {"red": "Красное", "white": "Белое", "rose": "Розовое", "sparkling": "Игристое"}


def recommend(catalog: Catalog, request: PairingRequest):
    category, title, explanation, temperature = RULES[request.dish]
    if request.preference != "any":
        category = PREFERENCES[request.preference]
        if category != RULES[request.dish][0]:
            explanation = "Учли ваш выбор стиля. " + {
                "Красное": "К лёгким блюдам выбирайте менее танинное красное; насыщенные вина оставьте для более плотных соусов.",
                "Белое": "Свежие белые вина поддержат лёгкие соусы, а более насыщенные подойдут к сливочным и запечённым блюдам.",
                "Розовое": "Розовое вино — лёгкий вариант для закусок, овощей и блюд с деликатными соусами.",
                "Игристое": "Игристые вина подходят к лёгким закускам; для сладкого десерта выбирайте сладкий стиль.",
            }[category]
    rows = [w for w in catalog.items if (category.casefold() in w.category.casefold() if category != "Игристое"
            else any(word in w.name.casefold() for word in ("брют", "игрист", "spumante", "петнат")))]
    if request.dish == "dessert":
        rows = [w for w in rows if "сладк" in w.name.casefold()]
    # Offer different producers rather than adjacent vintages of the same wine.
    selected: list[Wine] = []
    for wine in rows:
        if all(w.winery != wine.winery for w in selected):
            selected.append(wine)
        if len(selected) == 3:
            break
    return {"title": title, "explanation": explanation, "temperature": temperature,
            "note": "Общие рекомендации по стилю вина. Учитывайте соус и способ приготовления.",
            "method": "editorial_rules", "wines": selected}

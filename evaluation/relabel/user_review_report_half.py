"""The human review of the 180 misses on the held-out half, encoded as decisions.

Each entry keeps the reviewer's own words in `note`, so the encoding can be
audited against what was actually said. Where the reviewer named the right wine,
it was resolved to a catalog slug by reading the label at full resolution
(reviewer "user+claude"). Rerunning replaces these decisions.

    python -m evaluation.relabel.user_review_report_half
"""

import json

from scripts.relabel_tools import PREDICTIONS, labels, record

PREDS = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
L = labels()


def q(i):
    return f"q-{i:04d}"


def top(i, k=1):
    return PREDS[q(i)]["top5"][k - 1]["slug"]


def orig(i):
    return L[q(i)]["slug"]


DONE = []


def rec(i, decision, accepted=None, note="", reviewer="user"):
    record(q(i), decision, accepted, reviewer=reviewer, note=note)
    DONE.append(i)


def main():
    # No recognisable target: people, glasses, grapes, a metro station, crowds of bottles, junk.
    for i in [7, 33, 60, 99, 108, 109, 122, 123, 134, 139, 186, 216, 227, 331, 348, 352, 370, 399, 406, 485,
              494, 502, 538, 549, 582, 583, 588, 589, 602, 629, 635, 649, 657, 714, 716, 764, 769, 771, 778,
              790, 844, 857, 875, 911, 918]:
        rec(i, "remove", note="пользователь: удалить")

    # Model right, label wrong.
    for i, note in [(61, "верно модель предсказала"), (280, "модель права"), (318, "модель права"),
                    (346, "модель права"), (476, "модель права"), (493, "правильно"), (587, "модель права"),
                    (618, "фотка плохая, модель права"), (628, "разметка неверна, модель молодец"),
                    (653, "модель молодец, разметка неверна"), (693, "ответ модели похож на правду, разметка точно неверна"),
                    (736, "модель права"), (747, "модель права"), (777, "модель права"), (811, "модель права"),
                    (855, "модель верно для своей бутылки, разметка неверна")]:
        rec(i, "fix", [top(i)], note=f"пользователь: {note}; принят ответ модели")

    # Several bottles: the model's bottle and the labelled one are both valid answers.
    for i, note in [(2, "угадала левую, в разметке правая"), (196, "левую угадала, в разметке правая"),
                    (237, "три бутылки, ответ правильный"), (342, "много вина, выбранную угадала"),
                    (391, "правую бутылку правильно"), (393, "левую бутылку правильно"), (480, "угадала левую"),
                    (616, "три бутылки, центральную угадала"), (627, "для выбранной из двух верно"),
                    (672, "предсказала ту, что выбрала"), (684, "правильно ту, что выбрала"),
                    (809, "из трёх центральную угадала"), (898, "для выбранной бутылки верно")]:
        rec(i, "multi", sorted({orig(i), top(i)}), note=f"пользователь: {note}")

    # Named corrections, checked against the label text at full resolution.
    rec(8, "fix", [top(8, 1)], note="на этикетке «НРАВ РЕЗЕРВ»: модель права (разметка: РАШ Резерв)", reviewer="user+claude")
    rec(16, "keep", note="на этикетке «Gewurztraminer de Gaï-Kodzor 2022»: разметка верна, модель ошиблась "
                         "(пользователь склонялся к ответу модели по цвету надписи)", reviewer="claude")
    rec(374, "fix", ["vinogradniki-gay-kodzora-gewurztraminer-de-gai-kodzor-gevyurtstraminer-beloe-polusuhoe-135"],
        note="на этикетке «Gewurztraminer de Gaï-Kodzor 2022»", reviewer="user+claude")
    rec(105, "fix", ["risling"], note="на этикетке «Галицкий & Галицкий, РИСЛИНГ»; детектор взял ведёрко", reviewer="user+claude")
    rec(175, "fix", [top(175, 2)], note="совиньон блан = 2-й ответ модели", reviewer="user+claude")
    rec(176, "fix", ["novyj-svet-shardone"], note="на этикетке «Новый Свет, ШАРДОНЕ 2019»", reviewer="user+claude")
    rec(235, "fix", [top(235, 3)], note="пользователь: правильный ответ на 3-м месте")
    rec(277, "fix", [top(277, 5)], note="шардоне, не кокур = 5-й ответ модели", reviewer="user+claude")
    rec(341, "fix", [top(341, 3)], note="пользователь: третий вариант нейросети")
    rec(419, "fix", ["novyj-svet-vyderzhannoe-bryut"], note="«Новый Свет, выдержанное, брют, 2022»", reviewer="user+claude")
    rec(783, "fix", ["mantra-blanc-de-blancs"], note="на этикетке «MANTRA ESTATE BLANC DE BLANCS 2021»", reviewer="user+claude")
    rec(789, "fix", ["alma-valley-kaberne-sovinon-rezerv-krasnoe-suhoe-15"],
        note="на этикетке «ALMA VALLEY RESERVE 2021 CABERNET SAUVIGNON»; детектор взял фон", reviewer="user+claude")
    rec(803, "fix", [top(803, 1)], note="на пино нуар больше похоже = ответ модели", reviewer="user+claude")
    rec(909, "fix", ["mantra-estate-sauvignon-blanc-sovinon-blan-beloe-suhoe-133"],
        note="на этикетке «MANTRA ESTATE SAUVIGNON BLANC»", reviewer="user+claude")
    rec(97, "not_in_catalog", note="сильванер: в каталоге нет ни одного сильванера", reviewer="user+claude")
    rec(198, "not_in_catalog", note="Кокур 2024 на фото; в каталоге этот кокур только 2025", reviewer="user+claude")
    rec(377, "not_in_catalog", note="«Виталий Батрак, Пино Нуар», чёрная серия без медальона: в каталоге нет", reviewer="user+claude")
    rec(504, "not_in_catalog", note="«Виталий Батрак, Мерло», чёрная серия без медальона: в каталоге нет", reviewer="user+claude")
    rec(757, "not_in_catalog", note="«Chateau Tamagne ... Extra Brut 2024»: экстра брюта в каталоге нет", reviewer="user+claude")
    rec(349, "keep", note="пользователь: бомонд той же винодельни, мб другой год; разметка оставлена")

    # Label wrong, true wine not identified, model known wrong: scored as a miss.
    for i, note in [(10, "Солнечная долина, но не Мускатное Фестивальное"), (39, "и модель, и разметка неверны"),
                    (43, "и модель, и разметка неверны"), (74, "и модель, и разметка неверны"),
                    (150, "детектор этикетки + разметка неверна"), (200, "модель и разметка неверны"),
                    (267, "детектор этикетки + разметка неверна"), (397, "разметка неверна + детектор этикетки")]:
        rec(i, "wrong_unknown", note=f"пользователь: {note}")

    # Label wrong or doubtful, cannot tell whether the model is right: excluded.
    for i, note in [(824, "абрп, и это не розе (неясно, про разметку или про ответ модели)"), (389, "разметка"), (409, "разметка"),
                    (413, "разметка"), (415, "разметка"), (433, "разметка"), (439, "разметка"), (442, "разметка"),
                    (446, "разметка"), (453, "разметка"), (484, "разметка"), (509, "разметка"), (534, "разметка"),
                    (606, "разметка"), (612, "разметка"), (645, "разметка"), (719, "разметка"), (729, "разметка"),
                    (753, "разметка"), (782, "разметка"), (804, "разметка"), (808, "разметка"), (848, "разметка"),
                    (849, "модель ближе к правде, чем разметка"), (864, "разметка"), (868, "разметка"), (877, "разметка"),
                    (904, "разметка"), (67, "модель и разметка неверны, мб нет в каталоге"), (135, "мб нет в каталоге"),
                    (168, "скорее всего нет в каталоге"), (254, "скорее всего нет в каталоге"), (301, "мб нет в каталоге"),
                    (326, "разметка и модель неверны, но мб это то же вино")]:
        rec(i, "unsure", note=f"пользователь: {note}")

    # "абрп": the MODEL answered «Розовое полусладкое | Абрау-Дюрсо» again - its
    # hub answer (top-1 for 22 of 918 photos). The label is not challenged.
    for i in [218, 285, 500, 555, 609, 794]:
        rec(i, "keep", note="пользователь: абрп = модель снова ответила «Розовое полусладкое Абрау» (вино-магнит)")

    # Label right: honest model miss, label-detector miss, or the model picked another bottle.
    for i, note in [(27, "разметка похожа больше, чем ответ модели"), (31, "выбрала другую бутылку"),
                    (38, "честно плохой ответ"), (86, "честный промах"), (88, "честный промах"), (116, "честный промах"),
                    (158, "два вина, разметка верна для правой"), (178, "честный промах"), (184, "честный промах"),
                    (185, "честный промах"), (268, "честный промах"), (273, "честный промах"),
                    (274, "бутылка не совсем похожа, мб другой год"), (366, "хз, мб и верно"), (404, "честный промах"),
                    (440, "честный промах"), (461, "честно"), (514, "правильного нет даже в топ-5"),
                    (683, "честный промах"), (862, "честно"), (882, "честно"), (93, "промах детектора этикетки"),
                    (214, "промах детектора этикетки"), (242, "промах детектора этикетки"), (264, "промах детектора этикетки"),
                    (408, "промах детектора этикетки"), (497, "промах детектора этикетки"), (586, "промах детектора этикетки"),
                    (614, "промах детектора этикетки"), (710, "промах детектора этикетки"), (835, "этикеткой посчитался кирпич"),
                    (893, "промах детектора этикетки"), (179, "выбрала левую, которой мб нет"), (489, "выбрала левую, разметка верна"),
                    (584, "выбрала левые, которых мб нет"), (647, "выбрала ту, которой мб нет"),
                    (812, "из четырёх выбрала вторую слева и ошиблась"), (907, "выбрала ту, которой скорее всего нет")]:
        rec(i, "keep", note=f"пользователь: {note}")

    assert len(DONE) == len(set(DONE)) == 180, (len(DONE), len(set(DONE)))
    print("recorded", len(DONE), "decisions")


if __name__ == "__main__":
    main()

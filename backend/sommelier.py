"""LLM sommelier: retrieval over the catalog, then an LLM explains a choice.

The LLM is the local YandexGPT-5 Lite or, with VISCANER_SOMMELIER_BACKEND=openrouter,
any chat model behind the OpenRouter API. Retrieval is local in both cases.

1. Every wine is described by a short text built from its catalog card and its
   taste profile (scripts/sommelier_profiles.py) and embedded with bge-m3.
2. A question is embedded the same way; hard filters read from its wording
   (colour, sweetness, sparkling, dessert) narrow the catalog first.
3. The LLM sees the question and a numbered shortlist and cites wines as [N].
   Only cited shortlist entries are returned, so a wine outside the catalog can
   never reach the page.

Models are loaded lazily on first use and kept for the life of the process.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from backend.catalog import Catalog
from backend.config import Settings

logger = logging.getLogger(__name__)

SHORTLIST = 6


class SommelierUnavailable(RuntimeError):
    """A readable reason the answer could not be produced (shown to the visitor)."""


@dataclass
class Filters:
    category: str | None = None          # Красное / Белое / Розовое / Оранжевое
    sparkling: bool | None = None
    sweetness: set[str] | None = None    # allowed profile sweetness values
    not_sweet: bool = False              # a savoury dish: leave sweet wines out


SWEET_STYLES = {"сладкое", "полусладкое"}
# Savoury dishes. Cheese is left out on purpose: blue cheese and sweet wine is a classic.
SAVOURY = (r"\b(мяс|стейк|утк|утин|гус|кур|цыпл|индейк|шашлык|гриль|барбекю|бургер|рыб|лосос|форел|"
           r"тунц|морепродукт|кревет|миди|устриц|кальмар|паст|пицц|плов|баранин|ягнят|свинин|"
           r"говядин|телятин|дичь|оленин|колбас|овощ|салат|гриб|суш)")


COLOURS = [("красн", "Красное"), ("бел", "Белое"), ("розов", "Розовое"), ("розе", "Розовое"),
           ("оранж", "Оранжевое")]
SWEET = [
    ("полусладк", {"полусладкое"}),
    ("полусух", {"полусухое"}),
    ("сладк", {"сладкое", "полусладкое"}),
    ("сух", {"сухое", "брют", "экстра брют", "брют натюр"}),
]


def read_filters(question: str) -> Filters:
    """Hard constraints stated in plain words. Anything unclear is left to retrieval."""
    text = question.casefold().replace("ё", "е")
    found = Filters()
    for stem, category in COLOURS:
        if re.search(rf"\b{stem}", text):
            found.category = category
            break
    # The negation first: "не игристое" also contains "игрист".
    if re.search(r"\bне\s+игрист|\bбез\s+пузыр|\bтих(ое|ого|им)\b", text):
        found.sparkling = False
    elif re.search(r"\b(игрист|шампанск|брют|пузырьк|петнат|пет-нат)", text):
        found.sparkling = True
    for stem, allowed in SWEET:
        if re.search(rf"\b{stem}", text):
            found.sweetness = allowed
            break
    if found.sweetness is None and re.search(r"\b(десерт|торт|пирож|шоколад|мороженое)", text):
        found.sweetness = {"сладкое", "полусладкое"}
    # "Утка с вишнёвым соусом" matched a sweet Muscat on its cherry aromas: unless
    # sweetness was asked for, a savoury dish never gets a sweet wine.
    if found.sweetness is None and re.search(SAVOURY, text):
        found.not_sweet = True
    return found


def wine_text(wine, profile: dict) -> str:
    style = [wine.category]
    if profile.get("sweetness") and profile["sweetness"] != "неизвестно":
        style.append(profile["sweetness"])
    if profile.get("sparkling"):
        style.append("игристое")
    parts = [f"{wine.name}. {wine.winery}.", ", ".join(style) + ".",
             f"Регион: {wine.region}. Сорт: {wine.grapes}.", wine.description]
    if profile.get("aromas"):
        parts.append("Ароматы: " + ", ".join(profile["aromas"]) + ".")
    if profile.get("pairing"):
        parts.append("Подходит к: " + ", ".join(profile["pairing"]) + ".")
    return " ".join(p for p in parts if p)


def brief(n: int, wine, profile: dict) -> str:
    """One shortlist line for the prompt: only facts from the card and profile."""
    facts = [wine.category]
    for key in ("sweetness", "body", "acidity", "tannins"):
        value = profile.get(key)
        if value and value != "неизвестно":
            facts.append({"body": "тело ", "acidity": "кислотность ", "tannins": "танины "}.get(key, "") + value)
    if profile.get("sparkling"):
        facts.append("игристое")
    line = f"[{n}] {wine.name} — {wine.winery}, {wine.region}. Сорт: {wine.grapes}. {'; '.join(facts)}."
    if profile.get("aromas"):
        line += " Ароматы: " + ", ".join(profile["aromas"]) + "."
    if profile.get("pairing"):
        line += " К блюдам: " + ", ".join(profile["pairing"]) + "."
    if profile.get("serve_temp") and "неизв" not in profile["serve_temp"]:
        line += f" Подача: {profile['serve_temp']}."
    return line


INSTRUCTION = (
    "Ты сомелье винного сервиса, который рассказывает о российских винах. Ответь на вопрос гостя, "
    "выбрав от одного до трёх вин только из списка ниже. Называй вина их номером в квадратных "
    "скобках, например [2], и коротко объясняй выбор, опираясь только на данные из списка. "
    "Подбирай пару к самому блюду — его насыщенности, соусу и способу приготовления, — а не по "
    "совпадению отдельных слов с ароматами вина. К несладким блюдам не предлагай сладкие и "
    "полусладкие вина, если гость сам об этом не просит. "
    "Не упоминай вина, которых нет в списке, не называй цены, рейтинги и награды. "
    "Пиши по-русски, дружелюбно, не больше 90 слов."
)
INSTRUCTION_WINE = (
    "Ты сомелье винного сервиса, который рассказывает о российских винах. Гость держит в руках "
    "вино [1] и задаёт вопрос о нём. Отвечай про вино [1], опираясь только на данные из списка. "
    "Если гость просит похожее или альтернативу, можешь предложить вина [2]–[{n}] из списка, "
    "называя их номером в квадратных скобках. Не называй цены, рейтинги и награды. "
    "Пиши по-русски, дружелюбно, не больше 90 слов."
)


class Sommelier:
    def __init__(self, settings: Settings, catalog: Catalog):
        self.settings = settings
        self.catalog = catalog
        self.lock = threading.Lock()
        self.ready = False
        self.profiles: dict[str, dict] = {}
        self.slugs: list[str] = []
        self.embeddings = None
        self.embedder = self.embed_tokenizer = None
        self.llm = self.tokenizer = None

    # ------------------------------------------------------------------ models
    def load(self) -> None:
        with self.lock:
            if self.ready:
                return
            import torch
            from transformers import AutoModel, AutoTokenizer

            if self.remote and not self.settings.openrouter_api_key.get_secret_value():
                raise ValueError("VISCANER_OPENROUTER_API_KEY is required for the openrouter sommelier")
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            path = self.settings.sommelier_profiles_path
            for line in path.open(encoding="utf-8"):
                item = json.loads(line)
                if not item["problems"] and item["slug"] in self.catalog.wines:
                    self.profiles[item["slug"]] = item["profile"]
            self.embed_tokenizer = AutoTokenizer.from_pretrained(self.settings.sommelier_embedder_path)
            self.embedder = AutoModel.from_pretrained(
                self.settings.sommelier_embedder_path,
                dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device).eval()
            self.embeddings, self.slugs = self._index()
            if not self.remote:
                from transformers import AutoModelForCausalLM, BitsAndBytesConfig

                self.tokenizer = AutoTokenizer.from_pretrained(self.settings.sommelier_llm_path)
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.settings.sommelier_llm_path, device_map="cuda",
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16,
                                                           bnb_4bit_use_double_quant=True)).eval()
            self.ready = True

    @property
    def remote(self) -> bool:
        return self.settings.sommelier_backend == "openrouter"

    @property
    def model_name(self) -> str:
        if self.remote:
            return f"{self.settings.openrouter_model} (OpenRouter) + bge-m3"
        return "YandexGPT-5-Lite-8B-instruct (4-bit, local) + bge-m3"

    def _embed(self, texts: list[str], batch: int = 32):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch):
                enc = self.embed_tokenizer(texts[start:start + batch], padding=True, truncation=True,
                                           max_length=512, return_tensors="pt").to(self.device)
                cls = self.embedder(**enc).last_hidden_state[:, 0]      # bge-m3 dense = CLS
                out.append(torch.nn.functional.normalize(cls.float(), dim=-1))
        return torch.cat(out)

    def _index(self):
        """Embeddings of every wine, cached next to the profiles and keyed by their content."""
        torch = self.torch
        slugs = sorted(self.catalog.wines)
        texts = [wine_text(self.catalog.wines[s], self.profiles.get(s, {})) for s in slugs]
        signature = hashlib.sha256("\n".join(texts).encode()).hexdigest()
        cache = self.settings.sommelier_profiles_path.with_suffix(".index.pt")
        if cache.is_file():
            data = torch.load(cache, map_location=self.device, weights_only=True)
            if data.get("signature") == signature:
                return data["embeddings"], slugs
        embeddings = self._embed(texts)
        torch.save({"signature": signature, "embeddings": embeddings.cpu()}, cache)
        return embeddings, slugs

    # --------------------------------------------------------------- retrieval
    def _allowed(self, slug: str, filters: Filters) -> bool:
        wine, profile = self.catalog.wines[slug], self.profiles.get(slug, {})
        if filters.category and wine.category != filters.category:
            return False
        if filters.sparkling is not None and bool(profile.get("sparkling")) != filters.sparkling:
            return False
        if filters.sweetness and profile.get("sweetness") not in filters.sweetness:
            return False
        if filters.not_sweet and profile.get("sweetness") in SWEET_STYLES:
            return False
        return True

    def search(self, query: str, filters: Filters, k: int, exclude: set[str] = frozenset(),
               vector=None) -> list[str]:
        vector = self._embed([query])[0] if vector is None else vector
        scores = self.embeddings @ vector
        order = self.torch.argsort(scores, descending=True).tolist()
        picked, wineries = [], {}
        for index in order:
            slug = self.slugs[index]
            if slug in exclude or not self._allowed(slug, filters):
                continue
            # At most two wines per producer, so the shortlist is a real choice.
            winery = self.catalog.wines[slug].winery
            if wineries.get(winery, 0) >= 2:
                continue
            wineries[winery] = wineries.get(winery, 0) + 1
            picked.append(slug)
            if len(picked) == k:
                break
        return picked

    # -------------------------------------------------------------------- answer
    def ask(self, question: str, wine_slug: str | None = None) -> dict:
        self.load()
        started = time.perf_counter()
        filters = read_filters(question)
        if wine_slug:
            wine = self.catalog.wines[wine_slug]
            vector = self.embeddings[self.slugs.index(wine_slug)]
            similar = self.search("", Filters(category=filters.category, sparkling=filters.sparkling,
                                              sweetness=filters.sweetness, not_sweet=filters.not_sweet),
                                  SHORTLIST - 1, exclude={wine_slug}, vector=vector)
            shortlist = [wine_slug] + similar
            instruction = INSTRUCTION_WINE.format(n=len(shortlist))
        else:
            shortlist = self.search(question, filters, SHORTLIST)
            if not shortlist:             # the filters were too strict - fall back to meaning only
                shortlist = self.search(question, Filters(), SHORTLIST)
            instruction = INSTRUCTION
        listing = "\n".join(brief(n, self.catalog.wines[s], self.profiles.get(s, {}))
                            for n, s in enumerate(shortlist, 1))
        prompt = f"{instruction}\n\nСписок вин:\n{listing}\n\nВопрос гостя: {question}"
        answer = self._generate(prompt)
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
        cited = list(dict.fromkeys(n for n in cited if 1 <= n <= len(shortlist)))
        if wine_slug:                     # the asked-about wine always leads
            cited = [1] + [n for n in cited if n != 1]
        if not cited:                     # the model named nothing: show its top pick only
            cited = [1]
        # Numbers are shortlist positions; on the page they read as footnotes.
        return {"answer": answer.strip(), "wines": [self.catalog.wines[shortlist[n - 1]] for n in cited],
                "cited": cited, "shortlist": shortlist,
                "filters": {"category": filters.category, "sparkling": filters.sparkling,
                            "sweetness": sorted(filters.sweetness) if filters.sweetness else None,
                            "not_sweet": filters.not_sweet},
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "model": self.model_name}

    def _generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        if self.remote:
            return self._generate_openrouter(prompt, max_new_tokens)
        # YandexGPT-5 Lite's template has no system role: the instruction opens the user turn.
        messages = [{"role": "user", "content": prompt}]
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                 return_dict=True)["input_ids"].to("cuda")
        with self.lock, self.torch.inference_mode():
            out = self.llm.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                    repetition_penalty=1.05, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def _generate_openrouter(self, prompt: str, max_tokens: int) -> str:
        import httpx

        headers = {"Authorization": "Bearer " + self.settings.openrouter_api_key.get_secret_value(),
                   "X-Title": "VIscaNEr"}
        # Other tokenizers spend more tokens on Cyrillic than YandexGPT: allow twice as many.
        body = {"model": self.settings.openrouter_model, "max_tokens": max_tokens * 2, "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}]}
        try:
            with httpx.Client(timeout=self.settings.timeout_seconds, follow_redirects=False) as client:
                response = client.post(self.settings.openrouter_url, headers=headers, json=body)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            # Only the exception type: the message could echo the request headers.
            logger.warning("OpenRouter sommelier failed: %s", type(exc).__name__)
            raise SommelierUnavailable("Сомелье не ответил. Попробуйте спросить ещё раз.") from exc
        if not isinstance(content, str) or not content.strip():
            raise SommelierUnavailable("Сомелье вернул пустой ответ. Попробуйте спросить ещё раз.")
        return content

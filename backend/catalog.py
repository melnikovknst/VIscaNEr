"""The wine catalog: data/catalog.zip holds catalog.csv and one WebP photo per wine."""
import csv
import io
from functools import lru_cache
from urllib.parse import quote
from zipfile import ZipFile

from backend.config import Settings
from backend.schemas import Wine


class Catalog:
    def __init__(self, settings: Settings):
        if not settings.catalog_archive.is_file():
            raise FileNotFoundError(f"Каталог не найден: {settings.catalog_archive}. Выполните git lfs pull.")
        self.archive = ZipFile(settings.catalog_archive)
        text = self.archive.read("catalog.csv").decode("utf-8-sig")
        self.wines: dict[str, Wine] = {}
        for row in csv.DictReader(io.StringIO(text)):
            slug = row["slug"]
            if not slug or "/" in slug or "\\" in slug or slug in self.wines:
                raise ValueError(f"Некорректный или повторяющийся slug: {slug}")
            self.wines[slug] = Wine(
                **{key: row.get(key, "") for key in ("slug", "name", "category", "color", "region", "grapes", "description", "winery", "near_dup_group")},
                image_url=f"/api/catalog/{quote(slug)}/image",
                source_url=f"https://vino-svoe.ru/wines/{quote(slug)}",
            )
        if not self.wines:
            raise ValueError("Каталог пуст")
        self.items = list(self.wines.values())
        self.search_text = {w.slug: f"{w.name} {w.winery} {w.region} {w.grapes}".casefold() for w in self.items}

    def search(self, query: str = "", category: str = "", offset: int = 0, limit: int = 24):
        words = query.casefold().split()
        rows = [w for w in self.items if (not category or w.category == category)
                and all(word in self.search_text[w.slug] for word in words)]
        return {"items": rows[offset:offset + limit], "total": len(rows), "offset": offset, "limit": limit}

    @lru_cache(maxsize=256)
    def image(self, slug: str) -> tuple[bytes, str]:
        if slug not in self.wines:
            raise KeyError(slug)
        try:
            return self.archive.read(f"images/{slug}.webp"), "image/webp"
        except KeyError:
            raise FileNotFoundError(slug) from None

    def featured(self) -> list[Wine]:
        picks = []
        for text in ["резерв каберне", "рислинг", "пино нуар", "брют", "шардоне", "саперави"]:
            result = self.search(text, limit=1)["items"]
            if result and result[0] not in picks:
                picks.append(result[0])
        return picks or self.items[:6]

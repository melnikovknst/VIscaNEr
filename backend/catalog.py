import csv
import io
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile

from backend.config import Settings
from backend.schemas import Wine

ARCHIVE_PREFIX = "wine-scanner/data/"


class Catalog:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.catalog_path.is_file():
            text = settings.catalog_path.read_text(encoding="utf-8-sig")
        elif settings.catalog_archive.is_file():
            with ZipFile(settings.catalog_archive) as archive:
                text = archive.read(ARCHIVE_PREFIX + "catalog.csv").decode("utf-8-sig")
        else:
            raise FileNotFoundError("Каталог не найден. Скачайте datasets/wine-scanner_code-catalog.zip через Git LFS.")
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

    @lru_cache(maxsize=128)
    def image(self, slug: str) -> tuple[bytes, str]:
        if slug not in self.wines:
            raise KeyError(slug)
        variants = [("rgba", "webp", "image/webp"), ("rgba", "png", "image/png"), ("rgb", "jpg", "image/jpeg")]
        for directory, extension, mime in variants:
            path = self.settings.refs_root / directory / f"{slug}.{extension}"
            if path.is_file():
                return path.read_bytes(), mime
        if self.settings.catalog_archive.is_file():
            with ZipFile(self.settings.catalog_archive) as archive:
                for directory, extension, mime in variants:
                    try:
                        return archive.read(f"{ARCHIVE_PREFIX}refs/{directory}/{slug}.{extension}"), mime
                    except KeyError:
                        pass
        raise FileNotFoundError(slug)

    def featured(self) -> list[Wine]:
        picks = []
        for text in ["резерв каберне", "рислинг", "пино нуар", "брют", "шардоне", "саперави"]:
            result = self.search(text, limit=1)["items"]
            if result and result[0] not in picks:
                picks.append(result[0])
        return picks or self.items[:6]

import asyncio
import io
import logging
import re
import time
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from backend.catalog import Catalog
from backend.config import ROOT, Settings
from backend.pairing import recommend
from backend.metrics import read_metrics
from backend.providers import ModelUnavailable, create_provider
from backend.schemas import Match, PairingRequest, Prediction, ScanResult
from backend.storage import History

logger = logging.getLogger(__name__)


class BodyLimitMiddleware:
    """Bound actual incoming multipart bytes, including chunked uploads."""
    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.limit:
                return await JSONResponse({"detail": "Файл слишком большой. Максимум — 12 МБ (или лимит сервера)."}, 413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def buffered_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, buffered_receive, send)


def decode_image(content: bytes, settings: Settings) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(415, "Поддерживаются только JPG, PNG и WebP.")
                if source.width * source.height > settings.max_pixels:
                    raise HTTPException(413, "Изображение слишком большое. Уменьшите его до 24 мегапикселей.")
                if min(source.size) < 32:
                    raise HTTPException(422, "Фото слишком маленькое. Этикетка должна быть хорошо видна.")
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                return image
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(422, "Не удалось прочитать изображение. Загрузите исправный JPG, PNG или WebP.") from exc


def resolve_prediction(prediction: Prediction, catalog: Catalog, settings: Settings, elapsed_ms: int) -> ScanResult:
    # Never skip an unknown top-1 and silently promote a lower-ranked identity.
    if prediction.ranked:
        # The provider already decided the order (a cascade may have swapped
        # its top two); keep it, dropping only repeated slugs.
        seen: set[str] = set()
        candidates = [c for c in prediction.candidates if not (c.slug in seen or seen.add(c.slug))][:5]
    else:
        unique = {c.slug: c for c in sorted(prediction.candidates, key=lambda c: c.similarity)}
        candidates = sorted(unique.values(), key=lambda c: c.similarity, reverse=True)[:5]
    if any(c.slug not in catalog.wines for c in candidates):
        raise ModelUnavailable("Каталог и модель не синхронизированы. В ответе есть неизвестное вино.")
    top = candidates[0] if candidates else None
    if prediction.decision_margin is not None:
        margin = prediction.decision_margin
    else:
        margin = top.similarity - candidates[1].similarity if len(candidates) > 1 else None
    # Each stage is judged by its own separation. After the bottle model ran,
    # stage 1 was a near tie by construction, so its margin cannot gate the answer.
    required_margin = settings.min_resolver_margin if prediction.decision_basis == "resolver" else settings.min_margin
    status = "not_found"
    message = "Не удалось найти вино. Снимите этикетку крупнее, без бликов и соседних бутылок."
    wine = None
    uncertain_message = "Есть несколько похожих этикеток. Снимите название и год крупнее — пока точный результат не подтверждён."
    if top and not prediction.abstain and top.similarity >= settings.min_similarity:
        # A single candidate does not demonstrate separation from near-duplicates.
        if margin is not None and margin >= required_margin:
            status, wine, message = "matched", catalog.wines[top.slug], "Вино найдено в каталоге."
        else:
            status, message = "uncertain", uncertain_message
    elif (top and not prediction.abstain and settings.min_suggest_similarity is not None
          and top.similarity >= settings.min_suggest_similarity):
        # Too weak to answer, but the right wine is usually among the candidates:
        # let the visitor pick rather than report a failure.
        status, message = "uncertain", uncertain_message
    return ScanResult(id=str(uuid4()), status=status, wine=wine,
        candidates=[Match(wine=catalog.wines[c.slug], similarity=c.similarity) for c in candidates],
        similarity=top.similarity if top else None, margin=margin, elapsed_ms=elapsed_ms,
        model_version=prediction.model_version, provider=settings.model_provider,
        created_at=datetime.now(timezone.utc).isoformat(), message=message,
        decision_basis=prediction.decision_basis, pipeline=prediction.pipeline)


def create_app(settings: Settings | None = None, provider_override=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.catalog = Catalog(settings)
        app.state.history = History(settings.data_dir / "history.sqlite3", settings.history_limit)
        app.state.model_error = None
        app.state.inference_lock = asyncio.Lock()
        try:
            app.state.provider = provider_override or await run_in_threadpool(create_provider, settings)
            if settings.model_provider in {"local", "cascade"} and hasattr(app.state.provider, "slugs"):
                if set(app.state.provider.slugs) - app.state.catalog.wines.keys():
                    raise ValueError("Галерея содержит slug, отсутствующие в каталоге")
        except Exception:
            logger.exception("Model initialization failed")
            app.state.provider = None
            app.state.model_error = "Модель не загрузилась. Проверьте настройки и журнал backend."
        yield

    app = FastAPI(title="winescanner API", version="1.0.0", lifespan=lifespan,
                  description="Сканер российских вин. /predict — плоский ответ для оценщика; /api/scan — полная карточка.")
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_upload_mb * 1024 * 1024 + 65536)

    @app.exception_handler(ModelUnavailable)
    async def unavailable_handler(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    def session_id(request: Request, response: Response) -> str:
        current = request.cookies.get("viscaner_session", "")
        if not re.fullmatch(r"[a-f0-9]{32}", current):
            current = uuid4().hex
            response.set_cookie("viscaner_session", current, httponly=True, samesite="lax",
                                secure=request.url.scheme == "https", max_age=30 * 86400)
        return current

    @app.get("/api/health")
    def health():
        ready = app.state.provider is not None and settings.model_provider != "demo"
        return {"status": "ok", "provider": settings.model_provider, "model_ready": ready,
                "model_status": "error" if app.state.model_error else ("demo" if settings.model_provider == "demo" else "configured" if settings.model_provider == "remote" else "ready"),
                "message": app.state.model_error, "catalog_count": len(app.state.catalog.items),
                "max_upload_mb": settings.max_upload_mb}

    @app.get("/api/catalog")
    def catalog(q: str = Query("", max_length=200), category: str = "",
                offset: int = Query(0, ge=0), limit: int = Query(24, ge=1, le=100)):
        return app.state.catalog.search(q, category, offset, limit)

    @app.get("/api/metrics")
    def metrics():
        return read_metrics(settings.data_dir / "evaluation.json")

    @app.get("/api/catalog/meta")
    def catalog_meta():
        wines = app.state.catalog.items
        return {"total": len(wines), "categories": sorted({w.category for w in wines if w.category}),
                "winery_count": len({w.winery for w in wines if w.winery}),
                "featured": app.state.catalog.featured()}

    @app.get("/api/catalog/{slug}/image")
    def catalog_image(slug: str):
        try:
            data, mime = app.state.catalog.image(slug)
            return Response(data, media_type=mime, headers={"Cache-Control": "public, max-age=86400"})
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "Фото не найдено")

    @app.get("/api/catalog/{slug}")
    def wine_detail(slug: str):
        if slug not in app.state.catalog.wines:
            raise HTTPException(404, "Вино не найдено")
        return app.state.catalog.wines[slug]

    async def scan_image(file: UploadFile):
        start = time.perf_counter()
        try:
            raw = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        finally:
            await file.close()
        if len(raw) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"Файл должен быть не больше {settings.max_upload_mb} МБ.")
        image = await run_in_threadpool(decode_image, raw, settings)
        if app.state.provider is None:
            raise ModelUnavailable(app.state.model_error)
        if app.state.inference_lock.locked():
            raise HTTPException(429, "Сканер обрабатывает другое фото. Повторите через несколько секунд.", headers={"Retry-After": "3"})
        async with app.state.inference_lock:
            try:
                prediction = await run_in_threadpool(app.state.provider.predict, image)
            except ModelUnavailable:
                raise
            except Exception as exc:
                logger.exception("Inference failed")
                raise ModelUnavailable("Не удалось обработать фото. Проверьте сервис модели и попробуйте снова.") from exc
        result = resolve_prediction(prediction, app.state.catalog, settings, round((time.perf_counter() - start) * 1000))
        report = read_metrics(settings.data_dir / "evaluation.json")
        result.metrics = {**report, "matches_model_version": prediction.model_version in report.get("model_versions", [])}
        return result

    @app.post("/api/scan", response_model=ScanResult)
    async def scan(request: Request, response: Response, file: UploadFile = File(...)):
        result = await scan_image(file)
        sid = session_id(request, response)
        await run_in_threadpool(app.state.history.save, sid, result)
        return result

    @app.post("/predict")
    @app.post("/api/predict", include_in_schema=False)
    async def predict(file: UploadFile = File(...)):
        result = await scan_image(file)
        return {"slug": result.wine.slug if result.wine else None}

    @app.post("/api/demo", response_model=ScanResult)
    def demo(request: Request, response: Response):
        result = ScanResult(id=str(uuid4()), status="demo", wine=app.state.catalog.featured()[0],
            candidates=[], similarity=None, margin=None, elapsed_ms=0, model_version="example",
            provider="demo", created_at=datetime.now(timezone.utc).isoformat(),
            message="Пример карточки из каталога. Это демонстрация интерфейса, а не результат распознавания.")
        app.state.history.save(session_id(request, response), result)
        return result

    @app.get("/api/history")
    def history(request: Request, response: Response):
        return {"items": app.state.history.list(session_id(request, response))}

    @app.delete("/api/history", status_code=204)
    def clear_history(request: Request, response: Response):
        app.state.history.clear(session_id(request, response))

    @app.post("/api/pairing")
    def pairing(body: PairingRequest):
        return recommend(app.state.catalog, body)

    # One process serves the production build. The API remains independent in dev.
    dist = ROOT / "dist"
    if dist.is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/favicon.svg", include_in_schema=False)
        def favicon():
            return FileResponse(dist / "favicon.svg")

        @app.get("/", include_in_schema=False)
        def frontend():
            return FileResponse(dist / "index.html")

    return app


app = create_app()

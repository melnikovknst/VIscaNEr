# winescanner · сканер российских вин

Веб-приложение для знакомства с российскими винами: мобильный интерфейс, загрузка этикетки, карточка вина, каталог, коллекция, история и подбор вина к блюду. Исходный пайплайн обучения DINOv3 сохранён без изменений.

Каталог и 2103 фотографии настоящие. Пользовательский интерфейс показывает сканер, каталог и карточки без технических статусов и демонстрационных сценариев. Распознавание требует весов или API коллег. До подключения провайдера загрузка возвращает понятное сообщение о временной недоступности и предлагает поиск по каталогу; случайные результаты, фиктивные проценты точности и рейтинги не используются.

## Актуальный ML pipeline

Финальный pipeline состоит ровно из двух моделей:

```text
фотография пользователя
  → YOLO11n: выбирает бутылку прежде всего по confidence, иногда две при неоднозначности
  → DINOv3 ViT-B/16: одним batch обрабатывает 1–2 изображения бутылок
  → объединяет similarity по двум бутылкам только для неоднозначного случая
  → top-k wine_slug и cosine similarity
```

Каскада, DINO-S и дополнительного reranker в актуальном inference нет.

Пользователь наводит центральное перекрестье на этикетку нужной бутылки. YOLO
кандидаты получают оценку `confidence + 0.20 × axis_proximity`, где
`axis_proximity` зависит только от горизонтального расстояния центра bbox до
вертикальной оси перекрестья. Координата Y и сам факт пересечения bbox с
перекрестьем бонуса не дают. Поэтому небольшой перевес confidence у явно
боковой бутылки может быть исправлен, но слабая центральная детекция не победит
сильную только за счёт геометрии. Если две разные уверенные детекции близки и
по итоговой оценке, и по расстоянию до вертикальной оси, обе бутылки проходят
через DINO в одном batch. Для каждого `wine_slug` берётся
максимальная cosine similarity из двух бутылок, после чего строится единый top-k.
Во всех остальных случаях DINO получает только одну бутылку. Дальше применяется
та же политика, на которой собран новый датасет целых бутылок:

- YOLO confidence `>= 0.75` — в DINO передаётся crop бутылки с padding `6%`;
- YOLO confidence `< 0.75` — в DINO передаётся исходная фотография целиком;
- если YOLO ничего не нашёл — исходная фотография также передаётся целиком;
- второй DINO-вход включается только для двух различных боксов с confidence
  `>= 0.75`, близкими итоговыми scores и расстояниями до вертикальной оси;
- DINO получает square-padded изображение `224×224`, строит нормализованный
  embedding и ранжирует эталоны по cosine similarity.

Используемые файлы:

- inference entrypoint: `infer_wine.py`;
- DINO architecture/transforms: `dinov3_retrieval.py`;
- локальные backbone loaders: `deeptune_backbones.py`;
- joint YOLO бутылок и этикеток: `models/joint_yolo/best.pt`;
- исходный DINOv3-B backbone: `models/dinov3/model.safetensors`;
- обученный DINOv3-B: `models/trained_checkpoints/dinov3_vitb16_bottles_best_full.pt`;
- gallery: `datasets/bottle_classifier_crops/refs/`, автоматически извлекается
  из `datasets/bottle_classifier_crops.zip`, если директории ещё нет.

После клонирования нужно получить LFS-файлы и установить inference-зависимости:

```bash
git lfs pull
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-inference.txt
```

Один снимок:

```bash
.venv/bin/python infer_wine.py /path/to/photo.jpg --top-k 5
```

Целая директория с сохранением JSON:

```bash
.venv/bin/python infer_wine.py /path/to/photos \
  --top-k 5 \
  --output runs/inference/predictions.json
```

На первом запуске DINO один раз строит gallery embeddings и сохраняет cache в
`runs/inference/dinov3_vitb16_bottles_gallery.pt`. Последующие запуски используют
готовый cache. `device=auto` выбирает CUDA, затем MPS, затем CPU. Значение
`similarity` — cosine similarity, а не вероятность.

## Быстрый запуск

Нужны Python 3.11–3.13 и Node.js 22.12+ (проверено на Python 3.13 и Node 24). Выполняйте команды из корня репозитория.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
npm.cmd ci
Copy-Item .env.example .env
npm.cmd run dev
```

Открыть **http://127.0.0.1:5173**. Одна команда запускает Vite и FastAPI. Windows: используйте `npm.cmd`, если PowerShell блокирует `npm.ps1`. Если `python` не находится в PATH, укажите полный путь к интерпретатору.

На Linux/macOS: `python3 -m venv .venv`, `.venv/bin/pip install -r backend/requirements.txt`, `npm ci`, `cp .env.example .env`, `npm run dev`.

Каталог загружается прямо из `datasets/wine-scanner_code-catalog.zip`, распаковка не нужна. Если вместо архива лежит указатель LFS:

```sh
git lfs pull --include="datasets/wine-scanner_code-catalog.zip"
```

### Одна точка входа для демонстрации

```powershell
npm.cmd run build
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Интерфейс: **http://127.0.0.1:8000**. Swagger: **http://127.0.0.1:8000/docs**. Все шрифты и изображения локальные; в demo/local режимах внешний интернет приложению не нужен. Ссылка «Карточка источника» открывает исходную запись во внешнем каталоге.

Для телефона в той же сети запустите backend с `--host 0.0.0.0` и откройте `http://<IP-компьютера>:8000`. Кнопка камеры использует системный выбор фото с `capture="environment"`; на компьютере и в некоторых браузерах откроется выбор файла. Продаж и оплаты в приложении нет.

## Подключение модели

### Вариант A — файлы коллег

Установить дополнительные зависимости (на машине с GPU заранее выбрать подходящую CUDA-сборку PyTorch):

```powershell
.venv\Scripts\python.exe -m pip install -r backend/requirements-local.txt
```

В `.env`:

```dotenv
VISCANER_MODEL_PROVIDER=local
VISCANER_CHECKPOINT_PATH=models/trained_checkpoints/dinov3_vitb16_bottles_best_full.pt
VISCANER_GALLERY_PATH=runs/inference/dinov3_vitb16_bottles_gallery.pt
VISCANER_DEVICE=auto
VISCANER_DETECTOR_PATH=models/joint_yolo/best.pt
```

Потребуются обученный checkpoint и gallery, построенная именно этой версией
модели. Gallery cache автоматически создаётся первым запуском `infer_wine.py`.
Web-provider использует тот же confidence + vertical-axis YOLO selector,
fallback `< 0.75` и DINOv3-B, что и основной CLI. Исходный
`model.safetensors` backend-провайдеру отдельно не требуется: обученный
checkpoint содержит backbone.

Форматы из текущего обучения:

- `best.pt`: `model_state_dict`, `config`, `epoch` (обычный deployable checkpoint, **не** `last.pt` со всем состоянием optimizer/RNG).
- `gallery_embeddings.pt`: `embeddings` `[N,D]`, `wine_slugs` `[N]`; `labels` и `paths` могут присутствовать. Одна строка на уникальный slug, все slug должны существовать в каталоге.
- `DEVICE=auto` выбирает CUDA, затем MPS, затем CPU. Можно указать `cpu` или `cuda:0`.

Модель загружается и прогревается один раз при старте. YOLO выбирает целевую
бутылку прежде всего по confidence; уверенный box кропается, а при confidence
ниже `0.75` используется исходное фото. После замены весов перезапустите
backend. `/api/health` покажет ошибку загрузки, а каталог останется доступен.

### Вариант B — HTTP API коллег

```dotenv
VISCANER_MODEL_PROVIDER=remote
VISCANER_REMOTE_URL=http://127.0.0.1:9000/predict
VISCANER_REMOTE_API_KEY=
VISCANER_TIMEOUT_SECONDS=30
```

Backend отправляет **POST multipart/form-data**: поле `file` — нормализованное JPEG, поле `top_k=5`. При наличии ключа добавляет `Authorization: Bearer ...`. `REMOTE_URL` — полный URL обработчика, суффикс не дописывается. Пример ответа:

```json
{
  "model_version": "dinov3-run-2026-09-20",
  "abstain": false,
  "candidates": [
    {"slug": "slug-iz-catalog-csv", "similarity": 0.91},
    {"slug": "drugoy-slug-iz-catalog-csv", "similarity": 0.72}
  ]
}
```

Нужны до пяти реальных кандидатов и **cosine similarity в диапазоне [-1,1]**. Значения не являются вероятностями. Один кандидат без ближайшего конкурента не подтверждает достаточный отрыв и приводит к `uncertain`. `abstain=true` принудительно означает отсутствие достоверного результата. Пустой список допустим. Неизвестный slug, некорректный ответ, таймаут и ошибка сети возвращают понятный `503`; demo-подмена не выполняется. Если API коллег отличается, точка адаптации — `RemoteProvider.predict()` в `backend/providers.py`.

В remote-режиме `model_status=configured` означает корректную настройку, а не проверенную доступность внешнего сервиса; фактический ответ проверяется при сканировании.

## API

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/api/health` | Режим, готовность модели, число вин |
| POST | `/api/scan` | Фото в поле `file` → карточка, top-5, similarity, margin, время, метрики |
| POST | `/predict` | Фото в поле `file` → строго `{"slug":"..."}` или `{"slug":null}` |
| POST | `/api/demo` | Технический пример для интеграции, не используется пользовательским интерфейсом |
| GET | `/api/catalog?q=рислинг&category=Белое&offset=0&limit=24` | Поиск/фильтр/пагинация |
| GET | `/api/catalog/meta` | Категории, число вин и виноделен, подборка |
| GET | `/api/catalog/{slug}` | Карточка вина |
| GET | `/api/catalog/{slug}/image` | Фото из локального каталога |
| GET / DELETE | `/api/history` | История текущей cookie-сессии / её очистка |
| POST | `/api/pairing` | `{ "dish": "fish", "preference": "any" }` → вина к блюду |
| GET | `/api/metrics` | Результаты измеренного прогона или `null` до оценки |

```sh
curl -F "file=@label.jpg" http://127.0.0.1:8000/predict
```

На Windows используйте `curl.exe`. Алиас `/api/predict` поддерживает тот же формат. Название multipart-поля проверяйте по скрипту кейсодержателя, когда он будет передан: сейчас согласованный контракт — `file`. `/predict` не пишет историю. В режиме demo он возвращает `503`, а не выдуманный slug.

Состояния `/api/scan`: `matched` — одна итоговая карточка, `uncertain` — недостаточный отрыв, `not_found` — низкое сходство/отказ. Похожие вина показываются только при отсутствии подтверждённого совпадения. Уверенный результат сразу открывает одну карточку.

## Настройки

Все параметры имеют префикс `VISCANER_`; шаблон — `.env.example`.

| Переменная | По умолчанию | Значение |
|---|---|---|
| `MODEL_PROVIDER` | `demo` | `demo`, `local`, `remote` |
| `MIN_SIMILARITY` | `0.65` | Минимальный cosine score |
| `MIN_MARGIN` | `0.04` | Минимальная разница top-1 и top-2 |
| `MAX_UPLOAD_MB` | `12` | Ограничение фото |
| `MAX_PIXELS` | `24000000` | Защита от чрезмерного разрешения |
| `DATA_DIR` | `backend/data` | SQLite-история и отчёт оценки |
| `HISTORY_LIMIT` | `100` | Последних результатов на сессию |
| `CATALOG_PATH` | `datasets/wine-scanner/data/catalog.csv` | CSV имеет приоритет перед архивом |
| `CATALOG_ARCHIVE` | `datasets/wine-scanner_code-catalog.zip` | Архив каталога и изображений |
| `REFS_ROOT` | `datasets/wine-scanner/data/refs` | Распакованные `rgba/*.webp`, `rgb/*.jpg` |

Пороги предварительные: коллеги должны откалибровать их на отложенных полевых фото. **Точность 90–100% и SLA 3 секунды пока не измерены.**

## Оценка и проверки

```powershell
.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
.venv\Scripts\python.exe -m pytest backend/tests -q
npm.cmd run build
# При запущенном backend в demo-режиме, в отдельном терминале:
npm.cmd run test:e2e
```

Браузерные тесты запускаются в изолированном Edge (нужен установленный Microsoft Edge), проверяют desktop и iPhone viewport. Скриншоты — `tmp/ui-*.png`. Для CI с Chromium замените `channel` в `playwright.config.ts` и установите браузер через `npx playwright install chromium`.

После подключения реальной модели распакуйте `datasets/real_photos_v4.zip` и запустите:

```powershell
.venv\Scripts\python.exe -m scripts.evaluate_api --labels datasets/real_photos_v4/labels.csv --images datasets/real_photos_v4/queries
```

Исходные метки `real_photos_v4` местами ошибочны. Переразметка всех 918 фото лежит в
`evaluation/relabel/` (журнал решений с причинами) и экспортируется в
`evaluation/real_photos_v4_relabeled.csv`: у фото с несколькими бутылками или с
дублирующимися карточками каталога несколько верных ответов, фото без целевой бутылки
исключены. Пороги подбираются на половине `selection`, честная цифра — на `report`:

```powershell
.venv\Scripts\python.exe -m scripts.evaluate_relabeled
.venv\Scripts\python.exe -m scripts.evaluate_api --labels evaluation/real_photos_v4_relabeled.csv --images datasets/real_photos_v4/queries --split report
```

Скрипт считает top-1 micro-F1, set-retrieval micro-F1@5, Recall@5, точность с учётом честных отказов, coverage, медиану и p95 времени. Ошибки не выбрасываются из знаменателя; при них код выхода 1. Формулы сохранены в отчёте. **F1@5 и Recall@5 — разные метрики**: для сравнения с организаторами используйте их окончательное определение и официальный оценщик.

Отчёт `backend/data/evaluation.json` доступен в `/api/metrics` и в ответах сканера, с флагом соответствия версии модели. Без прогона F1 равен `null`: по одному запросу без правильного ответа измерить F1 нельзя. Demo-режим оценщик отклоняет.

## Docker

```sh
docker compose up --build
```

Открыть `http://127.0.0.1:8000`. Архив каталога монтируется read-only; история — в named volume. Этот лёгкий образ рассчитан на **demo/remote**. Локальную GPU-модель запускайте нативно или отдельным inference-сервисом: CUDA-окружение зависит от машины. Конфигурация Docker добавлена, запуск Docker в текущем окружении не проверялся.

## Границы текущей версии

- Финальные веса и внешний API пока не переданы; реальный локальный DINO-инференс ещё не прогонялся. Адаптер соответствует коду обучения, удалённый контракт проверен тестовым сервисом.
- В CSV нет рейтинга Роскачества: карточка честно показывает отсутствие данных.
- «К столу» использует понятные правила по стилям вина, а не LLM; это общие сочетания, не индивидуальная оценка конкретной бутылки.
- Коллекция хранится в localStorage, история — в SQLite по HttpOnly cookie. Авторизации и синхронизации между устройствами нет. История ограничена 100 записями/сессию; записи старше 30 дней удаляются при следующей записи.
- Фото не сохраняются в приложении; multipart может временно буферизоваться библиотекой. В remote-режиме нормализованное фото передаётся только настроенному сервису модели.
- Один процесс backend и одна одновременная операция inference: конкурентный запрос получает `429`. Нагрузочная инфраструктура, публичная авторизация и распределённые очереди не входят в локальное демо.

Архитектура — [ARCHITECTURE.md](ARCHITECTURE.md). Обучение label-DINO — [README_FULL_TRAINING.md](README_FULL_TRAINING.md). Whole-bottle ambiguity classifiers — [BOTTLE_CLASSIFIER.md](BOTTLE_CLASSIFIER.md). Датасеты и ограничения использования фотографий — [DATASETS.md](DATASETS.md).

Точная карта актуальных YOLO/DINO/OCR inference-файлов и весов для передачи
другому разработчику — [MODEL_INFERENCE_HANDOFF.md](MODEL_INFERENCE_HANDOFF.md).

Финальный однопроходный detector, одновременно возвращающий бутылку и её
этикетку, находится в [joint_yolo/README.md](joint_yolo/README.md).

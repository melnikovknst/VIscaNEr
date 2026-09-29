# VIscaNEr: актуальный Transformer inference handoff

Все пути указаны от корня репозитория. Это комплект для **текущего выбранного
пайплайна**, а не старого bottle-only inference и не ранних MLP/Transformer
экспериментов.

## 1. Что запускаем

```text
исходная фотография
  -> один joint YOLO forward: bottle + wine_label
  -> label crop и whole-bottle crop одной целевой бутылки
  -> frozen Stage-2C DINOv3-B для этикетки
  -> frozen Stage-2C DINOv3-B для бутылки
  -> frozen EasyOCR ru+en по label crop
  -> объединение Top-12 каждой ветки, максимум 36 кандидатов
  -> residual five-stream Transformer
  -> итоговый Top-K wine_slug
```

YOLO выбирает цель по `confidence + 0.20 * vertical_axis_proximity`. Координата
Y и простое пересечение перекрестия бонуса не дают. Бутылочный box с confidence
ниже `0.75` заменяется исходной фотографией. При настоящей неоднозначности
поддерживаются два label/bottle view, а финальные оценки wine-кандидатов
объединяются максимумом.

## 2. Все необходимые файлы

| Назначение | Актуальный путь |
|---|---|
| Единая точка inference | `five_stream_transformer/infer.py` |
| Архитектура residual Transformer | `five_stream_transformer/model.py` |
| Архитектура и transforms двух DINO-B | `five_stream_transformer/dino.py` |
| Загрузчик Stage-2C | `five_stream_transformer/stage2c.py` |
| OCR text embedding | `five_stream_transformer/text.py` |
| Joint YOLO логика и связывание боксов | `joint_yolo/infer.py` |
| Joint YOLO weights | `models/joint_yolo/best.pt` |
| Полный Stage-2C checkpoint двух DINO-B, epoch 17 | `models/stage2c/manual_stage2c_best.pt` |
| Последний residual Transformer, epoch 3 | `models/five_stream_transformer/residual_best.pt` |
| Предвычисленные признаки gallery | `models/five_stream_transformer/gallery_features.pt` |
| EasyOCR detector weights | `models/easyocr_ru_en/craft_mlt_25k.pth` |
| EasyOCR ru+en recognizer weights | `models/easyocr_ru_en/cyrillic_g2.pth` |
| Компактные label+bottle galleries, 2 x 2,103 refs | `datasets/inference_galleries.zip` |
| Python dependencies | `requirements-inference.txt` |

Оригинальный `models/dinov3/model.safetensors` для этого pipeline **не нужен**:
оба полных backbone уже находятся внутри Stage-2C checkpoint.

## 3. Получение репозитория

```bash
git clone https://github.com/melnikovknst/VIscaNEr.git
cd VIscaNEr
git lfs install
git lfs pull --include="models/joint_yolo/best.pt,models/stage2c/manual_stage2c_best.pt,models/five_stream_transformer/*.pt,models/easyocr_ru_en/*.pth,datasets/inference_galleries.zip"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-inference.txt
.venv/bin/python scripts/verify_latest_pipeline_assets.py
```

Windows PowerShell: заменить `.venv/bin/python` на
`.venv\Scripts\python.exe`.

## 4. Готовый CLI

Одна фотография:

```bash
.venv/bin/python -m five_stream_transformer.infer photo.jpg --top-k 5
```

Директория с фотографиями и JSON:

```bash
.venv/bin/python -m five_stream_transformer.infer ./photos \
  --top-k 5 \
  --device cuda \
  --output runs/inference/five_stream_predictions.json
```

Репозиторий уже содержит совместимый
`models/five_stream_transformer/gallery_features.pt`, поэтому обычный первый
запуск не пересчитывает 4,206 эталонных изображений. Если cache отсутствует или
его подпись не совпадает, автоматически:

1. распакуется `datasets/inference_galleries.zip`;
2. будут построены gallery features;
3. cache сохранится по пути, переданному через `--gallery-cache`.

Следующие запуски используют cache. На сервере cache лучше построить один раз
при деплое и хранить на постоянном диске.

## 5. Встраивание в backend

Модель надо создавать **один раз при старте процесса**, а не на каждый HTTP
запрос:

```python
from PIL import Image
from five_stream_transformer.infer import FiveStreamInferencePipeline

pipeline = FiveStreamInferencePipeline(device="cuda")

def recognize(path: str) -> dict:
    with Image.open(path) as image:
        return pipeline.predict(image, top_k=5)
```

Главный ответ находится в `result["candidates"][0]["wine_slug"]`. Поле
`score` — внутренний ranking logit Transformer, **не вероятность и не cosine
confidence**. Для UI нельзя показывать его как процент без отдельной
калибровки. В одном GPU-процессе inference следует сериализовать lock/очередью:
YOLO, EasyOCR и PyTorch-модели разделяют CUDA-контекст. Несколько worker-процессов
загрузят отдельную копию всех весов в память.

## 6. Проверенная совместимость артефактов

- Stage-2C: epoch 17, 2,103 класса, embedding 256;
- Transformer: `stage2c-five-stream-residual-v1`, epoch 3;
- Transformer записан с SHA256 Stage-2C
  `f037ecd909f28ddadb782602a6ea079bddd5afd549fe351dad892ed9d2665c6e`;
- YOLO classes: `0=bottle`, `1=wine_label`;
- обе galleries содержат одинаковые 2,103 `wine_slug`.

Контрольные SHA256 лежат рядом с весами и gallery archive.


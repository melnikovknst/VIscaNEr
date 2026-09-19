# Запуск DINOv3-B retrieval на Kaggle

Пайплайн обучает не классификатор «в вакууме», а embedding-модель: YOLO-кроп
этикетки используется как query, а чистое reference-изображение бутылки — как
gallery item. Итоговая проверка показывает Recall@1/5/10 отдельно для знакомых
и полностью отложенных wine identities.

## 1. Подготовить три приватных Kaggle Dataset

На Mac из корня проекта:

```bash
cd /Users/konstantinmelnikov/Desktop/work/VIscaNEr
source .venv/bin/activate
python -m pip install kaggle
python prepare_kaggle_dinov3_bundle.py --kaggle-username YOUR_KAGGLE_USERNAME
```

Скрипт создаст `kaggle_upload/` с тремя наборами:

- `viscaner-dinov3-data` — generator-target-aligned кропы: YOLO bbox принимается
  только внутри точно отслеженной целевой бутылки; также reference images и CSV;
- `viscaner-dinov3-vitb16-weights` — локальный `model.safetensors`;
- `viscaner-dinov3-code` — notebook, CLI, config и Python-модуль.

Исходники не перемещаются и не удаляются. На том же диске используются hard
links, поэтому staging-папка почти не занимает дополнительного места.

Если Kaggle API уже авторизован, загрузить приватные datasets можно так:

```bash
kaggle datasets create -p kaggle_upload/viscaner-dinov3-data -r zip
kaggle datasets create -p kaggle_upload/viscaner-dinov3-vitb16-weights -r zip
kaggle datasets create -p kaggle_upload/viscaner-dinov3-code -r zip
```

Флаг `--public` не добавлять: без него datasets создаются приватными. Формат
`dataset-metadata.json` и команды соответствуют официальной документации
[Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).

Можно вместо CLI вручную создать три private Dataset в интерфейсе Kaggle и
перетащить содержимое соответствующих папок.

Для замены уже существующих data/code datasets новыми версиями:

```bash
kaggle datasets version -p kaggle_upload/viscaner-dinov3-data \
  -m "Replace crops with generator-target-aligned DINO dataset" -r zip

kaggle datasets version -p kaggle_upload/viscaner-dinov3-code \
  -m "Add generator-target-aligned crop builder and DINO paths" -r zip
```

Dataset с базовыми DINOv3-весами не изменился, повторно загружать его не нужно.

## 2. Создать Kaggle Notebook

1. `Create → New Notebook`.
2. В правой панели `Add Input` подключить все три private Dataset.
3. В `Settings → Accelerator` выбрать `GPU`. Пайплайн использует первый CUDA
   GPU; TPU не нужен.
4. Internet можно включить на время установки `transformers`, затем выключить:
   модельные веса с Hugging Face не скачиваются.

Kaggle документирует включение GPU и режим `Save & Run All` в
[Notebooks guide](https://www.kaggle.com/docs/notebooks).

## 3. Первая cell: скопировать код и поставить недостающие пакеты

```python
from pathlib import Path
import shutil

SOURCE = Path("/kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-code")
PROJECT = Path("/kaggle/working/VIscaNEr")
PROJECT.mkdir(parents=True, exist_ok=True)

for source in SOURCE.rglob("*"):
    if source.is_file() and source.name != "dataset-metadata.json":
        destination = PROJECT / source.relative_to(SOURCE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

%pip install -q "transformers>=5.6,<6" "safetensors>=0.5,<1" "scikit-learn>=1.5,<2"
```

После установки лучше сделать `Restart Session`, затем выполнить первую cell
ещё раз: повторное копирование безопасно, а пакеты уже будут установлены.

## 4. Проверить mount paths

```python
from pathlib import Path

required = [
    Path("/kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/crops_metadata.csv"),
    Path("/kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/bottle_images_manifest.csv"),
    Path("/kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/crops/successful"),
    Path("/kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/refs"),
    Path("/kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-vitb16-weights/model.safetensors"),
]
for path in required:
    print(path.exists(), path)
assert all(path.exists() for path in required)
```

Если Kaggle показал другой slug, изменить только соответствующую часть путей.

## 5. Smoke test

Он проверяет загрузку весов, MPS/CUDA-независимый код, transforms, forward,
backward и сохранение checkpoint на маленькой подвыборке:

```python
!python /kaggle/working/VIscaNEr/train_dinov3_retrieval.py all \
  --config /kaggle/working/VIscaNEr/configs/dinov3_retrieval.yaml \
  --project-root /kaggle/working/VIscaNEr \
  --weights-path /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-vitb16-weights/model.safetensors \
  --crops-metadata-path /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/crops_metadata.csv \
  --bottle-manifest-path /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/bottle_images_manifest.csv \
  --crops-root /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/crops \
  --refs-root /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/refs \
  --device cuda --num-workers 4 --quick-smoke
```

Smoke test перезаписывает `best.pt` тестовым checkpoint. Перед полным запуском
это нормально: full run перезапишет его уже обученной моделью.

## 6. Полное обучение

Удалить только флаг `--quick-smoke`:

```python
!python /kaggle/working/VIscaNEr/train_dinov3_retrieval.py all \
  --config /kaggle/working/VIscaNEr/configs/dinov3_retrieval.yaml \
  --project-root /kaggle/working/VIscaNEr \
  --weights-path /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-vitb16-weights/model.safetensors \
  --crops-metadata-path /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/crops_metadata.csv \
  --bottle-manifest-path /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/bottle_images_manifest.csv \
  --crops-root /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/crops \
  --refs-root /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/refs \
  --device cuda --num-workers 4
```

Для гарантированного фонового выполнения использовать `Save Version → Save &
Run All`. Kaggle ограничивает одну GPU-сессию по времени, поэтому сначала нужен
smoke test. Если появляется CUDA OOM, в
`configs/dinov3_retrieval.yaml` уменьшить:

```yaml
stage1_identities_per_batch: 8
stage1_images_per_identity: 4
stage2_identities_per_batch: 4
stage2_images_per_identity: 2
eval_batch_size: 32
```

Input остаётся 224×224; снижать его не рекомендуется, потому что мелкий текст
на этикетках важен для различения похожих вин.

## 7. Где результаты

После успешного run скачать из Kaggle Outputs папку
`/kaggle/working/VIscaNEr/` или отдельные файлы:

- `models/dinov3_retrieval/best.pt` — лучший deployable checkpoint;
- `models/dinov3_retrieval/last.pt` — последняя эпоха;
- `runs/dinov3_retrieval/dinov3_vitb16_wine_retrieval/final_metrics.json`;
- `runs/dinov3_retrieval/dinov3_vitb16_wine_retrieval/gallery_embeddings.pt`;
- `runs/dinov3_retrieval/dinov3_vitb16_wine_retrieval/history.csv`;
- `training_curves.png` и `retrieval_failures.png`.

`best.pt` + `gallery_embeddings.pt` — два основных production-артефакта.
Первый превращает новый YOLO-кроп в embedding, второй содержит reference
embedding для каждого из 2 103 вин.

## 8. Validation-only на новых кропах

Готовый отдельный notebook находится в корне проекта:
`validate_dinov3_new_crops.ipynb`. После подключения четырёх inputs достаточно
нажать `Run All`; training-функции в нём не вызываются.

Команда `validate` заново строит `index.csv` из текущего Kaggle Dataset,
загружает существующий обученный `best.pt` и пересчитывает метрики. Обучение,
backward и изменение checkpoint не выполняются.

Сначала найти checkpoint из сохранённого output предыдущего Kaggle notebook:

```python
from pathlib import Path

checkpoints = sorted(Path("/kaggle/input").rglob("best.pt"))
for checkpoint in checkpoints:
    print(checkpoint)
assert checkpoints, "Добавьте output предыдущего training notebook через Add Input"
OLD_BEST = checkpoints[0]
```

Затем выполнить только validation:

```python
!python /kaggle/working/VIscaNEr/train_dinov3_retrieval.py validate \
  --config /kaggle/working/VIscaNEr/configs/dinov3_retrieval.yaml \
  --project-root /kaggle/working/VIscaNEr \
  --weights-path /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-vitb16-weights/model.safetensors \
  --crops-metadata-path /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/crops_metadata.csv \
  --bottle-manifest-path /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/bottle_images_manifest.csv \
  --crops-root /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/crops \
  --refs-root /kaggle/input/datasets/konstantinmelnikof/viscaner-dinov3-data/refs \
  --checkpoint "{OLD_BEST}" \
  --device cuda --num-workers 4
```

Результаты сохраняются отдельно от training run:

- `validation_only_metrics.json`;
- `validation_only_summary.json`;
- `validation_only_gallery_embeddings.pt`;
- `validation_only_failures.png`.

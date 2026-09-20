# Полное обучение DINOv3-B/16

Полный режим использует тот же target-aligned датасет, модель, transforms,
SupCon + CE loss и retrieval-валидацию, что и основной пайплайн. Отличие —
третья стадия с разморозкой всего backbone и точное продолжение долгой Kaggle
сессии.

## Стадии

1. Только projection/classification heads: 5 эпох, head LR `3e-4`.
2. Последние четыре блока + heads: 5 эпох, backbone LR `1e-5`.
3. Весь DINOv3 backbone + heads: до 100 эпох, backbone LR `5e-6`, head LR
   `1e-4`, warm-up 3 эпохи и cosine decay до 5% начального LR.

Третья стадия выполняет минимум 15 эпох и останавливается после 12 эпох без
улучшения `val_unseen Recall@1`. Та же метрика выбирает глобальный `best.pt`.
В логах и финальном JSON присутствуют Accuracy (= Recall@1), Recall@2,
Recall@5, Recall@10, MRR и ранги.

Обычный `configs/dinov3_retrieval.yaml` оставляет третью стадию выключенной.
Для полного режима используется `configs/dinov3_full_finetune.yaml`.

## Checkpoints

- `best.pt` — глобальный победитель по `val_unseen Recall@1` среди всех стадий;
- `best_full.pt` — лучший checkpoint именно полной разморозки;
- `last.pt` — точка продолжения с model, optimizer, scheduler, AMP scaler,
  RNG, историей и положением внутри стадии.

Свежий запуск не принимает существующие `.pt` в `models_dir`: выбери новый
`RUN_NAME` либо явно продолжай через `--resume-checkpoint`. При достижении
лимита сессии обучение останавливается на границе эпохи до hard timeout и
ставит `continuation_required: true` в `run_summary.json`.

## Kaggle

Импортируй `kaggle_notebooks/DINOv3-deeptune.ipynb` из code dataset, подключи:

- `viscaner-dinov3-code`;
- `viscaner-dinov3-data`;
- `viscaner-dinov3-vitb16-weights`.

Выбери CUDA GPU и запусти **Save Version → Save & Run All**. Notebook сначала
проверяет все три стадии на одном batch в отдельной директории, затем при
`RESUME_CHECKPOINT = None` создаёт новую модель строго из исходного
`model.safetensors`.

Для продолжения подключи сохранённый output предыдущего notebook и задай
полный путь к его `last.pt` в `RESUME_CHECKPOINT`. Рядом с `last.pt` должны
лежать `best.pt` и, если уже началась третья стадия, `best_full.pt`.

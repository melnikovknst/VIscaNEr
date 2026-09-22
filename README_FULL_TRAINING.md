# Полное обучение DINOv3-B

`kaggle_notebooks/DINOv3-deeptune.ipynb` поддерживает четыре независимых
варианта эксперимента: ViT-B/16 или ConvNeXt-B на target-aligned кропах
этикеток или на кропах бутылок. В первой ячейке выбираются `BACKBONE` и
`DATASET_KIND`. Для обеих архитектур используются локальные `safetensors`,
без Hugging Face Hub и сетевой загрузки модели.

Transforms, SupCon + CE loss, retrieval-валидация и правила разбиения общие.
Третья стадия размораживает весь выбранный backbone и поддерживает точное
продолжение долгой Kaggle-сессии.

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
- `viscaner-bottle-classifier-data`;
- `viscaner-dinov3-vitb16-weights` — общий weights dataset с
  `model.safetensors` и `dinov3-convnext-b.safetensors`.

Выбери `BACKBONE = 'vitb16'` или `'convnextb'`, затем
`DATASET_KIND = 'labels'` или `'bottles'`. Выбери CUDA GPU и запусти
**Save Version → Save & Run All**. Notebook сначала проверяет strict load и
все три стадии на одном batch в отдельной директории, затем при
`RESUME_CHECKPOINT = None` создаёт новую модель строго из выбранного исходного
checkpoint.

Для продолжения подключи сохранённый output предыдущего notebook и задай
полный путь к его `last.pt` в `RESUME_CHECKPOINT`. Рядом с `last.pt` должны
лежать `best.pt` и, если уже началась третья стадия, `best_full.pt`.

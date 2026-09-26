# PaddleOCR-VL Top-5 reranker

This optional stage reads the label crop produced by `infer_wine_labels.py`
and reranks only its DINO Top-5. It never adds a wine that DINO did not place
inside the Top-5.

The project did not fine-tune PaddleOCR-VL and has no custom OCR annotation
dataset. The only fitted values are the lexical blend `alpha=0.90` and the
relative evidence gate `0.10`, selected on the calibration half of 100 manually
labelled real photographs. These values are experimental and must be checked on
new held-out data before production use.

On Apple Silicon, use a separate environment because PaddleOCR/MLX dependencies
can conflict with the PyTorch inference environment:

```bash
python3.12 -m venv .venv_paddleocr_vl
.venv_paddleocr_vl/bin/python -m pip install -r ocr_reranker/requirements.txt
unzip -q models/paddleocr_vl/PaddleOCR-VL-1.6.zip -d models/paddleocr_vl
```

Start the local MLX model server:

```bash
.venv_paddleocr_vl/bin/python -u -m mlx_vlm.server \
  --host 127.0.0.1 --port 8111 \
  --model models/paddleocr_vl/PaddleOCR-VL-1.6 \
  --trust-remote-code --max-tokens 256 --vision-cache-size 32
```

In another terminal, first obtain DINO Top-5 and saved label crops, then OCR and
rerank them:

```bash
.venv/bin/python infer_wine_labels.py /path/to/photos \
  --top-k 5 --output runs/inference/labels_top5.json

PYTHONPATH=. .venv_paddleocr_vl/bin/python ocr_reranker/run_label_ocr.py \
  --input runs/inference/labels_top5.json \
  --output runs/inference/labels_top5_ocr.json
```

PaddleOCR-VL supplies the multilingual OCR model (Cyrillic and Latin were both
observed in the local 100-photo run). `reranker.py` normalizes/transliterates
the extracted text and compares it with the five catalogue slugs.

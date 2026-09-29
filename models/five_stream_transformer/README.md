# Latest residual five-stream Transformer

`residual_best.pt` is the deployable epoch-3 checkpoint downloaded from the
completed Kaggle notebook `viscaner-stage-2c-residual-transformer`.

It is not a standalone image encoder. It must be loaded with:

- `models/stage2c/manual_stage2c_best.pt` — both complete frozen DINOv3-B branches;
- `models/joint_yolo/best.pt` — joint bottle/label detector;
- `models/easyocr_ru_en/` — local-only Russian/English OCR weights;
- `data/inference_galleries.zip` — aligned 2,103-label and bottle references.

`gallery_features.pt` is the matching precomputed gallery cache. It makes the
first server start fast; the inference code validates its Stage-2C and gallery
signatures before use and rebuilds it if the assets change.

The checkpoint expects five streams and a candidate pool of at most 36 wines:

1. whole-bottle DINO pre-projection feature, 1536 values;
2. label DINO pre-projection feature, 1536 values;
3. whole-bottle retrieval embedding, 256 values;
4. label retrieval embedding, 256 values;
5. OCR character-ngram embedding, 512 values.

Use `python -m five_stream_transformer.infer` or `FiveStreamInferencePipeline`
from that module; the web service loads it through `backend/providers.py`.

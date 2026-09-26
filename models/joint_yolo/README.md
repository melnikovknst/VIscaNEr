# Joint bottle + label detector

`best.pt` is the deployable YOLO11n checkpoint with two classes:

- `0`: `bottle`;
- `1`: `wine_label`.

It performs both detections in one model forward. Use it through
`joint_yolo/infer.py`; the inference layer ranks labels as detector confidence
plus a bounded `0.20 ×` proximity bonus to the vertical target axis. Vertical
position and crossing the crosshair do not affect ranking. A second pair is
returned only for two genuinely close candidates.

Training configuration and measured validation metrics are recorded in
`metrics.json`. `last.pt` is retained for reproducible continuation, while
production inference should use `best.pt`.

Current checkpoint: 150 deep-adaptation epochs, best epoch 148. Validation:
Precision 0.9878, Recall 0.9099, mAP50 0.9281, mAP50-95 0.8665. The 211 new
manual images intentionally occur in both train and validation, alongside 12
old validation-only examples, so these numbers measure adaptation rather than
unbiased generalization. See `metrics.json` for per-class results and hashes.

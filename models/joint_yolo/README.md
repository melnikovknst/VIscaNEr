# Joint bottle + label detector

`best.pt` is the deployable YOLO11n checkpoint with two classes:

- `0`: `bottle`;
- `1`: `wine_label`.

It performs both detections in one model forward. Use it through
`joint_yolo/infer.py`; the inference layer ranks labels with 90% detector
confidence and 10% smooth crosshair proximity. Crossing the crosshair has no
hard bonus. A second pair is returned only for two genuinely close candidates.

Training configuration and measured validation metrics are recorded in
`metrics.json`. `last.pt` is retained for reproducible continuation, while
production inference should use `best.pt`.

Committed run: 29 epochs completed with early stopping, best epoch 9. Exact
synthetic validation: Precision 0.9118, Recall 0.9320, mAP50 0.9548,
mAP50-95 0.5839. See `metrics.json` for per-class results and hashes.

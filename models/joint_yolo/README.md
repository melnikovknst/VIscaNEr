# Joint bottle + label detector

`best.pt` is the deployable YOLO11n checkpoint with two classes:

- `0`: `bottle`;
- `1`: `wine_label`.

It performs both detections in one model forward. Use it through
`joint_yolo/infer.py`; the inference layer ranks labels as detector confidence
plus a bounded `0.20 ×` proximity bonus to the vertical target axis. Vertical
position and crossing the crosshair do not affect ranking. A second pair is
returned only for two genuinely close candidates.

The training configuration is recorded in
`metrics.json`. `last.pt` is retained for reproducible continuation, while
production inference should use `best.pt`.

Current checkpoint: 150 deep-adaptation epochs, best epoch 148.

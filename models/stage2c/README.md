# Complete Stage-2C checkpoint

`manual_stage2c_best.pt` is the completed epoch-17 Stage-2C artifact exported
from Kaggle. It contains the full state dictionaries for both adapted
DINOv3-B/16 branches (label and whole bottle), plus the earlier fusion state.

The Stage-2C residual Transformer uses only the two DINO branch state
dictionaries and freezes every DINO parameter. The checkpoint is stored with
Git LFS and verified against `stage2c_sha256.json` before Kaggle training.

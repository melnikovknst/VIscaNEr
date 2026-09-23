# Learned conditional resolver gate

`dino_s_gate.joblib` is a small scikit-learn model that decides whether the
whole-bottle DINOv3-S resolver is expected to improve DINOv3-B's Top-1 result.

The training target is utility:

- `+1`: DINO-S fixes a DINO-B Top-1 error;
- `0`: the Top-1 correctness does not change;
- `-1`: DINO-S breaks a correct DINO-B answer.

Ridge least-squares models and multinomial logistic models are trained on the
`train` split. The model and score threshold are selected on `val_seen`; only
the final selected model is evaluated on `val_unseen`. Wine identity and ground
truth are never input features. The runtime features are DINO-B Top-10 cosine
similarities/margins plus label/bottle detector confidence and bottle-crop
status.

Recreate the artifact with:

```bash
python run_dino_cascade.py --gating-strategy margin --ambiguity-margin 0.03
python train_cascade_gate.py
```

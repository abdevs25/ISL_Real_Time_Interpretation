# Training, evaluation and live demo (added files)

New files (drop them into the project root, keeping the folder layout):

| File | Role |
|---|---|
| `src/models/stgcn.py` | ST-GCN encoder (graph conv + 9x1 temporal conv, learnable edge weights) |
| `src/models/isl_model.py` | hand ST-GCN + body ST-GCN -> your FusionModel (cross-attention) -> BiLSTM; checkpoint save/load |
| `src/training/train_e2e.py` | real training on `skeleton_data`; metrics; ablation modes; seeds |
| `evaluate.py` | re-evaluate any checkpoint (accuracy, top-5, precision, recall, F1, confusion matrix) |
| `src/inference/segmenter.py` | cuts one clip per sign from the live stream (hands up -> hands down) |
| `src/inference/engine.py` | clip -> word + confidence + attention |
| `live_demo.py` | OpenCV webcam demo |
| `server.py` | optional FastAPI WebSocket backend for the web front end |
| `test_inference.py` | tests for segmenter (+ model shapes when torch is installed) |

## Order of commands (PowerShell, project root)

```powershell
pip install -r requirements.txt
python test_inference.py                                  # must print ok for every test, incl. "model shapes OK"
python -m src.training.train_e2e --smoke                  # 1-2 minutes, fake data, checks the whole wiring

# real run, ~50 signs first
python -m src.training.train_e2e --num_classes 50 --mode fused --seed 0 --amp   # --amp only with an NVIDIA GPU
python evaluate.py --ckpt checkpoints/e2e_fused_c50_seed0.pt --split test
python live_demo.py --ckpt checkpoints/e2e_fused_c50_seed0.pt
```

CPU only and too slow? add `--width 0.5` (half-width ST-GCN) and/or `--batch_size 16`.

Ablation table for the slides (hand / body / concat / fused, 3 seeds each):

```powershell
foreach ($m in "hand","body","concat","fused") { foreach ($s in 0,1,2) {
  python -m src.training.train_e2e --num_classes 50 --mode $m --seed $s } }
python -m src.training.train_e2e --summarize
```

## Choosing the classes
`--num_classes N` keeps the N signs with the most training clips (ties alphabetical).
For an exact list (e.g. INCLUDE-50) put one label per line in a text file and use `--class_list file.txt`
(labels must match `labels.json` exactly, e.g. `1. loud`).

## Honest-reporting notes
* Use `make_split.py --by-session` (your patched version) so sibling takes do not leak across splits. This is
  take-independent, NOT signer-independent.
* The test split is evaluated once per run, with the best-validation weights.
* Left/right mirror augmentation is OFF by default (`--flip_prob 0`): sign handedness can carry meaning.
* In the demo, "no sign" is handled by the segmenter + confidence threshold, not by a class. Words below
  `--threshold` show as "(unsure)" and are not added to the sentence.

## Demo tuning
* Signs are cut when hands go above `--raise_y` (shoulder-widths below the shoulder midpoint; default 1.6)
  and end after `--end_gap` idle frames. If it never triggers/never stops, change `--raise_y` (try 1.2-2.2).
* SPACE = manual start/stop. Use it as the safety net during the presentation.
* Do not mirror the camera image; training videos are not mirrored.

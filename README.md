# Cross-Attentive Skeleton-Based Recognition for Indian Sign Language (ISL)

Webcam video -> MediaPipe Holistic keypoints -> hand graph + body graph -> two ST-GCNs
-> cross-attention fusion -> BiLSTM -> predicted sign.

## Data pipeline (video -> training batches)

```
videos/Category/SignName/clip.mp4
   |  python extract_dataset.py --input videos --output skeleton_data
   v
skeleton_data/Category/SignName/clip.npz  +  manifest.csv      (50 joints x 3, normalized)
   |  python validate_dataset.py --data skeleton_data          (flags bad clips)
   |  python make_split.py --data skeleton_data                (train / val / test)
   v
skeleton_data/splits.csv + labels.json
   |  python dataset.py --data skeleton_data                   (prints one batch)
   v
DataLoader -> (hand, body, lengths, labels)
```

Setup: `pip install -r requirements.txt`

### Steps
1. Put videos in `videos/<Category>/<SignName>/<clip>.mp4`. The **SignName folder is the class label**.
2. Quick test first: `python extract_dataset.py --input videos --output skeleton_data --limit 3`
3. Full run: `python extract_dataset.py --input videos --output skeleton_data --skip-existing`
   (`--skip-existing` lets you stop with Ctrl+C and continue later.)
4. `python validate_dataset.py --data skeleton_data` and look at a clip with
   `python replay_skeleton.py skeleton_data/<Category>/<SignName>/<clip>.npz`
5. `python make_split.py --data skeleton_data` (add `--signer-regex "..."` only if signer ids are in the file names)
6. `python dataset.py --data skeleton_data` to confirm batches load.

No videos yet? `python make_fake_data.py --out fake_skeleton_data` makes fake clips so steps 5-6
can be tried. Run `python test_pipeline.py` to check graph / split / dataset code.

## Files for the data pipeline

| File | What it does |
|---|---|
| `skeleton_def.py` | The 50 joints and the bone lists (hand graph, body graph) |
| `stream_processor.py` | Holds short detection gaps (hand 5 frames, body 30) without looking at future frames |
| `normalize.py` | Shoulder-midpoint origin, shoulder-width scale; missing joints = 0 |
| `extract_dataset.py` | Batch video -> `.npz` skeleton per clip + `manifest.csv` |
| `validate_dataset.py`, `replay_skeleton.py` | Quality check and visual playback |
| `graph.py` | Adjacency matrices: hand (3,42,42) and body (3,8,8) |
| `make_split.py` | train/val/test split (`splits.csv`, `labels.json`) |
| `dataset.py` | PyTorch Dataset / DataLoader for the ST-GCN |

## Interface for the ST-GCN

```python
from dataset import make_loaders
from graph import build_graphs

loaders, info = make_loaders("skeleton_data", batch_size=32)   # info["num_classes"], info["labels"]
hand, body, lengths, labels = next(iter(loaders["train"]))
# hand (N, 3, 64, 42)   body (N, 3, 64, 8)   -> (N, C, T, V)
# lengths = real frame count (rest is zero padding),  labels = class index
g = build_graphs()
g["hand"]["A"]   # (3, 42, 42)       g["body"]["A"]   # (3, 8, 8)
```

## Notes
- `.npz` contents: `skeleton` (T,50,3), `status` (T,50: 0 missing, 1 detected, 2 held), plus detection rates.
- Clips longer than 64 frames are resampled to 64; shorter ones are zero-padded (`max_T` in `dataset.py`).
  The live demo uses a 30-frame window, so keep the two consistent when the model is wired in.
- The split is signer-independent only when `--signer-regex` is used. Otherwise it is a random split per sign
  and the script says so.
- Mirror augmentation is off by default (`flip_prob=0`): turn it on only if left/right handedness
  does not change the meaning of a sign.
- `checkpoints/*.pt` were trained on synthetic dummy data, not on INCLUDE.

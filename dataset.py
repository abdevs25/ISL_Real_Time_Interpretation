"""
dataset.py - PyTorch Dataset / DataLoader for the extracted ISL skeletons.

This is the interface the ST-GCN (Adithya) trains on.

    ds = SkeletonSignDataset("skeleton_data", split="train")
    hand, body, length, label = ds[0]

        hand   (3, max_T, 42)   float32   channels = x, y, z ;  T ; hand joints
        body   (3, max_T, 8)    float32   channels = x, y, z ;  T ; body joints
        length int              number of REAL frames (the rest is zero padding)
        label  int              class number (see ds.labels)

The layout (C, T, V) is what the standard ST-GCN code expects. Add the batch
dimension and you get (N, C, T, V). Use graph.py for the matching adjacency
matrices (hand: 42 nodes, body: 8 nodes).

Batches from the DataLoader come out as (hand, body, lengths, labels), the same
order train.py already uses.

What happens to each clip
    1. load the normalized skeleton (T, 50, 3) saved by extract_dataset.py
       (joints 0-41 = hands, 42-49 = body; missing joints are already 0)
    2. training only: light augmentation (scale, small rotation, shift, jitter,
       optional left/right mirror, random trim of the clip ends)
    3. longer than max_T -> resampled evenly down to max_T frames;
       shorter -> zero-padded and `length` records the real frame count

Needs: skeleton_data/splits.csv and labels.json (run make_split.py first).
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from skeleton_def import FLIP_PERM, BODY_OFFSET

DEFAULT_MAX_T = 64      # INCLUDE clips are usually 2-4 s (about 50-120 frames)


def resolve_npz(data_dir, rel):
    """Find a clip file even if the folder was moved between computers."""
    data_dir = Path(data_dir)
    parts = str(rel).replace("\\", "/").split("/")
    candidates = [data_dir.joinpath(*parts)]
    if data_dir.name in parts:                       # path included the folder name
        i = len(parts) - 1 - parts[::-1].index(data_dir.name)
        candidates.append(data_dir.joinpath(*parts[i + 1:]))
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"clip not found: {rel} (looked under {data_dir})")


def fit_length(skel, max_T):
    """(T,50,3) -> (max_T,50,3) by even resampling (if longer) or zero padding."""
    T = len(skel)
    if T > max_T:
        idx = np.linspace(0, T - 1, max_T).round().astype(int)
        return skel[idx], max_T
    if T < max_T:
        pad = np.zeros((max_T - T,) + skel.shape[1:], dtype=skel.dtype)
        return np.concatenate([skel, pad], axis=0), T
    return skel, T


def augment_clip(skel, status, rng, flip_prob=0.0):
    """
    Light, shape-preserving augmentation on a (T,50,3) clip.
    Joints that were never detected (status 0, stored as zeros) stay exactly 0.
    """
    skel = skel.copy()
    T = len(skel)

    # random trim: keep 90-100 % of the frames
    keep = int(round(T * rng.uniform(0.9, 1.0)))
    keep = max(1, min(T, keep))
    start = rng.integers(0, T - keep + 1)
    skel, status = skel[start:start + keep], status[start:start + keep]

    present = (status > 0)[..., None]                       # (T,50,1)

    if flip_prob > 0 and rng.random() < flip_prob:          # mirror left <-> right
        skel = skel[:, FLIP_PERM].copy()
        present = present[:, FLIP_PERM]
        skel[..., 0] = -skel[..., 0]

    theta = np.deg2rad(rng.uniform(-10, 10))                # small rotation in x-y
    c, s = np.cos(theta), np.sin(theta)
    x, y = skel[..., 0].copy(), skel[..., 1].copy()
    skel[..., 0], skel[..., 1] = c * x - s * y, s * x + c * y

    skel *= rng.uniform(0.9, 1.1)                           # scale
    skel[..., :2] += rng.uniform(-0.05, 0.05, size=2)       # shift
    skel += rng.normal(0.0, 0.005, size=skel.shape)         # jitter

    skel = np.where(present, skel, 0.0).astype(np.float32)  # keep missing joints at 0
    return skel


class SkeletonSignDataset(Dataset):
    def __init__(self, data_dir, split, max_T=DEFAULT_MAX_T, augment=False,
                 flip_prob=0.0, seed=0):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_T = max_T
        self.augment = augment
        self.flip_prob = flip_prob
        self.rng = np.random.default_rng(seed)

        with open(self.data_dir / "labels.json", encoding="utf-8") as f:
            self.labels = json.load(f)["labels"]
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

        with open(self.data_dir / "splits.csv", newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == split]

        # Clips are small (T x 50 x 3 floats) so keep them all in memory.
        self.items = []
        for r in rows:
            d = np.load(resolve_npz(self.data_dir, r["npz_path"]))
            self.items.append((
                d["skeleton"].astype(np.float32),
                d["status"],
                self.label_to_idx[r["label"]],
            ))

    @property
    def num_classes(self):
        return len(self.labels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        skel, status, label = self.items[i]
        if self.augment:
            skel = augment_clip(skel, status, self.rng, self.flip_prob)
        skel, length = fit_length(skel, self.max_T)

        # (T, V, C) -> (C, T, V), split into the two graphs
        hand = np.ascontiguousarray(skel[:, :BODY_OFFSET].transpose(2, 0, 1))
        body = np.ascontiguousarray(skel[:, BODY_OFFSET:].transpose(2, 0, 1))
        return (torch.from_numpy(hand), torch.from_numpy(body),
                torch.tensor(length, dtype=torch.long),
                torch.tensor(label, dtype=torch.long))


def make_loaders(data_dir="skeleton_data", batch_size=32, max_T=DEFAULT_MAX_T,
                 num_workers=0, augment_train=True, flip_prob=0.0, seed=0):
    """
    Returns (loaders, info).
        loaders = {"train": DataLoader, "val": DataLoader or None, "test": ... }
        info    = {"labels": [...], "num_classes": int, "sizes": {...}}
    A split with no clips gives None instead of a loader.
    """
    loaders, sizes = {}, {}
    labels = None
    for split in ("train", "val", "test"):
        ds = SkeletonSignDataset(data_dir, split, max_T=max_T,
                                 augment=(augment_train and split == "train"),
                                 flip_prob=flip_prob, seed=seed)
        labels = ds.labels
        sizes[split] = len(ds)
        if len(ds) == 0:
            print(f"WARNING: '{split}' split is empty")
            loaders[split] = None
            continue
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split == "train"),
                                    num_workers=num_workers)
    return loaders, {"labels": labels, "num_classes": len(labels), "sizes": sizes}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Print one batch from each split.")
    ap.add_argument("--data", default="skeleton_data")
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    loaders, info = make_loaders(a.data, batch_size=a.batch_size)
    print("classes:", info["num_classes"], " sizes:", info["sizes"])
    for split, loader in loaders.items():
        if loader is None:
            continue
        hand, body, lengths, labels = next(iter(loader))
        print(f"{split:5s} hand {tuple(hand.shape)} body {tuple(body.shape)} "
              f"lengths {tuple(lengths.shape)} labels {tuple(labels.shape)}")

"""
test_pipeline.py - checks for graph.py, make_split.py and dataset.py using FAKE skeleton data
(no videos, no webcam needed).

    python test_pipeline.py          (or: pytest test_pipeline.py)
"""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

import make_fake_data
import make_split as ms
import validate_dataset as vd
from graph import build_graphs, hop_distance
from skeleton_def import BODY_OFFSET


# ------------------------------------------------------------------ helpers
def _fake_folder(tmp, signs=4, signers=5, clips=3):
    out = Path(tmp) / "skeleton_data"
    import sys
    argv = sys.argv
    sys.argv = ["x", "--out", str(out), "--signs", str(signs),
                "--signers", str(signers), "--clips-per-signer", str(clips)]
    try:
        make_fake_data.main()
    finally:
        sys.argv = argv
    return out


def _run_split(out, signer_regex=None):
    rows = ms.load_manifest(out)
    for r in rows:
        r["npz_path"] = ms.clean_path(out, r["npz_path"])
    rows, indep = ms.make_splits(rows, 0.15, 0.15, seed=0, signer_regex=signer_regex)
    labels = sorted({r["label"] for r in rows})
    with open(out / "labels.json", "w") as f:
        json.dump({"labels": labels}, f)
    with open(out / "splits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["npz_path", "label", "signer", "split"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("npz_path", "label", "signer", "split")})
    return rows, indep


# ------------------------------------------------------------------ graph.py
def test_graph_shapes_and_properties():
    g = build_graphs()
    assert g["hand"]["A"].shape == (3, 42, 42)
    assert g["body"]["A"].shape == (3, 8, 8)
    assert g["hand"]["A_plain"].shape == (42, 42)

    # every node receives weights that add up to 1 (columns normalized)
    for name in ("hand", "body"):
        A = g[name]["A"]
        assert np.allclose(A.sum(axis=(0, 1)), 1.0, atol=1e-5), name
        assert np.allclose(g[name]["A_plain"].sum(axis=0), 1.0, atol=1e-5), name

    # hand graph: the two hands are NOT connected to each other
    A_hand = g["hand"]["A"].sum(axis=0)
    assert A_hand[:21, 21:].sum() == 0 and A_hand[21:, :21].sum() == 0

    # a bone exists where skeleton_def says so (thumb: wrist 0 - joint 1)
    assert (A_hand[0, 1] > 0) and (A_hand[1, 0] > 0)

    # every partition is used in the hand graph (same / closer / farther)
    for k in range(3):
        assert g["hand"]["A"][k].sum() > 0, f"hand partition {k} is empty"

    # distances from the wrist: fingertip 4 (thumb) is 4 bones away
    d = hop_distance(42, g["hand"]["edges"], g["hand"]["roots"])
    assert d[0] == 0 and d[21] == 0 and d[4] == 4


# ------------------------------------------------------------------ make_split.py
def test_random_split_covers_every_sign_and_has_no_overlap():
    with tempfile.TemporaryDirectory() as tmp:
        out = _fake_folder(tmp)
        rows, indep = _run_split(out)
        assert indep is False
        assert len({r["npz_path"] for r in rows}) == len(rows)          # nothing duplicated
        for label in {r["label"] for r in rows}:
            splits = {r["split"] for r in rows if r["label"] == label}
            assert {"train", "val", "test"} <= splits, (label, splits)


def test_signer_independent_split_has_disjoint_signers():
    with tempfile.TemporaryDirectory() as tmp:
        out = _fake_folder(tmp)
        rows, indep = _run_split(out, signer_regex=r"signer(\d+)")
        assert indep is True
        by_split = {s: {r["signer"] for r in rows if r["split"] == s}
                    for s in ("train", "val", "test")}
        assert by_split["train"] and by_split["test"]
        assert not (by_split["train"] & by_split["test"])
        assert not (by_split["train"] & by_split["val"])
        assert not (by_split["val"] & by_split["test"])


def test_tiny_dataset_does_not_crash():
    # e.g. only 2 signs with 2-3 clips each (the situation with very few videos)
    assert ms.counts_for(1, 0.15, 0.15) == (1, 0, 0)
    assert ms.counts_for(2, 0.15, 0.15) == (1, 0, 1)
    tr, va, te = ms.counts_for(3, 0.15, 0.15)
    assert tr >= 1 and tr + va + te == 3
    tr, va, te = ms.counts_for(20, 0.15, 0.15)
    assert (tr, va, te) == (14, 3, 3)


def test_clean_path_handles_windows_and_relative():
    assert ms.clean_path("skeleton_data", "skeleton_data\\Animals\\Dog\\a.npz") == "Animals/Dog/a.npz"
    assert ms.clean_path("skeleton_data", "skeleton_data/Animals/Dog/a.npz") == "Animals/Dog/a.npz"
    assert ms.clean_path("/abs/skeleton_data", "Animals/Dog/a.npz") == "Animals/Dog/a.npz"


# ------------------------------------------------------------------ validate_dataset.py on fake manifest
def test_validate_dataset_accepts_fake_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        out = _fake_folder(tmp, signs=2, signers=3, clips=2)
        cfg = dict(min_frames=8, min_body_rate=0.9, min_valid_rate=0.9, min_either_hand_rate=0.3)
        flagged, rows = vd.run(out, cfg)
        assert len(rows) == 12 and flagged == []


# ------------------------------------------------------------------ dataset.py (needs torch)
def test_dataset_and_dataloader():
    import torch
    from dataset import SkeletonSignDataset, make_loaders, fit_length, augment_clip

    with tempfile.TemporaryDirectory() as tmp:
        out = _fake_folder(tmp)
        _run_split(out, signer_regex=r"signer(\d+)")

        ds = SkeletonSignDataset(out, "train", max_T=64)
        hand, body, length, label = ds[0]
        assert tuple(hand.shape) == (3, 64, 42)
        assert tuple(body.shape) == (3, 64, 8)
        assert 1 <= int(length) <= 64
        assert 0 <= int(label) < ds.num_classes == 4
        # frames after `length` are zero padding
        assert float(hand[:, int(length):].abs().sum()) == 0.0
        assert float(body[:, int(length):].abs().sum()) == 0.0

        loaders, info = make_loaders(out, batch_size=8, max_T=64)
        assert info["num_classes"] == 4 and sum(info["sizes"].values()) == 4 * 5 * 3
        h, b, lengths, y = next(iter(loaders["train"]))
        assert tuple(h.shape)[1:] == (3, 64, 42) and tuple(b.shape)[1:] == (3, 64, 8)
        assert h.dtype == torch.float32 and y.dtype == torch.long
        assert len(lengths) == len(y) == h.shape[0]

        # evaluation data is deterministic (no augmentation)
        ds_val = SkeletonSignDataset(out, "val", max_T=64)
        a, _, _, _ = ds_val[0]
        c, _, _, _ = ds_val[0]
        assert torch.equal(a, c)

    # fit_length: long clips are resampled, short ones padded
    long_clip = np.random.rand(200, 50, 3).astype(np.float32)
    short_clip = np.random.rand(10, 50, 3).astype(np.float32)
    s, n = fit_length(long_clip, 64)
    assert s.shape == (64, 50, 3) and n == 64
    s, n = fit_length(short_clip, 64)
    assert s.shape == (64, 50, 3) and n == 10 and np.all(s[10:] == 0)

    # augmentation never invents joints: missing joints stay exactly 0
    skel = np.random.rand(30, 50, 3).astype(np.float32)
    status = np.ones((30, 50), dtype=np.int8)
    status[:, 0:21] = 0
    skel[:, 0:21] = 0
    aug = augment_clip(skel, status, np.random.default_rng(0), flip_prob=0.0)
    assert np.all(aug[:, 0:21] == 0) and np.abs(aug[:, 42:]).sum() > 0
    aug_flip = augment_clip(skel, status, np.random.default_rng(0), flip_prob=1.0)
    assert np.all(aug_flip[:, 21:42] == 0)      # left hand was missing -> after mirror, right hand is


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nall {len(tests)} checks passed")

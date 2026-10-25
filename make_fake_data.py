"""
make_fake_data.py - make FAKE skeleton clips in the same format extract_dataset.py writes,
so the rest of the pipeline can be tested without any videos.

    python make_fake_data.py --out fake_skeleton_data
    python make_split.py --data fake_skeleton_data --signer-regex "signer(\\d+)"
    python dataset.py --data fake_skeleton_data

The clips are NOT real signs: each fake "sign" is just a different hand motion
pattern. They only exist to prove the file formats, split and DataLoader work.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from skeleton_def import BODY_OFFSET, RIGHT_HAND_OFFSET

MANIFEST_FIELDS = ["npz_path", "category", "label", "num_frames", "fps", "width",
                   "height", "body_rate", "left_rate", "right_rate", "valid_rate"]


def fake_clip(cls, rng, T):
    """(T,50,3) skeleton + (T,50) status for fake sign number `cls`."""
    skel = np.zeros((T, 50, 3), dtype=np.float32)
    status = np.ones((T, 50), dtype=np.int8)
    t = np.linspace(0, 1, T, dtype=np.float32)

    # body: shoulders at +-0.5 (normalized), eyes/mouth above, elbows below
    body = np.array([[0.2, -0.9], [-0.2, -0.9], [0.1, -0.5], [-0.1, -0.5],
                     [0.5, 0.0], [-0.5, 0.0], [0.7, 0.9], [-0.7, 0.9]], dtype=np.float32)
    skel[:, BODY_OFFSET:, :2] = body + rng.normal(0, 0.01, (T, 8, 2))

    # hands: 21 joints = wrist + fixed finger offsets, wrist follows a class-specific path
    fingers = np.linspace(0, 1, 21, dtype=np.float32)[:, None] * np.array([0.0, -0.25], dtype=np.float32)
    freq, amp = 1 + cls, 0.2 + 0.05 * cls
    for off, side in ((0, 1.0), (RIGHT_HAND_OFFSET, -1.0)):
        wx = side * (0.7 + amp * np.sin(2 * np.pi * freq * t))
        wy = 0.9 - amp * np.cos(2 * np.pi * freq * t + cls)
        skel[:, off:off + 21, 0] = wx[:, None] + fingers[None, :, 0]
        skel[:, off:off + 21, 1] = wy[:, None] + fingers[None, :, 1]
    skel += rng.normal(0, 0.005, skel.shape).astype(np.float32)

    # simulate a few frames where a hand was not detected
    for off in (0, RIGHT_HAND_OFFSET):
        lost = rng.random(T) < 0.1
        status[lost, off:off + 21] = 0
        skel[lost, off:off + 21] = 0.0
    return skel, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fake_skeleton_data")
    ap.add_argument("--signs", type=int, default=4)
    ap.add_argument("--signers", type=int, default=5)
    ap.add_argument("--clips-per-signer", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    rows = []
    for c in range(a.signs):
        label = f"Sign{c + 1}"
        for s in range(a.signers):
            for k in range(a.clips_per_signer):
                T = int(rng.integers(40, 100))
                skel, status = fake_clip(c, rng, T)
                path = out / "FakeCategory" / label / f"signer{s}_clip{k}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                body_rate = float((status[:, BODY_OFFSET:] > 0).all(axis=1).mean())
                left = float((status[:, 0] > 0).mean())
                right = float((status[:, RIGHT_HAND_OFFSET] > 0).mean())
                np.savez_compressed(
                    path, skeleton=skel, status=status, fps=np.float32(30),
                    source_fps=np.float32(30), width=np.int32(640), height=np.int32(480),
                    num_frames=np.int32(T), body_rate=np.float32(body_rate),
                    left_rate=np.float32(left), right_rate=np.float32(right),
                    valid_rate=np.float32(1.0))
                rows.append({"npz_path": str(path), "category": "FakeCategory", "label": label,
                             "num_frames": T, "fps": 30.0, "width": 640, "height": 480,
                             "body_rate": body_rate, "left_rate": left, "right_rate": right,
                             "valid_rate": 1.0})
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} fake clips + manifest.csv to {out}")


if __name__ == "__main__":
    main()

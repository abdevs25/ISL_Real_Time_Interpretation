"""Tests for Step 7: replay_skeleton.py (headless render) and validate_dataset.py."""

import csv
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

import replay_skeleton as rs
import validate_dataset as vd
from skeleton_def import BODY_OFFSET

MISSING, DETECTED, HELD = 0, 1, 2

# ---------------------------------------------------------------------
# replay_skeleton: build a small synthetic sequence, render it, check the
# output video actually contains the right number of readable frames.
# ---------------------------------------------------------------------
T = 12
skeleton = np.zeros((T, 50, 3), dtype=np.float32)
status = np.ones((T, 50), dtype=np.int8)
skeleton[:, BODY_OFFSET + 4] = [0.5, 0, 0]    # L shoulder
skeleton[:, BODY_OFFSET + 5] = [-0.5, 0, 0]   # R shoulder
skeleton[:, 0] = [0.2, 0.3, 0]                # left wrist
status[5:8, 21:42] = MISSING                  # right hand missing for a few frames
status[8:10, 0:21] = HELD                     # left hand held for a couple frames

img = rs.render_frame(skeleton[0], status[0])
assert img.shape == (rs.CANVAS[1], rs.CANVAS[0], 3)
assert img.dtype == np.uint8
assert img.any(), "rendered frame should not be blank when joints are present"

img_missing = rs.render_frame(np.zeros((50, 3), np.float32),
                              np.zeros(50, dtype=np.int8))
# an all-missing frame should still render (just the text overlay, no skeleton)
assert img_missing.shape == img.shape

tmp = Path(tempfile.mkdtemp())
try:
    npz_path = tmp / "sample.npz"
    np.savez_compressed(npz_path, skeleton=skeleton, status=status,
                        fps=np.float32(10), source_fps=np.float32(10),
                        width=np.int32(320), height=np.int32(240),
                        num_frames=np.int32(T), body_rate=np.float32(1.0),
                        left_rate=np.float32(0.9), right_rate=np.float32(0.7),
                        valid_rate=np.float32(1.0))

    sk, st, fps = rs.load_sequence(npz_path)
    assert sk.shape == (T, 50, 3) and st.shape == (T, 50) and fps == 10.0

    out_video = tmp / "replay.mp4"
    rs.save_video(sk, st, fps, out_video)
    assert out_video.exists() and out_video.stat().st_size > 0

    cap = cv2.VideoCapture(str(out_video))
    n_read = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_read += 1
    cap.release()
    assert n_read == T, f"expected {T} frames in output video, got {n_read}"

    print("replay_skeleton tests passed.")

    # -------------------------------------------------------------
    # validate_dataset: build a fake manifest with one good and a few
    # deliberately bad rows, check each flagging rule fires correctly.
    # -------------------------------------------------------------
    data_dir = tmp / "dataset"
    data_dir.mkdir()
    fields = ["npz_path", "category", "label", "num_frames", "fps", "width",
              "height", "body_rate", "left_rate", "right_rate", "valid_rate"]
    rows = [
        dict(npz_path="a/good.npz", category="a", label="good", num_frames=30,
             fps=30, width=320, height=240, body_rate=0.99, left_rate=0.8,
             right_rate=0.1, valid_rate=0.99),
        dict(npz_path="a/short.npz", category="a", label="short", num_frames=3,
             fps=30, width=320, height=240, body_rate=0.99, left_rate=0.8,
             right_rate=0.1, valid_rate=0.99),
        dict(npz_path="a/nobody.npz", category="a", label="nobody", num_frames=30,
             fps=30, width=320, height=240, body_rate=0.2, left_rate=0.8,
             right_rate=0.1, valid_rate=0.2),
        dict(npz_path="a/nohands.npz", category="a", label="nohands", num_frames=30,
             fps=30, width=320, height=240, body_rate=0.99, left_rate=0.02,
             right_rate=0.01, valid_rate=0.99),
    ]
    with open(data_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    flagged, all_rows = vd.run(data_dir, vd.DEFAULTS)
    flagged_labels = {r["label"] for r in flagged}
    assert flagged_labels == {"short", "nobody", "nohands"}, flagged_labels
    assert "good" not in flagged_labels

    short_reasons = next(r["reasons"] for r in flagged if r["label"] == "short")
    assert "too short" in short_reasons
    nobody_reasons = next(r["reasons"] for r in flagged if r["label"] == "nobody")
    assert "low body detection" in nobody_reasons and "low valid-frame rate" in nobody_reasons
    nohands_reasons = next(r["reasons"] for r in flagged if r["label"] == "nohands")
    assert "very low hand detection" in nohands_reasons

    assert (data_dir / "flagged.csv").exists()
    with open(data_dir / "flagged.csv", encoding="utf-8") as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == 3

    # a run with nothing flagged should remove a stale flagged.csv
    clean_dir = tmp / "clean"
    clean_dir.mkdir()
    with open(clean_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(rows[0])
    (clean_dir / "flagged.csv").write_text("stale")
    flagged2, _ = vd.run(clean_dir, vd.DEFAULTS)
    assert flagged2 == [] and not (clean_dir / "flagged.csv").exists()

    print("validate_dataset tests passed.")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

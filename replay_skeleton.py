"""
replay_skeleton.py - Step 7: play back a saved .npz skeleton sequence, so you
can visually confirm the training data looks like a real sign (and matches
what the live webcam view showed for Steps 2-4).

The .npz stores NORMALIZED coordinates (shoulder midpoint = origin, shoulder
width = 1.0), which have no fixed pixel scale. This script un-normalizes them
back onto a blank canvas purely for display: shoulder width -> a fixed number
of pixels, and it's flipped/placed so the pose reads the same way the webcam
view did.

Usage:
    # live window (needs a display - run on your own machine, not a server)
    python replay_skeleton.py skeleton_data/Animals/Dog/MVI_0001.npz

    # save to a video file instead (works with no display, e.g. to send someone)
    python replay_skeleton.py skeleton_data/Animals/Dog/MVI_0001.npz --out review.mp4

Keys (live window): space = pause/resume, n = step one frame while paused, q = quit.
"""

import argparse

import cv2
import numpy as np

from skeleton_def import EDGES, RIGHT_HAND_OFFSET, BODY_OFFSET, NUM_JOINTS

MISSING, DETECTED, HELD = 0, 1, 2
CANVAS = (640, 640)
SHOULDER_PX = 160.0   # on-screen pixels per normalized unit (= shoulder width)


def joint_color(j):
    if j < RIGHT_HAND_OFFSET:
        return (255, 0, 0)      # left hand  - blue (BGR)
    if j < BODY_OFFSET:
        return (0, 0, 255)      # right hand - red
    return (0, 255, 0)          # body/face  - green


def render_frame(joints, status, canvas_size=CANVAS, px_per_unit=SHOULDER_PX):
    """joints (50,3) normalized, status (50,) -> a BGR image."""
    w, h = canvas_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = w // 2, h // 2

    pts = {}
    for j in range(NUM_JOINTS):
        if status[j] == MISSING:
            continue
        x = cx + int(joints[j, 0] * px_per_unit)
        y = cy + int(joints[j, 1] * px_per_unit)
        pts[j] = (x, y)

    for a, b in EDGES:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], joint_color(a), 2)
    for j, p in pts.items():
        col = (140, 140, 140) if status[j] == HELD else joint_color(j)
        cv2.circle(img, p, 4, col, -1)

    n_present = int((status != MISSING).sum())
    n_held = int((status == HELD).sum())
    cv2.putText(img, f"joints {n_present}/50 (held {n_held})", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def load_sequence(npz_path):
    with np.load(npz_path) as z:
        return (z["skeleton"].astype(np.float32), z["status"].astype(np.int8),
                float(z["fps"]))


def play_live(skeleton, status, fps, canvas=CANVAS, px=SHOULDER_PX):
    T = len(skeleton)
    i, paused = 0, False
    delay = max(1, int(1000 / max(fps, 1)))
    while True:
        img = render_frame(skeleton[i], status[i], canvas, px)
        cv2.putText(img, f"frame {i + 1}/{T}", (10, canvas[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Skeleton replay (space=pause, n=step, q=quit)", img)
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif paused and key == ord("n"):
            i = min(i + 1, T - 1)
        elif not paused:
            i = (i + 1) % T
    cv2.destroyAllWindows()


def save_video(skeleton, status, fps, out_path, canvas=CANVAS, px=SHOULDER_PX):
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                              max(fps, 1), canvas)
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")
    for i in range(len(skeleton)):
        img = render_frame(skeleton[i], status[i], canvas, px)
        cv2.putText(img, f"frame {i + 1}/{len(skeleton)}", (10, canvas[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(img)
    writer.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz_path")
    ap.add_argument("--out", default=None, help="save to this video file instead of a live window")
    a = ap.parse_args()

    skeleton, status, fps = load_sequence(a.npz_path)
    print(f"Loaded {a.npz_path}: {len(skeleton)} frames @ {fps:.1f} fps")

    if a.out:
        save_video(skeleton, status, fps, a.out)
        print(f"Saved replay to {a.out}")
    else:
        play_live(skeleton, status, fps)


if __name__ == "__main__":
    main()
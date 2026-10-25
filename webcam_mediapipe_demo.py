"""
MC-CASLR (ISL project) - live real-time skeleton pipeline (Steps 1-3, streaming).

Per frame:
    webcam -> MediaPipe Holistic -> 50-joint (50,3) array
           -> CausalFiller (hold short gaps, no look-ahead)
           -> SlidingWindow (last WINDOW_T frames)
           -> every WINDOW_STRIDE frames a (T, 50, 3) window is ready for the model

Keys:
    v  - toggle view: 50-joint skeleton  <->  full MediaPipe drawing
    p  - print the current (50, 3) array to the console
    q  - quit
"""

import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from skeleton_def import (
    EDGES, JOINT_NAMES, NUM_JOINTS, extract_frame,
    RIGHT_HAND_OFFSET, BODY_OFFSET,
)
from stream_processor import CausalFiller, SlidingWindow, MISSING, HELD
from normalize import normalize_frame

# ---- real-time settings (tune later; must match the training pipeline) ----
WINDOW_T = 30          # frames per window given to the model
WINDOW_STRIDE = 5      # emit a new window every N frames
MAX_HOLD_HANDS = 5     # frames a lost hand keeps its last position

# ---------------------------------------------------------------------
# 1. Download the model (only once)
# ---------------------------------------------------------------------
MODEL_PATH = "holistic_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading holistic landmarker model (first run only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

# ---------------------------------------------------------------------
# 2. Set up the HolisticLandmarker task
# ---------------------------------------------------------------------
BaseOptions = mp_python.BaseOptions
HolisticLandmarker = mp_vision.HolisticLandmarker
HolisticLandmarkerOptions = mp_vision.HolisticLandmarkerOptions
VisionRunningMode = mp_vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    min_face_detection_confidence=0.5,
    min_pose_detection_confidence=0.5,
    min_hand_landmarks_confidence=0.5,
)

# ---------------------------------------------------------------------
# 3. Drawing helpers
# ---------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def draw_landmarks(frame, landmarks, connections, color=(0, 255, 0)):
    if not landmarks:
        return
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in connections:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], color, 2)
    for p in points:
        cv2.circle(frame, p, 3, color, -1)


def joint_color(j):
    if j < RIGHT_HAND_OFFSET:
        return (255, 0, 0)      # left hand  - blue (BGR)
    if j < BODY_OFFSET:
        return (0, 0, 255)      # right hand - red
    return (0, 255, 0)          # body/face  - green


def draw_skeleton50(frame, joints, status):
    """Draw the 50 joints from the numeric array. Held joints are drawn grey."""
    h, w, _ = frame.shape
    pts = {}
    for j in range(NUM_JOINTS):
        if status[j] != MISSING:
            pts[j] = (int(joints[j, 0] * w), int(joints[j, 1] * h))
    for a, b in EDGES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], joint_color(a), 2)
    for j, p in pts.items():
        col = (160, 160, 160) if status[j] == HELD else joint_color(j)
        cv2.circle(frame, p, 3, col, -1)
    if 0 in pts:
        cv2.putText(frame, "L-wrist", pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1)
    if 21 in pts:
        cv2.putText(frame, "R-wrist", pts[21], cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)


def on_window(seq, status):
    """
    Called every WINDOW_STRIDE frames with the newest window.
    seq: (WINDOW_T, 50, 3) NORMALIZED coords, no NaN (missing joints = 0).
    status: (WINDOW_T, 50) int8: 0 missing, 1 detected, 2 held.
    -> this is the ML-ready array; the team's model consumes it here.
    """
    global windows_emitted
    windows_emitted += 1


# ---------------------------------------------------------------------
# 4. Run on the webcam
# ---------------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions/index.")

start_time = time.time()
show_50 = True
filler = CausalFiller(MAX_HOLD_HANDS)
window = SlidingWindow(WINDOW_T, WINDOW_STRIDE)
windows_emitted = 0
fps, last_t = 0.0, time.time()

with HolisticLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # ---- per-frame streaming pipeline ----
        joints, mask = extract_frame(result)              # raw (50,3), detected mask
        filled, status = filler.step(joints, mask)        # causal gap handling
        aspect = frame.shape[1] / frame.shape[0]
        norm, nstatus = normalize_frame(filled, status, aspect)   # Step 4
        window.push(norm, nstatus)
        if window.ready():
            seq, seq_status = window.get()                # (T, 50, 3)
            on_window(seq, seq_status)

        if show_50:
            draw_skeleton50(frame, filled, status)
        else:
            draw_landmarks(frame, result.pose_landmarks, POSE_CONNECTIONS, (0, 255, 0))
            draw_landmarks(frame, result.left_hand_landmarks, HAND_CONNECTIONS, (255, 0, 0))
            draw_landmarks(frame, result.right_hand_landmarks, HAND_CONNECTIONS, (0, 0, 255))

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)
        last_t = now

        line1 = (f"FPS {fps:4.1f} | detected {int(mask.sum())}/50 | "
                 f"L-hand {'Y' if mask[0] else 'N'} R-hand {'Y' if mask[21] else 'N'} "
                 f"body {int(mask[42:].sum())}/8")
        line2 = f"window {len(window)}/{WINDOW_T} | windows emitted {windows_emitted}"
        cv2.putText(frame, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        cv2.putText(frame, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)

        cv2.imshow("MC-CASLR - live 50-joint stream (v=view, p=print, q=quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("v"):
            show_50 = not show_50
        elif key == ord("p"):
            np.set_printoptions(precision=3, suppress=True)
            print(f"\n--- frame at t={timestamp_ms} ms ---")
            print("filled.shape:", filled.shape, "| detected:", int(mask.sum()), "/ 50")
            print("normalized: NaN count =", int(np.isnan(norm).sum()),
                  "| shoulder width =",
                  round(float(np.linalg.norm(norm[46, :2] - norm[47, :2])), 3))
            for j in (0, 8, 21, 29, 42, 44, 46, 48):
                tag = {0: "missing", 1: "", 2: "(held)"}[int(status[j])]
                print(f"  joint {j:2d} {JOINT_NAMES[j]:<11} raw", filled[j],
                      "-> norm", norm[j], tag)

cap.release()
cv2.destroyAllWindows()
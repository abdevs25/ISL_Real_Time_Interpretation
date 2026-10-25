"""Real-time ISL recognition demo (webcam -> skeleton -> trained model -> word).

    python live_demo.py --ckpt checkpoints/e2e_fused_c50_seed0.pt

How it works: MediaPipe Holistic -> 50 joints -> same causal fill + normalization as training
-> SignSegmenter cuts one clip per sign (hands raised -> hands lowered) -> model predicts.
Keys:  SPACE = manual start/stop of a sign (reliable fallback)   c = clear sentence
       v = hide/show skeleton                                    q = quit
Do NOT mirror the camera: the training videos are not mirrored, so left/right must match.
"""
import argparse
import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from normalize import normalize_frame
from skeleton_def import EDGES, NUM_JOINTS, RIGHT_HAND_OFFSET, BODY_OFFSET, extract_frame
from stream_processor import CausalFiller, MISSING, HELD
from src.inference.engine import SignRecognizer
from src.inference.segmenter import SignSegmenter

MODEL_PATH = "holistic_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")


def joint_color(j):
    return (255, 0, 0) if j < RIGHT_HAND_OFFSET else (0, 0, 255) if j < BODY_OFFSET else (0, 255, 0)


def draw_skeleton(frame, joints, status):
    h, w = frame.shape[:2]
    pts = {j: (int(joints[j, 0] * w), int(joints[j, 1] * h))
           for j in range(NUM_JOINTS) if status[j] != MISSING}
    for a, b in EDGES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], joint_color(a), 2)
    for j, p in pts.items():
        cv2.circle(frame, p, 3, (160, 160, 160) if status[j] == HELD else joint_color(j), -1)


def draw_panel(frame, state, result, sentence):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 150), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, state, (10, h - 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if result:
        ok = result["accepted"]
        txt = result["name"] if ok else f"(unsure) {result['name']}"
        cv2.putText(frame, txt, (10, h - 88), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if ok else (0, 165, 255), 2)
        cv2.rectangle(frame, (10, h - 76), (210, h - 62), (80, 80, 80), -1)
        cv2.rectangle(frame, (10, h - 76), (10 + int(200 * result["confidence"]), h - 62), (0, 200, 0), -1)
        cv2.putText(frame, f"{100 * result['confidence']:.0f}%", (220, h - 63),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "top: " + ", ".join(f"{n} {100 * p:.0f}%" for n, p in result["top"]),
                    (10, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        att = result["attention"]
        if att is not None:                                   # attention-over-time strip
            imp = att["time_importance"]
            strip = cv2.applyColorMap((imp[None, :] * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            strip = cv2.resize(strip, (w - 20, 16), interpolation=cv2.INTER_NEAREST)
            frame[h - 24:h - 8, 10:w - 10] = strip
            cv2.putText(frame, "attention over time", (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (200, 200, 200), 1)
    cv2.putText(frame, "Sentence: " + " ".join(sentence[-8:]), (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5, help="below this the word is shown as 'unsure'")
    ap.add_argument("--raise_y", type=float, default=1.6, help="wrist height that counts as 'signing'")
    ap.add_argument("--end_gap", type=int, default=12, help="idle frames that end a sign")
    a = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        print("Downloading holistic landmarker model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    rec = SignRecognizer(a.ckpt, threshold=a.threshold)
    print(f"loaded {a.ckpt}: {len(rec.labels)} classes, mode={rec.config['mode']}, device={rec.device}")
    seg = SignSegmenter(raise_y=a.raise_y, end_gap=a.end_gap)
    filler = CausalFiller(max_hold_hands=5, max_hold_body=30)

    options = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5, min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5)

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (check permissions / --camera index).")
    t0, last_ts = time.time(), -1
    show, manual, result, sentence = True, False, None, []

    with mp_vision.HolisticLandmarker.create_from_options(options) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(int((time.time() - t0) * 1000), last_ts + 1)
            last_ts = ts
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            joints, mask = extract_frame(res)
            filled, status = filler.step(joints, mask)
            norm, nstatus = normalize_frame(filled, status, frame.shape[1] / frame.shape[0])

            clip = seg.push(norm, nstatus)
            if clip is not None:
                manual = False
            if clip is not None:
                result = rec.predict(clip[0])
                if result["accepted"]:
                    sentence.append(result["name"])
                print(f"[{len(clip[0])} frames] -> {result['name']} ({result['confidence']:.2f})")

            if show:
                draw_skeleton(frame, filled, status)
            state = (f"MANUAL capture {seg.length} frames (SPACE to stop)" if manual else
                     f"SIGNING... {seg.length} frames" if seg.signing else
                     "idle - raise your hands to sign (or press SPACE)")
            draw_panel(frame, state, result, sentence)
            cv2.imshow("ISL live recognition (SPACE manual, c clear, v skeleton, q quit)", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("v"):
                show = not show
            elif k == ord("c"):
                sentence.clear(); result = None
            elif k == ord(" "):
                if manual:
                    clip = seg.force_finish()
                    manual = False
                    if clip is not None:
                        result = rec.predict(clip[0])
                        if result["accepted"]:
                            sentence.append(result["name"])
                else:
                    seg.begin_manual()
                    manual = True
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

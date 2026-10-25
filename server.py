"""Optional backend for the web front end (FastAPI + WebSocket). NOT covered by tests here.

    pip install fastapi uvicorn
    uvicorn server:app --host 0.0.0.0 --port 8000      (set ISL_CKPT=checkpoints/....pt first)

Protocol  ws://host:8000/ws
    client -> server : binary JPEG frames (one per message, ~15-30 fps, NOT mirrored)
                       or the text "start" / "stop" for manual capture
    server -> client : JSON  {"state": "idle|signing", "frames": int,
                              "prediction": null | {name, confidence, accepted, top:[[name,p]..],
                                                    time_importance:[...]}}
GET /labels  -> class names;   GET /health -> {"ok": true}
"""
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from normalize import normalize_frame
from skeleton_def import extract_frame
from stream_processor import CausalFiller
from src.inference.engine import SignRecognizer, display_name
from src.inference.segmenter import SignSegmenter

CKPT = os.environ.get("ISL_CKPT", "checkpoints/e2e_fused_c50_seed0.pt")
TASK = os.environ.get("ISL_TASK", "holistic_landmarker.task")
app = FastAPI(title="ISL recognition backend")
recognizer = SignRecognizer(CKPT, threshold=float(os.environ.get("ISL_THRESHOLD", "0.5")))


def new_landmarker():
    opts = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=TASK),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5, min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5)
    return mp_vision.HolisticLandmarker.create_from_options(opts)


def pack(pred):
    if pred is None:
        return None
    att = pred["attention"]
    return {"name": pred["name"], "confidence": pred["confidence"], "accepted": pred["accepted"],
            "top": [[n, p] for n, p in pred["top"]],
            "time_importance": None if att is None else att["time_importance"].tolist()}


@app.get("/health")
def health():
    return {"ok": True, "classes": len(recognizer.labels), "mode": recognizer.config["mode"]}


@app.get("/labels")
def labels():
    return [display_name(l) for l in recognizer.labels]


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    seg, filler, t0, last_ts = SignSegmenter(), CausalFiller(5, 30), time.time(), -1
    manual = False
    with new_landmarker() as lm:
        try:
            while True:
                msg = await sock.receive()
                if msg.get("text") in ("start", "stop"):
                    clip = None
                    if msg["text"] == "start":
                        seg.begin_manual(); manual = True
                    else:
                        clip, manual = seg.force_finish(), False
                    pred = recognizer.predict(clip[0]) if clip is not None else None
                    await sock.send_json({"state": "idle", "frames": 0, "prediction": pack(pred)})
                    continue
                data = msg.get("bytes")
                if not data:
                    continue
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                ts = max(int((time.time() - t0) * 1000), last_ts + 1)
                last_ts = ts
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                joints, mask = extract_frame(res)
                filled, status = filler.step(joints, mask)
                norm, nstatus = normalize_frame(filled, status, frame.shape[1] / frame.shape[0])
                clip = seg.push(norm, nstatus)
                pred = recognizer.predict(clip[0]) if clip is not None else None
                if clip is not None:
                    manual = False
                await sock.send_json({"state": "signing" if seg.signing else "idle",
                                      "frames": seg.length, "prediction": pack(pred)})
        except WebSocketDisconnect:
            pass

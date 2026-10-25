"""
skeleton_def.py - 50-joint ISL skeleton definition + per-frame extraction.

Joint order (fixed for the whole project):
    0-20   left hand  (MediaPipe hand landmarks 0-20)
    21-41  right hand (MediaPipe hand landmarks 0-20)
    42-49  body: L_eye, R_eye, mouth_L, mouth_R, L_shoulder, R_shoulder,
                 L_elbow, R_elbow   (MediaPipe pose 2,5,9,10,11,12,13,14)
"""

import numpy as np

NUM_JOINTS = 50
LEFT_HAND_OFFSET = 0
RIGHT_HAND_OFFSET = 21
BODY_OFFSET = 42

# MediaPipe pose indices for the 8 body joints, in joint order 42..49
POSE_INDICES = [2, 5, 9, 10, 11, 12, 13, 14]

JOINT_NAMES = (
    [f"L_hand_{i}" for i in range(21)]
    + [f"R_hand_{i}" for i in range(21)]
    + ["L_eye", "R_eye", "mouth_L", "mouth_R",
       "L_shoulder", "R_shoulder", "L_elbow", "R_elbow"]
)

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def _shift(edges, off):
    return [(a + off, b + off) for a, b in edges]


BODY_EDGES = [
    (42, 43), (44, 45),        # eye-eye, mouth-mouth
    (42, 44), (43, 45),        # eye-mouth (each side of face)
    (44, 46), (45, 47),        # mouth to shoulders (head-torso link)
    (46, 47),                  # shoulder-shoulder
    (46, 48), (47, 49),        # shoulder-elbow
]

# ---------------------------------------------------------------------
# Dual-graph split, for the dual ST-GCN (Methodology [3/4], step 4:
# "one graph for hands, one for the upper body"). These two graphs are
# what the ST-GCN teammate should build their adjacency matrices from.
# ---------------------------------------------------------------------

# HAND GRAPH: joints 0-41. Left and right hand are each internally connected
# (standard MediaPipe hand skeleton) but there is NO edge between the two
# hands here - keeping them disjoint mirrors that each hand is an
# independent shape, and a graph conv layer would otherwise mix
# left/right-hand features that have no anatomical connection.
HAND_GRAPH_EDGES = (
    _shift(HAND_EDGES, LEFT_HAND_OFFSET)
    + _shift(HAND_EDGES, RIGHT_HAND_OFFSET)
)
HAND_GRAPH_JOINTS = list(range(LEFT_HAND_OFFSET, BODY_OFFSET))   # 0..41

# BODY GRAPH: joints 42-49 only (face + shoulders + elbows). No links out
# to the hand joints - that separation is the whole point of the dual
# graph, per the Project Motivation slide ("treating hands and body as a
# single skeleton graph blurs fine hand-shape detail with coarse body
# posture").
BODY_GRAPH_EDGES = list(BODY_EDGES)
BODY_GRAPH_JOINTS = list(range(BODY_OFFSET, NUM_JOINTS))         # 42..49

# CROSS-STREAM EDGES: elbow -> same-side wrist. NOT part of either ST-GCN
# graph - these exist only so the combined view (drawing / sanity checks)
# renders as one connected skeleton instead of three floating pieces.
CROSS_STREAM_EDGES = [
    (48, 0),   # left elbow  -> left wrist
    (49, 21),  # right elbow -> right wrist
]

# Combined edge list, for on-screen drawing / visual QA only.
EDGES = HAND_GRAPH_EDGES + BODY_GRAPH_EDGES + CROSS_STREAM_EDGES

# For mirror augmentation: swap left <-> right, then negate x
FLIP_PERM = (
    list(range(21, 42)) + list(range(0, 21))
    + [43, 42, 45, 44, 47, 46, 49, 48]
)


def extract_frame(result, min_visibility=0.5):
    """
    Convert one HolisticLandmarkerResult into numeric skeleton data.

    Returns
    -------
    joints : np.ndarray, shape (50, 3), float32
        x, y, z per joint. Joints that were not detected are NaN
        (no filling/interpolation here - that is Step 3).
    mask : np.ndarray, shape (50,), bool
        True where the joint was actually detected in this frame.

    Coordinates are still RAW MediaPipe values (x, y normalized to [0, 1]
    by image width/height; z is relative depth). Normalization is Step 4.
    """
    joints = np.full((NUM_JOINTS, 3), np.nan, dtype=np.float32)
    mask = np.zeros(NUM_JOINTS, dtype=bool)

    def put(idx, lm):
        joints[idx] = (lm.x, lm.y, lm.z)
        mask[idx] = True

    # Hands: all 21 joints are present or absent together
    for lms, off in ((result.left_hand_landmarks, LEFT_HAND_OFFSET),
                     (result.right_hand_landmarks, RIGHT_HAND_OFFSET)):
        if lms and len(lms) == 21:
            for i, lm in enumerate(lms):
                put(off + i, lm)

    # Upper body: pick the 8 chosen pose landmarks.
    # Joints MediaPipe is unsure about (e.g. elbows cropped by the frame edge)
    # have low visibility; treat those as missing.
    pose = result.pose_landmarks
    if pose and len(pose) >= 33:
        for k, pose_idx in enumerate(POSE_INDICES):
            lm = pose[pose_idx]
            vis = getattr(lm, "visibility", None)
            if vis is None or vis >= min_visibility:
                put(BODY_OFFSET + k, lm)

    return joints, mask
"""
normalize.py - Step 4: per-frame skeleton normalization (works live, no look-ahead).

Input : filled (50, 3) raw MediaPipe coords (NaN where MISSING), status (50,)
Output: norm (50, 3) float32 with NO NaN, and status (50,)

What it does
------------
1. Aspect fix : x and z are multiplied by (width/height) so x, y, z share one
               unit ("image heights"). Without this, a 16:9 camera stretches
               the skeleton horizontally compared with a 4:3 camera.
2. Origin     : midpoint of the two shoulders -> (0, 0) in x, y.
3. Scale      : shoulder width (x,y distance) -> 1.0.
4. z          : body joints (42-49) are re-centered on the shoulder-midpoint z.
               Hand z is already relative to that hand's wrist, so it is only
               rescaled (it keeps the finger-curl depth information).
5. Missing    : MISSING joints are set to 0 (after normalization, when 0 is a
               well-defined point = shoulder midpoint). `status` still marks
               them, so the model / teammate can mask them.

If either shoulder is MISSING (nobody in view) the whole frame is returned as
zeros with every joint MISSING.
"""

import numpy as np

from skeleton_def import NUM_JOINTS, BODY_OFFSET

L_SHOULDER = BODY_OFFSET + 4   # 46
R_SHOULDER = BODY_OFFSET + 5   # 47
MISSING = 0
_EPS = 1e-4


def normalize_frame(filled, status, aspect=1.0):
    """
    filled : (50, 3) float, NaN where status == MISSING
    status : (50,) int8, 0 = MISSING
    aspect : image width / image height
    """
    empty_status = np.zeros(NUM_JOINTS, dtype=np.int8)
    zeros = np.zeros((NUM_JOINTS, 3), dtype=np.float32)

    if status[L_SHOULDER] == MISSING or status[R_SHOULDER] == MISSING:
        return zeros, empty_status

    pts = np.asarray(filled, dtype=np.float32).copy()
    pts[:, 0] *= aspect
    pts[:, 2] *= aspect

    l, r = pts[L_SHOULDER], pts[R_SHOULDER]
    origin_xy = (l[:2] + r[:2]) / 2.0
    scale = float(np.linalg.norm(l[:2] - r[:2]))
    if not np.isfinite(scale) or scale < _EPS:
        return zeros, empty_status
    z_origin = (l[2] + r[2]) / 2.0

    out = np.empty_like(pts)
    out[:, :2] = (pts[:, :2] - origin_xy) / scale
    out[:, 2] = pts[:, 2] / scale
    out[BODY_OFFSET:, 2] = (pts[BODY_OFFSET:, 2] - z_origin) / scale

    missing = status == MISSING
    out[missing] = 0.0
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out, status.copy()
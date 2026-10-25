"""Synthetic tests for normalize.normalize_frame (no webcam needed)."""

import numpy as np

from normalize import normalize_frame, L_SHOULDER, R_SHOULDER

rng = np.random.default_rng(0)
DETECTED = np.ones(50, dtype=np.int8)


def make_pose(aspect=1.0):
    """Random skeleton in 'image-height' units, rendered as MediaPipe-style
    normalized coords for an image of the given aspect (x, z divided by aspect)."""
    p = rng.normal(0.0, 0.1, size=(50, 3)).astype(np.float32)
    p[L_SHOULDER] = [0.15, 0.4, 0.02]     # signer's left shoulder
    p[R_SHOULDER] = [-0.15, 0.4, -0.02]
    p[:, 0] += 0.5 * aspect
    p[:, 1] += 0.3
    q = p.copy()
    q[:, 0] /= aspect
    q[:, 2] /= aspect
    return p, q


base, raw = make_pose(1.0)
out, st = normalize_frame(raw, DETECTED, aspect=1.0)

# shoulder midpoint -> origin, shoulder width -> 1
assert np.allclose((out[L_SHOULDER, :2] + out[R_SHOULDER, :2]) / 2, 0, atol=1e-5)
assert np.isclose(np.linalg.norm(out[L_SHOULDER, :2] - out[R_SHOULDER, :2]), 1.0, atol=1e-5)
# body z centered on shoulders
assert np.isclose(out[L_SHOULDER, 2], -out[R_SHOULDER, 2], atol=1e-5)

# translation invariance (person moves in the frame)
shifted = raw.copy()
shifted[:, 0] += 0.2
shifted[:, 1] -= 0.1
shifted[42:, 2] += 0.3   # hand z is wrist-relative, so only body z shifts with depth
out_t, _ = normalize_frame(shifted, DETECTED, 1.0)
assert np.allclose(out, out_t, atol=1e-5)

# scale invariance (person moves toward / away from the camera)
scaled = raw * 1.7
out_s, _ = normalize_frame(scaled, DETECTED, 1.0)
assert np.allclose(out, out_s, atol=1e-4)

# aspect-ratio invariance: same physical pose seen by 16:9 and 4:3 cameras
_, raw_169 = make_pose(16 / 9)
rng = np.random.default_rng(0)            # regenerate identical physical pose
base1, _ = make_pose(1.0)
def render(base_pose, aspect):
    p = base_pose.copy()
    p[:, 0] += 0.5 * (aspect - 1.0)
    q = p.copy()
    q[:, 0] /= aspect
    q[:, 2] /= aspect
    return q
o_a, _ = normalize_frame(render(base1, 16 / 9), DETECTED, 16 / 9)
o_b, _ = normalize_frame(render(base1, 4 / 3), DETECTED, 4 / 3)
assert np.allclose(o_a, o_b, atol=1e-4)

# missing joints -> exactly 0, status preserved, no NaN anywhere
st_m = DETECTED.copy()
st_m[21:42] = 0                              # right hand missing
raw_m = raw.copy()
raw_m[21:42] = np.nan
out_m, st_out = normalize_frame(raw_m, st_m, 1.0)
assert not np.isnan(out_m).any()
assert (out_m[21:42] == 0).all() and (st_out[21:42] == 0).all()
assert np.allclose(out_m[:21], out[:21])     # other joints unaffected

# no shoulders -> empty frame, everything MISSING
st_n = DETECTED.copy()
st_n[L_SHOULDER] = 0
raw_n = raw.copy()
raw_n[L_SHOULDER] = np.nan
out_n, st_n_out = normalize_frame(raw_n, st_n, 1.0)
assert (out_n == 0).all() and (st_n_out == 0).all()

# output dtype/shape
assert out.shape == (50, 3) and out.dtype == np.float32

print("All Step 4 tests passed.")
"""Synthetic tests for stream_processor (no webcam needed)."""

import numpy as np

from stream_processor import (
    CausalFiller, SlidingWindow, process_sequence_causal,
    MISSING, DETECTED, HELD,
)

T = 30
t = np.arange(T, dtype=np.float32)
truth = np.stack([t, 2 * t, 3 * t], axis=-1)[:, None, :].repeat(50, axis=1)
mask = np.ones((T, 50), dtype=bool)

mask[10:13, 0:21] = False      # left hand: 3-frame dropout   -> held
mask[20:, 21:42] = False       # right hand: lost from frame 20 -> held 5 then missing
mask[0:2, 42:50] = False       # body unseen at the start     -> missing
mask[10:20, 42:50] = False     # body 10-frame dropout        -> held

raw = truth.copy()
raw[~mask] = np.nan
out, st = process_sequence_causal(raw, mask, max_hold_hands=5)

# short hand gap: held at last valid value (frame 9)
assert (st[10:13, 0:21] == HELD).all()
assert np.allclose(out[10:13, 0:21], truth[9, 0:21])

# right hand: frames 20-24 held, 25+ missing
assert (st[20:25, 21:42] == HELD).all()
assert (st[25:, 21:42] == MISSING).all() and np.isnan(out[25:, 21:42]).all()

# body never seen before frame 2 -> missing (causal: cannot look ahead)
assert (st[0:2, 42:50] == MISSING).all() and np.isnan(out[0:2, 42:50]).all()

# body dropout held indefinitely
assert (st[10:20, 42:50] == HELD).all()
assert np.allclose(out[10:20, 42:50], truth[9, 42:50])

# detections untouched
assert (st[mask] == DETECTED).all() and np.allclose(out[mask], truth[mask])

# causality: result for frame k must not depend on future frames
f = CausalFiller(5)
partial = [f.step(raw[i], mask[i])[0] for i in range(15)]
assert np.allclose(np.stack(partial), out[:15], equal_nan=True)

# body held only up to max_hold_body (30) frames, then MISSING (person left)
f2 = CausalFiller(5, 30)
seen = np.zeros((50, 3), dtype=np.float32)
f2.step(seen, np.ones(50, dtype=bool))
gone = np.full((50, 3), np.nan, dtype=np.float32)
statuses = [f2.step(gone, np.zeros(50, dtype=bool))[1] for _ in range(35)]
assert (statuses[29][42:] == HELD).all()          # 30th missing frame still held
assert (statuses[30][42:] == MISSING).all()       # 31st -> missing

# sliding window: T=10, stride=5 -> first window at frame 10, then every 5
w = SlidingWindow(T=10, stride=5)
fired = []
for i in range(30):
    w.push(out[i], st[i])
    if w.ready():
        seq, s = w.get()
        assert seq.shape == (10, 50, 3) and s.shape == (10, 50)
        fired.append(i + 1)
assert fired == [10, 15, 20, 25, 30], fired

print("All streaming tests passed. Windows fired after frames:", fired)
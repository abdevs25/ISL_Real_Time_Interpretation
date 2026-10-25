"""
stream_processor.py - real-time (causal) skeleton stream processing.

CausalFiller  : handles missing joints frame-by-frame using ONLY past frames.
SlidingWindow : keeps the last T frames and yields (T, 50, 3) windows for the model.

The same classes are used to process INCLUDE videos frame-by-frame for training
data, so training and live inference see identical preprocessing.
"""

from collections import deque

import numpy as np

from skeleton_def import NUM_JOINTS, BODY_OFFSET

MISSING, DETECTED, HELD = 0, 1, 2
MAX_HOLD_HANDS = 5          # frames a lost hand keeps its last position
MAX_HOLD_BODY = 30          # frames a lost body joint is held (~1 s at 30 fps)


class CausalFiller:
    """
    Hands (0-41): if lost, hold the last valid position for up to
        max_hold_hands frames, then report MISSING (NaN). A hand that has
        never been seen stays MISSING.
    Body (42-49): hold the last valid position for up to max_hold_body frames
        (the head and shoulders barely move), then MISSING. This stops a
        frozen skeleton from lingering after the signer leaves the frame.
        Never seen -> MISSING.
    """

    def __init__(self, max_hold_hands=MAX_HOLD_HANDS, max_hold_body=MAX_HOLD_BODY):
        self.max_hold = np.full(NUM_JOINTS, max_hold_hands, dtype=np.int64)
        self.max_hold[BODY_OFFSET:] = max_hold_body
        self.reset()

    def reset(self):
        self.last = np.full((NUM_JOINTS, 3), np.nan, dtype=np.float32)
        self.age = np.full(NUM_JOINTS, 10**9, dtype=np.int64)  # frames since seen

    def step(self, joints, mask):
        """joints (50,3) with NaN where missing, mask (50,) -> (filled, status)."""
        out = joints.copy()
        status = np.where(mask, DETECTED, MISSING).astype(np.int8)

        self.last[mask] = joints[mask]
        self.age[mask] = 0
        self.age[~mask] += 1

        hold = (~mask) & (self.age <= self.max_hold)
        out[hold] = self.last[hold]
        status[hold] = HELD
        return out, status


class SlidingWindow:
    """Keep the last T frames; signal when a new window is due every `stride` frames."""

    def __init__(self, T=30, stride=5):
        self.T, self.stride = T, stride
        self.joints = deque(maxlen=T)
        self.status = deque(maxlen=T)
        self._since = 0

    def push(self, joints, status):
        self.joints.append(joints)
        self.status.append(status)
        self._since += 1

    def __len__(self):
        return len(self.joints)

    def ready(self):
        return len(self.joints) == self.T and self._since >= self.stride

    def get(self):
        """Return (T,50,3) joints and (T,50) status; resets the stride counter."""
        self._since = 0
        return np.stack(self.joints), np.stack(self.status)


def process_sequence_causal(joints_seq, mask_seq, max_hold_hands=MAX_HOLD_HANDS):
    """Offline helper for building training data: runs CausalFiller frame by frame."""
    f = CausalFiller(max_hold_hands)
    outs, sts = zip(*(f.step(j, m) for j, m in zip(joints_seq, mask_seq)))
    return np.stack(outs), np.stack(sts)
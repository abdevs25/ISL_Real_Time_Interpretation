"""Tests for the pieces that do not need a camera:  python test_inference.py
(model tests run only when torch is installed)"""
import numpy as np
from src.inference.segmenter import SignSegmenter
from stream_processor import DETECTED, MISSING


def frame(hands_up):
    j = np.zeros((50, 3), np.float32)
    s = np.zeros(50, np.int8)
    for w in (0, 21):
        s[w] = DETECTED
        j[w, 1] = 0.5 if hands_up else 2.6          # wrist y (down = larger)
    return j, s


def test_segment_one_sign():
    seg, out = SignSegmenter(), []
    for t in range(80):
        r = seg.push(*frame(hands_up=20 <= t < 50))
        if r is not None:
            out.append(r)
    assert len(out) == 1, len(out)
    seq, st = out[0]
    assert seq.shape[1:] == (50, 3) and st.shape[0] == seq.shape[0]
    assert 30 <= len(seq) <= 50, len(seq)


def test_idle_gives_nothing():
    seg = SignSegmenter()
    assert all(seg.push(*frame(False)) is None for _ in range(100))


def test_short_flicker_ignored():
    seg, out = SignSegmenter(), []
    for t in range(80):
        r = seg.push(*frame(hands_up=(t in (10, 11, 12, 13, 14))))
        if r is not None:
            out.append(r)
    assert out == []


def test_two_signs_and_manual():
    seg, out = SignSegmenter(), []
    for t in range(140):
        r = seg.push(*frame(hands_up=(10 <= t < 40) or (70 <= t < 110)))
        if r is not None:
            out.append(r)
    assert len(out) == 2
    seg.begin_manual()
    for _ in range(25):
        seg.push(*frame(True))
    assert seg.force_finish()[0].shape[0] == 25
    seg.begin_manual()                     # manual capture must survive idle frames
    for _ in range(40):
        assert seg.push(*frame(False)) is None
    assert seg.force_finish()[0].shape[0] == 40


def test_max_len():
    seg, out = SignSegmenter(max_len=60), []
    for _ in range(100):
        r = seg.push(*frame(True))
        if r is not None:
            out.append(r)
    assert out and len(out[0][0]) == 60


def test_model_shapes():
    try:
        import torch
    except ImportError:
        print("torch not installed - skipping model test")
        return
    from src.models.isl_model import ISLModel
    B, T = 4, 64
    hand, body = torch.randn(B, 3, T, 42), torch.randn(B, 3, T, 8)
    lengths = torch.tensor([64, 50, 30, 10])
    for mode in ("hand", "body", "concat", "fused"):
        m = ISLModel(7, mode=mode, widths=(16, 16, 32, 32, 64), dim=32, hidden=32, heads=4)
        out = m(hand, body, lengths)
        assert out.shape == (B, 7), (mode, out.shape)
        out.sum().backward()
    m.eval()
    logits, w = m(hand, body, lengths, return_weights=True)
    assert w["hand_from_body"].shape == (B, 32, 32)
    print("model shapes OK (all four modes + attention weights)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)

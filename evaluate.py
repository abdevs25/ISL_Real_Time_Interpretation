"""Evaluate a trained checkpoint on the val or test split.

    python evaluate.py --ckpt checkpoints/e2e_fused_c50_seed0.pt --split test
"""
import argparse
import os

import torch
import torch.nn as nn

from dataset import make_loaders
from src.models.isl_model import load_checkpoint
from src.training.train_e2e import evaluate, restrict_classes, save_report
from src.inference.engine import display_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="skeleton_data")
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, cfg = load_checkpoint(a.ckpt, device)
    loaders, _ = make_loaders(a.data, batch_size=32, max_T=cfg.get("max_T", 64), augment_train=False)
    restrict_classes(loaders, labels)                   # same classes, same order as training
    loader = loaders[a.split]
    res = evaluate(model, loader, device, nn.CrossEntropyLoss())
    print(f"{a.split}: {len(loader.dataset)} clips, {len(labels)} classes, mode={cfg['mode']}")
    print(f"  accuracy {res['acc']:.4f}   top-5 {res['top5']:.4f}")
    print(f"  macro precision {res['precision']:.4f}   recall {res['recall']:.4f}   F1 {res['f1']:.4f}")

    prefix = os.path.join(a.out, os.path.basename(a.ckpt)[:-3] + f"_{a.split}")
    save_report(prefix, res, labels)
    import numpy as np
    cm = np.load(prefix + "_confusion.npy")
    acc = np.diag(cm) / np.maximum(cm.sum(1), 1)
    worst = np.argsort(acc)[:10]
    print("  hardest signs:", ", ".join(f"{display_name(labels[i])} ({acc[i]:.2f})" for i in worst))
    print("  saved reports to", a.out)


if __name__ == "__main__":
    main()

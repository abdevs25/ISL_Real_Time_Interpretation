"""
make_split.py - build the train / val / test split from skeleton_data/manifest.csv.

Writes, inside the skeleton_data folder:
    splits.csv   one row per clip: npz_path, label, signer, split
    labels.json  the sign names in a fixed order (index = class number)

Usage:
    python make_split.py --data skeleton_data
    python make_split.py --data skeleton_data --val-frac 0.15 --test-frac 0.15 --seed 0

SESSION-GROUPED SPLIT  (--by-session, recommended for INCLUDE)
    INCLUDE clips are recorded as bursts of 3-4 takes of the same sign with
    consecutive file numbers (MVI_5177, MVI_5178, MVI_5179 ...). A plain random
    split puts take 1 in train and take 2 in test, which are near-duplicates, so
    accuracy looks better than it is. --by-session finds those bursts from the
    MVI numbers and keeps every burst inside ONE split. It is "take-independent",
    NOT signer-independent (signer ids are not in the file names).

SIGNER-INDEPENDENT SPLIT
    A signer-independent split means nobody who appears in the test set also
    appears in training. That is only possible if the signer can be read from
    the file or folder name. If it can, tell the script how with a regex whose
    FIRST GROUP is the signer id, e.g.

        python make_split.py --data skeleton_data --signer-regex "signer(\\d+)"

    If you do not pass --signer-regex, the split is a per-sign RANDOM split
    (stratified: every sign appears in every split when it has enough clips).
    That is NOT signer-independent and the script says so. Report it honestly
    in the presentation ("random stratified split") unless signer ids exist.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------
# Paths stored in the manifest may be relative to where extraction was run,
# or use Windows backslashes. Make them relative to the skeleton_data folder.
# ---------------------------------------------------------------------

def clean_path(data_dir, p):
    """Return the clip path relative to data_dir, with forward slashes."""
    data_dir = Path(data_dir)
    parts = list(Path(str(p).replace("\\", "/")).parts)
    if data_dir.name in parts:
        parts = parts[len(parts) - parts[::-1].index(data_dir.name):]
    return "/".join(parts)


def load_manifest(data_dir):
    path = Path(data_dir) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run extract_dataset.py first")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("manifest.csv is empty - nothing was extracted")
    return rows


# ---------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------

def counts_for(n, val_frac, test_frac):
    """How many of a sign's n clips go to (train, val, test)."""
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    while n - n_test - n_val < 1:          # always keep at least 1 for training
        if n_val > 0:
            n_val -= 1
        else:
            n_test -= 1
    return n - n_val - n_test, n_val, n_test


def split_random(rows, val_frac, test_frac, seed):
    """Per-sign random split. NOT signer-independent."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    for label in sorted(by_label):
        clips = sorted(by_label[label], key=lambda r: r["npz_path"])
        rng.shuffle(clips)
        n_train, n_val, _ = counts_for(len(clips), val_frac, test_frac)
        for i, r in enumerate(clips):
            r["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return rows


def split_by_signer(rows, val_frac, test_frac, seed):
    """Whole signers go to one split only, so test signers are never seen in training."""
    signers = sorted({r["signer"] for r in rows})
    if len(signers) < 3:
        raise SystemExit(
            f"Only {len(signers)} signer(s) found ({signers}). A signer-independent "
            "split needs at least 3 (train / val / test). Check --signer-regex, or "
            "drop the option to use the random stratified split instead."
        )
    rng = random.Random(seed)
    rng.shuffle(signers)
    per_signer = Counter(r["signer"] for r in rows)
    n_total = len(rows)
    want_test, want_val = round(n_total * test_frac), round(n_total * val_frac)

    assigned, test_n, val_n = {}, 0, 0
    for s in signers:
        if test_n < want_test:
            assigned[s], test_n = "test", test_n + per_signer[s]
        elif val_n < want_val:
            assigned[s], val_n = "val", val_n + per_signer[s]
        else:
            assigned[s] = "train"
    if "train" not in assigned.values():     # make sure training is never empty
        assigned[signers[-1]] = "train"
    for r in rows:
        r["split"] = assigned[r["signer"]]
    return rows


def mvi_number(path):
    m = re.search(r"MVI_(\d+)", str(path))
    return int(m.group(1)) if m else None


def assign_sessions(rows, gap=20):
    """Add r['session'] = burst index (0,1,2..) of consecutive clip numbers within each sign."""
    by_label = defaultdict(list)
    for r in rows:
        n = mvi_number(r["npz_path"])
        if n is None:
            raise SystemExit(f"--by-session needs MVI_<number> in the file name: {r['npz_path']}")
        by_label[r["label"]].append((n, r))
    for label, items in by_label.items():
        items.sort(key=lambda t: t[0])
        sess, prev = -1, None
        for n, r in items:
            if prev is None or n - prev > gap:
                sess += 1
            r["session"] = sess
            prev = n


def split_by_session(rows, val_frac, test_frac, seed, gap=20):
    """Per sign: whole bursts of takes go to one split. 1 burst -> train;
    2 -> train+test; 3+ -> at least one burst each for train/val/test."""
    assign_sessions(rows, gap)
    rng = random.Random(seed)
    by_label = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_label[r["label"]][r["session"]].append(r)
    for label in sorted(by_label):
        sessions = sorted(by_label[label])
        rng.shuffle(sessions)
        k = len(sessions)
        if k == 1:
            n_val, n_test = 0, 0
        elif k == 2:
            n_val, n_test = 0, 1
        else:
            n_test = max(1, round(k * test_frac))
            n_val = max(1, round(k * val_frac))
            while k - n_test - n_val < 1:
                n_val -= 1
        for i, s in enumerate(sessions):
            name = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
            for r in by_label[label][s]:
                r["split"] = name
    return rows


def make_splits(rows, val_frac=0.15, test_frac=0.15, seed=0, signer_regex=None, by_session=False):
    """Add 'signer' and 'split' to every row. Returns (rows, signer_independent)."""
    if signer_regex:
        pat = re.compile(signer_regex)
        for r in rows:
            m = pat.search(r["npz_path"].replace("\\", "/"))
            if not m:
                raise SystemExit(f"--signer-regex did not match: {r['npz_path']}")
            r["signer"] = m.group(1) if m.groups() else m.group(0)
        return split_by_signer(rows, val_frac, test_frac, seed), True
    for r in rows:
        r["signer"] = ""
    if by_session:
        return split_by_session(rows, val_frac, test_frac, seed), False
    return split_random(rows, val_frac, test_frac, seed), False


# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="skeleton_data", help="folder written by extract_dataset.py")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by-session", action="store_true",
                    help="keep consecutive takes (MVI numbers) of a sign in the same split")
    ap.add_argument("--signer-regex", default=None,
                    help="regex whose first group is the signer id (enables signer-independent split)")
    a = ap.parse_args()

    data_dir = Path(a.data)
    rows = load_manifest(data_dir)
    for r in rows:
        r["npz_path"] = clean_path(data_dir, r["npz_path"])

    seen, unique = set(), []
    for r in rows:
        key = (r["label"], mvi_number(r["npz_path"]) if mvi_number(r["npz_path"]) is not None else r["npz_path"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    if len(unique) != len(rows):
        print(f"NOTE: dropped {len(rows) - len(unique)} duplicate clip(s) (same sign + same MVI number)")
    rows = unique

    rows, signer_indep = make_splits(rows, a.val_frac, a.test_frac, a.seed, a.signer_regex, a.by_session)

    labels = sorted({r["label"] for r in rows})
    with open(data_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, indent=2, ensure_ascii=False)

    with open(data_dir / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["npz_path", "label", "signer", "split"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["npz_path"]):
            w.writerow({k: r[k] for k in ("npz_path", "label", "signer", "split")})

    # ---- report ----
    per = Counter(r["split"] for r in rows)
    print(f"Clips: {len(rows)}   signs: {len(labels)}")
    print(f"  train {per['train']}   val {per['val']}   test {per['test']}")
    print("  split type:", "SIGNER-INDEPENDENT" if signer_indep
          else ("session-grouped per sign (takes never shared; NOT signer-independent)"
                if a.by_session else "random per sign (NOT signer-independent)"))
    table = defaultdict(Counter)
    for r in rows:
        table[r["label"]][r["split"]] += 1
    print(f"\n  {'sign':<28}{'train':>6}{'val':>6}{'test':>6}")
    for label in labels:
        c = table[label]
        print(f"  {label:<28}{c['train']:>6}{c['val']:>6}{c['test']:>6}")

    warnings = []
    if per["val"] == 0:
        warnings.append("validation set is EMPTY (too few clips) - add more videos")
    if per["test"] == 0:
        warnings.append("test set is EMPTY (too few clips) - add more videos")
    missing = [l for l in labels if table[l]["train"] == 0]
    if missing:
        warnings.append(f"signs with no training clip: {missing}")
    for s in warnings:
        print("WARNING:", s)
    print(f"\nWrote {data_dir / 'splits.csv'} and {data_dir / 'labels.json'}")


if __name__ == "__main__":
    main()
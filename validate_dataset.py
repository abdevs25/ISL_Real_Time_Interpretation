"""
validate_dataset.py - Step 7: scan manifest.csv from extract_dataset.py and
flag videos likely to hurt training (bad camera framing, MediaPipe failing to
track, clips that are too short, etc.), so they can be reviewed or re-shot
before being handed to the ST-GCN teammate.

Usage:
    python validate_dataset.py --data skeleton_data
    python validate_dataset.py --data skeleton_data --min-frames 12 --min-body-rate 0.95

Writes skeleton_data/flagged.csv (one row per flagged video, with reasons)
and prints a summary. Exits with a non-zero status if anything was flagged,
so it can be used as a CI-style gate.
"""

import argparse
import csv
import sys
from pathlib import Path

DEFAULTS = dict(
    min_frames=8,          # clips shorter than this rarely carry a full sign
    min_body_rate=0.90,    # upper-body joints should almost always be visible
    min_valid_rate=0.90,   # frames where both shoulders were found (normalizable)
    min_either_hand_rate=0.30,  # at least one hand should be visible fairly often
)


def load_manifest(data_dir):
    path = Path(data_dir) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run extract_dataset.py first")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_row(row, cfg):
    reasons = []
    frames = int(row["num_frames"])
    body = float(row["body_rate"])
    valid = float(row["valid_rate"])
    left, right = float(row["left_rate"]), float(row["right_rate"])

    if frames < cfg["min_frames"]:
        reasons.append(f"too short ({frames} frames < {cfg['min_frames']})")
    if body < cfg["min_body_rate"]:
        reasons.append(f"low body detection ({body:.0%} < {cfg['min_body_rate']:.0%}) "
                        f"- check framing/lighting")
    if valid < cfg["min_valid_rate"]:
        reasons.append(f"low valid-frame rate ({valid:.0%}) - shoulders often "
                        f"missing, can't normalize these frames")
    if max(left, right) < cfg["min_either_hand_rate"]:
        reasons.append(f"very low hand detection (L{left:.0%}/R{right:.0%}) "
                        f"- likely a tracking failure, not just a one-handed sign")
    return reasons


def run(data_dir, cfg):
    rows = load_manifest(data_dir)
    if not rows:
        print("Manifest is empty - nothing to validate.")
        return [], rows

    flagged = []
    for row in rows:
        reasons = check_row(row, cfg)
        if reasons:
            flagged.append({**row, "reasons": "; ".join(reasons)})

    out_path = Path(data_dir) / "flagged.csv"
    if flagged:
        fields = list(rows[0].keys()) + ["reasons"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(flagged)
    elif out_path.exists():
        out_path.unlink()

    def pct(vals):
        return f"min {min(vals):.0%} / mean {sum(vals)/len(vals):.0%} / max {max(vals):.0%}"

    body_rates = [float(r["body_rate"]) for r in rows]
    valid_rates = [float(r["valid_rate"]) for r in rows]
    frame_counts = [int(r["num_frames"]) for r in rows]

    print(f"Checked {len(rows)} video(s) from {data_dir}/manifest.csv")
    print(f"  frames/video : min {min(frame_counts)} / mean "
          f"{sum(frame_counts)/len(frame_counts):.0f} / max {max(frame_counts)}")
    print(f"  body_rate    : {pct(body_rates)}")
    print(f"  valid_rate   : {pct(valid_rates)}")
    print(f"  labels       : {len({r['label'] for r in rows})}")
    print(f"\nFlagged: {len(flagged)}/{len(rows)} video(s)"
          + (f" -> {out_path}" if flagged else " (none)"))
    for r in flagged[:20]:
        print(f"  {r['npz_path']}: {r['reasons']}")
    if len(flagged) > 20:
        print(f"  ... and {len(flagged) - 20} more, see {out_path}")

    return flagged, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="output folder from extract_dataset.py")
    ap.add_argument("--min-frames", type=int, default=DEFAULTS["min_frames"])
    ap.add_argument("--min-body-rate", type=float, default=DEFAULTS["min_body_rate"])
    ap.add_argument("--min-valid-rate", type=float, default=DEFAULTS["min_valid_rate"])
    ap.add_argument("--min-either-hand-rate", type=float,
                    default=DEFAULTS["min_either_hand_rate"])
    a = ap.parse_args()
    cfg = dict(min_frames=a.min_frames, min_body_rate=a.min_body_rate,
               min_valid_rate=a.min_valid_rate,
               min_either_hand_rate=a.min_either_hand_rate)
    flagged, rows = run(a.data, cfg)
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
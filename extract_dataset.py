"""
Extract ML-ready 50-joint skeleton sequences from a video dataset.

Pipeline:
    video
      -> MediaPipe Holistic
      -> 50-joint extraction
      -> causal missing-joint filling
      -> skeleton normalization
      -> .npz skeleton sequence

Output:
    skeleton_data/
        <category>/
            <label>/
                <video_name>.npz
        manifest.csv
"""

import argparse
import csv
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from skeleton_def import extract_frame
from stream_processor import CausalFiller
from normalize import normalize_frame


# ---------------------------------------------------------------------
# MediaPipe model
# ---------------------------------------------------------------------

MODEL_PATH = "holistic_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/"
    "float16/latest/holistic_landmarker.task"
)


def ensure_model():
    """Download the MediaPipe model if it is not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading holistic landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


# ---------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------

def create_landmarker():
    BaseOptions = mp_python.BaseOptions
    HolisticLandmarker = mp_vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp_vision.HolisticLandmarkerOptions
    VisionRunningMode = mp_vision.RunningMode

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    return HolisticLandmarker.create_from_options(options)


# ---------------------------------------------------------------------
# Video extraction
# ---------------------------------------------------------------------

def extract_video(video_path, output_path):
    """
    Process one video and save its normalized skeleton sequence.

    Returns metadata for manifest.csv.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    aspect = width / height if height > 0 else 1.0

    # One CausalFiller is used for the entire video.
    filler = CausalFiller(max_hold_hands=5, max_hold_body=30)

    skeleton_frames = []
    status_frames = []

    frame_count = 0

    # Detection statistics
    body_detected_frames = 0
    valid_frames = 0
    left_hand_detected_frames = 0
    right_hand_detected_frames = 0

    # FIX: MediaPipe wants timestamps that always increase. We now use the
    # video's own clock (frame number / fps) instead of the wall clock, so the
    # result does not depend on how fast the computer happens to be.
    last_timestamp_ms = -1

    with create_landmarker() as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(frame_count * 1000.0 / fps)
            timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # ---------------------------------------------------------
            # Step 1: Extract raw 50-joint skeleton
            # ---------------------------------------------------------

            joints, mask = extract_frame(result)

            # ---------------------------------------------------------
            # Detection statistics
            # ---------------------------------------------------------

            if mask[42:].sum() == 8:
                body_detected_frames += 1

            if mask[0:21].any():
                left_hand_detected_frames += 1

            if mask[21:42].any():
                right_hand_detected_frames += 1

            # Both shoulders are required for normalization.
            if mask[46] and mask[47]:
                valid_frames += 1

            # ---------------------------------------------------------
            # Step 2: Causal missing-joint handling
            # ---------------------------------------------------------

            filled, status = filler.step(
                joints,
                mask
            )

            # ---------------------------------------------------------
            # Step 3: Normalize skeleton
            # ---------------------------------------------------------

            normalized, normalized_status = normalize_frame(
                filled,
                status,
                aspect
            )

            skeleton_frames.append(normalized)
            status_frames.append(normalized_status)

            frame_count += 1

    cap.release()

    # -------------------------------------------------------------
    # Check that something was actually extracted
    # -------------------------------------------------------------

    if frame_count == 0:
        raise RuntimeError(
            f"No frames could be read from {video_path}"
        )

    # Convert lists to arrays
    skeleton = np.stack(skeleton_frames).astype(np.float32)
    status = np.stack(status_frames).astype(np.int8)

    # Expected:
    # skeleton -> (T, 50, 3)
    # status   -> (T, 50)

    # -------------------------------------------------------------
    # Save .npz
    # -------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    body_rate = body_detected_frames / frame_count
    valid_rate = valid_frames / frame_count
    left_rate = left_hand_detected_frames / frame_count
    right_rate = right_hand_detected_frames / frame_count

    np.savez_compressed(
        output_path,
        skeleton=skeleton,
        status=status,
        fps=np.float32(fps),
        source_fps=np.float32(fps),
        width=np.int32(width),
        height=np.int32(height),
        num_frames=np.int32(frame_count),
        body_rate=np.float32(body_rate),
        left_rate=np.float32(left_rate),
        right_rate=np.float32(right_rate),
        valid_rate=np.float32(valid_rate),
    )

    return {
        "num_frames": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "body_rate": body_rate,
        "left_rate": left_rate,
        "right_rate": right_rate,
        "valid_rate": valid_rate,
    }



# ---------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".3gp",
}

MANIFEST_FIELDS = [
    "npz_path",
    "category",
    "label",
    "num_frames",
    "fps",
    "width",
    "height",
    "body_rate",
    "left_rate",
    "right_rate",
    "valid_rate",
]


def find_videos(input_dir):
    """Find all supported video files recursively."""

    input_dir = Path(input_dir)

    videos = []

    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)

    return sorted(videos)


def metadata_from_npz(npz_path):
    """Read the manifest numbers back from an already-extracted clip."""
    d = np.load(npz_path)
    return {
        "num_frames": int(d["num_frames"]),
        "fps": float(d["fps"]),
        "width": int(d["width"]),
        "height": int(d["height"]),
        "body_rate": float(d["body_rate"]),
        "left_rate": float(d["left_rate"]),
        "right_rate": float(d["right_rate"]),
        "valid_rate": float(d["valid_rate"]),
    }


def write_manifest(rows, output_dir):
    """Write manifest.csv. Called even if the run is interrupted (Ctrl+C)."""
    if not rows:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


# ---------------------------------------------------------------------
# Main dataset extraction
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Extract 50-joint skeleton sequences from videos.",
        epilog=(
            "Expected folder layout:  <input>/<Category>/<SignName>/<video>.mp4  "
            "(the SignName folder becomes the class label)."
        ),
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing the video dataset"
    )

    parser.add_argument(
        "--output",
        default="skeleton_data",
        help="Output folder for skeleton sequences"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N videos (quick test before a full run)"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have a .npz (resume an interrupted run)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    ensure_model()

    videos = find_videos(input_dir)

    if not videos:
        print(f"No video files found in {input_dir}")
        return

    if args.limit:
        videos = videos[:args.limit]

    print(f"Found {len(videos)} video(s).")
    print()

    manifest_rows = []
    failed = []

    try:

        for index, video_path in enumerate(videos, start=1):

            print(
                f"[{index}/{len(videos)}] "
                f"Processing: {video_path}"
            )

            # ---------------------------------------------------------
            # Preserve the dataset's relative folder structure.
            # ---------------------------------------------------------

            relative = video_path.relative_to(input_dir)

            output_path = (
                output_dir
                / relative.with_suffix(".npz")
            )

            try:

                if args.skip_existing and output_path.exists():
                    metadata = metadata_from_npz(output_path)
                    print("    already extracted - skipped")
                else:
                    metadata = extract_video(
                        video_path,
                        output_path
                    )

                    print(
                        f"    saved: {output_path}"
                    )

                    print(
                        f"    shape: "
                        f"({metadata['num_frames']}, 50, 3)"
                    )

                    print(
                        f"    body: {metadata['body_rate']:.1%} | "
                        f"left: {metadata['left_rate']:.1%} | "
                        f"right: {metadata['right_rate']:.1%} | "
                        f"valid: {metadata['valid_rate']:.1%}"
                    )

                # Use the first directory below the dataset root as category
                parts = relative.parts

                category = parts[0] if len(parts) > 1 else ""
                # FIX: the label is the folder directly under the category
                # (videos/<Category>/<SignName>/...). Using parent.name made a
                # clip sitting in a nested sub-folder (e.g. "79. short/extra/")
                # get the label "extra" instead of "79. short".
                if len(parts) > 2:
                    label = parts[1]
                else:
                    label = video_path.parent.name

                manifest_rows.append({
                    "npz_path": str(output_path),
                    "category": category,
                    "label": label,
                    "num_frames": metadata["num_frames"],
                    "fps": metadata["fps"],
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "body_rate": metadata["body_rate"],
                    "left_rate": metadata["left_rate"],
                    "right_rate": metadata["right_rate"],
                    "valid_rate": metadata["valid_rate"],
                })

            except Exception as exc:

                failed.append(str(video_path))
                print(
                    f"    ERROR: {exc}"
                )

            print()

    except KeyboardInterrupt:
        print("\nInterrupted - saving the manifest for what was finished.")
        print("Run again with --skip-existing to continue.\n")

    finally:
        # -------------------------------------------------------------
        # Write manifest (also when interrupted)
        # -------------------------------------------------------------
        manifest_path = write_manifest(manifest_rows, output_dir)

    if manifest_path:

        per_label = {}
        for r in manifest_rows:
            per_label[r["label"]] = per_label.get(r["label"], 0) + 1

        print("----------------------------------------")
        print("Dataset extraction complete.")
        print(f"Successful videos: {len(manifest_rows)}/{len(videos)}")
        print(f"Signs found: {len(per_label)}")
        for name, n in sorted(per_label.items()):
            print(f"    {name}: {n} clip(s)")
        if failed:
            print(f"Failed ({len(failed)}):")
            for p in failed:
                print(f"    {p}")
        print(f"Manifest: {manifest_path}")
        print("Next: python validate_dataset.py --data " + str(output_dir))
        print("----------------------------------------")


if __name__ == "__main__":
    main()
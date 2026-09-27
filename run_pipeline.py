"""
Convenience launcher showing the intended workflow.

Example:

python run_pipeline.py \
    --root-dir /path/to/dataset \
    --fps 30 \
    --epochs 20 \
    --batch-size 32
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--work-dir", default="paper_gaze_outputs")

    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    train_targets = work / "train_pseudo_gaze.pt"
    val_targets = work / "val_pseudo_gaze.pt"
    checkpoint = work / "best_gaze_student.pt"

    python = sys.executable

    run([
        python,
        "generate_pseudo_gaze.py",
        "--root-dir", args.root_dir,
        "--split", "train",
        "--output", str(train_targets),
        "--fps", str(args.fps),
        "--device", "cuda",
    ])

    run([
        python,
        "generate_pseudo_gaze.py",
        "--root-dir", args.root_dir,
        "--split", "val",
        "--output", str(val_targets),
        "--fps", str(args.fps),
        "--device", "cuda",
    ])

    run([
        python,
        "train_gaze_student.py",
        "--train-targets", str(train_targets),
        "--val-targets", str(val_targets),
        "--output", str(checkpoint),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ])


if __name__ == "__main__":
    main()

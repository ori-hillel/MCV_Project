from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gaze_data import collect_split_frames, load_rgb_tensor
from paper_gaze_teacher import PaperGazeConfig, PaperGazeTeacher


@torch.inference_mode()
def generate_split_targets(
    root_dir: str,
    split: str,
    output_file: str,
    fps: int,
    grid_size: int,
    device: str,
):
    videos = collect_split_frames(root_dir, split)

    config = PaperGazeConfig(
        grid_size=grid_size,
        history_frames=fps,  # paper: k = video fps
    )
    teacher = PaperGazeTeacher(config=config, device=device)

    samples = []

    total_frames = sum(len(v) for v in videos.values())
    processed = 0

    print(f"{split}: {len(videos)} videos, {total_frames} frames")
    print(f"Teacher device: {teacher.device}")

    for video_id, frame_paths in videos.items():
        teacher.reset()

        print(f"[{split}] video {video_id}: {len(frame_paths)} frames")

        for frame_index, path in enumerate(frame_paths):
            frame = load_rgb_tensor(path)
            heatmap, gaze_xy = teacher.step(frame)

            samples.append({
                "path": path,
                "video_id": video_id,
                "frame_index": frame_index,
                "heatmap": heatmap.cpu().to(torch.float16),
                "gaze_xy": gaze_xy.cpu().to(torch.float32),
            })

            processed += 1
            if processed % 500 == 0:
                print(f"  processed {processed}/{total_frames}")

    payload = {
        "metadata": {
            "root_dir": str(Path(root_dir).resolve()),
            "split": split,
            "fps": fps,
            "grid_size": grid_size,
            "method": "paper-inspired unsupervised energy/surprise teacher",
        },
        "samples": samples,
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_file)

    print(f"Saved {len(samples)} pseudo-gaze targets -> {output_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--grid-size", type=int, default=14)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. Defaults to CUDA when available, otherwise CPU.",
    )

    args = parser.parse_args()

    generate_split_targets(
        root_dir=args.root_dir,
        split=args.split,
        output_file=args.output,
        fps=args.fps,
        grid_size=args.grid_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

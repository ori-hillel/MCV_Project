from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def natural_key(text: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def _split_json_path(root_dir: str, split: str, category: str) -> str:
    return os.path.join(
        root_dir,
        "annotations",
        "bbox",
        "by_split",
        category,
        f"{split}.json",
    )


def _filenames_from_split_json(json_path: str) -> set[str]:
    """
    Read the split JSON and return authorized basenames.

    This supports the same common JSON layouts as the user's original dataset.
    """
    if not os.path.exists(json_path):
        return set()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    names: set[str] = set()

    if isinstance(data, dict) and "images" in data:
        for item in data["images"]:
            if isinstance(item, dict):
                name = item.get("file_name")
            else:
                name = item
            if name:
                names.add(os.path.basename(str(name)))

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = item.get("file_name")
            else:
                name = item
            if name:
                names.add(os.path.basename(str(name)))

    elif isinstance(data, dict):
        # Fallback for dictionaries keyed by filenames/video ids.
        for key in data.keys():
            names.add(os.path.basename(str(key)))

    return names


def collect_split_frames(root_dir: str, split: str) -> Dict[str, List[str]]:
    """
    Return:
        {
            video_id: [frame1, frame2, ... in chronological order],
            ...
        }

    We use the hand/tool split JSONs only to determine split membership.
    No hand/tool annotations are used as gaze supervision.
    """
    allowed = set()

    for category in ("hands", "tools"):
        allowed |= _filenames_from_split_json(
            _split_json_path(root_dir, split, category)
        )

    images_dir = os.path.join(root_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    videos: Dict[str, List[str]] = defaultdict(list)

    for video_id in sorted(os.listdir(images_dir), key=natural_key):
        video_dir = os.path.join(images_dir, video_id)
        if not os.path.isdir(video_dir):
            continue

        for frame_name in sorted(os.listdir(video_dir), key=natural_key):
            if not frame_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            if allowed and frame_name not in allowed:
                continue

            videos[video_id].append(os.path.join(video_dir, frame_name))

    return dict(videos)


def flatten_video_dict(videos: Dict[str, List[str]]) -> List[str]:
    paths: List[str] = []
    for video_id in sorted(videos.keys(), key=natural_key):
        paths.extend(videos[video_id])
    return paths


paper_frame_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # RGB in [0, 1]
])


student_image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_rgb_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return paper_frame_transform(image)


class PaperPseudoGazeDataset(Dataset):
    """
    Dataset for training the ViT student on paper-generated pseudo-gaze maps.

    pseudo_targets must be produced by generate_pseudo_gaze.py.
    """

    def __init__(self, target_file: str):
        payload = torch.load(target_file, map_location="cpu")

        self.samples = payload["samples"]
        self.metadata = payload.get("metadata", {})

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {target_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        path = sample["path"]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image_tensor = student_image_transform(image)

        heatmap = sample["heatmap"].float()
        gaze_xy = sample["gaze_xy"].float()

        return image_tensor, heatmap, gaze_xy, path

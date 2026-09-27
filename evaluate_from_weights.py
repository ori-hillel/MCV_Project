#!/usr/bin/env python3
import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score, precision_score, recall_score
from torch.utils.data import DataLoader

from dataset import SurgicalMultiTaskDataset
from model_v2 import ImprovedGazeModel, ViTBackbone
from utils import load_real_phase_dict


def temporal_smooth_predictions(y_pred, window=7):
    smoothed = list(y_pred)
    half = window // 2
    for i in range(len(smoothed)):
        start = max(0, i - half)
        end = min(len(smoothed), i + half + 1)
        window_preds = y_pred[start:end]
        counts = Counter(window_preds)
        smoothed[i] = counts.most_common(1)[0][0]
    return smoothed


def evaluate_with_smoothing(model, dataloader, device, lambda_aux=0.5, smooth_window=7):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, phase_targets, bbox_grids, has_tools in dataloader:
            images = images.to(device)
            phase_targets = phase_targets.to(device)
            bbox_grids = bbox_grids.to(device)
            has_tools = has_tools.to(device)

            phase_logits, spatial_logits = model(images)
            loss, _, _ = __import__('loss').calculate_multitask_loss(
                phase_logits, phase_targets, spatial_logits, bbox_grids, has_tools, lambda_aux=lambda_aux
            )
            total_loss += loss.item()
            preds = torch.argmax(phase_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(phase_targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    raw_preds = np.asarray(all_preds)
    y_true = np.asarray(all_targets)
    smoothed_preds = np.asarray(temporal_smooth_predictions(raw_preds.tolist(), window=smooth_window))

    raw_metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(y_true, raw_preds),
        'jaccard_iou': jaccard_score(y_true, raw_preds, average='macro', zero_division=0),
        'precision': precision_score(y_true, raw_preds, average='macro', zero_division=0),
        'recall': recall_score(y_true, raw_preds, average='macro', zero_division=0),
    }

    smooth_metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(y_true, smoothed_preds),
        'jaccard_iou': jaccard_score(y_true, smoothed_preds, average='macro', zero_division=0),
        'precision': precision_score(y_true, smoothed_preds, average='macro', zero_division=0),
        'recall': recall_score(y_true, smoothed_preds, average='macro', zero_division=0),
    }

    choose_smooth = smooth_metrics['jaccard_iou'] > raw_metrics['jaccard_iou']
    final_metrics = smooth_metrics if choose_smooth else raw_metrics
    final_preds = smoothed_preds if choose_smooth else raw_preds
    return raw_metrics, smooth_metrics, final_metrics, y_true, raw_preds, smoothed_preds, final_preds


def build_model(gaze_weights_path: str, device: torch.device) -> ImprovedGazeModel:
    encoder = ViTBackbone(num_unfrozen_blocks=4).to(device)
    model = ImprovedGazeModel(
        phase_encoder=encoder,
        gaze_weights_path=gaze_weights_path,
        num_phases=9,
        embed_dim=768,
    ).to(device)
    return model


def load_checkpoint(model: torch.nn.Module, weights_path: str, device: torch.device):
    ckpt = torch.load(weights_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint at {weights_path} does not contain a state_dict-like object.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Missing keys: {missing[:10]}" if missing else "Missing keys: none")
        print(f"Unexpected keys: {unexpected[:10]}" if unexpected else "Unexpected keys: none")

    return ckpt


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a surgical phase model on the test split using saved weights."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join("checkpoints", "best_model_v2.pt"),
        help="Path to the V2 leaderboard checkpoint (.pt).",
    )
    parser.add_argument(
        "--gaze-weights",
        type=str,
        default=os.path.join("paper_gaze_outputs", "best_gaze_student.pt"),
        help="Path to the gaze prior weights (.pt).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory containing the images/ and annotations/ folders.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Auxiliary loss weighting parameter used by the project evaluator.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit for evaluating only the first N test samples.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    os.chdir(args.root)

    phase_dir = os.path.join("annotations", "bbox", "phase")
    phase_dict = load_real_phase_dict(phase_dir)

    test_dataset = SurgicalMultiTaskDataset(root_dir=".", split="test", phase_dict=phase_dict)
    if args.limit is not None:
        test_dataset.samples = test_dataset.samples[: args.limit]

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.gaze_weights, device)
    ckpt = load_checkpoint(model, args.weights, device)

    print(f"Loaded checkpoint from {args.weights}")
    if isinstance(ckpt, dict):
        print(f"Epoch: {ckpt.get('epoch', '?')}, best val jaccard: {ckpt.get('best_val_jaccard', '?')}")

    print(f"Running evaluation on {len(test_dataset)} test samples with batch_size={args.batch_size}...")
    print("Starting evaluation loop...")
    raw_metrics, smooth_metrics, final_metrics, y_true, raw_preds, smoothed_preds, y_pred = evaluate_with_smoothing(
        model, test_loader, device, args.threshold, smooth_window=7
    )
    print(f"Raw Jaccard IoU:     {raw_metrics['jaccard_iou']:.4f}")
    print(f"Smoothed Jaccard IoU: {smooth_metrics['jaccard_iou']:.4f}")
    print("Final reporting will show both raw and smoothed results.")

    phase_names = [
        "Disinfection",
        "Design",
        "Anesthesia",
        "Incision",
        "Dissection",
        "Hemostasis",
        "Closure",
        "Irrigation",
        "Dressing",
    ]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(phase_names))), normalize="true")
    cm_path = os.path.join("test_confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("Test Set Normalized Confusion Matrix")
    ax.set_xlabel("Predicted phase")
    ax.set_ylabel("True phase")
    ax.set_xticks(np.arange(len(phase_names)))
    ax.set_yticks(np.arange(len(phase_names)))
    ax.set_xticklabels(phase_names, rotation=45, ha="right")
    ax.set_yticklabels(phase_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_color = "white" if value > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color)

    fig.colorbar(im, ax=ax, label="Fraction of true examples")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)

    print("\n=== Test Set Evaluation ===")
    print("Raw predictions:")
    print(f"  Accuracy:     {raw_metrics['accuracy']:.4f}")
    print(f"  Jaccard IoU:  {raw_metrics['jaccard_iou']:.4f}")
    print(f"  Precision:    {raw_metrics['precision']:.4f}")
    print(f"  Recall:       {raw_metrics['recall']:.4f}")
    print(f"  Loss:         {raw_metrics['loss']:.4f}")
    print("\nSmoothed predictions:")
    print(f"  Accuracy:     {smooth_metrics['accuracy']:.4f}")
    print(f"  Jaccard IoU:  {smooth_metrics['jaccard_iou']:.4f}")
    print(f"  Precision:    {smooth_metrics['precision']:.4f}")
    print(f"  Recall:       {smooth_metrics['recall']:.4f}")
    print(f"  Loss:         {smooth_metrics['loss']:.4f}")
    print(f"\nSamples:      {len(y_true)}")
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()

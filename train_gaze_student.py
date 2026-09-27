from __future__ import annotations

import argparse
from pathlib import Path
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gaze_data import PaperPseudoGazeDataset
from gaze_student import ViTGazeStudent, gaze_distribution

def gaze_distribution_loss(
    logits: torch.Tensor,
    target_heatmap: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    KL divergence between:
      - predicted 196-cell gaze distribution
      - normalized teacher surprise map
    """
    b = logits.shape[0]

    pred_log_probs = F.log_softmax(
        logits.reshape(b, -1),
        dim=1,
    )

    target = target_heatmap.reshape(b, -1).float()
    target = target + eps
    target = target / target.sum(dim=1, keepdim=True).clamp_min(eps)

    return F.kl_div(
        pred_log_probs,
        target,
        reduction="batchmean",
    )


@torch.inference_mode()
def mean_grid_distance(
    logits: torch.Tensor,
    gaze_xy: torch.Tensor,
) -> float:
    """
    Validation helper:
    distance in normalized image coordinates between the predicted
    max-probability cell center and the teacher's selected gaze point.

    This is NOT the paper's AAE because camera intrinsics/FOV are not provided.
    """
    b, _, h, w = logits.shape

    flat_idx = logits.reshape(b, -1).argmax(dim=1)
    gy = flat_idx // w
    gx = flat_idx % w

    pred_x = (gx.float() + 0.5) / w
    pred_y = (gy.float() + 0.5) / h
    pred = torch.stack([pred_x, pred_y], dim=1)

    return torch.linalg.vector_norm(pred - gaze_xy, dim=1).mean().item()


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler=None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_dist = 0.0
    total_count = 0

    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for images, targets, gaze_xy, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            gaze_xy = gaze_xy.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)
                loss = gaze_distribution_loss(logits, targets)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            bs = images.shape[0]
            total_loss += loss.item() * bs
            total_dist += mean_grid_distance(logits.detach(), gaze_xy) * bs
            total_count += bs

    return total_loss / total_count, total_dist / total_count


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_image(image):
    image = image.cpu() * IMAGENET_STD + IMAGENET_MEAN
    return image.clamp(0, 1)


@torch.inference_mode()
def visualize_random_predictions(
    model,
    dataset,
    device,
    epoch,
    num_images=5,
):
    model.eval()

    indices = random.sample(
        range(len(dataset)),
        min(num_images, len(dataset))
    )

    fig, axes = plt.subplots(
        1,
        len(indices),
        figsize=(4 * len(indices), 4)
    )

    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        image, _, teacher_gaze_xy, path = dataset[idx]

        x = image.unsqueeze(0).to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            logits = model(x)

        # [14,14] gaze probability map
        probs = gaze_distribution(logits)[0, 0]

        h, w = image.shape[-2:]

        # Upscale heatmap to image resolution
        heatmap = F.interpolate(
            probs[None, None],
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )[0, 0].cpu()

        # Predicted gaze point
        idx_max = int(probs.argmax().item())

        gh, gw = probs.shape

        gy = idx_max // gw
        gx = idx_max % gw

        pred_x = (gx + 0.5) / gw
        pred_y = (gy + 0.5) / gh

        rgb = denormalize_image(
            image
        ).permute(1, 2, 0).numpy()

        ax.imshow(rgb)

        # Heatmap overlay
        ax.imshow(
            heatmap.numpy(),
            alpha=0.45
        )

        # Predicted gaze
        ax.scatter(
            pred_x * w,
            pred_y * h,
            marker="x",
            s=100,
            linewidths=3,
            c="red"
        )

        # Teacher gaze
        teacher_x = float(teacher_gaze_xy[0])
        teacher_y = float(teacher_gaze_xy[1])

        ax.scatter(
            teacher_x * w,
            teacher_y * h,
            marker="o",
            s=80,
            facecolors="none",
            edgecolors="cyan",
            linewidths=2
        )

        ax.set_title(Path(path).name)
        ax.axis("off")

    plt.suptitle(
        f"Gaze predictions - epoch {epoch}"
    )

    plt.tight_layout()
    plt.show()


def train(
    train_targets,
    val_targets,
    output,
    epochs,
    batch_size,
    lr,
    weight_decay,
    num_workers,
    visualize_every=2,
    num_visualizations=5,
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable."
        )

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))

    torch.backends.cudnn.benchmark = True

    train_dataset = PaperPseudoGazeDataset(train_targets)
    val_dataset = PaperPseudoGazeDataset(val_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model = ViTGazeStudent().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    scaler = torch.amp.GradScaler("cuda")

    best_val = float("inf")

    output_path = Path(output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    for epoch in range(1, epochs + 1):

        # -----------------------
        # TRAIN
        # -----------------------
        train_loss, train_dist = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )

        # -----------------------
        # VALIDATION
        # -----------------------
        val_loss, val_dist = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
        )

        scheduler.step()

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_loss:.5f} | "
            f"Val loss: {val_loss:.5f} | "
            f"Train dist: {train_dist:.5f} | "
            f"Val dist: {val_dist:.5f}"
        )

        # -----------------------
        # SAVE BEST MODEL
        # -----------------------
        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_distance": val_dist,
                },
                output_path,
            )

            print(
                f"Saved best model -> {output_path}"
            )

        # -----------------------
        # VISUALIZE GAZE
        # -----------------------
        if (
            visualize_every > 0
            and epoch % visualize_every == 0
        ):
            visualize_random_predictions(
                model=model,
                dataset=val_dataset,
                device=device,
                epoch=epoch,
                num_images=num_visualizations,
            )

    return model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-targets", required=True)
    parser.add_argument("--val-targets", required=True)
    parser.add_argument("--output", default="checkpoints/best_gaze_student.pt")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    train(
        train_targets=args.train_targets,
        val_targets=args.val_targets,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PaperGazeConfig:
    """
    Practical implementation of the main gaze-prediction ideas from
    Aakur & Bagavathi, "Unsupervised Gaze Prediction in Egocentric Videos
    by Energy-based Surprise Modeling".

    Notes:
    - grid_size corresponds to the paper's N x N lattice.
    - history_frames corresponds to k; the paper sets k to video FPS.
    - beta_decay corresponds to the temporal weighting concept in Sec. 4.2.
    - pc and p_saccade are exposed because the paper describes the acceptors
      but does not fully specify every implementation detail numerically.
    """
    grid_size: int = 14
    history_frames: int = 30
    alpha: float = 2.0
    bond_scale: float = 1.0
    beta_decay: float = 0.95
    pc: float = 0.90
    p_saccade: float = 0.15
    center_sigma: float = 0.30


class PaperGazeTeacher:
    """
    Unsupervised, stateful, paper-inspired gaze teacher.

    Input:
        frame: [3, H, W], float tensor in [0, 1]

    Output:
        heatmap: [N, N], normalized to [0, 1]
        gaze_xy: [2], normalized coordinates in [0, 1]

    Important:
        Call reset() whenever a new video starts.
    """

    def __init__(
        self,
        config: Optional[PaperGazeConfig] = None,
        device: str | torch.device = "cuda",
    ):
        self.cfg = config or PaperGazeConfig()

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

        self.device = torch.device(device)
        self.history: Deque[torch.Tensor] = deque(maxlen=self.cfg.history_frames)
        self.prev_gaze: Optional[Tuple[int, int]] = None
        self.center_bias = self._make_center_bias().to(self.device)

    def reset(self) -> None:
        self.history.clear()
        self.prev_gaze = None

    def _make_center_bias(self) -> torch.Tensor:
        n = self.cfg.grid_size
        y = torch.linspace(-1.0, 1.0, n)
        x = torch.linspace(-1.0, 1.0, n)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        cb = torch.exp(
            -(xx.square() + yy.square()) /
            (2.0 * self.cfg.center_sigma * self.cfg.center_sigma)
        )
        return cb / cb.max().clamp_min(1e-8)

    def _feature_configuration(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Paper Sec. 4.2:
        construct an N x N lattice and populate each generator with
        appearance features from its image cell.

        Here each generator uses mean RGB in its cell.
        """
        pooled = F.adaptive_avg_pool2d(
            frame.unsqueeze(0),
            (self.cfg.grid_size, self.cfg.grid_size),
        )[0]  # [3, N, N]

        return pooled.permute(1, 2, 0).contiguous()  # [N, N, 3]

    @staticmethod
    def _bhattacharyya_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Stable Bhattacharyya-style appearance distance.

        The paper's appearance term is based on a normalized sqrt(g_i g_j)
        affinity. We turn that into surprise = 1 - affinity.
        """
        eps = 1e-8
        a = a.clamp_min(0.0) + eps
        b = b.clamp_min(0.0) + eps

        a = a / a.sum(dim=-1, keepdim=True)
        b = b / b.sum(dim=-1, keepdim=True)

        affinity = torch.sqrt(a * b).sum(dim=-1)
        return (1.0 - affinity).clamp_min(0.0)

    def _shift_with_mask(
        self,
        x: torch.Tensor,
        dy: int,
        dx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shifted = torch.roll(x, shifts=(dy, dx), dims=(0, 1))
        n = self.cfg.grid_size

        mask = torch.ones((n, n), dtype=torch.bool, device=x.device)

        if dy > 0:
            mask[:dy, :] = False
        elif dy < 0:
            mask[dy:, :] = False

        if dx > 0:
            mask[:, :dx] = False
        elif dx < 0:
            mask[:, dx:] = False

        return shifted, mask

    def _local_bond_surprise(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
    ) -> torch.Tensor:
        """
        Paper Sec. 4.2:
        connect generators with spatial locality and quantify bonds.

        We use a 3x3 local neighborhood, appearance distance, alpha clipping,
        and tanh-scaled bond strength.
        """
        n = self.cfg.grid_size
        total = torch.zeros((n, n), device=self.device)
        count = torch.zeros_like(total)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                previous_local, mask = self._shift_with_mask(previous, dy, dx)

                phi_a = self._bhattacharyya_distance(current, previous_local)

                # Paper Eq. (2): phi = min(alpha, phi_a + phi_m)
                # Optical flow is omitted by default; the paper reports that
                # adding optical flow hurt gaze performance in its ablation.
                phi = torch.clamp(phi_a, max=self.cfg.alpha)

                # Paper Eq. (1): bond term uses w_s * tanh(phi)
                bond = self.cfg.bond_scale * torch.tanh(phi)

                total += bond * mask
                count += mask

        return total / count.clamp_min(1.0)

    def _temporal_proposal(self, current: torch.Tensor) -> torch.Tensor:
        """
        Paper Sec. 4.2:
        aggregate the previous k configurations with greater weight
        on temporally local configurations.
        """
        n = self.cfg.grid_size
        if not self.history:
            return torch.zeros((n, n), device=self.device)

        energies = []
        weights = []

        # newest frame first
        for age, previous in enumerate(reversed(self.history)):
            energies.append(self._local_bond_surprise(current, previous))

            # Practical exponentially decaying weights.
            # beta_decay=0.95 => recent history receives largest weight.
            weights.append(self.cfg.beta_decay ** age)

        w = torch.tensor(weights, dtype=torch.float32, device=self.device)
        w = w / w.sum().clamp_min(1e-8)

        proposal = torch.zeros_like(energies[0])
        for wi, ei in zip(w, energies):
            proposal += wi * ei

        return proposal

    def _select_generator(self, energy: torch.Tensor) -> tuple[int, int]:
        """
        Paper Sec. 4.3:
        emulate fixation vs. saccade behavior using the previous predicted gaze.

        The exact numerical realization is under-specified in the paper,
        so this implementation makes that choice explicit:
        - with p_saccade, choose based on raw surprise;
        - otherwise divide by (1 + distance) to favor fixation near prior gaze.
        """
        n = self.cfg.grid_size

        if self.prev_gaze is None:
            idx = int(torch.argmax(energy).item())
            return idx // n, idx % n

        prev_y, prev_x = self.prev_gaze

        yy, xx = torch.meshgrid(
            torch.arange(n, device=self.device),
            torch.arange(n, device=self.device),
            indexing="ij",
        )
        distance = torch.sqrt(
            (yy.float() - float(prev_y)).square() +
            (xx.float() - float(prev_x)).square()
        )

        if torch.rand((), device=self.device) < self.cfg.p_saccade:
            selection_energy = energy
        else:
            selection_energy = energy / (1.0 + distance)

        idx = int(torch.argmax(selection_energy).item())
        return idx // n, idx % n

    @torch.inference_mode()
    def step(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process one frame in temporal order.
        """
        frame = frame.to(self.device, non_blocking=True)

        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(f"Expected [3,H,W] RGB tensor, got {tuple(frame.shape)}")

        current = self._feature_configuration(frame)
        temporal = self._temporal_proposal(current)

        # Paper Algorithm 1 first acceptor:
        # temporal proposal vs strong center bias.
        use_temporal = (
            len(self.history) > 0 and
            torch.rand((), device=self.device) < self.cfg.pc
        )
        energy = temporal if use_temporal else self.center_bias

        e_min = energy.min()
        e_max = energy.max()
        if (e_max - e_min) > 1e-8:
            heatmap = (energy - e_min) / (e_max - e_min)
        else:
            heatmap = torch.zeros_like(energy)

        gy, gx = self._select_generator(energy)
        self.prev_gaze = (gy, gx)

        # Paper Eq. (4): use the center of the selected grid cell.
        gaze_x = (gx + 0.5) / self.cfg.grid_size
        gaze_y = (gy + 0.5) / self.cfg.grid_size
        gaze_xy = torch.tensor(
            [gaze_x, gaze_y],
            dtype=torch.float32,
            device=self.device,
        )

        self.history.append(current.detach())

        return heatmap, gaze_xy

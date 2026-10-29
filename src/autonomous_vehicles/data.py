"""Procedural scenes with exact segmentation labels; no external data or weights."""

import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ("background", "road", "obstacle")
SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}


class RoadScenes(Dataset):
    """Index-stable scenes using independent random streams for each split.

    Image: float32 CHW [0, 1]. Mask: int64 HW, classes 0..2.
    These stylized scenes provide pipeline checks, not real-world performance evidence.
    """

    def __init__(self, count: int = 128, size: int = 64, seed: int = 42, split: str = "train"):
        if count < 1 or size < 32 or seed < 0 or split not in SPLIT_IDS:
            raise ValueError("Require count >= 1, size >= 32, seed >= 0 and a known split")
        self.count, self.size, self.seed, self.split = count, size, seed, split

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, SPLIT_IDS[self.split], index])
        )
        n = self.size
        y, x = np.mgrid[:n, :n]
        horizon = int(rng.integers(n // 4, n // 2))
        progress = np.clip((y - horizon) / (n - horizon), 0, 1)
        center = n / 2 + rng.uniform(-0.15, 0.15) * n * progress
        width = n * (0.04 + 0.4 * progress)
        road = (y >= horizon) & (np.abs(x - center) < width)
        mask = np.zeros((n, n), dtype=np.int64)
        mask[road] = 1
        for _ in range(int(rng.integers(1, 4))):
            bottom = int(rng.integers(horizon + n // 8, n))
            height = max(3, int((bottom / n) * n * rng.uniform(0.08, 0.16)))
            left = int(center[bottom, 0] + rng.uniform(-0.5, 0.5) * width[bottom, 0])
            mask[
                bottom - height : bottom,
                max(0, left - height // 2) : min(n, left + height // 2 + 1),
            ] = 2
        palette = np.array([[0.35, 0.65, 0.6], [0.3, 0.3, 0.33], [0.85, 0.25, 0.15]])
        image = palette[mask] * rng.uniform(0.7, 1.15)
        image += rng.normal(0, 0.035, image.shape)
        image = np.clip(image, 0, 1).astype(np.float32).transpose(2, 0, 1)
        return torch.from_numpy(image.copy()), torch.from_numpy(mask)

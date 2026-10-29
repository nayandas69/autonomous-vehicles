"""Seeded image transformations preserving the segmentation coordinate contract."""

from dataclasses import dataclass

import torch

from autonomous_vehicles.simulation.geometry import finite


@dataclass(frozen=True)
class AugmentationConfig:
    flip_probability: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    noise: float = 0.02

    def __post_init__(self):
        finite(*vars(self).values())
        if not 0 <= self.flip_probability <= 1:
            raise ValueError("Flip probability must lie in [0, 1]")
        if min(self.brightness, self.contrast, self.noise) < 0:
            raise ValueError("Augmentation strengths must be nonnegative")
        if max(self.brightness, self.contrast) > 1:
            raise ValueError("Brightness and contrast strengths must not exceed one")


def validate_sample(image: torch.Tensor, mask: torch.Tensor) -> None:
    if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
        raise ValueError("Image must be floating point CHW RGB")
    if mask.ndim != 2 or mask.shape != image.shape[1:] or mask.dtype != torch.int64:
        raise ValueError("Mask must be int64 HW aligned with image")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Image must contain finite values in [0, 1]")
    if image.device.type != "cpu" or mask.device.type != "cpu":
        raise ValueError("This reproducible augmentation profile requires CPU tensors")


class Augmenter:
    def __init__(self, config: AugmentationConfig | None = None, seed: int = 42):
        if seed < 0:
            raise ValueError("Seed must be nonnegative")
        self.config = AugmentationConfig() if config is None else config
        self.generator = torch.Generator().manual_seed(seed)

    def uniform(self, lower: float, upper: float) -> float:
        return lower + (upper - lower) * torch.rand((), generator=self.generator).item()

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        validate_sample(image, mask)
        image, mask = image.clone(), mask.clone()
        if self.uniform(0, 1) < self.config.flip_probability:
            image, mask = image.flip(-1), mask.flip(-1)
        brightness = self.uniform(1 - self.config.brightness, 1 + self.config.brightness)
        contrast = self.uniform(1 - self.config.contrast, 1 + self.config.contrast)
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = ((image - mean) * contrast + mean) * brightness
        noise = torch.randn(image.shape, generator=self.generator, dtype=image.dtype)
        return (image + noise * self.config.noise).clamp(0, 1), mask


def image_corruption(
    image: torch.Tensor, kind: str, severity: float, seed: int = 42
) -> torch.Tensor:
    finite(severity)
    if not 0 <= severity <= 1 or seed < 0:
        raise ValueError("Severity must be in [0, 1] and seed nonnegative")
    if image.ndim != 3 or image.shape[0] != 3 or image.device.type != "cpu":
        raise ValueError("Expected a CPU CHW RGB image")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("Image must be finite floating point")
    if kind == "darkness":
        return (image * (1 - severity)).clamp(0, 1)
    if kind == "fog":
        return (image * (1 - severity) + severity * 0.8).clamp(0, 1)
    if kind == "noise":
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
        return (image + noise * severity * 0.25).clamp(0, 1)
    if kind == "occlusion":
        result = image.clone()
        width = int(image.shape[-1] * severity)
        start = (image.shape[-1] - width) // 2
        result[:, :, start : start + width] = 0
        return result
    raise ValueError("Choose darkness, fog, noise or occlusion")

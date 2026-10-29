"""CPU inference timing with warmup and explicit measurement limitations."""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch


@dataclass(frozen=True)
class LatencyReport:
    iterations: int
    batch_size: int
    input_shape: tuple[int, ...]
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    images_per_second: float
    parameters: int
    threads: int

    def to_dict(self):
        return asdict(self)


@torch.inference_mode()
def profile_inference(
    model: torch.nn.Module, images: torch.Tensor, warmup: int = 3, iterations: int = 20
) -> LatencyReport:
    if images.ndim != 4 or len(images) < 1 or images.device.type != "cpu":
        raise ValueError("This profiler requires a nonempty CPU BCHW batch")
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("Images must be finite floating-point tensors")
    if warmup < 0 or iterations < 2:
        raise ValueError("Need nonnegative warmup and at least two measured iterations")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("Model parameters must be on CPU")
    was_training = model.training
    measurements = []
    try:
        model.eval()
        for _ in range(warmup):
            model(images)
        for _ in range(iterations):
            started = perf_counter()
            model(images)
            measurements.append(perf_counter() - started)
    finally:
        model.train(was_training)
    times = np.array(measurements)
    average = float(times.mean())
    return LatencyReport(
        iterations,
        len(images),
        tuple(images.shape),
        average,
        float(np.median(times)),
        float(np.quantile(times, 0.95)),
        len(images) / average,
        sum(parameter.numel() for parameter in model.parameters()),
        torch.get_num_threads(),
    )

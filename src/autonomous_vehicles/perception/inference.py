"""Batched segmentation inference and flip-averaged prediction."""

from dataclasses import dataclass
from time import perf_counter

import torch

from autonomous_vehicles.perception.boxes import Detection
from autonomous_vehicles.perception.components import mask_detections


@dataclass(frozen=True)
class Prediction:
    mask: torch.Tensor
    confidence: torch.Tensor
    detections: tuple[Detection, ...]
    elapsed_seconds: float


class Predictor:
    def __init__(self, model: torch.nn.Module, flip_average: bool = False):
        self.model = model.cpu().eval()
        self.flip_average = flip_average

    @torch.inference_mode()
    def probabilities(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3 or images.device.type != "cpu":
            raise ValueError("Expected CPU BCHW RGB images")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("Images must be finite floating point")
        if images.min() < 0 or images.max() > 1:
            raise ValueError("Images must lie in [0, 1]")
        probabilities = self.model(images).softmax(1)
        if self.flip_average:
            flipped = self.model(images.flip(-1)).softmax(1).flip(-1)
            probabilities = (probabilities + flipped) / 2
        return probabilities

    def predict(self, image: torch.Tensor, threshold: float = 0.5) -> Prediction:
        if image.ndim != 3:
            raise ValueError("Expected one CHW image")
        start = perf_counter()
        probabilities = self.probabilities(image.unsqueeze(0))[0]
        confidence, mask = probabilities.max(0)
        detections = mask_detections(probabilities.numpy(), threshold=threshold)
        return Prediction(mask, confidence, detections, perf_counter() - start)

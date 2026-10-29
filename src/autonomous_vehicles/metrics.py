"""Dataset-level confusion matrix metrics, never averages of batch IoUs."""

import torch


def segmentation_metrics(confusion: torch.Tensor) -> dict:
    confusion = confusion.to(torch.float64)
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    present = union > 0
    iou = intersection / union.clamp_min(1)
    return {
        "mean_iou": iou[present].mean().item() if present.any() else 0.0,
        "class_iou": [
            value.item() if exists else None for value, exists in zip(iou, present, strict=True)
        ],
        "pixel_accuracy": (intersection.sum() / confusion.sum().clamp_min(1)).item(),
    }


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    confusion = torch.zeros((3, 3), dtype=torch.int64)
    for image, target in loader:
        prediction = model(image).argmax(1)
        confusion += torch.bincount((target * 3 + prediction).flatten(), minlength=9).reshape(3, 3)
    return segmentation_metrics(confusion)

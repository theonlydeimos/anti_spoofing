import torch

from src.metrics.base_metric import BaseMetric


class AntiSpoofingAccuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        prediction_labels: torch.Tensor = torch.argmax(logits, dim=1)
        return (prediction_labels == labels).float().mean()

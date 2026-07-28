import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.calculate_eer import compute_eer


class AntiSpoofingAccuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        prediction_labels: torch.Tensor = torch.argmax(logits, dim=1)
        return (prediction_labels == labels).float().mean()


class AntiSpoofingEER(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch) -> float:
        if logits.ndim != 2:
            raise ValueError(
                f"Expected logits to be a 2D tensor, but got {logits.ndim}D tensor."
            )

        if logits.shape[1] != 2:
            raise ValueError(
                f"Expected logits to have shape (B, 2), but got {logits.shape}."
            )

        if labels.ndim != 1:
            raise ValueError(
                f"Expected labels to be a 1D tensor, but got {labels.ndim}D tensor."
            )

        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Expected logits and labels to have the same batch size, but got {logits.shape[0]} and {labels.shape[0]}."
            )

        if not torch.all(logits.isfinite()).item():
            raise ValueError("Logits contain non-finite values.")

        if not torch.all((labels == 0) | (labels == 1)).item():
            raise ValueError(
                f"Expected labels to be binary (0 or 1), but got {labels}."
            )

        if not torch.any(labels == 0).item() or not torch.any(labels == 1).item():
            raise ValueError("Expected labels to contain both classes 0 and 1.")

        scores: torch.Tensor = logits[:, 1] - logits[:, 0]
        if not torch.all(scores.isfinite()).item():
            raise ValueError("Scores contain non-finite values.")

        bonafide_scores = scores[labels == 1].detach().cpu().numpy()
        spoof_scores = scores[labels == 0].detach().cpu().numpy()

        eer, _ = compute_eer(bonafide_scores, spoof_scores)
        return float(eer)

import csv
from pathlib import Path

import torch


def compute_bonafide_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected logits with shape [N, 2], but got {logits.shape}.")

    return logits[:, 1] - logits[:, 0]


def write_score_csv(
    audio_file_ids: list[str],
    logits: torch.Tensor,
    output_path: str | Path,
) -> Path:
    if len(audio_file_ids) != logits.shape[0]:
        raise ValueError(
            "The number of utterance IDs must match the number of model outputs."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    scores = compute_bonafide_scores(logits).detach().cpu().tolist()

    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerows(zip(audio_file_ids, scores))

    return output_path

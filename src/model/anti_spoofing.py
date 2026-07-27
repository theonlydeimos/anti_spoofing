import torch
from torch import nn


class MaxFeatureMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.shape[1] % 2 != 0:
            raise ValueError("The number of channels must be even.")

        first_half, second_half = torch.chunk(
            input_tensor,
            chunks=2,
            dim=1,
        )

        return torch.maximum(first_half, second_half)

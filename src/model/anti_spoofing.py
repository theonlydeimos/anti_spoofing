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


class ConvMFMPair(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            MaxFeatureMap(),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(input_tensor)


class AntiSpoofingModel(nn.Module):
    def __init__(
        self,
        n_classes: int = 2,
        dropout_probability: float = 0.75,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # Conv_1, MFM_2
            ConvMFMPair(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                padding=2,
            ),
            # MaxPool_3
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            # Conv_4, MFM_5
            ConvMFMPair(
                in_channels=32,
                out_channels=32,
                kernel_size=1,
                padding=0,
            ),
            # BatchNorm_6
            nn.BatchNorm2d(num_features=32),
            # Conv_7, MFM_8
            ConvMFMPair(
                in_channels=32,
                out_channels=48,
                kernel_size=3,
                padding=1,
            ),
            # MaxPool_9
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            # BatchNorm_10
            nn.BatchNorm2d(num_features=48),
            # Conv_11, MFM_12
            ConvMFMPair(
                in_channels=48,
                out_channels=48,
                kernel_size=1,
                padding=0,
            ),
            # BatchNorm_13
            nn.BatchNorm2d(num_features=48),
            # Conv_14, MFM_15
            ConvMFMPair(
                in_channels=48,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            # MaxPool_16
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            # Conv_17, MFM_18
            ConvMFMPair(
                in_channels=64,
                out_channels=64,
                kernel_size=1,
                padding=0,
            ),
            # BatchNorm_19
            nn.BatchNorm2d(num_features=64),
            # Conv_20, MFM_21
            ConvMFMPair(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            # BatchNorm_22
            nn.BatchNorm2d(num_features=32),
            # Conv_23, MFM_24
            ConvMFMPair(
                in_channels=32,
                out_channels=32,
                kernel_size=1,
                padding=0,
            ),
            # BatchNorm_25
            nn.BatchNorm2d(num_features=32),
            # Conv_26, MFM_27
            ConvMFMPair(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            # MaxPool_28
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            # FC_29
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=23552, out_features=160),
            # MFM_30
            MaxFeatureMap(),
            # Dropout
            nn.Dropout(p=dropout_probability),
            # BatchNorm_31
            nn.BatchNorm1d(num_features=80),
            # FC_32
            nn.Linear(in_features=80, out_features=n_classes),
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        data_object: torch.Tensor,
        **batch,
    ) -> dict[str, torch.Tensor]:
        if data_object.ndim == 3:
            data_object = data_object.unsqueeze(dim=1)

        if data_object.ndim != 4 or data_object.shape[1] != 1:
            raise ValueError(
                "Expected data_object with shape [B, F, T] or [B, 1, F, T]."
            )

        return {"logits": self.net(data_object)}

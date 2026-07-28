import torch
import torchaudio
from torch import nn


class FrequencyCompressionFC(nn.Module):
    def __init__(
        self,
        n_input_frequency_bins: int,
        n_output_frequency_bins: int,
        sample_rate: int,
    ) -> None:
        super().__init__()
        self.n_input_frequency_bins = n_input_frequency_bins
        self.n_output_frequency_bins = n_output_frequency_bins
        self.sample_rate = sample_rate
        self.net = nn.Linear(
            in_features=n_input_frequency_bins,
            out_features=n_output_frequency_bins,
            bias=False,
        )
        self.initialize_as_linear_filter_bank()

    @torch.no_grad()
    def initialize_as_linear_filter_bank(self) -> None:
        filter_bank = torchaudio.functional.linear_fbanks(
            n_freqs=self.n_input_frequency_bins,
            f_min=0.0,
            f_max=self.sample_rate / 2,
            n_filter=self.n_output_frequency_bins,
            sample_rate=self.sample_rate,
        )
        self.net.weight.copy_(filter_bank.transpose(0, 1))

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        spectrogram = spectrogram.transpose(1, 2)
        projected_spectrogram = self.net(spectrogram)
        return projected_spectrogram.transpose(1, 2)


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
        n_input_frequency_bins: int = 257,
        n_projected_frequency_bins: int = 60,
        n_input_frames: int = 750,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()

        self.n_input_frequency_bins = n_input_frequency_bins
        self.n_input_frames = n_input_frames
        self.frequency_compression = FrequencyCompressionFC(
            n_input_frequency_bins=n_input_frequency_bins,
            n_output_frequency_bins=n_projected_frequency_bins,
            sample_rate=sample_rate,
        )

        n_pooled_frequency_bins = n_projected_frequency_bins
        n_pooled_frames = n_input_frames
        for _ in range(4):
            n_pooled_frequency_bins //= 2
            n_pooled_frames //= 2

        if n_pooled_frequency_bins == 0 or n_pooled_frames == 0:
            raise ValueError(
                "The projected spectrogram must be large enough for four pooling "
                "layers."
            )

        n_flattened_features = 32 * n_pooled_frequency_bins * n_pooled_frames

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
            nn.Linear(in_features=n_flattened_features, out_features=160),
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
        self.frequency_compression.initialize_as_linear_filter_bank()

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
        if data_object.ndim == 4 and data_object.shape[1] == 1:
            data_object = data_object.squeeze(dim=1)

        if data_object.ndim != 3:
            raise ValueError(
                "Expected data_object with shape [B, F, T] or [B, 1, F, T]."
            )

        if data_object.shape[1] != self.n_input_frequency_bins:
            raise ValueError(
                f"Expected {self.n_input_frequency_bins} frequency bins, "
                f"but got {data_object.shape[1]}."
            )

        if data_object.shape[2] != self.n_input_frames:
            raise ValueError(
                f"Expected {self.n_input_frames} time frames, "
                f"but got {data_object.shape[2]}."
            )

        data_object = self.frequency_compression(data_object)
        data_object = data_object.unsqueeze(dim=1)

        return {"logits": self.net(data_object)}

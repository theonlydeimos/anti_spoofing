import random
from collections.abc import Callable

import torch
import torchaudio
from torch import nn


class WaveformToSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int,
        power: float = 2.0,
        normalized: bool = False,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
    ) -> None:
        super().__init__()

        self.stft_config = {
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "power": power,
            "normalized": normalized,
            "window_fn": window_fn,
        }

        self.stft = torchaudio.transforms.Spectrogram(**self.stft_config)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram = self.stft(waveform)
        log_spectrogram = torch.log(spectrogram.clamp(min=1e-10))

        return log_spectrogram


class TrimPadTransform(nn.Module):
    def __init__(
        self,
        size_to_crop_to: int = 750,
        random_crop: bool = True,
    ) -> None:
        super().__init__()
        self.size_to_crop_to: int = size_to_crop_to
        self.random_crop: bool = random_crop

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        time_shape: int = spectrogram.shape[-1]
        if time_shape > self.size_to_crop_to:
            if self.random_crop:
                n = random.randint(0, time_shape - self.size_to_crop_to)
                spectrogram = spectrogram[..., n : n + self.size_to_crop_to]
            else:
                spectrogram = spectrogram[..., : self.size_to_crop_to]

        elif time_shape < self.size_to_crop_to:
            spectrogram = torch.nn.functional.pad(
                spectrogram,
                (0, self.size_to_crop_to - time_shape),
                mode="constant",
                value=0.0,
            )

        return spectrogram

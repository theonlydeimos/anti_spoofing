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
    ) -> None:
        super().__init__()

        self.stft_config = {
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "power": power,
            "normalized": normalized,
        }

        self.stft = torchaudio.transforms.Spectrogram(**self.stft_config)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram = self.stft(waveform)
        log_spectrogram = torch.log(spectrogram.clamp(min=1e-10))
        return log_spectrogram

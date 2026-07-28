from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import torchaudio

from src.datasets.base_dataset import BaseDataset


class AntiSpoofingDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        partition: Literal["train", "dev", "eval"],
        instance_transforms: None | Mapping[str, Callable] = None,
        limit: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)

        if partition == "train":
            self.protocol_path = (
                self.data_root
                / "asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
            )
            self.audio_root_path = (
                self.data_root / "asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_train/flac"
            )
            shuffle = True
        elif partition == "dev":
            self.protocol_path = (
                self.data_root
                / "asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
            )
            self.audio_root_path = (
                self.data_root / "asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_dev/flac"
            )
            shuffle = False
        elif partition == "eval":
            self.protocol_path = (
                self.data_root
                / "asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
            )
            self.audio_root_path = (
                self.data_root / "asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_eval/flac"
            )
            shuffle = False
        else:
            raise ValueError(
                f"Invalid partition: {partition}. Must be one either 'train' or 'dev' or 'eval'."
            )

        index: list[dict] = []

        with open(self.protocol_path, "r") as f:
            for line in f:
                speaker_id, audio_file_id, _, attack_id, key = line.strip().split()
                audio_file_name: Path = Path(f"{audio_file_id}.flac")
                key = 0 if key == "spoof" else 1

                index.append(
                    {
                        "path": self.audio_root_path / audio_file_name,
                        "speaker_id": speaker_id,
                        "audio_file_id": audio_file_id,
                        "label": key,
                        "attack_id": attack_id,
                    }
                )

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle,
            instance_transforms=instance_transforms,
        )

    def load_object(self, path: str | Path):
        return torchaudio.load(path)[0][0]

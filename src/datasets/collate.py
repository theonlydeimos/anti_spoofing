import torch


def collate_fn(dataset_items: list[dict]):
    result_batch = {}

    spectrograms = []
    labels = []
    for sample in dataset_items:
        spectrograms.append(sample["data_object"])
        labels.append(sample["labels"])

    result_batch["data_object"] = torch.stack(spectrograms, dim=0)
    result_batch["labels"] = torch.tensor(labels)

    if "audio_file_id" in dataset_items[0]:
        result_batch["audio_file_id"] = [
            sample["audio_file_id"] for sample in dataset_items
        ]

    return result_batch

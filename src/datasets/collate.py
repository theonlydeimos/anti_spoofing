import torch


def collate_fn(dataset_items: list[dict]):
    result_batch = {}

    waveforms = []
    lengths = []
    labels = []
    for sample in dataset_items:
        waveforms.append(sample["data_object"])
        lengths.append(sample["data_object"].shape[0])
        labels.append(sample["labels"])

    result_batch["data_object"] = torch.nn.utils.rnn.pad_sequence(
        waveforms, batch_first=True, padding_value=0.0
    )
    result_batch["audio_lengths"] = torch.tensor(lengths)
    result_batch["labels"] = torch.tensor(labels)
    return result_batch

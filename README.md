# Voice Anti-Spoofing

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch project for detecting synthetic and voice-converted speech in the
Logical Access partition of the
[ASVspoof 2019](https://www.asvspoof.org/index2019.html) dataset.

The repository provides configurable training and inference pipelines,
experiment tracking through Weights & Biases or Comet ML, Equal Error Rate
(EER) evaluation, checkpointing, and CSV score export.

## Installation

Python 3.11 is recommended.

```bash
git clone https://github.com/theonlydeimos/anti_spoofing.git
cd anti_spoofing

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Authenticate before using online experiment tracking:

```bash
wandb login
```

## Dataset

Download ASVspoof 2019 and place its Logical Access files in the layout expected
by the default configuration:

```text
/path/to/data/
└── asvpoof-2019-dataset/
    └── LA/
        └── LA/
            ├── ASVspoof2019_LA_train/flac/
            ├── ASVspoof2019_LA_dev/flac/
            ├── ASVspoof2019_LA_eval/flac/
            └── ASVspoof2019_LA_cm_protocols/
                ├── ASVspoof2019.LA.cm.train.trn.txt
                ├── ASVspoof2019.LA.cm.dev.trl.txt
                └── ASVspoof2019.LA.cm.eval.trl.txt
```

The dataset is not distributed with this repository.

## Training

```bash
python train.py \
  paths.data_root=/absolute/path/to/data \
  writer.entity=YOUR_WANDB_ENTITY \
  writer.run_name=asvspoof2019_la
```

Configuration values can be overridden from the command line:

```bash
python train.py \
  paths.data_root=/absolute/path/to/data \
  writer.run_name=experiment_seed_10 \
  trainer.seed=10 \
  trainer.n_epochs=30
```

Run artifacts are saved under `saved/<run_name>/`. Use a unique run name for
each experiment.

## Inference

```bash
python inference.py \
  paths.data_root=/absolute/path/to/data \
  inferencer.from_pretrained=/absolute/path/to/model_best.pth \
  inferencer.save_dir=saved/evaluation \
  inferencer.prediction_filename=predictions.csv
```

The generated CSV contains one utterance ID and one real-valued bonafide score
per row, without a header:

```text
LA_E_1000137,3.1840920448303223
LA_E_1000273,-1.7628471851348877
```

Higher scores indicate stronger support for genuine speech.

## References

1. G. Lavrentyeva et al.,
   [“STC Antispoofing Systems for the ASVspoof2019 Challenge”](https://arxiv.org/abs/1904.05576),
   Interspeech 2019.
2. X. Wang and J. Yamagishi,
   [“A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection”](https://arxiv.org/abs/2103.11326),
   Interspeech 2021.
3. X. Wu et al.,
   [“A Light CNN for Deep Face Representation with Noisy Labels”](https://arxiv.org/abs/1511.02683).

The project structure is based on the
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

See [LICENSE](LICENSE).

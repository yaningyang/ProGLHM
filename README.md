# ProGLHM

**Promoter recognition with genomic language representations and hybrid modelling**

ProGLHM recognizes promoter sequences using frozen Nucleotide Transformer v2 (NT-v2) representations and a trainable CNN, shared bidirectional Mamba, Transformer and attention classifier. This repository packages the original research implementation with commands for encoding, training, checkpoint saving/loading, held-out testing and FASTA prediction.

Each species has **two independent binary classifiers**: TATA promoters versus genomic background, and Non-TATA promoters versus genomic background. The subtype is chosen by the user; the model does not classify a sequence as TATA versus Non-TATA.

## Repository contents

```text
ProGLHM/
├── proglhm/               # Model and command-line implementation
├── datasets/              # The 16 supplied tasks, counts and SHA-256 manifest
├── examples/              # Example FASTA input
├── main.py                # Alternative command-line launcher
├── requirements.txt       # Author-specified dependency versions
├── pyproject.toml
├── README.md
└── COPYRIGHT
```

Run the commands below from this repository directory. NT-v2 weights and trained task checkpoints are downloaded or generated separately; they are not bundled with the source and FASTA data.

## Installation

Use Python 3.10 on Linux with an NVIDIA CUDA GPU. The pinned Mamba implementation uses compiled CUDA extensions; native Windows is not the supported training environment. Windows users can run the GPU workflow inside WSL2 with the corresponding NVIDIA driver and CUDA toolchain.

The six author-specified versions are:

```text
numpy==1.26.4
biopython==1.86
transformers==4.38.1
torch==2.2.0
mamba-ssm==1.2.0.post1
causal-conv1d==1.2.0.post2
```

Install PyTorch before the CUDA extensions. The following example uses CUDA 11.8; the NVIDIA driver, CUDA toolkit and C++ compiler must be compatible with that build.

```bash
conda create -n proglhm python=3.10 -y
conda activate proglhm
python -m pip install --upgrade pip
python -m pip install "setuptools>=68" wheel packaging ninja
python -m pip install numpy==1.26.4 biopython==1.86 transformers==4.38.1
python -m pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install causal-conv1d==1.2.0.post2 --no-build-isolation
python -m pip install mamba-ssm==1.2.0.post1 --no-build-isolation
python -m pip install -e . --no-deps
```

Check that the installed backend can execute a CUDA forward and backward pass:

```bash
python - <<'PY'
import torch
from mamba_ssm import Mamba

if not torch.cuda.is_available():
    raise RuntimeError("An NVIDIA CUDA device is required")
layer = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)
layer(x).square().mean().backward()
print("Mamba CUDA backend ready")
PY
```

The preprocessing and dataset-audit commands can run without a Mamba GPU backend. Training, held-out testing and prediction with the complete model require the real Mamba implementation; the application does not silently replace it with another layer.

## Tasks and data

```bash
python -m proglhm list-tasks
python -m proglhm audit --data-root datasets
```

| Species argument | Organism | Input length |
| --- | --- | --- |
| `A.thaliana` | *Arabidopsis thaliana* | 251 bp |
| `D.melanogaster` | *Drosophila melanogaster* | 300 bp |
| `D.rerio` | *Danio rerio* | 300 bp |
| `G.gallus` | *Gallus gallus* | 300 bp |
| `H.sapiens` | *Homo sapiens* | 300 bp |
| `M.musculus` | *Mus musculus* | 300 bp |
| `R.norvegicus` | *Rattus norvegicus* | 300 bp |
| `Z.mays` | *Zea mays* | 251 bp |

Use `--subtype TATA` or `--subtype non-TATA`. The directory form `Non_TATA` is also accepted. A task directory contains four original FASTA files:

```text
datasets/H.sapiens_Non_TATA/
├── Pos_train.fasta
├── Neg_train.fasta
├── Pos_test.fasta
└── Neg_test.fasta
```

`Pos` denotes promoters (class 1), and `Neg` denotes genomic background (class 0). The 64 FASTA files contain 280,944 sequences: 140,472 promoters and the same number of background windows, divided into 224,924 training-pool and 56,020 test records. The promoter totals are 22,881 TATA and 117,591 Non-TATA sequences. Both classes are balanced within every task and fixed split.

The provided training files form the training pool; 10% of their records are assigned to internal validation using a local random generator with seed 200. Test files are used only by an explicit testing command. The release preserves the supplied FASTA bytes. `datasets/manifest.json` is used by the audit command to check the FASTA file hashes and records the actual counts and sequence properties.

Positive sequences and subtype labels derive from the [Eukaryotic Promoter Database](https://epd.epfl.ch/). The study uses transcription-oriented windows from -200 to +50 bp in plants and -249 to +50 bp in animals, including both endpoints. Background sequences were sampled from the assembly-matched reference genomes described in the manuscript. Train/test chromosome labels are disjoint within each task; the two subtype tasks can hold out different chromosomes. The sequence audit does not independently remap genomic coordinates or exclude approximate homology.

## Frozen NT-v2 encoding

The encoder defaults to [`InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species). An online first run loads its tokenizer and weights through Hugging Face. The model uses custom Transformers code; loading it therefore uses `trust_remote_code=True`. An explicit revision identifies the encoder code and weights used by a run.

```bash
python -m proglhm encode \
  --species H.sapiens --subtype non-TATA --split train \
  --data-root datasets --cache-dir cache --batch-size 4
```

To use a local NT-v2 snapshot instead:

```bash
python -m proglhm encode \
  --species H.sapiens --subtype non-TATA --split train \
  --data-root datasets --cache-dir cache \
  --encoder /path/to/ntv2-snapshot --device cpu
```

For a Hub model, `--encoder-revision` accepts a revision or commit. Retain the resolved commit recorded by the cache and checkpoint when reproducing a run. Cached last-layer embeddings have 1,024 channels and are stored as float32 with the tokenizer's real attention mask, including its valid special tokens. Padding is zeroed. Every non-ACGT character is converted to `N` independently of its class. Sequences are never silently truncated; an input exceeding the 128-token limit is rejected.

A Hub-trained checkpoint uses the recorded Hub identity; `--local-files-only` uses an already downloaded Hugging Face cache. A checkpoint trained with a local encoder accepts an identical directory at another path via `--encoder`. Hub and local-directory identities are not interchangeable. A complete local snapshot must include the model configuration, tokenizer, custom code and weights.

Each float32 embedding occupies 0.5 MiB. Caching all 280,944 sequences requires approximately 137 GiB, so allocate disk space for the selected tasks. The frozen encoder is released before classifier training.

## Train and save one task model

```bash
python -m proglhm train \
  --species H.sapiens --subtype non-TATA \
  --data-root datasets --cache-dir cache --output-dir runs
```

This command encodes or reuses the selected task's training pool, trains its classifier, and saves its checkpoints under `runs/H.sapiens_Non_TATA/`. It does not load or evaluate the held-out test FASTA files.

| Setting | Default |
| --- | --- |
| Random seed | 200 |
| Maximum epochs | 200 |
| Training batch size | 32 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4`; parameters marked `_no_weight_decay` use 0 |
| Internal validation | Unstratified 10% of training-pool records |
| Checkpoint selection | Strict improvement in validation accuracy |
| Learning-rate schedule | ReduceLROnPlateau, factor 0.5, patience 10 |
| Early stopping | 20 epochs without validation-accuracy improvement |
| Training precision | CUDA automatic mixed precision; `--no-amp` disables it |

The training loader drops its incomplete final batch, following the source implementation. A small custom dataset must contain enough training records for the requested batch size.

The task directory contains:

| File | Contents |
| --- | --- |
| `best.pt` | Validation-selected model state, task/encoder/data configuration, selected epoch and validation scores |
| `last.pt` | Epoch-boundary training state: model, optimizer, scheduler, mixed-precision scaler, random generators, split indices and the selected best snapshot |
| `split.json` | Record indices and identity for the internal training/validation split |
| `history.csv` | Per-epoch training/validation metrics, learning rate and learned mixing coefficients |

The model configuration binds each checkpoint to its species, subtype, architecture, encoder identity and data fingerprint. Resume uses the same configuration and restores the last committed epoch boundary; it does not extend a completed run's configured epoch limit. Resume only from trusted `last.pt` files created by your own runs: restoring Python/NumPy random states requires unrestricted checkpoint deserialization.

Keep the entire task directory when resuming, including `last.pt` and `history.csv`. The recorded history is checked against the saved epoch boundary.

```bash
python -m proglhm train \
  --species H.sapiens --subtype non-TATA \
  --data-root datasets --cache-dir cache --output-dir runs --resume
```

To train all 16 classifiers separately:

```bash
python -m proglhm train-all \
  --data-root datasets --cache-dir cache --output-dir runs
```

## Load and test a saved model

```bash
python -m proglhm test \
  --species H.sapiens --subtype non-TATA \
  --checkpoint runs/H.sapiens_Non_TATA/best.pt \
  --data-root datasets --cache-dir cache --output-dir results
```

The command loads the requested checkpoint and evaluates only that species/subtype's `Pos_test.fasta` and `Neg_test.fasta`. It checks that the checkpoint task matches the command arguments. For automatic checkpoint selection, replace `--checkpoint` with `--model-dir runs`; this selects `<model-dir>/<task>/best.pt`.

Outputs are `results/<task>/predictions.csv` and `results/<task>/metrics.json`. The per-record CSV includes the original record ID, task, true label, promoter probability, predicted label/class, and the two logits.

Testing uses a default batch size of 64, matching the original implementation. Training retains its default batch size of 32.

The two logits are ordered as background (0) and promoter (1). The predicted label is their argmax; promoter probability is the positive-class softmax probability. Test outputs include per-record predictions, confusion counts, accuracy, positive-class F1, MCC, sensitivity, specificity and precision. A metric with a zero denominator is returned as 0.

```bash
python -m proglhm test-all \
  --checkpoint-root runs --data-root datasets \
  --cache-dir cache --output-dir results
```

## Predict a new FASTA file

Provide the species and the desired subtype-specific classifier explicitly:

```bash
python -m proglhm predict \
  --species H.sapiens --subtype non-TATA \
  --input examples/promoters.fasta \
  --model-dir runs --output predictions.csv
```

An explicit checkpoint can be used instead of `--model-dir`:

```bash
python -m proglhm predict \
  --species H.sapiens --subtype TATA \
  --input examples/promoters.fasta \
  --checkpoint runs/H.sapiens_TATA/best.pt \
  --output tata_predictions.csv --batch-size 4
```

Input windows must already have the species-appropriate length and transcriptional orientation used in training. The prediction command does not extract genomic windows, determine a strand, identify TSS coordinates, infer a species or assign a promoter subtype. Each FASTA record is evaluated against the background class by the selected task model.

`examples/promoters.fasta` contains two 300-bp windows from the released human Non-TATA test set: the first positive and first background record. Their labels are omitted from the example input. These windows demonstrate the command and do not form an additional independent evaluation set.

Prediction uses batch size **4** by default for both NT-v2 encoding and classifier inference. Its CSV contains `id`, `task`, `promoter_probability`, `predicted_label`, `predicted_class`, `background_logit` and `promoter_logit`. A sidecar `<output>.metadata.json` records input/checkpoint hashes and the encoder identity. Prediction does not write a label column for unlabeled input.

## Python loading API

```python
from proglhm import load_model

model, checkpoint = load_model(
    "runs/H.sapiens_Non_TATA/best.pt",
    species="H.sapiens",
    subtype="non-TATA",
    device="cuda",
)
```

The returned classifier is in evaluation mode. NT-v2 remains separate: `model((embeddings, valid_mask))` accepts float32 features shaped `[batch, tokens, 1024]` and a boolean right-padding mask, returning `[batch, 2]` logits. The token dimension must be a multiple of eight from 8 through 128, with at least one valid token per sequence. Use the CLI for FASTA-to-prediction processing, including frozen NT-v2 extraction.

## Implementation and validation

The trainable network retains the original computation: token projection; three CNN/pooling stages; a shared-parameter bidirectional Mamba branch before the first pool; masked global pooling and bounded residual feature-wise modulation; two Transformer encoder layers; and a separate masked attention readout with two output logits.

The classifier and training rules derive from the supplied main-model implementation. The release adds species/subtype selection, configurable paths, encoder identity metadata and separate testing/prediction commands. Its checkpoint wrapper stores this metadata alongside the original parameter names; older experiment checkpoint wrappers are not accepted directly by the release loader.

Preparation checks covered dataset integrity, interfaces, masking and checkpoint operations. Explicit encoder and Mamba stand-ins were used for the relevant integration checks. Real NT-v2 encoding and Mamba CUDA training have not been run under all six pinned versions; these require verification in the Linux CUDA environment before claiming a reproduced experimental run.

Trained task checkpoints are not included; the training command generates them.

## Copyright

The manuscript title is **ProGLHM: Promoter recognition by coupling genomic language representations with hybrid sequence modelling**. Cite the NT-v2 and Mamba sources when using those components.

Copyright (c) 2026 ProGLHM contributors. **All rights reserved.** The source is provided without an additional open-source license; see [COPYRIGHT](COPYRIGHT). NT-v2 weights, dependency software and upstream data retain their respective terms. The [NT-v2 model card](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species) identifies its weights as CC BY-NC-SA 4.0. The pinned [Mamba release](https://github.com/state-spaces/mamba/blob/v1.2.0.post1/LICENSE) uses Apache License 2.0; other dependencies retain their own licenses. EPD records and reference-genome sequences are not relicensed by this repository's copyright notice.


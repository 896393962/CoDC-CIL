# CoDC

Official PyTorch implementation of:

**CoDC: Unified Diffusion and Classification for Enhanced Class-Incremental Learning**

## Environment

Tested environment:

```text
GPU: NVIDIA GeForce RTX 3090 (24 GB)
CUDA: 12.4
Python: 3.11
PyTorch: CUDA-compatible build
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

Install the PyTorch and torchvision versions that match your local CUDA toolkit and driver.

## Data

Prepare datasets in ImageFolder format:

```text
dataset/
  train/
    class_000/
    class_001/
  test/
    class_000/
    class_001/
```

Set dataset paths in:

```text
ddpm/Diffusion/Train.py
```

Datasets and checkpoints are not included.

## Training

The main CoDC configuration is:

```text
ddpm/Main.py
```

Run:

```bash
cd ddpm
python Main.py
```

Default settings:

```text
T = 1000
channel = 128
channel_mult = [1, 2, 4, 4]
num_res_blocks = 3
dropout = 0.15
learning_rate = 1e-4
```

## Baselines

Standard CIL baselines are in:

```text
cil_survey/
```

Recent baseline utilities are in:

```text
recent_baselines/
```

Export collected reference results:

```bash
python recent_baselines/official_results.py --output official_recent_baselines.csv --print-markdown
```

## Structure

```text
ddpm/                         CoDC implementation
cil_survey/                   standard CIL baseline framework
recent_baselines/             recent baseline result utilities
third_party/improved_diffusion source-only diffusion utilities
```

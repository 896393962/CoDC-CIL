# CoDC: Unified Diffusion and Classification for Class-Incremental Learning

This repository contains a cleaned code snapshot for the paper:

**CoDC: Unified Diffusion and Classification for Enhanced Class-Incremental Learning**

The repository is prepared for manuscript review and public code availability. It includes the CoDC implementation, baseline-running utilities, and the selected baseline framework code needed to inspect or reproduce the reported experiments.

## Repository Contents

- `ddpm/`: core CoDC diffusion-classification implementation, PEDCC center generation utilities, diffusion trainers, and model definitions.
- `cil_survey/`: standard class-incremental learning baseline framework, including backbone definitions, model implementations, experiment configs, and shell scripts.
- `recent_baselines/`: scripts and records used to organize recent FeTrIL, SEED, DPCR, and related baseline comparisons.
- `third_party/improved_diffusion/`: source-only copy of the improved-diffusion utilities used for diffusion-related evaluation support.

## Excluded Files

The following files are intentionally excluded from this repository:

- benchmark datasets, including CIFAR-100 archives and extracted data;
- trained checkpoints and model weights (`.pt`, `.pth`, `.ckpt`);
- generated samples and intermediate outputs;
- logs, cached results, and temporary experiment artifacts;
- large third-party evaluation binaries and statistics files.

These files are excluded to keep the repository inspectable, lightweight, and suitable for public release. Datasets should be downloaded from their original public sources.

## Environment

The code was developed with Python and PyTorch on CUDA-enabled GPUs. A minimal environment can be prepared with:

```bash
pip install -r requirements.txt
```

For full training, use a CUDA PyTorch build compatible with the local GPU driver.

## Data Preparation

The training scripts expect image-folder style datasets. Update the dataset paths in:

```text
ddpm/Diffusion/Train.py
```

The original local paths are retained in the code to document the implementation used for the manuscript, but users should replace them with their own dataset paths before running experiments.

Expected structure:

```text
dataset_root/
  train/
    class_000/
    class_001/
    ...
  test/
    class_000/
    class_001/
    ...
```

## CoDC Training Entry Point

The main CoDC configuration is in:

```text
ddpm/Main.py
```

Run from the `ddpm` directory:

```bash
python Main.py
```

Important settings include:

- diffusion steps: `T = 1000`;
- UNet channel width: `128`;
- channel multipliers: `[1, 2, 4, 4]`;
- residual blocks per stage: `3`;
- dropout: `0.15`;
- learning rate: `1e-4`;
- generated replay sampling: DDIM-style sampling in the diffusion trainer.

## Recent Baseline Utilities

The `recent_baselines` directory provides scripts for organizing published or externally rerun baseline results and for launching official FeTrIL/SEED repositories when available.

Example:

```bash
python recent_baselines/official_results.py --output official_recent_baselines.csv --print-markdown
```

The launcher scripts do not vendor the official FeTrIL or SEED repositories. Clone those repositories separately and pass their local paths to the runner scripts.

## Notes for Reviewers

This repository is a cleaned public code snapshot. It does not backfill historical commits and does not include large datasets or checkpoints. The goal is to expose the implementation, configuration logic, and baseline scripts needed to verify the manuscript-level method description.

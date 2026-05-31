# Recent Baseline Utilities

This directory contains lightweight utilities used to organize recent baseline evidence for the CoDC manuscript.

## Export Published Reference Results

```bash
python official_results.py --output official_reference.csv --print-markdown
```

The exported records distinguish:

- `official-original`: values taken from the original FeTrIL or SEED papers;
- `third-party-reproduced`: values reproduced and reported by another paper;
- `cifar100_b20_inc20_n5`: the closest five-task CIFAR-100 protocol used for protocol-aware comparison with CoDC.

Do not describe third-party reproduced numbers as newly rerun official results.

## FeTrIL Runner

Clone the official FeTrIL repository separately, install its dependencies, and pass the repository path:

```bash
python run_fetril.py --repo /path/to/FeTrIL --config configs/cifar100_b50_t10.cf
```

To execute the command rather than only print it:

```bash
python run_fetril.py --repo /path/to/FeTrIL --config configs/cifar100_b50_t10.cf --execute
```

## SEED Runner

Clone the official SEED repository separately, install its dependencies, and pass the repository path:

```bash
python run_seed.py --repo /path/to/SEED --protocol equal_10_tasks
python run_seed.py --repo /path/to/SEED --protocol equal_20_tasks
python run_seed.py --repo /path/to/SEED --protocol equal_50_tasks
python run_seed.py --repo /path/to/SEED --protocol large_first
```

For a custom script:

```bash
python run_seed.py --repo /path/to/SEED --protocol custom --script my_cifar20x5_codc.sh --execute
```

## Environment Note

The runner scripts are wrappers. Full experiments require each official baseline repository to be configured with its own dependencies, dataset paths, class order, and random seed settings.

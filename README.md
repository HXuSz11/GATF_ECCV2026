# Geometry-Anchored Transport Framework (ECCV 2026) — Official Code

This repository contains the official implementation for the ECCV 2026 paper:

**Geometry-Anchored Transport Framework for Exemplar-Free Class-Incremental Learning**
Hongye Xu, Bartosz Krawczyk

------

## Overview

In exemplar-free class-incremental learning (EFCIL), models must learn new classes sequentially without storing raw examples from previous tasks. A common strategy is to store compact class-conditional statistics, such as prototypes or Gaussian distributions, and transport them into the current feature space for evaluation. However, representation drift can distort the legacy manifold, making transported statistics inaccurate and weakening Mahalanobis/Bayes classification.

We propose the **Geometry-Anchored Transport Framework (GATF)**, which integrates feature transport into the main training phase instead of treating it only as a post-hoc adaptation step.

The framework contains two key components:

- **Analytic Geometric Anchor (AGA):** a closed-form affine prior that captures macroscopic old-to-new feature drift using a geometry-aware GLS-style estimate.
- **Topology-Aware Evolution:** an end-to-end training objective that jointly regularizes the active backbone, the old-to-new residual adapter, and the new-to-old distiller to preserve the legacy feature topology.

Together, these components enable stable transport of old-class Gaussian statistics for exemplar-free Bayes/Mahalanobis evaluation.

------

## Codebase

Our implementation is based on the [AdaGauss](https://github.com/grypesc/AdaGauss) codebase and the [FACIL](https://github.com/mmasana/FACIL) benchmark-style exemplar-free continual learning framework.

The main method is implemented in:

```text
src/approach/gatf.py
```

The CUB-200 pretrained-backbone setting uses:

```text
src/approach/gatf_cub200.py
```

Experiment scripts are provided under:

```text
scripts/
```

------

## Setup

### 1) Create conda env + install dependencies

```bash
conda create -n yourenv python=3.10 -y
conda activate yourenv

# PyTorch (CUDA 12.6 wheels)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126

# Core libs
pip install timm==1.0.15 einops==0.8.1 \
  numpy==2.2.5 scipy==1.15.2 pandas==2.2.3 scikit-learn==1.6.1 \
  matplotlib==3.10.0 pillow==11.2.1 tqdm==4.67.1 pyyaml==6.0.2
```

> Notes:
>
> - For GPU training, ensure your NVIDIA driver supports CUDA 12.6.
> - The environment is the same as our BiCyc codebase.
> - If you want a minimal dependency set, keep only the core libraries required by the scripts you run.

------

## Datasets

### 1) Download datasets

Please download the datasets following your preferred local or cluster storage convention.

The supported datasets include:

- CIFAR-100
- TinyImageNet
- ImageNet-100
- CUB-200

### 2) Set dataset root path

Set the dataset root by editing:

```text
src/datasets/dataset_config.py
```

Modify the base path to your local dataset directory.

------

## Reproducing Experiments

We provide scripts under `scripts/`.

Each script supports the following environment variables:

- `GPU`: CUDA device ID.
- `RESULTS`: output directory for experiment results.

For example:

```bash
env GPU=0 RESULTS=./results_gatf/ bash scripts/cifar100_10x10_gatf.sh
```

------

### CIFAR-100, 10 tasks × 10 classes

```bash
env GPU=0 RESULTS=./results_gatf/ bash scripts/cifar100_10x10_gatf.sh
```

### TinyImageNet, 10 tasks × 20 classes

```bash
env GPU=1 RESULTS=./results_gatf/ bash scripts/tiny_10x20_gatf.sh
```

### ImageNet-100, 10 tasks × 10 classes

```bash
env GPU=0 RESULTS=./results_gatf/ bash scripts/imagenet100_10x10_gatf.sh
```

### CUB-200, 10 tasks × 20 classes

```bash
env GPU=0 RESULTS=./results_gatf/ bash scripts/cub200_10x20_gatf.sh
```

------

## Main Arguments

The main method is selected by:

```bash
--approach gatf
```

Important GATF-specific arguments include:

```bash
--distillation gatf
--lambda-geometry
--lambda-transport-cycle
--lambda-isometry
--preserve-pairwise-geometry
--lambda-pairwise-geometry

--prior-kind gls
--prior-mode train_adapt
--prior-ridge
--prior-batches
--gls-target-cov diag
--prior-update-interval
--prior-update-ema
--prior-update-ema-m
--no-adapt-train
```

The Gaussian/Bayes evaluation is enabled by:

```bash
--classifier bayes
--normalize
--rotation
```

------

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{xu2026gatf,
  title     = {Geometry-Anchored Transport Framework for Exemplar-Free Class-Incremental Learning},
  author    = {Xu, Hongye and Krawczyk, Bartosz},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

------

## Acknowledgements

This repository is built upon [AdaGauss](https://github.com/grypesc/AdaGauss),  [FACIL](https://github.com/mmasana/FACIL), and related exemplar-free continual learning codebases. We thank the respective authors for releasing their code.
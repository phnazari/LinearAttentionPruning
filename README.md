# DeltaNet Compression & Pruning

This repository provides a comprehensive suite of tools for compressing and pruning **DeltaNet** and **GatedDeltaNet** models. It focuses on structural pruning (reducing head dimensions and member counts) and low-rank compression to improve inference efficiency and reduce memory footprint while maintaining performance.


# Setup

```\bash
uv init --python 3.10
```

## 🚀 Features

- **Multiple Pruning Methods**:
  - **Strong RRQR (Rank-Revealing QR)**: Data-driven structural pruning using the Strong RRQR algorithm for optimal subspace selection.
  - **Wanda**: Pruning based on the Weights and Activations score.
  - **Taylor (LLM-Pruner)**: Gradient-based importance scoring for structural pruning.
  - **PCA**: Principal Component Analysis for low-rank key/query dimension reduction.
  - **L1 / Magnitude**: Data-free pruning based on weight or activation norms.
  - **Random**: Baseline random pruning.
- **Architectural Support**: Optimized for DeltaNet, DeltaNet2, and GatedDeltaNet with specific handling for `ShortConvolution` and shared kernels.
- **End-to-End Pipeline**: Scripts for pruning, DCP (DeepSpeed Checkpoint) conversion, and evaluation.
- **Flexible Strategies**: Support for pruning key dimensions (`head_k_dim`), value dimensions (`head_v_dim`), or entire heads.

## 📁 Repository Structure

```text
KeyReduction/
├── src/key_reduction/    # Core package logic
│   ├── pruners/          # Pruning algorithm implementations
│   └── utils/            # Shared utilities (plotting, etc.)
├── scripts/              # Execution scripts
│   ├── run_pruning/      # Individual pruning pipeline scripts
│   ├── multi_run/        # Scripts for automated multi-ratio runs
│   ├── benchmarking/     # Speed and memory benchmarking scripts
│   └── eval/             # Evaluation and table generation scripts
├── flame/                # Training and evaluation framework (submodule)
├── flash-linear-attention/ # Optimized kernels (submodule)
└── pyproject.toml        # Project configuration
```

## 🛠️ Installation

### 1. Setup Submodules
This repository depends on `flash-linear-attention` and `flame`.
```bash
git submodule update --init --recursive
```

### 2. Install Dependencies
```bash
# Install the core package and dependencies
pip install -e .

# Install submodules in editable mode if needed
pip install -e ./flash-linear-attention
pip install -e ./flame
```

## 📖 Usage

### Running a Pruning Pipeline
Each pruning method has a corresponding shell script in `scripts/run_pruning/`. These scripts handle the full pipeline: pruning -> conversion -> DCP save.

#### Example: Strong RRQR Pruning
```bash
BASE_MODEL_DIR="path/to/model" OUTPUT_DIR="path/to/output" bash scripts/run_pruning/run_rrqr.sh
```

#### Other Pipelines
- **Wanda**: `scripts/run_pruning/run_wanda.sh`
- **Taylor (Grad)**: `scripts/run_pruning/run_grad.sh`
- **PCA**: `scripts/run_pruning/run_pca.sh`
- **L1**: `scripts/run_pruning/run_l1.sh`

### Multi-Run Automation
To sweep across different pruning ratios automatically:
```bash
bash scripts/multi_run/run_multi_rrqr.sh 128 path/to/model path/to/output_base
```

## 📊 Evaluation & Benchmarking

### Benchmarking Throughput
```bash
COMPRESSED_MODEL="path/to/pruned_model" bash scripts/benchmarking/benchmark_mixer.sh
```

### Parallel Evaluation
```bash
COMPRESSED_BASE="path/to/checkpoints_dir" bash scripts/eval/eval_batch_parallel.sh delta_net 340m l2
```

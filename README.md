# KeyReduction: Structural Pruning for DeltaNet

This repository provides an end-to-end pipeline for training, compressing, and evaluating **DeltaNet** and **GatedDeltaNet** models. We focus on structural Q/K dimension reduction to improve efficiency while maintaining performance.

## 🚀 Quick Start

### 1. Training a Model
To train a model from scratch or continue training, use the pre-configured scripts or `train.sh` within the `flame` submodule.

```bash
# Example: Training a 340M DeltaNet on 10BT of fineweb-edu
# This script uses the flame submodule and optimized configurations
bash flame/flame/scripts/deltanet_340m.sh
```

### 2. Pruning a Model
We provide several structural pruning methods. You can prune either a local checkpoint or a model directly from the Hugging Face Hub.

#### Example: Strong RRQR (drrqr)
This method uses activations to identify which head dimensions are redundant.

```bash
# Pruning a local checkpoint
BASE_MODEL_DIR="./exp/gated-deltanet-340M/hf_checkpoint" \
OUTPUT_DIR="./exp/pruned_rrqr" \
PRUNING_RATIO=0.5 \
bash scripts/run_pruning/run_rrqr.sh

# Pruning from Hugging Face Hub
BASE_MODEL_DIR="m-a-p/1.3B-100B-GatedDeltaNet-pure" \
OUTPUT_DIR="./exp/pruned_wanda" \
PRUNING_RATIO=0.25 \
bash scripts/run_pruning/run_wanda.sh
```

### 3. LoRA Finetuning
After pruning, performance can be recovered by finetuning the model using LoRA.

```bash
# Apply LoRA to the pruned checkpoint
bash scripts/finetuning/run_lora.sh ./exp/pruned_rrqr ./exp/finetuned_rrqr
```

### 4. Evaluation with Multi-Evaluator
To evaluate your models (initial, pruned, and finetuned) across multiple benchmarks in parallel. This script automatically discovers all compressed and finetuned models in the provided directory.

```bash
# Evaluate all checkpoints in a folder
bash scripts/eval/eval_batch_parallel.sh ./exp/checkpoints
```

### 5. Benchmarking Performance
Verify the speedup and memory savings of your compressed models compared to the baseline.

```bash
# Benchmark throughput and latency
INITIAL_MODEL="path/to/baseline" \
COMPRESSED_MODEL="path/to/pruned" \
bash scripts/benchmarking/benchmark_forward.sh
```

### 6. State Rank Analysis
Analyze the rank utilization of recurrent states during forward passes to understand how well the model is utilizing its latent space.

```bash
# Compute rank utilization for a model
bash scripts/eval/run_effective_state_rank.sh /path/to/model ./outputs/rank_analysis
```
This script generates:
- **Rank Lists**: Per-head rank utilization tensors saved in `.pt` format.
- **Visualizations**: Combined plots showing rank evolution across layers.

## 🛠️ Installation & Setup

```bash
# 1. Clone with submodules
git clone --recursive https://github.com/phnazari/KeyReduction.git
cd KeyReduction

# 2. Setup environment
pip install -e .
pip install -e ./flash-linear-attention
pip install -e ./flame
```

## 📁 Repository Structure
- `src/key_reduction/`: Core pruning logic and algorithms.
- `scripts/run_pruning/`: Individual pipelines for RRQR, Wanda, Grad, L1, etc.
- `scripts/multi_run/`: Utilities for sweeping across multiple pruning ratios.
- `scripts/eval/`: Parallel evaluation and table generation.
- `scripts/benchmarking/`: Throughput and speedup verification.
- `flame/`: Training framework and base model implementations.

# KeyReduction: Structural Pruning for DeltaNet
This is the repository for the paper [The Key to State Reduction in Linear Attention: A Rank-based Perspective](google.com).

It allows for structured Q/K dimension reduction of DeltaNet and Gated DeltaNet models to improve efficiency while maintaining performance.

## Installation
First, clone this repository:
```bash
git clone git@github.com:phnazari/KeyReduction.git
```
Next, install the dependencies via `uv`:
```bash
uv venv --python=3.10
source .venv/bin/activate
uv sync
```
Now, you are ready to go!

## 🚀 Quick Start

### 1. Training a Model
To train a model from scratch or continue training, use the pre-configured scripts or `train.sh` within the `flame` submodule.

```bash
# Example: Training a 340M DeltaNet on 10BT of fineweb-edu
bash flame/flame/scripts/deltanet_340m.sh
```

### 2. Pruning a Model
We provide several structural pruning methods. You can prune either a local checkpoint or a model directly from the Hugging Face Hub.

#### Example: Deep RRQR (drrqr)
This method uses activations to identify which head dimensions are redundant.

```bash
BASE_MODEL_DIR="fla-hub/delta_net-1.3B-100B" \
OUTPUT_DIR="./exp/pruned_rrqr" \
PRUNING_RATIO=0.5 \
bash scripts/run_pruning/run_rrqr.sh
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

# Citation
If you find this repository helpful, please cite our work:
```
to fill in
```



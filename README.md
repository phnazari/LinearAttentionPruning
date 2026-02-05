<!--<p align="center">
  <img src="assets/logo.png" width="600" alt="KeyReduction">
</p>-->

<h1 align="center">KeyReduction</h1>

<p align="center">
  <b>The Key to State Reduction in Linear Attention: A Rank-based Perspective</b><br>
  <a href="https://arxiv.org/abs/2602.04852">Paper</a> • <a href="https://github.com/camail-official/LinearAttentionPruning">Repository</a>
</p>

---

This is the main repository accompanying the paper [The Key to State Reduction in Linear Attention: A Rank-based Perspective](google.com). It allows for structured Q/K dimension reduction of DeltaNet and Gated DeltaNet models to improve efficiency at a minimum loss of performance.

---

## Installation

First, clone this repository:
```bash
git clone git@github.com:phnazari/KeyReduction.git
cd KeyReduction
```

Next, install the dependencies via `uv`:
```bash
uv venv --python=3.10
source .venv/bin/activate
uv sync
```

## Quick Start

### 1. Training a Model
To train a model from scratch or continue training, use the pre-configured scripts or `train.sh` within the `flame` submodule.

```bash
bash flame/flame/scripts/deltanet_340m.sh
```

### 2. Pruning a Model
We provide several structural pruning methods. You can prune either a local checkpoint or a model directly from the Hugging Face Hub. Here, we show how to prune a pre-trained DeltaNet 1.3B model from [fla-hub](https://huggingface.co/fla-hub).

#### Example: Deep RRQR (DRRQR)
The following is an example for how to reduce the models key dimension by 50% using DRRQR:

```bash
bash scripts/run_pruning/run_rrqr.sh fla-hub/delta_net-1.3B-100B ./exp/pruned_rrqr 0.5
```

### 3. LoRA Finetuning
After pruning, performance can be recovered by finetuning the model using LoRA.

```bash
bash scripts/finetuning/run_lora.sh ./exp/pruned_rrqr/step-0 ./exp/finetuned_rrqr
```

### 4. Evaluation
We provide a script to evaluate the model across multiple benchmarks.

```bash
bash scripts/eval/eval_single_file.sh ./exp/finetuned_rrqr/checkpoints
```

## Additional Tools

### Benchmarking Performance
Verify the speedup and memory savings of your compressed models compared to the baseline.

```bash
bash scripts/benchmarking/benchmark_forward.sh fla-hub/delta_net-1.3B-100B exp/finetuned_rrqr/checkpoints
```

### State Rank Analysis
Analyze the rank utilization of recurrent states during forward passes to understand how well the model is utilizing its latent space.

```bash
bash scripts/eval/run_effective_state_rank.sh exp/finetuned_rrqr/checkpoints ./outputs/rank_analysis
```



## Citation
If you find this repository helpful, please cite our work:

```bibtex
@misc{nazari2026keystatereductionlinear,
      title={The Key to State Reduction in Linear Attention: A Rank-based Perspective}, 
      author={Philipp Nazari and T. Konstantin Rusch},
      year={2026},
      eprint={2602.04852},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.04852}, 
}
```

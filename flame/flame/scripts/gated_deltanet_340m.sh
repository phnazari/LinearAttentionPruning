#!/bin/bash

# Resolve repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FLAME_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Run training from the flame root
cd "$FLAME_ROOT"

NNODE=1 NGPU=8 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder outputs/gated_delta_net/340m/10BT \
  --model.config configs/gated_deltanet_340M.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 7e-4 \
  --lr_scheduler.warmup_steps 950 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 32 \
  --training.seq_len 2048 \
  --training.context_len 2048 \
  --training.gradient_accumulation_steps 1 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-10BT \
  --training.steps 19064 \
  --training.dataset_split train \
  --training.num_workers 14 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --checkpoint.enable_checkpoint \
  --checkpoint.folder checkpoints \
  --checkpoint.interval 1000 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 100
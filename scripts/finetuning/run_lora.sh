#!/bin/bash
# run_lora.sh
# Parameter-Efficient Finetuning (LoRA) for compressed DeltaNet models

# Robust path handling
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Ensure all submodules and package code are in PYTHONPATH

set -e  # Exit on error

# ============================================================================
# Configuration & Arguments
# ============================================================================

# Usage: bash scripts/finetuning/run_lora.sh <COMPRESSED_CHECKPOINT> <FINETUNED_OUTPUT_DIR> [TEACHER_MODEL]
# Or use the legacy schema:
# Usage: MODEL_NAME=delta_net PARAMS=340m RANK=64 METHOD=l2 bash scripts/finetuning/run_lora.sh

if [ -n "$1" ] && [ -n "$2" ]; then
    COMPRESSED_CHECKPOINT="$1"
    FINETUNED_OUTPUT_DIR="$2"
    TEACHER_DIR="$3"
else
    # Fallback to user's logic if no explicit paths provided
    MODEL_NAME=${MODEL_NAME:-"delta_net"}
    PARAMS=${PARAMS:-"340m"}
    RANK=${RANK:-64}
    METHOD=${METHOD:-"l2"}
    BASE_DUMP_DIR=${BASE_DUMP_DIR:-"/fast/pnazari/flame/dump"}
    
    if [[ "$PARAMS" == "340m" ]]; then
        TOKENS="10BT"
    else
        TOKENS="100BT"
    fi
    
    COMPRESSED_CHECKPOINT="${BASE_DUMP_DIR}/${MODEL_NAME}/${PARAMS}/${TOKENS}/checkpoints/${METHOD}_compressed_${RANK}/step-0"
    FINETUNED_OUTPUT_DIR="${BASE_DUMP_DIR}/${MODEL_NAME}/${PARAMS}/${TOKENS}/checkpoints/${METHOD}_finetuned_${RANK}"
fi

# Automatic Teacher determination (if not explicitly provided)
if [ -z "$TEACHER_DIR" ]; then
    if [[ "$PARAMS" == "340m" ]]; then
        TEACHER_DIR="${COMPRESSED_CHECKPOINT}/../../"
    elif [[ "$MODEL_NAME" == "gated_delta_net" ]]; then
        TEACHER_DIR="m-a-p/${PARAMS}-100B-GatedDeltaNet-pure"
    else
        TEACHER_DIR="fla-hub/${MODEL_NAME}-${PARAMS}-100B"
    fi
fi

# LoRA configuration
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q_proj,k_proj,v_proj"}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.00}
LORA_BIAS=${LORA_BIAS:-"none"}

# ============================================================================
# Run LoRA finetuning
# ============================================================================

echo "========================================="
echo "LoRA Finetuning Pipeline"
echo "========================================="
echo "  Repo Root:             $REPO_ROOT"
echo "  Compressed checkpoint: $COMPRESSED_CHECKPOINT"
echo "  Finetuned output:      $FINETUNED_OUTPUT_DIR"
echo "  Teacher model:         $TEACHER_DIR"
echo "  LoRA target modules:   $LORA_TARGET_MODULES"
echo "  LoRA rank:             $LORA_RANK (Alpha: $LORA_ALPHA)"
echo "========================================="

# Verify checkpoint exists
if [ ! -d "$COMPRESSED_CHECKPOINT" ]; then
  echo "ERROR: Compressed checkpoint not found at: $COMPRESSED_CHECKPOINT"
  exit 1
fi

# Clean up any existing checkpoint folder to avoid shape mismatches
if [ -d "$FINETUNED_OUTPUT_DIR/checkpoints" ]; then
  echo "⚠ Removing existing finetuned checkpoint folder..."
  rm -rf "$FINETUNED_OUTPUT_DIR/checkpoints"
fi

# Launch training via flame/train.sh
# We use one GPU as default for testing, but user can override via env
NNODE=${NNODE:-1}
NGPU=${NGPU:-1}
LOG_RANK=${LOG_RANK:-0}

NNODE=$NNODE NGPU=$NGPU LOG_RANK=$LOG_RANK bash "$REPO_ROOT/flame/train.sh" \
  --job.config_file "$REPO_ROOT/flame/flame/models/fla.toml" \
  --job.dump_folder "$FINETUNED_OUTPUT_DIR" \
  --model.config "${COMPRESSED_CHECKPOINT}/../config.json" \
  --model.tokenizer_path "${COMPRESSED_CHECKPOINT}/.." \
  --model.use_lora \
  --model.lora_target_modules "$LORA_TARGET_MODULES" \
  --model.lora_rank $LORA_RANK \
  --model.lora_alpha $LORA_ALPHA \
  --model.lora_dropout $LORA_DROPOUT \
  --model.lora_bias $LORA_BIAS \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-4 \
  --lr_scheduler.warmup_steps 100 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 16 \
  --training.seq_len 2048 \
  --training.context_len 2048 \
  --training.gradient_accumulation_steps 1 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-10BT \
  --training.steps 2000 \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --checkpoint.enable_checkpoint \
  --checkpoint.initial_load_path "$COMPRESSED_CHECKPOINT" \
  --checkpoint.folder checkpoints \
  --checkpoint.interval 500 \
  --checkpoint.keep_latest_k 3 \
  --metrics.log_freq 10 \
  --model.teacher_model_path "$TEACHER_DIR" \
  --training.distillation_loss_weight 1.0 \
  --training.distillation_temperature 2.0

echo ""
echo "========================================="
echo "LoRA finetuning complete!"
echo "  Finetuned model saved to: $FINETUNED_OUTPUT_DIR"
echo "========================================="

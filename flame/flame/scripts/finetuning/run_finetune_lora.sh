#!/bin/bash

# LoRA finetuning script for compressed DeltaNet2 models
# 
# This applies LoRA (Low-Rank Adaptation) to target modules for 
# parameter-efficient finetuning using the PEFT library

# Activate virtual environment
source /lustre/path/to/exp/flame/.venv/bin/activate

# Change to working directory
cd /path/to/work

# ============================================================================
# Configuration
# ============================================================================

#RANK=32  # Rank used during compression

MODEL_NAME=${1:-"delta_net"}
PARAMS=${2:-"340m"}
RANK=${3:-64}
METHOD=${4:-"l2"}  # Compression method identifier (e.g., l2, wanda, etc.)

if [[ "$PARAMS" == "340m" ]]; then
    TOKENS="10BT"
else
    TOKENS="100BT"
fi

# Path to compressed checkpoint
COMPRESSED_CHECKPOINT="/path/to/flame/dump/${MODEL_NAME}/${PARAMS}/${TOKENS}/checkpoints/${METHOD}_compressed_${RANK}/step-0"

# Path to save finetuned model (output)
FINETUNED_OUTPUT_DIR="/path/to/flame/dump/${MODEL_NAME}/${PARAMS}/${TOKENS}/checkpoints/long_${METHOD}_finetuned_${RANK}"

# LoRA configuration
# All linear layers: attention (q,k,v,o,b,g), MLP (gate,up,down), output (lm_head)
LORA_TARGET_MODULES="q_proj,k_proj,v_proj"
LORA_RANK=32
LORA_ALPHA=64  # alpha=2*rank
LORA_DROPOUT=0.00                                   # LoRA dropout rate
LORA_BIAS="none"                                    # Bias training: none, all, or lora_only

# --- AUTOMATIC TEACHER DETERMINATION ---
# Logic derived from submission queue patterns
if [[ "$PARAMS" == "340m" ]]; then
    # 1. Local path for smaller 340m models
    TEACHER_DIR="/path/to/flame/dump/${MODEL_NAME}/${PARAMS}/${TOKENS}/checkpoints"
elif [[ "$MODEL_NAME" == "gated_delta_net" ]]; then
    # 2. Specific Hub ID for Gated DeltaNet (e.g., 1.3B)
    TEACHER_DIR="m-a-p/${PARAMS}-100B-GatedDeltaNet-pure"
else
    # 3. Standard Hub ID for DeltaNet (1.3B, 2.7B, etc.)
    TEACHER_DIR="fla-hub/${MODEL_NAME}-${PARAMS}-100B"
fi

# ============================================================================
# Run LoRA finetuning
# ============================================================================

echo "========================================="
echo "LoRA Finetuning (PEFT)"
echo "========================================="
echo "  Compressed checkpoint: $COMPRESSED_CHECKPOINT"
echo "  Finetuned output: $FINETUNED_OUTPUT_DIR"
echo "  LoRA target modules: $LORA_TARGET_MODULES"
echo "  LoRA rank: $LORA_RANK"
echo "  LoRA alpha: $LORA_ALPHA"
echo "  LoRA dropout: $LORA_DROPOUT"
echo "  LoRA bias: $LORA_BIAS"
echo ""

# Verify checkpoint exists
if [ ! -d "$COMPRESSED_CHECKPOINT" ]; then
  echo "ERROR: Compressed checkpoint not found at: $COMPRESSED_CHECKPOINT"
  echo ""
  echo "Please run the compression pipeline first:"
  echo "  ./run_compression_pipeline.sh"
  echo ""
  exit 1
fi

echo "✓ Checkpoint found"
echo "✓ LoRA will be applied during training"
echo ""

# Clean up any existing checkpoint folder to avoid shape mismatches
if [ -d "$FINETUNED_OUTPUT_DIR/checkpoints" ]; then
  echo "⚠ Removing existing finetuned checkpoint folder to avoid shape mismatches..."
  rm -rf "$FINETUNED_OUTPUT_DIR/checkpoints"
fi

cd /lustre/path/to/exp/flame

# Adjust based on training dynamics
# CRITICAL: LoRA finetuning requires higher LR than full finetuning
# Typical LoRA LR: 1e-4 to 5e-4 (10-50x higher than full finetuning)
NNODE=1 NGPU=1 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
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
  --training.batch_size 4 \
  --training.seq_len 8192 \
  --training.context_len 8192 \
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

if [ $? -ne 0 ]; then
  echo "ERROR: LoRA finetuning failed"
  exit 1
fi

echo ""
echo "========================================="
echo "LoRA finetuning complete!"
echo "========================================="
echo "  Finetuned model saved to: $FINETUNED_OUTPUT_DIR"
echo ""
echo "Note:"
echo "  - LoRA was applied to: $LORA_TARGET_MODULES"
echo "  - LoRA rank: $LORA_RANK, alpha: $LORA_ALPHA"
echo "  - All base model parameters remained frozen"
echo ""
echo "To merge LoRA weights back into the base model, use:"
echo "  from peft import PeftModel"
echo "  model = PeftModel.from_pretrained(base_model, '$FINETUNED_OUTPUT_DIR')"
echo "  merged_model = model.merge_and_unload()"
echo ""

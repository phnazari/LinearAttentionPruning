#!/bin/bash
# eval_batch_parallel.sh
# Parallel evaluation for DeltaNet models via Ray

# Resolve repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Ensure all submodules and package code are in PYTHONPATH
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$REPO_ROOT/flame:$REPO_ROOT/flash-linear-attention:$PYTHONPATH"

set -e

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME=${1:-"delta_net"}
PARAMS=${2:-"340m"}
METHOD=${3:-"l1"}

if [[ "$PARAMS" == "340m" ]]; then
    TOKENS="10BT"
else
    TOKENS="100BT"
fi

# Define short names for models
case "${MODEL_NAME}" in
    "delta_net")
        MODEL_NAME_SHORT="dn"
        ;;
    "gated_delta_net")
        MODEL_NAME_SHORT="gdn"
        ;;
    *)
        MODEL_NAME_SHORT="${MODEL_NAME}" 
        ;;
esac

TOKENIZER_PATH="fla-hub/transformer-1.3B-100B"

# Logic-based default paths with override support
DEFAULT_OUTPUT_BASE="/fast/pnazari/flame/dump/eval_drrqr/eval_${MODEL_NAME_SHORT}_${PARAMS}"
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-$DEFAULT_OUTPUT_BASE}

DEFAULT_COMPRESSED_BASE="/fast/pnazari/flame/dump/${MODEL_NAME}/${PARAMS}/${TOKENS}/checkpoints"
COMPRESSED_BASE=${4:-${COMPRESSED_BASE:-$DEFAULT_COMPRESSED_BASE}}

BATCH_SIZE=8
MAX_LENGTH="10000"
DTYPE="bfloat16"
STEP=-1
TASKS="arc_easy,arc_challenge,hellaswag,winogrande,piqa,wikitext,lambada"

echo "=========================================="
echo "Parallel Evaluation via Ray"
echo "=========================================="
echo "Root:             $REPO_ROOT"
echo "Searching in:     $COMPRESSED_BASE"
echo "Model Prefix:     $MODEL_NAME"
echo "=========================================="

python -u "$REPO_ROOT/scripts/eval/eval_batch_parallel.py" \
    --tasks "${TASKS}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --output_dir "${OUTPUT_BASE_DIR}" \
    --compressed_base "${COMPRESSED_BASE}" \
    --model_prefix "${MODEL_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --dtype "${DTYPE}" \
    --step "${STEP}" \
    --method "${METHOD}"
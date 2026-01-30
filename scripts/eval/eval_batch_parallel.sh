#!/bin/bash
# eval_batch_parallel.sh
# Parallel evaluation for DeltaNet models via Ray

# Robust path handling
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/flame:$REPO_ROOT/flash-linear-attention:$PYTHONPATH"

set -e

# Configuration
MODEL_NAME=${1:-"delta_net"}
PARAMS=${2:-"340m"}
METHOD=${3:-"l2"}

if [[ "$PARAMS" == "340m" ]]; then
    TOKENS="10BT"
else
    TOKENS="100BT"
fi

# Define short names
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
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-"$REPO_ROOT/eval_results/eval_${MODEL_NAME_SHORT}_${PARAMS}"}
COMPRESSED_BASE=${COMPRESSED_BASE:-""}

if [ -z "$COMPRESSED_BASE" ]; then
    echo "Usage: COMPRESSED_BASE=/path/to/checkpoints bash $0 <MODEL_NAME> <PARAMS> <METHOD>"
    exit 1
fi

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
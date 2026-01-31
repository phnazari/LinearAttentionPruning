# Resolve repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Ensure all submodules and package code are in PYTHONPATH
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$REPO_ROOT/flame:$REPO_ROOT/flash-linear-attention:$PYTHONPATH"

# ============================================================================
# Configuration
# ============================================================================

CHECKPOINT_DIR=$1
OUTPUT_DIR=${2:-"outputs/eval_results"}
TASKS=${TASKS:-"wikitext"}

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Usage: bash $0 <CHECKPOINT_DIR> [OUTPUT_DIR]"
    exit 1
fi

# Evaluation Settings
BATCH_SIZE=8
MAX_LENGTH=10000        # <--- Added Max Length
STEP=-1                 # -1 for latest checkpoint
DEVICE="cuda"
DTYPE="bfloat16"

# Tasks
DEFAULT_TASKS="wikitext"  # "hellaswag,arc_easy,arc_challenge,winogrande,wikitext,piqa,lambada"
TASKS="${DEFAULT_TASKS}"

# ============================================================================
# Run Evaluation
# ============================================================================

echo "========================================="
echo "DeltaNet Evaluation"
echo "========================================="
echo "Checkpoint:  ${CHECKPOINT_DIR}"
echo "Step:        ${STEP}"
echo "Tasks:       ${TASKS}"
echo "Max Length:  ${MAX_LENGTH}"
echo "Batch Size:  ${BATCH_SIZE}"
echo "Output Dir:  ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "Starting evaluation..."

#if [ "${IS_HF_CHECKPOINT}" = true ]; then
# HuggingFace Format
python -u "${REPO_ROOT}/scripts/eval/eval_checkpoint.py \
    --pretrained "${CHECKPOINT_DIR}" \
    --tasks "${TASKS}" \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --device ${DEVICE} \
    --dtype ${DTYPE} \
    --output_path "${OUTPUT_DIR}/eval_harness_results.json"

echo ""
echo "✓ Evaluation completed successfully"
echo "========================================="
echo "Results saved to: ${OUTPUT_DIR}/eval_harness_results.json"
echo "========================================="
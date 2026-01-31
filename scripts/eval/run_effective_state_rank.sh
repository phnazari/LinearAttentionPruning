#!/bin/bash
# Example script for running effective state rank analysis
# This demonstrates how to use the adapted effective_state_rank.py script

# Resolve repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Ensure all submodules and package code are in PYTHONPATH
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$REPO_ROOT/flame:$REPO_ROOT/flash-linear-attention:$PYTHONPATH"

# Configuration
MODEL_PATH=${1:-"/path/to/model"}
OUTPUT_DIR=${2:-"${REPO_ROOT}/outputs/effective_rank_results"}

# Dataset configuration
DATASET="HuggingFaceFW/fineweb-edu"
DATASET_NAME="sample-10BT"

# Processing parameters
N_SAMPLES=1
BATCH_SIZE=1
SEQ_LEN=2048
NUM_WORKERS=4
SEED=54

if [ -z "$1" ] && [ "$MODEL_PATH" == "/path/to/model" ]; then
    echo "Usage: bash $0 <MODEL_PATH> [OUTPUT_DIR]"
    exit 1
fi

echo "=========================================="
echo "Effective State Rank Analysis"
echo "=========================================="
echo "Model:  ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run the analysis
python "${SCRIPT_DIR}/effective_state_rank.py" \
    --model_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --dataset_name "${DATASET_NAME}" \
    --n_samples "${N_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --seq_len "${SEQ_LEN}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "Results saved to: ${OUTPUT_DIR}/ranks/"
echo "Plots saved to: ${OUTPUT_DIR}/plots/"
echo "=========================================="

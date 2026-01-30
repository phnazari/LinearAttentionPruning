#!/bin/bash
# run_rrqr.sh
# Activation-Based RRQR Pruning pipeline for DeltaNet

# Robust path handling
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/flame:$REPO_ROOT/flash-linear-attention:$PYTHONPATH"

set -e  # Exit on error

# Pruning Settings
PRUNING_RATIO=${PRUNING_RATIO:-0.5}
PRUNING_STRATEGY=${PRUNING_STRATEGY:-"dimension"}

# Calibration Settings
DATASET="HuggingFaceFW/fineweb-edu"
DATASET_NAME="sample-10BT"
NUM_BATCHES=${NUM_BATCHES:-16}
BATCH_SIZE=1
SEQ_LEN=2048

# Configuration
# BASE_MODEL_DIR and OUTPUT_DIR should be provided by environment or arguments
if [ -z "$BASE_MODEL_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: BASE_MODEL_DIR=/path/to/model OUTPUT_DIR=/path/to/output bash $0"
    exit 1
fi

echo "=========================================="
echo "DeltaNet Activation-Based RRQR Pipeline"
echo "=========================================="
echo "Root:     $REPO_ROOT"
echo "Input:    $BASE_MODEL_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Ratio:    $PRUNING_RATIO"
echo "Strategy: $PRUNING_STRATEGY"
echo "=========================================="

# Step 1: Apply Pruning
echo "Step 1: Running RRQR pruning..."
python "$REPO_ROOT/src/key_reduction/pruners/rrqr.py" \
    --model_path "$BASE_MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --pruning_ratio "$PRUNING_RATIO" \
    --pruning_strategy "$PRUNING_STRATEGY" \
    --dataset "$DATASET" \
    --dataset_name "$DATASET_NAME" \
    --num_batches "$NUM_BATCHES" \
    --batch_size "$BATCH_SIZE" \
    --seq_len "$SEQ_LEN" \
    --num_workers 4 \
    --seed 42

echo ""
echo "✓ Pruning complete!"

# Step 2: Convert to DCP (Using flame module directly)
echo ""
echo "Step 2: Converting pruned model to DCP format..."
python -m flame.utils.convert_hf_to_dcp \
    --model "$OUTPUT_DIR" \
    --checkpoint "$OUTPUT_DIR/step-0"

echo ""
echo "✓ DCP checkpoint created at $OUTPUT_DIR/step-0"
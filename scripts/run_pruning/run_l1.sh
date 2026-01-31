#!/bin/bash
# run_l1.sh
# L1-Based Low-Rank Pruning pipeline for DeltaNet

# Robust path handling
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"


set -e  # Exit on error

# Pruning Settings
PRUNING_RATIO=${PRUNING_RATIO:-0.5}
PRUNING_STRATEGY=${PRUNING_STRATEGY:-"dimension"} # Options: 'dimension' (only Q/K reduction)

# Configuration
BASE_MODEL_DIR=${BASE_MODEL_DIR:-$1}
OUTPUT_DIR=${OUTPUT_DIR:-$2}

if [ -z "$BASE_MODEL_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: bash $0 <BASE_MODEL_DIR> <OUTPUT_DIR>"
    echo "Or: BASE_MODEL_DIR=/path/to/model OUTPUT_DIR=/path/to/output bash $0"
    exit 1
fi

echo "=========================================="
echo "DeltaNet L1 Pruning Pipeline"
echo "=========================================="
echo "Root:     $REPO_ROOT"
echo "Input:    $BASE_MODEL_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Ratio:    $PRUNING_RATIO"
echo "Strategy: $PRUNING_STRATEGY"
echo "=========================================="

# Step 1: Apply Pruning
echo "Step 1: Running L1 pruning (data-free)..."
python "$REPO_ROOT/src/key_reduction/pruners/l1.py" \
    --model_path "$BASE_MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --pruning_ratio "$PRUNING_RATIO" \
    --pruning_strategy "$PRUNING_STRATEGY" \
    --seed 42

echo ""
echo "✓ Pruning complete!"

# Step 2: Convert to DCP
echo ""
echo "Step 2: Converting pruned model to DCP format..."
python $REPO_ROOT/flame/flame/utils/convert_hf_to_dcp.py \
    --model "$OUTPUT_DIR" \
    --checkpoint "$OUTPUT_DIR/step-0"

echo ""
echo "✓ DCP checkpoint created at $OUTPUT_DIR/step-0"

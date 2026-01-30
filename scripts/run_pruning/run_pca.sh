#!/bin/bash
# run_pca.sh
# PCA-Based Low-Rank Pruning pipeline for DeltaNet

# Robust path handling
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/flame:$REPO_ROOT/flash-linear-attention:$PYTHONPATH"

set -e  # Exit on error

# Pruning Settings
TARGET_RANK=${TARGET_RANK:-64}
N_CALIBRATION_SAMPLES=${N_CALIBRATION_SAMPLES:-1024}
BATCH_SIZE=${BATCH_SIZE:-4}
SEQ_LEN=${SEQ_LEN:-2048}
VARIANCE_THRESHOLD=${VARIANCE_THRESHOLD:-0.95}

# Configuration
if [ -z "$BASE_MODEL_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: BASE_MODEL_DIR=/path/to/model OUTPUT_DIR=/path/to/output bash $0"
    exit 1
fi

echo "=========================================="
echo "DeltaNet PCA Pruning Pipeline"
echo "=========================================="
echo "Root:     $REPO_ROOT"
echo "Input:    $BASE_MODEL_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Rank:     $TARGET_RANK"
echo "Samples:  $N_CALIBRATION_SAMPLES"
echo "=========================================="

# Step 1: Apply Pruning
echo "Step 1: Running PCA pruning..."
python "$REPO_ROOT/src/key_reduction/pruners/pca.py" \
    --model_path "$BASE_MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_rank "$TARGET_RANK" \
    --n_calibration_samples "$N_CALIBRATION_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --seq_len "$SEQ_LEN" \
    --variance_threshold "$VARIANCE_THRESHOLD" \
    --compression_mode "pca" \
    --seed 42

echo ""
echo "✓ PCA Pruning complete!"

# Step 2: Convert to DCP
echo ""
echo "Step 2: Converting pruned model to DCP format..."
python -m flame.utils.convert_hf_to_dcp \
    --model "$OUTPUT_DIR" \
    --checkpoint "$OUTPUT_DIR/step-0"

echo ""
echo "✓ DCP checkpoint created at $OUTPUT_DIR/step-0"

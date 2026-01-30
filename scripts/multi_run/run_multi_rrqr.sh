#!/bin/bash
# run_multi_rrqr.sh
# Automates RRQR-based pruning for multiple ratios

# Robust path handling
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Define pruning ratios
RATIOS=(0.5)

# Define original head dimension
ORIGINAL_DIM=${1:-128}

# Base model path and output base provided by arguments
BASE_MODEL_DIR=${2:-""}
BASE_OUTPUT_DIR=${3:-""}

if [ -z "$BASE_MODEL_DIR" ] || [ -z "$BASE_OUTPUT_DIR" ]; then
    echo "Usage: bash $0 <ORIGINAL_DIM> <BASE_MODEL_DIR> <BASE_OUTPUT_DIR>"
    exit 1
fi

echo "Starting automated RRQR datapruning runs..."
echo "Original Dimension: $ORIGINAL_DIM"
echo "Repo Root:          $REPO_ROOT"
echo "-----------------------------------"

for RATIO in "${RATIOS[@]}"; do
    RANK=$(python3 -c "import math; print(math.floor($ORIGINAL_DIM * (1 - $RATIO)))")
    
    export PRUNING_RATIO=$RATIO
    export OUTPUT_DIR="${BASE_OUTPUT_DIR}/rrqr_data_compressed_${RANK}"
    export BASE_MODEL_DIR=$BASE_MODEL_DIR
    
    echo "Running RRQR Pruning: Ratio=$RATIO -> Calculated Rank=$RANK"
    
    # Call the updated RRQR pipeline script
    bash "$REPO_ROOT/scripts/run_pruning/run_rrqr.sh"
    
    echo "Completed Rank $RANK"
    echo "-----------------------------------"
done

echo "All RRQR pruning runs completed successfully!"

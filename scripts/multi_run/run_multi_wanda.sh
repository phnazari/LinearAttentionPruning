#!/bin/bash
# run_multi_wanda.sh
# Automates Wanda-based pruning for multiple ratios

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

echo "Starting automated Wanda pruning runs..."
echo "Original Dimension: $ORIGINAL_DIM"
echo "Repo Root:          $REPO_ROOT"
echo "-----------------------------------"

for RATIO in "${RATIOS[@]}"; do
    RANK=$(python3 -c "import math; print(math.floor($ORIGINAL_DIM * (1 - $RATIO)))")
    
    export PRUNING_RATIO=$RATIO
    export OUTPUT_DIR="${BASE_OUTPUT_DIR}/wanda_compressed_${RANK}"
    export BASE_MODEL_DIR=$BASE_MODEL_DIR
    
    echo "Running Wanda Pruning: Ratio=$RATIO -> Calculated Rank=$RANK"
    
    # Call the updated pruning script
    bash "$REPO_ROOT/scripts/run_pruning/run_wanda.sh"
    
    echo "Completed Rank $RANK"
    echo "-----------------------------------"
done

echo "All Wanda pruning runs completed successfully!"

#!/bin/bash
# run_multi_pca.sh
# Automates PCA-based pruning for multiple target ranks

# Robust path handling
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Define target ranks
RANKS=(64)

# Base model path and output base provided by arguments
BASE_MODEL_DIR=${1:-""}
BASE_OUTPUT_DIR=${2:-""}

if [ -z "$BASE_MODEL_DIR" ] || [ -z "$BASE_OUTPUT_DIR" ]; then
    echo "Usage: bash $0 <BASE_MODEL_DIR> <BASE_OUTPUT_DIR>"
    exit 1
fi

echo "Starting automated PCA pruning runs..."
echo "Repo Root:          $REPO_ROOT"
echo "-----------------------------------"

for RANK in "${RANKS[@]}"; do
    export TARGET_RANK=$RANK
    export OUTPUT_DIR="${BASE_OUTPUT_DIR}/pca_compressed_${RANK}"
    export BASE_MODEL_DIR=$BASE_MODEL_DIR
    
    echo "Running PCA Pruning: Target Rank=$RANK"
    
    # Call the updated pruning script
    bash "$REPO_ROOT/scripts/run_pruning/run_pca.sh"
    
    echo "Completed Rank $RANK"
    echo "-----------------------------------"
done

echo "All PCA pruning runs completed successfully!"

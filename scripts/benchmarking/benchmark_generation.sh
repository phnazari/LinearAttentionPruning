#!/bin/bash
# benchmark_generation.sh
# Benchmark autoregressive generation for initial vs compressed DeltaNet2 models

# Robust path handling
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/flame:$REPO_ROOT/flash-linear-attention:$PYTHONPATH"

set -e  # Exit on error

# Configuration
INITIAL_MODEL=${INITIAL_MODEL:-""}
COMPRESSED_MODEL=${COMPRESSED_MODEL:-""}

if [ -z "$INITIAL_MODEL" ] || [ -z "$COMPRESSED_MODEL" ]; then
    echo "Usage: INITIAL_MODEL=/path/to/initial COMPRESSED_MODEL=/path/to/compressed bash $0"
    exit 1
fi

# Generation settings
PROMPT_LENGTH=128
MAX_GEN_LENGTH=256
DATA="fla-hub/pg19"

echo "========================================="
echo "Generation Throughput Benchmark"
echo "========================================="
echo "Root:              $REPO_ROOT"
echo "  Initial model:     $INITIAL_MODEL"
echo "  Compressed model:  $COMPRESSED_MODEL"
echo "  Prompt length:     $PROMPT_LENGTH tokens"
echo "  Max gen length:    $MAX_GEN_LENGTH tokens"
echo "  Dataset:           $DATA"
echo "  KV cache:          enabled (default)"
echo "========================================="
echo ""

# Temporary files to store results
INITIAL_RESULTS=$(mktemp)
COMPRESSED_RESULTS=$(mktemp)

# Benchmark initial model
echo "Step 1/2: Benchmarking initial model generation..."
python "$REPO_ROOT/flame/benchmarks/benchmark_generation.py" \
  --path "$INITIAL_MODEL" \
  --data "$DATA" \
  --length "$PROMPT_LENGTH" \
  --maxlen "$MAX_GEN_LENGTH" | tee "$INITIAL_RESULTS"

# Benchmark compressed model
echo ""
echo "Step 2/2: Benchmarking compressed model generation..."
python "$REPO_ROOT/flame/benchmarks/benchmark_generation.py" \
  --path "$COMPRESSED_MODEL" \
  --data "$DATA" \
  --length "$PROMPT_LENGTH" \
  --maxlen "$MAX_GEN_LENGTH" | tee "$COMPRESSED_RESULTS"

# Extract key metrics for comparison
echo ""
echo "======================================"
echo "COMPARISON SUMMARY"
echo "======================================"

# Extract number of parameters
INITIAL_PARAMS=$(grep "Number of parameters:" "$INITIAL_RESULTS" | sed 's/.*: \([0-9,]*\).*/\1/' | tr -d ',')
COMPRESSED_PARAMS=$(grep "Number of parameters:" "$COMPRESSED_RESULTS" | sed 's/.*: \([0-9,]*\).*/\1/' | tr -d ',')

# Extract prompt and generation length
INITIAL_PROMPT=$(grep "Prompt length:" "$INITIAL_RESULTS" | sed -n 's/.*Prompt length: \([0-9]*\), generation length: \([0-9]*\).*/\1/p')
COMPRESSED_PROMPT=$(grep "Prompt length:" "$COMPRESSED_RESULTS" | sed -n 's/.*Prompt length: \([0-9]*\), generation length: \([0-9]*\).*/\1/p')
INITIAL_GEN=$(grep "Prompt length:" "$INITIAL_RESULTS" | sed -n 's/.*Prompt length: \([0-9]*\), generation length: \([0-9]*\).*/\2/p')
COMPRESSED_GEN=$(grep "Prompt length:" "$COMPRESSED_RESULTS" | sed -n 's/.*Prompt length: \([0-9]*\), generation length: \([0-9]*\).*/\2/p')

# Extract total time in ms
INITIAL_TIME=$(grep "Total prompt processing + decoding time:" "$INITIAL_RESULTS" | sed -n 's/.*decoding time: \([0-9]*\)ms.*/\1/p')
COMPRESSED_TIME=$(grep "Total prompt processing + decoding time:" "$COMPRESSED_RESULTS" | sed -n 's/.*decoding time: \([0-9]*\)ms.*/\1/p')

# Extract max memory used
INITIAL_MEMORY=$(grep "Max memory used:" "$INITIAL_RESULTS" | awk '{print $4}')
COMPRESSED_MEMORY=$(grep "Max memory used:" "$COMPRESSED_RESULTS" | awk '{print $4}')

# Calculate metrics
if [[ -n "$INITIAL_TIME" && -n "$INITIAL_GEN" ]]; then
    INITIAL_TPS=$(awk "BEGIN {printf \"%.2f\", $INITIAL_GEN / ($INITIAL_TIME / 1000)}")
    INITIAL_TIME_SEC=$(awk "BEGIN {printf \"%.2f\", $INITIAL_TIME / 1000}")
else
    INITIAL_TPS="N/A"
    INITIAL_TIME_SEC="N/A"
fi

if [[ -n "$COMPRESSED_TIME" && -n "$COMPRESSED_GEN" ]]; then
    COMPRESSED_TPS=$(awk "BEGIN {printf \"%.2f\", $COMPRESSED_GEN / ($COMPRESSED_TIME / 1000)}")
    COMPRESSED_TIME_SEC=$(awk "BEGIN {printf \"%.2f\", $COMPRESSED_TIME / 1000}")
else
    COMPRESSED_TPS="N/A"
    COMPRESSED_TIME_SEC="N/A"
fi

# Calculate ratios
if [[ "$INITIAL_TPS" != "N/A" && "$COMPRESSED_TPS" != "N/A" ]]; then
    SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $COMPRESSED_TPS / $INITIAL_TPS}")
else
    SPEEDUP="N/A"
fi

if [[ -n "$INITIAL_PARAMS" && -n "$COMPRESSED_PARAMS" ]]; then
    PARAM_REDUCTION=$(awk "BEGIN {printf \"%.1f\", (1 - $COMPRESSED_PARAMS / $INITIAL_PARAMS) * 100}")
else
    PARAM_REDUCTION="N/A"
fi

if [[ -n "$INITIAL_TIME" && -n "$COMPRESSED_TIME" ]]; then
    TIME_REDUCTION=$(awk "BEGIN {printf \"%.2f\", $INITIAL_TIME / $COMPRESSED_TIME}")
else
    TIME_REDUCTION="N/A"
fi

printf "%-35s %20s %20s %15s\n" "Metric" "Initial" "Compressed" "Speedup"
echo "-------------------------------------------------------------------------------------"
printf "%-35s %20s %20s\n" "Parameters" "$INITIAL_PARAMS" "$COMPRESSED_PARAMS"
printf "%-35s %20s %20s\n" "Prompt length" "$INITIAL_PROMPT" "$COMPRESSED_PROMPT"
printf "%-35s %20s %20s\n" "Generated tokens" "$INITIAL_GEN" "$COMPRESSED_GEN"
printf "%-35s %20s %20s %15s\n" "Total time (s)" "$INITIAL_TIME_SEC" "$COMPRESSED_TIME_SEC" "${TIME_REDUCTION}x"
printf "%-35s %20s %20s %15s\n" "Tokens/second" "$INITIAL_TPS" "$COMPRESSED_TPS" "${SPEEDUP}x"
printf "%-35s %20s %20s\n" "Peak Memory" "$INITIAL_MEMORY" "$COMPRESSED_MEMORY"
echo "-------------------------------------------------------------------------------------"
echo ""
echo "Parameter reduction: ${PARAM_REDUCTION}%"

# Clean up
rm -f "$INITIAL_RESULTS" "$COMPRESSED_RESULTS"

echo ""
echo "======================================"
echo "Benchmark completed successfully!"
echo "======================================"

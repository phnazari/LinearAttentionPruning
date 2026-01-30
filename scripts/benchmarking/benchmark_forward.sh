#!/bin/bash
# benchmark_forward.sh
# Benchmark forward-only throughput for initial vs compressed DeltaNet2 models

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

# Benchmark settings
BATCH_SIZE=32
SEQ_LEN=2048
WARMUP_STEPS=10
BENCHMARK_STEPS=50
DTYPE="bfloat16"

echo "========================================="
echo "Forward-Only Throughput Benchmark"
echo "========================================="
echo "Root:              $REPO_ROOT"
echo "Initial model:     $INITIAL_MODEL"
echo "Compressed model:  $COMPRESSED_MODEL"
echo "Batch size:        $BATCH_SIZE"
echo "Sequence length:   $SEQ_LEN"
echo "Warmup steps:      $WARMUP_STEPS"
echo "Benchmark steps:   $BENCHMARK_STEPS"
echo "Data type:         $DTYPE"
echo "========================================="
echo ""

# Temporary files to store results
INITIAL_RESULTS=$(mktemp)
COMPRESSED_RESULTS=$(mktemp)

# Benchmark initial model
echo "Step 1/2: Benchmarking initial model..."
python "$REPO_ROOT/flame/benchmarks/benchmark_forward.py" \
  --model "$INITIAL_MODEL" \
  --name "Initial" \
  --batch_size "$BATCH_SIZE" \
  --seq_len "$SEQ_LEN" \
  --warmup_steps "$WARMUP_STEPS" \
  --steps "$BENCHMARK_STEPS" \
  --dtype "$DTYPE" | tee "$INITIAL_RESULTS"

# Benchmark compressed model
echo ""
echo "Step 2/2: Benchmarking compressed model..."
python "$REPO_ROOT/flame/benchmarks/benchmark_forward.py" \
  --model "$COMPRESSED_MODEL" \
  --name "Compressed" \
  --batch_size "$BATCH_SIZE" \
  --seq_len "$SEQ_LEN" \
  --warmup_steps "$WARMUP_STEPS" \
  --steps "$BENCHMARK_STEPS" \
  --dtype "$DTYPE" | tee "$COMPRESSED_RESULTS"

# Extract key metrics for comparison
echo ""
echo "======================================"
echo "COMPARISON SUMMARY"
echo "======================================"

INITIAL_PARAMS=$(grep "Number of parameters:" "$INITIAL_RESULTS" | awk '{print $4}' | tr -d ',')
COMPRESSED_PARAMS=$(grep "Number of parameters:" "$COMPRESSED_RESULTS" | awk '{print $4}' | tr -d ',')

INITIAL_TRAINABLE_PARAMS=$(grep "Number of trainable parameters:" "$INITIAL_RESULTS" | awk '{print $5}' | tr -d ',')
COMPRESSED_TRAINABLE_PARAMS=$(grep "Number of trainable parameters:" "$COMPRESSED_RESULTS" | awk '{print $5}' | tr -d ',')

INITIAL_TPS=$(grep "Throughput:" "$INITIAL_RESULTS" | tail -1 | awk '{print $2}' | tr -d ',')
COMPRESSED_TPS=$(grep "Throughput:" "$COMPRESSED_RESULTS" | tail -1 | awk '{print $2}' | tr -d ',')

INITIAL_LATENCY=$(grep "Average latency per step:" "$INITIAL_RESULTS" | awk '{print $5}' | tr -d 'ms')
COMPRESSED_LATENCY=$(grep "Average latency per step:" "$COMPRESSED_RESULTS" | awk '{print $5}' | tr -d 'ms')

INITIAL_MEMORY=$(grep "Peak memory:" "$INITIAL_RESULTS" | tail -1 | awk '{print $3}')
COMPRESSED_MEMORY=$(grep "Peak memory:" "$COMPRESSED_RESULTS" | tail -1 | awk '{print $3}')

# Calculate ratios using awk
if [[ -n "$INITIAL_TPS" && -n "$COMPRESSED_TPS" ]]; then
    SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $COMPRESSED_TPS / $INITIAL_TPS}")
else
    SPEEDUP="N/A"
fi

if [[ -n "$INITIAL_PARAMS" && -n "$COMPRESSED_PARAMS" ]]; then
    PARAM_REDUCTION=$(awk "BEGIN {printf \"%.1f\", (1 - $COMPRESSED_PARAMS / $INITIAL_PARAMS) * 100}")
else
    PARAM_REDUCTION="N/A"
fi

printf "%-30s %20s %20s %15s\n" "Metric" "Initial" "Compressed" "Speedup"
echo "---------------------------------------------------------------------------------"
printf "%-30s %20s %20s %15s\n" "Parameters" "$INITIAL_PARAMS" "$COMPRESSED_PARAMS" "${SPEEDUP}x"
printf "%-30s %20s %20s\n" "Trainable Parameters" "$INITIAL_TRAINABLE_PARAMS" "$COMPRESSED_TRAINABLE_PARAMS"
printf "%-30s %20s %20s %15s\n" "Throughput (tokens/s)" "$INITIAL_TPS" "$COMPRESSED_TPS" "${SPEEDUP}x"
printf "%-30s %20s %20s\n" "Avg Latency (ms)" "$INITIAL_LATENCY" "$COMPRESSED_LATENCY"
printf "%-30s %20s %20s\n" "Peak Memory" "$INITIAL_MEMORY" "$COMPRESSED_MEMORY"
echo "---------------------------------------------------------------------------------"
echo ""
echo "Parameter reduction: ${PARAM_REDUCTION}%"

# Clean up
rm -f "$INITIAL_RESULTS" "$COMPRESSED_RESULTS"

echo ""
echo "======================================"
echo "Benchmark completed successfully!"
echo "======================================"

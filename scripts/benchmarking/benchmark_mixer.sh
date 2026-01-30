#!/bin/bash
# benchmark_mixer.sh
# Benchmarks only the sequence mixer for initial and compressed models.

# Robust path handling
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"


set -e  # Exit on error

# Configuration
COMPRESSED_MODEL=${COMPRESSED_MODEL:-$1}
INITIAL_MODEL=${INITIAL_MODEL:-"fla-hub/delta_net-1.3B-100B"}

if [ -z "$COMPRESSED_MODEL" ]; then
    echo "Usage: bash $0 <COMPRESSED_MODEL> [INITIAL_MODEL]"
    echo "Or: COMPRESSED_MODEL=/path/to/compressed bash $0"
    exit 1
fi

# Benchmark settings
BATCH_SIZE=32
SEQ_LEN=2048
WARMUP_STEPS=10
BENCHMARK_STEPS=50
DTYPE="bfloat16"

echo "========================================="
echo "Sequence Mixer Throughput Benchmark"
echo "========================================="
echo "Root:              $REPO_ROOT"
echo "Initial Model:     $INITIAL_MODEL"
echo "Compressed Model:  $COMPRESSED_MODEL"
echo "========================================="

run_benchmark() {
    local model=$1
    local name=$2
    echo "Benchmarking $name mixer..."
    python "$REPO_ROOT/flame/benchmarks/benchmark_mixer.py" \
        --model "$model" \
        --batch_size "$BATCH_SIZE" \
        --seq_len "$SEQ_LEN" \
        --warmup_steps "$WARMUP_STEPS" \
        --steps "$BENCHMARK_STEPS" \
        --dtype "$DTYPE"
}

# Run benchmarks and capture throughput
INITIAL_OUT=$(run_benchmark "$INITIAL_MODEL" "Initial")
echo "$INITIAL_OUT"
INITIAL_TPS=$(echo "$INITIAL_OUT" | grep "Throughput:" | awk '{print $2}' | tr -d ',')

echo ""

COMPRESSED_OUT=$(run_benchmark "$COMPRESSED_MODEL" "Compressed")
echo "$COMPRESSED_OUT"
COMPRESSED_TPS=$(echo "$COMPRESSED_OUT" | grep "Throughput:" | awk '{print $2}' | tr -d ',')

echo ""
echo "========================================="
echo "COMPARISON SUMMARY"
echo "========================================="
printf "%-25s %15s\n" "Model" "Throughput (tokens/s)"
echo "----------------------------------------------------"
printf "%-25s %15s\n" "Initial Mixer" "$INITIAL_TPS"
printf "%-25s %15s\n" "Compressed Mixer" "$COMPRESSED_TPS"
echo "----------------------------------------------------"

# Calculate speedup
if [[ -n "$INITIAL_TPS" && -n "$COMPRESSED_TPS" ]]; then
    SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $COMPRESSED_TPS / $INITIAL_TPS}")
    echo "Speedup: ${SPEEDUP}x"
else
    echo "Error: Could not extract throughput values."
fi
echo "========================================="

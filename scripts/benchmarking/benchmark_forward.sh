#!/bin/bash
# benchmark_forward.sh
#
# Benchmark forward-only throughput for initial vs compressed DeltaNet2 models
#
# This script measures pure inference performance (no backward pass) to isolate
# the impact of compression on forward pass speed.
#
# Usage:
#   cd /path/to/CompreSSM2/exp/flame
#   bash flame/scripts/benchmark_forward.sh

# Activate virtual environment
source /lustre/home/pnazari/workspace/CompreSSM2/exp/flame/.venv/bin/activate

# Change to flame directory
cd /lustre/home/pnazari/workspace/CompreSSM2/exp/flame

# Configuration (matching compression pipeline defaults)
#INITIAL_MODEL="fla-hub/delta_net-2.7B-100B"
INITIAL_MODEL="/fast/pnazari/flame/dump/gated_delta_net/340m/10BT/checkpoints"
#COMPRESSED_MODEL="/fast/pnazari/flame/dump/delta_net/340m/10BT/checkpoints/llm_pruner_ratio_0.5"
COMPRESSED_MODEL="/fast/pnazari/flame/dump/gated_delta_net/340m/10BT/checkpoints/compressed_16"

#INITIAL_MODEL="m-a-p/1.3B-100B-GatedDeltaNet-pure"
#COMPRESSED_MODEL="/fast/pnazari/flame/dump/gated_delta_net/1.3B/100BT/checkpoints/compressed_4"

# Benchmark settings
BATCH_SIZE=32
SEQ_LEN=2048
WARMUP_STEPS=10
BENCHMARK_STEPS=50
DTYPE="bfloat16"

echo "========================================="
echo "Forward-Only Throughput Benchmark"
echo "========================================="
echo "  Initial model:     $INITIAL_MODEL"
echo "  Compressed model:  $COMPRESSED_MODEL"
echo "  Batch size:        $BATCH_SIZE"
echo "  Sequence length:   $SEQ_LEN"
echo "  Warmup steps:      $WARMUP_STEPS"
echo "  Benchmark steps:   $BENCHMARK_STEPS"
echo "  Data type:         $DTYPE"
echo ""

# Check if models exist
#if [ ! -d "$INITIAL_MODEL" ]; then
#  echo "ERROR: Initial model not found at: $INITIAL_MODEL"
#  exit 1
#fi

if [ ! -d "$COMPRESSED_MODEL" ]; then
  echo "ERROR: Compressed model not found at: $COMPRESSED_MODEL"
  echo ""
  echo "Please run the compression pipeline first:"
  echo "  cd /lustre/home/pnazari/workspace/CompreSSM2"
  echo "  ./run_compression_pipeline.sh"
  echo ""
  exit 1
fi

# Temporary files to store results
INITIAL_RESULTS=$(mktemp)
COMPRESSED_RESULTS=$(mktemp)

# Benchmark initial model
echo "Step 1/2: Benchmarking initial model..."
python benchmarks/benchmark_forward.py \
  --model "$INITIAL_MODEL" \
  --name "Initial" \
  --batch_size "$BATCH_SIZE" \
  --seq_len "$SEQ_LEN" \
  --warmup_steps "$WARMUP_STEPS" \
  --steps "$BENCHMARK_STEPS" \
  --dtype "$DTYPE" | tee "$INITIAL_RESULTS"

if [ $? -ne 0 ]; then
  echo ""
  echo "ERROR: Initial model benchmark failed"
  rm -f "$INITIAL_RESULTS" "$COMPRESSED_RESULTS"
  exit 1
fi

# Benchmark compressed model
echo ""
echo "Step 2/2: Benchmarking compressed model..."
python benchmarks/benchmark_forward.py \
  --model "$COMPRESSED_MODEL" \
  --name "Compressed" \
  --batch_size "$BATCH_SIZE" \
  --seq_len "$SEQ_LEN" \
  --warmup_steps "$WARMUP_STEPS" \
  --steps "$BENCHMARK_STEPS" \
  --dtype "$DTYPE" | tee "$COMPRESSED_RESULTS"

if [ $? -ne 0 ]; then
  echo ""
  echo "ERROR: Compressed model benchmark failed"
  rm -f "$INITIAL_RESULTS" "$COMPRESSED_RESULTS"
  exit 1
fi

# Extract key metrics for comparison
echo ""
echo "======================================"
echo "COMPARISON SUMMARY"
echo "======================================"

INITIAL_PARAMS=$(grep "Number of parameters:" "$INITIAL_RESULTS" | awk '{print $4}' | tr -d ',')
COMPRESSED_PARAMS=$(grep "Number of parameters:" "$COMPRESSED_RESULTS" | awk '{print $4}' | tr -d ',')

# Extract trainable parameters
INITIAL_TRAINABLE_PARAMS=$(grep "Number of trainable parameters:" "$INITIAL_RESULTS" | awk '{print $5}' | tr -d ',')
COMPRESSED_TRAINABLE_PARAMS=$(grep "Number of trainable parameters:" "$COMPRESSED_RESULTS" | awk '{print $5}' | tr -d ',')

INITIAL_TPS=$(grep "Throughput:" "$INITIAL_RESULTS" | tail -1 | awk '{print $2}' | tr -d ',')
COMPRESSED_TPS=$(grep "Throughput:" "$COMPRESSED_RESULTS" | tail -1 | awk '{print $2}' | tr -d ',')

INITIAL_LATENCY=$(grep "Average latency per step:" "$INITIAL_RESULTS" | awk '{print $5}' | tr -d 'ms')
COMPRESSED_LATENCY=$(grep "Average latency per step:" "$COMPRESSED_RESULTS" | awk '{print $5}' | tr -d 'ms')

INITIAL_MEMORY=$(grep "Peak memory:" "$INITIAL_RESULTS" | tail -1 | awk '{print $3}')
COMPRESSED_MEMORY=$(grep "Peak memory:" "$COMPRESSED_RESULTS" | tail -1 | awk '{print $3}')

# Calculate ratios using awk
SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $COMPRESSED_TPS / $INITIAL_TPS}")
PARAM_REDUCTION=$(awk "BEGIN {printf \"%.1f\", (1 - $COMPRESSED_PARAMS / $INITIAL_PARAMS) * 100}")

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

if (( $(echo "$SPEEDUP > 1.0" | bc -l) )); then
  echo "✓ Compressed model is ${SPEEDUP}x FASTER"
else
  echo "✗ Compressed model is slower"
fi

# Clean up
rm -f "$INITIAL_RESULTS" "$COMPRESSED_RESULTS"

echo ""
echo "======================================"
echo "Benchmark completed successfully!"
echo "======================================"